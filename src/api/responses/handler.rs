use super::wire::CompletedResponseWire;
use crate::application::ResponsesService;
use axum::extract::rejection::JsonRejection;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde_json::Value;
use std::sync::Arc;

pub async fn responses(
    Extension(service): Extension<Arc<ResponsesService>>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    let Json(body) = match payload {
        Ok(payload) => payload,
        Err(error) => return crate::api::error::invalid_json(error.status(), error.body_text()),
    };
    let request = match super::wire::decode_request_with_limits(
        body,
        service.unknown_field_policy(),
        service.request_limits(),
    ) {
        Ok(request) => request,
        Err(error) => return crate::api::error::wire_request_error(error),
    };
    match service.execute(request).await {
        Ok(completed) => Json(CompletedResponseWire::from(completed)).into_response(),
        Err(error) => crate::api::error::responses_error(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::UnknownFieldPolicy;
    use crate::conversation::ModelId;
    use crate::inference::{
        InferenceGateway, InferenceGatewayError, ModelCompletion, ModelRequest, RawModelToolCall,
        TokenUsage,
    };
    use async_trait::async_trait;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use tower::ServiceExt;

    struct FakeGateway;

    #[async_trait]
    impl InferenceGateway for FakeGateway {
        async fn complete(
            &self,
            model: &ModelId,
            _request: ModelRequest,
        ) -> Result<ModelCompletion, InferenceGatewayError> {
            if model.as_str() == "missing" {
                return Err(InferenceGatewayError::ModelNotFound);
            }
            if model.as_str() == "context-too-small" {
                return Err(InferenceGatewayError::ContextLengthExceeded);
            }
            if model.as_str() == "slow" {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            if model.as_str() == "tool-model" {
                return Ok(ModelCompletion::ToolCall {
                    call: RawModelToolCall {
                        name: crate::conversation::ToolName::new("read_file").unwrap(),
                        arguments: r#"{ "path": "README.md" }"#.to_string(),
                    },
                    usage: TokenUsage::Measured {
                        input_tokens: 20,
                        output_tokens: 5,
                    },
                });
            }
            Ok(ModelCompletion::Text {
                content: "OK".to_string(),
                usage: TokenUsage::Measured {
                    input_tokens: 4,
                    output_tokens: 1,
                },
            })
        }
    }

    fn router_with_timeout(timeout: std::time::Duration) -> Router {
        let service = Arc::new(ResponsesService::new(
            Arc::new(FakeGateway),
            UnknownFieldPolicy::Compatible,
            timeout,
            crate::application::ResponsesRequestLimits {
                max_tool_schema_bytes: 4096,
                max_single_tool_schema_bytes: 2048,
                max_tools: 16,
                max_tool_argument_bytes: 2048,
                max_tool_result_bytes: 4096,
            },
        ));
        Router::new()
            .route("/v1/responses", post(responses))
            .layer(Extension(service))
    }

    fn router() -> Router {
        router_with_timeout(std::time::Duration::from_secs(1))
    }

    #[tokio::test]
    async fn non_stream_text_request_returns_openai_response_shape() {
        let response = router()
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"gemma4","input":"Return OK.","stream":false}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["object"], "response");
        assert_eq!(value["output"][0]["content"][0]["text"], "OK");
        assert_eq!(value["usage"]["total_tokens"], 5);
    }

    #[tokio::test]
    async fn request_validation_error_uses_openai_error_shape() {
        let response = router()
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"gemma4"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], "invalid_request");
        assert_eq!(value["error"]["param"], "input");
    }

    #[tokio::test]
    async fn model_not_found_uses_stable_public_error_code() {
        let response = router()
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"missing","input":"Hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], "model_not_found");
        assert_eq!(value["error"]["param"], "model");
    }

    #[tokio::test]
    async fn elapsed_request_uses_stable_upstream_timeout_error() {
        let response = router_with_timeout(std::time::Duration::from_millis(1))
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"slow","input":"Hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], "upstream_timeout");
        assert_eq!(value["error"]["type"], "server_error");
    }

    #[tokio::test]
    async fn context_limit_uses_stable_request_error() {
        let response = router()
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"context-too-small","input":"Hello","max_output_tokens":8192}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], "context_length_exceeded");
        assert_eq!(value["error"]["param"], "max_output_tokens");
    }

    #[tokio::test]
    async fn tool_request_returns_function_call_without_executing_it() {
        let response = router()
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{
                            "model": "tool-model",
                            "input": "Read README.md",
                            "tools": [{
                                "type": "function",
                                "name": "read_file",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"path": {"type": "string"}},
                                    "required": ["path"]
                                }
                            }],
                            "tool_choice": "auto"
                        }"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["output"][0]["type"], "function_call");
        assert_eq!(value["output"][0]["name"], "read_file");
        assert_eq!(value["output"][0]["arguments"], r#"{"path":"README.md"}"#);
        assert!(value["output"][0]["call_id"]
            .as_str()
            .is_some_and(|call_id| call_id.starts_with("call_")));
    }
}
