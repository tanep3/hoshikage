use super::wire::CompletedResponseWire;
use crate::application::ResponsesService;
use axum::extract::rejection::JsonRejection;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use futures_util::StreamExt;
use serde_json::Value;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

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
    if request.stream {
        return match service.execute_stream(request).await {
            Ok(events) => {
                let stream = events.map(|result| {
                    let event = match result {
                        Ok(event) => super::sse::to_sse_event(&event).unwrap_or_else(|_| {
                            Event::default().event("error").data(
                                r#"{"type":"error","code":"response_translation_failed","message":"Response translation failed"}"#,
                            )
                        }),
                        Err(_) => Event::default().event("error").data(
                            r#"{"type":"error","code":"generation_failed","message":"Generation failed"}"#,
                        ),
                    };
                    Ok::<Event, Infallible>(event)
                });
                Sse::new(stream)
                    .keep_alive(
                        KeepAlive::new()
                            .interval(Duration::from_secs(15))
                            .text("keep-alive"),
                    )
                    .into_response()
            }
            Err(error) => crate::api::error::responses_error(error),
        };
    }
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
        InferenceGateway, InferenceGatewayError, ModelActionStream, ModelCompletion, ModelRequest,
        ModelStreamAction, RawModelToolCall, TokenUsage,
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

        async fn stream(
            &self,
            model: &ModelId,
            _request: ModelRequest,
        ) -> Result<ModelActionStream, InferenceGatewayError> {
            if model.as_str() == "tool-stream" {
                return Ok(Box::pin(futures_util::stream::iter(vec![
                    Ok(ModelStreamAction::BeginFunctionCall {
                        name: crate::conversation::ToolName::new("read_file").unwrap(),
                    }),
                    Ok(ModelStreamAction::AppendArguments(
                        r#"{"path":"README.md"}"#.to_string(),
                    )),
                    Ok(ModelStreamAction::FinishFunctionCall),
                    Ok(ModelStreamAction::Complete {
                        usage: TokenUsage::Measured {
                            input_tokens: 8,
                            output_tokens: 4,
                        },
                    }),
                ])));
            }
            if model.as_str() == "failure-stream" {
                return Ok(Box::pin(futures_util::stream::iter(vec![
                    Ok(ModelStreamAction::BeginText),
                    Ok(ModelStreamAction::AppendText("partial".to_string())),
                    Err(InferenceGatewayError::UpstreamDisconnected),
                ])));
            }
            if model.as_str() == "incomplete-stream" {
                return Ok(Box::pin(futures_util::stream::iter(vec![
                    Ok(ModelStreamAction::BeginText),
                    Ok(ModelStreamAction::AppendText("partial".to_string())),
                ])));
            }
            Ok(Box::pin(futures_util::stream::iter(vec![
                Ok(ModelStreamAction::BeginText),
                Ok(ModelStreamAction::AppendText("OK".to_string())),
                Ok(ModelStreamAction::FinishText),
                Ok(ModelStreamAction::Complete {
                    usage: TokenUsage::Measured {
                        input_tokens: 4,
                        output_tokens: 1,
                    },
                }),
            ])))
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
    async fn stream_text_request_returns_ordered_sse_events() {
        let response = router()
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"gemma4","input":"Return OK.","stream":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        let types = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str::<Value>(data).unwrap())
            .map(|value| value["type"].as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            types,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
    }

    #[tokio::test]
    async fn stream_tool_request_returns_ordered_function_events_with_stable_ids() {
        let response = router()
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{
                            "model":"tool-stream",
                            "input":"Read README.md",
                            "stream":true,
                            "tools":[{
                                "type":"function",
                                "name":"read_file",
                                "parameters":{"type":"object"}
                            }]
                        }"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let events = String::from_utf8(body.to_vec())
            .unwrap()
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str::<Value>(data).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            events
                .iter()
                .map(|event| event["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let call_id = events[2]["item"]["call_id"].as_str().unwrap();
        assert_eq!(events[5]["item"]["call_id"], call_id);
        assert_eq!(events[6]["response"]["output"][0]["call_id"], call_id);
    }

    #[tokio::test]
    async fn stream_failure_never_emits_completed() {
        let response = router()
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"failure-stream","input":"Hello","stream":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("\"type\":\"error\""));
        assert!(body.contains("\"type\":\"response.failed\""));
        assert!(!body.contains("\"type\":\"response.completed\""));
    }

    #[tokio::test]
    async fn stream_end_without_terminal_action_is_upstream_disconnect() {
        let response = router()
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"incomplete-stream","input":"Hello","stream":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("\"code\":\"upstream_disconnected\""));
        assert!(body.contains("\"type\":\"response.failed\""));
        assert!(!body.contains("\"type\":\"response.completed\""));
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
