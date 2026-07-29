use crate::application::ResponsesServiceError;
use crate::inference::InferenceGatewayError;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
struct OpenAiErrorEnvelope {
    error: OpenAiErrorBody,
}

#[derive(Serialize)]
struct OpenAiErrorBody {
    message: String,
    r#type: &'static str,
    param: Option<String>,
    code: &'static str,
}

pub fn responses_error(error: ResponsesServiceError) -> Response {
    let (status, message, error_type, param, code) = match error {
        ResponsesServiceError::Inference(InferenceGatewayError::ModelNotFound) => (
            StatusCode::BAD_REQUEST,
            "Model was not found".to_string(),
            "invalid_request_error",
            Some("model".to_string()),
            "model_not_found",
        ),
        ResponsesServiceError::Inference(InferenceGatewayError::ModelLoadFailed) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Model failed to load".to_string(),
            "server_error",
            Some("model".to_string()),
            "model_load_failed",
        ),
        ResponsesServiceError::Inference(InferenceGatewayError::GenerationFailed) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Generation failed".to_string(),
            "server_error",
            None,
            "generation_failed",
        ),
        ResponsesServiceError::Inference(InferenceGatewayError::TranslationFailed)
        | ResponsesServiceError::InvalidToolArguments
        | ResponsesServiceError::Identity => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Response translation failed".to_string(),
            "server_error",
            None,
            "response_translation_failed",
        ),
        ResponsesServiceError::Inference(InferenceGatewayError::ServerBusy) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "Runtime is busy".to_string(),
            "server_error",
            None,
            "server_busy",
        ),
        ResponsesServiceError::Inference(InferenceGatewayError::ContextLengthExceeded) => (
            StatusCode::BAD_REQUEST,
            "max_output_tokens exceeds the model context limit".to_string(),
            "invalid_request_error",
            Some("max_output_tokens".to_string()),
            "context_length_exceeded",
        ),
        ResponsesServiceError::Inference(InferenceGatewayError::ToolCallingNotSupported) => (
            StatusCode::BAD_REQUEST,
            "Model does not support tool calling".to_string(),
            "invalid_request_error",
            Some("tools".to_string()),
            "tool_calling_not_supported",
        ),
        ResponsesServiceError::Inference(InferenceGatewayError::VisionNotSupported) => (
            StatusCode::BAD_REQUEST,
            "Model does not support image input".to_string(),
            "invalid_request_error",
            Some("input".to_string()),
            "vision_not_supported",
        ),
        ResponsesServiceError::Inference(InferenceGatewayError::InvalidToolSchema) => (
            StatusCode::BAD_REQUEST,
            "Tool schema is invalid".to_string(),
            "invalid_request_error",
            Some("tools".to_string()),
            "invalid_tool_schema",
        ),
        ResponsesServiceError::Inference(InferenceGatewayError::InvalidToolArguments) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Model generated invalid tool arguments".to_string(),
            "server_error",
            None,
            "invalid_tool_arguments",
        ),
        ResponsesServiceError::Inference(InferenceGatewayError::ToolChoiceViolation) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Model did not satisfy tool_choice".to_string(),
            "server_error",
            None,
            "response_translation_failed",
        ),
        ResponsesServiceError::Inference(InferenceGatewayError::UpstreamTimeout) => (
            StatusCode::GATEWAY_TIMEOUT,
            "Upstream timed out".to_string(),
            "server_error",
            None,
            "upstream_timeout",
        ),
        ResponsesServiceError::Inference(InferenceGatewayError::UpstreamDisconnected) => (
            StatusCode::BAD_GATEWAY,
            "Upstream disconnected".to_string(),
            "server_error",
            None,
            "upstream_disconnected",
        ),
    };
    (
        status,
        Json(OpenAiErrorEnvelope {
            error: OpenAiErrorBody {
                message,
                r#type: error_type,
                param,
                code,
            },
        }),
    )
        .into_response()
}

pub fn wire_request_error(error: crate::api::responses::wire::WireRequestError) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(OpenAiErrorEnvelope {
            error: OpenAiErrorBody {
                message: error.message,
                r#type: "invalid_request_error",
                param: error.param,
                code: error.code,
            },
        }),
    )
        .into_response()
}

pub fn invalid_json(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(OpenAiErrorEnvelope {
            error: OpenAiErrorBody {
                message: message.into(),
                r#type: "invalid_request_error",
                param: None,
                code: "invalid_request",
            },
        }),
    )
        .into_response()
}
