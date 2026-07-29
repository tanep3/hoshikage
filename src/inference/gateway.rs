use super::{ModelCompletion, ModelRequest, ModelStreamAction};
use crate::conversation::ModelId;
use async_trait::async_trait;
use futures_util::Stream;
use futures_util::StreamExt;
use std::pin::Pin;
use std::sync::Arc;
use thiserror::Error;

pub type ModelActionStream =
    Pin<Box<dyn Stream<Item = Result<ModelStreamAction, InferenceGatewayError>> + Send>>;

#[derive(Debug, Error)]
pub enum InferenceGatewayError {
    #[error("model was not found")]
    ModelNotFound,
    #[error("model failed to load")]
    ModelLoadFailed,
    #[error("generation failed")]
    GenerationFailed,
    #[error("upstream response translation failed")]
    TranslationFailed,
    #[error("runtime is busy")]
    ServerBusy,
    #[error("context length exceeded")]
    ContextLengthExceeded,
    #[error("model does not support tool calling")]
    ToolCallingNotSupported,
    #[error("model does not support image input")]
    VisionNotSupported,
    #[error("tool schema is invalid")]
    InvalidToolSchema,
    #[error("tool arguments are invalid")]
    InvalidToolArguments,
    #[error("model violated tool choice")]
    ToolChoiceViolation,
    #[error("upstream timed out")]
    UpstreamTimeout,
    #[error("upstream disconnected")]
    UpstreamDisconnected,
}

#[async_trait]
pub trait InferenceGateway: Send + Sync {
    async fn complete(
        &self,
        model: &ModelId,
        request: ModelRequest,
    ) -> Result<ModelCompletion, InferenceGatewayError>;

    async fn stream(
        &self,
        _model: &ModelId,
        _request: ModelRequest,
    ) -> Result<ModelActionStream, InferenceGatewayError> {
        Err(InferenceGatewayError::GenerationFailed)
    }
}

pub struct ModelManagerGateway {
    manager: Arc<crate::model::ModelManager>,
}

impl ModelManagerGateway {
    pub fn new(manager: Arc<crate::model::ModelManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl InferenceGateway for ModelManagerGateway {
    async fn complete(
        &self,
        model: &ModelId,
        request: ModelRequest,
    ) -> Result<ModelCompletion, InferenceGatewayError> {
        self.manager
            .complete_model_request(model, request)
            .await
            .map_err(map_hoshikage_error)
    }

    async fn stream(
        &self,
        model: &ModelId,
        request: ModelRequest,
    ) -> Result<ModelActionStream, InferenceGatewayError> {
        let stream = self
            .manager
            .stream_model_request(model, request)
            .await
            .map_err(map_hoshikage_error)?;
        Ok(Box::pin(
            stream.map(|result| result.map_err(map_hoshikage_error)),
        ))
    }
}

fn map_hoshikage_error(error: crate::error::HoshikageError) -> InferenceGatewayError {
    match error {
        crate::error::HoshikageError::ModelNotFound(_) => InferenceGatewayError::ModelNotFound,
        crate::error::HoshikageError::ModelLoadFailed(_)
        | crate::error::HoshikageError::ConfigError(_)
        | crate::error::HoshikageError::LibraryLoadError(_) => {
            InferenceGatewayError::ModelLoadFailed
        }
        crate::error::HoshikageError::ServerBusy => InferenceGatewayError::ServerBusy,
        crate::error::HoshikageError::ContextLengthExceeded => {
            InferenceGatewayError::ContextLengthExceeded
        }
        crate::error::HoshikageError::ToolCallingNotSupported => {
            InferenceGatewayError::ToolCallingNotSupported
        }
        crate::error::HoshikageError::VisionNotSupported => {
            InferenceGatewayError::VisionNotSupported
        }
        crate::error::HoshikageError::InvalidToolSchema => InferenceGatewayError::InvalidToolSchema,
        crate::error::HoshikageError::InvalidToolArguments
        | crate::error::HoshikageError::MultipleToolCalls => {
            InferenceGatewayError::InvalidToolArguments
        }
        crate::error::HoshikageError::ToolChoiceViolation => {
            InferenceGatewayError::ToolChoiceViolation
        }
        crate::error::HoshikageError::SerdeError(_) => InferenceGatewayError::TranslationFailed,
        crate::error::HoshikageError::ResponseTranslationFailed => {
            InferenceGatewayError::TranslationFailed
        }
        crate::error::HoshikageError::GenerationFailed => InferenceGatewayError::GenerationFailed,
        crate::error::HoshikageError::UpstreamDisconnected => {
            InferenceGatewayError::UpstreamDisconnected
        }
        crate::error::HoshikageError::UpstreamTimeout => InferenceGatewayError::UpstreamTimeout,
        crate::error::HoshikageError::HttpError(error) if error.is_timeout() => {
            InferenceGatewayError::UpstreamTimeout
        }
        crate::error::HoshikageError::HttpError(_) => InferenceGatewayError::UpstreamDisconnected,
        _ => InferenceGatewayError::GenerationFailed,
    }
}
