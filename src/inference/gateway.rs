use super::{ModelCompletion, ModelRequest};
use crate::conversation::ModelId;
use async_trait::async_trait;
use std::sync::Arc;
use thiserror::Error;

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
            .map_err(|error| match error {
                crate::error::HoshikageError::ModelNotFound(_) => {
                    InferenceGatewayError::ModelNotFound
                }
                crate::error::HoshikageError::ModelLoadFailed(_)
                | crate::error::HoshikageError::ConfigError(_)
                | crate::error::HoshikageError::LibraryLoadError(_) => {
                    InferenceGatewayError::ModelLoadFailed
                }
                crate::error::HoshikageError::ServerBusy => InferenceGatewayError::ServerBusy,
                crate::error::HoshikageError::ContextLengthExceeded => {
                    InferenceGatewayError::ContextLengthExceeded
                }
                crate::error::HoshikageError::SerdeError(_) => {
                    InferenceGatewayError::TranslationFailed
                }
                crate::error::HoshikageError::ResponseTranslationFailed => {
                    InferenceGatewayError::TranslationFailed
                }
                crate::error::HoshikageError::HttpError(error) if error.is_timeout() => {
                    InferenceGatewayError::UpstreamTimeout
                }
                crate::error::HoshikageError::HttpError(_) => {
                    InferenceGatewayError::UpstreamDisconnected
                }
                _ => InferenceGatewayError::GenerationFailed,
            })
    }
}
