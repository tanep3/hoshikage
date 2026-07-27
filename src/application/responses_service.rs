use crate::config::UnknownFieldPolicy;
use crate::conversation::{
    CallId, Conversation, ModelId, OutputItemId, ResponseId, ToolArguments, ToolName,
};
use crate::inference::{
    InferenceGateway, InferenceGatewayError, ModelCompletion, ModelRequest, ModelToolSet,
    SamplingOptions, TokenUsage, ToolChoice,
};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

#[derive(Clone)]
pub struct CompletedMessage {
    pub id: OutputItemId,
    pub text: String,
}

#[derive(Clone)]
pub struct CompletedFunctionCall {
    pub id: OutputItemId,
    pub call_id: CallId,
    pub name: ToolName,
    pub arguments: ToolArguments,
}

#[derive(Clone)]
pub enum CompletedOutput {
    Message(CompletedMessage),
    FunctionCall(CompletedFunctionCall),
}

#[derive(Clone)]
pub struct CompletedResponse {
    pub id: ResponseId,
    pub created_at: i64,
    pub model: ModelId,
    pub output: CompletedOutput,
    pub usage: TokenUsage,
}

pub struct NormalizedResponsesRequest {
    pub model: ModelId,
    pub conversation: Conversation,
    pub tools: ModelToolSet,
    pub tool_choice: ToolChoice,
    pub sampling: SamplingOptions,
    pub max_output_tokens: u32,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ResponsesRequestLimits {
    pub max_tool_schema_bytes: usize,
    pub max_single_tool_schema_bytes: usize,
    pub max_tools: usize,
    pub max_tool_argument_bytes: usize,
    pub max_tool_result_bytes: usize,
}

impl Default for ResponsesRequestLimits {
    fn default() -> Self {
        Self {
            max_tool_schema_bytes: 1_048_576,
            max_single_tool_schema_bytes: 262_144,
            max_tools: 128,
            max_tool_argument_bytes: 65_536,
            max_tool_result_bytes: 4_194_304,
        }
    }
}

#[derive(Debug, Error)]
pub enum ResponsesServiceError {
    #[error("{0}")]
    Inference(#[from] InferenceGatewayError),
    #[error("model returned invalid Tool arguments")]
    InvalidToolArguments,
    #[error("failed to construct response identity")]
    Identity,
}

pub struct ResponsesService {
    gateway: Arc<dyn InferenceGateway>,
    unknown_field_policy: UnknownFieldPolicy,
    request_timeout: Duration,
    request_limits: ResponsesRequestLimits,
}

impl ResponsesService {
    pub fn new(
        gateway: Arc<dyn InferenceGateway>,
        unknown_field_policy: UnknownFieldPolicy,
        request_timeout: Duration,
        request_limits: ResponsesRequestLimits,
    ) -> Self {
        Self {
            gateway,
            unknown_field_policy,
            request_timeout,
            request_limits,
        }
    }

    pub fn unknown_field_policy(&self) -> UnknownFieldPolicy {
        self.unknown_field_policy
    }

    pub fn request_limits(&self) -> ResponsesRequestLimits {
        self.request_limits
    }

    pub async fn execute(
        &self,
        request: NormalizedResponsesRequest,
    ) -> Result<CompletedResponse, ResponsesServiceError> {
        for field in &request.warnings {
            tracing::warn!(field, "Ignored unsupported Responses request field");
        }
        let model = request.model;
        let model_request = ModelRequest {
            conversation: request.conversation,
            tools: request.tools,
            tool_choice: request.tool_choice,
            sampling: request.sampling,
            max_output_tokens: request.max_output_tokens,
            stream: false,
        };
        let completion = tokio::time::timeout(
            self.request_timeout,
            self.gateway.complete(&model, model_request),
        )
        .await
        .map_err(|_| InferenceGatewayError::UpstreamTimeout)??;
        let (output, usage) = match completion {
            ModelCompletion::Text { content, usage } => (
                CompletedOutput::Message(CompletedMessage {
                    id: OutputItemId::new(format!("msg_{}", uuid::Uuid::new_v4().simple()))
                        .map_err(|_| ResponsesServiceError::Identity)?,
                    text: content,
                }),
                usage,
            ),
            ModelCompletion::ToolCall { call, usage } => (
                CompletedOutput::FunctionCall(CompletedFunctionCall {
                    id: OutputItemId::new(format!("fc_{}", uuid::Uuid::new_v4().simple()))
                        .map_err(|_| ResponsesServiceError::Identity)?,
                    call_id: CallId::new(format!("call_{}", uuid::Uuid::new_v4().simple()))
                        .map_err(|_| ResponsesServiceError::Identity)?,
                    name: call.name,
                    arguments: ToolArguments::parse(&call.arguments)
                        .map_err(|_| ResponsesServiceError::InvalidToolArguments)?,
                }),
                usage,
            ),
        };
        Ok(CompletedResponse {
            id: ResponseId::new(format!("resp_{}", uuid::Uuid::new_v4().simple()))
                .map_err(|_| ResponsesServiceError::Identity)?,
            created_at: chrono::Utc::now().timestamp(),
            model,
            output,
            usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct FakeGateway;
    struct SlowGateway;

    #[async_trait]
    impl InferenceGateway for FakeGateway {
        async fn complete(
            &self,
            model: &ModelId,
            request: ModelRequest,
        ) -> Result<ModelCompletion, InferenceGatewayError> {
            assert_eq!(model.as_str(), "gemma4");
            assert_eq!(request.conversation.summary().messages, 1);
            Ok(ModelCompletion::Text {
                content: "OK".to_string(),
                usage: TokenUsage::Measured {
                    input_tokens: 10,
                    output_tokens: 1,
                },
            })
        }
    }

    #[async_trait]
    impl InferenceGateway for SlowGateway {
        async fn complete(
            &self,
            _model: &ModelId,
            _request: ModelRequest,
        ) -> Result<ModelCompletion, InferenceGatewayError> {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(ModelCompletion::Text {
                content: "late".to_string(),
                usage: TokenUsage::Measured {
                    input_tokens: 1,
                    output_tokens: 1,
                },
            })
        }
    }

    fn request() -> NormalizedResponsesRequest {
        NormalizedResponsesRequest {
            model: ModelId::new("gemma4").unwrap(),
            conversation: Conversation::new(vec![crate::conversation::ConversationItem::Message(
                crate::conversation::Message::text(crate::conversation::Role::User, "Return OK.")
                    .unwrap(),
            )]),
            tools: ModelToolSet::default(),
            tool_choice: ToolChoice::None,
            sampling: SamplingOptions::default(),
            max_output_tokens: 64,
            warnings: Vec::new(),
        }
    }

    fn limits() -> ResponsesRequestLimits {
        ResponsesRequestLimits {
            max_tool_schema_bytes: 1024,
            max_single_tool_schema_bytes: 512,
            max_tools: 8,
            max_tool_argument_bytes: 512,
            max_tool_result_bytes: 1024,
        }
    }

    #[tokio::test]
    async fn service_returns_completed_text_without_wire_types_in_gateway() {
        let service = ResponsesService::new(
            Arc::new(FakeGateway),
            UnknownFieldPolicy::Compatible,
            Duration::from_secs(1),
            limits(),
        );

        let response = service.execute(request()).await.unwrap();

        assert_eq!(response.model.as_str(), "gemma4");
        let CompletedOutput::Message(message) = response.output else {
            panic!("response must contain a message");
        };
        assert_eq!(message.text, "OK");
        assert_eq!(
            response.usage,
            TokenUsage::Measured {
                input_tokens: 10,
                output_tokens: 1
            }
        );
    }

    #[tokio::test]
    async fn service_converts_elapsed_deadline_to_upstream_timeout() {
        let service = ResponsesService::new(
            Arc::new(SlowGateway),
            UnknownFieldPolicy::Compatible,
            Duration::from_millis(1),
            limits(),
        );

        let error = service.execute(request()).await.err().unwrap();

        assert!(matches!(
            error,
            ResponsesServiceError::Inference(InferenceGatewayError::UpstreamTimeout)
        ));
    }
}
