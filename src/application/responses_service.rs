use super::response_stream::{ResponseEvent, ResponseMachine, ResponseMachineError, StreamFailure};
use crate::config::UnknownFieldPolicy;
use crate::conversation::{
    CallId, Conversation, ModelId, OutputItemId, RequestId, ResponseId, ToolArguments, ToolName,
};
use crate::inference::{
    InferenceGateway, InferenceGatewayError, ModelCompletion, ModelRequest, ModelStreamAction,
    ModelToolSet, SamplingOptions, TokenUsage, ToolChoice,
};
use crate::observability::DebugCapture;
use futures_util::{Stream, StreamExt};
use std::pin::Pin;
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
    pub stream: bool,
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

pub type ResponseEventStream =
    Pin<Box<dyn Stream<Item = Result<ResponseEvent, ResponsesServiceError>> + Send>>;

pub struct ResponsesService {
    gateway: Arc<dyn InferenceGateway>,
    unknown_field_policy: UnknownFieldPolicy,
    request_timeout: Duration,
    request_limits: ResponsesRequestLimits,
    debug_capture: Option<DebugCapture>,
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
            debug_capture: None,
        }
    }

    pub fn with_debug_capture(mut self, debug_capture: DebugCapture) -> Self {
        self.debug_capture = Some(debug_capture);
        self
    }

    pub fn debug_capture(&self) -> Option<DebugCapture> {
        self.debug_capture.clone()
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
        self.execute_observed(request, new_request_id()?).await
    }

    pub async fn execute_observed(
        &self,
        request: NormalizedResponsesRequest,
        request_id: RequestId,
    ) -> Result<CompletedResponse, ResponsesServiceError> {
        let started_at = std::time::Instant::now();
        let model_name = request.model.as_str().to_string();
        let tools_count = request.tools.tools().len();
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
        let completion = match tokio::time::timeout(
            self.request_timeout,
            self.gateway.complete(&model, model_request),
        )
        .await
        {
            Ok(Ok(completion)) => completion,
            Ok(Err(error)) => {
                log_request_failure(
                    &request_id,
                    &model_name,
                    false,
                    tools_count,
                    started_at,
                    inference_error_class(&error),
                );
                return Err(error.into());
            }
            Err(_) => {
                log_request_failure(
                    &request_id,
                    &model_name,
                    false,
                    tools_count,
                    started_at,
                    "upstream_timeout",
                );
                return Err(InferenceGatewayError::UpstreamTimeout.into());
            }
        };
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
        let response = CompletedResponse {
            id: ResponseId::new(format!("resp_{}", uuid::Uuid::new_v4().simple()))
                .map_err(|_| ResponsesServiceError::Identity)?,
            created_at: chrono::Utc::now().timestamp(),
            model,
            output,
            usage,
        };
        let (input_tokens, output_tokens) = usage_counts(&response.usage);
        tracing::info!(
            request_id = request_id.as_str(),
            response_id = response.id.as_str(),
            model = model_name,
            stream = false,
            tools_count,
            input_tokens,
            output_tokens,
            elapsed_ms = started_at.elapsed().as_millis(),
            terminal = "completed",
            "Responses request completed"
        );
        Ok(response)
    }

    pub async fn execute_stream(
        &self,
        request: NormalizedResponsesRequest,
    ) -> Result<ResponseEventStream, ResponsesServiceError> {
        self.execute_stream_observed(request, new_request_id()?)
            .await
    }

    pub async fn execute_stream_observed(
        &self,
        request: NormalizedResponsesRequest,
        request_id: RequestId,
    ) -> Result<ResponseEventStream, ResponsesServiceError> {
        let started_at = std::time::Instant::now();
        let model_name = request.model.as_str().to_string();
        let tools_count = request.tools.tools().len();
        for field in &request.warnings {
            tracing::warn!(field, "Ignored unsupported Responses request field");
        }
        let model_request = ModelRequest {
            conversation: request.conversation,
            tools: request.tools,
            tool_choice: request.tool_choice,
            sampling: request.sampling,
            max_output_tokens: request.max_output_tokens,
            stream: true,
        };
        let upstream = match tokio::time::timeout(
            self.request_timeout,
            self.gateway.stream(&request.model, model_request),
        )
        .await
        {
            Ok(Ok(upstream)) => upstream,
            Ok(Err(error)) => {
                log_request_failure(
                    &request_id,
                    &model_name,
                    true,
                    tools_count,
                    started_at,
                    inference_error_class(&error),
                );
                return Err(error.into());
            }
            Err(_) => {
                log_request_failure(
                    &request_id,
                    &model_name,
                    true,
                    tools_count,
                    started_at,
                    "upstream_timeout",
                );
                return Err(InferenceGatewayError::UpstreamTimeout.into());
            }
        };
        let response_id = ResponseId::new(format!("resp_{}", uuid::Uuid::new_v4().simple()))
            .map_err(|_| ResponsesServiceError::Identity)?;
        let observed_response_id = response_id.clone();

        let stream = async_stream::stream! {
            let mut upstream = upstream;
            let mut machine = ResponseMachine::new(response_id);
            match machine.start() {
                Ok(events) => {
                    for event in events {
                        yield Ok(event);
                    }
                }
                Err(_) => {
                    yield Err(ResponsesServiceError::Identity);
                    return;
                }
            }
            while let Some(result) = upstream.next().await {
                let action = match result {
                    Ok(action) => action,
                    Err(error) => {
                        let failure = stream_failure(&error);
                        tracing::warn!(
                            request_id = request_id.as_str(),
                            response_id = observed_response_id.as_str(),
                            model = model_name,
                            stream = true,
                            tools_count,
                            elapsed_ms = started_at.elapsed().as_millis(),
                            terminal = "failed",
                            error_class = failure.code,
                            "Responses request failed"
                        );
                        if let Ok(events) = machine.fail(failure) {
                            for event in events {
                                yield Ok(event);
                            }
                        }
                        return;
                    }
                };
                if let ModelStreamAction::Complete { usage } = &action {
                    let (input_tokens, output_tokens) = usage_counts(usage);
                    tracing::info!(
                        request_id = request_id.as_str(),
                        response_id = observed_response_id.as_str(),
                        model = model_name,
                        stream = true,
                        tools_count,
                        input_tokens,
                        output_tokens,
                        elapsed_ms = started_at.elapsed().as_millis(),
                        terminal = "completed",
                        "Responses request completed"
                    );
                }
                let events = apply_stream_action(&mut machine, action);
                match events {
                    Ok(events) => {
                        for event in events {
                            yield Ok(event);
                        }
                    }
                    Err(_) => {
                        tracing::warn!(
                            request_id = request_id.as_str(),
                            response_id = observed_response_id.as_str(),
                            model = model_name,
                            stream = true,
                            tools_count,
                            elapsed_ms = started_at.elapsed().as_millis(),
                            terminal = "failed",
                            error_class = "response_translation_failed",
                            "Responses request failed"
                        );
                        if let Ok(events) = machine.fail(StreamFailure {
                            code: "response_translation_failed",
                            message: "Response translation failed",
                        }) {
                            for event in events {
                                yield Ok(event);
                            }
                        }
                        return;
                    }
                }
            }
            if !machine.is_terminal() {
                tracing::warn!(
                    request_id = request_id.as_str(),
                    response_id = observed_response_id.as_str(),
                    model = model_name,
                    stream = true,
                    tools_count,
                    elapsed_ms = started_at.elapsed().as_millis(),
                    terminal = "failed",
                    error_class = "upstream_disconnected",
                    "Responses request failed"
                );
                if let Ok(events) = machine.fail(StreamFailure {
                    code: "upstream_disconnected",
                    message: "Upstream disconnected",
                }) {
                    for event in events {
                        yield Ok(event);
                    }
                }
            }
        };
        Ok(Box::pin(stream))
    }
}

fn new_request_id() -> Result<RequestId, ResponsesServiceError> {
    RequestId::new(format!("req_{}", uuid::Uuid::new_v4().simple()))
        .map_err(|_| ResponsesServiceError::Identity)
}

fn usage_counts(usage: &TokenUsage) -> (u32, u32) {
    match usage {
        TokenUsage::Measured {
            input_tokens,
            output_tokens,
        }
        | TokenUsage::Estimated {
            input_tokens,
            output_tokens,
        } => (*input_tokens, *output_tokens),
    }
}

fn log_request_failure(
    request_id: &RequestId,
    model: &str,
    stream: bool,
    tools_count: usize,
    started_at: std::time::Instant,
    error_class: &str,
) {
    tracing::warn!(
        request_id = request_id.as_str(),
        model,
        stream,
        tools_count,
        elapsed_ms = started_at.elapsed().as_millis(),
        terminal = "failed",
        error_class,
        "Responses request failed"
    );
}

fn inference_error_class(error: &InferenceGatewayError) -> &'static str {
    match error {
        InferenceGatewayError::ModelNotFound => "model_not_found",
        InferenceGatewayError::ModelLoadFailed => "model_load_failed",
        InferenceGatewayError::ToolCallingNotSupported => "tool_calling_not_supported",
        InferenceGatewayError::InvalidToolSchema => "invalid_tool_schema",
        InferenceGatewayError::InvalidToolArguments => "invalid_tool_arguments",
        InferenceGatewayError::ToolChoiceViolation => "tool_choice_violation",
        InferenceGatewayError::ContextLengthExceeded => "context_length_exceeded",
        InferenceGatewayError::ServerBusy => "server_busy",
        InferenceGatewayError::UpstreamTimeout => "upstream_timeout",
        InferenceGatewayError::UpstreamDisconnected => "upstream_disconnected",
        InferenceGatewayError::GenerationFailed => "generation_failed",
        InferenceGatewayError::TranslationFailed => "response_translation_failed",
    }
}

fn apply_stream_action(
    machine: &mut ResponseMachine,
    action: ModelStreamAction,
) -> Result<Vec<ResponseEvent>, ResponseMachineError> {
    match action {
        ModelStreamAction::BeginText => machine.begin_text(
            OutputItemId::new(format!("msg_{}", uuid::Uuid::new_v4().simple()))
                .map_err(|_| ResponseMachineError::InvalidTransition)?,
        ),
        ModelStreamAction::AppendText(delta) => machine.append_text(delta).map(|event| vec![event]),
        ModelStreamAction::FinishText => machine.finish_text(),
        ModelStreamAction::BeginFunctionCall { name } => machine
            .begin_function_call(
                OutputItemId::new(format!("fc_{}", uuid::Uuid::new_v4().simple()))
                    .map_err(|_| ResponseMachineError::InvalidTransition)?,
                CallId::new(format!("call_{}", uuid::Uuid::new_v4().simple()))
                    .map_err(|_| ResponseMachineError::InvalidTransition)?,
                name,
            )
            .map(|event| vec![event]),
        ModelStreamAction::AppendArguments(delta) => machine
            .append_function_arguments(delta)
            .map(|event| vec![event]),
        ModelStreamAction::FinishFunctionCall => machine.finish_function_call(),
        ModelStreamAction::Complete { usage } => machine.complete(usage).map(|event| vec![event]),
    }
}

fn stream_failure(error: &InferenceGatewayError) -> StreamFailure {
    match error {
        InferenceGatewayError::UpstreamTimeout => StreamFailure {
            code: "upstream_timeout",
            message: "Upstream timed out",
        },
        InferenceGatewayError::UpstreamDisconnected => StreamFailure {
            code: "upstream_disconnected",
            message: "Upstream disconnected",
        },
        InferenceGatewayError::InvalidToolArguments => StreamFailure {
            code: "invalid_tool_arguments",
            message: "Tool arguments are invalid",
        },
        InferenceGatewayError::ToolChoiceViolation => StreamFailure {
            code: "tool_choice_violation",
            message: "Model violated tool choice",
        },
        InferenceGatewayError::ContextLengthExceeded => StreamFailure {
            code: "context_length_exceeded",
            message: "Context length exceeded",
        },
        InferenceGatewayError::TranslationFailed => StreamFailure {
            code: "response_translation_failed",
            message: "Response translation failed",
        },
        _ => StreamFailure {
            code: "generation_failed",
            message: "Generation failed",
        },
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
            stream: false,
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

    #[test]
    fn observability_uses_stable_error_classes() {
        assert_eq!(
            inference_error_class(&InferenceGatewayError::ModelNotFound),
            "model_not_found"
        );
        assert_eq!(
            inference_error_class(&InferenceGatewayError::ServerBusy),
            "server_busy"
        );
        assert_eq!(
            inference_error_class(&InferenceGatewayError::TranslationFailed),
            "response_translation_failed"
        );
    }
}
