use super::{ModelCompletion, ModelRequest, RawModelToolCall, ToolChoice};
use crate::conversation::{
    ContentPart, Conversation, ConversationItem, ToolArguments, ToolOutputContent,
};
use crate::model::{ModelConfig, ToolCallingMode, ToolParserId, ToolResultPolicy};

pub fn apply_tool_result_policy(
    request: &mut ModelRequest,
    config: &ModelConfig,
) -> crate::Result<()> {
    let ToolResultPolicy::HeadTail {
        max_bytes,
        head_bytes,
        tail_bytes,
    } = &config.tool_calling.result_policy
    else {
        return Ok(());
    };
    let (max_bytes, head_bytes, tail_bytes) = (*max_bytes, *head_bytes, *tail_bytes);
    let items = std::mem::take(&mut request.conversation).into_items();
    request.conversation = Conversation::new(
        items
            .into_iter()
            .map(|item| match item {
                ConversationItem::FunctionCallOutput(mut output) => {
                    output.content = truncate_tool_output_content(
                        output.content,
                        max_bytes,
                        head_bytes,
                        tail_bytes,
                    );
                    ConversationItem::FunctionCallOutput(output)
                }
                item => item,
            })
            .collect(),
    );
    Ok(())
}

fn truncate_tool_output_content(
    content: ToolOutputContent,
    max_bytes: usize,
    head_bytes: usize,
    tail_bytes: usize,
) -> ToolOutputContent {
    match content {
        ToolOutputContent::Text(text) => ToolOutputContent::Text(truncate_tool_result(
            &text, max_bytes, head_bytes, tail_bytes,
        )),
        ToolOutputContent::Items(mut parts) => {
            let text_parts = parts
                .iter()
                .filter(|part| matches!(part, ContentPart::Text(_)))
                .count();
            if text_parts == 0 {
                return ToolOutputContent::Items(parts);
            }
            let per_part = max_bytes / text_parts;
            for part in &mut parts {
                if let ContentPart::Text(text) = part {
                    *text = truncate_tool_result(text, per_part, head_bytes, tail_bytes);
                }
            }
            ToolOutputContent::Items(parts)
        }
    }
}

fn truncate_tool_result(
    content: &str,
    max_bytes: usize,
    head_bytes: usize,
    tail_bytes: usize,
) -> String {
    if content.len() <= max_bytes {
        return content.to_string();
    }
    let notice = format!(
        "\n[Hoshikage tool result truncated: original_bytes={}]\n",
        content.len()
    );
    if notice.len() >= max_bytes {
        return truncate_to_utf8_boundary("[Hoshikage tool result truncated]", max_bytes);
    }
    let available = max_bytes.saturating_sub(notice.len());
    let head_budget = head_bytes.min(available);
    let tail_budget = tail_bytes.min(available.saturating_sub(head_budget));
    let head_end = previous_char_boundary(content, head_budget);
    let tail_start = next_char_boundary(content, content.len().saturating_sub(tail_budget));
    format!(
        "{}{}{}",
        &content[..head_end],
        notice,
        &content[tail_start..]
    )
}

fn truncate_to_utf8_boundary(value: &str, max_bytes: usize) -> String {
    value[..previous_char_boundary(value, max_bytes)].to_string()
}

fn previous_char_boundary(value: &str, mut index: usize) -> usize {
    index = index.min(value.len());
    while index > 0 && !value.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn next_char_boundary(value: &str, mut index: usize) -> usize {
    index = index.min(value.len());
    while index < value.len() && !value.is_char_boundary(index) {
        index += 1;
    }
    index
}

pub fn validate_tool_request(request: &ModelRequest, config: &ModelConfig) -> crate::Result<()> {
    if request.conversation.items().iter().any(|item| {
        matches!(
            item,
            ConversationItem::FunctionCall(call)
                if call.arguments.canonical_json().len() > config.tool_calling.max_argument_bytes
        )
    }) {
        return Err(crate::error::HoshikageError::InvalidToolArguments);
    }
    if request.tools.tools().is_empty() {
        return Ok(());
    }
    if config.tool_calling.mode == ToolCallingMode::Disabled {
        return Err(crate::error::HoshikageError::ToolCallingNotSupported);
    }
    match (
        config.tool_calling.mode,
        config.tool_calling.effective_parser(),
    ) {
        (ToolCallingMode::Native, ToolParserId::LlamaServerNative)
        | (ToolCallingMode::Json, ToolParserId::GenericJson) => Ok(()),
        _ => Err(crate::error::HoshikageError::ConfigError(
            "configured Tool parser is not implemented for this mode".to_string(),
        )),
    }
}

pub fn validate_native_completion(
    completion: ModelCompletion,
    request: &ModelRequest,
    config: &ModelConfig,
) -> crate::Result<ModelCompletion> {
    match completion {
        ModelCompletion::Text { .. }
            if matches!(
                request.tool_choice,
                ToolChoice::Required | ToolChoice::Function(_)
            ) =>
        {
            Err(crate::error::HoshikageError::ToolChoiceViolation)
        }
        ModelCompletion::Text { .. } => Ok(completion),
        ModelCompletion::ToolCall { call, usage } => {
            if request.tools.tools().is_empty() || request.tool_choice == ToolChoice::None {
                return Err(crate::error::HoshikageError::ToolChoiceViolation);
            }
            let tool = request
                .tools
                .tools()
                .iter()
                .find(|tool| tool.name == call.name)
                .ok_or(crate::error::HoshikageError::InvalidToolArguments)?;
            if let ToolChoice::Function(required) = &request.tool_choice {
                if required != &call.name {
                    return Err(crate::error::HoshikageError::ToolChoiceViolation);
                }
            }
            if call.arguments.len() > config.tool_calling.max_argument_bytes {
                return Err(crate::error::HoshikageError::InvalidToolArguments);
            }
            let arguments = ToolArguments::parse(&call.arguments)
                .map_err(|_| crate::error::HoshikageError::InvalidToolArguments)?;
            if config.tool_calling.strict || tool.strict == Some(true) {
                let validator = jsonschema::draft7::new(&tool.parameters)
                    .map_err(|_| crate::error::HoshikageError::InvalidToolSchema)?;
                validator
                    .validate(arguments.value())
                    .map_err(|_| crate::error::HoshikageError::InvalidToolArguments)?;
            }
            Ok(ModelCompletion::ToolCall {
                call: RawModelToolCall {
                    name: call.name,
                    arguments: arguments.canonical_json().to_string(),
                },
                usage,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{Conversation, Message, Role, ToolName};
    use crate::inference::{ModelTool, ModelToolSet, SamplingOptions, TokenUsage};

    fn request(choice: ToolChoice) -> ModelRequest {
        ModelRequest {
            conversation: Conversation::new(vec![crate::conversation::ConversationItem::Message(
                Message::text(Role::User, "Read README").unwrap(),
            )]),
            tools: ModelToolSet::new(vec![ModelTool {
                name: ToolName::new("read_file").unwrap(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": false
                }),
                strict: Some(false),
            }]),
            tool_choice: choice,
            sampling: SamplingOptions::default(),
            max_output_tokens: 64,
            stream: false,
        }
    }

    fn config() -> ModelConfig {
        let mut config =
            ModelConfig::new_legacy("/models".to_string(), "model.gguf".to_string(), Vec::new());
        config.tool_calling.mode = ToolCallingMode::Native;
        config
    }

    fn call(arguments: &str) -> ModelCompletion {
        ModelCompletion::ToolCall {
            call: RawModelToolCall {
                name: ToolName::new("read_file").unwrap(),
                arguments: arguments.to_string(),
            },
            usage: TokenUsage::Measured {
                input_tokens: 10,
                output_tokens: 5,
            },
        }
    }

    #[test]
    fn strict_native_completion_canonicalizes_and_validates_arguments() {
        let validated = validate_native_completion(
            call(r#"{ "path": "README.md" }"#),
            &request(ToolChoice::Auto),
            &config(),
        )
        .unwrap();

        let ModelCompletion::ToolCall { call, .. } = validated else {
            panic!("completion must remain a Tool Call");
        };
        assert_eq!(call.arguments, r#"{"path":"README.md"}"#);
    }

    #[test]
    fn schema_and_tool_choice_violations_are_rejected() {
        let schema_error = validate_native_completion(
            call(r#"{"path":7}"#),
            &request(ToolChoice::Auto),
            &config(),
        )
        .err()
        .unwrap();
        assert!(matches!(
            schema_error,
            crate::error::HoshikageError::InvalidToolArguments
        ));

        let choice_error = validate_native_completion(
            call(r#"{"path":"README.md"}"#),
            &request(ToolChoice::None),
            &config(),
        )
        .err()
        .unwrap();
        assert!(matches!(
            choice_error,
            crate::error::HoshikageError::ToolChoiceViolation
        ));
    }

    #[test]
    fn disabled_bundle_rejects_tools_before_runtime_start() {
        let error = validate_tool_request(
            &request(ToolChoice::Auto),
            &ModelConfig::new_legacy("/models".to_string(), "model.gguf".to_string(), Vec::new()),
        )
        .err()
        .unwrap();

        assert!(matches!(
            error,
            crate::error::HoshikageError::ToolCallingNotSupported
        ));
    }

    #[test]
    fn head_tail_policy_truncates_on_utf8_boundaries_and_marks_the_result() {
        let call_id = crate::conversation::CallId::new("call_1").unwrap();
        let mut request = request(ToolChoice::Auto);
        request.conversation = Conversation::new(vec![
            ConversationItem::FunctionCall(crate::conversation::FunctionCall {
                call_id: call_id.clone(),
                name: ToolName::new("read_file").unwrap(),
                arguments: ToolArguments::parse(r#"{"path":"README.md"}"#).unwrap(),
            }),
            ConversationItem::FunctionCallOutput(crate::conversation::FunctionCallOutput {
                call_id,
                outcome: crate::conversation::ToolOutcome::Success,
                content: ToolOutputContent::Text("日本語".repeat(100)),
            }),
        ]);
        let mut config = config();
        config.tool_calling.result_policy = ToolResultPolicy::HeadTail {
            max_bytes: 128,
            head_bytes: 48,
            tail_bytes: 48,
        };

        apply_tool_result_policy(&mut request, &config).unwrap();

        let ConversationItem::FunctionCallOutput(output) = &request.conversation.items()[1] else {
            panic!("second item must be Tool output");
        };
        assert_eq!(output.outcome, crate::conversation::ToolOutcome::Success);
        let ToolOutputContent::Text(content) = &output.content else {
            panic!("output content must remain text");
        };
        assert!(content.len() <= 128);
        assert!(content.contains("tool result truncated"));
        assert!(std::str::from_utf8(content.as_bytes()).is_ok());
    }

    #[test]
    fn head_tail_policy_honors_even_a_tiny_explicit_byte_limit() {
        let content = truncate_tool_result("日本語".repeat(10).as_str(), 8, 0, 0);

        assert!(content.len() <= 8);
        assert!(std::str::from_utf8(content.as_bytes()).is_ok());
    }

    #[test]
    fn historical_tool_arguments_obey_the_bundle_limit() {
        let mut request = request(ToolChoice::Auto);
        request.conversation = Conversation::new(vec![ConversationItem::FunctionCall(
            crate::conversation::FunctionCall {
                call_id: crate::conversation::CallId::new("call_1").unwrap(),
                name: ToolName::new("read_file").unwrap(),
                arguments: ToolArguments::parse(r#"{"path":"README.md"}"#).unwrap(),
            },
        )]);
        let mut config = config();
        config.tool_calling.max_argument_bytes = 4;

        let error = validate_tool_request(&request, &config).err().unwrap();

        assert!(matches!(
            error,
            crate::error::HoshikageError::InvalidToolArguments
        ));
    }
}
