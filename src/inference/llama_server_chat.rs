use super::{ModelCompletion, ModelRequest, RawModelToolCall, TokenUsage, ToolChoice};
use crate::conversation::{ContentPart, ConversationItem, ModelId, ToolOutcome};
use crate::model::{ModelConfig, ThinkingMode};
use serde::Deserialize;
use serde_json::Value;

pub struct LlamaServerChatDefaults {
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

pub fn build_chat_request(
    model: &ModelId,
    request: &ModelRequest,
    model_config: &ModelConfig,
    defaults: &LlamaServerChatDefaults,
) -> crate::Result<serde_json::Value> {
    let messages = request
        .conversation
        .items()
        .iter()
        .map(|item| match item {
            ConversationItem::Message(message) => {
                let mut text = Vec::new();
                for part in &message.content {
                    match part {
                        ContentPart::Text(content) => text.push(content.as_str()),
                        ContentPart::Image(_) => {
                            return Err(crate::error::HoshikageError::InferenceError(
                                "Responses text request contained image content".to_string(),
                            ))
                        }
                    }
                }
                Ok(serde_json::json!({
                    "role": message.role.as_str(),
                    "content": text.join("\n")
                }))
            }
            ConversationItem::FunctionCall(call) => Ok(serde_json::json!({
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": [{
                    "id": call.call_id.as_str(),
                    "type": "function",
                    "function": {
                        "name": call.name.as_str(),
                        "arguments": call.arguments.value()
                    }
                }]
            })),
            ConversationItem::FunctionCallOutput(output) => {
                let content = match &output.outcome {
                    ToolOutcome::Success(content) => content.clone(),
                    ToolOutcome::Failure(content) => {
                        format!("Tool execution failed:\n{content}")
                    }
                    ToolOutcome::Rejected(content) => {
                        format!("Tool execution was rejected:\n{content}")
                    }
                    ToolOutcome::Cancelled(content) => {
                        format!("Tool execution was cancelled:\n{content}")
                    }
                };
                Ok(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": output.call_id.as_str(),
                    "content": content
                }))
            }
        })
        .collect::<crate::Result<Vec<_>>>()?;
    let mut stop = vec![
        "<|im_start|>".to_string(),
        "<|im_end|>".to_string(),
        "</s>".to_string(),
        "<|eot_id|>".to_string(),
        "<|endoftext|>".to_string(),
    ];
    for configured in &model_config.stop {
        if !stop.contains(configured) {
            stop.push(configured.clone());
        }
    }
    let mut body = serde_json::json!({
        "model": model.as_str(),
        "messages": messages,
        "stream": request.stream,
        "temperature": request.sampling.temperature.unwrap_or(defaults.temperature),
        "top_p": request.sampling.top_p.unwrap_or(defaults.top_p),
        "max_tokens": request.max_output_tokens,
        "repeat_penalty": defaults.repeat_penalty,
        "stop": stop,
    });
    if model_config.thinking.mode == ThinkingMode::Off {
        body["chat_template_kwargs"] = serde_json::json!({ "enable_thinking": false });
    }
    if request.stream {
        body["stream_options"] = serde_json::json!({ "include_usage": true });
    }
    if !request.tools.tools().is_empty() {
        body["tools"] = Value::Array(
            request
                .tools
                .tools()
                .iter()
                .map(|tool| {
                    let mut function = serde_json::Map::from_iter([
                        (
                            "name".to_string(),
                            Value::String(tool.name.as_str().to_string()),
                        ),
                        ("parameters".to_string(), tool.parameters.clone()),
                    ]);
                    if let Some(description) = &tool.description {
                        function.insert(
                            "description".to_string(),
                            Value::String(description.clone()),
                        );
                    }
                    serde_json::json!({
                        "type": "function",
                        "function": function
                    })
                })
                .collect(),
        );
        body["tool_choice"] = match &request.tool_choice {
            ToolChoice::Auto => Value::String("auto".to_string()),
            ToolChoice::None => Value::String("none".to_string()),
            ToolChoice::Required => Value::String("required".to_string()),
            ToolChoice::Function(name) => serde_json::json!({
                "type": "function",
                "function": {"name": name.as_str()}
            }),
        };
        body["parallel_tool_calls"] = Value::Bool(false);
    }
    Ok(body)
}

#[derive(Deserialize)]
struct UpstreamResponse {
    choices: Vec<UpstreamChoice>,
    usage: Option<UpstreamUsage>,
}

#[derive(Deserialize)]
struct UpstreamChoice {
    finish_reason: Option<String>,
    message: UpstreamMessage,
}

#[derive(Deserialize)]
struct UpstreamMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<UpstreamToolCall>,
}

#[derive(Deserialize)]
struct UpstreamToolCall {
    #[serde(rename = "type")]
    kind: String,
    function: UpstreamFunction,
}

#[derive(Deserialize)]
struct UpstreamFunction {
    name: String,
    arguments: Value,
}

#[derive(Deserialize)]
struct UpstreamUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

pub fn parse_chat_response(bytes: &[u8]) -> crate::Result<ModelCompletion> {
    let response: UpstreamResponse = serde_json::from_slice(bytes)
        .map_err(|_| crate::error::HoshikageError::ResponseTranslationFailed)?;
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or(crate::error::HoshikageError::ResponseTranslationFailed)?;
    let usage = response
        .usage
        .ok_or(crate::error::HoshikageError::ResponseTranslationFailed)?;
    let usage = TokenUsage::Measured {
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
    };
    if choice.message.tool_calls.len() > 1 {
        return Err(crate::error::HoshikageError::MultipleToolCalls);
    }
    if let Some(tool_call) = choice.message.tool_calls.into_iter().next() {
        if tool_call.kind != "function"
            || choice.finish_reason.as_deref() != Some("tool_calls")
            || choice
                .message
                .content
                .as_deref()
                .is_some_and(|content| !content.trim().is_empty())
        {
            return Err(crate::error::HoshikageError::ResponseTranslationFailed);
        }
        let name = crate::conversation::ToolName::new(tool_call.function.name)
            .map_err(|_| crate::error::HoshikageError::ResponseTranslationFailed)?;
        let arguments = match tool_call.function.arguments {
            Value::String(arguments) => arguments,
            Value::Object(_) => serde_json::to_string(&tool_call.function.arguments)?,
            _ => return Err(crate::error::HoshikageError::InvalidToolArguments),
        };
        return Ok(ModelCompletion::ToolCall {
            call: RawModelToolCall { name, arguments },
            usage,
        });
    }
    if choice.finish_reason.as_deref() == Some("tool_calls") {
        return Err(crate::error::HoshikageError::ResponseTranslationFailed);
    }
    let content = choice
        .message
        .content
        .ok_or(crate::error::HoshikageError::ResponseTranslationFailed)?;
    Ok(ModelCompletion::Text { content, usage })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{
        CallId, Conversation, FunctionCall, FunctionCallOutput, Message, Role, ToolArguments,
        ToolName, ToolOutcome,
    };
    use crate::inference::{ModelTool, ModelToolSet, SamplingOptions, ToolChoice};

    fn request() -> ModelRequest {
        ModelRequest {
            conversation: Conversation::new(vec![ConversationItem::Message(
                Message::text(Role::User, "Return OK.").unwrap(),
            )]),
            tools: ModelToolSet::default(),
            tool_choice: ToolChoice::None,
            sampling: SamplingOptions::default(),
            max_output_tokens: 64,
            stream: false,
        }
    }

    fn tool_request() -> ModelRequest {
        tool_request_with_outcome(ToolOutcome::Success("Hoshikage".to_string()))
    }

    fn tool_request_with_outcome(outcome: ToolOutcome) -> ModelRequest {
        let call_id = CallId::new("call_previous").unwrap();
        ModelRequest {
            conversation: Conversation::new(vec![
                ConversationItem::Message(Message::text(Role::User, "Read README").unwrap()),
                ConversationItem::FunctionCall(FunctionCall {
                    call_id: call_id.clone(),
                    name: ToolName::new("read_file").unwrap(),
                    arguments: ToolArguments::parse(r#"{"path":"README.md"}"#).unwrap(),
                }),
                ConversationItem::FunctionCallOutput(FunctionCallOutput { call_id, outcome }),
            ]),
            tools: ModelToolSet::new(vec![ModelTool {
                name: ToolName::new("read_file").unwrap(),
                description: Some("Read a file".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }),
                strict: Some(false),
            }]),
            tool_choice: ToolChoice::Auto,
            sampling: SamplingOptions::default(),
            max_output_tokens: 64,
            stream: false,
        }
    }

    #[test]
    fn adapter_builds_text_only_chat_request_and_preserves_thinking_off() {
        let mut config =
            ModelConfig::new_legacy("/models".to_string(), "model.gguf".to_string(), Vec::new());
        config.thinking.mode = ThinkingMode::Off;
        config.stop = vec!["BUNDLE_STOP".to_string(), "</s>".to_string()];

        let body = build_chat_request(
            &ModelId::new("gemma4").unwrap(),
            &request(),
            &config,
            &LlamaServerChatDefaults {
                temperature: 0.2,
                top_p: 0.8,
                repeat_penalty: 1.1,
            },
        )
        .unwrap();

        assert_eq!(body["stream"], false);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        let stop = body["stop"].as_array().unwrap();
        assert!(stop.contains(&serde_json::json!("BUNDLE_STOP")));
        assert_eq!(
            stop.iter()
                .filter(|value| **value == serde_json::json!("</s>"))
                .count(),
            1
        );
    }

    #[test]
    fn adapter_requires_usage_in_success_response() {
        let error = parse_chat_response(br#"{"choices":[{"message":{"content":"OK"}}]}"#)
            .err()
            .unwrap();

        assert!(matches!(
            error,
            crate::error::HoshikageError::ResponseTranslationFailed
        ));
    }

    #[test]
    fn adapter_preserves_native_tools_and_tool_result_history() {
        let body = build_chat_request(
            &ModelId::new("gemma4").unwrap(),
            &tool_request(),
            &ModelConfig::new_legacy("/models".to_string(), "model.gguf".to_string(), Vec::new()),
            &LlamaServerChatDefaults {
                temperature: 0.2,
                top_p: 0.8,
                repeat_penalty: 1.1,
            },
        )
        .unwrap();

        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
        assert_eq!(body["messages"][1]["role"], "assistant");
        assert_eq!(body["messages"][1]["tool_calls"][0]["id"], "call_previous");
        assert_eq!(body["messages"][2]["role"], "tool");
        assert_eq!(body["messages"][2]["tool_call_id"], "call_previous");
    }

    #[test]
    fn adapter_preserves_side_effect_tool_outcomes_for_model_recovery() {
        for (outcome, expected) in [
            (
                ToolOutcome::Success("saved".to_string()),
                "saved".to_string(),
            ),
            (
                ToolOutcome::Failure("database unavailable".to_string()),
                "Tool execution failed:\ndatabase unavailable".to_string(),
            ),
            (
                ToolOutcome::Rejected("user denied".to_string()),
                "Tool execution was rejected:\nuser denied".to_string(),
            ),
            (
                ToolOutcome::Cancelled("request cancelled".to_string()),
                "Tool execution was cancelled:\nrequest cancelled".to_string(),
            ),
        ] {
            let request = tool_request_with_outcome(outcome);

            let body = build_chat_request(
                &ModelId::new("gemma4").unwrap(),
                &request,
                &ModelConfig::new_legacy(
                    "/models".to_string(),
                    "model.gguf".to_string(),
                    Vec::new(),
                ),
                &LlamaServerChatDefaults {
                    temperature: 0.2,
                    top_p: 0.8,
                    repeat_penalty: 1.1,
                },
            )
            .unwrap();

            assert_eq!(body["messages"][2]["content"], expected);
        }
    }

    #[test]
    fn adapter_omits_absent_tool_description_instead_of_sending_null() {
        let mut request = tool_request();
        request.tools = ModelToolSet::new(vec![ModelTool {
            name: ToolName::new("read_file").unwrap(),
            description: None,
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        }]);
        let body = build_chat_request(
            &ModelId::new("gemma4").unwrap(),
            &request,
            &ModelConfig::new_legacy("/models".to_string(), "model.gguf".to_string(), Vec::new()),
            &LlamaServerChatDefaults {
                temperature: 0.2,
                top_p: 0.8,
                repeat_penalty: 1.1,
            },
        )
        .unwrap();

        assert!(body["tools"][0]["function"].get("description").is_none());
    }

    #[test]
    fn native_fixture_parses_to_single_domain_tool_call() {
        let completion = parse_chat_response(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/llama-server/10075/gemma4-native-tool-response.json"
        )))
        .unwrap();

        let ModelCompletion::ToolCall { call, usage } = completion else {
            panic!("fixture must parse as Tool Call");
        };
        assert_eq!(call.name.as_str(), "read_file");
        assert_eq!(call.arguments, r#"{"path":"README.md"}"#);
        assert_eq!(
            usage,
            TokenUsage::Measured {
                input_tokens: 75,
                output_tokens: 17
            }
        );
    }

    #[test]
    fn malformed_upstream_json_is_a_translation_failure() {
        let error = parse_chat_response(br#"{"choices":"broken"}"#)
            .err()
            .unwrap();

        assert!(matches!(
            error,
            crate::error::HoshikageError::ResponseTranslationFailed
        ));
    }

    #[test]
    fn multiple_native_tool_calls_are_rejected_for_semantic_recovery() {
        let error = parse_chat_response(
            br#"{
                "choices":[{
                    "finish_reason":"tool_calls",
                    "message":{"content":null,"tool_calls":[
                        {"type":"function","function":{"name":"read_file","arguments":"{}"}},
                        {"type":"function","function":{"name":"read_file","arguments":"{}"}}
                    ]}
                }],
                "usage":{"prompt_tokens":10,"completion_tokens":5}
            }"#,
        )
        .err()
        .unwrap();

        assert!(matches!(
            error,
            crate::error::HoshikageError::MultipleToolCalls
        ));
    }
}
