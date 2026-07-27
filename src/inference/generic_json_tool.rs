use super::{
    build_chat_request, LlamaServerChatDefaults, ModelCompletion, ModelRequest, ModelToolSet,
    RawModelToolCall, ToolChoice,
};
use crate::conversation::{Conversation, ConversationItem, Message, Role, ToolName};
use crate::model::ModelConfig;
use serde_json::{Map, Value};

pub fn build_generic_json_request(
    model: &crate::conversation::ModelId,
    request: &ModelRequest,
    config: &ModelConfig,
    defaults: &LlamaServerChatDefaults,
) -> crate::Result<Value> {
    let schema = output_schema(request)?;
    let tools = request
        .tools
        .tools()
        .iter()
        .map(|tool| {
            serde_json::json!({
                "name": tool.name.as_str(),
                "description": tool.description.as_deref(),
                "parameters": &tool.parameters
            })
        })
        .collect::<Vec<_>>();
    let instruction = format!(
        "You are in JSON Tool Calling mode. Return exactly one JSON object matching the supplied \
         response schema. Use type=function_call to request one provided tool, or type=final for \
         the final answer when allowed. Never execute a tool yourself. Available tools: {}",
        serde_json::to_string(&tools)?
    );
    let mut items = Vec::with_capacity(request.conversation.items().len() + 1);
    items.push(ConversationItem::Message(
        Message::text(Role::Developer, instruction)
            .map_err(|error| crate::error::HoshikageError::InferenceError(error.to_string()))?,
    ));
    items.extend(request.conversation.items().iter().cloned());
    let rewritten = ModelRequest {
        conversation: Conversation::new(items),
        tools: ModelToolSet::default(),
        tool_choice: ToolChoice::None,
        sampling: request.sampling.clone(),
        max_output_tokens: request.max_output_tokens,
        stream: false,
    };
    let mut body = build_chat_request(model, &rewritten, config, defaults)?;
    body["response_format"] = serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": "hoshikage_tool_output",
            "strict": true,
            "schema": schema
        }
    });
    Ok(body)
}

pub fn parse_generic_json_completion(
    completion: ModelCompletion,
    repair_invalid_json: bool,
) -> crate::Result<ModelCompletion> {
    let ModelCompletion::Text { content, usage } = completion else {
        return Err(crate::error::HoshikageError::ResponseTranslationFailed);
    };
    let value = parse_json_object(&content, repair_invalid_json)?;
    match value.get("type").and_then(Value::as_str) {
        Some("final") => {
            let content = value
                .get("content")
                .and_then(Value::as_str)
                .ok_or(crate::error::HoshikageError::ResponseTranslationFailed)?;
            Ok(ModelCompletion::Text {
                content: content.to_string(),
                usage,
            })
        }
        Some("function_call") => {
            let name = value
                .get("name")
                .and_then(Value::as_str)
                .ok_or(crate::error::HoshikageError::InvalidToolArguments)
                .and_then(|name| {
                    ToolName::new(name)
                        .map_err(|_| crate::error::HoshikageError::InvalidToolArguments)
                })?;
            let arguments = value
                .get("arguments")
                .filter(|arguments| arguments.is_object())
                .ok_or(crate::error::HoshikageError::InvalidToolArguments)?;
            Ok(ModelCompletion::ToolCall {
                call: RawModelToolCall {
                    name,
                    arguments: serde_json::to_string(arguments)?,
                },
                usage,
            })
        }
        _ => Err(crate::error::HoshikageError::ResponseTranslationFailed),
    }
}

fn output_schema(request: &ModelRequest) -> crate::Result<Value> {
    let allowed_tools = request
        .tools
        .tools()
        .iter()
        .filter(|tool| match &request.tool_choice {
            ToolChoice::Function(name) => &tool.name == name,
            _ => true,
        })
        .map(|tool| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "type": {"const": "function_call"},
                    "name": {"const": tool.name.as_str()},
                    "arguments": &tool.parameters
                },
                "required": ["type", "name", "arguments"],
                "additionalProperties": false
            })
        })
        .collect::<Vec<_>>();
    let final_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "type": {"const": "final"},
            "content": {"type": "string"}
        },
        "required": ["type", "content"],
        "additionalProperties": false
    });
    let variants = match &request.tool_choice {
        ToolChoice::None => vec![final_schema],
        ToolChoice::Required | ToolChoice::Function(_) => allowed_tools,
        ToolChoice::Auto => {
            let mut variants = allowed_tools;
            variants.push(final_schema);
            variants
        }
    };
    if variants.is_empty() {
        return Err(crate::error::HoshikageError::InvalidToolSchema);
    }
    Ok(serde_json::json!({"oneOf": variants}))
}

fn parse_json_object(content: &str, repair: bool) -> crate::Result<Map<String, Value>> {
    if let Ok(Value::Object(value)) = serde_json::from_str(content) {
        return Ok(value);
    }
    if !repair {
        return Err(crate::error::HoshikageError::InvalidToolArguments);
    }
    let stripped = strip_json_fence(content);
    if let Ok(Value::Object(value)) = serde_json::from_str(stripped) {
        tracing::warn!(repair = "json_fence", "Repaired generic Tool JSON");
        return Ok(value);
    }
    let without_trailing_commas = remove_trailing_commas(stripped);
    if let Ok(Value::Object(value)) = serde_json::from_str(&without_trailing_commas) {
        tracing::warn!(repair = "trailing_comma", "Repaired generic Tool JSON");
        return Ok(value);
    }
    Err(crate::error::HoshikageError::InvalidToolArguments)
}

fn strip_json_fence(content: &str) -> &str {
    let trimmed = content.trim();
    let Some(body) = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```JSON"))
        .or_else(|| trimmed.strip_prefix("```"))
    else {
        return trimmed;
    };
    body.strip_suffix("```").unwrap_or(body).trim()
}

fn remove_trailing_commas(content: &str) -> String {
    let chars = content.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(content.len());
    let mut in_string = false;
    let mut escaped = false;
    for (index, character) in chars.iter().copied().enumerate() {
        if in_string {
            output.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }
        if character == '"' {
            in_string = true;
            output.push(character);
            continue;
        }
        if character == ',' {
            let next = chars[index + 1..]
                .iter()
                .copied()
                .find(|next| !next.is_whitespace());
            if matches!(next, Some('}' | ']')) {
                continue;
            }
        }
        output.push(character);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{Message, ModelId, Role, ToolName};
    use crate::inference::{ModelTool, SamplingOptions, TokenUsage};

    fn request(choice: ToolChoice) -> ModelRequest {
        ModelRequest {
            conversation: Conversation::new(vec![ConversationItem::Message(
                Message::text(Role::User, "Inspect the file").unwrap(),
            )]),
            tools: ModelToolSet::new(vec![ModelTool {
                name: ToolName::new("inspect_file").unwrap(),
                description: Some("Inspect one file".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "line_start": {"type": "integer"}
                    },
                    "required": ["path"]
                }),
                strict: Some(false),
            }]),
            tool_choice: choice,
            sampling: SamplingOptions::default(),
            max_output_tokens: 256,
            stream: false,
        }
    }

    #[test]
    fn request_uses_dynamic_json_schema_without_native_tool_fields() {
        let body = build_generic_json_request(
            &ModelId::new("gemma4").unwrap(),
            &request(ToolChoice::Auto),
            &ModelConfig::new_legacy("/models".to_string(), "model.gguf".to_string(), Vec::new()),
            &LlamaServerChatDefaults {
                temperature: 0.2,
                top_p: 0.8,
                repeat_penalty: 1.1,
            },
        )
        .unwrap();

        assert!(body.get("tools").is_none());
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(
            body["response_format"]["json_schema"]["schema"]["oneOf"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn observed_complex_fixture_preserves_argument_types() {
        let fixture: Value = serde_json::from_slice(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/llama-server/10075/gemma4-generic-json-complex-response.json"
        )))
        .unwrap();
        let completion = ModelCompletion::Text {
            content: fixture["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .to_string(),
            usage: TokenUsage::Measured {
                input_tokens: 85,
                output_tokens: 114,
            },
        };

        let parsed = parse_generic_json_completion(completion, true).unwrap();
        let ModelCompletion::ToolCall { call, .. } = parsed else {
            panic!("fixture must parse as Tool Call");
        };
        let arguments: Value = serde_json::from_str(&call.arguments).unwrap();
        assert_eq!(arguments["line_start"], 17);
        assert_eq!(arguments["include_hidden"], false);
        assert!(arguments["tags"].is_array());
        assert!(arguments["options"].is_object());
    }

    #[test]
    fn deterministic_repair_only_removes_fence_and_trailing_comma() {
        let completion = ModelCompletion::Text {
            content: "```json\n{\"type\":\"final\",\"content\":\"OK\",}\n```".to_string(),
            usage: TokenUsage::Measured {
                input_tokens: 1,
                output_tokens: 1,
            },
        };

        let parsed = parse_generic_json_completion(completion, true).unwrap();
        assert!(matches!(
            parsed,
            ModelCompletion::Text { content, .. } if content == "OK"
        ));
    }
}
