use crate::config::UnknownFieldPolicy;
use crate::conversation::{
    CallId, ContentPart, Conversation, ConversationError, ConversationItem, FunctionCall,
    FunctionCallOutput, Message, ModelId, Role, ToolArguments, ToolName, ToolOutcome,
};
use crate::inference::{ModelTool, ModelToolSet, SamplingOptions, TokenUsage, ToolChoice};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeSet, HashSet};
use thiserror::Error;

#[derive(Serialize)]
pub struct CompletedResponseWire {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    pub status: &'static str,
    pub model: String,
    pub output: Vec<CompletedOutputItemWire>,
    pub usage: UsageWire,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum CompletedOutputItemWire {
    Message(OutputMessageWire),
    FunctionCall(OutputFunctionCallWire),
}

#[derive(Serialize)]
pub struct OutputMessageWire {
    pub id: String,
    pub r#type: &'static str,
    pub role: &'static str,
    pub status: &'static str,
    pub content: Vec<OutputContentWire>,
}

#[derive(Serialize)]
pub struct OutputFunctionCallWire {
    pub id: String,
    pub r#type: &'static str,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub status: &'static str,
}

#[derive(Serialize)]
pub struct OutputContentWire {
    pub r#type: &'static str,
    pub text: String,
    pub annotations: Vec<Value>,
}

#[derive(Debug, Serialize)]
pub struct UsageWire {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

impl From<crate::application::CompletedResponse> for CompletedResponseWire {
    fn from(response: crate::application::CompletedResponse) -> Self {
        let (input_tokens, output_tokens) = match response.usage {
            TokenUsage::Measured {
                input_tokens,
                output_tokens,
            }
            | TokenUsage::Estimated {
                input_tokens,
                output_tokens,
            } => (input_tokens, output_tokens),
        };
        let output = match response.output {
            crate::application::CompletedOutput::Message(message) => {
                CompletedOutputItemWire::Message(OutputMessageWire {
                    id: message.id.as_str().to_string(),
                    r#type: "message",
                    role: "assistant",
                    status: "completed",
                    content: vec![OutputContentWire {
                        r#type: "output_text",
                        text: message.text,
                        annotations: Vec::new(),
                    }],
                })
            }
            crate::application::CompletedOutput::FunctionCall(call) => {
                CompletedOutputItemWire::FunctionCall(OutputFunctionCallWire {
                    id: call.id.as_str().to_string(),
                    r#type: "function_call",
                    call_id: call.call_id.as_str().to_string(),
                    name: call.name.as_str().to_string(),
                    arguments: call.arguments.canonical_json().to_string(),
                    status: "completed",
                })
            }
        };
        Self {
            id: response.id.as_str().to_string(),
            object: "response",
            created_at: response.created_at,
            status: "completed",
            model: response.model.as_str().to_string(),
            output: vec![output],
            usage: UsageWire {
                input_tokens,
                output_tokens,
                total_tokens: input_tokens.saturating_add(output_tokens),
            },
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct WireRequestError {
    pub code: &'static str,
    pub param: Option<String>,
    pub message: String,
}

pub fn decode_request(
    value: Value,
    policy: UnknownFieldPolicy,
) -> Result<crate::application::NormalizedResponsesRequest, WireRequestError> {
    decode_request_with_limits(
        value,
        policy,
        crate::application::ResponsesRequestLimits::default(),
    )
}

pub fn decode_request_with_limits(
    value: Value,
    policy: UnknownFieldPolicy,
    limits: crate::application::ResponsesRequestLimits,
) -> Result<crate::application::NormalizedResponsesRequest, WireRequestError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("request body must be a JSON object", None))?;
    let supported = BTreeSet::from([
        "input",
        "instructions",
        "max_output_tokens",
        "model",
        "stream",
        "temperature",
        "tool_choice",
        "tools",
        "top_p",
    ]);
    let compatible_only = BTreeSet::from([
        "client_metadata",
        "include",
        "metadata",
        "parallel_tool_calls",
        "prompt_cache_key",
        "reasoning",
        "safety_identifier",
        "store",
        "text",
    ]);
    let mut warnings = Vec::new();
    for key in object.keys() {
        if supported.contains(key.as_str()) {
            continue;
        }
        if key == "previous_response_id" {
            continue;
        }
        if compatible_only.contains(key.as_str()) || !supported.contains(key.as_str()) {
            if policy == UnknownFieldPolicy::Strict {
                return Err(unsupported(
                    key,
                    format!("parameter {key} is not supported in strict mode"),
                ));
            }
            warnings.push(key.clone());
        }
    }
    warnings.sort();

    if object
        .get("stream")
        .is_some_and(|stream| !stream.is_null() && !stream.is_boolean())
    {
        return Err(invalid("stream must be a boolean", Some("stream")));
    }
    let tools = normalize_tools(object.get("tools"), policy, limits, &mut warnings)?;
    let tool_choice = normalize_tool_choice(object.get("tool_choice"), &tools)?;
    warnings.sort();

    let model = object
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("model is required and must be a string", Some("model")))
        .and_then(|model| {
            ModelId::new(model).map_err(|_| invalid("model is invalid", Some("model")))
        })?;
    let input = object
        .get("input")
        .ok_or_else(|| invalid("input is required", Some("input")))?;
    let mut items = Vec::new();
    if let Some(instructions) = object.get("instructions").filter(|value| !value.is_null()) {
        let instructions = instructions
            .as_str()
            .ok_or_else(|| invalid("instructions must be a string", Some("instructions")))?;
        items.push(ConversationItem::Message(
            Message::text(Role::Developer, instructions)
                .map_err(|_| invalid("instructions are invalid", Some("instructions")))?,
        ));
    }
    normalize_input(input, &mut items, limits)?;
    let conversation = Conversation::new(items);
    conversation.validate().map_err(conversation_error)?;
    if let Some(previous_response_id) = object
        .get("previous_response_id")
        .filter(|value| !value.is_null())
    {
        let valid_id = previous_response_id
            .as_str()
            .is_some_and(|value| !value.trim().is_empty());
        let has_message = conversation
            .items()
            .iter()
            .any(|item| matches!(item, ConversationItem::Message(_)));
        let has_completed_call = conversation
            .items()
            .iter()
            .any(|item| matches!(item, ConversationItem::FunctionCallOutput(_)));
        if !valid_id || !has_message || !has_completed_call {
            return Err(WireRequestError {
                code: "previous_response_not_supported",
                param: Some("previous_response_id".to_string()),
                message: "previous_response_id requires a complete stateless Tool history"
                    .to_string(),
            });
        }
        warnings.push("previous_response_id".to_string());
        warnings.sort();
    }

    let sampling = SamplingOptions {
        temperature: optional_f32(object, "temperature")?,
        top_p: optional_f32(object, "top_p")?,
        presence_penalty: None,
        frequency_penalty: None,
    };
    if sampling
        .temperature
        .is_some_and(|temperature| !(0.0..=2.0).contains(&temperature))
    {
        return Err(invalid(
            "temperature must be between 0 and 2",
            Some("temperature"),
        ));
    }
    if sampling
        .top_p
        .is_some_and(|top_p| !(0.0 < top_p && top_p <= 1.0))
    {
        return Err(invalid(
            "top_p must be greater than 0 and at most 1",
            Some("top_p"),
        ));
    }
    let max_output_tokens = match object
        .get("max_output_tokens")
        .filter(|value| !value.is_null())
    {
        Some(value) => {
            let value = value.as_u64().ok_or_else(|| {
                invalid(
                    "max_output_tokens must be a positive integer",
                    Some("max_output_tokens"),
                )
            })?;
            u32::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    invalid(
                        "max_output_tokens must be a positive 32-bit integer",
                        Some("max_output_tokens"),
                    )
                })?
        }
        None => 1024,
    };
    let stream = object
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    Ok(crate::application::NormalizedResponsesRequest {
        model,
        conversation,
        tools,
        tool_choice,
        sampling,
        max_output_tokens,
        stream,
        warnings,
    })
}

fn normalize_input(
    input: &Value,
    items: &mut Vec<ConversationItem>,
    limits: crate::application::ResponsesRequestLimits,
) -> Result<(), WireRequestError> {
    if let Some(text) = input.as_str() {
        items.push(ConversationItem::Message(
            Message::text(Role::User, text)
                .map_err(|_| invalid("input is invalid", Some("input")))?,
        ));
        return Ok(());
    }
    let input_items = input
        .as_array()
        .ok_or_else(|| invalid("input must be a string or an item array", Some("input")))?;
    if input_items.is_empty() {
        return Err(invalid("input must not be empty", Some("input")));
    }
    for item in input_items {
        let object = item
            .as_object()
            .ok_or_else(|| invalid("input item must be an object", Some("input")))?;
        match object.get("type").and_then(Value::as_str) {
            Some("message") => {
                let role = object
                    .get("role")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("message role is required", Some("input")))
                    .and_then(|role| {
                        Role::parse(role)
                            .map_err(|_| invalid("message role is invalid", Some("input")))
                    })?;
                let content = object
                    .get("content")
                    .ok_or_else(|| invalid("message content is required", Some("input")))?;
                let content = normalize_message_content(content)?;
                items.push(ConversationItem::Message(
                    Message::new(role, content)
                        .map_err(|_| invalid("message content is invalid", Some("input")))?,
                ));
            }
            Some("function_call") => {
                items.push(ConversationItem::FunctionCall(normalize_function_call(
                    object, limits,
                )?));
            }
            Some("function_call_output") => {
                items.push(ConversationItem::FunctionCallOutput(
                    normalize_function_call_output(object, limits)?,
                ));
            }
            _ => {
                return Err(invalid("unsupported input item type", Some("input")));
            }
        }
    }
    Ok(())
}

fn normalize_function_call(
    object: &serde_json::Map<String, Value>,
    limits: crate::application::ResponsesRequestLimits,
) -> Result<FunctionCall, WireRequestError> {
    let call_id = object
        .get("call_id")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("function_call call_id is required", Some("input")))
        .and_then(|value| {
            CallId::new(value)
                .map_err(|_| invalid("function_call call_id is invalid", Some("input")))
        })?;
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("function_call name is required", Some("input")))
        .and_then(|value| {
            ToolName::new(value)
                .map_err(|_| invalid("function_call name is invalid", Some("input")))
        })?;
    let arguments = object
        .get("arguments")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            invalid(
                "function_call arguments must be a JSON string",
                Some("input"),
            )
        })
        .and_then(|value| {
            if value.len() > limits.max_tool_argument_bytes {
                return Err(WireRequestError {
                    code: "invalid_tool_arguments",
                    param: Some("input".to_string()),
                    message: "function_call arguments exceed the configured size limit".to_string(),
                });
            }
            ToolArguments::parse(value).map_err(|_| WireRequestError {
                code: "invalid_tool_arguments",
                param: Some("input".to_string()),
                message: "function_call arguments are not valid JSON".to_string(),
            })
        })?;
    Ok(FunctionCall {
        call_id,
        name,
        arguments,
    })
}

fn normalize_function_call_output(
    object: &serde_json::Map<String, Value>,
    limits: crate::application::ResponsesRequestLimits,
) -> Result<FunctionCallOutput, WireRequestError> {
    let call_id = object
        .get("call_id")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("function_call_output call_id is required", Some("input")))
        .and_then(|value| {
            CallId::new(value)
                .map_err(|_| invalid("function_call_output call_id is invalid", Some("input")))
        })?;
    let output = object
        .get("output")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            invalid(
                "function_call_output output must be a string",
                Some("input"),
            )
        })?
        .to_string();
    if output.len() > limits.max_tool_result_bytes {
        return Err(WireRequestError {
            code: "context_length_exceeded",
            param: Some("input".to_string()),
            message: "function_call_output exceeds the configured size limit".to_string(),
        });
    }
    let outcome = match object.get("status").and_then(Value::as_str) {
        None | Some("completed" | "success") => ToolOutcome::Success(output),
        Some("failed" | "error") => ToolOutcome::Failure(output),
        Some("rejected") => ToolOutcome::Rejected(output),
        Some("cancelled") => ToolOutcome::Cancelled(output),
        Some(_) => {
            return Err(invalid(
                "function_call_output status is invalid",
                Some("input"),
            ));
        }
    };
    Ok(FunctionCallOutput { call_id, outcome })
}

fn normalize_message_content(content: &Value) -> Result<Vec<ContentPart>, WireRequestError> {
    if let Some(text) = content.as_str() {
        return Ok(vec![ContentPart::Text(text.to_string())]);
    }
    let parts = content
        .as_array()
        .ok_or_else(|| invalid("message content must be a string or array", Some("input")))?;
    if parts.is_empty() {
        return Err(invalid("message content must not be empty", Some("input")));
    }
    parts
        .iter()
        .map(|part| {
            let object = part
                .as_object()
                .ok_or_else(|| invalid("content part must be an object", Some("input")))?;
            match object.get("type").and_then(Value::as_str) {
                Some("input_text" | "output_text") => object
                    .get("text")
                    .and_then(Value::as_str)
                    .map(|text| ContentPart::Text(text.to_string()))
                    .ok_or_else(|| invalid("text content is required", Some("input"))),
                Some("input_image") => Err(WireRequestError {
                    code: "unsupported_parameter",
                    param: Some("input".to_string()),
                    message: "Image input is not available in the text-only Responses phase"
                        .to_string(),
                }),
                _ => Err(invalid("unsupported content part type", Some("input"))),
            }
        })
        .collect()
}

fn normalize_tools(
    value: Option<&Value>,
    policy: UnknownFieldPolicy,
    limits: crate::application::ResponsesRequestLimits,
    warnings: &mut Vec<String>,
) -> Result<ModelToolSet, WireRequestError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(ModelToolSet::default());
    };
    let definitions = value
        .as_array()
        .ok_or_else(|| invalid("tools must be an array", Some("tools")))?;
    let mut tools = Vec::new();
    let mut names = HashSet::new();
    let mut total_schema_bytes = 0_usize;

    for (index, definition) in definitions.iter().enumerate() {
        let object = definition
            .as_object()
            .ok_or_else(|| invalid("tool definition must be an object", Some("tools")))?;
        let tool_type = object
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("tool type is required", Some("tools")))?;
        if matches!(tool_type, "namespace" | "web_search") {
            if policy == UnknownFieldPolicy::Strict {
                return Err(WireRequestError {
                    code: "unsupported_tool_type",
                    param: Some("tools".to_string()),
                    message: format!("tool type {tool_type} is not supported in strict mode"),
                });
            }
            warnings.push(format!("tools[{index}].type={tool_type}"));
            continue;
        }
        if tool_type != "function" {
            return Err(WireRequestError {
                code: "unsupported_tool_type",
                param: Some("tools".to_string()),
                message: format!("tool type {tool_type} is not supported"),
            });
        }
        if tools.len() >= limits.max_tools {
            return Err(WireRequestError {
                code: "invalid_tool_schema",
                param: Some("tools".to_string()),
                message: "function tool count exceeds the configured limit".to_string(),
            });
        }
        let name = object
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_tool_schema("function tool name is required"))
            .and_then(|name| {
                ToolName::new(name)
                    .map_err(|_| invalid_tool_schema("function tool name is invalid"))
            })?;
        if !names.insert(name.as_str().to_string()) {
            return Err(invalid_tool_schema("function tool names must be unique"));
        }
        let description = object
            .get("description")
            .filter(|value| !value.is_null())
            .map(|value| {
                value
                    .as_str()
                    .map(ToString::to_string)
                    .ok_or_else(|| invalid_tool_schema("tool description must be a string"))
            })
            .transpose()?;
        let parameters = object
            .get("parameters")
            .ok_or_else(|| invalid_tool_schema("function tool parameters are required"))?;
        if !parameters.is_object() {
            return Err(invalid_tool_schema(
                "function tool parameters must be a JSON Schema object",
            ));
        }
        let schema_bytes = serde_json::to_vec(parameters)
            .map_err(|_| invalid_tool_schema("function tool parameters cannot be serialized"))?
            .len();
        if schema_bytes > limits.max_single_tool_schema_bytes {
            return Err(invalid_tool_schema(
                "function tool schema exceeds the per-tool size limit",
            ));
        }
        total_schema_bytes = total_schema_bytes.saturating_add(schema_bytes);
        if total_schema_bytes > limits.max_tool_schema_bytes {
            return Err(invalid_tool_schema(
                "function tool schemas exceed the total size limit",
            ));
        }
        jsonschema::draft7::new(parameters).map_err(|_| {
            invalid_tool_schema("function tool parameters are not valid JSON Schema")
        })?;
        let strict = object
            .get("strict")
            .filter(|value| !value.is_null())
            .map(|value| {
                value
                    .as_bool()
                    .ok_or_else(|| invalid_tool_schema("tool strict must be a boolean"))
            })
            .transpose()?;
        tools.push(ModelTool {
            name,
            description,
            parameters: parameters.clone(),
            strict,
        });
    }

    Ok(ModelToolSet::new(tools))
}

fn normalize_tool_choice(
    value: Option<&Value>,
    tools: &ModelToolSet,
) -> Result<ToolChoice, WireRequestError> {
    match value {
        None | Some(Value::Null) => Ok(ToolChoice::Auto),
        Some(Value::String(choice)) if choice == "auto" => Ok(ToolChoice::Auto),
        Some(Value::String(choice)) if choice == "none" => Ok(ToolChoice::None),
        Some(Value::String(choice)) if choice == "required" && !tools.tools().is_empty() => {
            Ok(ToolChoice::Required)
        }
        Some(Value::String(choice)) if choice == "required" => Err(invalid(
            "tool_choice required needs at least one function tool",
            Some("tool_choice"),
        )),
        Some(Value::Object(choice))
            if choice.get("type").and_then(Value::as_str) == Some("function") =>
        {
            let name = choice
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("named tool_choice requires name", Some("tool_choice")))
                .and_then(|name| {
                    ToolName::new(name).map_err(|_| {
                        invalid("named tool_choice name is invalid", Some("tool_choice"))
                    })
                })?;
            if !tools.tools().iter().any(|tool| tool.name == name) {
                return Err(invalid(
                    "named tool_choice does not match a provided function tool",
                    Some("tool_choice"),
                ));
            }
            Ok(ToolChoice::Function(name))
        }
        Some(_) => Err(unsupported(
            "tool_choice",
            "tool_choice format is not supported",
        )),
    }
}

fn conversation_error(error: ConversationError) -> WireRequestError {
    match error {
        ConversationError::OrphanCallOutput(_) => WireRequestError {
            code: "orphan_function_call_output",
            param: Some("input".to_string()),
            message: "function_call_output has no preceding function_call".to_string(),
        },
        ConversationError::InvalidToolArguments(_) => WireRequestError {
            code: "invalid_tool_arguments",
            param: Some("input".to_string()),
            message: "function_call arguments are invalid".to_string(),
        },
        _ => invalid("input conversation is invalid", Some("input")),
    }
}

fn invalid_tool_schema(message: impl Into<String>) -> WireRequestError {
    WireRequestError {
        code: "invalid_tool_schema",
        param: Some("tools".to_string()),
        message: message.into(),
    }
}

fn optional_f32(
    object: &serde_json::Map<String, Value>,
    name: &'static str,
) -> Result<Option<f32>, WireRequestError> {
    object
        .get(name)
        .map(|value| {
            if value.is_null() {
                return Ok(None);
            }
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .map(|value| value as f32)
                .map(Some)
                .ok_or_else(|| invalid(format!("{name} must be a number"), Some(name)))
        })
        .transpose()
        .map(Option::flatten)
}

fn invalid(message: impl Into<String>, param: Option<&str>) -> WireRequestError {
    WireRequestError {
        code: "invalid_request",
        param: param.map(ToString::to_string),
        message: message.into(),
    }
}

fn unsupported(param: &str, message: impl Into<String>) -> WireRequestError {
    WireRequestError {
        code: "unsupported_parameter",
        param: Some(param.to_string()),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_input_and_instructions_normalize_to_conversation() {
        let decoded = decode_request(
            serde_json::json!({
                "model": "gemma4",
                "instructions": "Answer briefly.",
                "input": "Return OK.",
                "temperature": 0.1,
                "top_p": 0.9,
                "max_output_tokens": 64,
                "stream": false
            }),
            UnknownFieldPolicy::Compatible,
        )
        .unwrap();

        assert_eq!(decoded.model.as_str(), "gemma4");
        assert_eq!(decoded.conversation.summary().messages, 2);
        assert_eq!(decoded.sampling.temperature, Some(0.1));
        assert_eq!(decoded.sampling.top_p, Some(0.9));
        assert_eq!(decoded.max_output_tokens, 64);
    }

    #[test]
    fn message_item_array_preserves_roles_and_content_order() {
        let decoded = decode_request(
            serde_json::json!({
                "model": "gemma4",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [
                            {"type": "input_text", "text": "Rule 1"},
                            {"type": "input_text", "text": "Rule 2"}
                        ]
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Question"
                    }
                ]
            }),
            UnknownFieldPolicy::Compatible,
        )
        .unwrap();

        assert_eq!(decoded.conversation.summary().messages, 2);
        let ConversationItem::Message(message) = &decoded.conversation.items()[0] else {
            panic!("first item must be a message");
        };
        assert_eq!(message.content.len(), 2);
        assert_eq!(message.text_content(), "Rule 1\nRule 2");
    }

    #[test]
    fn nullable_optional_fields_are_treated_as_unspecified() {
        let decoded = decode_request(
            serde_json::json!({
                "model": "gemma4",
                "input": "Hello",
                "instructions": null,
                "stream": null,
                "temperature": null,
                "top_p": null,
                "max_output_tokens": null,
                "tools": null,
                "tool_choice": null
            }),
            UnknownFieldPolicy::Compatible,
        )
        .unwrap();

        assert_eq!(decoded.conversation.summary().messages, 1);
        assert_eq!(decoded.max_output_tokens, 1024);
        assert_eq!(decoded.sampling, SamplingOptions::default());
    }

    #[test]
    fn unknown_top_level_is_warning_in_compatible_and_error_in_strict() {
        let value = serde_json::json!({
            "model": "gemma4",
            "input": "Hello",
            "future_field": true
        });

        let compatible = decode_request(value.clone(), UnknownFieldPolicy::Compatible).unwrap();
        assert_eq!(compatible.warnings, vec!["future_field"]);

        let strict = decode_request(value, UnknownFieldPolicy::Strict)
            .err()
            .unwrap();
        assert_eq!(strict.code, "unsupported_parameter");
        assert_eq!(strict.param.as_deref(), Some("future_field"));
    }

    #[test]
    fn streaming_flag_is_preserved_for_the_sse_service_path() {
        let stream = decode_request(
            serde_json::json!({
                "model": "gemma4",
                "input": "Hello",
                "stream": true
            }),
            UnknownFieldPolicy::Compatible,
        )
        .unwrap();
        assert!(stream.stream);
    }

    #[test]
    fn function_tools_and_named_choice_normalize_to_domain_contract() {
        let decoded = decode_request(
            serde_json::json!({
                "model": "gemma4",
                "input": "Hello",
                "tools": [{
                    "type": "function",
                    "name": "read_file",
                    "description": "Read one file",
                    "strict": true,
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": false
                    }
                }],
                "tool_choice": {"type": "function", "name": "read_file"}
            }),
            UnknownFieldPolicy::Compatible,
        )
        .unwrap();

        assert_eq!(decoded.tools.tools().len(), 1);
        assert_eq!(decoded.tools.tools()[0].name.as_str(), "read_file");
        assert_eq!(decoded.tools.tools()[0].strict, Some(true));
        assert!(matches!(
            decoded.tool_choice,
            ToolChoice::Function(ref name) if name.as_str() == "read_file"
        ));
    }

    #[test]
    fn function_call_and_output_preserve_identity_and_outcome() {
        let decoded = decode_request(
            serde_json::json!({
                "model": "gemma4",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": "{\"path\":\"README.md\"}"
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "Hoshikage"
                    }
                ]
            }),
            UnknownFieldPolicy::Compatible,
        )
        .unwrap();

        let summary = decoded.conversation.summary();
        assert_eq!(summary.function_calls, 1);
        assert_eq!(summary.function_outputs, 1);
        let ConversationItem::FunctionCallOutput(output) = &decoded.conversation.items()[1] else {
            panic!("second item must be function_call_output");
        };
        assert!(matches!(
            output.outcome,
            ToolOutcome::Success(ref content) if content == "Hoshikage"
        ));
    }

    #[test]
    fn orphan_function_output_has_stable_error_code() {
        let error = decode_request(
            serde_json::json!({
                "model": "gemma4",
                "input": [{
                    "type": "function_call_output",
                    "call_id": "call_missing",
                    "output": "orphan"
                }]
            }),
            UnknownFieldPolicy::Compatible,
        )
        .err()
        .unwrap();

        assert_eq!(error.code, "orphan_function_call_output");
        assert_eq!(error.param.as_deref(), Some("input"));
    }

    #[test]
    fn previous_response_id_is_ignored_only_with_complete_stateless_tool_history() {
        let complete = decode_request(
            serde_json::json!({
                "model": "gemma4",
                "previous_response_id": "resp_previous",
                "input": [
                    {"type": "message", "role": "user", "content": "Read README"},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": "{\"path\":\"README.md\"}"
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "Hoshikage"
                    }
                ]
            }),
            UnknownFieldPolicy::Compatible,
        )
        .unwrap();
        assert!(complete
            .warnings
            .contains(&"previous_response_id".to_string()));

        let error = decode_request(
            serde_json::json!({
                "model": "gemma4",
                "previous_response_id": "resp_previous",
                "input": "Continue"
            }),
            UnknownFieldPolicy::Compatible,
        )
        .err()
        .unwrap();
        assert_eq!(error.code, "previous_response_not_supported");
        assert_eq!(error.param.as_deref(), Some("previous_response_id"));
    }

    #[test]
    fn known_codex_auxiliary_tools_warn_in_compatible_and_fail_in_strict() {
        let value = serde_json::json!({
            "model": "gemma4",
            "input": "Hello",
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "parameters": {"type": "object"}
                },
                {"type": "namespace", "name": "multi_agent"},
                {"type": "web_search"}
            ]
        });

        let compatible = decode_request(value.clone(), UnknownFieldPolicy::Compatible).unwrap();
        assert_eq!(compatible.tools.tools().len(), 1);
        assert_eq!(
            compatible.warnings,
            ["tools[1].type=namespace", "tools[2].type=web_search"]
        );

        let strict = decode_request(value, UnknownFieldPolicy::Strict)
            .err()
            .unwrap();
        assert_eq!(strict.code, "unsupported_tool_type");
    }

    #[test]
    fn invalid_or_duplicate_tool_schemas_are_rejected() {
        let invalid_schema = decode_request(
            serde_json::json!({
                "model": "gemma4",
                "input": "Hello",
                "tools": [{
                    "type": "function",
                    "name": "read_file",
                    "parameters": {"type": "not-a-json-schema-type"}
                }]
            }),
            UnknownFieldPolicy::Compatible,
        )
        .err()
        .unwrap();
        assert_eq!(invalid_schema.code, "invalid_tool_schema");

        let duplicate = decode_request(
            serde_json::json!({
                "model": "gemma4",
                "input": "Hello",
                "tools": [
                    {"type": "function", "name": "read_file", "parameters": {}},
                    {"type": "function", "name": "read_file", "parameters": {}}
                ]
            }),
            UnknownFieldPolicy::Compatible,
        )
        .err()
        .unwrap();
        assert_eq!(duplicate.code, "invalid_tool_schema");
    }

    #[test]
    fn completed_text_serializes_as_responses_output_message() {
        let completed = crate::application::CompletedResponse {
            id: crate::conversation::ResponseId::new("resp_test").unwrap(),
            created_at: 1_780_000_000,
            model: ModelId::new("gemma4").unwrap(),
            output: crate::application::CompletedOutput::Message(
                crate::application::CompletedMessage {
                    id: crate::conversation::OutputItemId::new("msg_test").unwrap(),
                    text: "OK".to_string(),
                },
            ),
            usage: TokenUsage::Measured {
                input_tokens: 10,
                output_tokens: 1,
            },
        };

        let value = serde_json::to_value(CompletedResponseWire::from(completed)).unwrap();

        assert_eq!(value["object"], "response");
        assert_eq!(value["status"], "completed");
        assert_eq!(value["output"][0]["type"], "message");
        assert_eq!(value["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(value["output"][0]["content"][0]["text"], "OK");
        assert_eq!(value["usage"]["total_tokens"], 11);
    }

    #[test]
    fn completed_tool_call_serializes_as_responses_function_call() {
        let completed = crate::application::CompletedResponse {
            id: crate::conversation::ResponseId::new("resp_tool").unwrap(),
            created_at: 1_780_000_000,
            model: ModelId::new("gemma4").unwrap(),
            output: crate::application::CompletedOutput::FunctionCall(
                crate::application::CompletedFunctionCall {
                    id: crate::conversation::OutputItemId::new("fc_tool").unwrap(),
                    call_id: CallId::new("call_tool").unwrap(),
                    name: ToolName::new("read_file").unwrap(),
                    arguments: ToolArguments::parse(r#"{"path":"README.md"}"#).unwrap(),
                },
            ),
            usage: TokenUsage::Measured {
                input_tokens: 20,
                output_tokens: 5,
            },
        };

        let value = serde_json::to_value(CompletedResponseWire::from(completed)).unwrap();

        assert_eq!(value["output"][0]["type"], "function_call");
        assert_eq!(value["output"][0]["call_id"], "call_tool");
        assert_eq!(value["output"][0]["name"], "read_file");
        assert_eq!(value["output"][0]["arguments"], r#"{"path":"README.md"}"#);
        assert_eq!(value["output"][0]["status"], "completed");
    }

    #[test]
    fn responses_wire_conversion_p95_is_below_fifty_milliseconds() {
        let request = serde_json::json!({
            "model": "gemma4",
            "instructions": "Answer briefly.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Return OK."}]
                }
            ],
            "max_output_tokens": 64
        });
        let mut samples = Vec::with_capacity(500);
        for _ in 0..500 {
            let started = std::time::Instant::now();
            let decoded = decode_request(request.clone(), UnknownFieldPolicy::Compatible).unwrap();
            let completed = crate::application::CompletedResponse {
                id: crate::conversation::ResponseId::new("resp_perf").unwrap(),
                created_at: 1_780_000_000,
                model: decoded.model,
                output: crate::application::CompletedOutput::Message(
                    crate::application::CompletedMessage {
                        id: crate::conversation::OutputItemId::new("msg_perf").unwrap(),
                        text: "OK".to_string(),
                    },
                ),
                usage: TokenUsage::Measured {
                    input_tokens: 10,
                    output_tokens: 1,
                },
            };
            let _ = serde_json::to_value(CompletedResponseWire::from(completed)).unwrap();
            samples.push(started.elapsed());
        }
        samples.sort_unstable();
        let p95 = samples[(samples.len() * 95 / 100).saturating_sub(1)];
        println!("Responses wire conversion p95: {p95:?}");

        assert!(
            p95 < std::time::Duration::from_millis(50),
            "Responses wire conversion p95 was {p95:?}"
        );
    }
}
