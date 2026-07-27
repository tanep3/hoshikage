use crate::config::UnknownFieldPolicy;
use crate::conversation::{ContentPart, Conversation, ConversationItem, Message, ModelId, Role};
use crate::inference::{SamplingOptions, TokenUsage};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Debug, Serialize)]
pub struct CompletedResponseWire {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    pub status: &'static str,
    pub model: String,
    pub output: Vec<OutputItemWire>,
    pub usage: UsageWire,
}

#[derive(Debug, Serialize)]
pub struct OutputItemWire {
    pub id: String,
    pub r#type: &'static str,
    pub role: &'static str,
    pub status: &'static str,
    pub content: Vec<OutputContentWire>,
}

#[derive(Debug, Serialize)]
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
        Self {
            id: response.id.as_str().to_string(),
            object: "response",
            created_at: response.created_at,
            status: "completed",
            model: response.model.as_str().to_string(),
            output: vec![OutputItemWire {
                id: response.message.id.as_str().to_string(),
                r#type: "message",
                role: "assistant",
                status: "completed",
                content: vec![OutputContentWire {
                    r#type: "output_text",
                    text: response.message.text,
                    annotations: Vec::new(),
                }],
            }],
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
            return Err(unsupported(
                "previous_response_id",
                "previous_response_id is not supported by this stateless provider",
            ));
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

    if object.get("stream").and_then(Value::as_bool) == Some(true) {
        return Err(unsupported(
            "stream",
            "streaming Responses are not available in this release phase",
        ));
    }
    if object
        .get("stream")
        .is_some_and(|stream| !stream.is_null() && !stream.is_boolean())
    {
        return Err(invalid("stream must be a boolean", Some("stream")));
    }
    if object
        .get("tools")
        .is_some_and(|tools| !tools.is_null() && tools.as_array().is_none())
    {
        return Err(invalid("tools must be an array", Some("tools")));
    }
    if object
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty())
    {
        return Err(WireRequestError {
            code: "tool_calling_not_supported",
            param: Some("tools".to_string()),
            message: "Tool calling is not available in the text-only Responses phase".to_string(),
        });
    }
    validate_tool_choice(object.get("tool_choice"))?;

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
    normalize_input(input, &mut items)?;
    let conversation = Conversation::new(items);
    conversation
        .validate()
        .map_err(|_| invalid("input conversation is invalid", Some("input")))?;

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

    Ok(crate::application::NormalizedResponsesRequest {
        model,
        conversation,
        sampling,
        max_output_tokens,
        warnings,
    })
}

fn normalize_input(
    input: &Value,
    items: &mut Vec<ConversationItem>,
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
        if object.get("type").and_then(Value::as_str) != Some("message") {
            return Err(invalid(
                "only message input items are supported in the text-only phase",
                Some("input"),
            ));
        }
        let role = object
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("message role is required", Some("input")))
            .and_then(|role| {
                Role::parse(role).map_err(|_| invalid("message role is invalid", Some("input")))
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
    Ok(())
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

fn validate_tool_choice(value: Option<&Value>) -> Result<(), WireRequestError> {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(choice)) if matches!(choice.as_str(), "auto" | "none") => Ok(()),
        Some(Value::String(choice)) if choice == "required" => Err(WireRequestError {
            code: "tool_calling_not_supported",
            param: Some("tool_choice".to_string()),
            message: "Required Tool calling is not available in this release phase".to_string(),
        }),
        Some(_) => Err(WireRequestError {
            code: "tool_calling_not_supported",
            param: Some("tool_choice".to_string()),
            message: "Named Tool calling is not available in this release phase".to_string(),
        }),
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
    fn streaming_and_tool_requests_are_rejected_in_text_only_phase() {
        let stream = decode_request(
            serde_json::json!({
                "model": "gemma4",
                "input": "Hello",
                "stream": true
            }),
            UnknownFieldPolicy::Compatible,
        )
        .err()
        .unwrap();
        assert_eq!(stream.code, "unsupported_parameter");
        assert_eq!(stream.param.as_deref(), Some("stream"));

        let tools = decode_request(
            serde_json::json!({
                "model": "gemma4",
                "input": "Hello",
                "tools": [{"type": "function", "name": "read_file"}]
            }),
            UnknownFieldPolicy::Compatible,
        )
        .err()
        .unwrap();
        assert_eq!(tools.code, "tool_calling_not_supported");
        assert_eq!(tools.param.as_deref(), Some("tools"));
    }

    #[test]
    fn completed_text_serializes_as_responses_output_message() {
        let completed = crate::application::CompletedResponse {
            id: crate::conversation::ResponseId::new("resp_test").unwrap(),
            created_at: 1_780_000_000,
            model: ModelId::new("gemma4").unwrap(),
            message: crate::application::CompletedMessage {
                id: crate::conversation::OutputItemId::new("msg_test").unwrap(),
                text: "OK".to_string(),
            },
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
                message: crate::application::CompletedMessage {
                    id: crate::conversation::OutputItemId::new("msg_perf").unwrap(),
                    text: "OK".to_string(),
                },
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
