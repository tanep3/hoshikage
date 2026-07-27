use super::{ModelCompletion, ModelRequest, TokenUsage};
use crate::conversation::{ContentPart, ConversationItem, ModelId};
use crate::model::{ModelConfig, ThinkingMode};
use serde::Deserialize;

pub struct LlamaServerTextDefaults {
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

pub fn build_text_request(
    model: &ModelId,
    request: &ModelRequest,
    model_config: &ModelConfig,
    defaults: &LlamaServerTextDefaults,
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
            ConversationItem::FunctionCall(_) | ConversationItem::FunctionCallOutput(_) => {
                Err(crate::error::HoshikageError::InferenceError(
                    "Responses text request contained Tool history".to_string(),
                ))
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
        "stream": false,
        "temperature": request.sampling.temperature.unwrap_or(defaults.temperature),
        "top_p": request.sampling.top_p.unwrap_or(defaults.top_p),
        "max_tokens": request.max_output_tokens,
        "repeat_penalty": defaults.repeat_penalty,
        "stop": stop,
    });
    if model_config.thinking.mode == ThinkingMode::Off {
        body["chat_template_kwargs"] = serde_json::json!({ "enable_thinking": false });
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
    message: UpstreamMessage,
}

#[derive(Deserialize)]
struct UpstreamMessage {
    content: String,
}

#[derive(Deserialize)]
struct UpstreamUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

pub fn parse_text_response(bytes: &[u8]) -> crate::Result<ModelCompletion> {
    let response: UpstreamResponse = serde_json::from_slice(bytes)?;
    let content = response
        .choices
        .into_iter()
        .next()
        .map(|choice| choice.message.content)
        .ok_or(crate::error::HoshikageError::ResponseTranslationFailed)?;
    let usage = response
        .usage
        .ok_or(crate::error::HoshikageError::ResponseTranslationFailed)?;
    Ok(ModelCompletion::Text {
        content,
        usage: TokenUsage::Measured {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{Conversation, Message, Role};
    use crate::inference::{ModelToolSet, SamplingOptions, ToolChoice};

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

    #[test]
    fn adapter_builds_text_only_chat_request_and_preserves_thinking_off() {
        let mut config =
            ModelConfig::new_legacy("/models".to_string(), "model.gguf".to_string(), Vec::new());
        config.thinking.mode = ThinkingMode::Off;
        config.stop = vec!["BUNDLE_STOP".to_string(), "</s>".to_string()];

        let body = build_text_request(
            &ModelId::new("gemma4").unwrap(),
            &request(),
            &config,
            &LlamaServerTextDefaults {
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
        let error = parse_text_response(br#"{"choices":[{"message":{"content":"OK"}}]}"#)
            .err()
            .unwrap();

        assert!(matches!(
            error,
            crate::error::HoshikageError::ResponseTranslationFailed
        ));
    }

    #[test]
    fn malformed_upstream_json_is_a_translation_failure() {
        let error = parse_text_response(br#"{"choices":"broken"}"#)
            .err()
            .unwrap();

        assert!(matches!(error, crate::error::HoshikageError::SerdeError(_)));
    }
}
