use axum::{
    extract::State,
    http::StatusCode,
    response::sse::Event,
    response::{IntoResponse, Response, Sse},
    Json,
};
use async_stream::stream;
use std::convert::Infallible;
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

pub async fn chat_completion(
    State(manager): State<Arc<crate::model::ModelManager>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let stream_response = req.stream.unwrap_or(false);
    let model_name = req.model.clone();

    if req.messages.is_empty() {
            return error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                "validation_error",
                "messages must not be empty",
                "invalid_request",
            );
    }

    let model_config = match manager.get_model(&model_name).await {
        Ok(config) => config,
        Err(_) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "model_not_found",
                "指定されたモデルが見つかりません",
                "invalid_request",
            );
        }
    };

    let prompt = match manager.build_prompt(&model_name, &req.messages).await {
        Ok(prompt) => prompt,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
                e.to_string(),
                "internal_server_error",
            );
        }
    };

    let max_tokens = req
        .max_tokens
        .unwrap_or(if stream_response { 2096 } else { 1024 }) as i32;
    let max_tokens = if stream_response {
        max_tokens.min(2096)
    } else {
        max_tokens.min(1024)
    };

    let mut stop_sequences = vec![
        "<|im_start|>".to_string(),
        "<|im_end|>".to_string(),
        "</s>".to_string(),
        "<|eot_id|>".to_string(),
        "<|endoftext|>".to_string(),
    ];
    for stop in &model_config.stop {
        if !stop_sequences.contains(stop) {
            stop_sequences.push(stop.clone());
        }
    }

    let params = crate::inference::llama_wrapper::InferenceParams {
        temperature: req
            .temperature
            .unwrap_or(manager.default_temperature()),
        top_p: req.top_p.unwrap_or(manager.default_top_p()),
        max_tokens,
        stop_sequences,
    };

    if stream_response {
        return stream_response_handler(manager, model_name, prompt, params);
    }

    match manager.generate(&model_name, &prompt, params).await {
        Ok((output, prompt_tokens, completion_tokens)) => {
            let response = ChatCompletionResponse {
                id: uuid::Uuid::new_v4().to_string(),
                object: "chat.completion".to_string(),
                created: chrono::Utc::now().timestamp(),
                model: req.model,
                choices: vec![Choice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: output,
                    },
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            };

            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference_failed",
            e.to_string(),
            "internal_server_error",
        ),
    }
}

fn error_response(
    status: StatusCode,
    code: &str,
    message: impl Into<String>,
    error_type: &str,
) -> Response {
    (status, Json(ErrorResponse {
        error: ErrorBody {
            code: code.to_string(),
            message: message.into(),
            error_type: error_type.to_string(),
            param: None,
        },
    }))
        .into_response()
}

fn stream_response_handler(
    manager: Arc<crate::model::ModelManager>,
    model_name: String,
    prompt: String,
    params: crate::inference::llama_wrapper::InferenceParams,
) -> Response {
    let created = chrono::Utc::now().timestamp();
    let id = uuid::Uuid::new_v4().to_string();
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel::<Result<String>>();

    tokio::spawn({
        let manager = manager.clone();
        let model_name = model_name.clone();
        let prompt = prompt.clone();
        let params = params.clone();
        let sender_clone = sender.clone();
        async move {
            if let Err(e) = manager
                .generate_stream(model_name, prompt, params, sender_clone)
                .await
            {
                let _ = sender.send(Err(e));
            }
        }
    });

    let stream = stream! {
        let mut sent_role = false;
        while let Some(chunk) = receiver.recv().await {
            match chunk {
                Ok(text) => {
                    let delta = if sent_role {
                        ChunkDelta {
                            content: Some(text),
                            role: None,
                        }
                    } else {
                        sent_role = true;
                        ChunkDelta {
                            content: Some(text),
                            role: Some("assistant".to_string()),
                        }
                    };

                    let payload = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta,
                            finish_reason: None,
                        }],
                    };

                    let data = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
                    yield Ok::<Event, Infallible>(Event::default().data(data));
                }
                Err(e) => {
                    let error_payload = ErrorResponse {
                        error: ErrorBody {
                            code: "inference_failed".to_string(),
                            message: e.to_string(),
                            error_type: "internal_server_error".to_string(),
                            param: None,
                        },
                    };
                    let data = serde_json::to_string(&error_payload).unwrap_or_else(|_| "{}".to_string());
                    yield Ok::<Event, Infallible>(Event::default().data(data));
                    break;
                }
            }
        }

        let finish_payload = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta::default(),
                finish_reason: Some("stop".to_string()),
            }],
        };
        let finish_data = serde_json::to_string(&finish_payload).unwrap_or_else(|_| "{}".to_string());
        yield Ok::<Event, Infallible>(Event::default().data(finish_data));

        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
    };

    Sse::new(stream).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_prompt() {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful".to_string(),
            },
        ];

        let prompt = messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(prompt, "user: Hello\nsystem: You are helpful");
    }

    #[test]
    fn test_chat_completion_request() {
        let req = ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            temperature: Some(0.7),
            top_p: None,
            max_tokens: Some(100),
            stream: None,
            presence_penalty: None,
            frequency_penalty: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("test-model"));
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"max_tokens\":100"));
    }
}
