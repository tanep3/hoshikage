use crate::error::Result;
use crate::model::ThinkingMode;
use async_stream::stream;
use axum::{
    body::Body,
    extract::State,
    http::{header::HeaderName, HeaderValue, StatusCode},
    response::sse::Event,
    response::{IntoResponse, Response, Sse},
    Json,
};
use base64::Engine;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Deserialize, Serialize)]
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
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub repeat_last_n: Option<u32>,
    #[serde(default)]
    pub diffusion_steps: Option<i32>,
    #[serde(default)]
    pub diffusion_algorithm: Option<i32>,
    #[serde(default)]
    pub diffusion_schedule: Option<i32>,
    #[serde(default)]
    pub diffusion_cfg_scale: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: ChatContent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrlContent },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ImageUrlContent {
    Object {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    Url(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedImageInput {
    pub source: NormalizedImageSource,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizedImageSource {
    DataUrl { media_type: String, data: String },
    LocalPath(PathBuf),
}

impl ChatContent {
    pub fn to_prompt_text(&self) -> String {
        match self {
            ChatContent::Text(text) => text.clone(),
            ChatContent::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ChatContentPart::Text { text } => Some(text.as_str()),
                    ChatContentPart::ImageUrl { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    pub fn normalized_images(&self) -> std::result::Result<Vec<NormalizedImageInput>, String> {
        match self {
            ChatContent::Text(_) => Ok(Vec::new()),
            ChatContent::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ChatContentPart::Text { .. } => None,
                    ChatContentPart::ImageUrl { image_url } => Some(image_url.normalize()),
                })
                .collect(),
        }
    }
}

impl ChatMessage {
    pub fn text_content(&self) -> String {
        self.content.to_prompt_text()
    }

    fn to_conversation_message(&self) -> std::result::Result<crate::conversation::Message, String> {
        let role =
            crate::conversation::Role::parse(&self.role).map_err(|error| error.to_string())?;
        let content = match &self.content {
            ChatContent::Text(text) => vec![crate::conversation::ContentPart::Text(text.clone())],
            ChatContent::Parts(parts) => parts
                .iter()
                .map(|part| match part {
                    ChatContentPart::Text { text } => {
                        crate::conversation::ContentPart::Text(text.clone())
                    }
                    ChatContentPart::ImageUrl { image_url } => {
                        let (source, detail) = match image_url {
                            ImageUrlContent::Object { url, detail } => {
                                (url.clone(), detail.clone())
                            }
                            ImageUrlContent::Url(url) => (url.clone(), None),
                        };
                        crate::conversation::ContentPart::Image(crate::conversation::ImageInput {
                            source,
                            detail,
                        })
                    }
                })
                .collect(),
        };
        crate::conversation::Message::new(role, content).map_err(|error| error.to_string())
    }
}

impl From<String> for ChatContent {
    fn from(value: String) -> Self {
        ChatContent::Text(value)
    }
}

impl From<&str> for ChatContent {
    fn from(value: &str) -> Self {
        ChatContent::Text(value.to_string())
    }
}

impl ImageUrlContent {
    fn normalize(&self) -> std::result::Result<NormalizedImageInput, String> {
        let (url, detail) = match self {
            ImageUrlContent::Object { url, detail } => (url.as_str(), detail.clone()),
            ImageUrlContent::Url(url) => (url.as_str(), None),
        };

        let source = normalize_image_source(url)?;
        Ok(NormalizedImageInput { source, detail })
    }
}

fn normalize_image_source(raw: &str) -> std::result::Result<NormalizedImageSource, String> {
    if let Some(data_url) = raw.strip_prefix("data:") {
        let (metadata, data) = data_url
            .split_once(',')
            .ok_or_else(|| "image data URL must contain a comma separator".to_string())?;
        let media_type = metadata
            .split(';')
            .next()
            .ok_or_else(|| "image data URL media type is missing".to_string())?;

        if !metadata.split(';').any(|part| part == "base64") {
            return Err("image data URL must be base64 encoded".to_string());
        }
        if !matches!(media_type, "image/png" | "image/jpeg" | "image/jpg") {
            return Err(format!("unsupported image media type: {}", media_type));
        }
        if data.is_empty()
            || !data
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'='))
        {
            return Err("image data URL contains invalid base64 data".to_string());
        }

        return Ok(NormalizedImageSource::DataUrl {
            media_type: media_type.to_string(),
            data: data.to_string(),
        });
    }

    if let Some(path) = raw.strip_prefix("file://") {
        return normalize_local_image_path(path);
    }

    if raw.starts_with("http://") || raw.starts_with("https://") {
        return Err("external image URLs are not supported".to_string());
    }

    normalize_local_image_path(raw)
}

fn normalize_local_image_path(raw: &str) -> std::result::Result<NormalizedImageSource, String> {
    let path = PathBuf::from(raw);
    if !path.is_absolute() {
        return Err("local image path must be absolute".to_string());
    }

    Ok(NormalizedImageSource::LocalPath(path))
}

fn validate_vision_input(messages: &[ChatMessage]) -> std::result::Result<bool, String> {
    let mut has_images = false;
    for message in messages {
        let images = message.content.normalized_images()?;
        has_images |= !images.is_empty();
    }

    Ok(has_images)
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

    let has_vision_input = match validate_vision_input(&req.messages) {
        Ok(has_vision_input) => has_vision_input,
        Err(message) => {
            return error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                "vision_input_error",
                message,
                "invalid_request",
            );
        }
    };

    if has_vision_input && model_config.mmproj.is_none() {
        return error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "vision_not_configured",
            "画像入力には mmproj が設定されたモデルが必要です",
            "invalid_request",
        );
    }

    if manager.uses_managed_llama_server() {
        return managed_chat_completion(manager, req, model_config).await;
    }

    if has_vision_input {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            "vision_generation_not_connected",
            "画像推論経路はまだ接続されていません",
            "invalid_request",
        );
    }

    let conversation_messages = match req
        .messages
        .iter()
        .map(ChatMessage::to_conversation_message)
        .collect::<std::result::Result<Vec<_>, _>>()
    {
        Ok(messages) => messages,
        Err(message) => {
            return error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                "validation_error",
                message,
                "invalid_request",
            );
        }
    };

    let prompt = match manager
        .build_prompt(&model_name, &conversation_messages)
        .await
    {
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
        temperature: req.temperature.unwrap_or(manager.default_temperature()),
        top_p: req.top_p.unwrap_or(manager.default_top_p()),
        max_tokens,
        stop_sequences,
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        repeat_penalty: req
            .repeat_penalty
            .unwrap_or(manager.default_repeat_penalty()),
        repeat_last_n: req.repeat_last_n.unwrap_or(manager.default_repeat_last_n()) as usize,
        diffusion_steps: req.diffusion_steps,
        diffusion_algorithm: req.diffusion_algorithm,
        diffusion_schedule: req.diffusion_schedule,
        diffusion_cfg_scale: req.diffusion_cfg_scale,
    };

    if stream_response {
        match manager.is_diffusion_model(&model_name).await {
            Ok(true) => {
                return stream_response_single(manager, model_name, prompt, params);
            }
            Ok(false) => {
                return stream_response_handler(manager, model_name, prompt, params);
            }
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "inference_failed",
                    e.to_string(),
                    "internal_server_error",
                );
            }
        }
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
                        content: ChatContent::Text(output),
                    },
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            };

            with_fallback_headers((StatusCode::OK, Json(response)).into_response(), &manager)
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference_failed",
            e.to_string(),
            "internal_server_error",
        ),
    }
}

async fn managed_chat_completion(
    manager: Arc<crate::model::ModelManager>,
    req: ChatCompletionRequest,
    model_config: crate::model::ModelConfig,
) -> Response {
    let body = match build_managed_upstream_body(&manager, &req, &model_config) {
        Ok(body) => body,
        Err(message) => {
            return error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                "vision_input_error",
                message,
                "invalid_request",
            )
        }
    };
    let runtime_lease = match manager.acquire_managed_llama_server(&req.model).await {
        Ok(lease) => lease,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
                e.to_string(),
                "internal_server_error",
            )
        }
    };

    let upstream = match manager.send_managed_chat(&runtime_lease, &body).await {
        Ok(response) => response,
        Err(e) => {
            manager.mark_managed_lease_unhealthy(
                &req.model,
                &runtime_lease,
                format!("upstream request failed: {e}"),
            );
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
                e.to_string(),
                "internal_server_error",
            );
        }
    };

    let status = StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let content_type = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.to_string());

    if req.stream.unwrap_or(false) {
        let mut upstream_stream = upstream.bytes_stream();
        let stream = stream! {
            let _runtime_lease = runtime_lease;
            while let Some(chunk) = upstream_stream.next().await {
                yield chunk.map_err(std::io::Error::other);
            }
        };
        let mut response = Response::builder()
            .status(status)
            .body(Body::from_stream(stream))
            .unwrap_or_else(|_| {
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "inference_failed",
                    "failed to build streaming response",
                    "internal_server_error",
                )
            });
        if let Some(content_type) = content_type {
            if let Ok(content_type) = HeaderValue::from_str(&content_type) {
                response
                    .headers_mut()
                    .insert(axum::http::header::CONTENT_TYPE, content_type);
            }
        }
        return with_fallback_headers(response, &manager);
    }

    match upstream.bytes().await {
        Ok(bytes) => {
            if let Err(error) = runtime_lease.finish() {
                tracing::error!(error = %error, "Failed to finish managed runtime lease");
            }
            let mut response = Response::builder()
                .status(status)
                .body(Body::from(bytes))
                .unwrap_or_else(|_| {
                    error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "inference_failed",
                        "failed to build upstream response",
                        "internal_server_error",
                    )
                });
            if let Some(content_type) = content_type {
                if let Ok(content_type) = HeaderValue::from_str(&content_type) {
                    response
                        .headers_mut()
                        .insert(axum::http::header::CONTENT_TYPE, content_type);
                }
            }
            with_fallback_headers(response, &manager)
        }
        Err(e) => {
            manager.mark_managed_lease_unhealthy(
                &req.model,
                &runtime_lease,
                format!("upstream response failed: {e}"),
            );
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
                e.to_string(),
                "internal_server_error",
            )
        }
    }
}

fn build_managed_upstream_body(
    manager: &crate::model::ModelManager,
    req: &ChatCompletionRequest,
    model_config: &crate::model::ModelConfig,
) -> std::result::Result<serde_json::Value, String> {
    let mut body = serde_json::to_value(req).map_err(|err| err.to_string())?;
    let Some(object) = body.as_object_mut() else {
        return Err("request body must serialize to an object".to_string());
    };

    object.insert(
        "messages".to_string(),
        serde_json::to_value(prepare_messages_for_upstream(&req.messages)?)
            .map_err(|err| err.to_string())?,
    );
    object.insert(
        "temperature".to_string(),
        serde_json::json!(req.temperature.unwrap_or(manager.default_temperature())),
    );
    object.insert(
        "top_p".to_string(),
        serde_json::json!(req.top_p.unwrap_or(manager.default_top_p())),
    );
    object.insert(
        "max_tokens".to_string(),
        serde_json::json!(req.max_tokens.unwrap_or(if req.stream.unwrap_or(false) {
            2096
        } else {
            1024
        })),
    );
    object.insert(
        "stop".to_string(),
        serde_json::json!(merged_stop_sequences(req.stop.as_deref(), model_config)),
    );
    object.insert(
        "repeat_penalty".to_string(),
        serde_json::json!(req
            .repeat_penalty
            .unwrap_or(manager.default_repeat_penalty())),
    );
    if model_config.thinking.mode == ThinkingMode::Off {
        object.insert(
            "chat_template_kwargs".to_string(),
            serde_json::json!({ "enable_thinking": false }),
        );
    }

    Ok(body)
}

fn prepare_messages_for_upstream(
    messages: &[ChatMessage],
) -> std::result::Result<Vec<ChatMessage>, String> {
    messages
        .iter()
        .map(|message| {
            Ok(ChatMessage {
                role: message.role.clone(),
                content: prepare_content_for_upstream(&message.content)?,
            })
        })
        .collect()
}

fn prepare_content_for_upstream(content: &ChatContent) -> std::result::Result<ChatContent, String> {
    match content {
        ChatContent::Text(text) => Ok(ChatContent::Text(text.clone())),
        ChatContent::Parts(parts) => Ok(ChatContent::Parts(
            parts
                .iter()
                .map(|part| match part {
                    ChatContentPart::Text { text } => {
                        Ok(ChatContentPart::Text { text: text.clone() })
                    }
                    ChatContentPart::ImageUrl { image_url } => Ok(ChatContentPart::ImageUrl {
                        image_url: prepare_image_url_for_upstream(image_url)?,
                    }),
                })
                .collect::<std::result::Result<Vec<_>, String>>()?,
        )),
    }
}

fn prepare_image_url_for_upstream(
    image_url: &ImageUrlContent,
) -> std::result::Result<ImageUrlContent, String> {
    let detail = match image_url {
        ImageUrlContent::Object { detail, .. } => detail.clone(),
        ImageUrlContent::Url(_) => None,
    };

    let normalized = image_url.normalize()?;
    let url = match normalized.source {
        NormalizedImageSource::DataUrl { media_type, data } => {
            format!("data:{};base64,{}", media_type, data)
        }
        NormalizedImageSource::LocalPath(path) => {
            let media_type = image_media_type(&path)?;
            let data = std::fs::read(&path)
                .map_err(|err| format!("failed to read local image {}: {}", path.display(), err))?;
            format!(
                "data:{};base64,{}",
                media_type,
                base64::engine::general_purpose::STANDARD.encode(data)
            )
        }
    };

    Ok(ImageUrlContent::Object { url, detail })
}

fn image_media_type(path: &Path) -> std::result::Result<&'static str, String> {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => Ok("image/png"),
        Some("jpg") | Some("jpeg") => Ok("image/jpeg"),
        _ => Err(format!(
            "unsupported local image extension: {}",
            path.display()
        )),
    }
}

fn merged_stop_sequences(
    request_stop: Option<&[String]>,
    model_config: &crate::model::ModelConfig,
) -> Vec<String> {
    let mut stop_sequences = default_stop_sequences();
    for stop in model_config
        .stop
        .iter()
        .chain(request_stop.unwrap_or(&[]).iter())
    {
        if !stop_sequences.contains(stop) {
            stop_sequences.push(stop.clone());
        }
    }
    stop_sequences
}

fn default_stop_sequences() -> Vec<String> {
    vec![
        "<|im_start|>".to_string(),
        "<|im_end|>".to_string(),
        "</s>".to_string(),
        "<|eot_id|>".to_string(),
        "<|endoftext|>".to_string(),
    ]
}

fn error_response(
    status: StatusCode,
    code: &str,
    message: impl Into<String>,
    error_type: &str,
) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: ErrorBody {
                code: code.to_string(),
                message: message.into(),
                error_type: error_type.to_string(),
                param: None,
            },
        }),
    )
        .into_response()
}

fn with_fallback_headers(
    mut response: Response,
    manager: &Arc<crate::model::ModelManager>,
) -> Response {
    let status = manager.runtime_status();
    let Some(fallback) = status.last_fallback else {
        return response;
    };

    let headers = response.headers_mut();
    headers.insert(
        HeaderName::from_static("x-hoshikage-fallback"),
        HeaderValue::from_static("speculation"),
    );
    if let Ok(reason) = HeaderValue::from_str(&fallback.reason) {
        headers.insert(
            HeaderName::from_static("x-hoshikage-fallback-reason"),
            reason,
        );
    }
    if let Ok(mode) = HeaderValue::from_str(&format!("{:?}", fallback.requested_mode)) {
        headers.insert(
            HeaderName::from_static("x-hoshikage-fallback-requested-mode"),
            mode,
        );
    }

    response
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

    with_fallback_headers(Sse::new(stream).into_response(), &manager)
}

fn stream_response_single(
    manager: Arc<crate::model::ModelManager>,
    model_name: String,
    prompt: String,
    params: crate::inference::llama_wrapper::InferenceParams,
) -> Response {
    let created = chrono::Utc::now().timestamp();
    let id = uuid::Uuid::new_v4().to_string();
    let manager_for_stream = manager.clone();

    let stream = stream! {
        let output = match manager_for_stream.generate(&model_name, &prompt, params).await {
            Ok((output, _, _)) => output,
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
                yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
                return;
            }
        };

        let payload = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    content: Some(output),
                    role: Some("assistant".to_string()),
                },
                finish_reason: Some("stop".to_string()),
            }],
        };

        let data = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
        yield Ok::<Event, Infallible>(Event::default().data(data));
        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
    };

    with_fallback_headers(Sse::new(stream).into_response(), &manager)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::model::{ModelConfig, ModelManager};

    #[test]
    fn test_format_prompt() {
        let messages = [
            ChatMessage {
                role: "user".to_string(),
                content: ChatContent::Text("Hello".to_string()),
            },
            ChatMessage {
                role: "system".to_string(),
                content: ChatContent::Text("You are helpful".to_string()),
            },
        ];

        let prompt = messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.text_content()))
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
                content: ChatContent::Text("Hello".to_string()),
            }],
            temperature: Some(0.7),
            top_p: None,
            max_tokens: Some(100),
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            repeat_penalty: None,
            repeat_last_n: None,
            diffusion_steps: None,
            diffusion_algorithm: None,
            diffusion_schedule: None,
            diffusion_cfg_scale: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("test-model"));
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"max_tokens\":100"));
    }

    #[test]
    fn string_content_deserializes_as_text() {
        let json = r#"{"role":"user","content":"Hello"}"#;
        let message: ChatMessage = serde_json::from_str(json).unwrap();

        assert_eq!(message.content, ChatContent::Text("Hello".to_string()));
        assert_eq!(message.text_content(), "Hello");
    }

    #[test]
    fn parts_content_preserves_text_and_detects_data_url_image() {
        let json = r#"{
            "role": "user",
            "content": [
                { "type": "text", "text": "この画像を説明して" },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,aGVsbG8=",
                        "detail": "auto"
                    }
                }
            ]
        }"#;
        let message: ChatMessage = serde_json::from_str(json).unwrap();

        assert_eq!(message.text_content(), "この画像を説明して");
        let images = message.content.normalized_images().unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].detail.as_deref(), Some("auto"));
        assert_eq!(
            images[0].source,
            NormalizedImageSource::DataUrl {
                media_type: "image/png".to_string(),
                data: "aGVsbG8=".to_string()
            }
        );
    }

    #[test]
    fn file_url_image_normalizes_to_absolute_path() {
        let image = ImageUrlContent::Url("file:///tmp/sample.png".to_string())
            .normalize()
            .unwrap();

        assert_eq!(
            image.source,
            NormalizedImageSource::LocalPath(PathBuf::from("/tmp/sample.png"))
        );
    }

    #[test]
    fn external_image_url_is_rejected() {
        let image = ImageUrlContent::Url("https://example.test/image.png".to_string());
        let err = image.normalize().unwrap_err();

        assert_eq!(err, "external image URLs are not supported");
    }

    #[test]
    fn relative_image_path_is_rejected() {
        let image = ImageUrlContent::Url("images/sample.png".to_string());
        let err = image.normalize().unwrap_err();

        assert_eq!(err, "local image path must be absolute");
    }

    #[test]
    fn managed_upstream_body_merges_defaults_and_model_stop_sequences() {
        let manager = ModelManager::new(Config::default());
        let model_config = ModelConfig {
            stop: vec!["<custom-stop>".to_string()],
            ..ModelConfig::new_legacy("/models".to_string(), "main.gguf".to_string(), Vec::new())
        };
        let req = ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: ChatContent::Text("Hello".to_string()),
            }],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            stop: Some(vec!["<request-stop>".to_string()]),
            presence_penalty: None,
            frequency_penalty: None,
            repeat_penalty: None,
            repeat_last_n: None,
            diffusion_steps: None,
            diffusion_algorithm: None,
            diffusion_schedule: None,
            diffusion_cfg_scale: None,
        };

        let body = build_managed_upstream_body(&manager, &req, &model_config).unwrap();

        assert!((body["temperature"].as_f64().unwrap() - 0.2).abs() < 0.0001);
        assert!((body["top_p"].as_f64().unwrap() - 0.8).abs() < 0.0001);
        assert_eq!(body["max_tokens"], serde_json::json!(1024));
        assert!(body["stop"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("<custom-stop>")));
        assert!(body["stop"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("<request-stop>")));
    }

    #[test]
    fn managed_upstream_body_disables_template_thinking_for_thinking_off_bundle() {
        let manager = ModelManager::new(Config::default());
        let mut model_config =
            ModelConfig::new_legacy("/models".to_string(), "main.gguf".to_string(), Vec::new());
        model_config.thinking.mode = ThinkingMode::Off;
        let req = ChatCompletionRequest {
            model: "unsloth-gemma4-12b-qat-thinking-off".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: ChatContent::Text("Hello".to_string()),
            }],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            repeat_penalty: None,
            repeat_last_n: None,
            diffusion_steps: None,
            diffusion_algorithm: None,
            diffusion_schedule: None,
            diffusion_cfg_scale: None,
        };

        let body = build_managed_upstream_body(&manager, &req, &model_config).unwrap();

        assert_eq!(
            body["chat_template_kwargs"]["enable_thinking"],
            serde_json::json!(false)
        );
    }

    #[test]
    fn managed_upstream_body_converts_local_image_path_to_data_url() {
        let image_path =
            std::env::temp_dir().join(format!("hoshikage-test-image-{}.png", uuid::Uuid::new_v4()));
        std::fs::write(&image_path, b"hello").unwrap();

        let manager = ModelManager::new(Config::default());
        let model_config =
            ModelConfig::new_legacy("/models".to_string(), "main.gguf".to_string(), Vec::new());
        let req = ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: ChatContent::Parts(vec![ChatContentPart::ImageUrl {
                    image_url: ImageUrlContent::Url(image_path.to_string_lossy().to_string()),
                }]),
            }],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            repeat_penalty: None,
            repeat_last_n: None,
            diffusion_steps: None,
            diffusion_algorithm: None,
            diffusion_schedule: None,
            diffusion_cfg_scale: None,
        };

        let body = build_managed_upstream_body(&manager, &req, &model_config).unwrap();
        let url = body["messages"][0]["content"][0]["image_url"]["url"]
            .as_str()
            .unwrap();

        assert_eq!(url, "data:image/png;base64,aGVsbG8=");
        let _ = std::fs::remove_file(image_path);
    }
}
