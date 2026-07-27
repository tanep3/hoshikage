use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use std::sync::Arc;

#[derive(Debug, Serialize)]
pub struct ModelData {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelData>,
    pub models: Vec<CodexModelData>,
}

const CODEX_BASE_INSTRUCTIONS: &str = "You are an agentic coding assistant. Follow the user and \
developer instructions, use the available tools when they are needed, continue until the task is \
complete, and report tool results accurately. Never claim that an action succeeded unless its \
tool result confirms success.";

#[derive(Debug, Serialize)]
pub struct CodexModelData {
    pub slug: String,
    pub display_name: String,
    pub description: String,
    pub supported_reasoning_levels: Vec<CodexReasoningLevel>,
    pub shell_type: &'static str,
    pub visibility: &'static str,
    pub supported_in_api: bool,
    pub priority: i32,
    pub availability_nux: Option<serde_json::Value>,
    pub upgrade: Option<serde_json::Value>,
    pub base_instructions: &'static str,
    pub include_skills_usage_instructions: bool,
    pub supports_reasoning_summary_parameter: bool,
    pub default_reasoning_summary: &'static str,
    pub support_verbosity: bool,
    pub default_verbosity: Option<&'static str>,
    pub apply_patch_tool_type: Option<&'static str>,
    pub web_search_tool_type: &'static str,
    pub truncation_policy: CodexTruncationPolicy,
    pub supports_parallel_tool_calls: bool,
    pub supports_image_detail_original: bool,
    pub context_window: i64,
    pub max_context_window: i64,
    pub experimental_supported_tools: Vec<String>,
    pub input_modalities: Vec<&'static str>,
    pub supports_search_tool: bool,
    pub use_responses_lite: bool,
}

#[derive(Debug, Serialize)]
pub struct CodexReasoningLevel {
    pub effort: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Serialize)]
pub struct CodexTruncationPolicy {
    pub mode: &'static str,
    pub limit: i64,
}

impl CodexModelData {
    fn new(id: String, context_window: u32, tools: bool, vision: bool) -> Self {
        let context_window = i64::from(context_window);
        let mut input_modalities = vec!["text"];
        if vision {
            input_modalities.push("image");
        }

        Self {
            display_name: id.clone(),
            description: "Local model served by Hoshikage".to_string(),
            slug: id,
            supported_reasoning_levels: Vec::new(),
            shell_type: if tools { "shell_command" } else { "disabled" },
            visibility: "list",
            supported_in_api: true,
            priority: 0,
            availability_nux: None,
            upgrade: None,
            base_instructions: CODEX_BASE_INSTRUCTIONS,
            include_skills_usage_instructions: true,
            supports_reasoning_summary_parameter: false,
            default_reasoning_summary: "none",
            support_verbosity: false,
            default_verbosity: None,
            apply_patch_tool_type: None,
            web_search_tool_type: "text",
            truncation_policy: CodexTruncationPolicy {
                mode: "tokens",
                limit: context_window.saturating_mul(3) / 4,
            },
            supports_parallel_tool_calls: false,
            supports_image_detail_original: vision,
            context_window,
            max_context_window: context_window,
            experimental_supported_tools: Vec::new(),
            input_modalities,
            supports_search_tool: false,
            use_responses_lite: false,
        }
    }
}

pub async fn models(
    State(manager): State<Arc<crate::model::ModelManager>>,
) -> Json<ModelListResponse> {
    let model_infos = manager.list_hoshikage_models().await;

    let data = model_infos
        .iter()
        .map(|model| ModelData {
            id: model.id.clone(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "tane".to_string(),
        })
        .collect();
    let models = model_infos
        .into_iter()
        .map(|model| CodexModelData::new(model.id, model.context_window, model.tools, model.vision))
        .collect();

    Json(ModelListResponse {
        object: "list".to_string(),
        data,
        models,
    })
}

#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime: Option<crate::model::RuntimeStatusSnapshot>,
}

pub async fn status(
    State(manager): State<Arc<crate::model::ModelManager>>,
) -> Json<StatusResponse> {
    Json(StatusResponse {
        status: "ok".to_string(),
        runtime: Some(manager.runtime_status()),
    })
}

#[derive(Debug, Serialize)]
pub struct HoshikageModelListResponse {
    pub object: String,
    pub data: Vec<crate::model::HoshikageModelInfo>,
}

pub async fn hoshikage_models(
    State(manager): State<Arc<crate::model::ModelManager>>,
) -> Json<HoshikageModelListResponse> {
    Json(HoshikageModelListResponse {
        object: "list".to_string(),
        data: manager.list_hoshikage_models().await,
    })
}

pub async fn hoshikage_model(
    State(manager): State<Arc<crate::model::ModelManager>>,
    Path(name): Path<String>,
) -> Response {
    match manager.get_hoshikage_model(&name).await {
        Ok(model) => Json(model).into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "code": "model_not_found",
                    "message": "Model was not found",
                    "type": "invalid_request_error",
                    "param": "model"
                }
            })),
        )
            .into_response(),
    }
}

#[derive(Debug, Serialize)]
pub struct VersionResponse {
    pub version: String,
}

pub async fn version() -> Json<VersionResponse> {
    Json(VersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_data_serialization() {
        let data = ModelData {
            id: "test-model".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "tane".to_string(),
        };

        let json = serde_json::to_string(&data).unwrap();
        assert!(json.contains("test-model"));
    }

    #[test]
    fn model_list_serializes_openai_and_codex_catalog_shapes() {
        let response = ModelListResponse {
            object: "list".to_string(),
            data: vec![ModelData {
                id: "test-model".to_string(),
                object: "model".to_string(),
                created: 1686935002,
                owned_by: "tane".to_string(),
            }],
            models: vec![CodexModelData::new(
                "test-model".to_string(),
                65_536,
                true,
                false,
            )],
        };

        let json = serde_json::to_value(response).unwrap();
        assert_eq!(json["data"][0]["id"], "test-model");
        assert_eq!(json["models"][0]["slug"], "test-model");
        assert_eq!(json["models"][0]["context_window"], 65_536);
        assert_eq!(json["models"][0]["shell_type"], "shell_command");
        assert_eq!(json["models"][0]["supports_parallel_tool_calls"], false);
        assert_eq!(
            json["models"][0]["input_modalities"],
            serde_json::json!(["text"])
        );
        assert_eq!(
            json["models"][0]["apply_patch_tool_type"],
            serde_json::Value::Null
        );
        assert!(json["models"][0]["base_instructions"]
            .as_str()
            .is_some_and(|instructions| !instructions.is_empty()));
    }

    #[test]
    fn test_status_response() {
        let response = StatusResponse {
            status: "ok".to_string(),
            runtime: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("ok"));
    }

    #[test]
    fn test_version_response() {
        let response = VersionResponse {
            version: "1.0.0".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("1.0.0"));
    }
}
