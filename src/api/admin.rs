use crate::model::{ModelConfig, SpeculationConfig, SpeculationMode, ThinkingConfig};
use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize, Serialize)]
pub struct AddModelRequest {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drafter: Option<String>,
    #[serde(default)]
    pub speculation: SpeculationConfig,
    #[serde(default)]
    pub thinking: ThinkingConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_ctx: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_gpu_layers: Option<i32>,
}

#[derive(Debug, Serialize)]
pub struct AddModelResponse {
    pub success: bool,
    pub message: String,
}

pub async fn add_model(
    State(manager): State<Arc<crate::model::ModelManager>>,
    Json(req): Json<AddModelRequest>,
) -> Json<AddModelResponse> {
    let model_path = std::path::PathBuf::from(&req.path);
    let model_file = match model_path.file_name().and_then(|f| f.to_str()) {
        Some(name) => name.to_string(),
        None => {
            return Json(AddModelResponse {
                success: false,
                message: "Invalid model path".to_string(),
            })
        }
    };
    let model_dir = model_path
        .parent()
        .and_then(|p| p.to_str())
        .unwrap_or(".")
        .to_string();

    let mut speculation = req.speculation;
    if speculation.is_off() && req.drafter.is_some() {
        speculation.modes = vec![SpeculationMode::Mtp];
    }
    if req.drafter.is_none() {
        speculation.modes.clear();
    }

    let config = ModelConfig {
        mmproj: req.mmproj,
        drafter: req.drafter,
        speculation,
        thinking: req.thinking,
        n_ctx: req.n_ctx,
        n_gpu_layers: req.n_gpu_layers,
        ..ModelConfig::new_legacy(model_dir, model_file, req.stop)
    };

    let name = req.name.clone();
    match manager.add_model(name.clone(), config).await {
        Ok(()) => Json(AddModelResponse {
            success: true,
            message: format!("Model '{}' added successfully", name),
        }),
        Err(e) => {
            tracing::error!("Failed to add model: {}", e);
            Json(AddModelResponse {
                success: false,
                message: format!("Failed to add model: {}", e),
            })
        }
    }
}

pub async fn remove_model(
    State(manager): State<Arc<crate::model::ModelManager>>,
    Path(name): Path<String>,
) -> Json<AddModelResponse> {
    let name = name.clone();
    match manager.remove_model(&name).await {
        Ok(()) => Json(AddModelResponse {
            success: true,
            message: format!("Model '{}' removed successfully", name),
        }),
        Err(e) => {
            tracing::error!("Failed to remove model: {}", e);
            Json(AddModelResponse {
                success: false,
                message: format!("Failed to remove model: {}", e),
            })
        }
    }
}

pub async fn reload_models(
    State(manager): State<Arc<crate::model::ModelManager>>,
) -> Json<AddModelResponse> {
    match manager.load_models().await {
        Ok(()) => Json(AddModelResponse {
            success: true,
            message: "Models reloaded successfully".to_string(),
        }),
        Err(e) => {
            tracing::error!("Failed to reload models: {}", e);
            Json(AddModelResponse {
                success: false,
                message: format!("Failed to reload models: {}", e),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FallbackMode, ThinkingMode};

    #[test]
    fn test_add_model_request_serialization() {
        let req = AddModelRequest {
            name: "test-model".to_string(),
            path: "/models/model.gguf".to_string(),
            stop: vec!["</s>".to_string()],
            mmproj: Some("/models/mmproj.gguf".to_string()),
            drafter: None,
            speculation: SpeculationConfig {
                modes: Vec::new(),
                fallback: FallbackMode::Warn,
            },
            thinking: ThinkingConfig {
                mode: ThinkingMode::Off,
            },
            n_ctx: Some(8192),
            n_gpu_layers: Some(-1),
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("test-model"));
        assert!(json.contains("/models/model.gguf"));
        assert!(json.contains("/models/mmproj.gguf"));
        assert!(json.contains("</s>"));
    }

    #[test]
    fn test_add_model_response_serialization() {
        let resp = AddModelResponse {
            success: true,
            message: "Model added".to_string(),
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("true"));
        assert!(json.contains("Model added"));
    }

    #[test]
    fn test_error_response_serialization() {
        let resp = AddModelResponse {
            success: false,
            message: "Test error".to_string(),
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("false"));
        assert!(json.contains("Test error"));
    }
}
