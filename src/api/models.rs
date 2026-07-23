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
}

pub async fn models(
    State(manager): State<Arc<crate::model::ModelManager>>,
) -> Json<ModelListResponse> {
    let model_names = manager.list_models().await;

    let data = model_names
        .iter()
        .map(|name| ModelData {
            id: name.clone(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "tane".to_string(),
        })
        .collect();

    Json(ModelListResponse {
        object: "list".to_string(),
        data,
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
                    "message": "指定されたモデルが見つかりません",
                    "type": "invalid_request",
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
