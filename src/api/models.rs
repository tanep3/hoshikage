use axum::{extract::State, Json};
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
}

pub async fn status() -> Json<StatusResponse> {
    Json(StatusResponse {
        status: "ok".to_string(),
    })
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
