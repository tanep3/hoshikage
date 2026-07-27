use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use std::sync::Arc;

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ReadyResponse {
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct CapabilitiesResponse {
    pub responses_api: bool,
    pub streaming: bool,
    pub function_calling: bool,
    pub parallel_tool_calls: bool,
    pub vision: bool,
}

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

pub async fn ready(State(manager): State<Arc<crate::model::ModelManager>>) -> Response {
    if manager.is_ready() {
        Json(ReadyResponse {
            status: "ready",
            reason: None,
        })
        .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadyResponse {
                status: "not_ready",
                reason: Some("runtime_coordinator_unavailable"),
            }),
        )
            .into_response()
    }
}

pub async fn capabilities() -> Json<CapabilitiesResponse> {
    Json(CapabilitiesResponse {
        responses_api: true,
        streaming: true,
        function_calling: true,
        parallel_tool_calls: false,
        vision: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn phase_four_capabilities_advertise_streaming_without_parallel_tools() {
        let response = capabilities().await;

        assert!(response.responses_api);
        assert!(response.streaming);
        assert!(response.function_calling);
        assert!(!response.parallel_tool_calls);
        assert!(!response.vision);
    }
}
