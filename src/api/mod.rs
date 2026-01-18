pub mod admin;
pub mod chat;
pub mod models;

pub use chat::ChatMessage;

use axum::{
    routing::{delete, get, post},
    Router,
};
use std::sync::Arc;

pub fn create_router(manager: Arc<crate::model::ModelManager>) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/v1/models", get(models::models))
        .route("/v1/status", get(models::status))
        .route("/v1/api/version", get(models::version))
        .route("/v1/chat/completions", post(chat::chat_completion))
        .route("/admin/models", post(admin::add_model))
        .route("/admin/models/:name", delete(admin::remove_model))
        .route("/admin/reload", post(admin::reload_models))
        .with_state(manager)
}

async fn root() -> &'static str {
    "Hoshikage API Server"
}
