pub mod admin;
pub mod chat;
pub mod models;

pub use chat::ChatMessage;

use axum::{
    middleware,
    routing::{delete, get, post},
    Router,
};
use std::sync::Arc;

pub fn create_router(manager: Arc<crate::model::ModelManager>) -> Router {
    base_router(manager)
}

pub fn create_router_with_auth(
    manager: Arc<crate::model::ModelManager>,
    auth: crate::security::AuthState,
) -> Router {
    base_router(manager).layer(middleware::from_fn_with_state(
        auth,
        crate::security::authenticate,
    ))
}

fn base_router(manager: Arc<crate::model::ModelManager>) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/v1/models", get(models::models))
        .route("/v1/status", get(models::status))
        .route("/v1/hoshikage/models", get(models::hoshikage_models))
        .route("/v1/hoshikage/models/:name", get(models::hoshikage_model))
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

#[cfg(test)]
mod auth_integration_tests {
    use super::*;
    use crate::config::Config;
    use crate::security::{
        AuthPolicy, AuthState, FileTokenStore, SecretToken, TokenName, TokenStore,
        TokenVerifierRecord,
    };
    use axum::body::Body;
    use axum::http::{header, Request, StatusCode};
    use tower::ServiceExt;

    fn store_path() -> std::path::PathBuf {
        std::env::temp_dir()
            .join(format!("hoshikage-router-auth-{}", uuid::Uuid::new_v4()))
            .join("tokens.json")
    }

    fn manager() -> Arc<crate::model::ModelManager> {
        Arc::new(crate::model::ModelManager::new(Config::default()))
    }

    #[tokio::test]
    async fn loopback_policy_allows_requests_without_a_token() {
        let path = store_path();
        let router = create_router_with_auth(
            manager(),
            AuthState {
                policy: AuthPolicy::LoopbackOpen,
                store: Arc::new(FileTokenStore::new(path)),
            },
        );

        let response = router
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn bearer_policy_rejects_missing_token_and_accepts_valid_token() {
        let path = store_path();
        let store = Arc::new(FileTokenStore::new(path.clone()));
        let name = TokenName::new("codex-lan").unwrap();
        let token = SecretToken::generate();
        store
            .create(TokenVerifierRecord::new(&name, &token))
            .await
            .unwrap();
        let router = create_router_with_auth(
            manager(),
            AuthState {
                policy: AuthPolicy::BearerRequired,
                store,
            },
        );

        let rejected = router
            .clone()
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::UNAUTHORIZED);

        let accepted = router
            .oneshot(
                Request::get("/")
                    .header(
                        header::AUTHORIZATION,
                        format!("Bearer {}", token.expose_secret()),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);

        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
}
