pub mod admin;
pub mod chat;
pub mod error;
pub mod health;
pub mod models;
pub mod responses;

pub use chat::ChatMessage;

use axum::{
    extract::DefaultBodyLimit,
    middleware,
    routing::{delete, get, post},
    Extension, Router,
};
use std::sync::Arc;

pub fn create_router(manager: Arc<crate::model::ModelManager>) -> Router {
    let responses = responses_service(&manager);
    protected_router(manager, responses).merge(public_router())
}

pub fn create_router_with_auth(
    manager: Arc<crate::model::ModelManager>,
    auth: crate::security::AuthState,
) -> Router {
    let responses = responses_service(&manager);
    protected_router(manager, responses)
        .layer(middleware::from_fn_with_state(
            auth,
            crate::security::authenticate,
        ))
        .merge(public_router())
}

fn responses_service(
    manager: &Arc<crate::model::ModelManager>,
) -> Arc<crate::application::ResponsesService> {
    Arc::new(crate::application::ResponsesService::new(
        Arc::new(crate::inference::ModelManagerGateway::new(manager.clone())),
        manager.responses_unknown_field_policy(),
        std::time::Duration::from_secs(manager.responses_timeout_secs()),
    ))
}

fn protected_router(
    manager: Arc<crate::model::ModelManager>,
    responses: Arc<crate::application::ResponsesService>,
) -> Router {
    let max_request_bytes = manager.max_request_bytes();
    Router::new()
        .route("/", get(root))
        .route("/ready", get(health::ready))
        .route("/v1/models", get(models::models))
        .route("/v1/status", get(models::status))
        .route("/v1/capabilities", get(health::capabilities))
        .route("/v1/hoshikage/models", get(models::hoshikage_models))
        .route("/v1/hoshikage/models/:name", get(models::hoshikage_model))
        .route("/v1/api/version", get(models::version))
        .route("/v1/chat/completions", post(chat::chat_completion))
        .route(
            "/v1/responses",
            post(responses::handler::responses).layer(DefaultBodyLimit::max(max_request_bytes)),
        )
        .route("/admin/models", post(admin::add_model))
        .route("/admin/models/:name", delete(admin::remove_model))
        .route("/admin/reload", post(admin::reload_models))
        .with_state(manager)
        .layer(Extension(responses))
}

fn public_router() -> Router {
    Router::new().route("/health", get(health::health))
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
    use axum::body::{to_bytes, Body};
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

        let ready_rejected = router
            .clone()
            .oneshot(Request::get("/ready").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(ready_rejected.status(), StatusCode::UNAUTHORIZED);

        let health = router
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);

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

    #[tokio::test]
    async fn responses_route_enforces_configured_body_limit_with_json_error() {
        let manager = Arc::new(crate::model::ModelManager::new(Config {
            max_request_bytes: 64,
            max_tool_result_bytes: 32,
            ..Config::default()
        }));
        let response = create_router(manager)
            .oneshot(
                Request::post("/v1/responses")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(format!(
                        r#"{{"model":"missing","input":"{}"}}"#,
                        "x".repeat(128)
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], "invalid_request");
    }
}
