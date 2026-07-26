use super::{AuthPolicy, FileTokenStore, SecretToken, TokenStore};
use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use std::sync::Arc;

#[derive(Clone)]
pub struct AuthState {
    pub policy: AuthPolicy,
    pub store: Arc<FileTokenStore>,
}

impl AuthState {
    pub async fn validated(
        policy: AuthPolicy,
        store: Arc<FileTokenStore>,
    ) -> Result<Self, super::TokenStoreError> {
        if policy.requires_bearer() && store.load().await?.is_empty() {
            return Err(super::TokenStoreError::NoTokensConfigured);
        }
        Ok(Self { policy, store })
    }
}

pub async fn authenticate(
    State(state): State<AuthState>,
    request: Request,
    next: Next,
) -> Response {
    if !state.policy.requires_bearer() {
        return next.run(request).await;
    }

    let token = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .and_then(|value| SecretToken::parse(value.to_string()).ok());
    let authorized = match token {
        Some(token) => state
            .store
            .load()
            .await
            .map(|verifiers| verifiers.verify(&token))
            .unwrap_or(false),
        None => false,
    };
    if !authorized {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({
                "error": {
                    "message": "Invalid or missing bearer token",
                    "type": "authentication_error",
                    "param": null,
                    "code": "invalid_api_key"
                }
            })),
        )
            .into_response();
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_path() -> std::path::PathBuf {
        std::env::temp_dir()
            .join(format!("hoshikage-auth-state-{}", uuid::Uuid::new_v4()))
            .join("tokens.json")
    }

    #[tokio::test]
    async fn non_loopback_state_rejects_an_empty_token_store() {
        let store = Arc::new(FileTokenStore::new(store_path()));

        let result = AuthState::validated(AuthPolicy::BearerRequired, store).await;

        assert!(matches!(
            result,
            Err(super::super::TokenStoreError::NoTokensConfigured)
        ));
    }

    #[tokio::test]
    async fn loopback_state_allows_an_empty_token_store() {
        let store = Arc::new(FileTokenStore::new(store_path()));

        let result = AuthState::validated(AuthPolicy::LoopbackOpen, store).await;

        assert!(result.is_ok());
    }
}
