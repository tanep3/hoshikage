use crate::config::Config;
use crate::error::{HoshikageError, Result};
use crate::security::{FileTokenStore, SecretToken, TokenName, TokenStore, TokenVerifierRecord};

fn store(config: &Config) -> Result<FileTokenStore> {
    Ok(FileTokenStore::new(config.auth_token_path()?))
}

fn map_error(error: impl std::fmt::Display) -> HoshikageError {
    HoshikageError::ConfigError(error.to_string())
}

pub async fn create_token(name: String) -> Result<()> {
    let config = Config::load()?;
    let name = TokenName::new(name).map_err(map_error)?;
    let token = SecretToken::generate();
    store(&config)?
        .create(TokenVerifierRecord::new(&name, &token))
        .await
        .map_err(map_error)?;
    println!("{}", token.expose_secret());
    Ok(())
}

pub async fn rotate_token(name: String) -> Result<()> {
    let config = Config::load()?;
    let name = TokenName::new(name).map_err(map_error)?;
    let token = SecretToken::generate();
    store(&config)?
        .rotate(&name, TokenVerifierRecord::new(&name, &token))
        .await
        .map_err(map_error)?;
    println!("{}", token.expose_secret());
    Ok(())
}

pub async fn revoke_token(name: String) -> Result<()> {
    let config = Config::load()?;
    let name = TokenName::new(name).map_err(map_error)?;
    store(&config)?.revoke(&name).await.map_err(map_error)?;
    println!("Revoked token {}", name.as_str());
    Ok(())
}

pub async fn list_tokens() -> Result<()> {
    let config = Config::load()?;
    let tokens = store(&config)?.list().await.map_err(map_error)?;
    if tokens.is_empty() {
        println!("No tokens configured.");
        return Ok(());
    }
    for token in tokens {
        println!(
            "{}\t{}\tcreated={}\tupdated={}",
            token.name, token.public_id, token.created_at, token.updated_at
        );
    }
    Ok(())
}
