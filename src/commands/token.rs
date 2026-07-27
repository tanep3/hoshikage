use crate::config::Config;
use crate::error::{HoshikageError, Result};
use crate::i18n::Language;
use crate::security::{
    FileTokenStore, SecretToken, StoredTokenRecord, TokenMetadata, TokenName, TokenStore,
};

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
        .create(StoredTokenRecord::new(&name, &token))
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
        .rotate(&name, StoredTokenRecord::new(&name, &token))
        .await
        .map_err(map_error)?;
    println!("{}", token.expose_secret());
    Ok(())
}

pub async fn revoke_token(name: String, language: Language) -> Result<()> {
    let config = Config::load()?;
    let name = TokenName::new(name).map_err(map_error)?;
    store(&config)?.revoke(&name).await.map_err(map_error)?;
    match language {
        Language::En => println!("Revoked token {}", name.as_str()),
        Language::Ja => println!("Token {} を無効化しました", name.as_str()),
    }
    Ok(())
}

pub async fn list_tokens(language: Language) -> Result<()> {
    let config = Config::load()?;
    let tokens = store(&config)?.list().await.map_err(map_error)?;
    if tokens.is_empty() {
        println!(
            "{}",
            language.select("No tokens configured.", "Tokenは設定されていません。")
        );
        return Ok(());
    }
    for token in tokens {
        println!("{}", TokenListEntry(&token));
    }
    Ok(())
}

struct TokenListEntry<'a>(&'a TokenMetadata);

impl std::fmt::Display for TokenListEntry<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let token = self.0;
        let secret = token
            .token()
            .map(SecretToken::expose_secret)
            .unwrap_or("<unavailable: rotate required>");
        write!(
            formatter,
            "{}\t{}\tpublic_id={}\tcreated={}\tupdated={}",
            token.name, secret, token.public_id, token.created_at, token.updated_at
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroize::Zeroize;

    #[tokio::test]
    async fn administrator_list_line_contains_the_complete_token() {
        let path = std::env::temp_dir()
            .join(format!("hoshikage-token-cli-{}", uuid::Uuid::new_v4()))
            .join("tokens.json");
        let store = FileTokenStore::new(path.clone());
        let name = TokenName::new("codex-desktop").unwrap();
        let secret = SecretToken::generate();
        store
            .create(StoredTokenRecord::new(&name, &secret))
            .await
            .unwrap();

        let listed = store.list().await.unwrap();
        let mut line = TokenListEntry(&listed[0]).to_string();
        assert!(line.contains("codex-desktop"));
        assert!(line.contains(secret.expose_secret()));
        assert!(line.contains(secret.public_id()));
        line.zeroize();

        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
}
