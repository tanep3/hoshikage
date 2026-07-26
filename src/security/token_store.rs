use super::{SecretToken, TokenName};
use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::Utc;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use subtle::ConstantTimeEq;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenStoreError {
    #[error("token store IO failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("token store data is invalid: {0}")]
    InvalidData(#[from] serde_json::Error),
    #[error("token store permissions are unsafe")]
    UnsafePermissions,
    #[error("token name already exists: {0}")]
    DuplicateName(String),
    #[error("token name was not found: {0}")]
    NotFound(String),
    #[error("at least one bearer token is required")]
    NoTokensConfigured,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenVerifierRecord {
    pub name: String,
    pub public_id: String,
    pub digest: String,
    pub created_at: i64,
    pub updated_at: i64,
}

impl TokenVerifierRecord {
    pub fn new(name: &TokenName, token: &SecretToken) -> Self {
        let now = Utc::now().timestamp();
        Self {
            name: name.as_str().to_string(),
            public_id: token.public_id().to_string(),
            digest: token_digest(token),
            created_at: now,
            updated_at: now,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenMetadata {
    pub name: String,
    pub public_id: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Default)]
pub struct TokenVerifierSet {
    records: Vec<TokenVerifierRecord>,
}

impl TokenVerifierSet {
    pub fn new(records: Vec<TokenVerifierRecord>) -> Self {
        Self { records }
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn verify(&self, token: &SecretToken) -> bool {
        let candidate = digest_bytes(token);
        let mut matched = 0_u8;
        let dummy = [0_u8; 32];

        for record in &self.records {
            let stored = URL_SAFE_NO_PAD
                .decode(&record.digest)
                .unwrap_or_else(|_| dummy.to_vec());
            let stored: [u8; 32] = stored.try_into().unwrap_or(dummy);
            let public_id_matches = u8::from(record.public_id == token.public_id());
            matched |= candidate.ct_eq(&stored).unwrap_u8() & public_id_matches;
        }
        matched == 1
    }
}

#[async_trait]
pub trait TokenStore: Send + Sync {
    async fn load(&self) -> Result<TokenVerifierSet, TokenStoreError>;
    async fn create(&self, record: TokenVerifierRecord) -> Result<(), TokenStoreError>;
    async fn rotate(
        &self,
        name: &TokenName,
        record: TokenVerifierRecord,
    ) -> Result<(), TokenStoreError>;
    async fn revoke(&self, name: &TokenName) -> Result<(), TokenStoreError>;
    async fn list(&self) -> Result<Vec<TokenMetadata>, TokenStoreError>;
}

#[derive(Clone)]
pub struct FileTokenStore {
    path: PathBuf,
}

#[derive(Default, Serialize, Deserialize)]
struct TokenFile {
    version: u32,
    records: Vec<TokenVerifierRecord>,
}

impl FileTokenStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn with_exclusive_lock<T>(
        &self,
        operation: impl FnOnce() -> Result<T, TokenStoreError>,
    ) -> Result<T, TokenStoreError> {
        let parent = self.path.parent().ok_or_else(|| {
            TokenStoreError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "token store path has no parent",
            ))
        })?;
        std::fs::create_dir_all(parent)?;
        let lock_path = parent.join(".auth_tokens.lock");
        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);
        #[cfg(unix)]
        options.mode(0o600);
        let lock = options.open(lock_path)?;
        lock.lock_exclusive()?;
        let result = operation();
        let unlock_result = lock.unlock();
        match (result, unlock_result) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(TokenStoreError::Io(error)),
        }
    }

    fn read_file(&self) -> Result<TokenFile, TokenStoreError> {
        if !self.path.exists() {
            return Ok(TokenFile {
                version: 1,
                records: Vec::new(),
            });
        }
        validate_permissions(&self.path)?;
        let body = std::fs::read(&self.path)?;
        let file: TokenFile = serde_json::from_slice(&body)?;
        if file.version != 1 {
            return Err(TokenStoreError::InvalidData(serde_json::Error::io(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unsupported token store version",
                ),
            )));
        }
        Ok(file)
    }

    fn write_file(&self, file: &TokenFile) -> Result<(), TokenStoreError> {
        let parent = self.path.parent().ok_or_else(|| {
            TokenStoreError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "token store path has no parent",
            ))
        })?;
        std::fs::create_dir_all(parent)?;
        let temporary = parent.join(format!(
            ".auth_tokens.{}.tmp",
            uuid::Uuid::new_v4().simple()
        ));
        let result = (|| -> Result<(), TokenStoreError> {
            let mut options = OpenOptions::new();
            options.create_new(true).write(true);
            #[cfg(unix)]
            options.mode(0o600);
            let mut output = options.open(&temporary)?;
            output.write_all(&serde_json::to_vec_pretty(file)?)?;
            output.sync_all()?;
            std::fs::rename(&temporary, &self.path)?;
            validate_permissions(&self.path)?;
            if let Ok(directory) = OpenOptions::new().read(true).open(parent) {
                let _ = directory.sync_all();
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        result
    }
}

#[async_trait]
impl TokenStore for FileTokenStore {
    async fn load(&self) -> Result<TokenVerifierSet, TokenStoreError> {
        Ok(TokenVerifierSet::new(self.read_file()?.records))
    }

    async fn create(&self, record: TokenVerifierRecord) -> Result<(), TokenStoreError> {
        self.with_exclusive_lock(move || {
            let mut file = self.read_file()?;
            if file.records.iter().any(|item| item.name == record.name) {
                return Err(TokenStoreError::DuplicateName(record.name));
            }
            if file
                .records
                .iter()
                .any(|item| item.public_id == record.public_id)
            {
                return Err(TokenStoreError::InvalidData(serde_json::Error::io(
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "token public ID already exists",
                    ),
                )));
            }
            file.records.push(record);
            self.write_file(&file)
        })
    }

    async fn rotate(
        &self,
        name: &TokenName,
        mut record: TokenVerifierRecord,
    ) -> Result<(), TokenStoreError> {
        self.with_exclusive_lock(move || {
            let mut file = self.read_file()?;
            let existing = file
                .records
                .iter_mut()
                .find(|item| item.name == name.as_str())
                .ok_or_else(|| TokenStoreError::NotFound(name.as_str().to_string()))?;
            record.name = name.as_str().to_string();
            record.created_at = existing.created_at;
            record.updated_at = Utc::now().timestamp();
            *existing = record;
            self.write_file(&file)
        })
    }

    async fn revoke(&self, name: &TokenName) -> Result<(), TokenStoreError> {
        self.with_exclusive_lock(|| {
            let mut file = self.read_file()?;
            let before = file.records.len();
            file.records.retain(|item| item.name != name.as_str());
            if file.records.len() == before {
                return Err(TokenStoreError::NotFound(name.as_str().to_string()));
            }
            self.write_file(&file)
        })
    }

    async fn list(&self) -> Result<Vec<TokenMetadata>, TokenStoreError> {
        let mut records = self
            .read_file()?
            .records
            .into_iter()
            .map(|record| TokenMetadata {
                name: record.name,
                public_id: record.public_id,
                created_at: record.created_at,
                updated_at: record.updated_at,
            })
            .collect::<Vec<_>>();
        records.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(records)
    }
}

fn digest_bytes(token: &SecretToken) -> [u8; 32] {
    Sha256::digest(token.expose_secret().as_bytes()).into()
}

fn token_digest(token: &SecretToken) -> String {
    URL_SAFE_NO_PAD.encode(digest_bytes(token))
}

#[cfg(unix)]
fn validate_permissions(path: &Path) -> Result<(), TokenStoreError> {
    if path.metadata()?.mode() & 0o077 != 0 {
        return Err(TokenStoreError::UnsafePermissions);
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_permissions(_path: &Path) -> Result<(), TokenStoreError> {
    Err(TokenStoreError::UnsafePermissions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_path() -> PathBuf {
        std::env::temp_dir()
            .join(format!("hoshikage-token-{}", uuid::Uuid::new_v4()))
            .join("tokens.json")
    }

    #[tokio::test]
    async fn create_rotate_revoke_updates_verification_immediately() {
        let path = store_path();
        let store = FileTokenStore::new(path.clone());
        let name = TokenName::new("codex-lan").unwrap();
        let first = SecretToken::generate();
        store
            .create(TokenVerifierRecord::new(&name, &first))
            .await
            .unwrap();
        assert!(store.load().await.unwrap().verify(&first));

        let second = SecretToken::generate();
        store
            .rotate(&name, TokenVerifierRecord::new(&name, &second))
            .await
            .unwrap();
        let verifiers = store.load().await.unwrap();
        assert!(!verifiers.verify(&first));
        assert!(verifiers.verify(&second));

        store.revoke(&name).await.unwrap();
        assert!(!store.load().await.unwrap().verify(&second));
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[tokio::test]
    async fn named_tokens_are_independent() {
        let path = store_path();
        let store = FileTokenStore::new(path.clone());
        let codex_name = TokenName::new("codex-lan").unwrap();
        let yatagarasu_name = TokenName::new("yatagarasu-lan").unwrap();
        let codex = SecretToken::generate();
        let yatagarasu = SecretToken::generate();
        store
            .create(TokenVerifierRecord::new(&codex_name, &codex))
            .await
            .unwrap();
        store
            .create(TokenVerifierRecord::new(&yatagarasu_name, &yatagarasu))
            .await
            .unwrap();

        let verifiers = store.load().await.unwrap();
        assert!(verifiers.verify(&codex));
        assert!(verifiers.verify(&yatagarasu));

        store.revoke(&codex_name).await.unwrap();
        let verifiers = store.load().await.unwrap();
        assert!(!verifiers.verify(&codex));
        assert!(verifiers.verify(&yatagarasu));
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unsafe_permissions_fail_closed() {
        use std::os::unix::fs::PermissionsExt;

        let path = store_path();
        let store = FileTokenStore::new(path.clone());
        let name = TokenName::new("codex-lan").unwrap();
        let token = SecretToken::generate();
        store
            .create(TokenVerifierRecord::new(&name, &token))
            .await
            .unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

        assert!(matches!(
            store.load().await,
            Err(TokenStoreError::UnsafePermissions)
        ));
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
}
