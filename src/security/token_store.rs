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
use zeroize::Zeroize;

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

#[derive(Serialize, Deserialize)]
pub struct StoredTokenRecord {
    pub name: String,
    pub public_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    token: Option<String>,
    pub digest: String,
    pub created_at: i64,
    pub updated_at: i64,
}

impl StoredTokenRecord {
    pub fn new(name: &TokenName, token: &SecretToken) -> Self {
        let now = Utc::now().timestamp();
        Self {
            name: name.as_str().to_string(),
            public_id: token.public_id().to_string(),
            token: Some(token.expose_secret().to_string()),
            digest: token_digest(token),
            created_at: now,
            updated_at: now,
        }
    }

    fn verifier(&self) -> TokenVerifierRecord {
        TokenVerifierRecord {
            public_id: self.public_id.clone(),
            digest: self.digest.clone(),
        }
    }

    fn validate(&self) -> Result<(), TokenStoreError> {
        let Some(value) = &self.token else {
            return Ok(());
        };
        let token = SecretToken::parse(value.clone()).map_err(|error| {
            invalid_data(format!("stored token {} is invalid: {error}", self.name))
        })?;
        if token.public_id() != self.public_id || token_digest(&token) != self.digest {
            return Err(invalid_data(format!(
                "stored token {} does not match its verifier",
                self.name
            )));
        }
        Ok(())
    }
}

impl Drop for StoredTokenRecord {
    fn drop(&mut self) {
        if let Some(token) = &mut self.token {
            token.zeroize();
        }
    }
}

#[derive(Clone)]
struct TokenVerifierRecord {
    public_id: String,
    digest: String,
}

pub struct TokenMetadata {
    pub name: String,
    pub public_id: String,
    pub created_at: i64,
    pub updated_at: i64,
    token: Option<SecretToken>,
}

impl TokenMetadata {
    pub fn token(&self) -> Option<&SecretToken> {
        self.token.as_ref()
    }
}

#[derive(Clone, Default)]
pub struct TokenVerifierSet {
    records: Vec<TokenVerifierRecord>,
}

impl TokenVerifierSet {
    fn new(records: Vec<TokenVerifierRecord>) -> Self {
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
    async fn create(&self, record: StoredTokenRecord) -> Result<(), TokenStoreError>;
    async fn rotate(
        &self,
        name: &TokenName,
        record: StoredTokenRecord,
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
    records: Vec<StoredTokenRecord>,
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
        restrict_file_permissions(&lock, self.path.parent().unwrap().join(".auth_tokens.lock"))?;
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
        let mut body = std::fs::read(&self.path)?;
        let parsed = serde_json::from_slice(&body);
        body.zeroize();
        let file: TokenFile = parsed?;
        if !matches!(file.version, 1 | 2) {
            return Err(TokenStoreError::InvalidData(serde_json::Error::io(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unsupported token store version",
                ),
            )));
        }
        for record in &file.records {
            record.validate()?;
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
            restrict_file_permissions(&output, temporary.clone())?;
            let mut body = serde_json::to_vec_pretty(file)?;
            let write_result = output.write_all(&body);
            body.zeroize();
            write_result?;
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
        Ok(TokenVerifierSet::new(
            self.read_file()?
                .records
                .iter()
                .map(StoredTokenRecord::verifier)
                .collect(),
        ))
    }

    async fn create(&self, record: StoredTokenRecord) -> Result<(), TokenStoreError> {
        self.with_exclusive_lock(move || {
            let mut file = self.read_file()?;
            if file.records.iter().any(|item| item.name == record.name) {
                return Err(TokenStoreError::DuplicateName(record.name.clone()));
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
            file.version = 2;
            self.write_file(&file)
        })
    }

    async fn rotate(
        &self,
        name: &TokenName,
        mut record: StoredTokenRecord,
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
            file.version = 2;
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
            .map(|mut record| {
                let token = record
                    .token
                    .take()
                    .map(SecretToken::parse)
                    .transpose()
                    .map_err(|error| invalid_data(error.to_string()))?;
                Ok(TokenMetadata {
                    name: record.name.clone(),
                    public_id: record.public_id.clone(),
                    created_at: record.created_at,
                    updated_at: record.updated_at,
                    token,
                })
            })
            .collect::<Result<Vec<_>, TokenStoreError>>()?;
        records.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(records)
    }
}

fn invalid_data(message: impl Into<String>) -> TokenStoreError {
    TokenStoreError::InvalidData(serde_json::Error::io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message.into(),
    )))
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

#[cfg(unix)]
fn restrict_file_permissions(_file: &std::fs::File, _path: PathBuf) -> Result<(), TokenStoreError> {
    Ok(())
}

#[cfg(windows)]
fn restrict_file_permissions(_file: &std::fs::File, path: PathBuf) -> Result<(), TokenStoreError> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{GetLastError, LocalFree};
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::{
        SetFileSecurityW, DACL_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION,
    };

    let sddl = "D:P(A;;FA;;;SY)(A;;FA;;;OW)"
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let path = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let mut descriptor = std::ptr::null_mut();
    let converted = unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.as_ptr(),
            SDDL_REVISION_1,
            &mut descriptor,
            std::ptr::null_mut(),
        )
    };
    if converted == 0 {
        return Err(windows_permission_error(unsafe { GetLastError() }));
    }
    let applied = unsafe {
        SetFileSecurityW(
            path.as_ptr(),
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            descriptor,
        )
    };
    let error = if applied == 0 {
        Some(unsafe { GetLastError() })
    } else {
        None
    };
    unsafe {
        LocalFree(descriptor);
    }
    match error {
        Some(error) => Err(windows_permission_error(error)),
        None => Ok(()),
    }
}

#[cfg(windows)]
fn validate_permissions(path: &Path) -> Result<(), TokenStoreError> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Authorization::{GetNamedSecurityInfoW, SE_FILE_OBJECT};
    use windows_sys::Win32::Security::{
        CreateWellKnownSid, EqualSid, GetAce, GetSecurityDescriptorControl,
        WinCreatorOwnerRightsSid, WinLocalSystemSid, ACCESS_ALLOWED_ACE, ACL,
        DACL_SECURITY_INFORMATION, OWNER_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, PSID,
        SECURITY_MAX_SID_SIZE, SE_DACL_PROTECTED,
    };
    use windows_sys::Win32::Storage::FileSystem::FILE_ALL_ACCESS;
    use windows_sys::Win32::System::SystemServices::ACCESS_ALLOWED_ACE_TYPE;

    let path = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let mut owner: PSID = std::ptr::null_mut();
    let mut dacl: *mut ACL = std::ptr::null_mut();
    let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
    let status = unsafe {
        GetNamedSecurityInfoW(
            path.as_ptr(),
            SE_FILE_OBJECT,
            OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
            &mut owner,
            std::ptr::null_mut(),
            &mut dacl,
            std::ptr::null_mut(),
            &mut descriptor,
        )
    };
    if status != 0 {
        return Err(windows_permission_error(status));
    }

    let result = (|| {
        if owner.is_null() || dacl.is_null() {
            return Err(TokenStoreError::UnsafePermissions);
        }
        let mut control = 0_u16;
        let mut revision = 0_u32;
        if unsafe { GetSecurityDescriptorControl(descriptor, &mut control, &mut revision) } == 0
            || control & SE_DACL_PROTECTED == 0
        {
            return Err(TokenStoreError::UnsafePermissions);
        }
        let mut owner_rights_sid = [0_u8; SECURITY_MAX_SID_SIZE as usize];
        let mut owner_rights_sid_len = owner_rights_sid.len() as u32;
        if unsafe {
            CreateWellKnownSid(
                WinCreatorOwnerRightsSid,
                std::ptr::null_mut(),
                owner_rights_sid.as_mut_ptr().cast(),
                &mut owner_rights_sid_len,
            )
        } == 0
        {
            return Err(TokenStoreError::UnsafePermissions);
        }

        let mut system_sid = [0_u8; SECURITY_MAX_SID_SIZE as usize];
        let mut system_sid_len = system_sid.len() as u32;
        if unsafe {
            CreateWellKnownSid(
                WinLocalSystemSid,
                std::ptr::null_mut(),
                system_sid.as_mut_ptr().cast(),
                &mut system_sid_len,
            )
        } == 0
        {
            return Err(TokenStoreError::UnsafePermissions);
        }

        let ace_count = unsafe { (*dacl).AceCount };
        if ace_count != 2 {
            return Err(TokenStoreError::UnsafePermissions);
        }
        let mut owner_found = false;
        let mut system_found = false;
        for index in 0..u32::from(ace_count) {
            let mut raw_ace = std::ptr::null_mut();
            if unsafe { GetAce(dacl, index, &mut raw_ace) } == 0 || raw_ace.is_null() {
                return Err(TokenStoreError::UnsafePermissions);
            }
            let ace = unsafe { &*(raw_ace.cast::<ACCESS_ALLOWED_ACE>()) };
            if u32::from(ace.Header.AceType) != ACCESS_ALLOWED_ACE_TYPE
                || ace.Header.AceFlags != 0
                || ace.Mask != FILE_ALL_ACCESS
            {
                return Err(TokenStoreError::UnsafePermissions);
            }
            let sid = std::ptr::addr_of!(ace.SidStart).cast_mut().cast();
            if unsafe { EqualSid(sid, owner_rights_sid.as_mut_ptr().cast()) } != 0 {
                owner_found = true;
            } else if unsafe { EqualSid(sid, system_sid.as_mut_ptr().cast()) } != 0 {
                system_found = true;
            } else {
                return Err(TokenStoreError::UnsafePermissions);
            }
        }
        if owner_found && system_found {
            Ok(())
        } else {
            Err(TokenStoreError::UnsafePermissions)
        }
    })();
    unsafe {
        LocalFree(descriptor);
    }
    result
}

#[cfg(windows)]
fn windows_permission_error(error: u32) -> TokenStoreError {
    TokenStoreError::Io(std::io::Error::from_raw_os_error(error as i32))
}

#[cfg(not(any(unix, windows)))]
fn restrict_file_permissions(_file: &std::fs::File, _path: PathBuf) -> Result<(), TokenStoreError> {
    Err(TokenStoreError::UnsafePermissions)
}

#[cfg(not(any(unix, windows)))]
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
            .create(StoredTokenRecord::new(&name, &first))
            .await
            .unwrap();
        assert!(store.load().await.unwrap().verify(&first));
        let listed = store.list().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(
            listed[0].token().unwrap().expose_secret(),
            first.expose_secret()
        );
        let mut persisted = std::fs::read_to_string(&path).unwrap();
        assert!(persisted.contains(first.expose_secret()));
        assert!(persisted.contains(r#""version": 2"#));
        persisted.zeroize();

        let second = SecretToken::generate();
        store
            .rotate(&name, StoredTokenRecord::new(&name, &second))
            .await
            .unwrap();
        let verifiers = store.load().await.unwrap();
        assert!(!verifiers.verify(&first));
        assert!(verifiers.verify(&second));
        let listed = store.list().await.unwrap();
        assert_eq!(
            listed[0].token().unwrap().expose_secret(),
            second.expose_secret()
        );

        store.revoke(&name).await.unwrap();
        assert!(!store.load().await.unwrap().verify(&second));
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn legacy_digest_only_tokens_verify_until_rotated_into_recoverable_storage() {
        use std::os::unix::fs::PermissionsExt;

        let path = store_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let name = TokenName::new("legacy-client").unwrap();
        let legacy = SecretToken::generate();
        let body = serde_json::json!({
            "version": 1,
            "records": [{
                "name": name.as_str(),
                "public_id": legacy.public_id(),
                "digest": token_digest(&legacy),
                "created_at": 1,
                "updated_at": 1
            }]
        });
        std::fs::write(&path, serde_json::to_vec_pretty(&body).unwrap()).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        let store = FileTokenStore::new(path.clone());
        assert!(store.load().await.unwrap().verify(&legacy));
        assert!(store.list().await.unwrap()[0].token().is_none());

        let replacement = SecretToken::generate();
        store
            .rotate(&name, StoredTokenRecord::new(&name, &replacement))
            .await
            .unwrap();
        let listed = store.list().await.unwrap();
        assert_eq!(
            listed[0].token().unwrap().expose_secret(),
            replacement.expose_secret()
        );
        assert!(!store.load().await.unwrap().verify(&legacy));
        assert!(store.load().await.unwrap().verify(&replacement));
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
            .create(StoredTokenRecord::new(&codex_name, &codex))
            .await
            .unwrap();
        store
            .create(StoredTokenRecord::new(&yatagarasu_name, &yatagarasu))
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
            .create(StoredTokenRecord::new(&name, &token))
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
