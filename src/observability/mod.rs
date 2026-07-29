use crate::conversation::{ToolArguments, ToolOutputContent};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use thiserror::Error;

pub struct Redacted<T>(pub T);

impl<T> fmt::Debug for Redacted<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<redacted>")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolPayloadSummary {
    pub bytes: usize,
    pub json_valid: bool,
    pub schema_valid: Option<bool>,
    pub truncated: bool,
}

impl ToolPayloadSummary {
    pub fn from_arguments(arguments: &ToolArguments, schema_valid: Option<bool>) -> Self {
        Self {
            bytes: arguments.canonical_json().len(),
            json_valid: true,
            schema_valid,
            truncated: false,
        }
    }

    pub fn from_tool_output(content: &ToolOutputContent, truncated: bool) -> Self {
        Self {
            bytes: content.encoded_bytes(),
            json_valid: false,
            schema_valid: None,
            truncated,
        }
    }
}

const DEFAULT_CAPTURE_RETENTION: Duration = Duration::from_secs(24 * 60 * 60);
const DEFAULT_CAPTURE_DIRECTORY_BYTES: u64 = 100 * 1024 * 1024;
const DEFAULT_SINGLE_CAPTURE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum DebugCaptureError {
    #[error("debug capture I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("debug capture serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("debug capture record exceeds the configured size limit")]
    RecordTooLarge,
    #[error("invalid debug capture request id")]
    InvalidRequestId,
    #[error("debug capture worker failed")]
    Worker,
    #[error("debug capture lock is poisoned")]
    Lock,
}

#[derive(Debug, Clone)]
pub struct DebugCapture {
    directory: PathBuf,
    retention: Duration,
    max_directory_bytes: u64,
    max_single_capture_bytes: usize,
    lock: Arc<Mutex<()>>,
}

impl DebugCapture {
    pub fn new(directory: PathBuf) -> Result<Self, DebugCaptureError> {
        Self::with_limits(
            directory,
            DEFAULT_CAPTURE_RETENTION,
            DEFAULT_CAPTURE_DIRECTORY_BYTES,
            DEFAULT_SINGLE_CAPTURE_BYTES,
        )
    }

    fn with_limits(
        directory: PathBuf,
        retention: Duration,
        max_directory_bytes: u64,
        max_single_capture_bytes: usize,
    ) -> Result<Self, DebugCaptureError> {
        prepare_private_directory(&directory)?;
        let capture = Self {
            directory,
            retention,
            max_directory_bytes,
            max_single_capture_bytes,
            lock: Arc::new(Mutex::new(())),
        };
        capture.prune_sync()?;
        Ok(capture)
    }

    pub async fn capture(
        &self,
        request_id: &str,
        kind: &str,
        payload: &serde_json::Value,
    ) -> Result<(), DebugCaptureError> {
        if !request_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(DebugCaptureError::InvalidRequestId);
        }
        let this = self.clone();
        let request_id = request_id.to_string();
        let kind = kind.to_string();
        let payload = sanitize_capture_value(payload.clone());
        tokio::task::spawn_blocking(move || {
            let _guard = this.lock.lock().map_err(|_| DebugCaptureError::Lock)?;
            this.write_record(&request_id, &kind, &payload)?;
            this.prune_sync()
        })
        .await
        .map_err(|_| DebugCaptureError::Worker)?
    }

    pub async fn prune(&self) -> Result<(), DebugCaptureError> {
        let this = self.clone();
        tokio::task::spawn_blocking(move || {
            let _guard = this.lock.lock().map_err(|_| DebugCaptureError::Lock)?;
            this.prune_sync()
        })
        .await
        .map_err(|_| DebugCaptureError::Worker)?
    }

    fn write_record(
        &self,
        request_id: &str,
        kind: &str,
        payload: &serde_json::Value,
    ) -> Result<(), DebugCaptureError> {
        use std::io::Write;

        let record = serde_json::to_vec(&serde_json::json!({
            "captured_at": chrono::Utc::now().to_rfc3339(),
            "kind": kind,
            "payload": payload,
        }))?;
        if record.len() > self.max_single_capture_bytes {
            return Err(DebugCaptureError::RecordTooLarge);
        }
        let path = self.directory.join(format!("{request_id}.jsonl"));
        let mut options = std::fs::OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&path)?;
        file.write_all(&record)?;
        file.write_all(b"\n")?;
        restrict_file_permissions(&path)?;
        Ok(())
    }

    fn prune_sync(&self) -> Result<(), DebugCaptureError> {
        let now = SystemTime::now();
        let mut files = capture_files(&self.directory)?;
        for file in &files {
            let age = now.duration_since(file.modified).unwrap_or(Duration::ZERO);
            if age > self.retention {
                std::fs::remove_file(&file.path)?;
            }
        }
        files = capture_files(&self.directory)?;
        let mut total_bytes = files.iter().map(|file| file.bytes).sum::<u64>();
        for file in files {
            if total_bytes <= self.max_directory_bytes {
                break;
            }
            std::fs::remove_file(&file.path)?;
            total_bytes = total_bytes.saturating_sub(file.bytes);
        }
        Ok(())
    }
}

#[derive(Debug)]
struct CaptureFile {
    path: PathBuf,
    modified: SystemTime,
    bytes: u64,
}

fn capture_files(directory: &Path) -> Result<Vec<CaptureFile>, DebugCaptureError> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
            continue;
        }
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            files.push(CaptureFile {
                path,
                modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                bytes: metadata.len(),
            });
        }
    }
    files.sort_by_key(|file| file.modified);
    Ok(files)
}

fn sanitize_capture_value(mut value: serde_json::Value) -> serde_json::Value {
    match &mut value {
        serde_json::Value::Object(object) => {
            object.retain(|key, _| {
                !matches!(
                    key.to_ascii_lowercase().as_str(),
                    "authorization" | "api_key" | "token" | "metadata"
                )
            });
            for value in object.values_mut() {
                *value = sanitize_capture_value(value.take());
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                *value = sanitize_capture_value(value.take());
            }
        }
        _ => {}
    }
    value
}

fn prepare_private_directory(path: &Path) -> Result<(), DebugCaptureError> {
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
        if std::fs::metadata(path)?.permissions().mode() & 0o077 != 0 {
            return Err(DebugCaptureError::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "debug capture directory is accessible by group or other users",
            )));
        }
    }
    Ok(())
}

fn restrict_file_permissions(_path: &Path) -> Result<(), DebugCaptureError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(_path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacted_debug_never_exposes_wrapped_secret() {
        let output = format!("{:?}", Redacted("super-secret-tool-result"));

        assert_eq!(output, "<redacted>");
        assert!(!output.contains("super-secret"));
    }

    #[test]
    fn tool_summary_contains_shape_only() {
        let arguments = ToolArguments::parse(r#"{"path":"private/customer-record.txt"}"#).unwrap();
        let summary = ToolPayloadSummary::from_arguments(&arguments, Some(true));
        let output = format!("{summary:?}");

        assert_eq!(summary.bytes, arguments.canonical_json().len());
        assert!(summary.json_valid);
        assert_eq!(summary.schema_valid, Some(true));
        assert!(!output.contains("customer-record"));
    }

    #[tokio::test]
    async fn debug_capture_redacts_secrets_and_uses_private_permissions() {
        let root = std::env::temp_dir().join(format!(
            "hoshikage-debug-capture-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let capture = DebugCapture::with_limits(
            root.clone(),
            std::time::Duration::from_secs(86_400),
            100 * 1024 * 1024,
            16 * 1024 * 1024,
        )
        .unwrap();
        capture
            .capture(
                "req_test",
                "request",
                &serde_json::json!({
                    "input": "safe prompt",
                    "authorization": "Bearer secret",
                    "Authorization": "Bearer uppercase-secret",
                    "metadata": {"customer": "hidden"},
                    "token": "secret-token"
                }),
            )
            .await
            .unwrap();

        let content = std::fs::read_to_string(root.join("req_test.jsonl")).unwrap();
        assert!(content.contains("safe prompt"));
        assert!(!content.contains("Bearer secret"));
        assert!(!content.contains("uppercase-secret"));
        assert!(!content.contains("secret-token"));
        assert!(!content.contains("customer"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&root).unwrap().permissions().mode() & 0o077,
                0
            );
            assert_eq!(
                std::fs::metadata(root.join("req_test.jsonl"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o077,
                0
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn debug_capture_prunes_expired_files() {
        let root = std::env::temp_dir().join(format!(
            "hoshikage-debug-capture-prune-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let expired = root.join("expired.jsonl");
        std::fs::write(&expired, "old").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));

        let capture = DebugCapture::with_limits(
            root.clone(),
            std::time::Duration::from_millis(1),
            100 * 1024 * 1024,
            16 * 1024 * 1024,
        )
        .unwrap();
        capture.prune().await.unwrap();

        assert!(!expired.exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}
