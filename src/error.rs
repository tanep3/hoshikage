use std::ffi::NulError;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, HoshikageError>;

#[derive(Debug, Error)]
pub enum HoshikageError {
    #[error("モデルロードに失敗: {0}")]
    ModelLoadFailed(String),

    #[error("推論エラー: {0}")]
    InferenceError(String),

    #[error("モデルが見つかりません: {0}")]
    ModelNotFound(String),

    #[error("設定エラー: {0}")]
    ConfigError(String),

    #[error("ライブラリロードエラー: {0}")]
    LibraryLoadError(String),

    #[error("IOエラー: {0}")]
    IoError(#[from] std::io::Error),

    #[error("シリアライズエラー: {0}")]
    SerdeError(#[from] serde_json::Error),

    #[error("HTTPエラー: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("文字列エラー: {0}")]
    StringError(#[from] NulError),

    #[error("その他のエラー: {0}")]
    Other(String),
}

impl From<libloading::Error> for HoshikageError {
    fn from(err: libloading::Error) -> Self {
        HoshikageError::LibraryLoadError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages() {
        let err = HoshikageError::ModelLoadFailed("test".to_string());
        assert_eq!(err.to_string(), "モデルロードに失敗: test");

        let err = HoshikageError::InferenceError("test".to_string());
        assert_eq!(err.to_string(), "推論エラー: test");

        let err = HoshikageError::ModelNotFound("test".to_string());
        assert_eq!(err.to_string(), "モデルが見つかりません: test");

        let err = HoshikageError::ConfigError("test".to_string());
        assert_eq!(err.to_string(), "設定エラー: test");

        let err = HoshikageError::LibraryLoadError("test".to_string());
        assert_eq!(err.to_string(), "ライブラリロードエラー: test");

        let err = HoshikageError::Other("custom".to_string());
        assert_eq!(err.to_string(), "その他のエラー: custom");
    }

    #[test]
    fn test_from_conversions() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let hoshikage_err: HoshikageError = io_err.into();
        assert!(matches!(hoshikage_err, HoshikageError::IoError(_)));

        let serde_err = serde_json::from_str::<serde_json::Value>("invalid").unwrap_err();
        let hoshikage_err: HoshikageError = serde_err.into();
        assert!(matches!(hoshikage_err, HoshikageError::SerdeError(_)));
    }

    #[test]
    fn test_result_type() {
        fn returns_ok() -> Result<String> {
            Ok("success".to_string())
        }

        fn returns_error() -> Result<String> {
            Err(HoshikageError::ModelLoadFailed("failed".to_string()))
        }

        assert!(returns_ok().is_ok());
        assert!(returns_error().is_err());
    }
}
