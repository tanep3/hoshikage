use crate::error::Result;
use crate::i18n::Language;
use crate::{config::Config, model::ModelRegistry};
use reqwest::Client;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct RemoveModelResponse {
    pub success: bool,
    pub message: String,
}

async fn check_server_running(port: u16) -> bool {
    let url = format!("http://127.0.0.1:{}/v1/status", port);
    Client::new()
        .get(&url)
        .send()
        .await
        .map(|res| res.status().is_success())
        .unwrap_or(false)
}

async fn remove_via_api(port: u16, name: String, language: Language) -> Result<()> {
    let url = format!("http://127.0.0.1:{}/admin/models/{}", port, name);
    let client = Client::new();

    let mut last_error = None;
    for attempt in 1..=3 {
        match client.delete(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let resp: RemoveModelResponse = response.json().await?;
                    if resp.success {
                        match language {
                            Language::En => println!("Removed model: {name}"),
                            Language::Ja => println!("モデルを削除しました: {name}"),
                        }
                        return Ok(());
                    } else {
                        return Err(crate::error::HoshikageError::Other(resp.message));
                    }
                }
                last_error = Some(format!("HTTP {}", response.status()));
            }
            Err(e) => {
                last_error = Some(e.to_string());
            }
        }

        if attempt < 3 {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    }

    Err(crate::error::HoshikageError::Other(format!(
        "Failed to remove model via API: {}",
        last_error.unwrap_or("Unknown error".to_string())
    )))
}

async fn remove_directly_with_config(
    name: String,
    language: Language,
    runtime_config: Config,
) -> Result<()> {
    let model_map_path = runtime_config.model_map_path()?;

    if !model_map_path.exists() {
        return Err(crate::error::HoshikageError::Other(
            "Model map file not found".to_string(),
        ));
    }

    let registry = ModelRegistry::new(runtime_config);
    if !registry.load().await? {
        return Err(crate::error::HoshikageError::Other(
            "Model map file not found".to_string(),
        ));
    }
    if !registry.remove(&name).await? {
        return Err(crate::error::HoshikageError::ModelNotFound(name));
    }

    match language {
        Language::En => println!("Removed model: {name}"),
        Language::Ja => println!("モデルを削除しました: {name}"),
    }
    Ok(())
}

async fn remove_directly(name: String, language: Language) -> Result<()> {
    remove_directly_with_config(name, language, Config::load()?).await
}

pub async fn remove_model(label: String, port: u16, language: Language) -> Result<()> {
    if check_server_running(port).await {
        remove_via_api(port, label, language).await
    } else {
        remove_directly(label, language).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::model::ModelConfig;
    use std::path::PathBuf;

    fn test_model(name: &str) -> ModelConfig {
        ModelConfig {
            ..ModelConfig::new_legacy("/models".to_string(), format!("{name}.gguf"), Vec::new())
        }
    }

    fn test_config(path: PathBuf) -> Config {
        Config {
            model_map_file: Some(path),
            ..Config::default()
        }
    }

    #[tokio::test]
    async fn direct_remove_preserves_remaining_models_and_valid_json() {
        let directory = std::env::temp_dir().join(format!("hoshikage-rm-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("model_map.json");
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&std::collections::HashMap::from([
                ("keep".to_string(), test_model("keep")),
                ("remove".to_string(), test_model("remove")),
            ]))
            .unwrap(),
        )
        .unwrap();

        remove_directly_with_config(
            "remove".to_string(),
            Language::En,
            test_config(path.clone()),
        )
        .await
        .unwrap();

        let persisted: std::collections::HashMap<String, ModelConfig> =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert!(persisted.contains_key("keep"));
        assert!(!persisted.contains_key("remove"));

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{"success": true, "message": "Model removed"}"#;
        let resp: RemoveModelResponse = serde_json::from_str(json).unwrap();
        assert!(resp.success);
        assert_eq!(resp.message, "Model removed");
    }
}
