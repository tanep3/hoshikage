use crate::error::Result;
use crate::model::ModelConfig;
use fs2::FileExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize)]
struct AddModelRequest {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub stop: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct AddModelResponse {
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

async fn add_via_api(port: u16, name: String, config: ModelConfig) -> Result<()> {
    let url = format!("http://127.0.0.1:{}/admin/models", port);
    let client = Client::new();

    let req = AddModelRequest {
        name: name.clone(),
        path: PathBuf::from(&config.path)
            .join(&config.model)
            .to_string_lossy()
            .to_string(),
        stop: config.stop,
    };

    let mut last_error = None;
    for attempt in 1..=3 {
        match client.post(&url).json(&req).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let resp: AddModelResponse = response.json().await?;
                    if resp.success {
                        println!("{}", resp.message);
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
        "Failed to add model via API: {}",
        last_error.unwrap_or("Unknown error".to_string())
    )))
}

fn add_directly(name: String, config: ModelConfig) -> Result<()> {
    let config_dir = dirs::config_dir().ok_or_else(|| {
        crate::error::HoshikageError::ConfigError("Config directory not found".to_string())
    })?;

    let hoshikage_dir = config_dir.join("hoshikage");
    std::fs::create_dir_all(&hoshikage_dir)?;

    let model_map_path = hoshikage_dir.join("model_map.json");

    let mut models: std::collections::HashMap<String, ModelConfig> = if model_map_path.exists() {
        let content = std::fs::read_to_string(&model_map_path)?;
        serde_json::from_str(&content)?
    } else {
        std::collections::HashMap::new()
    };

    use std::fs::OpenOptions;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&model_map_path)?;

    file.try_lock_exclusive()
        .map_err(|e| crate::error::HoshikageError::Other(format!("Failed to lock file: {}", e)))?;

    models.insert(name.clone(), config);

    let content = serde_json::to_string_pretty(&models)?;
    std::fs::write(&model_map_path, &content)?;

    println!("Added model: {}", name);
    Ok(())
}

pub async fn add_model(
    path: String,
    label: String,
    stop_words: Vec<String>,
    port: u16,
) -> Result<()> {
    let file_path = PathBuf::from(&path);

    if !file_path.exists() {
        return Err(crate::error::HoshikageError::Other(format!(
            "Model file not found: {}",
            path
        )));
    }

    let file_name = file_path
        .file_name()
        .and_then(|f| f.to_str())
        .ok_or_else(|| crate::error::HoshikageError::Other("Invalid model path".to_string()))?
        .to_string();

    let parent_dir = file_path
        .parent()
        .and_then(|p| p.to_str())
        .unwrap_or(".")
        .to_string();

    let config = ModelConfig {
        path: parent_dir,
        model: file_name,
        stop: stop_words,
    };

    if check_server_running(port).await {
        add_via_api(port, label, config).await
    } else {
        add_directly(label, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_serialization() {
        let config = ModelConfig {
            path: "/models".to_string(),
            model: "test.gguf".to_string(),
            stop: vec!["</s>".to_string()],
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("test.gguf"));
    }
}
