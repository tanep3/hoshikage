use crate::error::Result;
use fs2::FileExt;
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

async fn remove_via_api(port: u16, name: String) -> Result<()> {
    let url = format!("http://127.0.0.1:{}/admin/models/{}", port, name);
    let client = Client::new();

    let mut last_error = None;
    for attempt in 1..=3 {
        match client.delete(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let resp: RemoveModelResponse = response.json().await?;
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
        "Failed to remove model via API: {}",
        last_error.unwrap_or("Unknown error".to_string())
    )))
}

fn remove_directly(name: String) -> Result<()> {
    let config_dir = dirs::config_dir().ok_or_else(|| {
        crate::error::HoshikageError::ConfigError("Config directory not found".to_string())
    })?;

    let hoshikage_dir = config_dir.join("hoshikage");
    let model_map_path = hoshikage_dir.join("model_map.json");

    if !model_map_path.exists() {
        return Err(crate::error::HoshikageError::Other(
            "Model map file not found".to_string(),
        ));
    }

    use std::fs::OpenOptions;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&model_map_path)?;

    file.try_lock_exclusive()
        .map_err(|e| crate::error::HoshikageError::Other(format!("Failed to lock file: {}", e)))?;

    let content = std::fs::read_to_string(&model_map_path)?;
    let mut models: std::collections::HashMap<String, crate::model::ModelConfig> =
        serde_json::from_str(&content)?;

    if models.remove(&name).is_none() {
        return Err(crate::error::HoshikageError::ModelNotFound(name));
    }

    let new_content = serde_json::to_string_pretty(&models)?;
    std::fs::write(&model_map_path, &new_content)?;

    println!("Removed model: {}", name);
    Ok(())
}

pub async fn remove_model(label: String, port: u16) -> Result<()> {
    if check_server_running(port).await {
        remove_via_api(port, label).await
    } else {
        remove_directly(label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_deserialization() {
        let json = r#"{"success": true, "message": "Model removed"}"#;
        let resp: RemoveModelResponse = serde_json::from_str(json).unwrap();
        assert!(resp.success);
        assert_eq!(resp.message, "Model removed");
    }
}
