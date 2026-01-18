use crate::error::Result;
use reqwest::Client;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ModelData {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelData>,
}

async fn list_via_api(port: u16) -> Result<()> {
    let url = format!("http://127.0.0.1:{}/v1/models", port);
    let client = Client::new();

    let mut last_error = None;
    for attempt in 1..=3 {
        match client.get(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let resp: ModelListResponse = response.json().await?;

                    println!("Registered models:");
                    println!("------------------");

                    if resp.data.is_empty() {
                        println!("No models registered");
                    } else {
                        for model in &resp.data {
                            println!("  - {}", model.id);
                        }
                    }
                    println!();
                    println!("Total: {} model(s)", resp.data.len());
                    return Ok(());
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
        "Failed to list models via API: {}",
        last_error.unwrap_or("Unknown error".to_string())
    )))
}

fn list_directly() -> Result<()> {
    let config_dir = dirs::config_dir().ok_or_else(|| {
        crate::error::HoshikageError::ConfigError("Config directory not found".to_string())
    })?;

    let hoshikage_dir = config_dir.join("hoshikage");
    let model_map_path = hoshikage_dir.join("model_map.json");

    if !model_map_path.exists() {
        println!("No models registered");
        return Ok(());
    }

    let content = std::fs::read_to_string(&model_map_path)?;
    let models: std::collections::HashMap<String, crate::model::ModelConfig> =
        serde_json::from_str(&content)?;

    println!("Registered models:");
    println!("------------------");

    if models.is_empty() {
        println!("No models registered");
    } else {
        for (name, config) in &models {
            println!("  - {} ({})", name, config.model);
        }
    }

    println!();
    println!("Total: {} model(s)", models.len());

    Ok(())
}

pub async fn list_models(port: u16) -> Result<()> {
    if check_server_running(port).await {
        list_via_api(port).await
    } else {
        list_directly()
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "object": "list",
            "data": [
                {"id": "model1", "object": "model", "created": 123, "owned_by": "tane"}
            ]
        }"#;
        let resp: ModelListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].id, "model1");
    }
}
