use crate::error::Result;
use crate::i18n::Language;
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

async fn list_via_api(port: u16, details: bool, language: Language) -> Result<()> {
    if details {
        println!(
            "{}",
            language.select(
                "Detailed model listing is available when the server is stopped.",
                "詳細なモデル一覧はサーバー停止中に利用できます。"
            )
        );
        println!(
            "{}",
            language.select(
                "The OpenAI-compatible /v1/models API intentionally returns model IDs only.",
                "OpenAI互換の/v1/models APIは意図的にモデルIDだけを返します。"
            )
        );
        println!();
    }

    let url = format!("http://127.0.0.1:{}/v1/models", port);
    let client = Client::new();

    let mut last_error = None;
    for attempt in 1..=3 {
        match client.get(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let resp: ModelListResponse = response.json().await?;

                    println!(
                        "{}",
                        language.select("Registered models:", "登録済みモデル:")
                    );
                    println!("------------------");

                    if resp.data.is_empty() {
                        println!(
                            "{}",
                            language.select("No models registered", "登録モデルはありません")
                        );
                    } else {
                        for model in &resp.data {
                            println!("  - {}", model.id);
                        }
                    }
                    println!();
                    match language {
                        Language::En => println!("Total: {} model(s)", resp.data.len()),
                        Language::Ja => println!("合計: {}モデル", resp.data.len()),
                    }
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

fn list_directly(details: bool, language: Language) -> Result<()> {
    let config_dir = dirs::config_dir().ok_or_else(|| {
        crate::error::HoshikageError::ConfigError("Config directory not found".to_string())
    })?;

    let hoshikage_dir = config_dir.join("hoshikage");
    let model_map_path = hoshikage_dir.join("model_map.json");

    if !model_map_path.exists() {
        println!(
            "{}",
            language.select("No models registered", "登録モデルはありません")
        );
        return Ok(());
    }

    let content = std::fs::read_to_string(&model_map_path)?;
    let models: std::collections::HashMap<String, crate::model::ModelConfig> =
        serde_json::from_str(&content)?;

    println!(
        "{}",
        language.select("Registered models:", "登録済みモデル:")
    );
    println!("------------------");

    if models.is_empty() {
        println!(
            "{}",
            language.select("No models registered", "登録モデルはありません")
        );
    } else {
        for (name, config) in &models {
            if details {
                print_model_details(name, config);
            } else {
                println!("  - {} ({})", name, config.model);
            }
        }
    }

    println!();
    match language {
        Language::En => println!("Total: {} model(s)", models.len()),
        Language::Ja => println!("合計: {}モデル", models.len()),
    }

    Ok(())
}

pub async fn list_models(port: u16, details: bool, language: Language) -> Result<()> {
    if check_server_running(port).await {
        list_via_api(port, details, language).await
    } else {
        list_directly(details, language)
    }
}

fn print_model_details(name: &str, config: &crate::model::ModelConfig) {
    println!("  - {}", name);
    println!("    main: {}", config.main_model_path().display());
    if let Some(mmproj) = &config.mmproj {
        println!("    mmproj: {}", mmproj);
    }
    if let Some(drafter) = &config.drafter {
        println!("    drafter: {}", drafter);
    }
    println!("    speculation: {:?}", config.speculation.modes);
    if let Some(draft_n_max) = config.speculation.draft_n_max {
        println!("    spec_draft_n_max: {}", draft_n_max);
    }
    println!("    thinking: {:?}", config.thinking.mode);
    if let Some(max_reasoning_tokens) = config.thinking.max_reasoning_tokens {
        println!("    max_reasoning_tokens: {}", max_reasoning_tokens);
    } else {
        println!("    max_reasoning_tokens: unlimited");
    }
    println!("    min_final_tokens: {}", config.thinking.min_final_tokens);
    println!("    generation: {:?}", config.generation);
    if let Some(cache_type_k) = config.llama_server.cache_type_k {
        println!("    cache_type_k: {}", cache_type_k.as_str());
    }
    if let Some(cache_type_v) = config.llama_server.cache_type_v {
        println!("    cache_type_v: {}", cache_type_v.as_str());
    }
    if let Some(n_ctx) = config.n_ctx {
        println!("    n_ctx: {}", n_ctx);
    }
    if let Some(n_gpu_layers) = config.n_gpu_layers {
        println!("    n_gpu_layers: {}", n_gpu_layers);
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
