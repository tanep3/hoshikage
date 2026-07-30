use crate::error::Result;
use crate::i18n::Language;
use crate::{
    commands::doctor::check_candidate_model,
    config::Config,
    model::{
        FallbackMode, ModelConfig, ModelRegistry, SpeculationConfig, SpeculationMode,
        ThinkingConfig, ThinkingMode, ToolCallingConfig,
    },
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::path::PathBuf;

#[derive(Debug, Serialize)]
struct AddModelRequest {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drafter: Option<String>,
    #[serde(default)]
    pub speculation: SpeculationConfig,
    #[serde(default)]
    pub thinking: ThinkingConfig,
    #[serde(
        default,
        skip_serializing_if = "ToolCallingConfig::is_disabled_default"
    )]
    pub tool_calling: ToolCallingConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_ctx: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_gpu_layers: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct AddModelResponse {
    pub success: bool,
    pub message: String,
}

pub struct AddModelOptions {
    pub path: String,
    pub label: String,
    pub stop_words: Vec<String>,
    pub mmproj: Option<String>,
    pub mtp: bool,
    pub mtp_drafter: Option<String>,
    pub draft_model: Option<String>,
    pub spec_draft_n_max: Option<NonZeroU32>,
    pub thinking_off: bool,
    pub n_ctx: Option<u32>,
    pub n_gpu_layers: Option<i32>,
    pub check: bool,
    pub port: u16,
    pub language: Language,
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

async fn add_via_api(
    port: u16,
    name: String,
    config: ModelConfig,
    language: Language,
) -> Result<()> {
    let url = format!("http://127.0.0.1:{}/admin/models", port);
    let client = Client::new();

    let req = AddModelRequest {
        name: name.clone(),
        path: PathBuf::from(&config.path)
            .join(&config.model)
            .to_string_lossy()
            .to_string(),
        stop: config.stop,
        mmproj: config.mmproj,
        drafter: config.drafter,
        speculation: config.speculation,
        thinking: config.thinking,
        tool_calling: config.tool_calling,
        n_ctx: config.n_ctx,
        n_gpu_layers: config.n_gpu_layers,
    };

    let mut last_error = None;
    for attempt in 1..=3 {
        match client.post(&url).json(&req).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let resp: AddModelResponse = response.json().await?;
                    if resp.success {
                        println!(
                            "{}",
                            match language {
                                Language::En => format!("Added model: {name}"),
                                Language::Ja => format!("モデルを追加しました: {name}"),
                            }
                        );
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

async fn add_directly_with_config(
    name: String,
    model_config: ModelConfig,
    language: Language,
    runtime_config: Config,
) -> Result<()> {
    let registry = ModelRegistry::new(runtime_config);
    registry.load().await?;
    registry.insert(name.clone(), model_config).await?;

    match language {
        Language::En => println!("Added model: {name}"),
        Language::Ja => println!("モデルを追加しました: {name}"),
    }
    Ok(())
}

async fn add_directly(name: String, config: ModelConfig, language: Language) -> Result<()> {
    add_directly_with_config(name, config, language, Config::load()?).await
}

pub async fn add_model(options: AddModelOptions) -> Result<()> {
    let AddModelOptions {
        path,
        label,
        stop_words,
        mmproj,
        mtp,
        mtp_drafter,
        draft_model,
        spec_draft_n_max,
        thinking_off,
        n_ctx,
        n_gpu_layers,
        check,
        port,
        language,
    } = options;
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

    let (speculation, drafter) =
        build_speculation_config(mtp, mtp_drafter, draft_model, spec_draft_n_max)?;

    let config = ModelConfig {
        mmproj,
        drafter,
        speculation,
        thinking: ThinkingConfig {
            mode: if thinking_off {
                ThinkingMode::Off
            } else {
                ThinkingMode::Auto
            },
        },
        n_ctx,
        n_gpu_layers,
        ..ModelConfig::new_legacy(parent_dir, file_name, stop_words)
    };

    if check && !check_candidate_model(&label, &config, language)? {
        return Err(crate::error::HoshikageError::ConfigError(
            "Model bundle check failed; registration was not saved".to_string(),
        ));
    }

    if check_server_running(port).await {
        add_via_api(port, label, config, language).await
    } else {
        add_directly(label, config, language).await
    }
}

fn build_speculation_config(
    mtp: bool,
    mtp_drafter: Option<String>,
    draft_model: Option<String>,
    draft_n_max: Option<NonZeroU32>,
) -> Result<(SpeculationConfig, Option<String>)> {
    let mut modes = Vec::new();
    if mtp || mtp_drafter.is_some() {
        modes.push(SpeculationMode::Mtp);
    }
    if draft_model.is_some() {
        modes.push(SpeculationMode::DraftModel);
    }
    if draft_n_max.is_some() && modes.is_empty() {
        return Err(crate::error::HoshikageError::ConfigError(
            "--spec-draft-n-max requires --mtp, --mtp-drafter, or --draft-model".to_string(),
        ));
    }

    let drafter = match (mtp_drafter, draft_model) {
        (Some(mtp), Some(draft)) if mtp == draft => Some(mtp),
        (Some(_mtp), Some(_draft)) => {
            return Err(crate::error::HoshikageError::ConfigError(
                "current model bundle format supports one speculation auxiliary model path; use the same path for --mtp-drafter and --draft-model or register one mode at a time".to_string(),
            ))
        }
        (Some(mtp), None) => Some(mtp),
        (None, Some(draft)) => Some(draft),
        (None, None) => None,
    };
    Ok((
        SpeculationConfig {
            modes,
            draft_n_max,
            fallback: FallbackMode::Warn,
        },
        drafter,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;
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

    #[test]
    fn test_config_serialization() {
        let config = ModelConfig {
            ..ModelConfig::new_legacy(
                "/models".to_string(),
                "test.gguf".to_string(),
                vec!["</s>".to_string()],
            )
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("test.gguf"));
        assert!(json.contains("base_path"));
    }

    #[test]
    fn built_in_mtp_does_not_require_a_drafter_file() {
        let (speculation, drafter) =
            build_speculation_config(true, None, None, NonZeroU32::new(6)).unwrap();

        assert!(speculation.has_mode(SpeculationMode::Mtp));
        assert_eq!(speculation.draft_n_max.map(|value| value.get()), Some(6));
        assert_eq!(drafter, None);
    }

    #[test]
    fn draft_n_max_requires_an_enabled_speculation_mode() {
        let error = build_speculation_config(false, None, None, NonZeroU32::new(6)).unwrap_err();

        assert!(error.to_string().contains("--spec-draft-n-max requires"));
    }

    #[tokio::test]
    async fn direct_add_preserves_existing_models_and_valid_json() {
        let directory =
            std::env::temp_dir().join(format!("hoshikage-add-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("model_map.json");
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&std::collections::HashMap::from([(
                "existing".to_string(),
                test_model("existing"),
            )]))
            .unwrap(),
        )
        .unwrap();

        add_directly_with_config(
            "new".to_string(),
            test_model("new"),
            Language::En,
            test_config(path.clone()),
        )
        .await
        .unwrap();

        let persisted: std::collections::HashMap<String, ModelConfig> =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert!(persisted.contains_key("existing"));
        assert!(persisted.contains_key("new"));

        std::fs::remove_dir_all(directory).unwrap();
    }
}
