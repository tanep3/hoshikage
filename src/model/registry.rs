use super::ModelConfig;
use crate::config::Config;
use crate::error::{HoshikageError, Result};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct ModelRegistry {
    models: Arc<RwLock<HashMap<String, ModelConfig>>>,
    config: Config,
}

impl ModelRegistry {
    pub fn new(config: Config) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    pub async fn load(&self) -> Result<bool> {
        let path = self.config.model_map_path()?;
        if !path.exists() {
            return Ok(false);
        }
        let content = std::fs::read_to_string(&path)?;
        let models: HashMap<String, ModelConfig> = serde_json::from_str(&content)?;
        for (name, config) in &models {
            config
                .tool_calling
                .validate()
                .map_err(|error| HoshikageError::ConfigError(format!("model {name}: {error}")))?;
        }
        *self.models.write().await = models;
        Ok(true)
    }

    pub async fn replace(&self, models: HashMap<String, ModelConfig>) {
        *self.models.write().await = models;
    }

    pub async fn save(&self) -> Result<()> {
        let models = self.models.read().await;
        self.save_snapshot(&models)
    }

    fn save_snapshot(&self, models: &HashMap<String, ModelConfig>) -> Result<()> {
        let path = self.config.model_map_path()?;
        let content = serde_json::to_vec_pretty(models)?;
        let parent = path.parent().ok_or_else(|| {
            HoshikageError::ConfigError("model map path has no parent directory".to_string())
        })?;
        std::fs::create_dir_all(parent)?;
        let temporary = parent.join(format!(
            ".{}.{}.tmp",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model_map.json"),
            uuid::Uuid::new_v4().simple()
        ));

        let write_result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)?;
            file.write_all(&content)?;
            file.sync_all()?;
            std::fs::rename(&temporary, &path)?;
            if let Ok(directory) = OpenOptions::new().read(true).open(parent) {
                let _ = directory.sync_all();
            }
            Ok(())
        })();
        if write_result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        write_result
    }

    pub async fn get(&self, name: &str) -> Result<ModelConfig> {
        self.models
            .read()
            .await
            .get(name)
            .cloned()
            .ok_or_else(|| HoshikageError::ModelNotFound(name.to_string()))
    }

    pub async fn insert(&self, name: String, config: ModelConfig) -> Result<()> {
        config
            .tool_calling
            .validate()
            .map_err(HoshikageError::ConfigError)?;
        let mut models = self.models.write().await;
        let previous = models.insert(name.clone(), config);
        if let Err(error) = self.save_snapshot(&models) {
            match previous {
                Some(previous) => {
                    models.insert(name, previous);
                }
                None => {
                    models.remove(&name);
                }
            }
            return Err(error);
        }
        Ok(())
    }

    pub async fn remove(&self, name: &str) -> Result<bool> {
        let mut models = self.models.write().await;
        let Some(removed) = models.remove(name) else {
            return Ok(false);
        };
        if let Err(error) = self.save_snapshot(&models) {
            models.insert(name.to_string(), removed);
            return Err(error);
        }
        Ok(true)
    }

    pub async fn names(&self) -> Vec<String> {
        self.models.read().await.keys().cloned().collect()
    }

    pub async fn snapshot(&self) -> HashMap<String, ModelConfig> {
        self.models.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{SpeculationConfig, ThinkingConfig};
    use std::path::PathBuf;

    fn config(path: PathBuf) -> Config {
        Config {
            model_map_file: Some(path),
            ..Config::default()
        }
    }

    fn model() -> ModelConfig {
        ModelConfig {
            path: "/models".to_string(),
            model: "model.gguf".to_string(),
            stop: Vec::new(),
            mmproj: None,
            drafter: None,
            speculation: SpeculationConfig::default(),
            thinking: ThinkingConfig::default(),
            llama_server: crate::model::LlamaServerModelConfig::default(),
            generation: crate::model::GenerationMode::Autoregressive,
            tool_calling: crate::model::ToolCallingConfig::default(),
            n_ctx: Some(16384),
            n_gpu_layers: Some(0),
        }
    }

    #[tokio::test]
    async fn registry_persists_and_reloads_model_snapshot() {
        let directory =
            std::env::temp_dir().join(format!("hoshikage-registry-{}", uuid::Uuid::new_v4()));
        let path = directory.join("model_map.json");
        let registry = ModelRegistry::new(config(path.clone()));

        registry
            .insert("gemma4".to_string(), model())
            .await
            .unwrap();
        assert!(path.exists());

        let reloaded = ModelRegistry::new(config(path.clone()));
        assert!(reloaded.load().await.unwrap());
        assert_eq!(reloaded.get("gemma4").await.unwrap().n_ctx, Some(16384));

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn invalid_json_never_replaces_existing_memory_snapshot() {
        let directory =
            std::env::temp_dir().join(format!("hoshikage-registry-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("model_map.json");
        let registry = ModelRegistry::new(config(path.clone()));
        registry
            .replace(HashMap::from([("gemma4".to_string(), model())]))
            .await;
        std::fs::write(&path, "{invalid").unwrap();

        assert!(registry.load().await.is_err());
        assert!(registry.get("gemma4").await.is_ok());

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn failed_save_rolls_back_memory_mutation() {
        let directory =
            std::env::temp_dir().join(format!("hoshikage-registry-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let not_a_directory = directory.join("not-a-directory");
        std::fs::write(&not_a_directory, "file").unwrap();
        let registry = ModelRegistry::new(config(not_a_directory.join("model_map.json")));
        registry
            .replace(HashMap::from([("existing".to_string(), model())]))
            .await;

        assert!(registry.insert("new".to_string(), model()).await.is_err());
        assert!(registry.get("existing").await.is_ok());
        assert!(registry.get("new").await.is_err());

        std::fs::remove_dir_all(directory).unwrap();
    }
}
