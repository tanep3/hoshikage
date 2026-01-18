use crate::config::Config;
use crate::error::Result;
use crate::inference::llama_wrapper::InferenceParams;
use crate::inference::LlamaWrapper;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub path: String,
    pub model: String,
    #[serde(default)]
    pub stop: Vec<String>,
}

struct InferenceState {
    wrapper: Option<LlamaWrapper>,
    current_model: Option<String>,
    ramdisk_file: Option<PathBuf>,
    last_access: Instant,
}

pub struct ModelManager {
    models: Arc<RwLock<HashMap<String, ModelConfig>>>,
    config: Config,
    inference: Arc<Mutex<InferenceState>>,
    semaphore: Arc<Semaphore>,
}

impl ModelManager {
    pub fn new(config: Config) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            config,
            inference: Arc::new(Mutex::new(InferenceState {
                wrapper: None,
                current_model: None,
                ramdisk_file: None,
                last_access: Instant::now(),
            })),
            semaphore: Arc::new(Semaphore::new(1)),
        }
    }

    pub async fn load_models(&self) -> Result<()> {
        let model_map_path = self.config.model_map_path()?;

        if model_map_path.exists() {
            let content = std::fs::read_to_string(&model_map_path)?;
            let models: HashMap<String, ModelConfig> = serde_json::from_str(&content)?;
            let mut models_guard = self.models.write().await;
            *models_guard = models;

            tracing::info!(
                "Loaded {} models from {}",
                models_guard.len(),
                model_map_path.display()
            );
        } else {
            if let Some(models) = self.scan_model_dir()? {
                let mut models_guard = self.models.write().await;
                *models_guard = models;
                drop(models_guard);
                self.save_models().await?;
                tracing::info!(
                    "Model map file not found. Scanned model directory and saved to {}",
                    model_map_path.display()
                );
            } else {
                tracing::warn!("Model map file not found: {}", model_map_path.display());
            }
        }

        Ok(())
    }

    pub async fn save_models(&self) -> Result<()> {
        let model_map_path = self.config.model_map_path()?;

        let content = serde_json::to_string_pretty(&*self.models.read().await)?;

        if let Some(parent) = model_map_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(&model_map_path, &content)?;

        tracing::info!(
            "Saved {} models to {}",
            content.lines().count(),
            model_map_path.display()
        );

        Ok(())
    }

    pub async fn get_model(&self, name: &str) -> Result<ModelConfig> {
        let models = self.models.read().await;

        models
            .get(name)
            .cloned()
            .ok_or_else(|| crate::error::HoshikageError::ModelNotFound(name.to_string()))
    }

    pub async fn add_model(&self, name: String, config: ModelConfig) -> Result<()> {
        let mut models = self.models.write().await;

        models.insert(name.clone(), config);
        drop(models);
        self.save_models().await?;

        tracing::info!("Added model: {}", name);

        Ok(())
    }

    pub async fn remove_model(&self, name: &str) -> Result<()> {
        let mut models = self.models.write().await;

        if models.remove(name).is_some() {
            drop(models);
            self.save_models().await?;
            tracing::info!("Removed model: {}", name);
        }

        Ok(())
    }

    pub async fn list_models(&self) -> Vec<String> {
        let models = self.models.read().await;
        models.keys().cloned().collect()
    }

    pub fn default_temperature(&self) -> f32 {
        self.config.default_temperature
    }

    pub fn default_top_p(&self) -> f32 {
        self.config.default_top_p
    }

    pub fn start_idle_monitor(self: Arc<Self>) {
        let idle_timeout = self.config.idle_timeout;
        let great_timeout = self.config.great_timeout;
        if idle_timeout == 0 && great_timeout == 0 {
            return;
        }

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                let mut state = match self.inference.lock() {
                    Ok(state) => state,
                    Err(e) => {
                        tracing::error!("Inference state lock poisoned: {}", e);
                        continue;
                    }
                };

                let idle_secs = state.last_access.elapsed().as_secs();

                if idle_timeout > 0 && idle_secs >= idle_timeout {
                    if let Some(wrapper) = state.wrapper.as_mut() {
                        wrapper.unload();
                        state.current_model = None;
                        tracing::info!("Unloaded model due to idle timeout");
                    }
                }

                if great_timeout > 0 && idle_secs >= great_timeout * 60 {
                    if let Some(path) = state.ramdisk_file.take() {
                        if let Err(e) = std::fs::remove_file(&path) {
                            tracing::warn!("Failed to remove ramdisk file {}: {}", path.display(), e);
                        } else {
                            tracing::info!("Removed ramdisk file {}", path.display());
                        }
                    }
                }
            }
        });
    }

    pub async fn generate(
        &self,
        model_name: &str,
        prompt: &str,
        params: InferenceParams,
    ) -> Result<(String, u32, u32)> {
        let _permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| crate::error::HoshikageError::Other(format!("Semaphore error: {}", e)))?;

        let model_config = self.get_model(model_name).await?;
        let mut state = self
            .inference
            .lock()
            .map_err(|e| crate::error::HoshikageError::Other(format!("Lock error: {}", e)))?;

        self.load_model_if_needed(&mut state, model_name, &model_config)?;

        {
            let wrapper = state.wrapper.as_mut().ok_or_else(|| {
                crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
            })?;
            wrapper.prepare_for_inference(&self.config)?;
        }

        let wrapper = state.wrapper.as_ref().ok_or_else(|| {
            crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
        })?;

        let prompt_tokens = wrapper.count_tokens(prompt)? as u32;
        let output = wrapper.generate(prompt, &params)?;
        let completion_tokens = wrapper.count_tokens(&output)? as u32;
        state.last_access = Instant::now();

        Ok((output, prompt_tokens, completion_tokens))
    }

    pub async fn build_prompt(
        &self,
        model_name: &str,
        messages: &[crate::api::ChatMessage],
    ) -> Result<String> {
        let model_config = self.get_model(model_name).await?;
        let mut state = self
            .inference
            .lock()
            .map_err(|e| crate::error::HoshikageError::Other(format!("Lock error: {}", e)))?;

        self.load_model_if_needed(&mut state, model_name, &model_config)?;

        let prompt = {
            let wrapper = state.wrapper.as_mut().ok_or_else(|| {
                crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
            })?;
            wrapper.prepare_for_inference(&self.config)?;
            wrapper.format_chat_prompt(messages)?
        };

        state.last_access = Instant::now();
        Ok(prompt)
    }

    pub async fn generate_stream(
        &self,
        model_name: String,
        prompt: String,
        params: InferenceParams,
        sender: tokio::sync::mpsc::UnboundedSender<Result<String>>,
    ) -> Result<()> {
        let _permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| crate::error::HoshikageError::Other(format!("Semaphore error: {}", e)))?;

        let model_config = self.get_model(&model_name).await?;
        let mut state = self
            .inference
            .lock()
            .map_err(|e| crate::error::HoshikageError::Other(format!("Lock error: {}", e)))?;

        self.load_model_if_needed(&mut state, &model_name, &model_config)?;

        {
            let wrapper = state.wrapper.as_mut().ok_or_else(|| {
                crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
            })?;
            wrapper.prepare_for_inference(&self.config)?;
        }

        let wrapper = state.wrapper.as_ref().ok_or_else(|| {
            crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
        })?;

        let result = wrapper.generate_with_callback(&prompt, &params, |chunk| {
            sender
                .send(Ok(chunk))
                .map_err(|e| crate::error::HoshikageError::Other(e.to_string()))
        });

        state.last_access = Instant::now();

        if let Err(e) = result {
            let _ = sender.send(Err(e));
        }

        Ok(())
    }

    fn load_model_if_needed(
        &self,
        state: &mut InferenceState,
        model_name: &str,
        model_config: &ModelConfig,
    ) -> Result<()> {
        let needs_reload = state.wrapper.is_none()
            || state.current_model.as_deref() != Some(model_name)
            || !state.wrapper.as_ref().map(|w| w.is_loaded()).unwrap_or(false);

        if !needs_reload {
            return Ok(());
        }

        if let Some(wrapper) = state.wrapper.as_mut() {
            wrapper.unload();
        }

        let lib_path = self.config.resolve_lib_path()?;
        let mut wrapper = LlamaWrapper::new(lib_path)?;
        let (model_path, ramdisk_file) = self.resolve_model_path(model_config)?;
        let model_path_str = model_path.to_str().ok_or_else(|| {
            crate::error::HoshikageError::ModelLoadFailed("Invalid model path".to_string())
        })?;
        wrapper.load_model(model_path_str, &self.config)?;

        state.wrapper = Some(wrapper);
        state.current_model = Some(model_name.to_string());
        state.ramdisk_file = ramdisk_file;
        state.last_access = Instant::now();

        Ok(())
    }

    fn resolve_model_path(&self, config: &ModelConfig) -> Result<(PathBuf, Option<PathBuf>)> {
        let model_path = PathBuf::from(&config.path).join(&config.model);

        if cfg!(target_os = "linux") {
            if let Some(ramdisk_path) = &self.config.ramdisk_path {
                let ramdisk_dir = PathBuf::from(ramdisk_path);
                std::fs::create_dir_all(&ramdisk_dir)?;

                if let Ok(entries) = std::fs::read_dir(&ramdisk_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                            let _ = std::fs::remove_file(&path);
                        }
                    }
                }

                let dest_path = ramdisk_dir.join(&config.model);
                std::fs::copy(&model_path, &dest_path)?;
                return Ok((dest_path.clone(), Some(dest_path)));
            }
        }

        Ok((model_path, None))
    }

    fn scan_model_dir(&self) -> Result<Option<HashMap<String, ModelConfig>>> {
        let base_dir = self
            .config
            .model_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("models"));

        if !base_dir.exists() {
            return Ok(None);
        }

        let mut models = HashMap::new();
        for entry in std::fs::read_dir(&base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
                continue;
            }
            let file_name = match path.file_name().and_then(|f| f.to_str()) {
                Some(name) => name.to_string(),
                None => continue,
            };
            let label = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&file_name)
                .to_string();

            models.insert(
                label,
                ModelConfig {
                    path: base_dir.to_string_lossy().to_string(),
                    model: file_name,
                    stop: Vec::new(),
                },
            );
        }

        if models.is_empty() {
            Ok(None)
        } else {
            Ok(Some(models))
        }
    }
}
