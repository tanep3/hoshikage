use crate::config::{Config, RuntimeBackendKind};
use crate::error::Result;
use crate::inference::llama_wrapper::InferenceParams;
use crate::inference::{
    LlamaFfiBackend, LlamaLoadRequest, LlamaServerCommandSpec, LlamaServerLaunchConfig,
    LlamaServerProcess, LoadedRuntimeInfo, RuntimeBackend, ThinkingController,
    ThinkingStreamFilter,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default, rename = "base_path", alias = "path")]
    pub path: String,
    pub model: String,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_ctx: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_gpu_layers: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpeculationConfig {
    #[serde(
        default,
        alias = "mode",
        deserialize_with = "deserialize_speculation_modes"
    )]
    pub modes: Vec<SpeculationMode>,
    #[serde(default)]
    pub fallback: FallbackMode,
}

impl Default for SpeculationConfig {
    fn default() -> Self {
        Self {
            modes: Vec::new(),
            fallback: FallbackMode::Warn,
        }
    }
}

#[cfg(test)]
mod runtime_status_tests {
    use super::*;

    #[test]
    fn tracks_active_managed_requests_in_status() {
        let manager = ModelManager::new(Config::default());

        {
            let mut state = manager.inference.lock().unwrap();
            state.current_model = Some("test-model".to_string());
        }

        manager.begin_managed_request("test-model");
        manager.begin_managed_request("test-model");

        let status = manager.runtime_status();
        assert_eq!(status.active_requests, 2);

        manager.finish_managed_request("test-model");

        let status = manager.runtime_status();
        assert_eq!(status.active_requests, 1);
    }
}

impl SpeculationConfig {
    pub fn has_mode(&self, mode: SpeculationMode) -> bool {
        self.modes.iter().any(|configured| configured == &mode)
    }

    pub fn is_off(&self) -> bool {
        self.modes.is_empty() || self.has_mode(SpeculationMode::Off)
    }

    pub fn primary_mode(&self) -> SpeculationMode {
        self.modes
            .iter()
            .find(|mode| **mode != SpeculationMode::Off)
            .cloned()
            .unwrap_or(SpeculationMode::Off)
    }
}

fn deserialize_speculation_modes<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<SpeculationMode>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Modes {
        One(SpeculationMode),
        Many(Vec<SpeculationMode>),
    }

    let modes = Option::<Modes>::deserialize(deserializer)?;
    let modes = match modes {
        Some(Modes::One(SpeculationMode::Off)) | None => Vec::new(),
        Some(Modes::One(mode)) => vec![mode],
        Some(Modes::Many(modes)) => modes
            .into_iter()
            .filter(|mode| *mode != SpeculationMode::Off)
            .collect(),
    };
    Ok(modes)
}

fn resolve_bundle_path(base_path: &str, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        PathBuf::from(base_path).join(path)
    }
}

fn resolve_runtime_aux_path(
    base_path: &str,
    configured_path: Option<&str>,
    ramdisk_bundle: Option<&RamdiskBundleState>,
    ramdisk_file_name: &str,
) -> Option<PathBuf> {
    configured_path.map(|path| {
        ramdisk_bundle
            .map(|bundle| bundle.dir.join(ramdisk_file_name))
            .unwrap_or_else(|| resolve_bundle_path(base_path, path))
    })
}

fn bundle_total_bytes(files: &[BundleFile]) -> Result<u64> {
    let mut total = 0_u64;
    for file in files {
        total = total
            .checked_add(std::fs::metadata(&file.source)?.len())
            .ok_or_else(|| {
                crate::error::HoshikageError::Other(
                    "Bundle size overflow while checking RAM disk capacity".to_string(),
                )
            })?;
    }
    Ok(total)
}

fn ensure_ramdisk_capacity(path: &PathBuf, required_bytes: u64) -> Result<()> {
    let available = available_bytes(path)?;
    if available < required_bytes {
        return Err(crate::error::HoshikageError::Other(format!(
            "RAM disk capacity is insufficient: required {} bytes, available {} bytes",
            required_bytes, available
        )));
    }
    Ok(())
}

#[cfg(target_family = "unix")]
fn available_bytes(path: &PathBuf) -> Result<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(path.as_os_str().as_bytes())?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    let rc = unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error().into());
    }

    let stat = unsafe { stat.assume_init() };
    Ok(stat.f_bavail as u64 * stat.f_frsize as u64)
}

#[cfg(not(target_family = "unix"))]
fn available_bytes(_path: &PathBuf) -> Result<u64> {
    Ok(u64::MAX)
}

#[cfg(test)]
mod ramdisk_tests {
    use super::*;

    fn unique_temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("hoshikage-{}-{}", name, uuid::Uuid::new_v4()))
    }

    fn write_file(path: &PathBuf, content: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    #[test]
    fn resolves_bundle_relative_and_absolute_paths() {
        let absolute = PathBuf::from("/tmp/mmproj.gguf");

        assert_eq!(
            resolve_bundle_path("/models/base", "main.gguf"),
            PathBuf::from("/models/base/main.gguf")
        );
        assert_eq!(
            resolve_bundle_path("/models/base", absolute.to_str().unwrap()),
            absolute
        );
    }

    #[test]
    fn materializes_bundle_to_managed_current_dir() {
        let source_dir = unique_temp_dir("bundle-source");
        let ramdisk_dir = unique_temp_dir("ramdisk");
        write_file(&source_dir.join("main-source.gguf"), "main");
        write_file(&source_dir.join("mmproj-source.gguf"), "mmproj");
        write_file(&source_dir.join("drafter-source.gguf"), "drafter");

        let config = ModelConfig {
            mmproj: Some("mmproj-source.gguf".to_string()),
            drafter: Some(
                source_dir
                    .join("drafter-source.gguf")
                    .to_string_lossy()
                    .to_string(),
            ),
            ..ModelConfig::new_legacy(
                source_dir.to_string_lossy().to_string(),
                "main-source.gguf".to_string(),
                Vec::new(),
            )
        };
        let manager = ModelManager::new(Config::default());

        let bundle = manager
            .materialize_ramdisk_bundle("test-model", &config, ramdisk_dir.to_str().unwrap())
            .unwrap();

        assert_eq!(bundle.dir, ramdisk_dir.join("hoshikage/current"));
        assert_eq!(
            std::fs::read_to_string(bundle.dir.join("main.gguf")).unwrap(),
            "main"
        );
        assert_eq!(
            std::fs::read_to_string(bundle.dir.join("mmproj.gguf")).unwrap(),
            "mmproj"
        );
        assert_eq!(
            std::fs::read_to_string(bundle.dir.join("drafter.gguf")).unwrap(),
            "drafter"
        );
        assert!(bundle.dir.join("manifest.json").exists());
        let manifest = std::fs::read_to_string(bundle.dir.join("manifest.json")).unwrap();
        assert!(manifest.contains("hoshikage/current/main.gguf"));
        assert!(!manifest.contains("current.tmp"));
        assert!(!ramdisk_dir.join("hoshikage/current.tmp").exists());

        let _ = std::fs::remove_dir_all(source_dir);
        let _ = std::fs::remove_dir_all(ramdisk_dir);
    }

    #[test]
    fn materialize_clears_old_managed_cache_only() {
        let source_dir = unique_temp_dir("bundle-source");
        let ramdisk_dir = unique_temp_dir("ramdisk");
        write_file(&source_dir.join("main.gguf"), "new-main");
        write_file(&ramdisk_dir.join("hoshikage/current/old.gguf"), "old");
        write_file(&ramdisk_dir.join("user-file.gguf"), "keep");

        let config = ModelConfig::new_legacy(
            source_dir.to_string_lossy().to_string(),
            "main.gguf".to_string(),
            Vec::new(),
        );
        let manager = ModelManager::new(Config::default());

        let bundle = manager
            .materialize_ramdisk_bundle("test-model", &config, ramdisk_dir.to_str().unwrap())
            .unwrap();

        assert!(!bundle.dir.join("old.gguf").exists());
        assert_eq!(
            std::fs::read_to_string(bundle.dir.join("main.gguf")).unwrap(),
            "new-main"
        );
        assert_eq!(
            std::fs::read_to_string(ramdisk_dir.join("user-file.gguf")).unwrap(),
            "keep"
        );

        let _ = std::fs::remove_dir_all(source_dir);
        let _ = std::fs::remove_dir_all(ramdisk_dir);
    }

    #[test]
    fn load_request_uses_model_overrides_and_ramdisk_aux_paths() {
        let mut runtime_config = Config::default();
        runtime_config.n_ctx = 4096;
        runtime_config.n_gpu_layers = 10;
        let manager = ModelManager::new(runtime_config);
        let model_config = ModelConfig {
            mmproj: Some("mmproj-source.gguf".to_string()),
            drafter: Some("/models/drafter-source.gguf".to_string()),
            n_ctx: Some(8192),
            n_gpu_layers: Some(-1),
            ..ModelConfig::new_legacy(
                "/models".to_string(),
                "main-source.gguf".to_string(),
                Vec::new(),
            )
        };
        let ramdisk_bundle = RamdiskBundleState {
            model_name: "test-model".to_string(),
            dir: PathBuf::from("/dev/shm/hoshikage/current"),
            files: Vec::new(),
            total_bytes: 0,
        };

        let request = manager.build_load_request(
            &model_config,
            PathBuf::from("/dev/shm/hoshikage/current/main.gguf"),
            Some(&ramdisk_bundle),
        );

        assert_eq!(request.n_ctx, 8192);
        assert_eq!(request.n_gpu_layers, -1);
        assert_eq!(
            request.mmproj,
            Some(PathBuf::from("/dev/shm/hoshikage/current/mmproj.gguf"))
        );
        assert_eq!(
            request.draft_model,
            Some(PathBuf::from("/dev/shm/hoshikage/current/drafter.gguf"))
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpeculationMode {
    Off,
    Mtp,
    DraftModel,
}

impl Default for SpeculationMode {
    fn default() -> Self {
        Self::Off
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FallbackMode {
    Warn,
    Strict,
    Off,
}

impl Default for FallbackMode {
    fn default() -> Self {
        Self::Warn
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ThinkingConfig {
    #[serde(default)]
    pub mode: ThinkingMode,
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            mode: ThinkingMode::Auto,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingMode {
    Auto,
    Off,
}

impl Default for ThinkingMode {
    fn default() -> Self {
        Self::Auto
    }
}

impl ModelConfig {
    pub fn new_legacy(path: String, model: String, stop: Vec<String>) -> Self {
        Self {
            path,
            model,
            stop,
            mmproj: None,
            drafter: None,
            speculation: SpeculationConfig::default(),
            thinking: ThinkingConfig::default(),
            n_ctx: None,
            n_gpu_layers: None,
        }
    }

    pub fn main_model_path(&self) -> PathBuf {
        PathBuf::from(&self.path).join(&self.model)
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn legacy_model_config_deserializes_from_path() {
        let json = r#"{
            "path": "/models",
            "model": "main.gguf",
            "stop": ["</s>"]
        }"#;

        let config: ModelConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.path, "/models");
        assert_eq!(config.model, "main.gguf");
        assert_eq!(config.stop, vec!["</s>"]);
        assert!(config.speculation.is_off());
        assert_eq!(config.thinking.mode, ThinkingMode::Auto);
    }

    #[test]
    fn bundle_model_config_deserializes_from_base_path() {
        let json = r#"{
            "base_path": "/models/gemma4",
            "model": "main.gguf",
            "mmproj": "mmproj.gguf",
            "drafter": "mtp.gguf",
            "speculation": {
                "mode": "mtp",
                "fallback": "warn"
            },
            "thinking": {
                "mode": "off"
            },
            "n_ctx": 8192,
            "n_gpu_layers": -1
        }"#;

        let config: ModelConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.path, "/models/gemma4");
        assert_eq!(
            config.main_model_path(),
            PathBuf::from("/models/gemma4/main.gguf")
        );
        assert_eq!(config.mmproj.as_deref(), Some("mmproj.gguf"));
        assert_eq!(config.drafter.as_deref(), Some("mtp.gguf"));
        assert!(config.speculation.has_mode(SpeculationMode::Mtp));
        assert_eq!(config.thinking.mode, ThinkingMode::Off);
        assert_eq!(config.n_ctx, Some(8192));
        assert_eq!(config.n_gpu_layers, Some(-1));
    }

    #[test]
    fn model_config_serializes_base_path() {
        let config = ModelConfig::new_legacy(
            "/models".to_string(),
            "main.gguf".to_string(),
            vec!["</s>".to_string()],
        );

        let json = serde_json::to_string(&config).unwrap();

        assert!(json.contains("base_path"));
        assert!(!json.contains("\"path\""));
    }
}

struct InferenceState {
    backend: Option<Box<dyn RuntimeBackend>>,
    managed_server: Option<LlamaServerProcess>,
    managed_loaded_info: Option<LoadedRuntimeInfo>,
    current_model: Option<String>,
    ramdisk_bundle: Option<RamdiskBundleState>,
    last_fallback: Option<RuntimeFallbackEvent>,
    active_requests: usize,
    last_access: Instant,
}

struct ManagedServerStart {
    process: LlamaServerProcess,
    loaded_info: LoadedRuntimeInfo,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RuntimeFallbackEvent {
    pub kind: String,
    pub model: String,
    pub requested_mode: SpeculationMode,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeStatusSnapshot {
    pub loaded: bool,
    pub current_model: Option<String>,
    pub loaded_info: Option<LoadedRuntimeInfoSnapshot>,
    pub last_fallback: Option<RuntimeFallbackEvent>,
    pub active_requests: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct LoadedRuntimeInfoSnapshot {
    pub main_model_loaded: bool,
    pub mmproj_loaded: bool,
    pub draft_model_loaded: bool,
    pub n_ctx: u32,
    pub n_gpu_layers: i32,
    pub vision_supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_marker: Option<String>,
    pub speculation_enabled: bool,
    pub speculation_mode: SpeculationMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculation_fallback_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HoshikageModelInfo {
    pub id: String,
    pub main_model_configured: bool,
    pub vision: bool,
    pub mmproj_configured: bool,
    pub mtp_configured: bool,
    pub draft_model_configured: bool,
    pub thinking: ThinkingMode,
    pub fallback: FallbackMode,
}

impl From<LoadedRuntimeInfo> for LoadedRuntimeInfoSnapshot {
    fn from(info: LoadedRuntimeInfo) -> Self {
        Self {
            main_model_loaded: true,
            mmproj_loaded: info.mmproj.is_some(),
            draft_model_loaded: info.draft_model.is_some(),
            n_ctx: info.n_ctx,
            n_gpu_layers: info.n_gpu_layers,
            vision_supported: info.vision_supported,
            vision_marker: info.vision_marker,
            speculation_enabled: info.speculation_enabled,
            speculation_mode: info.speculation_mode,
            speculation_fallback_reason: info.speculation_fallback_reason,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RamdiskBundleState {
    pub model_name: String,
    pub dir: PathBuf,
    pub files: Vec<PathBuf>,
    pub total_bytes: u64,
}

#[derive(Debug, Clone)]
struct BundleFile {
    role: BundleFileRole,
    source: PathBuf,
    dest_name: &'static str,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum BundleFileRole {
    Main,
    Mmproj,
    Drafter,
}

impl BundleFileRole {
    fn as_str(self) -> &'static str {
        match self {
            BundleFileRole::Main => "main",
            BundleFileRole::Mmproj => "mmproj",
            BundleFileRole::Drafter => "drafter",
        }
    }
}

#[derive(Debug, Serialize)]
struct RamdiskManifest {
    model_name: String,
    files: Vec<RamdiskManifestFile>,
    total_bytes: u64,
}

#[derive(Debug, Serialize)]
struct RamdiskManifestFile {
    role: BundleFileRole,
    source: String,
    dest: String,
    bytes: u64,
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
                backend: None,
                managed_server: None,
                managed_loaded_info: None,
                current_model: None,
                ramdisk_bundle: None,
                last_fallback: None,
                active_requests: 0,
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
        } else if let Some(models) = self.scan_model_dir()? {
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

    pub async fn list_hoshikage_models(&self) -> Vec<HoshikageModelInfo> {
        let models = self.models.read().await;
        let mut data = models
            .iter()
            .map(|(name, config)| HoshikageModelInfo {
                id: name.clone(),
                main_model_configured: !config.model.is_empty(),
                vision: config.mmproj.is_some(),
                mmproj_configured: config.mmproj.is_some(),
                mtp_configured: config.speculation.has_mode(SpeculationMode::Mtp),
                draft_model_configured: config.speculation.has_mode(SpeculationMode::DraftModel)
                    && config.drafter.is_some(),
                thinking: config.thinking.mode.clone(),
                fallback: config.speculation.fallback.clone(),
            })
            .collect::<Vec<_>>();
        data.sort_by(|a, b| a.id.cmp(&b.id));
        data
    }

    pub async fn get_hoshikage_model(&self, name: &str) -> Result<HoshikageModelInfo> {
        let config = self.get_model(name).await?;
        Ok(HoshikageModelInfo {
            id: name.to_string(),
            main_model_configured: !config.model.is_empty(),
            vision: config.mmproj.is_some(),
            mmproj_configured: config.mmproj.is_some(),
            mtp_configured: config.speculation.has_mode(SpeculationMode::Mtp),
            draft_model_configured: config.speculation.has_mode(SpeculationMode::DraftModel)
                && config.drafter.is_some(),
            thinking: config.thinking.mode,
            fallback: config.speculation.fallback,
        })
    }

    pub fn runtime_status(&self) -> RuntimeStatusSnapshot {
        let state = self
            .inference
            .lock()
            .map_err(|e| crate::error::HoshikageError::Other(format!("Lock error: {}", e)))
            .ok();

        let Some(mut state) = state else {
            return RuntimeStatusSnapshot {
                loaded: false,
                current_model: None,
                loaded_info: None,
                last_fallback: None,
                active_requests: 0,
            };
        };

        let managed_loaded = state
            .managed_server
            .as_mut()
            .map(|server| server.is_running())
            .unwrap_or(false);
        let backend_status = state.backend.as_ref().map(|backend| backend.status());
        RuntimeStatusSnapshot {
            loaded: managed_loaded
                || backend_status
                    .as_ref()
                    .map(|status| status.loaded)
                    .unwrap_or(false),
            current_model: state.current_model.clone(),
            loaded_info: if managed_loaded {
                state
                    .managed_loaded_info
                    .clone()
                    .map(LoadedRuntimeInfoSnapshot::from)
            } else {
                backend_status
                    .and_then(|status| status.loaded_info)
                    .map(LoadedRuntimeInfoSnapshot::from)
            },
            last_fallback: state.last_fallback.clone(),
            active_requests: state.active_requests,
        }
    }

    pub fn default_temperature(&self) -> f32 {
        self.config.default_temperature
    }

    pub fn default_top_p(&self) -> f32 {
        self.config.default_top_p
    }

    pub fn default_repeat_penalty(&self) -> f32 {
        self.config.repeat_penalty
    }

    pub fn default_repeat_last_n(&self) -> u32 {
        self.config.repeat_last_n
    }

    pub fn uses_managed_llama_server(&self) -> bool {
        self.config.runtime_backend == RuntimeBackendKind::LlamaServerManaged
    }

    pub fn begin_managed_request(&self, model_name: &str) {
        let Ok(mut state) = self.inference.lock() else {
            tracing::error!("Failed to lock inference state when beginning managed request");
            return;
        };
        if state.current_model.as_deref() == Some(model_name) {
            state.active_requests = state.active_requests.saturating_add(1);
            state.last_access = Instant::now();
        }
    }

    pub fn finish_managed_request(&self, model_name: &str) {
        let Ok(mut state) = self.inference.lock() else {
            tracing::error!("Failed to lock inference state when finishing managed request");
            return;
        };
        if state.current_model.as_deref() == Some(model_name) {
            state.active_requests = state.active_requests.saturating_sub(1);
            state.last_access = Instant::now();
        }
    }

    pub async fn ensure_managed_llama_server(&self, model_name: &str) -> Result<String> {
        let _permit =
            self.semaphore.clone().acquire_owned().await.map_err(|e| {
                crate::error::HoshikageError::Other(format!("Semaphore error: {}", e))
            })?;

        let model_config = self.get_model(model_name).await?;
        let endpoint = self.llama_server_base_url();
        let mut started = false;

        {
            let mut state = self
                .inference
                .lock()
                .map_err(|e| crate::error::HoshikageError::Other(format!("Lock error: {}", e)))?;

            let managed_running = state
                .managed_server
                .as_mut()
                .map(|server| server.is_running())
                .unwrap_or(false);
            let current_loaded =
                state.current_model.as_deref() == Some(model_name) && managed_running;

            if !current_loaded {
                if state.active_requests > 0 {
                    return Err(crate::error::HoshikageError::Other(format!(
                        "managed runtime is busy with {} active request(s)",
                        state.active_requests
                    )));
                }
                self.stop_loaded_runtime(&mut state);
                let (model_path, ramdisk_bundle) =
                    self.resolve_model_path(model_name, &model_config)?;
                let request =
                    self.build_load_request(&model_config, model_path, ramdisk_bundle.as_ref());
                let start = self.start_managed_server_process(model_name, &request)?;

                state.managed_server = Some(start.process);
                state.managed_loaded_info = Some(start.loaded_info);
                state.current_model = Some(model_name.to_string());
                state.ramdisk_bundle = ramdisk_bundle;
                state.last_fallback = None;
                started = true;
            }

            state.last_access = Instant::now();
        }

        if started {
            if let Err(err) = self.wait_for_managed_server(&endpoint).await {
                self.mark_managed_server_unhealthy(model_name);
                if model_config.speculation.fallback == FallbackMode::Warn
                    && !model_config.speculation.is_off()
                {
                    self.start_managed_server_without_speculation(model_name, &model_config)
                        .await?;
                } else {
                    return Err(err);
                }
            }
        }

        Ok(endpoint)
    }

    async fn start_managed_server_without_speculation(
        &self,
        model_name: &str,
        model_config: &ModelConfig,
    ) -> Result<()> {
        let endpoint = self.llama_server_base_url();
        let fallback_reason =
            "managed llama-server did not become healthy with speculation options".to_string();

        {
            let mut state = self
                .inference
                .lock()
                .map_err(|e| crate::error::HoshikageError::Other(format!("Lock error: {}", e)))?;
            self.stop_loaded_runtime(&mut state);

            let (model_path, ramdisk_bundle) = self.resolve_model_path(model_name, model_config)?;
            let mut request =
                self.build_load_request(model_config, model_path, ramdisk_bundle.as_ref());
            let requested_mode = request.speculation.primary_mode();
            request.speculation = SpeculationConfig::default();
            request.draft_model = None;
            request.n_rs_seq = 0;

            let start = self.start_managed_server_process(model_name, &request)?;
            state.managed_server = Some(start.process);
            state.managed_loaded_info = Some(LoadedRuntimeInfo {
                speculation_fallback_reason: Some(fallback_reason.clone()),
                ..start.loaded_info
            });
            state.current_model = Some(model_name.to_string());
            state.ramdisk_bundle = ramdisk_bundle;
            state.last_fallback = Some(RuntimeFallbackEvent {
                kind: "speculation".to_string(),
                model: model_name.to_string(),
                requested_mode,
                reason: fallback_reason,
            });
            state.last_access = Instant::now();
        }

        if let Err(err) = self.wait_for_managed_server(&endpoint).await {
            self.mark_managed_server_unhealthy(model_name);
            return Err(err);
        }

        Ok(())
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

                if state.active_requests > 0 {
                    continue;
                }

                if idle_timeout > 0 && idle_secs >= idle_timeout {
                    if state.backend.is_some() || state.managed_server.is_some() {
                        self.stop_loaded_runtime(&mut state);
                        state.current_model = None;
                        tracing::info!("Unloaded model due to idle timeout");
                    }
                }

                if great_timeout > 0 && idle_secs >= great_timeout * 60 {
                    if let Some(bundle) = state.ramdisk_bundle.take() {
                        if let Err(e) = std::fs::remove_dir_all(&bundle.dir) {
                            tracing::warn!(
                                "Failed to remove ramdisk bundle {}: {}",
                                bundle.dir.display(),
                                e
                            );
                        } else {
                            tracing::info!(
                                model = bundle.model_name,
                                bytes = bundle.total_bytes,
                                files = bundle.files.len(),
                                "Removed ramdisk bundle {}",
                                bundle.dir.display()
                            );
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
        let _permit =
            self.semaphore.clone().acquire_owned().await.map_err(|e| {
                crate::error::HoshikageError::Other(format!("Semaphore error: {}", e))
            })?;

        let model_config = self.get_model(model_name).await?;
        let mut state = self
            .inference
            .lock()
            .map_err(|e| crate::error::HoshikageError::Other(format!("Lock error: {}", e)))?;

        self.load_model_if_needed(&mut state, model_name, &model_config)?;

        {
            let backend = state.backend.as_mut().ok_or_else(|| {
                crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
            })?;
            backend.prepare_for_inference()?;
        }

        let backend = state.backend.as_ref().ok_or_else(|| {
            crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
        })?;

        let prompt_tokens = backend.count_tokens(prompt)? as u32;

        if backend.is_diffusion_model()? {
            tracing::info!("Using diffusion generation for model: {}", model_name);
            let (diff_output, _, _) = backend.generate_with_diffusion(prompt, &params)?;
            let thinking_decision = ThinkingController::decide(&model_config.thinking);
            let diff_output =
                ThinkingController::strip_output_if_needed(&thinking_decision, &diff_output);
            let diff_completion = backend.count_tokens(&diff_output)? as u32;
            if thinking_decision.strip_thinking {
                tracing::debug!(
                    model = model_name,
                    "Thinking off safety filtering applied to diffusion response"
                );
            }
            state.last_access = Instant::now();
            return Ok((diff_output, prompt_tokens, diff_completion));
        }

        let raw_output = backend.generate(prompt, &params)?;
        let thinking_decision = ThinkingController::decide(&model_config.thinking);
        let output = ThinkingController::strip_output_if_needed(&thinking_decision, &raw_output);
        if thinking_decision.strip_thinking && output != raw_output {
            tracing::debug!(
                model = model_name,
                removed_bytes = raw_output.len().saturating_sub(output.len()),
                "Thinking off safety filtering removed hidden block content"
            );
        }
        let completion_tokens = backend.count_tokens(&output)? as u32;
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
            let backend = state.backend.as_mut().ok_or_else(|| {
                crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
            })?;
            backend.prepare_for_inference()?;
            let prompt = backend.format_chat_prompt(messages)?;
            let thinking_decision = ThinkingController::decide(&model_config.thinking);
            let prompt =
                ThinkingController::apply_prompt_policy_if_needed(&thinking_decision, &prompt);
            if thinking_decision.strip_thinking {
                tracing::debug!(
                    model = model_name,
                    "Thinking off prompt policy applied before inference"
                );
            }
            prompt
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
        let _permit =
            self.semaphore.clone().acquire_owned().await.map_err(|e| {
                crate::error::HoshikageError::Other(format!("Semaphore error: {}", e))
            })?;

        let model_config = self.get_model(&model_name).await?;
        let mut state = self
            .inference
            .lock()
            .map_err(|e| crate::error::HoshikageError::Other(format!("Lock error: {}", e)))?;

        self.load_model_if_needed(&mut state, &model_name, &model_config)?;

        {
            let backend = state.backend.as_mut().ok_or_else(|| {
                crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
            })?;
            backend.prepare_for_inference()?;
        }

        let backend = state.backend.as_ref().ok_or_else(|| {
            crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
        })?;

        let thinking_decision = ThinkingController::decide(&model_config.thinking);
        let mut stream_filter = ThinkingStreamFilter::new(&thinking_decision);
        let result = backend.generate_stream(&prompt, &params, &mut |chunk| {
            for visible_chunk in stream_filter.push(&chunk) {
                if !visible_chunk.is_empty() {
                    sender
                        .send(Ok(visible_chunk))
                        .map_err(|e| crate::error::HoshikageError::Other(e.to_string()))?;
                }
            }
            Ok(())
        });

        if result.is_ok() {
            for visible_chunk in stream_filter.finish() {
                if !visible_chunk.is_empty() {
                    sender
                        .send(Ok(visible_chunk))
                        .map_err(|e| crate::error::HoshikageError::Other(e.to_string()))?;
                }
            }
        }

        if thinking_decision.strip_thinking && stream_filter.stripped_bytes() > 0 {
            tracing::debug!(
                model = model_name,
                removed_bytes = stream_filter.stripped_bytes(),
                "Thinking off safety filtering removed hidden block content from stream"
            );
        }

        state.last_access = Instant::now();

        if let Err(e) = result {
            let _ = sender.send(Err(e));
        }

        Ok(())
    }

    pub async fn is_diffusion_model(&self, model_name: &str) -> Result<bool> {
        if self.uses_managed_llama_server() {
            return Ok(false);
        }

        let model_config = self.get_model(model_name).await?;
        let mut state = self
            .inference
            .lock()
            .map_err(|e| crate::error::HoshikageError::Other(format!("Lock error: {}", e)))?;

        self.load_model_if_needed(&mut state, model_name, &model_config)?;

        let backend = state.backend.as_ref().ok_or_else(|| {
            crate::error::HoshikageError::InferenceError("Model not loaded".to_string())
        })?;
        let is_diffusion = backend.is_diffusion_model()?;

        state.last_access = Instant::now();
        Ok(is_diffusion)
    }

    fn llama_server_base_url(&self) -> String {
        format!(
            "http://{}:{}",
            self.config.llama_server_host, self.config.llama_server_port
        )
    }

    fn start_managed_server_process(
        &self,
        model_name: &str,
        request: &LlamaLoadRequest,
    ) -> Result<ManagedServerStart> {
        let runtime_dir = self.config.llama_cpp_runtime_dir()?;
        let log_file = self.managed_server_log_file(
            model_name,
            if request.speculation.is_off() {
                "normal"
            } else {
                "speculation"
            },
        )?;
        let launch = LlamaServerLaunchConfig {
            server_path: runtime_dir.join(if cfg!(target_os = "windows") {
                "llama-server.exe"
            } else {
                "llama-server"
            }),
            host: self.config.llama_server_host.clone(),
            port: self.config.llama_server_port,
            alias: model_name.to_string(),
            log_file: Some(log_file),
            sleep_idle_secs: self.config.llama_server_sleep_idle_secs,
            request: request.clone(),
        };
        let command_spec = LlamaServerCommandSpec::from_launch_config(&launch);
        let process = LlamaServerProcess::start(command_spec)?;

        let loaded_info = LoadedRuntimeInfo {
            main_model: request.main_model.clone(),
            mmproj: request.mmproj.clone(),
            draft_model: request.draft_model.clone(),
            n_ctx: request.n_ctx,
            n_gpu_layers: request.n_gpu_layers,
            vision_supported: request.mmproj.is_some(),
            vision_marker: request.mmproj.as_ref().map(|_| "llama-server".to_string()),
            speculation_enabled: !request.speculation.is_off(),
            speculation_mode: request.speculation.primary_mode(),
            speculation_fallback_reason: None,
        };

        tracing::info!(
            model = model_name,
            main = %loaded_info.main_model.display(),
            n_ctx = loaded_info.n_ctx,
            n_gpu_layers = loaded_info.n_gpu_layers,
            mmproj = loaded_info.mmproj.as_ref().map(|path| path.display().to_string()),
            draft_model = loaded_info.draft_model.as_ref().map(|path| path.display().to_string()),
            speculation_mode = ?loaded_info.speculation_mode,
            "Started managed llama-server runtime"
        );

        Ok(ManagedServerStart {
            process,
            loaded_info,
        })
    }

    async fn wait_for_managed_server(&self, endpoint: &str) -> Result<()> {
        let url = format!("{}/health", endpoint);
        let client = reqwest::Client::new();
        let deadline =
            Instant::now() + Duration::from_secs(self.config.llama_server_startup_timeout_secs);

        while Instant::now() < deadline {
            match client.get(&url).send().await {
                Ok(response) if response.status().is_success() => return Ok(()),
                _ => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }

        Err(crate::error::HoshikageError::ModelLoadFailed(format!(
            "llama-server did not become healthy: {}",
            url
        )))
    }

    fn managed_server_log_file(&self, model_name: &str, phase: &str) -> Result<PathBuf> {
        let config_dir = dirs::config_dir().ok_or_else(|| {
            crate::error::HoshikageError::ConfigError("Config directory not found".to_string())
        })?;
        let log_dir = config_dir.join("hoshikage").join("logs");
        std::fs::create_dir_all(&log_dir)?;
        let safe_name = model_name
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                    ch
                } else {
                    '_'
                }
            })
            .collect::<String>();
        Ok(log_dir.join(format!("llama-server-{}-{}.log", safe_name, phase)))
    }

    pub fn mark_managed_server_unhealthy(&self, model_name: &str) {
        let Ok(mut state) = self.inference.lock() else {
            return;
        };
        if state.current_model.as_deref() != Some(model_name) {
            return;
        }
        if let Some(server) = state.managed_server.as_mut() {
            server.stop();
        }
        state.managed_server = None;
        state.managed_loaded_info = None;
        state.current_model = None;
    }

    fn stop_loaded_runtime(&self, state: &mut InferenceState) {
        if let Some(backend) = state.backend.as_mut() {
            backend.unload();
        }
        if let Some(server) = state.managed_server.as_mut() {
            server.stop();
        }
        state.backend = None;
        state.managed_server = None;
        state.managed_loaded_info = None;
        state.active_requests = 0;
    }

    fn load_model_if_needed(
        &self,
        state: &mut InferenceState,
        model_name: &str,
        model_config: &ModelConfig,
    ) -> Result<()> {
        if self.uses_managed_llama_server() {
            return Err(crate::error::HoshikageError::ConfigError(
                "prompt-based FFI inference path is disabled while HOSHIKAGE_RUNTIME_BACKEND=llama-server-managed".to_string(),
            ));
        }

        let needs_reload = state.backend.is_none()
            || state.current_model.as_deref() != Some(model_name)
            || !state
                .backend
                .as_ref()
                .map(|backend| backend.status().loaded)
                .unwrap_or(false);

        if !needs_reload {
            return Ok(());
        }

        self.stop_loaded_runtime(state);

        let lib_path = self.config.resolve_lib_path()?;
        let mut backend = LlamaFfiBackend::new(lib_path, self.config.clone())?;
        let (model_path, ramdisk_bundle) = self.resolve_model_path(model_name, model_config)?;
        let request =
            self.build_load_request(model_config, model_path.clone(), ramdisk_bundle.as_ref());
        let loaded = backend.load(&request)?;

        tracing::info!(
            model = model_name,
            main = %loaded.main_model.display(),
            n_ctx = loaded.n_ctx,
            n_gpu_layers = loaded.n_gpu_layers,
            mmproj = loaded.mmproj.as_ref().map(|path| path.display().to_string()),
            draft_model = loaded.draft_model.as_ref().map(|path| path.display().to_string()),
            vision_supported = loaded.vision_supported,
            vision_marker = loaded.vision_marker.as_deref(),
            speculation_enabled = loaded.speculation_enabled,
            speculation_mode = ?loaded.speculation_mode,
            speculation_fallback_reason = loaded.speculation_fallback_reason.as_deref(),
            "Loaded runtime backend"
        );

        state.backend = Some(Box::new(backend));
        state.current_model = Some(model_name.to_string());
        state.ramdisk_bundle = ramdisk_bundle;
        state.last_fallback =
            loaded
                .speculation_fallback_reason
                .clone()
                .map(|reason| RuntimeFallbackEvent {
                    kind: "speculation".to_string(),
                    model: model_name.to_string(),
                    requested_mode: request.speculation.primary_mode(),
                    reason,
                });
        state.last_access = Instant::now();

        Ok(())
    }

    fn build_load_request(
        &self,
        config: &ModelConfig,
        main_model: PathBuf,
        ramdisk_bundle: Option<&RamdiskBundleState>,
    ) -> LlamaLoadRequest {
        LlamaLoadRequest {
            main_model,
            mmproj: resolve_runtime_aux_path(
                &config.path,
                config.mmproj.as_deref(),
                ramdisk_bundle,
                "mmproj.gguf",
            ),
            draft_model: resolve_runtime_aux_path(
                &config.path,
                config.drafter.as_deref(),
                ramdisk_bundle,
                "drafter.gguf",
            ),
            n_ctx: config.n_ctx.unwrap_or(self.config.n_ctx),
            n_gpu_layers: config.n_gpu_layers.unwrap_or(self.config.n_gpu_layers),
            n_rs_seq: if config.speculation.has_mode(SpeculationMode::Mtp) {
                16
            } else {
                0
            },
            speculation: config.speculation.clone(),
            thinking: config.thinking.clone(),
        }
    }

    fn resolve_model_path(
        &self,
        model_name: &str,
        config: &ModelConfig,
    ) -> Result<(PathBuf, Option<RamdiskBundleState>)> {
        let model_path = config.main_model_path();

        if cfg!(target_os = "linux") {
            if let Some(ramdisk_path) = &self.config.ramdisk_path {
                let bundle = self.materialize_ramdisk_bundle(model_name, config, ramdisk_path)?;
                let model_path = bundle.dir.join("main.gguf");
                return Ok((model_path, Some(bundle)));
            }
        }

        Ok((model_path, None))
    }

    fn materialize_ramdisk_bundle(
        &self,
        model_name: &str,
        config: &ModelConfig,
        ramdisk_path: &str,
    ) -> Result<RamdiskBundleState> {
        let ramdisk_root = PathBuf::from(ramdisk_path);
        std::fs::create_dir_all(&ramdisk_root)?;

        let lock_path = ramdisk_root.join("hoshikage.lock");
        let lock_file = File::create(&lock_path)?;
        fs2::FileExt::try_lock_exclusive(&lock_file).map_err(|e| {
            crate::error::HoshikageError::Other(format!(
                "RAM disk cache is locked by another Hoshikage process: {}",
                e
            ))
        })?;

        let managed_dir = ramdisk_root.join("hoshikage");
        let current_dir = managed_dir.join("current");
        let tmp_dir = managed_dir.join("current.tmp");

        if managed_dir.exists() {
            std::fs::remove_dir_all(&managed_dir)?;
        }
        std::fs::create_dir_all(&tmp_dir)?;

        let result = (|| {
            let files = self.bundle_files(config)?;
            let total_bytes = bundle_total_bytes(&files)?;
            ensure_ramdisk_capacity(&ramdisk_root, total_bytes)?;

            let mut manifest_files = Vec::with_capacity(files.len());
            let mut copied_files = Vec::with_capacity(files.len());

            for file in files {
                let bytes = std::fs::metadata(&file.source)?.len();
                let dest = tmp_dir.join(file.dest_name);
                std::fs::copy(&file.source, &dest)?;
                copied_files.push(dest.clone());
                manifest_files.push(RamdiskManifestFile {
                    role: file.role,
                    source: file.source.to_string_lossy().to_string(),
                    dest: current_dir
                        .join(file.dest_name)
                        .to_string_lossy()
                        .to_string(),
                    bytes,
                });
            }

            let manifest = RamdiskManifest {
                model_name: model_name.to_string(),
                files: manifest_files,
                total_bytes,
            };
            std::fs::write(
                tmp_dir.join("manifest.json"),
                serde_json::to_string_pretty(&manifest)?,
            )?;

            std::fs::rename(&tmp_dir, &current_dir)?;

            let files = copied_files
                .into_iter()
                .map(|path| current_dir.join(path.file_name().unwrap_or_default()))
                .collect::<Vec<_>>();

            tracing::info!(
                model = model_name,
                bytes = total_bytes,
                files = files.len(),
                "Materialized model bundle on RAM disk"
            );

            Ok(RamdiskBundleState {
                model_name: model_name.to_string(),
                dir: current_dir,
                files,
                total_bytes,
            })
        })();

        if result.is_err() {
            let _ = std::fs::remove_dir_all(&managed_dir);
        }

        result
    }

    fn bundle_files(&self, config: &ModelConfig) -> Result<Vec<BundleFile>> {
        let mut files = vec![BundleFile {
            role: BundleFileRole::Main,
            source: config.main_model_path(),
            dest_name: "main.gguf",
        }];

        if let Some(mmproj) = &config.mmproj {
            files.push(BundleFile {
                role: BundleFileRole::Mmproj,
                source: resolve_bundle_path(&config.path, mmproj),
                dest_name: "mmproj.gguf",
            });
        }

        if let Some(drafter) = &config.drafter {
            files.push(BundleFile {
                role: BundleFileRole::Drafter,
                source: resolve_bundle_path(&config.path, drafter),
                dest_name: "drafter.gguf",
            });
        }

        for file in &files {
            if !file.source.exists() {
                return Err(crate::error::HoshikageError::ModelLoadFailed(format!(
                    "Bundle {} file not found: {}",
                    file.role.as_str(),
                    file.source.display()
                )));
            }
            if !file.source.is_file() {
                return Err(crate::error::HoshikageError::ModelLoadFailed(format!(
                    "Bundle {} path is not a file: {}",
                    file.role.as_str(),
                    file.source.display()
                )));
            }
        }

        Ok(files)
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
                    ..ModelConfig::new_legacy(
                        base_dir.to_string_lossy().to_string(),
                        file_name,
                        Vec::new(),
                    )
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
