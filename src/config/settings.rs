use crate::error::Result;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub host: String,
    pub log_level: String,
    pub log_file_path: Option<String>,
    pub idle_timeout: u64,
    pub great_timeout: u64,
    pub ramdisk_path: Option<String>,
    pub n_gpu_layers: i32,
    pub n_ctx: u32,
    pub default_temperature: f32,
    pub default_top_p: f32,
    pub model_dir: Option<PathBuf>,
    pub model_map_file: Option<PathBuf>,
    pub lib_path: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: 3030,
            host: "127.0.0.1".to_string(),
            log_level: "info".to_string(),
            log_file_path: None,
            idle_timeout: 300,
            great_timeout: 60,
            ramdisk_path: None,
            n_gpu_layers: -1,
            n_ctx: 4096,
            default_temperature: 0.2,
            default_top_p: 0.8,
            model_dir: None,
            model_map_file: None,
            lib_path: None,
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let mut config = Self::default();

        if let Ok(env_path) = std::env::var("HOSHIKAGE_CONFIG_PATH") {
            dotenvy::from_path(&env_path).ok();
        }

        let mut config_dir = dirs::config_dir().ok_or_else(|| {
            crate::error::HoshikageError::ConfigError("Config directory not found".to_string())
        })?;

        config_dir.push("hoshikage");

        let env_path = config_dir.join(".env");
        if env_path.exists() {
            dotenvy::from_path(&env_path).ok();
        }

        if let Ok(port) = std::env::var("PORT") {
            config.port = port.parse().unwrap_or(3030);
        }

        if let Ok(host) = std::env::var("HOST") {
            config.host = host;
        }

        if let Ok(log_level) = std::env::var("RUST_LOG") {
            config.log_level = log_level;
        }

        if let Ok(log_file) = std::env::var("LOG_FILE_PATH") {
            config.log_file_path = Some(log_file);
        }

        if let Ok(idle_timeout) = std::env::var("IDLE_TIMEOUT") {
            config.idle_timeout = idle_timeout.parse().unwrap_or(300);
        }

        if let Ok(great_timeout) = std::env::var("GREAT_TIMEOUT") {
            config.great_timeout = great_timeout.parse().unwrap_or(60);
        }

        if let Ok(ramdisk_path) = std::env::var("RAMDISK_PATH") {
            config.ramdisk_path = if ramdisk_path.is_empty() {
                None
            } else {
                Some(ramdisk_path)
            };
        }

        if let Ok(n_gpu_layers) = std::env::var("N_GPU_LAYERS") {
            config.n_gpu_layers = n_gpu_layers.parse().unwrap_or(-1);
        }

        if let Ok(n_ctx) = std::env::var("N_CTX") {
            config.n_ctx = n_ctx.parse().unwrap_or(4096);
        }

        if let Ok(temperature) = std::env::var("TEMPERATURE") {
            config.default_temperature = temperature.parse().unwrap_or(0.2);
        }

        if let Ok(top_p) = std::env::var("TOP_P") {
            config.default_top_p = top_p.parse().unwrap_or(0.8);
        }

        if let Ok(model_dir) = std::env::var("MODEL_DIR") {
            config.model_dir = Some(PathBuf::from(model_dir));
        }

        if let Ok(model_map_file) = std::env::var("MODEL_MAP_FILE") {
            config.model_map_file = Some(PathBuf::from(model_map_file));
        }

        if let Ok(lib_path) = std::env::var("HOSHIKAGE_LIB_PATH") {
            config.lib_path = Some(lib_path);
        }

        Ok(config)
    }

    pub fn model_map_path(&self) -> Result<PathBuf> {
        if let Some(path) = &self.model_map_file {
            Ok(path.clone())
        } else {
            let config_dir = dirs::config_dir().ok_or_else(|| {
                crate::error::HoshikageError::ConfigError("Config directory not found".to_string())
            })?;

            let mut config_dir = config_dir;
            config_dir.push("hoshikage");

            Ok(config_dir.join("model_map.json"))
        }
    }

    pub fn resolve_lib_path(&self) -> Result<PathBuf> {
        let lib_name = if cfg!(target_os = "windows") {
            "llama.dll"
        } else if cfg!(target_os = "macos") {
            "libllama.dylib"
        } else {
            "libllama.so"
        };

        if let Some(path) = &self.lib_path {
            let path = PathBuf::from(path);
            if path.is_dir() {
                return Ok(path.join(lib_name));
            }
            return Ok(path);
        }

        if let Some(config_dir) = dirs::config_dir() {
            let candidate = config_dir.join("hoshikage").join("lib").join(lib_name);
            if candidate.exists() {
                return Ok(candidate);
            }
        }

        Ok(PathBuf::from(lib_name))
    }
}
