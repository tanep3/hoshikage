use crate::error::Result;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownFieldPolicy {
    Compatible,
    Strict,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeBackendKind {
    LlamaServerManaged,
    LlamaFfi,
}

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
    pub repeat_penalty: f32,
    pub repeat_last_n: u32,
    pub model_dir: Option<PathBuf>,
    pub model_map_file: Option<PathBuf>,
    pub runtime_backend: RuntimeBackendKind,
    pub llama_cpp_runtime_dir: Option<PathBuf>,
    pub llama_server_host: String,
    pub llama_server_port: u16,
    pub llama_server_startup_timeout_secs: u64,
    pub llama_server_sleep_idle_secs: Option<u64>,
    pub responses_unknown_field_policy: UnknownFieldPolicy,
    pub max_request_bytes: usize,
    pub max_tool_schema_bytes: usize,
    pub max_single_tool_schema_bytes: usize,
    pub max_tools: usize,
    pub max_tool_argument_bytes: usize,
    pub max_tool_result_bytes: usize,
    pub responses_queue_capacity: usize,
    pub responses_queue_timeout_ms: u64,
    pub responses_timeout_secs: u64,
    pub first_token_timeout_secs: u64,
    pub stream_idle_timeout_secs: u64,
    pub generation_timeout_secs: u64,
    pub auth_token_file: Option<PathBuf>,
    pub debug_capture: bool,
    pub lib_path: Option<String>,
    // Diffusion parameters
    pub diffusion_steps: i32,
    pub diffusion_algorithm: i32,
    pub diffusion_schedule: i32,
    pub diffusion_cfg_scale: f32,
    pub diffusion_max_tokens: i32,
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
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            model_dir: None,
            model_map_file: None,
            runtime_backend: RuntimeBackendKind::LlamaServerManaged,
            llama_cpp_runtime_dir: None,
            llama_server_host: "127.0.0.1".to_string(),
            llama_server_port: 13030,
            llama_server_startup_timeout_secs: 120,
            llama_server_sleep_idle_secs: None,
            responses_unknown_field_policy: UnknownFieldPolicy::Compatible,
            max_request_bytes: 8_388_608,
            max_tool_schema_bytes: 1_048_576,
            max_single_tool_schema_bytes: 262_144,
            max_tools: 128,
            max_tool_argument_bytes: 65_536,
            max_tool_result_bytes: 4_194_304,
            responses_queue_capacity: 4,
            responses_queue_timeout_ms: 30_000,
            responses_timeout_secs: 900,
            first_token_timeout_secs: 120,
            stream_idle_timeout_secs: 120,
            generation_timeout_secs: 600,
            auth_token_file: None,
            debug_capture: false,
            lib_path: None,
            // Diffusion defaults
            diffusion_steps: 50,
            diffusion_algorithm: 4,
            diffusion_schedule: 0,
            diffusion_cfg_scale: 0.0,
            diffusion_max_tokens: 0,
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let mut config = Self::default();

        if let Ok(env_path) = std::env::var("HOSHIKAGE_CONFIG_PATH") {
            load_dotenv(&PathBuf::from(env_path))?;
        }

        let mut config_dir = dirs::config_dir().ok_or_else(|| {
            crate::error::HoshikageError::ConfigError("Config directory not found".to_string())
        })?;

        config_dir.push("hoshikage");

        let env_path = config_dir.join(".env");
        if env_path.exists() {
            load_dotenv(&env_path)?;
        }

        if let Some(port) = parse_env("PORT")? {
            config.port = port;
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

        if let Some(idle_timeout) = parse_env("IDLE_TIMEOUT")? {
            config.idle_timeout = idle_timeout;
        }

        if let Some(great_timeout) = parse_env("GREAT_TIMEOUT")? {
            config.great_timeout = great_timeout;
        }

        if let Ok(ramdisk_path) = std::env::var("RAMDISK_PATH") {
            config.ramdisk_path = if ramdisk_path.is_empty() {
                None
            } else {
                Some(ramdisk_path)
            };
        }

        if let Some(n_gpu_layers) = parse_env("N_GPU_LAYERS")? {
            config.n_gpu_layers = n_gpu_layers;
        }

        if let Some(n_ctx) = parse_env("N_CTX")? {
            config.n_ctx = n_ctx;
        }

        if let Some(temperature) = parse_env("TEMPERATURE")? {
            config.default_temperature = temperature;
        }

        if let Some(top_p) = parse_env("TOP_P")? {
            config.default_top_p = top_p;
        }

        if let Some(repeat_penalty) = parse_env("REPEAT_PENALTY")? {
            config.repeat_penalty = repeat_penalty;
        }

        if let Some(repeat_last_n) = parse_env("REPEAT_LAST_N")? {
            config.repeat_last_n = repeat_last_n;
        }

        if let Ok(model_dir) = std::env::var("MODEL_DIR") {
            config.model_dir = Some(PathBuf::from(model_dir));
        }

        if let Ok(model_map_file) = std::env::var("MODEL_MAP_FILE") {
            config.model_map_file = Some(PathBuf::from(model_map_file));
        }

        if let Ok(runtime_backend) = std::env::var("HOSHIKAGE_RUNTIME_BACKEND") {
            config.runtime_backend = RuntimeBackendKind::parse(&runtime_backend)?;
        }

        if let Ok(runtime_dir) = std::env::var("HOSHIKAGE_LLAMA_CPP_RUNTIME_DIR") {
            config.llama_cpp_runtime_dir = Some(PathBuf::from(runtime_dir));
        }

        if let Ok(host) = std::env::var("HOSHIKAGE_LLAMA_SERVER_HOST") {
            config.llama_server_host = host;
        }

        if let Some(port) = parse_env("HOSHIKAGE_LLAMA_SERVER_PORT")? {
            config.llama_server_port = port;
        }

        if let Some(timeout) = parse_env("HOSHIKAGE_LLAMA_SERVER_STARTUP_TIMEOUT_SECS")? {
            config.llama_server_startup_timeout_secs = timeout;
        }

        if let Ok(sleep_idle) = std::env::var("HOSHIKAGE_LLAMA_SERVER_SLEEP_IDLE_SECS") {
            config.llama_server_sleep_idle_secs = match sleep_idle.as_str() {
                "" | "off" | "disabled" => None,
                value => Some(parse_value(
                    "HOSHIKAGE_LLAMA_SERVER_SLEEP_IDLE_SECS",
                    value,
                )?),
            };
        }

        if let Ok(policy) = std::env::var("RESPONSES_UNKNOWN_FIELD_POLICY") {
            config.responses_unknown_field_policy = UnknownFieldPolicy::parse(&policy)?;
        }
        apply_responses_environment(&mut config)?;

        if let Ok(lib_path) = std::env::var("HOSHIKAGE_LIB_PATH") {
            config.lib_path = Some(lib_path);
        }

        if let Some(diffusion_steps) = parse_env("DIFFUSION_STEPS")? {
            config.diffusion_steps = diffusion_steps;
        }

        if let Some(diffusion_algorithm) = parse_env("DIFFUSION_ALGORITHM")? {
            config.diffusion_algorithm = diffusion_algorithm;
        }

        if let Some(diffusion_schedule) = parse_env("DIFFUSION_SCHEDULE")? {
            config.diffusion_schedule = diffusion_schedule;
        }

        if let Some(diffusion_cfg_scale) = parse_env("DIFFUSION_CFG_SCALE")? {
            config.diffusion_cfg_scale = diffusion_cfg_scale;
        }

        if let Some(diffusion_max_tokens) = parse_env("DIFFUSION_MAX_TOKENS")? {
            config.diffusion_max_tokens = diffusion_max_tokens;
        }

        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_request_bytes == 0
            || self.max_tool_schema_bytes == 0
            || self.max_single_tool_schema_bytes == 0
            || self.max_tools == 0
            || self.max_tool_argument_bytes == 0
            || self.max_tool_result_bytes == 0
            || self.responses_queue_capacity == 0
        {
            return Err(crate::error::HoshikageError::ConfigError(
                "Responses size and queue limits must be greater than zero".to_string(),
            ));
        }
        if self.max_single_tool_schema_bytes > self.max_tool_schema_bytes {
            return Err(crate::error::HoshikageError::ConfigError(
                "single Tool Schema limit exceeds total Tool Schema limit".to_string(),
            ));
        }
        if self.max_tool_result_bytes > self.max_request_bytes {
            return Err(crate::error::HoshikageError::ConfigError(
                "Tool Result limit exceeds request body limit".to_string(),
            ));
        }
        if self
            .auth_token_file
            .as_ref()
            .is_some_and(|path| path.as_os_str().is_empty())
        {
            return Err(crate::error::HoshikageError::ConfigError(
                "HOSHIKAGE_AUTH_TOKEN_FILE must not be empty".to_string(),
            ));
        }
        for (name, value) in [
            (
                "HOSHIKAGE_RESPONSES_QUEUE_TIMEOUT_MS",
                self.responses_queue_timeout_ms,
            ),
            (
                "HOSHIKAGE_RESPONSES_TIMEOUT_SECS",
                self.responses_timeout_secs,
            ),
            (
                "HOSHIKAGE_FIRST_TOKEN_TIMEOUT_SECS",
                self.first_token_timeout_secs,
            ),
            (
                "HOSHIKAGE_STREAM_IDLE_TIMEOUT_SECS",
                self.stream_idle_timeout_secs,
            ),
            (
                "HOSHIKAGE_GENERATION_TIMEOUT_SECS",
                self.generation_timeout_secs,
            ),
        ] {
            if value == 0 {
                return Err(crate::error::HoshikageError::ConfigError(format!(
                    "{name} must be greater than zero"
                )));
            }
        }
        Ok(())
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

    pub fn llama_cpp_runtime_dir(&self) -> Result<PathBuf> {
        if let Some(path) = &self.llama_cpp_runtime_dir {
            return Ok(path.clone());
        }

        let config_dir = dirs::config_dir().ok_or_else(|| {
            crate::error::HoshikageError::ConfigError("Config directory not found".to_string())
        })?;

        Ok(config_dir.join("hoshikage").join("llama.cpp"))
    }

    pub fn auth_token_path(&self) -> Result<PathBuf> {
        if let Some(path) = &self.auth_token_file {
            return Ok(path.clone());
        }
        let config_dir = dirs::config_dir().ok_or_else(|| {
            crate::error::HoshikageError::ConfigError("Config directory not found".to_string())
        })?;
        Ok(config_dir.join("hoshikage").join("auth_tokens.json"))
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

impl RuntimeBackendKind {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "llama-server-managed" | "managed" | "server" => Ok(Self::LlamaServerManaged),
            "llama-ffi" | "ffi" => Ok(Self::LlamaFfi),
            other => Err(crate::error::HoshikageError::ConfigError(format!(
                "unsupported HOSHIKAGE_RUNTIME_BACKEND: {}",
                other
            ))),
        }
    }
}

impl UnknownFieldPolicy {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "compatible" => Ok(Self::Compatible),
            "strict" => Ok(Self::Strict),
            other => Err(crate::error::HoshikageError::ConfigError(format!(
                "unsupported RESPONSES_UNKNOWN_FIELD_POLICY: {other}"
            ))),
        }
    }
}

fn parse_value<T>(key: &str, value: &str) -> Result<T>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    value.parse().map_err(|error| {
        crate::error::HoshikageError::ConfigError(format!("invalid {key} value {value:?}: {error}"))
    })
}

fn load_dotenv(path: &std::path::Path) -> Result<()> {
    dotenvy::from_path(path).map_err(|error| {
        crate::error::HoshikageError::ConfigError(format!(
            "failed to load config file {}: {error}",
            path.display()
        ))
    })?;
    Ok(())
}

fn parse_env<T>(key: &str) -> Result<Option<T>>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    match std::env::var(key) {
        Ok(value) => parse_value(key, &value).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(crate::error::HoshikageError::ConfigError(format!(
            "invalid {key}: {error}"
        ))),
    }
}

fn apply_responses_environment(config: &mut Config) -> Result<()> {
    macro_rules! apply {
        ($key:literal, $field:ident) => {
            if let Some(value) = parse_env($key)? {
                config.$field = value;
            }
        };
    }
    apply!("HOSHIKAGE_MAX_REQUEST_BYTES", max_request_bytes);
    apply!("HOSHIKAGE_MAX_TOOL_SCHEMA_BYTES", max_tool_schema_bytes);
    apply!(
        "HOSHIKAGE_MAX_SINGLE_TOOL_SCHEMA_BYTES",
        max_single_tool_schema_bytes
    );
    apply!("HOSHIKAGE_MAX_TOOLS", max_tools);
    apply!("HOSHIKAGE_MAX_TOOL_ARGUMENT_BYTES", max_tool_argument_bytes);
    apply!("HOSHIKAGE_MAX_TOOL_RESULT_BYTES", max_tool_result_bytes);
    apply!(
        "HOSHIKAGE_RESPONSES_QUEUE_CAPACITY",
        responses_queue_capacity
    );
    apply!(
        "HOSHIKAGE_RESPONSES_QUEUE_TIMEOUT_MS",
        responses_queue_timeout_ms
    );
    apply!("HOSHIKAGE_RESPONSES_TIMEOUT_SECS", responses_timeout_secs);
    apply!(
        "HOSHIKAGE_FIRST_TOKEN_TIMEOUT_SECS",
        first_token_timeout_secs
    );
    apply!(
        "HOSHIKAGE_STREAM_IDLE_TIMEOUT_SECS",
        stream_idle_timeout_secs
    );
    apply!("HOSHIKAGE_GENERATION_TIMEOUT_SECS", generation_timeout_secs);
    if let Ok(path) = std::env::var("HOSHIKAGE_AUTH_TOKEN_FILE") {
        config.auth_token_file = Some(PathBuf::from(path));
    }
    if let Ok(value) = std::env::var("HOSHIKAGE_DEBUG_CAPTURE") {
        config.debug_capture = match value.as_str() {
            "1" | "true" | "on" => true,
            "0" | "false" | "off" => false,
            _ => {
                return Err(crate::error::HoshikageError::ConfigError(format!(
                    "invalid HOSHIKAGE_DEBUG_CAPTURE value {value:?}"
                )))
            }
        };
    }
    Ok(())
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn defaults_match_phase_zero_decisions() {
        let config = Config::default();
        assert_eq!(config.max_request_bytes, 8_388_608);
        assert_eq!(config.max_tool_result_bytes, 4_194_304);
        assert_eq!(config.responses_queue_capacity, 4);
        assert_eq!(config.responses_queue_timeout_ms, 30_000);
        assert_eq!(config.responses_timeout_secs, 900);
        assert_eq!(config.generation_timeout_secs, 600);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn invalid_numeric_values_are_not_silently_defaulted() {
        let error = parse_value::<u16>("PORT", "not-a-number").unwrap_err();
        assert!(matches!(
            error,
            crate::error::HoshikageError::ConfigError(_)
        ));
    }

    #[test]
    fn cross_field_limits_fail_closed() {
        let config = Config {
            max_single_tool_schema_bytes: 2_000,
            max_tool_schema_bytes: 1_000,
            ..Config::default()
        };
        assert!(config.validate().is_err());

        let config = Config {
            max_tool_result_bytes: 9_000,
            max_request_bytes: 8_000,
            ..Config::default()
        };
        assert!(config.validate().is_err());
    }
}
