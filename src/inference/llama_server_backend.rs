use crate::inference::LlamaLoadRequest;
use crate::model::{SpeculationMode, ThinkingMode};
use crate::Result;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

#[derive(Debug, Clone)]
pub struct LlamaServerLaunchConfig {
    pub server_path: PathBuf,
    pub host: String,
    pub port: u16,
    pub alias: String,
    pub log_file: Option<PathBuf>,
    pub sleep_idle_secs: Option<u64>,
    pub cache_type_k: Option<crate::config::KvCacheType>,
    pub cache_type_v: Option<crate::config::KvCacheType>,
    pub request: LlamaLoadRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaServerCommandSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
}

pub struct LlamaServerProcess {
    child: Child,
    spec: LlamaServerCommandSpec,
}

impl LlamaServerCommandSpec {
    pub fn from_launch_config(config: &LlamaServerLaunchConfig) -> Self {
        let mut args = vec![
            "-m".to_string(),
            config.request.main_model.display().to_string(),
            "--host".to_string(),
            config.host.clone(),
            "--port".to_string(),
            config.port.to_string(),
            "--alias".to_string(),
            config.alias.clone(),
            "-c".to_string(),
            config.request.n_ctx.to_string(),
            "-np".to_string(),
            "1".to_string(),
            "-ngl".to_string(),
            config.request.n_gpu_layers.to_string(),
        ];

        if let Some(mmproj) = config.request.mmproj.as_ref() {
            args.push("--mmproj".to_string());
            args.push(mmproj.display().to_string());
        }

        let spec_types = speculation_types(&config.request.speculation.modes);
        if !spec_types.is_empty() {
            args.push("--spec-type".to_string());
            args.push(spec_types.join(","));
            args.push("-fa".to_string());
            args.push("on".to_string());
            args.push("--spec-draft-p-min".to_string());
            args.push("0.10".to_string());
            if let Some(draft_n_max) = config.request.speculation.draft_n_max {
                args.push("--spec-draft-n-max".to_string());
                args.push(draft_n_max.get().to_string());
            }
        }

        if let Some(draft_model) = config.request.draft_model.as_ref() {
            args.push("--model-draft".to_string());
            args.push(draft_model.display().to_string());
            args.push("--n-gpu-layers-draft".to_string());
            args.push("99".to_string());
        }

        if config.request.thinking.mode == ThinkingMode::Off {
            args.push("--reasoning".to_string());
            args.push("off".to_string());
            args.push("--reasoning-budget".to_string());
            args.push("0".to_string());
        }

        if let Some(log_file) = config.log_file.as_ref() {
            args.push("--log-file".to_string());
            args.push(log_file.display().to_string());
        }

        if let Some(sleep_idle_secs) = config.sleep_idle_secs {
            args.push("--sleep-idle-seconds".to_string());
            args.push(sleep_idle_secs.to_string());
        }
        let cache_type_k = config
            .request
            .llama_server
            .cache_type_k
            .or(config.cache_type_k);
        let cache_type_v = config
            .request
            .llama_server
            .cache_type_v
            .or(config.cache_type_v);
        append_cache_type(&mut args, "--cache-type-k", cache_type_k);
        append_cache_type(&mut args, "--cache-type-v", cache_type_v);
        if config.request.draft_model.is_some() {
            append_cache_type(&mut args, "--cache-type-k-draft", cache_type_k);
            append_cache_type(&mut args, "--cache-type-v-draft", cache_type_v);
        }

        Self {
            program: config.server_path.clone(),
            args,
        }
    }
}

fn append_cache_type(
    args: &mut Vec<String>,
    option: &str,
    cache_type: Option<crate::config::KvCacheType>,
) {
    if let Some(cache_type) = cache_type {
        args.push(option.to_string());
        args.push(cache_type.as_str().to_string());
    }
}

impl LlamaServerProcess {
    pub fn start(spec: LlamaServerCommandSpec) -> Result<Self> {
        if !spec.program.is_file() {
            return Err(crate::error::HoshikageError::ConfigError(format!(
                "llama-server not found: {}",
                spec.program.display()
            )));
        }

        let child = Command::new(&spec.program)
            .args(&spec.args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|err| {
                crate::error::HoshikageError::ModelLoadFailed(format!(
                    "failed to start llama-server {}: {}",
                    spec.program.display(),
                    err
                ))
            })?;

        Ok(Self { child, spec })
    }

    pub fn command_spec(&self) -> &LlamaServerCommandSpec {
        &self.spec
    }

    pub fn is_running(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(Some(_status)) => false,
            Ok(None) => true,
            Err(_) => false,
        }
    }

    pub fn stop(&mut self) {
        if self.is_running() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

impl Drop for LlamaServerProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

fn speculation_types(modes: &[SpeculationMode]) -> Vec<&'static str> {
    let mut types = Vec::new();
    if modes.iter().any(|mode| mode == &SpeculationMode::Mtp) {
        types.push("draft-mtp");
    }
    if modes
        .iter()
        .any(|mode| mode == &SpeculationMode::DraftModel)
    {
        types.push("draft-simple");
    }
    types
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FallbackMode, SpeculationConfig, ThinkingConfig};

    fn launch_config(request: LlamaLoadRequest) -> LlamaServerLaunchConfig {
        LlamaServerLaunchConfig {
            server_path: PathBuf::from("/runtime/llama-server"),
            host: "127.0.0.1".to_string(),
            port: 41001,
            alias: "gemma4".to_string(),
            log_file: None,
            sleep_idle_secs: None,
            cache_type_k: None,
            cache_type_v: None,
            request,
        }
    }

    fn base_request() -> LlamaLoadRequest {
        LlamaLoadRequest {
            main_model: PathBuf::from("/models/main.gguf"),
            mmproj: None,
            draft_model: None,
            n_ctx: 8192,
            n_gpu_layers: -1,
            n_rs_seq: 0,
            speculation: SpeculationConfig::default(),
            thinking: ThinkingConfig::default(),
            llama_server: crate::model::LlamaServerModelConfig::default(),
        }
    }

    #[test]
    fn command_spec_includes_base_server_options() {
        let spec = LlamaServerCommandSpec::from_launch_config(&launch_config(base_request()));

        assert_eq!(spec.program, PathBuf::from("/runtime/llama-server"));
        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["-m", "/models/main.gguf"]));
        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["--host", "127.0.0.1"]));
        assert!(spec.args.windows(2).any(|pair| pair == ["--port", "41001"]));
        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["--alias", "gemma4"]));
        assert!(spec.args.windows(2).any(|pair| pair == ["-c", "8192"]));
        assert!(spec.args.windows(2).any(|pair| pair == ["-np", "1"]));
        assert!(spec.args.windows(2).any(|pair| pair == ["-ngl", "-1"]));
    }

    #[test]
    fn command_spec_includes_log_file_when_configured() {
        let mut config = launch_config(base_request());
        config.log_file = Some(PathBuf::from("/logs/llama-server.log"));

        let spec = LlamaServerCommandSpec::from_launch_config(&config);

        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["--log-file", "/logs/llama-server.log"]));
    }

    #[test]
    fn command_spec_includes_sleep_idle_when_configured() {
        let mut config = launch_config(base_request());
        config.sleep_idle_secs = Some(2);

        let spec = LlamaServerCommandSpec::from_launch_config(&config);

        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["--sleep-idle-seconds", "2"]));
    }

    #[test]
    fn command_spec_includes_quantized_kv_cache_for_main_and_draft_contexts() {
        let mut config = launch_config(base_request());
        config.cache_type_k = Some(crate::config::KvCacheType::Q8Zero);
        config.cache_type_v = Some(crate::config::KvCacheType::Q4Zero);
        config.request.draft_model = Some(PathBuf::from("/models/draft.gguf"));

        let spec = LlamaServerCommandSpec::from_launch_config(&config);

        for expected in [
            ["--cache-type-k", "q8_0"],
            ["--cache-type-v", "q4_0"],
            ["--cache-type-k-draft", "q8_0"],
            ["--cache-type-v-draft", "q4_0"],
        ] {
            assert!(spec.args.windows(2).any(|pair| pair == expected));
        }
    }

    #[test]
    fn bundle_cache_settings_override_global_defaults() {
        let mut config = launch_config(base_request());
        config.cache_type_k = Some(crate::config::KvCacheType::F16);
        config.cache_type_v = Some(crate::config::KvCacheType::F16);
        config.request.llama_server.cache_type_k = Some(crate::config::KvCacheType::Q8Zero);
        config.request.llama_server.cache_type_v = Some(crate::config::KvCacheType::Q4Zero);

        let spec = LlamaServerCommandSpec::from_launch_config(&config);

        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["--cache-type-k", "q8_0"]));
        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["--cache-type-v", "q4_0"]));
        assert!(!spec
            .args
            .windows(2)
            .any(|pair| pair == ["--cache-type-k", "f16"]));
        assert!(!spec
            .args
            .windows(2)
            .any(|pair| pair == ["--cache-type-v", "f16"]));
    }

    #[test]
    fn command_spec_includes_vision_speculation_and_thinking_options() {
        let mut request = base_request();
        request.mmproj = Some(PathBuf::from("/models/mmproj.gguf"));
        request.draft_model = Some(PathBuf::from("/models/draft.gguf"));
        request.speculation = SpeculationConfig {
            modes: vec![SpeculationMode::Mtp, SpeculationMode::DraftModel],
            draft_n_max: std::num::NonZeroU32::new(6),
            fallback: FallbackMode::Warn,
        };
        request.thinking = ThinkingConfig {
            mode: ThinkingMode::Off,
            ..ThinkingConfig::default()
        };

        let spec = LlamaServerCommandSpec::from_launch_config(&launch_config(request));

        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["--mmproj", "/models/mmproj.gguf"]));
        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["--spec-type", "draft-mtp,draft-simple"]));
        assert!(spec.args.windows(2).any(|pair| pair == ["-fa", "on"]));
        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["--spec-draft-p-min", "0.10"]));
        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["--spec-draft-n-max", "6"]));
        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["--model-draft", "/models/draft.gguf"]));
        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["--n-gpu-layers-draft", "99"]));
        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["--reasoning", "off"]));
        assert!(spec
            .args
            .windows(2)
            .any(|pair| pair == ["--reasoning-budget", "0"]));
    }

    #[test]
    fn process_start_rejects_missing_server() {
        let spec = LlamaServerCommandSpec {
            program: PathBuf::from("/missing/llama-server"),
            args: Vec::new(),
        };

        let result = LlamaServerProcess::start(spec);

        assert!(result.is_err());
    }
}
