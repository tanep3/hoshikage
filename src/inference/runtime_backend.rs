use crate::config::Config;
use crate::error::Result;
use crate::inference::llama_wrapper::InferenceParams;
use crate::inference::{
    LlamaWrapper, SpeculationAdapter, SpeculationAdapterMode, SpeculationSession,
    SpeculationSessionConfig, VisionRuntime,
};
use crate::model::{FallbackMode, SpeculationConfig, SpeculationMode, ThinkingConfig};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct LlamaLoadRequest {
    pub main_model: PathBuf,
    pub mmproj: Option<PathBuf>,
    pub draft_model: Option<PathBuf>,
    pub n_ctx: u32,
    pub n_gpu_layers: i32,
    pub n_rs_seq: u32,
    pub speculation: SpeculationConfig,
    pub thinking: ThinkingConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedRuntimeInfo {
    pub main_model: PathBuf,
    pub mmproj: Option<PathBuf>,
    pub draft_model: Option<PathBuf>,
    pub n_ctx: u32,
    pub n_gpu_layers: i32,
    pub vision_supported: bool,
    pub vision_marker: Option<String>,
    pub speculation_enabled: bool,
    pub speculation_mode: SpeculationMode,
    pub speculation_fallback_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeBackendStatus {
    pub loaded: bool,
    pub loaded_info: Option<LoadedRuntimeInfo>,
}

pub trait RuntimeBackend: Send {
    fn load(&mut self, request: &LlamaLoadRequest) -> Result<LoadedRuntimeInfo>;
    fn unload(&mut self);
    fn prepare_for_inference(&mut self) -> Result<()>;
    fn format_chat_prompt(&self, messages: &[crate::api::ChatMessage]) -> Result<String>;
    fn count_tokens(&self, text: &str) -> Result<i32>;
    fn is_diffusion_model(&self) -> Result<bool>;
    fn generate(&self, prompt: &str, params: &InferenceParams) -> Result<String>;
    fn generate_stream(
        &self,
        prompt: &str,
        params: &InferenceParams,
        on_token: &mut dyn FnMut(String) -> Result<()>,
    ) -> Result<String>;
    fn generate_with_diffusion(
        &self,
        prompt: &str,
        params: &InferenceParams,
    ) -> Result<(String, u32, u32)>;
    fn status(&self) -> RuntimeBackendStatus;
}

pub struct LlamaFfiBackend {
    wrapper: LlamaWrapper,
    libllama_path: PathBuf,
    base_config: Config,
    effective_config: Config,
    loaded_info: Option<LoadedRuntimeInfo>,
    vision: Option<VisionRuntime>,
    speculation_session: Option<SpeculationSession>,
    speculation_adapter: Option<SpeculationAdapter>,
}

impl LlamaFfiBackend {
    pub fn new(lib_path: PathBuf, config: Config) -> Result<Self> {
        let wrapper = LlamaWrapper::new(lib_path.clone())?;
        Ok(Self {
            wrapper,
            libllama_path: lib_path,
            base_config: config.clone(),
            effective_config: config,
            loaded_info: None,
            vision: None,
            speculation_session: None,
            speculation_adapter: None,
        })
    }

    fn config_for_request(&self, request: &LlamaLoadRequest) -> Config {
        let mut config = self.base_config.clone();
        config.n_ctx = request.n_ctx;
        config.n_gpu_layers = request.n_gpu_layers;
        config
    }
}

impl RuntimeBackend for LlamaFfiBackend {
    fn load(&mut self, request: &LlamaLoadRequest) -> Result<LoadedRuntimeInfo> {
        self.vision = None;
        self.speculation_session = None;
        self.speculation_adapter = SpeculationAdapter::load(&self.libllama_path)
            .map_err(|err| {
                tracing::debug!(error = %err, "Speculation adapter is not available");
                err
            })
            .ok();

        let config = self.config_for_request(request);
        let main_model = request.main_model.to_str().ok_or_else(|| {
            crate::error::HoshikageError::ModelLoadFailed("Invalid model path".to_string())
        })?;

        self.wrapper
            .load_model_with_runtime_options(main_model, &config, request.n_rs_seq)?;
        self.effective_config = config.clone();

        if let Some(mmproj_path) = request.mmproj.as_ref() {
            let text_model = self.wrapper.model_ptr()?;
            let vision = VisionRuntime::load(&self.libllama_path, mmproj_path, text_model)?;
            tracing::info!(
                mmproj = %vision.mmproj_path().display(),
                marker = vision.marker(),
                "Loaded Vision runtime"
            );
            self.vision = Some(vision);
        }

        let speculation_result = self.load_speculation_runtime(request, &config)?;

        let loaded_info = LoadedRuntimeInfo {
            main_model: request.main_model.clone(),
            mmproj: request.mmproj.clone(),
            draft_model: request.draft_model.clone(),
            n_ctx: request.n_ctx,
            n_gpu_layers: request.n_gpu_layers,
            vision_supported: self
                .vision
                .as_ref()
                .map(|vision| vision.supports_vision())
                .unwrap_or(false),
            vision_marker: self
                .vision
                .as_ref()
                .map(|vision| vision.marker().to_string()),
            speculation_enabled: speculation_result.enabled,
            speculation_mode: speculation_result.mode,
            speculation_fallback_reason: speculation_result.fallback_reason,
        };
        self.loaded_info = Some(loaded_info.clone());
        Ok(loaded_info)
    }

    fn unload(&mut self) {
        self.vision = None;
        self.speculation_session = None;
        self.speculation_adapter = None;
        self.wrapper.unload();
        self.loaded_info = None;
    }

    fn prepare_for_inference(&mut self) -> Result<()> {
        self.wrapper.prepare_for_inference(&self.effective_config)
    }

    fn format_chat_prompt(&self, messages: &[crate::api::ChatMessage]) -> Result<String> {
        self.wrapper.format_chat_prompt(messages)
    }

    fn count_tokens(&self, text: &str) -> Result<i32> {
        self.wrapper.count_tokens(text)
    }

    fn is_diffusion_model(&self) -> Result<bool> {
        self.wrapper.is_diffusion_model()
    }

    fn generate(&self, prompt: &str, params: &InferenceParams) -> Result<String> {
        if let Some(session) = self.speculation_session.as_ref() {
            return self
                .wrapper
                .generate_with_speculation(prompt, params, session);
        }
        self.wrapper.generate(prompt, params)
    }

    fn generate_stream(
        &self,
        prompt: &str,
        params: &InferenceParams,
        on_token: &mut dyn FnMut(String) -> Result<()>,
    ) -> Result<String> {
        if let Some(session) = self.speculation_session.as_ref() {
            return self
                .wrapper
                .generate_with_speculation_callback(prompt, params, session, on_token);
        }
        self.wrapper
            .generate_with_callback(prompt, params, on_token)
    }

    fn generate_with_diffusion(
        &self,
        prompt: &str,
        params: &InferenceParams,
    ) -> Result<(String, u32, u32)> {
        self.wrapper
            .generate_with_diffusion(prompt, params, &self.effective_config)
    }

    fn status(&self) -> RuntimeBackendStatus {
        RuntimeBackendStatus {
            loaded: self.wrapper.is_loaded(),
            loaded_info: self.loaded_info.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SpeculationLoadResult {
    enabled: bool,
    mode: SpeculationMode,
    fallback_reason: Option<String>,
}

impl LlamaFfiBackend {
    fn load_speculation_runtime(
        &mut self,
        request: &LlamaLoadRequest,
        config: &Config,
    ) -> Result<SpeculationLoadResult> {
        match request.speculation.primary_mode() {
            SpeculationMode::Off => Ok(SpeculationLoadResult {
                enabled: false,
                mode: SpeculationMode::Off,
                fallback_reason: None,
            }),
            SpeculationMode::Mtp => {
                let Some(adapter) = self.speculation_adapter.as_ref() else {
                    return Self::handle_speculation_unavailable(
                        &request.speculation,
                        SpeculationMode::Mtp,
                        "speculation_adapter_not_available".to_string(),
                    );
                };

                if !adapter.supports_mtp()? {
                    return Self::handle_speculation_unavailable(
                        &request.speculation,
                        SpeculationMode::Mtp,
                        "speculation_adapter_not_available".to_string(),
                    );
                }

                match adapter.init_session(
                    self.wrapper.model_ptr()?,
                    self.wrapper.context_ptr()?,
                    &SpeculationSessionConfig {
                        mode: SpeculationAdapterMode::Mtp,
                        n_draft_max: 16,
                        n_seq: 1,
                        n_ctx: config.n_ctx,
                        n_gpu_layers_draft: 0,
                        draft_model_path: request.draft_model.clone(),
                    },
                ) {
                    Ok(session) => {
                        self.speculation_session = Some(session);
                        Ok(SpeculationLoadResult {
                            enabled: true,
                            mode: SpeculationMode::Mtp,
                            fallback_reason: None,
                        })
                    }
                    Err(err) => Self::handle_speculation_unavailable(
                        &request.speculation,
                        SpeculationMode::Mtp,
                        format!("mtp_adapter_init_failed: {}", err),
                    ),
                }
            }
            SpeculationMode::DraftModel => {
                let adapter = self.speculation_adapter.as_ref();
                let adapter_supports_draft_model = adapter
                    .map(|adapter| adapter.supports_draft_model())
                    .transpose()?
                    .unwrap_or(false);

                if adapter_supports_draft_model {
                    Self::handle_speculation_unavailable(
                        &request.speculation,
                        SpeculationMode::DraftModel,
                        "draft_model_generation_loop_not_connected".to_string(),
                    )
                } else {
                    Self::handle_speculation_unavailable(
                        &request.speculation,
                        SpeculationMode::DraftModel,
                        "draft_model_adapter_not_available".to_string(),
                    )
                }
            }
        }
    }

    fn handle_speculation_unavailable(
        config: &SpeculationConfig,
        requested_mode: SpeculationMode,
        reason: String,
    ) -> Result<SpeculationLoadResult> {
        match config.fallback {
            FallbackMode::Strict => Err(crate::error::HoshikageError::InferenceError(format!(
                "Speculation unavailable for {:?}: {}",
                requested_mode, reason
            ))),
            FallbackMode::Warn => {
                tracing::warn!(
                    requested_mode = ?requested_mode,
                    reason = %reason,
                    "Speculation fallback to normal generation"
                );
                Ok(SpeculationLoadResult {
                    enabled: false,
                    mode: SpeculationMode::Off,
                    fallback_reason: Some(reason),
                })
            }
            FallbackMode::Off => Ok(SpeculationLoadResult {
                enabled: false,
                mode: SpeculationMode::Off,
                fallback_reason: None,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FallbackMode, SpeculationMode, ThinkingMode};

    #[test]
    fn load_request_preserves_bundle_fields() {
        let request = LlamaLoadRequest {
            main_model: PathBuf::from("/models/main.gguf"),
            mmproj: Some(PathBuf::from("/models/mmproj.gguf")),
            draft_model: Some(PathBuf::from("/models/drafter.gguf")),
            n_ctx: 8192,
            n_gpu_layers: -1,
            n_rs_seq: 16,
            speculation: SpeculationConfig {
                modes: vec![SpeculationMode::Mtp],
                fallback: FallbackMode::Warn,
            },
            thinking: ThinkingConfig {
                mode: ThinkingMode::Off,
            },
        };

        assert_eq!(request.main_model, PathBuf::from("/models/main.gguf"));
        assert_eq!(request.mmproj, Some(PathBuf::from("/models/mmproj.gguf")));
        assert_eq!(
            request.draft_model,
            Some(PathBuf::from("/models/drafter.gguf"))
        );
        assert_eq!(request.n_ctx, 8192);
        assert_eq!(request.n_gpu_layers, -1);
        assert!(request.speculation.has_mode(SpeculationMode::Mtp));
        assert_eq!(request.thinking.mode, ThinkingMode::Off);
    }

    #[test]
    fn strict_speculation_unavailable_returns_error() {
        let result = LlamaFfiBackend::handle_speculation_unavailable(
            &SpeculationConfig {
                modes: vec![SpeculationMode::Mtp],
                fallback: FallbackMode::Strict,
            },
            SpeculationMode::Mtp,
            "missing".to_string(),
        );

        assert!(result.is_err());
    }

    #[test]
    fn warn_speculation_unavailable_falls_back_to_normal_generation() {
        let result = LlamaFfiBackend::handle_speculation_unavailable(
            &SpeculationConfig {
                modes: vec![SpeculationMode::Mtp],
                fallback: FallbackMode::Warn,
            },
            SpeculationMode::Mtp,
            "missing".to_string(),
        )
        .unwrap();

        assert!(!result.enabled);
        assert_eq!(result.mode, SpeculationMode::Off);
        assert_eq!(result.fallback_reason.as_deref(), Some("missing"));
    }
}
