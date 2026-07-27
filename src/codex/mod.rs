use crate::model::ModelConfig;
use serde::Serialize;
use std::collections::HashMap;

pub const MINIMUM_CODEX_CONTEXT: u32 = 16_384;

pub fn effective_context_window(model: &ModelConfig, default_context_window: u32) -> u32 {
    model.n_ctx.unwrap_or(default_context_window)
}

pub fn is_codex_compatible(model: &ModelConfig, default_context_window: u32) -> bool {
    effective_context_window(model, default_context_window) >= MINIMUM_CODEX_CONTEXT
        && !model.model.is_empty()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexProfileMode {
    Interactive,
    Unattended,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexConfigOptions {
    pub model: String,
    pub mode: CodexProfileMode,
    pub base_url: String,
    pub authenticated: bool,
    pub default_context_window: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CodexModelCatalog {
    pub object: &'static str,
    pub data: Vec<CodexModelEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CodexModelEntry {
    pub id: String,
    pub context_window: u32,
    pub codex_compatible: bool,
    pub capabilities: CodexModelCapabilities,
    pub tool_calling: CodexToolCalling,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CodexModelCapabilities {
    pub responses: bool,
    pub streaming: bool,
    pub tools: bool,
    pub vision: bool,
    pub reasoning: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CodexToolCalling {
    pub mode: crate::model::ToolCallingMode,
    pub parser: crate::model::ToolParserId,
    pub fallback: crate::model::ToolFallback,
    pub strict: bool,
}

#[derive(Serialize)]
struct CodexConfigDocument<'a> {
    model: &'a str,
    model_provider: &'static str,
    approval_policy: &'static str,
    sandbox_mode: &'static str,
    model_context_window: u32,
    model_auto_compact_token_limit: u32,
    tool_output_token_limit: u32,
    model_reasoning_summary: &'static str,
    model_providers: CodexProviders<'a>,
}

#[derive(Serialize)]
struct CodexProviders<'a> {
    hoshikage: CodexProvider<'a>,
}

#[derive(Serialize)]
struct CodexProvider<'a> {
    name: &'static str,
    base_url: &'a str,
    wire_api: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    requires_openai_auth: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    env_key: Option<&'static str>,
    request_max_retries: u8,
    stream_max_retries: u8,
}

pub fn build_model_catalog(
    models: &HashMap<String, ModelConfig>,
    default_context_window: u32,
) -> CodexModelCatalog {
    let mut data = models
        .iter()
        .map(|(id, model)| {
            let context_window = effective_context_window(model, default_context_window);
            CodexModelEntry {
                id: id.clone(),
                context_window,
                codex_compatible: is_codex_compatible(model, default_context_window),
                capabilities: CodexModelCapabilities {
                    responses: true,
                    streaming: true,
                    tools: model.tool_calling.mode != crate::model::ToolCallingMode::Disabled,
                    vision: model.mmproj.is_some(),
                    reasoning: false,
                },
                tool_calling: CodexToolCalling {
                    mode: model.tool_calling.mode,
                    parser: model.tool_calling.effective_parser(),
                    fallback: model.tool_calling.fallback,
                    strict: model.tool_calling.strict,
                },
            }
        })
        .collect::<Vec<_>>();
    data.sort_by(|left, right| left.id.cmp(&right.id));
    CodexModelCatalog {
        object: "codex_model_catalog",
        data,
    }
}

pub fn render_codex_config(
    model: &ModelConfig,
    options: &CodexConfigOptions,
) -> crate::Result<String> {
    if options.model.is_empty() {
        return Err(crate::error::HoshikageError::ConfigError(
            "Codex model ID must not be empty".to_string(),
        ));
    }
    let url = reqwest::Url::parse(&options.base_url).map_err(|_| {
        crate::error::HoshikageError::ConfigError(
            "Codex provider base URL must be an absolute HTTP URL".to_string(),
        )
    })?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(crate::error::HoshikageError::ConfigError(
            "Codex provider base URL must be an absolute HTTP URL".to_string(),
        ));
    }
    let context_window = effective_context_window(model, options.default_context_window);
    if context_window < MINIMUM_CODEX_CONTEXT {
        return Err(crate::error::HoshikageError::ConfigError(format!(
            "Model context window {context_window} is below the Codex minimum {MINIMUM_CODEX_CONTEXT}"
        )));
    }
    let approval_policy = match options.mode {
        CodexProfileMode::Interactive => "on-request",
        CodexProfileMode::Unattended => "never",
    };
    let document = CodexConfigDocument {
        model: &options.model,
        model_provider: "hoshikage",
        approval_policy,
        sandbox_mode: "workspace-write",
        model_context_window: context_window,
        model_auto_compact_token_limit: context_window.saturating_mul(3) / 4,
        tool_output_token_limit: (context_window / 4).min(8_192),
        model_reasoning_summary: "none",
        model_providers: CodexProviders {
            hoshikage: CodexProvider {
                name: "Hoshikage",
                base_url: &options.base_url,
                wire_api: "responses",
                requires_openai_auth: (!options.authenticated).then_some(false),
                env_key: options.authenticated.then_some("HOSHIKAGE_API_KEY"),
                request_max_retries: 1,
                stream_max_retries: 1,
            },
        },
    };
    toml::to_string_pretty(&document)
        .map_err(|error| crate::error::HoshikageError::ConfigError(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        SpeculationConfig, ThinkingConfig, ToolCallingConfig, ToolCallingMode, ToolFallback,
        ToolParserId,
    };

    fn model(context: Option<u32>, tools: ToolCallingMode, vision: bool) -> ModelConfig {
        ModelConfig {
            path: "/private/models".to_string(),
            model: "secret.gguf".to_string(),
            stop: Vec::new(),
            mmproj: vision.then(|| "mmproj.gguf".to_string()),
            drafter: None,
            speculation: SpeculationConfig::default(),
            thinking: ThinkingConfig::default(),
            tool_calling: ToolCallingConfig {
                mode: tools,
                parser: (tools == ToolCallingMode::Native)
                    .then_some(ToolParserId::LlamaServerNative),
                fallback: ToolFallback::Json,
                strict: true,
                repair_invalid_json: true,
                max_argument_bytes: 65_536,
                result_policy: crate::model::ToolResultPolicy::Reject,
            },
            n_ctx: context,
            n_gpu_layers: Some(99),
        }
    }

    #[test]
    fn catalog_is_sorted_and_exposes_capabilities_without_model_paths() {
        let models = HashMap::from([
            (
                "zeta".to_string(),
                model(Some(8_192), ToolCallingMode::Disabled, false),
            ),
            (
                "alpha".to_string(),
                model(Some(16_384), ToolCallingMode::Native, true),
            ),
        ]);

        let catalog = build_model_catalog(&models, 4_096);
        assert_eq!(
            catalog
                .data
                .iter()
                .map(|entry| entry.id.as_str())
                .collect::<Vec<_>>(),
            ["alpha", "zeta"]
        );
        assert!(catalog.data[0].codex_compatible);
        assert!(catalog.data[0].capabilities.tools);
        assert!(catalog.data[0].capabilities.vision);
        assert!(!catalog.data[1].codex_compatible);
        assert!(!serde_json::to_string(&catalog)
            .unwrap()
            .contains("/private/models"));
    }

    #[test]
    fn interactive_config_uses_bundle_limits_and_local_provider() {
        let output = render_codex_config(
            &model(Some(16_384), ToolCallingMode::Native, false),
            &CodexConfigOptions {
                model: "gemma4".to_string(),
                mode: CodexProfileMode::Interactive,
                base_url: "http://127.0.0.1:3030/v1".to_string(),
                authenticated: false,
                default_context_window: 4_096,
            },
        )
        .unwrap();

        assert!(output.contains("model = \"gemma4\""));
        assert!(output.contains("approval_policy = \"on-request\""));
        assert!(output.contains("model_context_window = 16384"));
        assert!(output.contains("model_auto_compact_token_limit = 12288"));
        assert!(output.contains("tool_output_token_limit = 4096"));
        assert!(output.contains("wire_api = \"responses\""));
        assert!(output.contains("requires_openai_auth = false"));
        assert!(!output.contains("env_key"));
    }

    #[test]
    fn unattended_authenticated_config_is_explicit_and_does_not_write_any_file() {
        let output = render_codex_config(
            &model(None, ToolCallingMode::Native, false),
            &CodexConfigOptions {
                model: "lan-model".to_string(),
                mode: CodexProfileMode::Unattended,
                base_url: "http://192.168.1.10:3030/v1".to_string(),
                authenticated: true,
                default_context_window: 32_768,
            },
        )
        .unwrap();

        assert!(output.contains("approval_policy = \"never\""));
        assert!(output.contains("sandbox_mode = \"workspace-write\""));
        assert!(output.contains("model_context_window = 32768"));
        assert!(output.contains("env_key = \"HOSHIKAGE_API_KEY\""));
        assert!(!output.contains("requires_openai_auth"));
    }

    #[test]
    fn config_rejects_context_below_codex_minimum() {
        let error = render_codex_config(
            &model(Some(8_192), ToolCallingMode::Native, false),
            &CodexConfigOptions {
                model: "small".to_string(),
                mode: CodexProfileMode::Interactive,
                base_url: "http://127.0.0.1:3030/v1".to_string(),
                authenticated: false,
                default_context_window: 4_096,
            },
        )
        .err()
        .unwrap();

        assert!(matches!(
            error,
            crate::error::HoshikageError::ConfigError(_)
        ));
    }
}
