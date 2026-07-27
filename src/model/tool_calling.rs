use serde::{Deserialize, Serialize};

const DEFAULT_MAX_ARGUMENT_BYTES: usize = 65_536;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallingMode {
    Native,
    Json,
    #[default]
    Disabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolParserId {
    LlamaServerNative,
    GenericJson,
    Qwen,
    Llama,
    Mistral,
    Hermes,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolFallback {
    None,
    #[default]
    Json,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ToolResultPolicy {
    #[default]
    Reject,
    HeadTail {
        max_bytes: usize,
        head_bytes: usize,
        tail_bytes: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallingConfig {
    #[serde(default)]
    pub mode: ToolCallingMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parser: Option<ToolParserId>,
    #[serde(default)]
    pub fallback: ToolFallback,
    #[serde(default = "default_true")]
    pub strict: bool,
    #[serde(default = "default_true")]
    pub repair_invalid_json: bool,
    #[serde(default = "default_max_argument_bytes")]
    pub max_argument_bytes: usize,
    #[serde(default)]
    pub result_policy: ToolResultPolicy,
}

impl Default for ToolCallingConfig {
    fn default() -> Self {
        Self {
            mode: ToolCallingMode::Disabled,
            parser: None,
            fallback: ToolFallback::Json,
            strict: true,
            repair_invalid_json: true,
            max_argument_bytes: DEFAULT_MAX_ARGUMENT_BYTES,
            result_policy: ToolResultPolicy::Reject,
        }
    }
}

impl ToolCallingConfig {
    pub fn is_disabled_default(&self) -> bool {
        self == &Self::default()
    }

    pub fn effective_parser(&self) -> ToolParserId {
        self.parser.unwrap_or(match self.mode {
            ToolCallingMode::Native => ToolParserId::LlamaServerNative,
            ToolCallingMode::Json | ToolCallingMode::Disabled => ToolParserId::GenericJson,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.max_argument_bytes == 0 {
            return Err("tool_calling.max_argument_bytes must be greater than zero".to_string());
        }
        match (self.mode, self.effective_parser()) {
            (ToolCallingMode::Native, ToolParserId::LlamaServerNative)
            | (ToolCallingMode::Json, ToolParserId::GenericJson)
            | (ToolCallingMode::Disabled, ToolParserId::GenericJson) => {}
            _ => {
                return Err(
                    "tool_calling parser is not implemented for the configured mode".to_string(),
                );
            }
        }
        if let ToolResultPolicy::HeadTail {
            max_bytes,
            head_bytes,
            tail_bytes,
        } = &self.result_policy
        {
            if *max_bytes == 0 || head_bytes.saturating_add(*tail_bytes) > *max_bytes {
                return Err("tool_calling head_tail byte limits are invalid".to_string());
            }
        }
        Ok(())
    }
}

const fn default_true() -> bool {
    true
}

const fn default_max_argument_bytes() -> usize {
    DEFAULT_MAX_ARGUMENT_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_config_is_disabled_and_omitted_when_serialized() {
        #[derive(Deserialize, Serialize)]
        struct Bundle {
            #[serde(
                default,
                skip_serializing_if = "ToolCallingConfig::is_disabled_default"
            )]
            tool_calling: ToolCallingConfig,
        }

        let bundle: Bundle = serde_json::from_str("{}").unwrap();
        assert_eq!(bundle.tool_calling.mode, ToolCallingMode::Disabled);
        assert_eq!(serde_json::to_value(bundle).unwrap(), serde_json::json!({}));
    }

    #[test]
    fn native_and_json_modes_resolve_default_parsers() {
        let native = ToolCallingConfig {
            mode: ToolCallingMode::Native,
            ..ToolCallingConfig::default()
        };
        let json = ToolCallingConfig {
            mode: ToolCallingMode::Json,
            ..ToolCallingConfig::default()
        };

        assert_eq!(native.effective_parser(), ToolParserId::LlamaServerNative);
        assert_eq!(json.effective_parser(), ToolParserId::GenericJson);
    }

    #[test]
    fn unknown_parser_is_rejected_during_deserialization() {
        let error = serde_json::from_value::<ToolCallingConfig>(serde_json::json!({
            "mode": "native",
            "parser": "load-arbitrary-library"
        }))
        .err()
        .unwrap();

        assert!(error.to_string().contains("unknown variant"));
    }

    #[test]
    fn mode_parser_mismatch_and_invalid_head_tail_are_rejected() {
        let mismatch = ToolCallingConfig {
            mode: ToolCallingMode::Native,
            parser: Some(ToolParserId::GenericJson),
            ..ToolCallingConfig::default()
        };
        assert!(mismatch.validate().is_err());

        let invalid_policy = ToolCallingConfig {
            result_policy: ToolResultPolicy::HeadTail {
                max_bytes: 8,
                head_bytes: 6,
                tail_bytes: 6,
            },
            ..ToolCallingConfig::default()
        };
        assert!(invalid_policy.validate().is_err());
    }
}
