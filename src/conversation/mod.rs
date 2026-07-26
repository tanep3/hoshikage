use serde_json::Value;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConversationError {
    #[error("{kind} must not be empty")]
    EmptyIdentifier { kind: &'static str },
    #[error("{kind} exceeds {max} bytes")]
    IdentifierTooLong { kind: &'static str, max: usize },
    #[error("{kind} contains an unsupported character")]
    InvalidIdentifierCharacter { kind: &'static str },
    #[error("message content must not be empty")]
    EmptyMessage,
    #[error("function call id is duplicated: {0}")]
    DuplicateCallId(String),
    #[error("function call output has no preceding call: {0}")]
    OrphanCallOutput(String),
    #[error("function call has multiple outputs: {0}")]
    DuplicateCallOutput(String),
    #[error("tool arguments are not valid JSON: {0}")]
    InvalidToolArguments(String),
}

fn validate_identifier(
    kind: &'static str,
    value: String,
    max: usize,
) -> Result<String, ConversationError> {
    if value.is_empty() {
        return Err(ConversationError::EmptyIdentifier { kind });
    }
    if value.len() > max {
        return Err(ConversationError::IdentifierTooLong { kind, max });
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(ConversationError::InvalidIdentifierCharacter { kind });
    }
    Ok(value)
}

macro_rules! identifier {
    ($name:ident, $kind:literal, $max:literal) => {
        #[derive(Clone, PartialEq, Eq, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, ConversationError> {
                validate_identifier($kind, value.into(), $max).map(Self)
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter
                    .debug_tuple(stringify!($name))
                    .field(&self.0)
                    .finish()
            }
        }
    };
}

identifier!(ModelId, "model id", 128);
identifier!(ResponseId, "response id", 128);
identifier!(OutputItemId, "output item id", 128);
identifier!(CallId, "call id", 128);
identifier!(ToolName, "tool name", 64);
identifier!(RequestId, "request id", 128);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }

    pub fn parse(value: &str) -> Result<Self, ConversationError> {
        match value {
            "system" => Ok(Self::System),
            "developer" => Ok(Self::Developer),
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            _ => Err(ConversationError::InvalidIdentifierCharacter { kind: "role" }),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct ImageInput {
    pub source: String,
    pub detail: Option<String>,
}

#[derive(Clone, PartialEq, Eq)]
pub enum ContentPart {
    Text(String),
    Image(ImageInput),
}

#[derive(Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
}

impl Message {
    pub fn new(role: Role, content: Vec<ContentPart>) -> Result<Self, ConversationError> {
        if content.is_empty() {
            return Err(ConversationError::EmptyMessage);
        }
        Ok(Self { role, content })
    }

    pub fn text(role: Role, content: impl Into<String>) -> Result<Self, ConversationError> {
        Self::new(role, vec![ContentPart::Text(content.into())])
    }

    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text(text) => Some(text.as_str()),
                ContentPart::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Clone, PartialEq)]
pub struct ToolArguments {
    canonical_json: String,
    value: Value,
}

impl ToolArguments {
    pub fn parse(arguments: &str) -> Result<Self, ConversationError> {
        let value: Value = serde_json::from_str(arguments)
            .map_err(|error| ConversationError::InvalidToolArguments(error.to_string()))?;
        Self::from_value(value)
    }

    pub fn from_value(value: Value) -> Result<Self, ConversationError> {
        let canonical_json = serde_json::to_string(&value)
            .map_err(|error| ConversationError::InvalidToolArguments(error.to_string()))?;
        Ok(Self {
            canonical_json,
            value,
        })
    }

    pub fn canonical_json(&self) -> &str {
        &self.canonical_json
    }

    pub fn value(&self) -> &Value {
        &self.value
    }
}

#[derive(Clone, PartialEq)]
pub struct FunctionCall {
    pub call_id: CallId,
    pub name: ToolName,
    pub arguments: ToolArguments,
}

#[derive(Clone, PartialEq, Eq)]
pub enum ToolOutcome {
    Success(String),
    Failure(String),
    Rejected(String),
    Cancelled(String),
}

#[derive(Clone, PartialEq, Eq)]
pub struct FunctionCallOutput {
    pub call_id: CallId,
    pub outcome: ToolOutcome,
}

#[derive(Clone, PartialEq)]
pub enum ConversationItem {
    Message(Message),
    FunctionCall(FunctionCall),
    FunctionCallOutput(FunctionCallOutput),
}

#[derive(Clone, Default, PartialEq)]
pub struct Conversation {
    items: Vec<ConversationItem>,
}

impl Conversation {
    pub fn new(items: Vec<ConversationItem>) -> Self {
        Self { items }
    }

    pub fn items(&self) -> &[ConversationItem] {
        &self.items
    }

    pub fn validate(&self) -> Result<ConversationIndex<'_>, ConversationError> {
        let mut calls = HashMap::new();
        let mut outputs = HashSet::new();

        for item in &self.items {
            match item {
                ConversationItem::Message(message) if message.content.is_empty() => {
                    return Err(ConversationError::EmptyMessage);
                }
                ConversationItem::FunctionCall(call) => {
                    if calls.insert(&call.call_id, call).is_some() {
                        return Err(ConversationError::DuplicateCallId(
                            call.call_id.as_str().to_string(),
                        ));
                    }
                }
                ConversationItem::FunctionCallOutput(output) => {
                    if !calls.contains_key(&output.call_id) {
                        return Err(ConversationError::OrphanCallOutput(
                            output.call_id.as_str().to_string(),
                        ));
                    }
                    if !outputs.insert(&output.call_id) {
                        return Err(ConversationError::DuplicateCallOutput(
                            output.call_id.as_str().to_string(),
                        ));
                    }
                }
                ConversationItem::Message(_) => {}
            }
        }

        Ok(ConversationIndex { calls })
    }

    pub fn summary(&self) -> ConversationSummary {
        let mut summary = ConversationSummary::default();
        for item in &self.items {
            match item {
                ConversationItem::Message(_) => summary.messages += 1,
                ConversationItem::FunctionCall(_) => summary.function_calls += 1,
                ConversationItem::FunctionCallOutput(_) => summary.function_outputs += 1,
            }
        }
        summary
    }
}

pub struct ConversationIndex<'a> {
    calls: HashMap<&'a CallId, &'a FunctionCall>,
}

impl<'a> ConversationIndex<'a> {
    pub fn call(&self, call_id: &CallId) -> Option<&'a FunctionCall> {
        self.calls.get(call_id).copied()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConversationSummary {
    pub messages: usize,
    pub function_calls: usize,
    pub function_outputs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(call_id: &str) -> FunctionCall {
        FunctionCall {
            call_id: CallId::new(call_id).unwrap(),
            name: ToolName::new("read_file").unwrap(),
            arguments: ToolArguments::parse(r#"{"path":"README.md"}"#).unwrap(),
        }
    }

    #[test]
    fn tool_arguments_are_canonicalized_and_invalid_json_is_rejected() {
        let arguments = ToolArguments::parse(r#"{ "path": "README.md" }"#).unwrap();
        assert_eq!(arguments.canonical_json(), r#"{"path":"README.md"}"#);
        assert!(ToolArguments::parse(r#"{"path":"README.md""#).is_err());
    }

    #[test]
    fn validation_builds_call_index_without_exposing_payloads() {
        let call = call("call_1");
        let call_id = call.call_id.clone();
        let conversation = Conversation::new(vec![
            ConversationItem::Message(Message::text(Role::User, "Read the file").unwrap()),
            ConversationItem::FunctionCall(call),
            ConversationItem::FunctionCallOutput(FunctionCallOutput {
                call_id: call_id.clone(),
                outcome: ToolOutcome::Success("secret file body".to_string()),
            }),
        ]);

        let index = conversation.validate().unwrap();
        assert_eq!(index.call(&call_id).unwrap().name.as_str(), "read_file");
        assert_eq!(
            conversation.summary(),
            ConversationSummary {
                messages: 1,
                function_calls: 1,
                function_outputs: 1,
            }
        );
    }

    #[test]
    fn validation_rejects_orphan_duplicate_and_repeated_outputs() {
        let orphan_id = CallId::new("call_orphan").unwrap();
        let orphan = Conversation::new(vec![ConversationItem::FunctionCallOutput(
            FunctionCallOutput {
                call_id: orphan_id,
                outcome: ToolOutcome::Success("output".to_string()),
            },
        )]);
        assert!(matches!(
            orphan.validate(),
            Err(ConversationError::OrphanCallOutput(_))
        ));

        let duplicate = Conversation::new(vec![
            ConversationItem::FunctionCall(call("call_1")),
            ConversationItem::FunctionCall(call("call_1")),
        ]);
        assert!(matches!(
            duplicate.validate(),
            Err(ConversationError::DuplicateCallId(_))
        ));

        let call = call("call_2");
        let output = FunctionCallOutput {
            call_id: call.call_id.clone(),
            outcome: ToolOutcome::Success("output".to_string()),
        };
        let repeated = Conversation::new(vec![
            ConversationItem::FunctionCall(call),
            ConversationItem::FunctionCallOutput(output.clone()),
            ConversationItem::FunctionCallOutput(output),
        ]);
        assert!(matches!(
            repeated.validate(),
            Err(ConversationError::DuplicateCallOutput(_))
        ));
    }

    #[test]
    fn identifiers_reject_empty_long_and_unsafe_values() {
        assert!(ModelId::new("").is_err());
        assert!(ToolName::new("bad tool").is_err());
        assert!(CallId::new("x".repeat(129)).is_err());
        assert_eq!(
            ToolName::new("multi_agent.spawn").unwrap().as_str(),
            "multi_agent.spawn"
        );
    }
}
