use crate::conversation::{Conversation, ToolName};
use serde_json::Value;

#[derive(Clone, Default, PartialEq)]
pub struct ModelToolSet {
    tools: Vec<ModelTool>,
}

impl ModelToolSet {
    pub fn new(tools: Vec<ModelTool>) -> Self {
        Self { tools }
    }

    pub fn tools(&self) -> &[ModelTool] {
        &self.tools
    }
}

#[derive(Clone, PartialEq)]
pub struct ModelTool {
    pub name: ToolName,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function(ToolName),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
}

#[derive(Clone, PartialEq)]
pub struct ModelRequest {
    pub conversation: Conversation,
    pub tools: ModelToolSet,
    pub tool_choice: ToolChoice,
    pub sampling: SamplingOptions,
    pub max_output_tokens: u32,
    pub stream: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenUsage {
    Measured {
        input_tokens: u32,
        output_tokens: u32,
    },
    Estimated {
        input_tokens: u32,
        output_tokens: u32,
    },
}

#[derive(Clone, PartialEq)]
pub struct RawModelToolCall {
    pub name: ToolName,
    pub arguments: String,
}

#[derive(Clone, PartialEq)]
pub enum ModelCompletion {
    Text {
        content: String,
        usage: TokenUsage,
    },
    ToolCall {
        call: RawModelToolCall,
        usage: TokenUsage,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFinishReason {
    Stop,
    ToolCall,
    Length,
}

#[derive(Clone, PartialEq)]
pub enum ModelDelta {
    Text(String),
    ToolCallStarted { index: usize },
    ToolName(String),
    ToolArguments(String),
    ToolCallFinished,
    Usage(TokenUsage),
    Finished(ModelFinishReason),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{ConversationItem, Message, Role};

    #[test]
    fn model_request_contains_no_wire_or_api_types() {
        let request = ModelRequest {
            conversation: Conversation::new(vec![ConversationItem::Message(
                Message::text(Role::User, "hello").unwrap(),
            )]),
            tools: ModelToolSet::default(),
            tool_choice: ToolChoice::Auto,
            sampling: SamplingOptions::default(),
            max_output_tokens: 128,
            stream: false,
        };

        assert_eq!(request.conversation.summary().messages, 1);
        assert!(request.tools.tools().is_empty());
    }
}
