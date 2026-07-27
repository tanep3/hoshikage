use super::{ModelDelta, ModelFinishReason, ModelToolSet, TokenUsage, ToolChoice};
use crate::conversation::{ToolArguments, ToolName};

#[derive(Clone, Debug, PartialEq)]
pub enum ModelStreamAction {
    BeginText,
    AppendText(String),
    FinishText,
    BeginFunctionCall { name: ToolName },
    AppendArguments(String),
    FinishFunctionCall,
    Complete { usage: TokenUsage },
}

enum Classification {
    Unknown,
    Text,
    FunctionCall,
}

pub struct NativeStreamStrategy {
    tools: ModelToolSet,
    tool_choice: ToolChoice,
    strict: bool,
    max_argument_bytes: usize,
    classification: Classification,
    name: String,
    arguments: String,
    arguments_bytes: usize,
    finish_reason: Option<ModelFinishReason>,
    usage: Option<TokenUsage>,
    output_finished: bool,
}

impl NativeStreamStrategy {
    pub fn new(
        tools: ModelToolSet,
        tool_choice: ToolChoice,
        strict: bool,
        max_argument_bytes: usize,
    ) -> Self {
        Self {
            tools,
            tool_choice,
            strict,
            max_argument_bytes,
            classification: Classification::Unknown,
            name: String::new(),
            arguments: String::new(),
            arguments_bytes: 0,
            finish_reason: None,
            usage: None,
            output_finished: false,
        }
    }

    pub fn push(&mut self, delta: ModelDelta) -> crate::Result<Vec<ModelStreamAction>> {
        let mut actions = Vec::new();
        match delta {
            ModelDelta::Text(text) => match self.classification {
                Classification::Unknown => {
                    if matches!(
                        self.tool_choice,
                        ToolChoice::Required | ToolChoice::Function(_)
                    ) {
                        return Err(crate::error::HoshikageError::ToolChoiceViolation);
                    }
                    self.classification = Classification::Text;
                    actions.push(ModelStreamAction::BeginText);
                    actions.push(ModelStreamAction::AppendText(text));
                }
                Classification::Text => actions.push(ModelStreamAction::AppendText(text)),
                Classification::FunctionCall => {
                    return Err(crate::error::HoshikageError::ResponseTranslationFailed);
                }
            },
            ModelDelta::ToolCallStarted { index } => {
                if index != 0
                    || !matches!(self.classification, Classification::Unknown)
                    || self.tool_choice == ToolChoice::None
                {
                    return Err(crate::error::HoshikageError::MultipleToolCalls);
                }
            }
            ModelDelta::ToolName(fragment) => {
                if matches!(self.classification, Classification::Text) {
                    return Err(crate::error::HoshikageError::ResponseTranslationFailed);
                }
                self.name.push_str(&fragment);
                if self.name.len() > 64 {
                    return Err(crate::error::HoshikageError::InvalidToolArguments);
                }
            }
            ModelDelta::ToolArguments(fragment) => {
                self.begin_function_if_ready(&mut actions)?;
                self.arguments_bytes = self.arguments_bytes.saturating_add(fragment.len());
                if self.arguments_bytes > self.max_argument_bytes {
                    return Err(crate::error::HoshikageError::InvalidToolArguments);
                }
                self.arguments.push_str(&fragment);
                actions.push(ModelStreamAction::AppendArguments(fragment));
            }
            ModelDelta::ToolCallFinished => {
                self.begin_function_if_ready(&mut actions)?;
                if !matches!(self.classification, Classification::FunctionCall)
                    || self.output_finished
                {
                    return Err(crate::error::HoshikageError::ResponseTranslationFailed);
                }
                self.validate_arguments()?;
                self.output_finished = true;
                actions.push(ModelStreamAction::FinishFunctionCall);
            }
            ModelDelta::Usage(usage) => {
                if self.usage.is_some() {
                    return Err(crate::error::HoshikageError::ResponseTranslationFailed);
                }
                self.usage = Some(usage);
            }
            ModelDelta::Finished(reason) => {
                if self.finish_reason.is_some() {
                    return Err(crate::error::HoshikageError::ResponseTranslationFailed);
                }
                match (&self.classification, reason) {
                    (Classification::Text, ModelFinishReason::Stop) => {
                        self.output_finished = true;
                        actions.push(ModelStreamAction::FinishText);
                    }
                    (Classification::FunctionCall, ModelFinishReason::ToolCall)
                        if self.output_finished => {}
                    (_, ModelFinishReason::Length) => {
                        return Err(crate::error::HoshikageError::GenerationFailed);
                    }
                    _ => return Err(crate::error::HoshikageError::ResponseTranslationFailed),
                }
                self.finish_reason = Some(reason);
            }
        }
        Ok(actions)
    }

    pub fn finish(mut self) -> crate::Result<ModelStreamAction> {
        if !self.output_finished || self.finish_reason.is_none() {
            return Err(crate::error::HoshikageError::UpstreamDisconnected);
        }
        let usage = self
            .usage
            .take()
            .ok_or(crate::error::HoshikageError::ResponseTranslationFailed)?;
        Ok(ModelStreamAction::Complete { usage })
    }

    fn begin_function_if_ready(
        &mut self,
        actions: &mut Vec<ModelStreamAction>,
    ) -> crate::Result<()> {
        match self.classification {
            Classification::Text => {
                return Err(crate::error::HoshikageError::ResponseTranslationFailed);
            }
            Classification::FunctionCall => return Ok(()),
            Classification::Unknown => {}
        }
        let name = ToolName::new(self.name.clone())
            .map_err(|_| crate::error::HoshikageError::InvalidToolArguments)?;
        if !self.tools.tools().iter().any(|tool| tool.name == name) {
            return Err(crate::error::HoshikageError::InvalidToolArguments);
        }
        if let ToolChoice::Function(required) = &self.tool_choice {
            if required != &name {
                return Err(crate::error::HoshikageError::ToolChoiceViolation);
            }
        }
        self.classification = Classification::FunctionCall;
        actions.push(ModelStreamAction::BeginFunctionCall { name });
        Ok(())
    }

    fn validate_arguments(&self) -> crate::Result<()> {
        let name = ToolName::new(self.name.clone())
            .map_err(|_| crate::error::HoshikageError::InvalidToolArguments)?;
        let tool = self
            .tools
            .tools()
            .iter()
            .find(|tool| tool.name == name)
            .ok_or(crate::error::HoshikageError::InvalidToolArguments)?;
        let arguments = ToolArguments::parse(&self.arguments)
            .map_err(|_| crate::error::HoshikageError::InvalidToolArguments)?;
        if self.strict || tool.strict == Some(true) {
            let validator = jsonschema::draft7::new(&tool.parameters)
                .map_err(|_| crate::error::HoshikageError::InvalidToolSchema)?;
            validator
                .validate(arguments.value())
                .map_err(|_| crate::error::HoshikageError::InvalidToolArguments)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{LlamaServerSseDecoder, ModelTool};

    fn tools() -> ModelToolSet {
        ModelToolSet::new(vec![ModelTool {
            name: ToolName::new("read_file").unwrap(),
            description: None,
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        }])
    }

    #[test]
    fn native_tool_name_is_validated_before_function_output_begins() {
        let fixture = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/llama-server/10075/gemma4-native-tool-stream.sse"
        ));
        let mut sse = LlamaServerSseDecoder::default();
        let mut strategy = NativeStreamStrategy::new(tools(), ToolChoice::Auto, true, 1024);
        let mut actions = Vec::new();
        for chunk in fixture.chunks(11) {
            for delta in sse.push(chunk).unwrap() {
                actions.extend(strategy.push(delta).unwrap());
            }
        }
        sse.finish().unwrap();
        actions.push(strategy.finish().unwrap());

        assert!(matches!(
            actions.first(),
            Some(ModelStreamAction::BeginFunctionCall { name })
                if name.as_str() == "read_file"
        ));
        assert!(matches!(
            actions.last(),
            Some(ModelStreamAction::Complete { .. })
        ));
    }

    #[test]
    fn mixed_text_and_tool_output_fails_closed() {
        let mut strategy = NativeStreamStrategy::new(tools(), ToolChoice::Auto, true, 1024);
        strategy
            .push(ModelDelta::Text("answer".to_string()))
            .unwrap();

        let error = strategy
            .push(ModelDelta::ToolCallStarted { index: 0 })
            .err()
            .unwrap();
        assert!(matches!(
            error,
            crate::error::HoshikageError::MultipleToolCalls
        ));
    }

    #[test]
    fn argument_limit_is_checked_per_fragment() {
        let mut strategy = NativeStreamStrategy::new(tools(), ToolChoice::Auto, true, 4);
        strategy
            .push(ModelDelta::ToolCallStarted { index: 0 })
            .unwrap();
        strategy
            .push(ModelDelta::ToolName("read_file".to_string()))
            .unwrap();

        let error = strategy
            .push(ModelDelta::ToolArguments("12345".to_string()))
            .err()
            .unwrap();
        assert!(matches!(
            error,
            crate::error::HoshikageError::InvalidToolArguments
        ));
    }

    #[test]
    fn streamed_arguments_are_validated_against_strict_schema() {
        let strict_tools = ModelToolSet::new(vec![ModelTool {
            name: ToolName::new("read_file").unwrap(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
            strict: Some(true),
        }]);
        let mut strategy = NativeStreamStrategy::new(strict_tools, ToolChoice::Auto, true, 1024);
        strategy
            .push(ModelDelta::ToolCallStarted { index: 0 })
            .unwrap();
        strategy
            .push(ModelDelta::ToolName("read_file".to_string()))
            .unwrap();
        strategy
            .push(ModelDelta::ToolArguments(r#"{"path":7}"#.to_string()))
            .unwrap();

        let error = strategy.push(ModelDelta::ToolCallFinished).err().unwrap();
        assert!(matches!(
            error,
            crate::error::HoshikageError::InvalidToolArguments
        ));
    }
}
