use super::{ModelDelta, ModelFinishReason, TokenUsage};
use serde_json::Value;

#[derive(Default)]
pub struct LlamaServerSseDecoder {
    buffer: Vec<u8>,
    done: bool,
    tool_started: bool,
    tool_finished: bool,
    finish_reason: Option<ModelFinishReason>,
    usage: Option<TokenUsage>,
}

impl LlamaServerSseDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> crate::Result<Vec<ModelDelta>> {
        if self.done {
            return Err(crate::error::HoshikageError::ResponseTranslationFailed);
        }
        self.buffer.extend_from_slice(bytes);
        let mut deltas = Vec::new();
        while let Some((frame_end, separator_len)) = find_frame(&self.buffer) {
            let frame = self.buffer.drain(..frame_end).collect::<Vec<_>>();
            self.buffer.drain(..separator_len);
            self.decode_frame(&frame, &mut deltas)?;
        }
        Ok(deltas)
    }

    pub fn finish(mut self) -> crate::Result<TokenUsage> {
        if !self.buffer.iter().all(u8::is_ascii_whitespace) {
            let frame = std::mem::take(&mut self.buffer);
            self.decode_frame(&frame, &mut Vec::new())?;
        }
        if !self.done
            || !self.buffer.iter().all(u8::is_ascii_whitespace)
            || self.finish_reason.is_none()
        {
            return Err(crate::error::HoshikageError::UpstreamDisconnected);
        }
        self.usage
            .ok_or(crate::error::HoshikageError::ResponseTranslationFailed)
    }

    fn decode_frame(&mut self, frame: &[u8], deltas: &mut Vec<ModelDelta>) -> crate::Result<()> {
        let frame = std::str::from_utf8(frame)
            .map_err(|_| crate::error::HoshikageError::ResponseTranslationFailed)?;
        let data = frame
            .lines()
            .filter_map(|line| line.trim_end_matches('\r').strip_prefix("data:"))
            .map(str::trim_start)
            .collect::<Vec<_>>()
            .join("\n");
        if data.is_empty() {
            return Ok(());
        }
        if data == "[DONE]" {
            self.done = true;
            return Ok(());
        }
        let value: Value = serde_json::from_str(&data)
            .map_err(|_| crate::error::HoshikageError::ResponseTranslationFailed)?;
        if value.get("error").is_some() {
            return Err(crate::error::HoshikageError::GenerationFailed);
        }
        if let Some(usage) = value.get("usage") {
            let input_tokens = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or(crate::error::HoshikageError::ResponseTranslationFailed)?;
            let output_tokens = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or(crate::error::HoshikageError::ResponseTranslationFailed)?;
            let usage = TokenUsage::Measured {
                input_tokens,
                output_tokens,
            };
            self.usage = Some(usage.clone());
            deltas.push(ModelDelta::Usage(usage));
        }
        let Some(choice) = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            return Ok(());
        };
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        if let Some(content) = delta.get("content").and_then(Value::as_str) {
            if !content.is_empty() {
                deltas.push(ModelDelta::Text(content.to_string()));
            }
        }
        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tool_call in tool_calls {
                let index = tool_call.get("index").and_then(Value::as_u64).unwrap_or(0);
                if index != 0 {
                    return Err(crate::error::HoshikageError::MultipleToolCalls);
                }
                if !self.tool_started {
                    self.tool_started = true;
                    deltas.push(ModelDelta::ToolCallStarted { index: 0 });
                }
                if let Some(function) = tool_call.get("function") {
                    if let Some(name) = function.get("name").and_then(Value::as_str) {
                        if !name.is_empty() {
                            deltas.push(ModelDelta::ToolName(name.to_string()));
                        }
                    }
                    if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                        if !arguments.is_empty() {
                            deltas.push(ModelDelta::ToolArguments(arguments.to_string()));
                        }
                    }
                }
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            let reason = match reason {
                "stop" => ModelFinishReason::Stop,
                "tool_calls" => {
                    if !self.tool_started || self.tool_finished {
                        return Err(crate::error::HoshikageError::ResponseTranslationFailed);
                    }
                    self.tool_finished = true;
                    deltas.push(ModelDelta::ToolCallFinished);
                    ModelFinishReason::ToolCall
                }
                "length" => ModelFinishReason::Length,
                _ => return Err(crate::error::HoshikageError::ResponseTranslationFailed),
            };
            self.finish_reason = Some(reason);
            deltas.push(ModelDelta::Finished(reason));
        }
        Ok(())
    }
}

fn find_frame(buffer: &[u8]) -> Option<(usize, usize)> {
    buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|index| (index, 2))
        .or_else(|| {
            buffer
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| (index, 4))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_tool_fixture_decodes_across_arbitrary_byte_boundaries() {
        let fixture = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/llama-server/10075/gemma4-native-tool-stream.sse"
        ));
        let mut decoder = LlamaServerSseDecoder::default();
        let mut deltas = Vec::new();
        for chunk in fixture.chunks(7) {
            deltas.extend(decoder.push(chunk).unwrap());
        }
        let usage = decoder.finish().unwrap();

        let arguments = deltas
            .iter()
            .filter_map(|delta| match delta {
                ModelDelta::ToolArguments(fragment) => Some(fragment.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(arguments, r#"{"path":"README.md"}"#);
        assert_eq!(
            usage,
            TokenUsage::Measured {
                input_tokens: 75,
                output_tokens: 17
            }
        );
        assert!(deltas.contains(&ModelDelta::Finished(ModelFinishReason::ToolCall)));
    }

    #[test]
    fn text_stream_preserves_unicode_split_inside_utf8_codepoint() {
        let fixture = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"星影\"},",
            "\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,",
            "\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n"
        );
        let mut decoder = LlamaServerSseDecoder::default();
        let mut deltas = Vec::new();
        for byte in fixture.as_bytes() {
            deltas.extend(decoder.push(std::slice::from_ref(byte)).unwrap());
        }
        decoder.finish().unwrap();

        assert!(deltas.contains(&ModelDelta::Text("星影".to_string())));
        assert!(deltas.contains(&ModelDelta::Finished(ModelFinishReason::Stop)));
    }

    #[test]
    fn missing_done_is_upstream_disconnect() {
        let mut decoder = LlamaServerSseDecoder::default();
        decoder
            .push(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
            .unwrap();

        assert!(matches!(
            decoder.finish().err().unwrap(),
            crate::error::HoshikageError::UpstreamDisconnected
        ));
    }
}
