use crate::model::{ThinkingConfig, ThinkingMode};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkingDecision {
    pub effective_mode: ThinkingMode,
    pub strip_thinking: bool,
    pub runtime_budget_tokens: Option<i32>,
    pub diagnostic: Option<String>,
}

pub struct ThinkingController;

impl ThinkingController {
    pub fn decide(config: &ThinkingConfig) -> ThinkingDecision {
        match config.mode {
            ThinkingMode::Auto => ThinkingDecision {
                effective_mode: ThinkingMode::Auto,
                strip_thinking: false,
                runtime_budget_tokens: None,
                diagnostic: None,
            },
            ThinkingMode::On => ThinkingDecision {
                effective_mode: ThinkingMode::On,
                strip_thinking: false,
                runtime_budget_tokens: config
                    .max_reasoning_tokens
                    .and_then(|value| i32::try_from(value).ok()),
                diagnostic: None,
            },
            ThinkingMode::Off => ThinkingDecision {
                effective_mode: ThinkingMode::Off,
                strip_thinking: true,
                runtime_budget_tokens: Some(0),
                diagnostic: None,
            },
        }
    }

    pub fn apply_prompt_policy_if_needed(decision: &ThinkingDecision, prompt: &str) -> String {
        if !decision.strip_thinking {
            return prompt.to_string();
        }

        remove_trailing_thinking_markers(prompt)
    }

    pub fn strip_output_if_needed(decision: &ThinkingDecision, output: &str) -> String {
        if !decision.strip_thinking {
            return output.to_string();
        }

        let mut filter = ThinkingStreamFilter::new(decision);
        let mut stripped = String::new();
        for chunk in filter.push(output) {
            stripped.push_str(&chunk);
        }
        for chunk in filter.finish() {
            stripped.push_str(&chunk);
        }
        stripped
    }
}

pub struct ThinkingStreamFilter {
    enabled: bool,
    buffer: String,
    inside_block: Option<&'static str>,
    stripped_bytes: usize,
}

impl ThinkingStreamFilter {
    pub fn new(decision: &ThinkingDecision) -> Self {
        Self {
            enabled: decision.strip_thinking,
            buffer: String::new(),
            inside_block: None,
            stripped_bytes: 0,
        }
    }

    pub fn push(&mut self, chunk: &str) -> Vec<String> {
        if !self.enabled {
            return vec![chunk.to_string()];
        }

        self.buffer.push_str(chunk);
        self.drain(false)
    }

    pub fn finish(&mut self) -> Vec<String> {
        if !self.enabled {
            return Vec::new();
        }

        self.drain(true)
    }

    pub fn stripped_bytes(&self) -> usize {
        self.stripped_bytes
    }

    fn drain(&mut self, finish: bool) -> Vec<String> {
        let mut output = Vec::new();

        loop {
            if let Some(close_tag) = self.inside_block {
                if let Some(close_pos) = find_case_insensitive(&self.buffer, close_tag) {
                    let end = close_pos + close_tag.len();
                    self.stripped_bytes += self.buffer[..end].len();
                    self.buffer.drain(..end);
                    self.inside_block = None;
                    continue;
                }

                let keep = if finish {
                    0
                } else {
                    max_close_tag_len().saturating_sub(1)
                };
                if self.buffer.len() > keep {
                    let discard_to = floor_char_boundary(&self.buffer, self.buffer.len() - keep);
                    self.stripped_bytes += self.buffer[..discard_to].len();
                    self.buffer.drain(..discard_to);
                }

                if finish && !self.buffer.is_empty() {
                    self.stripped_bytes += self.buffer.len();
                    self.buffer.clear();
                }
                break;
            }

            if let Some((open_pos, open_tag, close_tag)) = find_earliest_open_tag(&self.buffer) {
                if open_pos > 0 {
                    output.push(self.buffer[..open_pos].to_string());
                }
                let end = open_pos + open_tag.len();
                self.stripped_bytes += self.buffer[open_pos..end].len();
                self.buffer.drain(..end);
                self.inside_block = Some(close_tag);
                continue;
            }

            if finish {
                if !self.buffer.is_empty() {
                    output.push(std::mem::take(&mut self.buffer));
                }
                break;
            }

            let keep = max_open_tag_len().saturating_sub(1);
            if self.buffer.len() > keep {
                let emit_to = floor_char_boundary(&self.buffer, self.buffer.len() - keep);
                output.push(self.buffer[..emit_to].to_string());
                self.buffer.drain(..emit_to);
            }
            break;
        }

        output
    }
}

const ALL_THINKING_TAGS: [(&str, &str); 5] = [
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<thought>", "</thought>"),
    ("<|channel>thought", "<channel|>"),
    ("<|START_THINKING|>", "<|END_THINKING|>"),
];

fn find_earliest_open_tag(buffer: &str) -> Option<(usize, &'static str, &'static str)> {
    ALL_THINKING_TAGS
        .iter()
        .filter_map(|(open, close)| {
            find_case_insensitive(buffer, open).map(|pos| (pos, *open, *close))
        })
        .min_by_key(|(pos, _, _)| *pos)
}

fn find_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .to_ascii_lowercase()
        .find(&needle.to_ascii_lowercase())
}

fn max_open_tag_len() -> usize {
    ALL_THINKING_TAGS
        .iter()
        .map(|(open, _)| open.len())
        .max()
        .unwrap_or(0)
}

fn max_close_tag_len() -> usize {
    ALL_THINKING_TAGS
        .iter()
        .map(|(_, close)| close.len())
        .max()
        .unwrap_or(0)
}

fn remove_trailing_thinking_markers(prompt: &str) -> String {
    let mut output = prompt.to_string();

    loop {
        let trimmed_len = output.trim_end().len();
        let trailing = output[trimmed_len..].to_string();
        let lowered_prefix = output[..trimmed_len].to_ascii_lowercase();
        let mut changed = false;

        for (open_tag, _) in ALL_THINKING_TAGS {
            let open_lower = open_tag.to_ascii_lowercase();
            if lowered_prefix.ends_with(&open_lower) {
                output.truncate(trimmed_len - open_tag.len());
                output.push_str(&trailing);
                changed = true;
                break;
            }
        }

        if !changed {
            break;
        }
    }

    output
}

fn floor_char_boundary(value: &str, index: usize) -> usize {
    let mut boundary = index.min(value.len());
    while boundary > 0 && !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    boundary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn off_decision() -> ThinkingDecision {
        ThinkingController::decide(&ThinkingConfig {
            mode: ThinkingMode::Off,
            ..ThinkingConfig::default()
        })
    }

    #[test]
    fn auto_keeps_output_unchanged() {
        let decision = ThinkingController::decide(&ThinkingConfig {
            mode: ThinkingMode::Auto,
            ..ThinkingConfig::default()
        });

        assert_eq!(
            ThinkingController::strip_output_if_needed(&decision, "<think>hidden</think>visible"),
            "<think>hidden</think>visible"
        );
    }

    #[test]
    fn off_strips_closed_think_block() {
        let decision = off_decision();

        assert_eq!(
            ThinkingController::strip_output_if_needed(
                &decision,
                "before <think>hidden</think> after"
            ),
            "before  after"
        );
    }

    #[test]
    fn off_strips_unclosed_think_block_to_end() {
        let decision = off_decision();

        assert_eq!(
            ThinkingController::strip_output_if_needed(&decision, "answer<thinking>hidden"),
            "answer"
        );
    }

    #[test]
    fn off_strips_case_insensitive_tags() {
        let decision = off_decision();

        assert_eq!(
            ThinkingController::strip_output_if_needed(&decision, "<Thinking>hidden</Thinking>ok"),
            "ok"
        );
    }

    #[test]
    fn off_strips_gemma4_thought_channel() {
        let decision = off_decision();

        assert_eq!(
            ThinkingController::strip_output_if_needed(
                &decision,
                "<|channel>thought\nhidden<channel|>ok"
            ),
            "ok"
        );
    }

    #[test]
    fn off_strips_uppercase_thinking_marker() {
        let decision = off_decision();

        assert_eq!(
            ThinkingController::strip_output_if_needed(
                &decision,
                "<|START_THINKING|>hidden<|END_THINKING|>ok"
            ),
            "ok"
        );
    }

    #[test]
    fn off_removes_trailing_thinking_prompt_marker() {
        let decision = off_decision();

        assert_eq!(
            ThinkingController::apply_prompt_policy_if_needed(
                &decision,
                "<|turn>model\n<|channel>thought\n"
            ),
            "<|turn>model\n\n"
        );
    }

    #[test]
    fn stream_filter_handles_split_tags() {
        let decision = off_decision();
        let mut filter = ThinkingStreamFilter::new(&decision);
        let mut visible = String::new();

        for chunk in ["hel", "lo <thi", "nk>hidden</thi", "nk> world"] {
            for out in filter.push(chunk) {
                visible.push_str(&out);
            }
        }
        for out in filter.finish() {
            visible.push_str(&out);
        }

        assert_eq!(visible, "hello  world");
        assert!(filter.stripped_bytes() > 0);
    }
}
