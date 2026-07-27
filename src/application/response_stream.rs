use crate::conversation::{CallId, OutputItemId, ResponseId, ToolArguments, ToolName};
use crate::inference::TokenUsage;
use thiserror::Error;

#[derive(Clone, PartialEq)]
pub enum StreamOutput {
    Text {
        item_id: OutputItemId,
        text: String,
    },
    FunctionCall {
        item_id: OutputItemId,
        call_id: CallId,
        name: ToolName,
        arguments: String,
    },
}

#[derive(Clone, PartialEq)]
pub enum ResponseEvent {
    Created {
        sequence_number: u64,
        response_id: ResponseId,
    },
    InProgress {
        sequence_number: u64,
        response_id: ResponseId,
    },
    OutputItemAdded {
        sequence_number: u64,
        output: StreamOutput,
    },
    ContentPartAdded {
        sequence_number: u64,
        item_id: OutputItemId,
    },
    TextDelta {
        sequence_number: u64,
        item_id: OutputItemId,
        delta: String,
    },
    TextDone {
        sequence_number: u64,
        item_id: OutputItemId,
        text: String,
    },
    ContentPartDone {
        sequence_number: u64,
        item_id: OutputItemId,
        text: String,
    },
    FunctionArgumentsDelta {
        sequence_number: u64,
        item_id: OutputItemId,
        delta: String,
    },
    FunctionArgumentsDone {
        sequence_number: u64,
        item_id: OutputItemId,
        name: ToolName,
        arguments: ToolArguments,
    },
    OutputItemDone {
        sequence_number: u64,
        output: StreamOutput,
    },
    Completed {
        sequence_number: u64,
        response_id: ResponseId,
        output: StreamOutput,
        usage: TokenUsage,
    },
    Error {
        sequence_number: u64,
        response_id: ResponseId,
        failure: StreamFailure,
    },
    Failed {
        sequence_number: u64,
        response_id: ResponseId,
        failure: StreamFailure,
    },
}

impl ResponseEvent {
    pub fn sequence_number(&self) -> u64 {
        match self {
            Self::Created {
                sequence_number, ..
            }
            | Self::InProgress {
                sequence_number, ..
            }
            | Self::OutputItemAdded {
                sequence_number, ..
            }
            | Self::ContentPartAdded {
                sequence_number, ..
            }
            | Self::TextDelta {
                sequence_number, ..
            }
            | Self::TextDone {
                sequence_number, ..
            }
            | Self::ContentPartDone {
                sequence_number, ..
            }
            | Self::FunctionArgumentsDelta {
                sequence_number, ..
            }
            | Self::FunctionArgumentsDone {
                sequence_number, ..
            }
            | Self::OutputItemDone {
                sequence_number, ..
            }
            | Self::Completed {
                sequence_number, ..
            }
            | Self::Error {
                sequence_number, ..
            }
            | Self::Failed {
                sequence_number, ..
            } => *sequence_number,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamFailure {
    pub code: &'static str,
    pub message: &'static str,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ResponseMachineError {
    #[error("response stream transition is invalid")]
    InvalidTransition,
    #[error("function arguments are invalid")]
    InvalidToolArguments,
}

enum MachineState {
    New,
    InProgress { output: Option<OutputDraft> },
    Completed,
    Failed,
}

enum OutputDraft {
    Text {
        item_id: OutputItemId,
        text: String,
        finished: bool,
    },
    FunctionCall {
        item_id: OutputItemId,
        call_id: CallId,
        name: ToolName,
        arguments: String,
        finished: bool,
    },
}

pub struct ResponseMachine {
    response_id: ResponseId,
    sequence_number: u64,
    state: MachineState,
}

impl ResponseMachine {
    pub fn new(response_id: ResponseId) -> Self {
        Self {
            response_id,
            sequence_number: 0,
            state: MachineState::New,
        }
    }

    pub fn start(&mut self) -> Result<Vec<ResponseEvent>, ResponseMachineError> {
        if !matches!(self.state, MachineState::New) {
            return Err(ResponseMachineError::InvalidTransition);
        }
        self.state = MachineState::InProgress { output: None };
        let created_response_id = self.response_id.clone();
        let in_progress_response_id = self.response_id.clone();
        Ok(vec![
            self.next(move |sequence_number| ResponseEvent::Created {
                sequence_number,
                response_id: created_response_id,
            }),
            self.next(move |sequence_number| ResponseEvent::InProgress {
                sequence_number,
                response_id: in_progress_response_id,
            }),
        ])
    }

    pub fn begin_text(
        &mut self,
        item_id: OutputItemId,
    ) -> Result<Vec<ResponseEvent>, ResponseMachineError> {
        let MachineState::InProgress { output } = &mut self.state else {
            return Err(ResponseMachineError::InvalidTransition);
        };
        if output.is_some() {
            return Err(ResponseMachineError::InvalidTransition);
        }
        *output = Some(OutputDraft::Text {
            item_id: item_id.clone(),
            text: String::new(),
            finished: false,
        });
        Ok(vec![
            self.next(|sequence_number| ResponseEvent::OutputItemAdded {
                sequence_number,
                output: StreamOutput::Text {
                    item_id: item_id.clone(),
                    text: String::new(),
                },
            }),
            self.next(|sequence_number| ResponseEvent::ContentPartAdded {
                sequence_number,
                item_id,
            }),
        ])
    }

    pub fn append_text(&mut self, delta: String) -> Result<ResponseEvent, ResponseMachineError> {
        let MachineState::InProgress {
            output:
                Some(OutputDraft::Text {
                    item_id,
                    text,
                    finished: false,
                }),
        } = &mut self.state
        else {
            return Err(ResponseMachineError::InvalidTransition);
        };
        text.push_str(&delta);
        let item_id = item_id.clone();
        Ok(self.next(|sequence_number| ResponseEvent::TextDelta {
            sequence_number,
            item_id,
            delta,
        }))
    }

    pub fn finish_text(&mut self) -> Result<Vec<ResponseEvent>, ResponseMachineError> {
        let MachineState::InProgress {
            output:
                Some(OutputDraft::Text {
                    item_id,
                    text,
                    finished,
                }),
        } = &mut self.state
        else {
            return Err(ResponseMachineError::InvalidTransition);
        };
        if *finished {
            return Err(ResponseMachineError::InvalidTransition);
        }
        *finished = true;
        let item_id = item_id.clone();
        let text = text.clone();
        Ok(vec![
            self.next(|sequence_number| ResponseEvent::TextDone {
                sequence_number,
                item_id: item_id.clone(),
                text: text.clone(),
            }),
            self.next(|sequence_number| ResponseEvent::ContentPartDone {
                sequence_number,
                item_id: item_id.clone(),
                text: text.clone(),
            }),
            self.next(|sequence_number| ResponseEvent::OutputItemDone {
                sequence_number,
                output: StreamOutput::Text { item_id, text },
            }),
        ])
    }

    pub fn begin_function_call(
        &mut self,
        item_id: OutputItemId,
        call_id: CallId,
        name: ToolName,
    ) -> Result<ResponseEvent, ResponseMachineError> {
        let MachineState::InProgress { output } = &mut self.state else {
            return Err(ResponseMachineError::InvalidTransition);
        };
        if output.is_some() {
            return Err(ResponseMachineError::InvalidTransition);
        }
        *output = Some(OutputDraft::FunctionCall {
            item_id: item_id.clone(),
            call_id: call_id.clone(),
            name: name.clone(),
            arguments: String::new(),
            finished: false,
        });
        Ok(self.next(|sequence_number| ResponseEvent::OutputItemAdded {
            sequence_number,
            output: StreamOutput::FunctionCall {
                item_id,
                call_id,
                name,
                arguments: String::new(),
            },
        }))
    }

    pub fn append_function_arguments(
        &mut self,
        delta: String,
    ) -> Result<ResponseEvent, ResponseMachineError> {
        let MachineState::InProgress {
            output:
                Some(OutputDraft::FunctionCall {
                    item_id,
                    arguments,
                    finished: false,
                    ..
                }),
        } = &mut self.state
        else {
            return Err(ResponseMachineError::InvalidTransition);
        };
        arguments.push_str(&delta);
        let item_id = item_id.clone();
        Ok(
            self.next(|sequence_number| ResponseEvent::FunctionArgumentsDelta {
                sequence_number,
                item_id,
                delta,
            }),
        )
    }

    pub fn finish_function_call(&mut self) -> Result<Vec<ResponseEvent>, ResponseMachineError> {
        let MachineState::InProgress {
            output:
                Some(OutputDraft::FunctionCall {
                    item_id,
                    call_id,
                    name,
                    arguments,
                    finished,
                }),
        } = &mut self.state
        else {
            return Err(ResponseMachineError::InvalidTransition);
        };
        if *finished {
            return Err(ResponseMachineError::InvalidTransition);
        }
        let parsed = ToolArguments::parse(arguments)
            .map_err(|_| ResponseMachineError::InvalidToolArguments)?;
        *finished = true;
        let item_id = item_id.clone();
        let call_id = call_id.clone();
        let name = name.clone();
        let canonical = parsed.canonical_json().to_string();
        *arguments = canonical.clone();
        Ok(vec![
            self.next(|sequence_number| ResponseEvent::FunctionArgumentsDone {
                sequence_number,
                item_id: item_id.clone(),
                name: name.clone(),
                arguments: parsed,
            }),
            self.next(|sequence_number| ResponseEvent::OutputItemDone {
                sequence_number,
                output: StreamOutput::FunctionCall {
                    item_id,
                    call_id,
                    name,
                    arguments: canonical,
                },
            }),
        ])
    }

    pub fn complete(&mut self, usage: TokenUsage) -> Result<ResponseEvent, ResponseMachineError> {
        let MachineState::InProgress {
            output: Some(output),
        } = &self.state
        else {
            return Err(ResponseMachineError::InvalidTransition);
        };
        let output = match output {
            OutputDraft::Text {
                item_id,
                text,
                finished: true,
            } => StreamOutput::Text {
                item_id: item_id.clone(),
                text: text.clone(),
            },
            OutputDraft::FunctionCall {
                item_id,
                call_id,
                name,
                arguments,
                finished: true,
            } => StreamOutput::FunctionCall {
                item_id: item_id.clone(),
                call_id: call_id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
            },
            _ => return Err(ResponseMachineError::InvalidTransition),
        };
        self.state = MachineState::Completed;
        let response_id = self.response_id.clone();
        Ok(self.next(move |sequence_number| ResponseEvent::Completed {
            sequence_number,
            response_id,
            output,
            usage,
        }))
    }

    pub fn fail(
        &mut self,
        failure: StreamFailure,
    ) -> Result<Vec<ResponseEvent>, ResponseMachineError> {
        if !matches!(self.state, MachineState::InProgress { .. }) {
            return Err(ResponseMachineError::InvalidTransition);
        }
        self.state = MachineState::Failed;
        let error_response_id = self.response_id.clone();
        let failed_response_id = self.response_id.clone();
        Ok(vec![
            self.next(move |sequence_number| ResponseEvent::Error {
                sequence_number,
                response_id: error_response_id,
                failure,
            }),
            self.next(move |sequence_number| ResponseEvent::Failed {
                sequence_number,
                response_id: failed_response_id,
                failure,
            }),
        ])
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self.state, MachineState::Completed | MachineState::Failed)
    }

    fn next(&mut self, build: impl FnOnce(u64) -> ResponseEvent) -> ResponseEvent {
        let sequence_number = self.sequence_number;
        self.sequence_number = self.sequence_number.saturating_add(1);
        build(sequence_number)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(event: &ResponseEvent) -> &'static str {
        match event {
            ResponseEvent::Created { .. } => "response.created",
            ResponseEvent::InProgress { .. } => "response.in_progress",
            ResponseEvent::OutputItemAdded { .. } => "response.output_item.added",
            ResponseEvent::ContentPartAdded { .. } => "response.content_part.added",
            ResponseEvent::TextDelta { .. } => "response.output_text.delta",
            ResponseEvent::TextDone { .. } => "response.output_text.done",
            ResponseEvent::ContentPartDone { .. } => "response.content_part.done",
            ResponseEvent::FunctionArgumentsDelta { .. } => {
                "response.function_call_arguments.delta"
            }
            ResponseEvent::FunctionArgumentsDone { .. } => "response.function_call_arguments.done",
            ResponseEvent::OutputItemDone { .. } => "response.output_item.done",
            ResponseEvent::Completed { .. } => "response.completed",
            ResponseEvent::Error { .. } => "error",
            ResponseEvent::Failed { .. } => "response.failed",
        }
    }

    fn usage() -> TokenUsage {
        TokenUsage::Measured {
            input_tokens: 1,
            output_tokens: 1,
        }
    }

    #[test]
    fn text_state_machine_matches_codex_fixture_order() {
        let mut machine = ResponseMachine::new(ResponseId::new("resp_fixture").unwrap());
        let mut events = machine.start().unwrap();
        events.extend(
            machine
                .begin_text(OutputItemId::new("msg_fixture").unwrap())
                .unwrap(),
        );
        events.push(machine.append_text("OK".to_string()).unwrap());
        events.extend(machine.finish_text().unwrap());
        events.push(machine.complete(usage()).unwrap());

        assert_eq!(
            events.iter().map(kind).collect::<Vec<_>>(),
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_eq!(
            events
                .iter()
                .map(ResponseEvent::sequence_number)
                .collect::<Vec<_>>(),
            (0..9).collect::<Vec<_>>()
        );
        assert_eq!(
            machine.append_text("late".to_string()).err().unwrap(),
            ResponseMachineError::InvalidTransition
        );
    }

    #[test]
    fn function_state_machine_matches_codex_fixture_order() {
        let mut machine = ResponseMachine::new(ResponseId::new("resp_fixture").unwrap());
        let mut events = machine.start().unwrap();
        events.push(
            machine
                .begin_function_call(
                    OutputItemId::new("fc_fixture").unwrap(),
                    CallId::new("call_fixture").unwrap(),
                    ToolName::new("exec_command").unwrap(),
                )
                .unwrap(),
        );
        events.push(
            machine
                .append_function_arguments(r#"{"cmd":"printf phase0-tool-ok"}"#.to_string())
                .unwrap(),
        );
        events.extend(machine.finish_function_call().unwrap());
        events.push(machine.complete(usage()).unwrap());

        assert_eq!(
            events.iter().map(kind).collect::<Vec<_>>(),
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_eq!(
            events
                .iter()
                .map(ResponseEvent::sequence_number)
                .collect::<Vec<_>>(),
            (0..7).collect::<Vec<_>>()
        );
    }

    #[test]
    fn failure_is_terminal_and_never_emits_completed() {
        let mut machine = ResponseMachine::new(ResponseId::new("resp_fixture").unwrap());
        let mut events = machine.start().unwrap();
        events.extend(
            machine
                .fail(StreamFailure {
                    code: "upstream_disconnected",
                    message: "Upstream disconnected",
                })
                .unwrap(),
        );

        assert_eq!(
            events.iter().map(kind).collect::<Vec<_>>(),
            [
                "response.created",
                "response.in_progress",
                "error",
                "response.failed"
            ]
        );
        assert_eq!(
            machine.complete(usage()).err().unwrap(),
            ResponseMachineError::InvalidTransition
        );
    }
}
