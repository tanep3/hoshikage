use crate::application::{ResponseEvent, StreamOutput};
use crate::inference::TokenUsage;
use axum::response::sse::Event;
use serde_json::{json, Value};

pub fn to_sse_event(event: &ResponseEvent) -> Result<Event, serde_json::Error> {
    let value = response_event_value(event);
    Ok(Event::default()
        .event(event_type(event))
        .data(serde_json::to_string(&value)?))
}

pub fn response_event_value(event: &ResponseEvent) -> Value {
    match event {
        ResponseEvent::Created {
            sequence_number,
            response_id,
        } => json!({
            "type": "response.created",
            "sequence_number": sequence_number,
            "response": empty_response(response_id.as_str(), "in_progress")
        }),
        ResponseEvent::InProgress {
            sequence_number,
            response_id,
        } => json!({
            "type": "response.in_progress",
            "sequence_number": sequence_number,
            "response": empty_response(response_id.as_str(), "in_progress")
        }),
        ResponseEvent::OutputItemAdded {
            sequence_number,
            output,
        } => json!({
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item": output_value(output, "in_progress")
        }),
        ResponseEvent::ContentPartAdded {
            sequence_number,
            item_id,
        } => json!({
            "type": "response.content_part.added",
            "sequence_number": sequence_number,
            "item_id": item_id.as_str(),
            "output_index": 0,
            "content_index": 0,
            "part": text_part("")
        }),
        ResponseEvent::TextDelta {
            sequence_number,
            item_id,
            delta,
        } => json!({
            "type": "response.output_text.delta",
            "sequence_number": sequence_number,
            "item_id": item_id.as_str(),
            "output_index": 0,
            "content_index": 0,
            "delta": delta,
            "logprobs": []
        }),
        ResponseEvent::TextDone {
            sequence_number,
            item_id,
            text,
        } => json!({
            "type": "response.output_text.done",
            "sequence_number": sequence_number,
            "item_id": item_id.as_str(),
            "output_index": 0,
            "content_index": 0,
            "text": text,
            "logprobs": []
        }),
        ResponseEvent::ContentPartDone {
            sequence_number,
            item_id,
            text,
        } => json!({
            "type": "response.content_part.done",
            "sequence_number": sequence_number,
            "item_id": item_id.as_str(),
            "output_index": 0,
            "content_index": 0,
            "part": text_part(text)
        }),
        ResponseEvent::FunctionArgumentsDelta {
            sequence_number,
            item_id,
            delta,
        } => json!({
            "type": "response.function_call_arguments.delta",
            "sequence_number": sequence_number,
            "item_id": item_id.as_str(),
            "output_index": 0,
            "delta": delta
        }),
        ResponseEvent::FunctionArgumentsDone {
            sequence_number,
            item_id,
            name,
            arguments,
        } => json!({
            "type": "response.function_call_arguments.done",
            "sequence_number": sequence_number,
            "item_id": item_id.as_str(),
            "output_index": 0,
            "name": name.as_str(),
            "arguments": arguments.canonical_json()
        }),
        ResponseEvent::OutputItemDone {
            sequence_number,
            output,
        } => json!({
            "type": "response.output_item.done",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item": output_value(output, "completed")
        }),
        ResponseEvent::Completed {
            sequence_number,
            response_id,
            output,
            usage,
        } => json!({
            "type": "response.completed",
            "sequence_number": sequence_number,
            "response": {
                "id": response_id.as_str(),
                "object": "response",
                "status": "completed",
                "output": [output_value(output, "completed")],
                "usage": usage_value(usage)
            }
        }),
        ResponseEvent::Error {
            sequence_number,
            failure,
            ..
        } => json!({
            "type": "error",
            "sequence_number": sequence_number,
            "code": failure.code,
            "message": failure.message,
            "param": Value::Null
        }),
        ResponseEvent::Failed {
            sequence_number,
            response_id,
            failure,
        } => json!({
            "type": "response.failed",
            "sequence_number": sequence_number,
            "response": {
                "id": response_id.as_str(),
                "object": "response",
                "status": "failed",
                "output": [],
                "error": {
                    "code": failure.code,
                    "message": failure.message
                }
            }
        }),
    }
}

fn event_type(event: &ResponseEvent) -> &'static str {
    match event {
        ResponseEvent::Created { .. } => "response.created",
        ResponseEvent::InProgress { .. } => "response.in_progress",
        ResponseEvent::OutputItemAdded { .. } => "response.output_item.added",
        ResponseEvent::ContentPartAdded { .. } => "response.content_part.added",
        ResponseEvent::TextDelta { .. } => "response.output_text.delta",
        ResponseEvent::TextDone { .. } => "response.output_text.done",
        ResponseEvent::ContentPartDone { .. } => "response.content_part.done",
        ResponseEvent::FunctionArgumentsDelta { .. } => "response.function_call_arguments.delta",
        ResponseEvent::FunctionArgumentsDone { .. } => "response.function_call_arguments.done",
        ResponseEvent::OutputItemDone { .. } => "response.output_item.done",
        ResponseEvent::Completed { .. } => "response.completed",
        ResponseEvent::Error { .. } => "error",
        ResponseEvent::Failed { .. } => "response.failed",
    }
}

fn empty_response(response_id: &str, status: &str) -> Value {
    json!({
        "id": response_id,
        "object": "response",
        "status": status,
        "output": []
    })
}

fn text_part(text: &str) -> Value {
    json!({
        "type": "output_text",
        "annotations": [],
        "logprobs": [],
        "text": text
    })
}

fn output_value(output: &StreamOutput, status: &str) -> Value {
    match output {
        StreamOutput::Text { item_id, text } => json!({
            "id": item_id.as_str(),
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": if status == "completed" {
                vec![text_part(text)]
            } else {
                Vec::new()
            }
        }),
        StreamOutput::FunctionCall {
            item_id,
            call_id,
            name,
            arguments,
        } => json!({
            "id": item_id.as_str(),
            "type": "function_call",
            "status": status,
            "call_id": call_id.as_str(),
            "name": name.as_str(),
            "arguments": arguments
        }),
    }
}

fn usage_value(usage: &TokenUsage) -> Value {
    let (input_tokens, output_tokens) = match usage {
        TokenUsage::Measured {
            input_tokens,
            output_tokens,
        }
        | TokenUsage::Estimated {
            input_tokens,
            output_tokens,
        } => (*input_tokens, *output_tokens),
    };
    json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens.saturating_add(output_tokens)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::ResponseMachine;
    use crate::conversation::{CallId, OutputItemId, ResponseId, ToolName};

    fn fixture(relative: &str) -> Vec<Value> {
        let body = match relative {
            "text" => include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/codex/0.144.x/expected-text-events.jsonl"
            )),
            "tool" => include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/codex/0.144.x/expected-tool-events.jsonl"
            )),
            _ => unreachable!(),
        };
        body.lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn usage() -> TokenUsage {
        TokenUsage::Measured {
            input_tokens: 1,
            output_tokens: 1,
        }
    }

    #[test]
    fn text_events_match_captured_codex_fixture_exactly() {
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
            events.iter().map(response_event_value).collect::<Vec<_>>(),
            fixture("text")
        );
    }

    #[test]
    fn function_events_match_captured_codex_fixture_exactly() {
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
            events.iter().map(response_event_value).collect::<Vec<_>>(),
            fixture("tool")
        );
    }
}
