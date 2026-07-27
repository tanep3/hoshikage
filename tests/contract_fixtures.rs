use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

fn fixture_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(relative)
}

fn read_json(relative: &str) -> Value {
    let path = fixture_path(relative);
    let body = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&body)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

fn read_json_lines(relative: &str) -> Vec<Value> {
    let path = fixture_path(relative);
    let body = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    body.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line).unwrap_or_else(|error| {
                panic!(
                    "failed to parse {} line {}: {error}",
                    path.display(),
                    index + 1
                )
            })
        })
        .collect()
}

fn read_sse_data(relative: &str) -> (Vec<Value>, bool) {
    let path = fixture_path(relative);
    let body = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let mut done = false;
    let events = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| {
            if data == "[DONE]" {
                done = true;
                None
            } else {
                Some(serde_json::from_str(data).unwrap_or_else(|error| {
                    panic!("failed to parse SSE data in {}: {error}", path.display())
                }))
            }
        })
        .collect();
    (events, done)
}

fn event_types(relative: &str) -> Vec<String> {
    read_json_lines(relative)
        .into_iter()
        .map(|event| {
            event["type"]
                .as_str()
                .expect("event type must be a string")
                .to_owned()
        })
        .collect()
}

#[test]
fn codex_0_144_request_contract_is_preserved() {
    let text = read_json("codex/0.144.x/text-request.json");
    let tool = read_json("codex/0.144.x/tool-request.json");

    for request in [&text, &tool] {
        assert_eq!(request["stream"], true);
        assert_eq!(request["store"], false);
        assert_eq!(request["tool_choice"], "auto");
        assert_eq!(request["parallel_tool_calls"], false);
        assert!(request["input"].is_array());
        assert!(request["tools"].is_array());
    }

    let tool_types: Vec<&str> = tool["tools"]
        .as_array()
        .expect("tools must be an array")
        .iter()
        .map(|tool| tool["type"].as_str().expect("tool type must be a string"))
        .collect();
    assert!(tool_types.contains(&"function"));
    assert!(tool_types.contains(&"namespace"));
    assert!(tool_types.contains(&"web_search"));
}

#[test]
fn codex_tool_result_preserves_call_identity_and_json_arguments() {
    let request = read_json("codex/0.144.x/tool-output-request.json");
    let input = request["input"].as_array().expect("input must be an array");
    let function_call = input
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("function_call fixture item");
    let function_output = input
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("function_call_output fixture item");

    assert_eq!(function_call["call_id"], function_output["call_id"]);
    let arguments = function_call["arguments"]
        .as_str()
        .expect("Codex function arguments must be a JSON string");
    let parsed: Value = serde_json::from_str(arguments).expect("arguments must contain valid JSON");
    assert_eq!(parsed["cmd"], "printf phase0-tool-ok");
    assert!(function_output["output"]
        .as_str()
        .expect("tool output must be a string")
        .contains("phase0-tool-ok"));
}

#[test]
fn codex_text_sse_event_order_is_preserved() {
    assert_eq!(
        event_types("codex/0.144.x/expected-text-events.jsonl"),
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
}

#[test]
fn codex_function_call_sse_event_order_is_preserved() {
    let events = read_json_lines("codex/0.144.x/expected-tool-events.jsonl");
    let types: Vec<&str> = events
        .iter()
        .map(|event| event["type"].as_str().expect("event type must be a string"))
        .collect();
    assert_eq!(
        types,
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

    let done = &events[4];
    assert_eq!(done["name"], "exec_command");
    serde_json::from_str::<Value>(
        done["arguments"]
            .as_str()
            .expect("done arguments must be a JSON string"),
    )
    .expect("done arguments must contain valid JSON");
}

#[test]
fn llama_native_function_arguments_are_json_strings() {
    let response = read_json("llama-server/10075/native-tool-response.json");
    let tool_call = &response["choices"][0]["message"]["tool_calls"][0];
    let arguments = tool_call["function"]["arguments"]
        .as_str()
        .expect("llama-server arguments must be a string");
    let parsed: Value = serde_json::from_str(arguments).expect("arguments must contain valid JSON");

    assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(tool_call["function"]["name"], "read_file");
    assert_eq!(parsed["path"], "README.md");
}

#[test]
fn llama_native_stream_reassembles_arguments_and_finishes_with_usage() {
    let (events, done) = read_sse_data("llama-server/10075/native-tool-stream.sse");
    let mut arguments = String::new();
    let mut saw_tool_finish = false;

    for event in &events {
        if let Some(fragment) =
            event["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str()
        {
            arguments.push_str(fragment);
        }
        if event["choices"][0]["finish_reason"] == "tool_calls" {
            saw_tool_finish = true;
        }
    }

    let usage = events.last().expect("final usage event");
    assert!(done);
    assert!(saw_tool_finish);
    assert!(usage["choices"]
        .as_array()
        .expect("choices must be an array")
        .is_empty());
    assert!(usage["usage"]["total_tokens"].is_number());
    let parsed: Value =
        serde_json::from_str(&arguments).expect("streamed arguments must reassemble as JSON");
    assert_eq!(parsed["path"], "README.md");
}

#[test]
fn generic_json_bundle_emits_required_function_call_shape() {
    let response = read_json("llama-server/10075/generic-json-tool-response.json");
    let content = response["choices"][0]["message"]["content"]
        .as_str()
        .expect("generic JSON content must be a string");
    let parsed: Value = serde_json::from_str(content).expect("content must contain valid JSON");

    assert_eq!(parsed["type"], "function_call");
    assert_eq!(parsed["name"], "read_file");
    assert_eq!(parsed["arguments"]["path"], "README.md");
}

#[test]
fn malformed_arguments_fixture_remains_invalid_for_recovery_tests() {
    let response = read_json("llama-server/10075/malformed-arguments.json");
    let arguments = response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .expect("malformed arguments must still be represented as a string");

    assert!(serde_json::from_str::<Value>(arguments).is_err());
    assert_eq!(response["_fixture"]["provenance"], "synthetic-mutation");
}

#[test]
fn fixture_manifests_and_text_continuations_remain_parseable() {
    let codex_manifest = read_json("codex/0.144.x/manifest.json");
    let llama_manifest = read_json("llama-server/10075/manifest.json");
    let text = read_json("llama-server/10075/native-text-response.json");
    let tool_result = read_json("llama-server/10075/tool-result-response.json");

    assert_eq!(codex_manifest["client"]["version"], "0.144.5");
    assert_eq!(llama_manifest["server"]["build"], "b10075-76f46ad29");
    assert_eq!(
        llama_manifest["bundles"]["standard"]["name"],
        "unsloth-gemma4-12b-qat-thinking-off"
    );
    assert_eq!(
        llama_manifest["bundles"]["standard"]["minimum_context_tokens"],
        16384
    );
    assert_eq!(
        llama_manifest["bundles"]["standard"]["recommended_context_tokens"],
        32768
    );
    assert_eq!(text["choices"][0]["finish_reason"], "stop");
    assert_eq!(tool_result["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        tool_result["_fixture"]["input_contract"]["tool_call_id"],
        "preserved"
    );
}

#[test]
fn standard_gemma4_bundle_supports_native_tool_call_and_result_continuation() {
    let tool_call = read_json("llama-server/10075/gemma4-native-tool-response.json");
    let tool_result = read_json("llama-server/10075/gemma4-tool-result-response.json");
    let arguments = tool_call["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .expect("Gemma native arguments must be a string");
    let parsed: Value =
        serde_json::from_str(arguments).expect("Gemma native arguments must contain valid JSON");

    assert_eq!(tool_call["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        tool_call["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "read_file"
    );
    assert_eq!(parsed["path"], "README.md");
    assert_eq!(tool_result["choices"][0]["finish_reason"], "stop");
    assert!(tool_result["choices"][0]["message"]["content"]
        .as_str()
        .expect("Gemma tool result continuation must be text")
        .contains("Hoshikage"));
}

#[test]
fn standard_gemma4_native_stream_reassembles_valid_arguments() {
    let (events, done) = read_sse_data("llama-server/10075/gemma4-native-tool-stream.sse");
    let arguments: String = events
        .iter()
        .filter_map(|event| {
            event["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str()
        })
        .collect();

    assert!(done);
    assert_eq!(
        events
            .iter()
            .find_map(|event| event["choices"][0]["finish_reason"].as_str()),
        Some("tool_calls")
    );
    assert!(events.last().expect("Gemma stream usage event")["choices"]
        .as_array()
        .expect("choices must be an array")
        .is_empty());
    let parsed: Value =
        serde_json::from_str(&arguments).expect("Gemma streamed arguments must form valid JSON");
    assert_eq!(parsed["path"], "README.md");
}

#[test]
fn standard_gemma4_generic_json_preserves_complex_argument_types() {
    let response = read_json("llama-server/10075/gemma4-generic-json-complex-response.json");
    let content = response["choices"][0]["message"]["content"]
        .as_str()
        .expect("Gemma generic JSON content must be a string");
    let parsed: Value = serde_json::from_str(content).expect("Gemma content must be valid JSON");
    let arguments = &parsed["arguments"];

    assert_eq!(parsed["type"], "function_call");
    assert_eq!(parsed["name"], "inspect_file");
    assert_eq!(arguments["path"], "資料/My File.md");
    assert_eq!(arguments["line_start"], 17);
    assert_eq!(arguments["include_hidden"], false);
    assert_eq!(arguments["tags"], serde_json::json!(["alpha", "日本語"]));
    assert_eq!(arguments["options"]["mode"], "strict");
    assert_eq!(arguments["options"]["retry"], 2);
    assert_eq!(
        response["_fixture"]["thinking_parameter"],
        "enable_thinking=false"
    );
}
