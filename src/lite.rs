//! A resumable, transport-free agent state machine, not native `Agent` parity.
//!
//! The host owns network requests, durable state, clocks, authorization, and
//! actions. `request` builds a non-streaming provider body; `accept` records a
//! validated response. Before running any returned call, the host must consult
//! its authorizer (for example [`crate::permissions`]). Feed authorized outputs
//! or denial/error outputs to `results`, then request the next turn. Nothing in
//! this module executes tools or grants authority.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;

const LIMIT: usize = 65_536;

/// Provider configuration. Models are 1–256 bytes, without whitespace/control
/// characters; output tokens must be in `1..=8192`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provider {
    pub kind: ProviderKind,
    pub model: String,
    pub max_output_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Anthropic,
    OpenaiCompatible,
}

/// A host-provided tool definition. The input schema must describe an object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reply {
    pub text: String,
    pub calls: Vec<Call>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

impl From<&Call> for crate::agent::ToolCall {
    fn from(call: &Call) -> Self {
        Self {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.to_string(),
        }
    }
}

/// Serializable canonical conversation and pending tool IDs.
///
/// State, request bodies, and incoming responses are each bounded to 65,536
/// serialized JSON bytes. Bounds are checked before every successful transition;
/// an oversized initial value is rejected by `request`/`accept`, since `new` is
/// infallible. Failures leave the agent unchanged. Serialization contains opaque
/// provider thinking and may contain sensitive evidence: store it accordingly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "State")]
pub struct Agent {
    instructions: String,
    messages: Vec<Message>,
    pending: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "content", rename_all = "snake_case")]
enum Message {
    Evidence(String),
    Assistant(Vec<Value>),
    Results(Vec<Value>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    instructions: String,
    messages: Vec<Message>,
    pending: Vec<String>,
}

impl TryFrom<State> for Agent {
    type Error = String;

    fn try_from(state: State) -> Result<Self, String> {
        let mut messages = state.messages.into_iter();
        let Some(Message::Evidence(evidence)) = messages.next() else {
            return Err("state must begin with evidence".into());
        };
        let mut agent = Self {
            instructions: state.instructions,
            messages: vec![Message::Evidence(evidence)],
            pending: Vec::new(),
        };
        bounded(&agent)?;
        for message in messages {
            match message {
                Message::Assistant(content) => {
                    agent.accept(&ProviderKind::Anthropic, json!({"content":content}))?;
                }
                Message::Results(results) => agent.results(results)?,
                Message::Evidence(_) => return Err("unexpected evidence in state".into()),
            }
        }
        if agent.pending != state.pending {
            return Err("invalid pending IDs in state".into());
        }
        Ok(agent)
    }
}

impl Agent {
    pub fn new(instructions: String, evidence: Value) -> Self {
        Self {
            instructions,
            messages: vec![Message::Evidence(evidence.to_string())],
            pending: Vec::new(),
        }
    }

    /// Build a request without performing I/O. Pending results must be supplied
    /// first. Repeated calls produce the same body and do not alter state.
    pub fn request(&self, provider: &Provider, tools: &[Tool]) -> Result<Value, String> {
        bounded(self)?;
        if !self.pending.is_empty() {
            return Err("tool results are pending".into());
        }
        if provider.model.is_empty()
            || provider.model.len() > 256
            || provider
                .model
                .chars()
                .any(|c| c.is_whitespace() || c.is_control())
            || !(1..=8192).contains(&provider.max_output_tokens)
        {
            return Err("invalid model or output token limit".into());
        }
        let mut names = HashSet::new();
        for tool in tools {
            if tool.name.trim().is_empty()
                || !names.insert(&tool.name)
                || !tool.input_schema.is_object()
                || tool.input_schema["type"] != "object"
            {
                return Err("invalid tool definition".into());
            }
        }
        let mut messages = Vec::new();
        if provider.kind == ProviderKind::OpenaiCompatible {
            messages.push(json!({"role":"system","content":self.instructions}));
        }
        for message in &self.messages {
            match message {
                Message::Evidence(text) => messages.push(json!({"role":"user","content":text})),
                Message::Assistant(blocks) if provider.kind == ProviderKind::OpenaiCompatible => {
                    let text: String = blocks
                        .iter()
                        .filter(|b| b["type"] == "text")
                        .filter_map(|b| b["text"].as_str())
                        .collect();
                    let calls: Vec<Value> = blocks.iter().filter(|b| b["type"] == "tool_use")
                        .map(|b| json!({"id":b["id"],"type":"function","function":{"name":b["name"],"arguments":b["input"].to_string()}})).collect();
                    let mut message = json!({"role":"assistant","content":text});
                    if !calls.is_empty() {
                        message["tool_calls"] = json!(calls);
                    }
                    messages.push(message);
                }
                Message::Results(blocks) if provider.kind == ProviderKind::OpenaiCompatible => {
                    for block in blocks {
                        messages.push(json!({"role":"tool","tool_call_id":block["tool_use_id"],"content":block["content"]}));
                    }
                }
                Message::Assistant(blocks) => {
                    messages.push(json!({"role":"assistant","content":blocks}))
                }
                Message::Results(blocks) => messages.push(json!({"role":"user","content":blocks})),
            }
        }
        let tools: Vec<Value> = tools.iter().map(|t| match provider.kind {
            ProviderKind::Anthropic => json!({"name":t.name,"description":t.description,"input_schema":t.input_schema}),
            ProviderKind::OpenaiCompatible => json!({"type":"function","function":{"name":t.name,"description":t.description,"parameters":t.input_schema}}),
        }).collect();
        let mut body = json!({"model":provider.model,"max_tokens":provider.max_output_tokens,"messages":messages});
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        match provider.kind {
            ProviderKind::Anthropic => body["system"] = json!(self.instructions),
            ProviderKind::OpenaiCompatible => body["stream"] = json!(false),
        }
        bounded(&body)?;
        Ok(body)
    }

    /// Accept one complete, non-streaming provider response. Thinking blocks are
    /// preserved verbatim in canonical state but never returned as reply text.
    pub fn accept(&mut self, kind: &ProviderKind, value: Value) -> Result<Reply, String> {
        bounded(self)?;
        bounded(&value)?;
        if !self.pending.is_empty() {
            return Err("tool results are pending".into());
        }
        let blocks = match kind {
            ProviderKind::Anthropic => {
                if value.get("role").is_some_and(|r| r != "assistant")
                    || value.get("type").is_some_and(|t| t != "message")
                    || value.get("error").is_some()
                {
                    return Err("invalid provider response".into());
                }
                value["content"]
                    .as_array()
                    .ok_or("invalid provider content")?
                    .clone()
            }
            ProviderKind::OpenaiCompatible => {
                let choices = value["choices"].as_array().ok_or("invalid choices")?;
                if choices.len() != 1 || value.get("error").is_some() {
                    return Err("expected one response choice".into());
                }
                let message = &choices[0]["message"];
                if message["role"] != "assistant" {
                    return Err("invalid assistant message".into());
                }
                let mut blocks = Vec::new();
                match message.get("content") {
                    Some(Value::String(text)) => blocks.push(json!({"type":"text","text":text})),
                    Some(Value::Null) | None => {}
                    _ => return Err("invalid assistant text".into()),
                }
                if let Some(calls) = message.get("tool_calls") {
                    for call in calls.as_array().ok_or("invalid tool calls")? {
                        if call["type"] != "function" {
                            return Err("invalid tool call type".into());
                        }
                        let arguments: Value = serde_json::from_str(
                            call["function"]["arguments"]
                                .as_str()
                                .ok_or("invalid tool arguments")?,
                        )
                        .map_err(|_| "invalid tool arguments")?;
                        blocks.push(json!({"type":"tool_use","id":call["id"],"name":call["function"]["name"],"input":arguments}));
                    }
                }
                if message.get("function_call").is_some() {
                    return Err("legacy function calls are unsupported".into());
                }
                blocks
            }
        };
        if blocks.is_empty() {
            return Err("empty provider response".into());
        }
        let mut reply = Reply {
            text: String::new(),
            calls: Vec::new(),
        };
        let mut ids = HashSet::new();
        for block in &blocks {
            match block["type"].as_str() {
                Some("text") => reply
                    .text
                    .push_str(block["text"].as_str().ok_or("invalid text")?),
                Some("thinking")
                    if block["thinking"].is_string() && block["signature"].is_string() => {}
                Some("redacted_thinking") if block["data"].is_string() => {}
                Some("tool_use") => {
                    let id = nonempty(&block["id"])?;
                    let name = nonempty(&block["name"])?;
                    if !ids.insert(id) || !block["input"].is_object() {
                        return Err("duplicate call ID or non-object arguments".into());
                    }
                    reply.calls.push(Call {
                        id: id.into(),
                        name: name.into(),
                        arguments: block["input"].clone(),
                    });
                }
                _ => return Err("unsupported provider content".into()),
            }
        }
        if reply.calls.len() > 8 {
            return Err("too many tool calls".into());
        }
        let mut next = self.clone();
        next.pending = reply.calls.iter().map(|c| c.id.clone()).collect();
        next.messages.push(Message::Assistant(blocks));
        bounded(&next)?;
        *self = next;
        Ok(reply)
    }

    /// Supply exactly one result per pending call, in any order. Each result is
    /// an Anthropic-shaped `{ "type": "tool_result", "tool_use_id": "…",
    /// "content": "…", "is_error": false }` block; `is_error` is optional.
    /// Content may also be an array of text blocks. Denials are host-generated
    /// error results, not execution performed by this agent.
    pub fn results(&mut self, results: Vec<Value>) -> Result<(), String> {
        bounded(self)?;
        bounded(&results)?;
        if self.pending.is_empty() || results.len() != self.pending.len() {
            return Err("expected exactly the pending tool results".into());
        }
        let mut ids = HashSet::new();
        for result in &results {
            let id = nonempty(&result["tool_use_id"])?;
            if result["type"] != "tool_result"
                || !self.pending.iter().any(|p| p == id)
                || !ids.insert(id)
                || result.get("is_error").is_some_and(|v| !v.is_boolean())
            {
                return Err("invalid tool result or ID".into());
            }
            match &result["content"] {
                Value::String(_) => {}
                Value::Array(blocks)
                    if blocks
                        .iter()
                        .all(|b| b["type"] == "text" && b["text"].is_string()) => {}
                _ => return Err("invalid result content".into()),
            }
        }
        let mut next = self.clone();
        next.pending.clear();
        next.messages.push(Message::Results(results));
        bounded(&next)?;
        *self = next;
        Ok(())
    }
}

fn nonempty(value: &Value) -> Result<&str, String> {
    value
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| "expected nonempty string".into())
}

fn bounded(value: &impl Serialize) -> Result<(), String> {
    if serde_json::to_vec(value).map_err(|e| e.to_string())?.len() > LIMIT {
        Err("65,536 byte limit reached".into())
    } else {
        Ok(())
    }
}
