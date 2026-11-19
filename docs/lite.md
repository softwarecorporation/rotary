# Rotary lite host contract

`rx4::lite::Agent` owns the portable provider conversation and tool-result state machine. It compiles to `wasm32-unknown-unknown` with `--no-default-features --features lite`; it is not a replacement for every capability of native `rx4::Agent`.

## Host usage

```rust
use rx4::lite::{Agent, Provider, ProviderKind, Tool};
use serde_json::json;

let mut agent = Agent::new("Investigate using configured diagnostics.".into(), json!({"status":503}));
let provider = Provider { kind: ProviderKind::Anthropic, model: "configured-model".into(), max_output_tokens: 1024 };
let tools = vec![Tool { name: "inspect".into(), description: "Read configured diagnostics".into(), input_schema: json!({"type":"object","properties":{},"additionalProperties":false}) }];
let body = agent.request(&provider, &tools)?;
// Host sends body using fetch and its server-side provider secret, then:
let reply = agent.accept(&provider.kind, json!({"content":[{"type":"tool_use","id":"c1","name":"inspect","input":{}}]}))?;
// Host validates the registered tool/schema and consults its authorizer before I/O.
let call: rx4::agent::ToolCall = (&reply.calls[0]).into();
let mut policy = rx4::permissions::Policy::deny_all();
policy.allowlist.push("inspect".into());
assert_eq!(rx4::permissions::authorize(&policy, &call.name, &call.arguments, None), rx4::permissions::Decision::Allow);
agent.results(vec![json!({"type":"tool_result","tool_use_id":"c1","content":"worker stopped"})])?;
let continuation = agent.request(&provider, &tools)?;
# Ok::<(), String>(())
```

The response above is an illustrative fixture, not a live provider call. The host sends real requests and passes full parsed responses to `accept`. Tool definitions do not execute tools or establish authority. Unknown tools, argument schemas and production permissions remain host checks; `Ask` must never be treated as `Allow`. Return denied calls as error results or stop the investigation.

## State and supported protocols

- Anthropic Messages and OpenAI-compatible Chat Completions, non-streaming, text and object-valued tool calls. Model name and output limits are explicit.
- Serializable conversation and pending IDs; restoration validates the history. Invalid responses/results leave state unchanged. Results must match all pending IDs exactly once; another request is rejected until they arrive.
- Repeated `request` calls are deterministic body construction, not execution or a charged-request counter. Hosts prevent duplicate response acceptance, implement deadlines, bound turns and spending, and decide when a no-tool response completes the task.
- Up to eight calls per response; serialized state, request, response and result bodies are limited to 65536 bytes. An oversized initial value fails on use. These bounds are not token pricing or invoice caps.
- Anthropic signed thinking is retained verbatim for continuation and excluded from reply summaries. Serialized agent state can contain sensitive evidence and opaque provider data; never log it indiscriminately. Use the shared `rx4::secrets` redactor on application evidence and audit output, not signed provider blocks.
- The same `ToolCall`, permission evaluator, shell scanner, and redactor are shared with the native API. Tokio's portable channel support is present; its runtime, filesystem and timers are not.

No shell/filesystem tools, MCP, native provider transport, streaming, SQLite/session backend, compaction, background subprocesses, computer use, native sandbox, autonomous scheduler, retry policy or production approval policy is supplied by lite. Workers hosts use fetch, durable storage and alarms for those host responsibilities. Operating-system capabilities cannot be enabled inside Workers by adding native features.

## Checks

```sh
cargo build --locked --lib --no-default-features --features lite --target wasm32-unknown-unknown
cargo test --locked --no-default-features --features lite --lib --test lite
cargo test --locked --lib
cargo check --locked --all-features
```
