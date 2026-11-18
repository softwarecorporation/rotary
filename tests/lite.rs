#![cfg(feature = "lite")]

use rx4::lite::{Agent, Provider, ProviderKind, Tool};
use rx4::permissions::{authorize, Decision, Policy};
use serde_json::{json, Value};

fn provider(kind: ProviderKind) -> Provider {
    Provider {
        kind,
        model: "host-model".into(),
        max_output_tokens: 1024,
    }
}

fn tools() -> Vec<Tool> {
    vec![Tool {
        name: "lookup".into(),
        description: "Look up a supplied key".into(),
        input_schema: json!({"type":"object","properties":{"key":{"type":"string"}}}),
    }]
}

fn response(kind: &ProviderKind, calls: &[(&str, &str, Value)]) -> Value {
    match kind {
        ProviderKind::Anthropic => {
            let mut blocks = vec![json!({"type":"text","text":"Checking."})];
            blocks.extend(calls.iter().map(
                |(id, name, args)| json!({"type":"tool_use","id":id,"name":name,"input":args}),
            ));
            json!({"type":"message","role":"assistant","content":blocks})
        }
        ProviderKind::OpenaiCompatible => {
            json!({"choices":[{"message":{"role":"assistant","content":"Checking.","tool_calls":calls.iter().map(|(id, name, args)|json!({"id":id,"type":"function","function":{"name":name,"arguments":args.to_string()}})).collect::<Vec<_>>()}}]})
        }
    }
}

fn result(id: &str) -> Value {
    json!({"type":"tool_result","tool_use_id":id,"content":"Found."})
}

fn state(agent: &Agent) -> Value {
    serde_json::to_value(agent).unwrap()
}

#[test]
fn both_providers_run_real_resumable_turns_with_host_permissions() {
    for kind in [ProviderKind::Anthropic, ProviderKind::OpenaiCompatible] {
        let p = provider(kind.clone());
        let mut agent = Agent::new("Use supplied tools.".into(), json!({"key":"item"}));
        let request = agent.request(&p, &tools()).unwrap();
        assert_eq!(request["model"], "host-model");
        assert_eq!(request["max_tokens"], 1024);
        assert_eq!(request, agent.request(&p, &tools()).unwrap());
        if kind == ProviderKind::Anthropic {
            assert_eq!(request["system"], "Use supplied tools.");
            assert_eq!(request["tools"][0]["input_schema"]["type"], "object");
            assert_eq!(request["messages"][0]["content"], r#"{"key":"item"}"#);
        } else {
            assert_eq!(request["messages"][0]["role"], "system");
            assert_eq!(
                request["tools"][0]["function"]["parameters"]["type"],
                "object"
            );
            assert_eq!(request["stream"], false);
        }
        let reply = agent
            .accept(
                &kind,
                response(&kind, &[("a", "lookup", json!({"key":"item"}))]),
            )
            .unwrap();
        assert_eq!(reply.text, "Checking.");
        assert_eq!(reply.calls.len(), 1);
        let call: rx4::agent::ToolCall = (&reply.calls[0]).into();
        assert_eq!(call.id, "a");
        assert_eq!(
            serde_json::from_str::<Value>(&call.arguments).unwrap(),
            json!({"key":"item"})
        );
        assert_eq!(
            authorize(&Policy::deny_all(), &call.name, &call.arguments, None),
            Decision::Deny
        );
        let mut policy = Policy::deny_all();
        policy.allowlist.push("lookup".into());
        assert_eq!(
            authorize(&policy, &call.name, &call.arguments, None),
            Decision::Allow
        );
        assert!(agent.request(&p, &tools()).is_err());

        // Durable restore while calls are pending must still require results.
        let mut restored: Agent = serde_json::from_value(state(&agent)).unwrap();
        assert!(restored.request(&p, &tools()).is_err());
        restored.results(vec![result("a")]).unwrap();
        let next = restored.request(&p, &tools()).unwrap();
        if kind == ProviderKind::Anthropic {
            assert_eq!(next["messages"][1]["content"][1]["id"], "a");
            assert_eq!(next["messages"][2]["content"][0], result("a"));
        } else {
            assert_eq!(
                next["messages"][2]["tool_calls"][0]["function"]["arguments"],
                r#"{"key":"item"}"#
            );
            assert_eq!(
                next["messages"][3],
                json!({"role":"tool","tool_call_id":"a","content":"Found."})
            );
        }
        let resumed: Agent = serde_json::from_value(state(&restored)).unwrap();
        assert_eq!(next, resumed.request(&p, &tools()).unwrap());
        let final_reply = restored.accept(&kind, response(&kind, &[])).unwrap();
        assert!(final_reply.calls.is_empty());
        assert_eq!(final_reply.text, "Checking.");
    }
}

#[test]
fn signed_thinking_is_opaque_and_survives_restore_and_continuation() {
    let p = provider(ProviderKind::Anthropic);
    let mut agent = Agent::new("Investigate.".into(), json!("evidence"));
    let blocks = json!([
        {"type":"thinking","thinking":"PRIVATE","signature":"opaque/signature==","extra":{"text":"NOT REPLY"}},
        {"type":"redacted_thinking","data":"opaque redacted data"},
        {"type":"text","text":"Public."},
        {"type":"tool_use","id":"t","name":"lookup","input":{}}
    ]);
    let reply = agent.accept(&p.kind, json!({"content":blocks})).unwrap();
    assert_eq!(reply.text, "Public.");
    assert!(!serde_json::to_string(&reply).unwrap().contains("PRIVATE"));
    let mut restored: Agent = serde_json::from_value(state(&agent)).unwrap();
    restored.results(vec![result("t")]).unwrap();
    assert_eq!(
        restored.request(&p, &tools()).unwrap()["messages"][1]["content"],
        blocks
    );
    let compatible = restored
        .request(&provider(ProviderKind::OpenaiCompatible), &tools())
        .unwrap();
    assert_eq!(compatible["messages"][2]["content"], "Public.");
    assert!(!compatible.to_string().contains("PRIVATE"));
}

#[test]
fn invalid_results_and_pending_responses_are_atomic_for_both_providers() {
    for kind in [ProviderKind::Anthropic, ProviderKind::OpenaiCompatible] {
        let mut agent = Agent::new("instructions".into(), json!({}));
        agent
            .accept(
                &kind,
                response(
                    &kind,
                    &[("a", "lookup", json!({})), ("b", "lookup", json!({}))],
                ),
            )
            .unwrap();
        let before = state(&agent);
        for results in [
            vec![],
            vec![result("a")],
            vec![result("a"), result("a")],
            vec![result("a"), result("unknown")],
            vec![
                result("a"),
                json!({"type":"tool_result","content":"missing ID"}),
            ],
            vec![
                result("a"),
                json!({"type":"tool_result","tool_use_id":"b","content":42}),
            ],
        ] {
            assert!(agent.results(results).is_err());
            assert_eq!(state(&agent), before);
        }
        assert!(agent.accept(&kind, response(&kind, &[])).is_err());
        assert_eq!(state(&agent), before);
        agent.results(vec![result("b"), result("a")]).unwrap();
        let after = state(&agent);
        assert!(agent.results(vec![result("a"), result("b")]).is_err());
        assert_eq!(state(&agent), after);
        let mut corrupted = before;
        corrupted["pending"] = json!(["unknown"]);
        assert!(serde_json::from_value::<Agent>(corrupted).is_err());
    }
}

#[test]
fn invalid_provider_responses_never_change_state() {
    for kind in [ProviderKind::Anthropic, ProviderKind::OpenaiCompatible] {
        let mut agent = Agent::new("instructions".into(), json!({}));
        let before = state(&agent);
        let nine: Vec<_> = (0..9)
            .map(|i| (i.to_string(), "lookup", json!({})))
            .collect();
        let nine: Vec<_> = nine
            .iter()
            .map(|(id, name, args)| (id.as_str(), *name, args.clone()))
            .collect();
        for bad in [
            json!({}),
            json!({"error":"failed"}),
            response(&kind, &[("", "lookup", json!({}))]),
            response(&kind, &[("a", "", json!({}))]),
            response(&kind, &[("a", "lookup", json!([]))]),
            response(
                &kind,
                &[("a", "lookup", json!({})), ("a", "lookup", json!({}))],
            ),
            response(&kind, &nine),
        ] {
            assert!(agent.accept(&kind, bad).is_err());
            assert_eq!(state(&agent), before);
        }
    }
    let mut agent = Agent::new(String::new(), json!(null));
    let before = state(&agent);
    for bad in [
        json!({"choices":[{"message":{"role":"assistant","content":12}}]}),
        json!({"choices":[{"message":{"role":"user","content":"no"}}]}),
        json!({"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"lookup","arguments":"not JSON"}}]}}]}),
    ] {
        assert!(agent.accept(&ProviderKind::OpenaiCompatible, bad).is_err());
        assert_eq!(state(&agent), before);
    }
    for bad in [
        json!({"content":[{"type":"thinking","thinking":"unsigned"}]}),
        json!({"content":[{"type":"text","text":42}]}),
        json!({"content":[]}),
    ] {
        assert!(agent.accept(&ProviderKind::Anthropic, bad).is_err());
        assert_eq!(state(&agent), before);
    }
}

#[test]
fn context_response_results_and_configuration_are_bounded() {
    for kind in [ProviderKind::Anthropic, ProviderKind::OpenaiCompatible] {
        let p = provider(kind.clone());
        let mut agent = Agent::new("instructions".into(), json!({}));
        let before = state(&agent);
        let oversized = response(
            &kind,
            &[("a", "lookup", json!({"data":"x".repeat(65_536)}))],
        );
        assert!(agent.accept(&kind, oversized).is_err());
        assert_eq!(state(&agent), before);
        let huge = Agent::new("x".repeat(65_536), json!({}));
        assert!(huge.request(&p, &tools()).is_err());
        assert!(serde_json::from_value::<Agent>(state(&huge)).is_err());
        let huge_evidence = Agent::new(String::new(), json!("x".repeat(65_536)));
        assert!(huge_evidence.request(&p, &tools()).is_err());
        let mut huge_tools = tools();
        huge_tools[0].description = "x".repeat(65_536);
        assert!(agent.request(&p, &huge_tools).is_err());
        for model in [String::new(), "x".repeat(257), "bad model".into()] {
            assert!(agent
                .request(&Provider { model, ..p.clone() }, &tools())
                .is_err());
        }
        for max_output_tokens in [0, 8193] {
            assert!(agent
                .request(
                    &Provider {
                        max_output_tokens,
                        ..p.clone()
                    },
                    &tools()
                )
                .is_err());
        }
        agent
            .accept(&kind, response(&kind, &[("a", "lookup", json!({}))]))
            .unwrap();
        let before = state(&agent);
        assert!(agent
            .results(vec![
                json!({"type":"tool_result","tool_use_id":"a","content":"x".repeat(65_536)})
            ])
            .is_err());
        assert_eq!(state(&agent), before);
    }
    // Each response fits on its own; the cumulative canonical context does not.
    let mut agent = Agent::new(String::new(), json!({}));
    let chunk = json!({"content":[{"type":"text","text":"x".repeat(33_000)}]});
    agent
        .accept(&ProviderKind::Anthropic, chunk.clone())
        .unwrap();
    let before = state(&agent);
    assert!(agent.accept(&ProviderKind::Anthropic, chunk).is_err());
    assert_eq!(state(&agent), before);
}

#[test]
fn valid_boundary_configuration_and_eight_calls_continue_with_denial_results() {
    for kind in [ProviderKind::Anthropic, ProviderKind::OpenaiCompatible] {
        let p = Provider {
            kind: kind.clone(),
            model: "m".repeat(256),
            max_output_tokens: 8192,
        };
        let mut agent = Agent::new("Use tools.".into(), json!({}));
        assert!(agent.request(&p, &[]).unwrap().get("tools").is_none());
        let calls: Vec<_> = (0..8)
            .map(|i| (i.to_string(), "lookup", json!({})))
            .collect();
        let calls: Vec<_> = calls
            .iter()
            .map(|(id, name, args)| (id.as_str(), *name, args.clone()))
            .collect();
        let reply = agent.accept(&kind, response(&kind, &calls)).unwrap();
        assert_eq!(reply.calls.len(), 8);
        let results = reply.calls.iter().map(|call| {
            let native: rx4::agent::ToolCall = call.into();
            assert_eq!(authorize(&Policy::deny_all(), &native.name, &native.arguments, None), Decision::Deny);
            json!({"type":"tool_result","tool_use_id":call.id,"content":"Denied by host.","is_error":true})
        }).collect();
        agent.results(results).unwrap();
        assert!(agent.request(&p, &tools()).is_ok());
        assert!(agent
            .accept(&kind, response(&kind, &[]))
            .unwrap()
            .calls
            .is_empty());
    }
}
