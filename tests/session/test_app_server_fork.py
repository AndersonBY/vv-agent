"""Fork creation and inherited context work on every SQL backend."""

from dataclasses import replace
from hashlib import sha256

from vv_agent import Agent, RunConfig, ScriptedModelProvider, ThreadForkParams
from vv_agent.app_server import AppServer, AppServerClient, ChannelTransport, DefaultAppServerHost
from vv_agent.memory import MemoryManager
from vv_agent.model import ModelRef
from vv_agent.session.context import project_context
from vv_agent.session.context_view import context_view
from vv_agent.session.records import SessionSpec
from vv_agent.types import LLMResponse, Message, ToolArtifactRef, ToolCall
from vv_agent.workspace.local import LocalWorkspaceBackend

from .test_compaction import SUMMARY, tool_history


def send(server, identity, method, params):
    server.processor.process_message("test", {"jsonrpc": "2.0", "id": identity, "method": method, "params": params})
    while (wire := server.transport.receive_outbound(timeout=5)).get("id") != identity:
        pass
    assert "result" in wire, wire
    return wire["result"]


def server_for(store, host):
    server = AppServer(store=store, host=host, transport=ChannelTransport(connection_id="test", outbound_capacity=10000))
    send(server, "init", "initialize", {"clientInfo": {"name": "test"}})
    return server


def run_turn(server, sid, identity, expected=None):
    result = send(
        server,
        "turn-" + identity,
        "turn/start",
        {"threadId": sid, "inputId": identity, "expectedTurnId": expected, "input": [{"type": "text", "text": identity}]},
    )
    server.run_adapter.join()
    return result["turnId"]


def test_selected_model_settings_limits_and_clear_are_frozen_default(store, tmp_path, monkeypatch):
    calls = []
    base = ScriptedModelProvider.from_callback("test", "base", lambda r: calls.append(r) or LLMResponse("base"))
    selected = ScriptedModelProvider.from_callback(
        "test", "small", lambda r: calls.append(r) or LLMResponse("small")
    ).with_token_limits(64000, 8000)
    selected_config = selected.resolve(ModelRef.named("small"))
    host = DefaultAppServerHost(agent=Agent("test", "Be precise."), run_config=RunConfig(model_provider=base))
    original = host.resolve_model
    monkeypatch.setattr(
        host, "resolve_model", lambda req: (selected.llm, selected_config) if req["model"] == "small" else original(req)
    )
    a = server_for(store, host)
    try:
        send(a, "create", "thread/start", {"inputId": "source"})
        send(
            a,
            "select",
            "thread/model/select",
            {
                "threadId": "source",
                "inputId": "select",
                "selection": {"model": "small", "model_settings": {"max_tokens": 1000}, "route": {"opaque": "route"}},
            },
        )
        first = run_turn(a, "source", "first")
        original_records = store.read_state("source")[1]
        client = AppServerClient(server=a, transport=a.transport)
        fork = client.fork_thread(ThreadForkParams("source", "copy", through_turn_id=first, workspace={"root": "prepared"}))
        sid = fork["threadId"]
        assert context_view(store, sid)["limits"] == context_view(store, "source")["limits"]
        assert context_view(store, sid)["limits"]["outputReserve"] == 1000
        t1 = run_turn(a, sid, "new")
        assert calls[-1].model == "small" and calls[-1].model_settings.max_tokens == 1000
        assert context_view(store, sid)["limits"]["contextWindow"] == 64000
        send(
            a, "other", "thread/model/select", {"threadId": sid, "inputId": "other", "selection": {"model": "base", "route": {}}}
        )
        t2 = run_turn(a, sid, "other-model", t1)
        assert calls[-1].model == "base"
        send(a, "clear", "thread/model/select", {"threadId": sid, "inputId": "clear", "selection": None})
        run_turn(a, sid, "restore", t2)
        assert calls[-1].model == "small" and calls[-1].model_settings.max_tokens == 1000
        assert store.read_state("source")[1] == original_records
    finally:
        a.kernel.close()


def test_accepted_summary_artifacts_and_prepared_workspace_binding(store, tmp_path, monkeypatch):
    backend = LocalWorkspaceBackend(tmp_path / "source")
    target = LocalWorkspaceBackend(tmp_path / "prepared")
    artifact_path = ".vv-agent/artifacts/session-source/old.txt"
    body = "archived evidence"
    backend.write_text_exclusive(artifact_path, body)
    # Artifact roots differ from business workspaces; fork copies only missing immutable evidence.
    assert not target.exists(artifact_path)
    artifact = ToolArtifactRef(artifact_path, "text/plain", "utf-8", len(body.encode()), sha256(body.encode()).hexdigest())
    history = tool_history()
    history[3].artifact_ref = artifact
    history.append(Message("user", "latest request"))
    calls = []

    def complete(request):
        calls.append(request)
        if request.metadata.get("purpose") == "compaction":
            return LLMResponse(SUMMARY)
        if request.messages[-1].content == "fork-turn":
            return LLMResponse("", [ToolCall("recover", "read_file", {"path": artifact_path})])
        return LLMResponse("done")

    provider = ScriptedModelProvider.from_callback("test", "m", complete)

    host = DefaultAppServerHost(
        agent=Agent("test", "Be precise."),
        run_config=RunConfig(model_provider=provider, workspace_backend=backend, metadata={"memory_keep_recent_messages": 1}),
    )
    seen = []

    def binding(req):
        seen.append(req)
        assert host._run_config is not None
        return replace(host._run_config, workspace_backend=target if req.metadata.get("workspace") else backend)

    monkeypatch.setattr(host, "build_run_config", binding)
    a = server_for(store, host)
    try:
        with store.atomic() as tx:
            tx.create(
                SessionSpec(
                    "source",
                    "local",
                    ".",
                    attributes={
                        "app_server": {"agent_key": "default", "cwd": None, "metadata": {}},
                        "seed": {"messages": [m.to_dict() for m in history], "shared_state": {}},
                    },
                ),
                consumers=("events", "traces", "app_server"),
            )
        first = run_turn(a, "source", "first")
        send(a, "compact", "thread/compact/start", {"threadId": "source", "actionId": "summary"})
        a.run_adapter.join()
        state, records, _ = store.read_state("source")
        compactions = [r for r in records if r.record.kind == "context_compacted"]
        assert compactions, str(
            [
                (
                    r.record.kind,
                    r.record._payload.get("error"),
                    r.record._payload.get("status"),
                    r.record._payload.get("reason"),
                    r.record._payload.get("result"),
                )
                for r in records
                if r.record.kind in {"turn_ended", "input_applied", "op_completed"}
            ]
        )
        compact = compactions[-1]
        assert compact.record._payload["evidence_manifest"]["artifacts"][0]["artifact_ref"] == artifact.to_dict()
        params = {"threadId": "source", "forkId": "copy", "workspace": {"snapshot": "prepared"}}
        fork = send(a, "fork", "thread/fork", params)
        sid = fork["threadId"]
        baseline = store.read_state(sid)[1][0].record._payload["attributes"]["app_server_context"]["task"]
        assert baseline["initial_messages"] == [] and baseline["initial_shared_state"] == {}
        assert baseline["user_prompt"] == "" and "vv_session" not in baseline["metadata"]
        copied = project_context(store.read_state(sid)[1], store.read_state(sid)[0])
        assert [m.to_dict() for m in copied] == [m.to_dict() for m in project_context(records, state)]
        assert MemoryManager().compaction_evidence(copied)["artifacts"][0]["artifact_ref"] == artifact.to_dict()
        assert context_view(store, sid)["lastCompaction"] == context_view(store, "source")["lastCompaction"]
        assert target.read_bytes(artifact_path) == body.encode()
        assert any(req.metadata.get("workspace") == params["workspace"] for req in seen)
        assert len(store.read_state(sid)[1]) == 1 and not store.peek_inbox(sid)
        # Earlier complete boundaries cannot inherit a later accepted summary.
        early = send(a, "early", "thread/fork", {"threadId": "source", "forkId": "early", "throughTurnId": first})
        assert context_view(store, early["threadId"])["lastCompaction"] is None
        run_turn(a, sid, "fork-turn")
        assert any(m.name == "memory_summary" for m in calls[-1].messages)
        assert any(m.role == "tool" and body in m.content for m in calls[-1].messages)
        assert store.read_state("source")[1] == records
        (target._artifact_root / artifact_path).write_text("corrupt")
        a.processor.process_message(
            "test", {"jsonrpc": "2.0", "id": "bad", "method": "thread/fork", "params": params | {"forkId": "corrupt"}}
        )
        while (wire := a.transport.receive_outbound(timeout=5)).get("id") != "bad":
            pass
        assert wire["error"]["code"] == -32602 and "artifact" in wire["error"]["message"]
        assert send(a, "retry", "thread/fork", params) == fork | {"replayed": True}
    finally:
        a.kernel.close()
