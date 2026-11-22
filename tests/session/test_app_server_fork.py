"""Fork creation and inherited context work on every SQL backend."""

from dataclasses import replace
from hashlib import sha256
from time import monotonic, sleep

from vv_agent import Agent, RunConfig, ScriptedModelProvider, ThreadForkParams
from vv_agent.app_server import AppServer, AppServerClient, ChannelTransport, DefaultAppServerHost
from vv_agent.kernel._content import Principal
from vv_agent.kernel._requirements import SessionRequirements
from vv_agent.memory import MemoryManager
from vv_agent.model import ModelRef
from vv_agent.session.content_runtime import SESSION_PRINCIPAL_ATTRIBUTES, SessionContentWriter
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.context import project_context
from vv_agent.session.context_view import context_view
from vv_agent.session.fork import fork_thread
from vv_agent.session.projection import (
    fold_hydrated_records,
    hydrate_record_payload,
    hydrate_records,
    hydration_for_session_store,
)
from vv_agent.session.reducer import _fold_factory_for
from vv_agent.session.runtime import static_model_selection_requirements
from vv_agent.session.surfaces import _verified_routing_cache
from vv_agent.types import LLMResponse, Message, ToolArtifactRef, ToolCall
from vv_agent.workspace.local import LocalWorkspaceBackend

from .helpers import create_fixture_session, wrap_fixture_store
from .test_compaction import SUMMARY, tool_history


def request(server, identity, method, params):
    server.processor.process_message("test", {"jsonrpc": "2.0", "id": identity, "method": method, "params": params})
    while (wire := server.transport.receive_outbound(timeout=5)).get("id") != identity:
        pass
    return wire


def send(server, identity, method, params):
    wire = request(server, identity, method, params)
    assert "result" in wire, wire
    return wire["result"]


def server_for(store, host, state_path):
    server = AppServer(
        store=store,
        content_key_provider=LocalKeyProvider(b"f" * 32, state_path=state_path),
        host=host,
        transport=ChannelTransport(connection_id="test", outbound_capacity=10000),
    )
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
    turn_id = result["turnId"]
    deadline = monotonic() + 5
    while monotonic() < deadline:
        server.kernel._worker.wake(sid)
        server.run_adapter.join()
        state, records, _ = server.kernel.store.read_state(sid)
        if state.active_turn_id is None and any(
            row.record.kind == "turn_ended" and row.record.turn_id == turn_id for row in records
        ):
            return turn_id
        sleep(0.01)
    state, records, _ = server.kernel.store.read_state(sid)
    detail = [
        (
            row.record.kind,
            row.record.operation_id,
            row.record._payload.get("status"),
            row.record._payload.get("error"),
            row.record._payload.get("reason"),
            row.record._payload.get("result"),
        )
        for row in records
        if row.record.kind in {"input_applied", "turn_started", "op_completed", "turn_ended"}
    ]
    raise AssertionError(
        f"Turn did not finish: {turn_id}; active={state.active_turn_id}; "
        f"phase={state.phase}; next_drive_delay={server.kernel.store.next_drive_delay_ms(sid)}; "
        f"lease={server.kernel.store.lease_status(sid)}; "
        f"worker_error={server.kernel._worker.last_error!r}; "
        f"records={detail}"
    )


def set_write_level(store, level):
    store._rows("UPDATE sk_settings SET write_level=%s WHERE singleton=1", (level,))


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
    set_write_level(store, 2)
    a = server_for(store, host, tmp_path / "selected-model-keys.sqlite3")
    try:
        send(a, "create", "thread/start", {"inputId": "source"})
        source_creation = store.read_state("source")[1][0].record
        assert source_creation.feature_level == 2
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
        source_routing, _ = _verified_routing_cache(store, "source")
        handler_version = source_routing.requirements.handler_version
        assert handler_version is not None
        selected_requirements = static_model_selection_requirements(
            {"model": "small", "route": {"opaque": "route"}},
            resolve_model=host.resolve_model,
            handler_version=handler_version,
        )
        initial_requirements = static_model_selection_requirements(
            {"model": "base", "route": {}},
            resolve_model=host.resolve_model,
            handler_version=handler_version,
        )
        expected_bootstrap = SessionRequirements.normalized(
            level=2,
            handler_version=handler_version,
            capabilities={
                capability for capability in source_routing.requirements.capabilities if not capability.startswith("model:")
            }
            | set(selected_requirements.capabilities),
        )
        client = AppServerClient(server=a, transport=a.transport)
        fork = client.fork_thread(ThreadForkParams("source", "copy", through_turn_id=first, workspace={"root": "prepared"}))
        sid = fork["threadId"]
        fork_creation = store.read_state(sid)[1][0].record
        assert fork_creation.feature_level == 2
        fork_bootstrap = SessionRequirements.from_dict(fork_creation._payload["bootstrap"])
        assert fork_bootstrap == expected_bootstrap
        assert set(selected_requirements.capabilities) <= set(fork_bootstrap.capabilities)
        assert set(initial_requirements.capabilities).isdisjoint(fork_bootstrap.capabilities)
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
        cleared_state, cleared_records, _ = store.read_state(sid)
        hydration = hydration_for_session_store(store, sid)
        clear_payload = next(
            payload
            for row in cleared_records
            if row.record.kind == "input_applied"
            and (payload := hydrate_record_payload(row, hydration))["input"]["input_id"] == "clear"
        )
        assert clear_payload["disposition"] == "applied"
        assert clear_payload["input"]["payload"] == {"selection": None}
        assert "model_binding" not in clear_payload
        assert cleared_state.model_selection is None
        assert calls[-1].model == "small" and calls[-1].model_settings.max_tokens == 1000
        assert store.read_state("source")[1] == original_records
    finally:
        a.kernel.close()


def test_level_two_fork_can_select_model_before_first_turn(store, tmp_path, monkeypatch):
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
    set_write_level(store, 2)
    server = server_for(store, host, tmp_path / "fork-before-turn-keys.sqlite3")
    try:
        send(server, "create", "thread/start", {"inputId": "source"})
        source_creation = store.read_state("source")[1][0].record
        fork = send(server, "fork", "thread/fork", {"threadId": "source", "forkId": "copy"})
        sid = fork["threadId"]
        fork_creation = store.read_state(sid)[1][0].record
        assert source_creation.feature_level == 2
        assert fork_creation.feature_level == 2
        assert fork_creation._payload["bootstrap"] == source_creation._payload["bootstrap"]

        send(
            server,
            "select",
            "thread/model/select",
            {"threadId": sid, "inputId": "select", "selection": {"model": "small", "route": {}}},
        )
        run_turn(server, sid, "fork-turn")

        assert calls[-1].model == "small"
        assert not store.read_state("source")[0].turns
        assert not store.read_state(sid)[0].active_turn_id
    finally:
        server.kernel.close()


def test_level_one_fork_stays_level_one_and_keeps_turn_flow(store, tmp_path):
    calls = []
    provider = ScriptedModelProvider.from_callback("test", "m", lambda r: calls.append(r) or LLMResponse("done"))
    host = DefaultAppServerHost(agent=Agent("test", "Answer."), run_config=RunConfig(model_provider=provider))
    set_write_level(store, 1)
    server = server_for(store, host, tmp_path / "fork-level-one-keys.sqlite3")
    try:
        send(server, "create", "thread/start", {"inputId": "source"})
        fork = send(server, "fork", "thread/fork", {"threadId": "source", "forkId": "copy"})
        sid = fork["threadId"]
        source_creation = store.read_state("source")[1][0].record
        fork_creation = store.read_state(sid)[1][0].record
        assert source_creation.feature_level == 1
        assert fork_creation.feature_level == 1
        assert "bootstrap" not in source_creation._payload
        assert "bootstrap" not in fork_creation._payload
        assert store._one("SELECT write_level FROM sk_settings WHERE singleton=1")[0] == 1

        rejected = request(
            server,
            "select",
            "thread/model/select",
            {"threadId": sid, "inputId": "select", "selection": {"model": "m", "route": {}}},
        )

        assert rejected["error"]["code"] == -32602
        assert rejected["error"]["data"]["reason"] == "model_selection_unavailable"
        assert not store.peek_inbox(sid)
        run_turn(server, sid, "fork-turn")
        assert calls[-1].model == "m"
        assert not store.read_state("source")[0].turns
        assert not store.read_state(sid)[0].active_turn_id
    finally:
        server.kernel.close()


def test_level_two_fork_rejects_level_one_target_with_typed_reason(store, tmp_path):
    provider = ScriptedModelProvider.new("test", "m", [])
    host = DefaultAppServerHost(agent=Agent("test", "Answer."), run_config=RunConfig(model_provider=provider))
    set_write_level(store, 2)
    server = server_for(store, host, tmp_path / "fork-ceiling-keys.sqlite3")
    try:
        send(server, "create", "thread/start", {"inputId": "source"})
        assert store.read_state("source")[1][0].record.feature_level == 2
        set_write_level(store, 1)

        rejected = request(
            server,
            "fork",
            "thread/fork",
            {"threadId": "source", "forkId": "copy"},
        )

        assert rejected["error"]["code"] == -32602
        assert rejected["error"]["data"] == {"reason": "model_selection_unavailable"}
    finally:
        server.kernel.close()


def test_direct_fork_copies_persisted_principal_attributes_once(store, tmp_path):
    host = DefaultAppServerHost(
        agent=Agent("test", "Be precise."),
        run_config=RunConfig(
            model_provider=ScriptedModelProvider.new("test", "m", [LLMResponse("unused")]),
            workspace=tmp_path,
        ),
    )
    server = server_for(store, host, tmp_path / "fork-keys.sqlite3")
    try:
        send(server, "create", "thread/start", {"inputId": "source"})
        source_created = store.read("source", limit=1).records[0].record.payload
        source_principal = Principal.from_dict(source_created["principal"])

        forked = fork_thread(
            server.kernel,
            host,
            {"threadId": "source", "forkId": "principal-copy"},
            principal=source_principal,
        )
        destination = forked["threadId"]
        destination_created = store.read(destination, limit=1).records[0].record.payload
        source_ref = source_principal.attributes
        destination_ref = Principal.from_dict(destination_created["principal"]).attributes

        with store.atomic() as tx:
            source_writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id="source")
            destination_writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=destination)
            assert destination_writer.get_json(destination_ref) == source_writer.get_json(source_ref)
            principal_links = [
                link
                for link in destination_writer.durable.read_links(
                    store.tenant_id,
                    destination,
                    record_seq=1,
                )
                if link.slot == SESSION_PRINCIPAL_ATTRIBUTES
            ]

        assert destination_ref.key_id != source_ref.key_id
        assert destination_ref.labels == source_ref.labels
        assert [link.ref for link in principal_links] == [destination_ref]
    finally:
        server.kernel.close()


def test_accepted_summary_artifacts_preserve_bytes_without_cross_session_read_grant(store, tmp_path, monkeypatch):
    store = wrap_fixture_store(store)
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
    a = server_for(store, host, tmp_path / "summary-keys.sqlite3")
    try:
        create_fixture_session(
            store,
            "source",
            attributes={
                "app_server": {"agent_key": "default", "cwd": None, "metadata": {}},
                "seed": {"messages": [message.to_dict() for message in history], "shared_state": {}},
            },
            consumers=("events", "traces", "app_server"),
            subject_id="source-user",
        )
        first = run_turn(a, "source", "first")
        send(a, "compact", "thread/compact/start", {"threadId": "source", "actionId": "summary"})
        a.run_adapter.join()
        _, records, _ = store.read_state("source")
        hydration = hydration_for_session_store(store, "source")
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
        compact_payload = hydrate_record_payload(compact, hydration)
        assert compact_payload["evidence_manifest"]["artifacts"][0]["artifact_ref"] == artifact.to_dict()
        params = {"threadId": "source", "forkId": "copy", "workspace": {"snapshot": "prepared"}}
        fork = send(a, "fork", "thread/fork", params)
        sid = fork["threadId"]
        fork_creation = store.read_state(sid)[1][0]
        fork_attributes = hydrate_record_payload(fork_creation, hydration_for_session_store(store, sid))["attributes"]
        baseline = fork_attributes["app_server_context"]["task"]
        assert baseline["initial_messages"] == [] and baseline["initial_shared_state"] == {}
        assert baseline["user_prompt"] == "" and "vv_session" not in baseline["metadata"]
        copied = [Message.from_dict(message) for message in fork_attributes["seed"]["messages"]]
        hydrated_records = hydrate_records(records, hydration)
        logical_state = fold_hydrated_records(hydrated_records, records, _fold_factory_for(store), hydration)
        source_messages = project_context(
            hydrated_records,
            logical_state,
            _fold_factory=_fold_factory_for(store),
        )
        assert [m.to_dict() for m in copied] == [m.to_dict() for m in source_messages]
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
        fork_hydration = hydration_for_session_store(store, sid)
        recovery = next(
            hydrate_record_payload(row, fork_hydration)["result"]
            for row in store.read_state(sid)[1]
            if row.record.kind == "op_completed"
            and hydrate_record_payload(row, fork_hydration)["result"].get("tool_call_id") == "recover"
        )
        assert recovery["error_code"] == "artifact_read_denied", recovery
        assert not any(m.role == "tool" and body in m.content for m in calls[-1].messages)
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
