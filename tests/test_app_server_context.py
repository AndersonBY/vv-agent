"""A2 context projection and manual compaction through real App Server producers."""

import json
from contextlib import ExitStack
from dataclasses import replace
from threading import Event

import pytest
from jsonschema import Draft202012Validator
from test_app_server_distributed import node, request, submission

from vv_agent import Agent, ScriptedModelProvider
from vv_agent.app_server.item_mapper import map_run_event
from vv_agent.app_server.schema import json_schema_bundle
from vv_agent.session.projection import project_records
from vv_agent.session.replay import list_events
from vv_agent.types import LLMResponse, Message

SUMMARY = json.dumps({"original_user_messages": ["original request"], "current_work_state": "Continue the task"})


def history():
    return [Message("user", "original request"), Message("assistant", "old facts " * 4000)]


def create(server, *, seeded=False):
    if seeded:
        from vv_agent.session.records import SessionSpec

        with server.kernel.store.atomic() as tx:
            tx.create(
                SessionSpec(
                    "thread",
                    "local",
                    ".",
                    attributes={
                        "app_server": {"agent_key": "default", "cwd": None, "metadata": {}},
                        "seed": {"messages": [m.to_dict() for m in history()], "shared_state": {}},
                    },
                ),
                consumers=("app_server", "events", "traces"),
            )
        server.run_adapter.subscribe("thread", server.transport.connection_id)
        return
    result = request(server, "create", "thread/start", {"inputId": "thread"})
    assert "result" in result, result


def notices(store):
    after = through = None
    result = []
    for _ in range(1000):
        page = list_events(store, "thread", limit=1, after=after, through=through)
        for entry in page["events"]:
            from vv_agent.events import event_from_dict

            e = event_from_dict(entry["event"])
            result.extend(map_run_event(e, thread_id="thread", turn_id=e.run_id).additional_notifications)
        after, through = page["nextCursor"], page["throughCursor"]
        if not page["more"]:
            return [(m, p) for m, p in result if m.startswith("contextCompaction/")]
    pytest.fail("unbounded replay")


def test_context_is_read_only_stable_across_nodes_and_unknown_limits(tmp_path, monkeypatch):
    provider = ScriptedModelProvider.new(
        "test", "unknown", [LLMResponse("answer", raw={"reasoning_content": "PRIVATE"})]
    ).with_token_limits(None, None)
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider)
        b = node(stack, tmp_path / "store", "B", provider)
        create(a)
        before = request(a, "before", "thread/context/read", {"threadId": "thread"})["result"]
        assert before["limits"]["contextWindow"] is None
        assert before["limits"]["outputReserve"] == 16000
        request(a, "turn", "turn/start", submission())
        a.run_adapter.join()
        expected = request(a, "read", "thread/context/read", {"threadId": "thread"})["result"]
        assert expected["contextVersion"] != before["contextVersion"]
        assert expected["estimates"]["estimated"] is True
        assert set(expected["estimates"]["parts"]) == {"instructions", "tools", "history", "summary", "pendingInput"}
        assert "PRIVATE" not in json.dumps(expected) and "reasoning_content" not in json.dumps(expected)
        # Prohibit leases, transactions, state reads with row locks, and host callbacks.
        for name in ("read_state", "atomic", "acquire"):
            monkeypatch.setattr(b.kernel.store, name, lambda *a, **k: pytest.fail("read mutated or locked"))
        monkeypatch.setattr(b.run_adapter._host, "resolve_agent", lambda *a: pytest.fail("read invoked host"))
        assert request(b, "read", "thread/context/read", {"threadId": "thread"})["result"] == expected
        schema = json.loads(json_schema_bundle()["ThreadContextReadResponse"])
        Draft202012Validator(schema).validate(expected)
        monkeypatch.undo()


def test_manual_compaction_idle_receipt_replay_conflict_stale_and_reuse(tmp_path):
    seen = []

    def complete(req):
        seen.append(req.metadata.get("purpose"))
        return LLMResponse(SUMMARY if req.metadata.get("purpose") == "compaction" else "done")

    provider = ScriptedModelProvider.from_callback("test", "m", complete)
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider)
        b = node(stack, tmp_path / "store", "B", provider)
        # Freeze history in the admitted turn, then compact the idle session.
        a.run_adapter._host._run_config = replace(
            a.run_adapter._host._run_config, initial_messages=history(), metadata={"memory_keep_recent_messages": 1}
        )
        b.run_adapter._host._run_config = a.run_adapter._host._run_config
        create(a, seeded=True)
        request(a, "turn", "turn/start", submission())
        a.run_adapter.join()
        version = request(a, "read", "thread/context/read", {"threadId": "thread"})["result"]["contextVersion"]
        params = {"threadId": "thread", "actionId": "compact", "expectedContextVersion": version}
        receipt = request(b, "compact", "thread/compact/start", params)
        assert "result" in receipt, receipt
        receipt = receipt["result"]
        b.run_adapter.join()
        assert receipt["accepted"] and not receipt["replayed"]
        after = request(a, "after", "thread/context/read", {"threadId": "thread"})["result"]
        assert after["contextVersion"] != version
        assert after["lastCompaction"]["replacedMessageCount"] > 0
        assert "Continue the task" in after["lastCompaction"]["summary"]
        assert after["counts"]["rawHistoryMessages"] > after["counts"]["modelVisibleHistoryMessages"]
        assert request(a, "replay", "thread/compact/start", params)["result"] == receipt | {"replayed": True}
        assert (
            request(a, "conflict", "thread/compact/start", {"threadId": "thread", "actionId": "compact"})["error"]["code"]
            == -32024
        )
        assert request(a, "stale", "thread/compact/start", params | {"actionId": "stale"})["error"]["code"] == -32028
        assert seen == ["primary", "compaction"]
        lifecycle = notices(a.kernel.store)
        assert [m for m, _ in lifecycle] == ["contextCompaction/started", "contextCompaction/completed"]
        assert {p["requestId"] for _, p in lifecycle} == {receipt["requestId"]}
        assert request(b, "reuse", "thread/compact/start", {"threadId": "thread", "actionId": "reuse"})["result"]["accepted"]
        b.run_adapter.join()
        assert seen == ["primary", "compaction"]
        assert (
            request(b, "same", "thread/context/read", {"threadId": "thread"})["result"]["contextVersion"]
            == after["contextVersion"]
        )
        lifecycle = notices(a.kernel.store)
        # Full and one-record replay produce identical stable event IDs.
        projected = []
        for e in project_records(a.kernel.store.read_state("thread")[1]):
            projected.extend(map_run_event(e, thread_id="thread", turn_id=e.run_id).additional_notifications)
        assert lifecycle == [(m, p) for m, p in projected if m.startswith("contextCompaction/")]


@pytest.mark.parametrize("cancel", [False, True])
def test_compaction_during_running_turn_preserves_concurrent_input_or_cancel(tmp_path, cancel):
    entered, release = Event(), Event()
    calls = []

    def complete(req):
        calls.append(req)
        if len(calls) == 1:
            entered.set()
            assert release.wait(10)
            return LLMResponse("first result")
        if req.metadata.get("purpose") == "compaction":
            return LLMResponse(SUMMARY)
        return LLMResponse("done")

    provider = ScriptedModelProvider.from_callback("test", "m", complete)
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", provider)
        b = node(stack, tmp_path / "store", "B", provider)
        a.run_adapter._host._run_config = replace(
            a.run_adapter._host._run_config, initial_messages=history(), metadata={"memory_keep_recent_messages": 1}
        )
        b.run_adapter._host._run_config = a.run_adapter._host._run_config
        create(a, seeded=True)
        try:
            start = request(a, "turn", "turn/start", submission())["result"]
            assert entered.wait(5)
            receipt = request(b, "compact", "thread/compact/start", {"threadId": "thread", "actionId": "compact"})
            assert "result" in receipt, receipt
            if cancel:
                answer = request(
                    b, "cancel", "turn/interrupt", {"threadId": "thread", "expectedTurnId": start["turnId"], "actionId": "stop"}
                )
            else:
                answer = request(
                    b, "input", "turn/steer", submission(identity="late", expected=start["turnId"], text="KEEP LATE INPUT")
                )
            assert "result" in answer, answer
        finally:
            release.set()
        a.run_adapter.join()
        b.run_adapter.join()
        state, records, _ = a.kernel.store.read_state("thread")
        if cancel:
            assert state.turns[start["turnId"]].ended
            assert next(r.record for r in reversed(records) if r.record.kind == "turn_ended")._payload["status"] == "cancelled"
            assert not any(r.record.kind == "context_compacted" for r in records)
            assert any(m == "contextCompaction/failed" for m, _ in notices(a.kernel.store))
        else:
            from vv_agent.session.context import project_context

            assert any(m.content == "KEEP LATE INPUT" for m in project_context(records, state))
            assert any(r.record.kind == "context_compacted" for r in records)
            assert any("KEEP LATE INPUT" in str(req.messages) for req in calls[1:])


def test_model_catalog_capabilities_use_resolved_limits_and_explicit_unknowns(tmp_path):
    from vv_agent.app_server.host import model_summary
    from vv_agent.cli import _build_app_server_host
    from vv_agent.config import resolve_model_endpoint

    models = {
        "known": {
            "id": "known-id",
            "endpoints": ["local"],
            "context_length": 100000,
            "max_output_tokens": 9000,
            "function_call_available": True,
            "native_multimodal": True,
            "reasoning_available": True,
            "supported_reasoning_efforts": ["low", "high"],
            "default_reasoning_effort": "high",
            "prompt_cache_available": False,
            "continuation_available": True,
        },
        "unknown": {"id": "unknown-id", "endpoints": ["local"]},
    }
    catalog = {
        "VERSION": "2",
        "backends": {"test": {"models": models}},
        "endpoints": [{"id": "local", "api_key": "placeholder", "api_base": "https://example.invalid/v1"}],
    }
    settings = tmp_path / "catalog.json"
    settings.write_text(json.dumps(catalog))
    for name, known in [("known", True), ("unknown", False)]:
        host = _build_app_server_host(settings_file=settings, backend="test", model=name, workspace=tmp_path, max_cycles=3)
        from vv_agent.app_server import ModelListRequest

        result = host.list_models(ModelListRequest()).to_dict()["models"][0]
        assert result == model_summary(resolve_model_endpoint(catalog, "test", name)).to_dict()
        assert result["id"] == name + "-id"
        if known:
            assert result["contextLength"] == 100000 and result["maxOutputTokens"] == 9000
            assert result["supportsTools"] is True and result["supportsVision"] is True
            assert result["inputModalities"] == ["text", "image"]
            assert result["supportsReasoning"] is True and result["supportedReasoningEfforts"] == ["low", "high"]
            assert result["defaultReasoningEffort"] == "high"
            assert result["supportsPromptCache"] is False and result["supportsContinuation"] is True
        else:
            assert all(value is None for key, value in result.items() if key not in {"id", "provider", "displayName", "metadata"})
        assert "api_key" not in json.dumps(result)
        schema = json.loads(json_schema_bundle()["ClientRequest"])["$defs"]["ModelSummary"]
        Draft202012Validator(schema).validate(result)


@pytest.mark.parametrize(
    "params",
    [
        {"threadId": "thread"},
        {"threadId": "thread", "actionId": "bad/id"},
        {"threadId": "thread", "actionId": "ok", "expectedContextVersion": None},
        {"threadId": "thread", "actionId": "ok", "expectedContextVersion": "0" * 64 + "\n"},
        {"threadId": "thread", "actionId": "ok", "extra": True},
    ],
)
def test_compaction_rejects_malformed_params_before_writes(tmp_path, params):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", ScriptedModelProvider.new("test", "m", []))
        create(server)
        assert request(server, "bad", "thread/compact/start", params)["error"]["code"] == -32602
        assert not server.kernel.store.peek_inbox("thread")


def test_pending_input_and_model_selection_change_context_version_without_drive(tmp_path, monkeypatch):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", ScriptedModelProvider.new("test", "m", []))
        create(server)
        monkeypatch.setattr(server.run_adapter, "recover", lambda *a, **k: None)
        first = server.run_adapter.read_context("thread")
        request(server, "turn", "turn/start", submission(text="PENDING"))
        pending = server.run_adapter.read_context("thread")
        assert pending["contextVersion"] != first["contextVersion"]
        assert pending["counts"]["pendingInputs"] == 1
        assert pending["estimates"]["parts"]["pendingInput"] > 0
        request(
            server,
            "select",
            "thread/model/select",
            {"threadId": "thread", "inputId": "select", "selection": {"model": "new", "route": {}}},
        )
        assert server.run_adapter.read_context("thread")["contextVersion"] != pending["contextVersion"]


@pytest.mark.parametrize("cancel", [False, True])
def test_input_arriving_during_summary_cannot_be_overwritten(tmp_path, cancel):
    primary_entered, release_primary, summary_entered, release_summary = Event(), Event(), Event(), Event()
    primary_count = 0

    def complete(req):
        nonlocal primary_count
        if req.metadata.get("purpose") == "compaction":
            summary_entered.set()
            assert release_summary.wait(10)
            return LLMResponse(SUMMARY)
        primary_count += 1
        if primary_count == 1:
            primary_entered.set()
            assert release_primary.wait(10)
        return LLMResponse("done")

    with ExitStack() as stack:
        provider = ScriptedModelProvider.from_callback("test", "m", complete)
        a = node(stack, tmp_path / "store", "A", provider)
        b = node(stack, tmp_path / "store", "B", provider)
        for server in (a, b):
            server.run_adapter._host._run_config = replace(
                server.run_adapter._host._run_config, metadata={"memory_keep_recent_messages": 1}
            )
        create(a, seeded=True)
        try:
            turn = request(a, "turn", "turn/start", submission())["result"]["turnId"]
            assert primary_entered.wait(5)
            assert "result" in request(b, "compact", "thread/compact/start", {"threadId": "thread", "actionId": "compact"})
            release_primary.set()
            assert summary_entered.wait(5)
            if cancel:
                response = request(
                    b, "stop", "turn/interrupt", {"threadId": "thread", "expectedTurnId": turn, "actionId": "stop"}
                )
            else:
                response = request(b, "late", "turn/steer", submission(identity="late", expected=turn, text="SURVIVES SUMMARY"))
            assert "result" in response
        finally:
            release_primary.set()
            release_summary.set()
        a.run_adapter.join()
        b.run_adapter.join()
        state, records, _ = a.kernel.store.read_state("thread")
        if cancel:
            assert next(r.record for r in reversed(records) if r.record.kind == "turn_ended")._payload["status"] == "cancelled"
            assert not any(r.record.kind == "context_compacted" for r in records)
            assert any(m == "contextCompaction/failed" for m, _ in notices(a.kernel.store))
        else:
            from vv_agent.session.context import project_context

            assert any(m.content == "SURVIVES SUMMARY" for m in project_context(records, state))
            assert any(r.record.kind == "context_compacted" for r in records)


def test_canonical_a2_fixture_matches_real_producer(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    from scripts.app_server_a2_fixtures import produce

    fixture = Path(__file__).parent / "fixtures/parity/app_server_a2.json"
    assert produce() == json.loads(fixture.read_text())


def test_compaction_admission_race_has_one_durable_request(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    with ExitStack() as stack:
        provider = ScriptedModelProvider.new("test", "m", [])
        a = node(stack, tmp_path / "store", "A", provider)
        b = node(stack, tmp_path / "store", "B", provider)
        create(a)
        for server in (a, b):
            monkeypatch.setattr(server.run_adapter, "recover", lambda *a, **k: None)
        version = a.run_adapter.read_context("thread")["contextVersion"]
        barrier = Barrier(2)

        def submit(server):
            barrier.wait()
            return request(
                server,
                "compact",
                "thread/compact/start",
                {
                    "threadId": "thread",
                    "actionId": "same",
                    "expectedContextVersion": version,
                },
            )["result"]

        with ThreadPoolExecutor(2) as pool:
            left, right = list(pool.map(submit, (a, b)))
        assert left["requestId"] == right["requestId"] and left["inputSeq"] == right["inputSeq"]
        assert sorted([left["replayed"], right["replayed"]]) == [False, True]
        assert len(a.kernel.store.peek_inbox("thread")) == 1


def test_context_methods_are_authorized_before_store_access(tmp_path, monkeypatch):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", ScriptedModelProvider.new("test", "m", []))
        monkeypatch.setattr(server.run_adapter._host, "authorize", lambda *args: False)
        monkeypatch.setattr(server.kernel.store, "read", lambda *a, **k: pytest.fail("unauthorized read"))
        for method, params in (
            ("thread/context/read", {"threadId": "unknown"}),
            ("thread/compact/start", {"threadId": "unknown", "actionId": "compact"}),
        ):
            assert request(server, method, method, params)["error"]["code"] == -32027
        monkeypatch.undo()


def test_concurrent_creation_freezes_one_baseline_across_different_nodes(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    with ExitStack() as stack:
        provider = ScriptedModelProvider.new("test", "m", [])
        a = node(stack, tmp_path / "store", "A", provider, agent=Agent("test", "Node A instructions"))
        b = node(stack, tmp_path / "store", "B", provider, agent=Agent("test", "Node B instructions"))
        barrier = Barrier(2)

        def create_on(server):
            barrier.wait()
            return request(server, "create", "thread/start", {"inputId": "thread"})

        with ThreadPoolExecutor(2) as pool:
            responses = list(pool.map(create_on, (a, b)))
        assert all("result" in response for response in responses), responses
        assert sorted(response["result"]["replayed"] for response in responses) == [False, True]
        assert a.run_adapter.read_context("thread") == b.run_adapter.read_context("thread")


def test_baseline_and_maintenance_do_not_run_prompt_or_output_guardrails(tmp_path):
    from vv_agent.guardrails import GuardrailResult

    guarded_inputs, guarded_outputs = [], []

    def check_input(ctx, text):
        guarded_inputs.append(text)
        return GuardrailResult.allow() if text else GuardrailResult.block("empty prompt")

    def check_output(ctx, value):
        guarded_outputs.append(value)
        return GuardrailResult.allow()

    def complete(req):
        return LLMResponse(SUMMARY if req.metadata.get("purpose") == "compaction" else "done")

    with ExitStack() as stack:
        provider = ScriptedModelProvider.from_callback("test", "m", complete)
        agent = Agent("test", "Answer.", input_guardrails=[check_input], output_guardrails=[check_output])
        server = node(stack, tmp_path / "store", "A", provider, agent=agent)
        server.run_adapter._host._run_config = replace(
            server.run_adapter._host._run_config, metadata={"memory_keep_recent_messages": 1}
        )
        create(server)
        assert guarded_inputs == []
        request(server, "turn", "turn/start", submission())
        server.run_adapter.join()
        assert guarded_inputs == ["hello"] and guarded_outputs == ["done"]
        request(server, "compact", "thread/compact/start", {"threadId": "thread", "actionId": "compact"})
        server.run_adapter.join()
        assert guarded_inputs == ["hello"] and guarded_outputs == ["done"]
        assert any(m == "contextCompaction/completed" for m, _ in notices(server.kernel.store))


def test_steer_during_idle_compaction_is_preserved_for_next_request(tmp_path):
    entered, release = Event(), Event()
    calls = []

    def complete(req):
        calls.append(req)
        if req.metadata.get("purpose") == "compaction":
            entered.set()
            assert release.wait(10)
            return LLMResponse(SUMMARY)
        return LLMResponse("done")

    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", ScriptedModelProvider.from_callback("test", "m", complete))
        server.run_adapter._host._run_config = replace(
            server.run_adapter._host._run_config, metadata={"memory_keep_recent_messages": 1}
        )
        create(server, seeded=True)
        request(server, "first", "turn/start", submission())
        server.run_adapter.join()
        try:
            assert "result" in request(server, "compact", "thread/compact/start", {"threadId": "thread", "actionId": "compact"})
            assert entered.wait(5)
            tid = server.kernel.store.read_state("thread")[0].active_turn_id
            response = request(server, "late", "turn/steer", submission(identity="late", expected=tid, text="MAINTENANCE INPUT"))
            assert "result" in response
        finally:
            release.set()
        server.run_adapter.join()
        from vv_agent.session.context import project_context

        state, records, _ = server.kernel.store.read_state("thread")
        assert any(m.content == "MAINTENANCE INPUT" for m in project_context(records, state))
        assert "result" in request(server, "next", "turn/start", submission(identity="next", expected=tid))
        server.run_adapter.join()
        assert any(m.content == "MAINTENANCE INPUT" for m in calls[-1].messages)


def test_typed_client_context_and_compaction_methods():
    from vv_agent import RunConfig, ThreadCompactStartParams, ThreadContextReadParams
    from vv_agent.app_server import AppServerClient, ClientInfo, DefaultAppServerHost, ThreadStartParams

    client = AppServerClient.for_host(
        DefaultAppServerHost(
            agent=Agent("test", "Answer."),
            run_config=RunConfig(model_provider=ScriptedModelProvider.new("test", "m", [])),
        )
    )
    try:
        client.initialize(ClientInfo(name="test"))
        client.start_thread(ThreadStartParams(agent_key="default", input_id="thread"))
        view = client.read_context(ThreadContextReadParams("thread"))
        params = ThreadCompactStartParams("thread", "compact", view["contextVersion"])
        receipt = client.compact_thread(params)
        client._server.run_adapter.join()
        assert client.compact_thread(params) == receipt | {"replayed": True}
    finally:
        client.close()
        client._server.kernel.close()


def test_busy_context_cut_is_retryable_and_never_locks(tmp_path, monkeypatch):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", ScriptedModelProvider.new("test", "m", []))
        create(server)
        original = server.kernel.store.read

        def moving_head(sid, **kwargs):
            page = original(sid, **kwargs)
            return replace(page, head_seq=page.head_seq + 1) if kwargs.get("limit") == 1 else page

        monkeypatch.setattr(server.kernel.store, "read", moving_head)
        monkeypatch.setattr(server.kernel.store, "atomic", lambda: pytest.fail("read locked"))
        assert request(server, "read", "thread/context/read", {"threadId": "thread"})["error"]["code"] == -32001
        monkeypatch.undo()
