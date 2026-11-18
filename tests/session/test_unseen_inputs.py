"""Unseen inputs survive every context replacement until a primary receipt commits."""

import json
import multiprocessing
import runpy
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import pytest
from support.s1_fixture_values import append_fixture_records, clear_fixture_state_cache, read_fixture_state

from vv_agent.llm.scripted import ScriptStep
from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
from vv_agent.memory import MemoryManager
from vv_agent.memory.input_protection import UnseenInputError
from vv_agent.runtime.hooks import BaseRuntimeHook, BeforeLLMPatch
from vv_agent.session.context import protected_start
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem
from vv_agent.types import LLMResponse, Message

from .fold_history import extend_logical
from .helpers import (
    create_fixture_session,
    fixture_child_spec,
    fixture_content_reader,
    push_fixture_input,
    wrap_fixture_store,
)
from .test_recovery_matrix import records_of, start
from .test_recovery_matrix import runtime as base_runtime

SUMMARY = json.dumps({"current_work_state": "Summarized old work"})


@pytest.fixture
def store(store):
    return wrap_fixture_store(store)


def runtime(store, database, steps, tools=(), *, sid="s", **kwargs):
    kwargs.setdefault("content_reader", fixture_content_reader(store, sid))
    return base_runtime(database, steps, tools, **kwargs)


def read_runtime_state(database):
    from .conftest import open_store

    with open_store(database) as store:
        store = wrap_fixture_store(store)
        clear_fixture_state_cache(store)
        state, rows, watermark = read_fixture_state(store, "s")
        return state, rows, watermark


def test_new_session_declares_protection(tmp_path):
    from vv_agent.session.projection import hydration_for_session_store
    from vv_agent.session.records import Record
    from vv_agent.session.sqlite import SQLiteStore

    with SQLiteStore.standalone(str(tmp_path / "unseen.sqlite")) as store:
        store.install_schema()
        store = wrap_fixture_store(store)
        create_fixture_session(
            store,
            "new",
            attributes={
                "seed": {
                    "messages": [],
                    "shared_state": {},
                    "unseen_input_protection": True,
                    "response_rejection": True,
                }
            },
        )
        row = store.read("new", limit=1).records[0]
        hydration = hydration_for_session_store(store, "new")
        from vv_agent.session.projection import hydrate_record_payload

        attributes = hydrate_record_payload(row, hydration)["attributes"]
        assert attributes["seed"] == {
            "messages": [],
            "shared_state": {},
            "unseen_input_protection": True,
            "response_rejection": True,
        }
        Record.parse(row.record.encode())


@pytest.mark.parametrize("enabled", [None, False, True])
def test_creation_replay_keeps_frozen_policy(store, enabled):
    from vv_agent.session.store import Conflict

    seed = {"messages": [], "shared_state": {}}
    if enabled is not None:
        seed["unseen_input_protection"] = enabled
    attributes = {"seed": seed}
    first = create_fixture_session(store, "replay", attributes=attributes)
    assert not first.replayed
    before = store.read("replay").records[0].record.encode()
    replay = create_fixture_session(store, "replay", attributes=attributes)
    assert replay.replayed
    assert store.read("replay").records[0].record.encode() == before
    with pytest.raises(Conflict):
        create_fixture_session(
            store,
            "replay",
            attributes={"seed": seed | {"unseen_input_protection": not bool(enabled)}},
        )


@pytest.mark.parametrize("drop_ratio", [0, 0.95])
def test_summary_and_emergency_keep_unseen_tail(drop_ratio):
    messages = [Message("system", "system"), Message("assistant", "seen " * 4000)]
    messages += [Message("user", f"unseen-{i}") for i in range(12)]
    manager = MemoryManager(keep_recent_messages=1)
    manager.protected_start = 2
    plan = manager.plan_summary(messages, drop_ratio=drop_ratio)
    assert plan is not None
    assert plan.tail == messages[2:]
    assert plan.prefix == messages[1:2]


def test_protection_rolls_back_to_whole_tool_batch():
    messages = [
        Message("user", "seen"),
        Message("assistant", "", tool_calls=[{"id": "a", "type": "function", "function": {"name": "tool", "arguments": "{}"}}]),
        Message("tool", "unseen", tool_call_id="a"),
        Message("user", "last"),
    ]
    manager = MemoryManager(keep_recent_messages=1)
    manager.protected_start = 2
    plan = manager.plan_summary(messages)
    assert plan is not None
    assert plan.tail == messages[1:]


@pytest.mark.parametrize("count", [1, 12])
def test_steer_survives_ptl_in_real_primary_request(store, database, count):
    requests = []

    def initial(request):
        requests.append(request)
        # Arrives while the successful primary is in flight, beyond its context_version.
        for index in range(count):
            push_fixture_input(store, "s", InboxItem(f"steer-{index}", "steer", {"content": f"ORIGINAL-{index}"}))
        return LLMResponse("seen facts " * 4000)

    def too_long(request):
        requests.append(request)
        raise RuntimeError("maximum context length exceeded")

    def summary(request):
        requests.append(request)
        assert request.metadata["purpose"] == "compaction"
        assert "ORIGINAL-" not in request.messages[0].content
        return LLMResponse(SUMMARY)

    def answer(request):
        requests.append(request)
        for index in range(count):
            assert sum(m.content == f"ORIGINAL-{index}" for m in request.messages) == 1
        assert any(m.name == "memory_summary" for m in request.messages)
        return LLMResponse("done")

    start(store)
    rt = runtime(store, database, [initial, too_long, summary, answer])
    rt.config = replace(rt.config, no_tool_policy="continue", max_cycles=2)
    rt.memory_manager = MemoryManager(compact_threshold=100000, keep_recent_messages=1)
    assert drive(store, "s", runtime=rt)
    assert len(requests) == 4
    assert len(records_of(store, "context_compacted")) == 1
    applied = [r for r in records_of(store, "input_applied") if r.payload["input"]["kind"] == "steer"]
    assert len(applied) == count
    state, records, _ = read_runtime_state(database)
    assert protected_start(records, state) is None, (
        state._protection.covered,
        state._protection.inputs,
        [
            (row.seq, row.record.kind, row.record.operation_id, row.record.payload.get("context_version"))
            for row in records
            if row.record.kind in {"input_applied", "op_planned", "op_completed"}
        ],
    )


@pytest.mark.parametrize("stage", ["before_memory", "before_llm"])
def test_hook_cannot_remove_unseen_input_or_commit_plan(store, database, stage):
    class RemoveInput(BaseRuntimeHook):
        def before_memory_compact(self, event):
            return [m for m in event.messages if m.role != "user"] if stage == "before_memory" else None

        def before_llm(self, event):
            return BeforeLLMPatch(messages=[m for m in event.messages if m.role != "user"])

    start(store)
    rt = runtime(store, database, [])
    rt.config = replace(rt.config, hooks=[RemoveInput()])
    rt.hooks = __import__("vv_agent.runtime.hooks", fromlist=["RuntimeHookManager"]).RuntimeHookManager(rt.config.hooks)
    with pytest.raises(UnseenInputError) as error:
        drive(store, "s", runtime=rt)
    assert error.value.code == "unseen_input_removed"
    assert not records_of(store, "op_planned")
    assert (
        not any(r.payload["stage"] == "before_memory" for r in records_of(store, "boundary_recorded"))
        if stage == "before_memory"
        else True
    )


def test_unseen_capacity_ends_without_summary_or_retry(store, database):
    create_fixture_session(
        store,
        "s",
        attributes={
            "seed": {
                "messages": [],
                "shared_state": {},
                "unseen_input_protection": True,
                "response_rejection": True,
            }
        },
    )
    push_fixture_input(store, "s", InboxItem("large", "user", {"content": "new input " * 10000}))
    rt = runtime(store, database, [])
    rt.memory_manager = MemoryManager(model_context_window=1000, reserved_output_tokens=100, autocompact_buffer_tokens=0)
    drive(store, "s", runtime=rt)
    _, rows, _ = read_fixture_state(store, "s")
    terminal = next(row.record for row in rows if row.record.kind == "turn_ended")
    assert terminal.payload["status"] == "failed"
    assert terminal.payload["reason"] == "unseen_input_capacity_exceeded"
    assert terminal.payload["unconfirmed_operations"] == []
    assert not records_of(store, "op_planned")
    assert not records_of(store, "op_unknown")
    assert not records_of(store, "context_compacted")


def test_acceptance_rejects_forged_tail():
    messages = [Message("assistant", "old " * 4000), Message("user", "protected")]
    manager = MemoryManager(protected_start=1, keep_recent_messages=1)
    plan = manager.plan_summary(messages)
    assert plan is not None
    plan.tail.clear()
    with pytest.raises(UnseenInputError):
        manager.accept_summary(plan, SUMMARY)


def test_user_reply_survives_ptl(store, database):
    start(store)
    rt = runtime(store, database, [LLMResponse("seen " * 2000)])
    rt.config = replace(rt.config, no_tool_policy="wait_user")
    rt.memory_manager = MemoryManager(compact_threshold=100000, keep_recent_messages=1)
    drive(store, "s", runtime=rt)
    park = records_of(store, "turn_parked")[-1]
    push_fixture_input(
        store,
        "s",
        InboxItem(
            "reply",
            "user",
            {"content": {"interaction_id": park.payload["interaction_id"], "text": "USER-ORIGINAL"}},
            target_turn_id=park.turn_id,
            generation=0,
        ),
    )

    def reject(_):
        raise RuntimeError("maximum context length exceeded")

    def answer(request):
        assert sum(m.content == "USER-ORIGINAL" for m in request.messages) == 1
        return LLMResponse("done")

    rt.llm.steps[:] = [reject, LLMResponse(SUMMARY), answer]
    drive(store, "s", runtime=rt)
    assert len(records_of(store, "context_compacted")) == 1
    assert len([r for r in records_of(store, "input_applied") if r.payload["input"]["input_id"] == "reply"]) == 1


def test_tool_user_reply_keeps_whole_batch(store, database):
    from vv_agent.types import ToolCall

    start(store)
    rt = runtime(
        store,
        database,
        [LLMResponse("seen facts " * 4000), LLMResponse("", [ToolCall("question", "ask_user", {"question": "Choose"})])],
    )
    rt.config = replace(rt.config, no_tool_policy="continue", max_cycles=3)
    rt.memory_manager = MemoryManager(compact_threshold=100000, keep_recent_messages=1)
    drive(store, "s", runtime=rt)
    park = records_of(store, "op_parked")[-1]
    push_fixture_input(
        store,
        "s",
        InboxItem(
            "reply",
            "user",
            {"content": {"interaction_id": park.payload["handle"]["interaction_id"], "text": "TOOL-ORIGINAL"}},
            target_turn_id=park.turn_id,
            generation=0,
        ),
    )

    def reject(_):
        raise RuntimeError("maximum context length exceeded")

    def answer(request):
        tools = [m for m in request.messages if m.role == "tool" and m.tool_call_id == "question"]
        assert len(tools) == 1 and "TOOL-ORIGINAL" in tools[0].content
        assert any(m.tool_calls and m.tool_calls[0]["id"] == "question" for m in request.messages)
        return LLMResponse("done")

    rt.llm.steps[:] = [reject, LLMResponse(SUMMARY), answer]
    drive(store, "s", runtime=rt)
    assert len(records_of(store, "context_compacted")) == 1
    assert not rt.llm.steps


@pytest.mark.parametrize("count", [1, 12])
def test_native_children_survive_ptl(store, database, count):
    from vv_agent.session.children import ChildSession, child_delivery
    from vv_agent.types import ToolCall

    from .test_children import child_task

    names = [f"child-{index}" for index in range(count)]

    def reject(_):
        raise RuntimeError("maximum context length exceeded")

    def summary(request):
        assert "Child completed:" not in request.messages[0].content
        return LLMResponse(SUMMARY)

    def answer(request):
        notifications = [
            json.loads(m.content.removeprefix("Child completed: "))
            for m in request.messages
            if isinstance(m.content, str) and m.content.startswith("Child completed: ")
        ]
        assert sorted(n["session_id"] for n in notifications) == sorted(names)
        return LLMResponse("done")

    def hook(point, record):
        if point == "after_commit" and record.kind == "op_completed" and "/tool/" in (record.operation_id or ""):
            for sid in names:
                drive(store, sid, runtime=runtime(store, database, [LLMResponse(sid + " ORIGINAL")], sid=sid))
                with store.atomic() as tx:
                    child_delivery(store, tx, sid)

    start(store)
    rt = runtime(
        store,
        database,
        [LLMResponse("seen facts " * 4000, [ToolCall("children", "child_task", {})]), reject, summary, answer],
        [child_task],
        hook=hook,
    )
    rt.children["child_task"] = lambda _: tuple(
        ChildSession(fixture_child_spec(store, "s", sid), "work", background=True) for sid in names
    )
    rt.memory_manager = MemoryManager(compact_threshold=100000, keep_recent_messages=1)
    drive(store, "s", runtime=rt)
    _, rows, _ = read_fixture_state(store, "s")
    terminal = next(row.record for row in rows if row.record.kind == "turn_ended")
    assert terminal.payload["status"] == "completed"
    assert (
        len(
            [
                row
                for row in rows
                if row.record.kind == "input_applied" and row.record.payload["reason"] == "background child notification"
            ]
        )
        == count
    )
    assert len(records_of(store, "context_compacted")) == 1


@pytest.mark.parametrize("cut", ["planned", "receipt", "accepted"])
def test_restart_preserves_unseen_input_once(store, database, cut):
    class Crash(BaseException):
        pass

    def initial(_):
        push_fixture_input(store, "s", InboxItem("late", "steer", {"content": "RESTART-ORIGINAL"}))
        return LLMResponse("seen facts " * 4000)

    def reject(_):
        raise RuntimeError("maximum context length exceeded")

    def hook(point, record):
        if point != "after_commit":
            return
        match = (
            record.kind == "context_compacted"
            if cut == "accepted"
            else record.kind == ("op_planned" if cut == "planned" else "op_completed")
            and "/compaction/" in (record.operation_id or "")
        )
        if match:
            raise Crash

    def answer(request):
        assert sum(m.content == "RESTART-ORIGINAL" for m in request.messages) == 1
        return LLMResponse("done")

    start(store)
    rt = runtime(store, database, [initial, reject, LLMResponse(SUMMARY), answer], hook=hook)
    rt.config = replace(rt.config, no_tool_policy="continue", max_cycles=2)
    rt.memory_manager = MemoryManager(compact_threshold=100000, keep_recent_messages=1)
    with pytest.raises(Crash):
        drive(store, "s", runtime=rt)
    rt.hook = lambda *_: None
    clear_fixture_state_cache(store)
    drive(store, "s", runtime=rt)
    assert len(records_of(store, "context_compacted")) == 1
    assert len([r for r in records_of(store, "op_planned") if r.payload["purpose"] == "compaction"]) == 1
    assert len([r for r in records_of(store, "input_applied") if r.payload["input"]["input_id"] == "late"]) == 1


@contextmanager
def rejected_http(status):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls.append(self.path)
            body = json.dumps({"error": {"message": "Synthetic failure", "type": "test_error"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with patch("openai._base_client.SyncAPIClient._sleep_for_retry", lambda *a, **k: None):
            yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("status", [403, 503])
def test_real_http_endpoint_retry_and_unknown_do_not_advance_coverage(store, database, status):
    start(store)
    with rejected_http(status) as (url, calls):
        rt = runtime(store, database, [])
        rt.llm = VvLlmClient(
            [EndpointTarget(name, "synthetic-unused", url + "/" + name) for name in ("a", "b")],
            backend="openai",
            randomize_endpoints=False,
        )
        drive(store, "s", runtime=rt)
    state, records, _ = read_fixture_state(store, "s")
    assert protected_start(records, state) == 1
    assert len(calls) >= 2
    assert len(records_of(store, "op_planned")) == 2
    assert bool(records_of(store, "op_unknown")) is (status == 503)
    terminal = next(row.record for row in records if row.record.kind == "turn_ended")
    assert terminal.payload["reason"] == ("model_request_rejected" if status == 403 else "model_outcome_unknown")


def test_unknown_retry_success_covers_only_original_request(store, database):
    def lost(_):
        push_fixture_input(store, "s", InboxItem("late", "steer", {"content": "RETRY-ORIGINAL"}))
        raise TimeoutError("synthetic acknowledgement loss")

    def retry(request):
        assert not any(m.content == "RETRY-ORIGINAL" for m in request.messages)
        return LLMResponse("response to original request")

    def answer(request):
        assert sum(m.content == "RETRY-ORIGINAL" for m in request.messages) == 1
        return LLMResponse("done")

    def hook(point, record):
        if point == "after_commit" and record.kind == "op_completed" and record.attempt == 2:
            state, records, _ = read_fixture_state(store, "s")
            assert protected_start(records, state) is not None

    start(store)
    rt = runtime(store, database, [lost, retry, answer], hook=hook)
    drive(store, "s", runtime=rt)
    assert not rt.llm.steps
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"


@pytest.mark.parametrize("version", ["v1", "999999", "0", "01"])
def test_declared_primary_rejects_invalid_context_reference(store, database, version):
    from vv_agent.session.reducer import Fold, TransitionError

    start(store)
    drive(store, "s", runtime=runtime(store, database, [LLMResponse("done")]))
    records = [row.record for row in read_fixture_state(store, "s")[1]]
    index = next(i for i, record in enumerate(records) if record.kind == "op_planned")
    plan = records[index]
    records = [*records[:index], replace(plan, payload=plan.payload | {"context_version": version})]
    consumed = [InboxItem(**r.payload["input"]) for r in records if r.kind == "input_applied"]
    with pytest.raises(TransitionError, match="invalid input-protection context version"):
        extend_logical(Fold(), records, consumed_inputs=consumed)


def test_0333_producer_is_rejected_by_current_codec():
    from vv_agent.session.records import Record, RecordError

    fixture = json.loads((Path(__file__).parent / "fixtures" / "legacy-unseen-0333.json").read_text())
    with pytest.raises(RecordError):
        Record.parse(fixture["records"][0].encode())


def test_unseen_fixture_matches_local_contract():
    from support.contract import fixture_root

    produce = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "unseen_input_fixtures.py"))["generate"]
    assert produce() == json.loads((fixture_root() / "unseen_input_protection.json").read_text())


@pytest.mark.parametrize("policy", ["unseen_input_protection", "response_rejection"])
@pytest.mark.parametrize("enabled", [False, True])
def test_custom_children_inherit_parent_choice(store, database, enabled, policy):
    from vv_agent.session.children import ChildSession

    from .test_children import parent_runtime

    create_fixture_session(store, "s", attributes={"seed": {policy: enabled}})
    push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))
    rt = parent_runtime(store, database)
    rt.content_reader = fixture_content_reader(store, "s")
    rt.children["child_task"] = lambda _: ChildSession(
        replace(
            fixture_child_spec(store, "s", "child"),
            attributes={"seed": {policy: not enabled}},
        ),
        "work",
    )
    drive(store, "s", runtime=rt)
    from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store

    child_row = store.read("child", limit=1).records[0]
    child_payload = hydrate_record_payload(child_row, hydration_for_session_store(store, "child"))
    assert child_payload["attributes"]["seed"][policy] is enabled


@pytest.mark.parametrize("policy", ["unseen_input_protection", "response_rejection"])
@pytest.mark.parametrize("enabled", [False, True])
def test_app_server_fork_inherits_choice(store, tmp_path, monkeypatch, enabled, policy):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider
    from vv_agent.app_server import DefaultAppServerHost
    from vv_agent.session import app_server as session_app_server

    from .test_app_server_fork import run_turn, send, server_for

    provider = ScriptedModelProvider.new("test", "test", [LLMResponse("done")])
    host = DefaultAppServerHost(agent=Agent("test", "test", model="test"), run_config=RunConfig(model_provider=provider))
    server = server_for(store, host, tmp_path / "unseen-fork-keys.sqlite")
    try:
        publish_created = session_app_server.publish_session_created_content

        def seed_created_content(writer, *, principal, workspace, attributes):
            return publish_created(
                writer,
                principal=principal,
                workspace=workspace,
                attributes=attributes | {"seed": {policy: enabled}},
            )

        monkeypatch.setattr(session_app_server, "publish_session_created_content", seed_created_content)
        send(server, "create", "thread/start", {"inputId": "source"})
        turn = run_turn(server, "source", "first")
        result = send(server, "fork", "thread/fork", {"threadId": "source", "forkId": "copy", "throughTurnId": turn})
        from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store

        forked_row = store.read(result["threadId"], limit=1).records[0]
        forked_payload = hydrate_record_payload(forked_row, hydration_for_session_store(store, result["threadId"]))
        assert forked_payload["attributes"]["seed"][policy] is enabled
    finally:
        server.kernel.close()


def kill_protected_worker(database, pipe, cut):
    from .conftest import open_store

    with open_store(database) as store:
        store = wrap_fixture_store(store)

        def initial(_):
            push_fixture_input(store, "s", InboxItem("late", "steer", {"content": "KILL-ORIGINAL"}))
            return LLMResponse("seen facts " * 4000)

        def reject(_):
            raise RuntimeError("maximum context length exceeded")

        def hook(point, record):
            match = (
                record.kind == "context_compacted"
                if cut == "accepted"
                else record.kind == "op_completed" and "/compaction/" in (record.operation_id or "")
            )
            if point == "after_commit" and match:
                pipe.send(("barrier", None))
                pipe.recv()

        rt = runtime(store, database, [initial, reject, LLMResponse(SUMMARY)], hook=hook)
        rt.config = replace(rt.config, no_tool_policy="continue", max_cycles=2)
        rt.memory_manager = MemoryManager(compact_threshold=100000, keep_recent_messages=1)
        drive(store, "s", runtime=rt)


@pytest.mark.persistent_store
@pytest.mark.parametrize("cut", ["receipt", "accepted"])
def test_process_death_preserves_unseen_suffix(store, database, cut):
    from .test_recovery_matrix import kill, receive

    start(store)
    ctx = multiprocessing.get_context("spawn")
    pipe, child = ctx.Pipe()
    proc = ctx.Process(target=kill_protected_worker, args=(database, child, cut))
    proc.start()
    child.close()
    try:
        receive(pipe, "barrier")
        kill(proc, pipe, database)

        def answer(request):
            assert sum(m.content == "KILL-ORIGINAL" for m in request.messages) == 1
            return LLMResponse("done")

        rt = runtime(store, database, [answer])
        rt.config = replace(rt.config, no_tool_policy="continue", max_cycles=2)
        rt.memory_manager = MemoryManager(compact_threshold=100000, keep_recent_messages=1)
        drive(store, "s", runtime=rt)
        assert len(records_of(store, "context_compacted")) == 1
        assert len([r for r in records_of(store, "op_planned") if r.payload["purpose"] == "compaction"]) == 1
        assert len([r for r in records_of(store, "input_applied") if r.payload["input"]["input_id"] == "late"]) == 1
    finally:
        if proc.is_alive():
            kill(proc, pipe, database)


@pytest.mark.parametrize("attributes", [None, {"host": {"id": 7}}, {"seed": {"messages": [], "shared_state": {}}}])
@pytest.mark.persistent_store
def test_s1_database_tx_create_replay(database, attributes):
    """S1 create replays exact bytes after reopening the real SQLite/PostgreSQL database."""
    from vv_agent.session.store import Conflict

    from .conftest import open_store

    with open_store(database) as store:
        store.install_schema()
        store = wrap_fixture_store(store)
        first = create_fixture_session(store, "legacy", workspace="/tmp", attributes=attributes, consumers=("events",))
        assert not first.replayed
        before = store.read("legacy").records[0].record.encode()
    with open_store(database) as store:
        store = wrap_fixture_store(store)
        replay = create_fixture_session(
            store,
            "legacy",
            workspace="/tmp",
            attributes=attributes,
            consumers=("events",),
        )
        assert replay.replayed
        assert store.read("legacy").records[0].record.encode() == before
        for change in (True, False):
            attrs = json.loads(json.dumps(attributes or {}))
            attrs.setdefault("seed", {})["unseen_input_protection"] = change
            with pytest.raises(Conflict):
                create_fixture_session(store, "legacy", workspace="/tmp", attributes=attrs, consumers=("events",))
        with pytest.raises(Conflict):
            create_fixture_session(
                store,
                "legacy",
                workspace="/tmp",
                attributes=attributes,
                consumers=("events",),
                subject_id="other",
            )
        with pytest.raises(Conflict):
            create_fixture_session(store, "legacy", workspace="/different", attributes=attributes, consumers=("events",))
        with pytest.raises(Conflict):
            create_fixture_session(
                store,
                "legacy",
                workspace="/tmp",
                attributes=(attributes or {}) | {"other": 1},
                consumers=("events",),
            )
        with pytest.raises(Conflict):
            create_fixture_session(store, "legacy", workspace="/tmp", attributes=attributes, consumers=("different",))
        assert store.read("legacy").records[0].record.encode() == before


@pytest.mark.parametrize("with_steer", [False, True])
def test_provider_callback_and_steer_keep_whole_group_after_ptl(store, database, with_steer):
    from vv_agent.tools.function import function_tool
    from vv_agent.types import ToolCall

    @function_tool
    def effect() -> str:
        raise AssertionError("provider binding owns execution")

    provider = CallbackProvider()
    start(store)
    # A separate old prefix gives the summary something eligible to consume.
    rt = runtime(
        store,
        database,
        [LLMResponse("seen facts " * 4000), LLMResponse("", [ToolCall("e", "effect", {})])],
        [effect],
        providers={"effect": provider},
    )
    rt.config = replace(rt.config, no_tool_policy="continue", max_cycles=3)
    rt.memory_manager = MemoryManager(compact_threshold=100000, keep_recent_messages=1)
    drive(store, "s", runtime=rt)
    state, _, _ = read_fixture_state(store, "s")
    payload = {
        "operation_id": provider.plan.operation_id,
        "attempt": 1,
        "request_digest": provider.plan.payload["request_digest"],
        "provider_binding": provider.plan.payload["provider_binding"],
        "result": provider.result,
        "usage": {},
        "evidence": ["proof"],
    }
    push_fixture_input(store, "s", InboxItem("provider-reply", "provider_result", payload, state.active_turn_id, 0))
    if with_steer:
        push_fixture_input(store, "s", InboxItem("late-steer", "steer", {"content": "STEER-ORIGINAL"}))

    def reject(request):
        assert any(m.content == "PROVIDER-INPUT-ORIGINAL" for m in request.messages)
        raise RuntimeError("maximum context length exceeded")

    def summary(request):
        assert "PROVIDER-INPUT-ORIGINAL" not in request.messages[0].content
        assert "STEER-ORIGINAL" not in request.messages[0].content
        return LLMResponse(SUMMARY)

    def answer(request):
        index = next(i for i, m in enumerate(request.messages) if m.tool_calls)
        assert request.messages[index].tool_calls[0]["id"] == "e"
        assert request.messages[index + 1].tool_call_id == "e"
        assert sum(m.content == "PROVIDER-INPUT-ORIGINAL" for m in request.messages) == 1
        assert sum(m.content == "STEER-ORIGINAL" for m in request.messages) == int(with_steer)
        assert any(m.name == "memory_summary" for m in request.messages)
        return LLMResponse("done")

    rt.llm.steps[:] = [reject, summary, answer]
    drive(store, "s", runtime=rt)
    assert not rt.llm.steps
    assert len(records_of(store, "context_compacted")) == 1
    assert len([r for r in records_of(store, "input_applied") if r.payload["input"]["input_id"] == "provider-reply"]) == 1
    state, rows, _ = read_fixture_state(store, "s")
    assert protected_start(rows, state) is None


HTTP_PORT = 0


@pytest.mark.parametrize("mode", ["normal", "ptl", "retry", "unknown"])
def test_http_adapter_primary_coverage(store, database, mode):
    http_adapter_primary_coverage(store, database, mode)


def http_adapter_primary_coverage(store, database, mode):
    """Local HTTP -> provider SDK -> real VvLlmClient -> kernel receipts."""
    calls, coverage = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({"path": self.path, "request": request})
            if mode == "normal" or (mode == "retry" and "/b/" in self.path):
                status, content = 200, "done"
            elif mode == "ptl":
                status = 400 if len(calls) == 2 else 200
                content = "seen facts " * 4000 if len(calls) == 1 else SUMMARY if len(calls) == 3 else "done"
                if len(calls) == 1:
                    push_fixture_input(store, "s", InboxItem("http-late", "steer", {"content": "HTTP-ORIGINAL"}))
            else:
                status, content = (403 if mode == "retry" else 503), "failure"
            if status != 200:
                body = json.dumps(
                    {
                        "error": {
                            "message": "maximum context length exceeded" if mode == "ptl" else "Synthetic failure",
                            "type": "invalid_request_error",
                            "code": "context_length_exceeded" if mode == "ptl" else "synthetic_failure",
                        }
                    }
                ).encode()
                mime = "application/json"
            else:
                value = {"id": "completion-fixed", "created": 1, "model": "scripted"}
                if request.get("stream"):
                    chunk = value | {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
                    }
                    end = value | {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    body = ("data: " + json.dumps(chunk) + "\n\ndata: " + json.dumps(end) + "\n\ndata: [DONE]\n\n").encode()
                    mime = "text/event-stream"
                else:
                    body = json.dumps(
                        value
                        | {
                            "object": "chat.completion",
                            "choices": [
                                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                            ],
                        }
                    ).encode()
                    mime = "application/json"
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    receipt_seq = 0

    def hook(point, record):
        nonlocal receipt_seq
        if point == "after_commit":
            state, rows, _ = read_fixture_state(store, "s")
            for row in rows:
                r = row.record
                if row.seq > receipt_seq and r.kind in {"op_completed", "op_unknown"}:
                    coverage.append(
                        {
                            "kind": r.kind,
                            "operation_id": r.operation_id,
                            "error_code": r.payload.get("result", {}).get("error_code"),
                            "protected_start": protected_start(rows, state),
                        }
                    )
            receipt_seq = rows[-1].seq

    start(store)
    server = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        rt = runtime(store, database, [], hook=hook)
        if mode == "ptl":
            rt.config = replace(rt.config, no_tool_policy="continue", max_cycles=2)
        rt.memory_manager = MemoryManager(compact_threshold=100000, keep_recent_messages=1)
        rt.llm = VvLlmClient(
            [EndpointTarget(name, "synthetic-unused", f"http://127.0.0.1:{server.server_port}/{name}") for name in ("a", "b")],
            backend="openai",
            randomize_endpoints=False,
        )
        with patch("openai._base_client.SyncAPIClient._sleep_for_retry", lambda *a, **k: None):
            drive(store, "s", runtime=rt)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    state, rows, _ = read_fixture_state(store, "s")
    if mode == "unknown":
        assert protected_start(rows, state) == 1
        assert any(fact["kind"] == "op_unknown" for fact in coverage)
    else:
        assert protected_start(rows, state) is None
        terminal = next(row.record for row in rows if row.record.kind == "turn_ended").payload
        assert terminal["status"] == ("failed" if mode == "ptl" else "completed")
        if mode == "ptl":
            assert terminal["reason"] == "max_cycles"  # continue policy ends at the configured cycle limit
    if mode == "retry":
        assert coverage[0]["error_code"] == "model_request_rejected", coverage
        assert coverage[0]["protected_start"] == 1
        assert coverage[-1]["protected_start"] is None
    if mode == "ptl":
        assert coverage[1]["error_code"] == "prompt_too_long" and coverage[1]["protected_start"] is not None
        assert coverage[2]["protected_start"] is not None  # summary cannot advance primary coverage
        assert "HTTP-ORIGINAL" not in json.dumps(calls[2]["request"])
        assert sum(message.get("content") == "HTTP-ORIGINAL" for message in calls[3]["request"]["messages"]) == 1
        assert len(state.compactions) == 1
    return {"http_requests": calls, "coverage_at_receipts": coverage}


@pytest.mark.parametrize("decision", ["deny", "timeout"])
def test_approval_reason_keeps_original_group_after_ptl(store, database, decision):
    from vv_agent.tools.function import function_tool
    from vv_agent.types import ToolCall

    @function_tool(needs_approval=True)
    def gated() -> str:
        raise AssertionError("denial must not execute the tool")

    start(store)
    rt = runtime(
        store,
        database,
        [LLMResponse("seen facts " * 4000), LLMResponse("", [ToolCall("gate", "gated", {})])],
        [gated],
    )
    rt.config = replace(rt.config, no_tool_policy="continue", max_cycles=3)
    rt.memory_manager = MemoryManager(compact_threshold=100000, keep_recent_messages=1)
    drive(store, "s", runtime=rt)
    park = records_of(store, "op_parked")[-1]
    state, _, _ = read_fixture_state(store, "s")
    attempt = state.operations[park.operation_id].attempts[park.attempt]
    assert attempt.wait is not None
    handle = attempt.wait["handle"]
    assert park.payload["handle"]["scope"] != handle["scope"]
    assert handle["scope"] == ["gated"]
    payload = {
        "operation_id": park.operation_id,
        "attempt": park.attempt,
        "request_id": handle["request_id"],
        "request_digest": handle["request_digest"],
        "scope": handle["scope"],
        "decision": decision,
        "reason": "APPROVAL-ORIGINAL",
    }
    push_fixture_input(store, "s", InboxItem("approval-reply", "approval_answer", payload, park.turn_id, 0))
    push_fixture_input(store, "s", InboxItem("late-steer", "steer", {"content": "LATE-STEER"}))

    def reject(request):
        assert any("APPROVAL-ORIGINAL" in str(m.content) for m in request.messages)
        raise RuntimeError("maximum context length exceeded")

    def summary(request):
        assert "APPROVAL-ORIGINAL" not in request.messages[0].content
        assert request.metadata["purpose"] == "compaction"
        return LLMResponse(SUMMARY)

    def answer(request):
        index = next(i for i, m in enumerate(request.messages) if m.tool_calls)
        assert request.messages[index].tool_calls[0]["id"] == "gate"
        assert request.messages[index + 1].tool_call_id == "gate"
        assert "APPROVAL-ORIGINAL" in request.messages[index + 1].content
        return LLMResponse("done")

    rt.llm.steps[:] = [reject, summary, answer]
    drive(store, "s", runtime=rt)
    assert not rt.llm.steps
    assert len(records_of(store, "context_compacted")) == 1


class CallbackProvider:
    def __init__(self, *, unknown=False):
        self.unknown = unknown

    def submit(self, plan, *, context):
        from vv_agent.session.providers import Accepted, Unknown
        from vv_agent.types import ToolExecutionResult

        self.plan = plan
        self.result = ToolExecutionResult(tool_call_id=plan.payload["request"]["id"], content="PROVIDER-INPUT-ORIGINAL").to_dict()
        self.handle = {
            "kind": "provider",
            "provider": plan.payload["provider_binding"],
            "job_id": "job",
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "evidence": "proof",
            "query_ref": "query",
            "cancel_ref": None,
        }
        return Unknown("late tool acknowledgement") if self.unknown else Accepted(self.handle)

    def query(self, handle):
        from vv_agent.session.providers import Accepted

        return Accepted(handle)

    def cancel(self, handle):
        from vv_agent.session.providers import Accepted

        return Accepted(handle)

    def authenticate(self, item, plan):
        return item.payload["result"] == self.result


@pytest.mark.parametrize("delivery", ["correction", "audit"])
def test_provider_callback_only_protects_projected_content(store, database, delivery):
    from vv_agent.session.context import project_context
    from vv_agent.tools.function import function_tool
    from vv_agent.types import ToolCall

    @function_tool
    def effect() -> str:
        raise AssertionError("provider owns execution")

    provider = CallbackProvider(unknown=delivery == "correction")
    observed = []

    def push():
        plan = provider.plan
        payload = {
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "provider_binding": plan.payload["provider_binding"],
            "result": provider.result,
            "usage": {},
            "evidence": ["proof"],
        }
        push_fixture_input(store, "s", InboxItem("late-provider", "provider_result", payload, plan.turn_id, 0))

    def in_flight(request):
        assert any("tool_outcome_unknown" in str(m.content) for m in request.messages)
        push()
        return LLMResponse("response to the request before the correction")

    def answer(request):
        state, rows, _ = read_fixture_state(store, "s")
        messages = project_context(rows, state)
        index = next(i for i, m in enumerate(messages) if str(m.content).startswith("Correction for "))
        observed.append((protected_start(rows, state), index))
        assert sum("PROVIDER-INPUT-ORIGINAL" in str(m.content) for m in request.messages) == 1
        return LLMResponse("done")

    start(store)
    steps: list[ScriptStep] = [LLMResponse("seen old group", [ToolCall("e", "effect", {})])]
    if delivery == "correction":
        steps += [in_flight, answer]
    rt = runtime(store, database, steps, [effect], providers={"effect": provider})
    rt.memory_manager = MemoryManager(compact_threshold=100000, keep_recent_messages=1)
    drive(store, "s", runtime=rt)
    if delivery == "audit":
        state, _, _ = read_fixture_state(store, "s")
        push_fixture_input(store, "s", InboxItem("cancel", "control", {"action": "cancel"}, state.active_turn_id, 0))
        drive(store, "s", runtime=rt)
        push()
        drive(store, "s", runtime=rt)
    state, rows, _ = read_fixture_state(store, "s")
    receipts = [
        row.record for row in rows if row.record.kind == "op_completed" and row.record.operation_id == provider.plan.operation_id
    ]
    assert len(receipts) == 1 and receipts[0].payload["context"] == delivery
    if delivery == "audit":
        assert protected_start(rows, state) is None
        assert not any("PROVIDER-INPUT-ORIGINAL" in str(m.content) for m in project_context(rows, state))
    else:
        assert len(observed) == 1 and observed[0][0] == observed[0][1], observed
        assert protected_start(rows, state) is None
    assert (
        len(
            [
                row
                for row in rows
                if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "late-provider"
            ]
        )
        == 1
    )


@pytest.mark.parametrize("reply", ["user", "deny", "timeout"])
def test_cancelled_interaction_does_not_protect_audit_content(store, database, reply):
    from vv_agent.session.context import project_context
    from vv_agent.tools.function import function_tool
    from vv_agent.types import ToolCall

    @function_tool(needs_approval=True)
    def gated() -> str:
        raise AssertionError("cancelled approval must not execute")

    start(store)
    call = ToolCall("wait", "ask_user", {"question": "Choose"}) if reply == "user" else ToolCall("wait", "gated", {})
    rt = runtime(store, database, [LLMResponse("", [call])], [] if reply == "user" else [gated])
    drive(store, "s", runtime=rt)
    park = records_of(store, "op_parked")[-1]
    handle = park.payload["handle"]
    if reply == "user":
        kind = "user"
        payload = {"content": {"interaction_id": handle["interaction_id"], "text": "CANCELLED-ORIGINAL"}}
    else:
        kind = "approval_answer"
        payload = {
            "operation_id": park.operation_id,
            "attempt": park.attempt,
            "request_id": handle["request_id"],
            "request_digest": handle["request_digest"],
            "scope": handle["scope"],
            "decision": reply,
            "reason": "CANCELLED-ORIGINAL",
        }
    push_fixture_input(store, "s", InboxItem("cancel", "control", {"action": "cancel"}, park.turn_id, 0))
    push_fixture_input(store, "s", InboxItem("cancelled-reply", kind, payload, park.turn_id, 0))
    drive(store, "s", runtime=rt)
    state, rows, _ = read_fixture_state(store, "s")
    assert records_of(store, "turn_ended")[-1].payload["status"] == "cancelled"
    assert not any("CANCELLED-ORIGINAL" in str(m.content) for m in project_context(rows, state))
    assert protected_start(rows, state) is None


@pytest.mark.parametrize("reply", ["provider", "user", "deny", "timeout"])
@pytest.mark.parametrize("prepend", [False, True])
def test_normal_callback_after_snapshot_reaches_next_primary(store, database, reply, prepend):
    from vv_agent.kernel._content_publication import make_record_before_publication
    from vv_agent.kernel._reducer import logical_boundary_payloads, logical_boundary_sources
    from vv_agent.session.context import project_context
    from vv_agent.session.records import digest
    from vv_agent.tools.function import function_tool
    from vv_agent.types import ToolCall

    @function_tool(needs_approval=True)
    def gated() -> str:
        raise AssertionError("denied tool must not execute")

    @function_tool
    def effect() -> str:
        raise AssertionError("provider owns execution")

    provider = CallbackProvider()
    start(store)
    call = (
        ToolCall("callback", "effect", {})
        if reply == "provider"
        else ToolCall("callback", "ask_user", {"question": "Choose"})
        if reply == "user"
        else ToolCall("callback", "gated", {})
    )
    rt = runtime(store, database, [LLMResponse("", [call])], [effect, gated], providers={"effect": provider})
    drive(store, "s", runtime=rt)
    state, rows, _ = read_fixture_state(store, "s")
    assert protected_start(rows, state) is None
    source = project_context(rows, state)
    replacement = ([Message("system", "HOST")] if prepend else []) + source
    boundary = make_record_before_publication(
        "boundary_recorded",
        session_id="s",
        turn_id=state.active_turn_id,
        payload={
            "stage": "before_memory",
            "boundary_id": "inflight-snapshot",
            "source_operation_id": None,
            "source_digest": digest([m.to_dict() for m in source]),
            "data": {"messages": [m.to_dict() for m in replacement], "shared_state": {}},
        },
    )
    with (
        logical_boundary_payloads({boundary.record_id: boundary._payload}),
        logical_boundary_sources({boundary.record_id: source}),
    ):
        append_fixture_records(store, "s", [boundary], commit_id="snapshot")
    clear_fixture_state_cache(store)
    _, hydrated, _ = read_fixture_state(store, "s")
    park = next(row.record for row in reversed(hydrated) if row.record.kind == "op_parked")
    handle = park.payload["handle"]
    if reply == "provider":
        plan = provider.plan
        payload = {
            "operation_id": plan.operation_id,
            "attempt": 1,
            "request_digest": plan.payload["request_digest"],
            "provider_binding": plan.payload["provider_binding"],
            "result": provider.result,
            "usage": {},
            "evidence": ["proof"],
        }
        kind, marker = "provider_result", "PROVIDER-INPUT-ORIGINAL"
    elif reply == "user":
        payload = {"content": {"interaction_id": handle["interaction_id"], "text": "CALLBACK-ORIGINAL"}}
        kind, marker = "user", "CALLBACK-ORIGINAL"
    else:
        payload = {
            "operation_id": park.operation_id,
            "attempt": park.attempt,
            "request_id": handle["request_id"],
            "request_digest": handle["request_digest"],
            "scope": handle["scope"],
            "decision": reply,
            "reason": "CALLBACK-ORIGINAL",
        }
        kind, marker = "approval_answer", "CALLBACK-ORIGINAL"
    push_fixture_input(store, "s", InboxItem("callback-result", kind, payload, park.turn_id, 0))

    def answer(request):
        state0, committed, _ = read_fixture_state(store, "s")
        projected = project_context(committed, state0)
        assert any(marker in str(m.content) for m in request.messages)
        assert [m.to_dict() for m in request.messages] == [m.to_dict() for m in projected]
        assert protected_start(committed, state0) == next(i for i, m in enumerate(projected) if m.tool_calls)
        clear_fixture_state_cache(store)
        cold, cold_rows, _ = read_fixture_state(store, "s")
        assert [m.to_dict() for m in project_context(cold_rows, cold)] == [m.to_dict() for m in projected]
        return LLMResponse("done")

    rt.llm.steps[:] = [answer]
    drive(store, "s", runtime=rt)
    receipt = next(r for r in records_of(store, "op_completed") if r.operation_id == park.operation_id)
    assert receipt.payload["context"] == "normal"
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"
