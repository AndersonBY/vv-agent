"""sub_task_status uses accepted child observations instead of blocking one worker."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from vv_agent import Agent, RunConfig, SubAgentConfig
from vv_agent.session.delegation import ChildTasks, SubTaskStatusProvider
from vv_agent.session.kernel import drive
from vv_agent.session.projection import SessionHydration, hydrate_records, hydration_for_session_store, project_records
from vv_agent.session.providers import Accepted, Definitive
from vv_agent.session.records import InboxItem
from vv_agent.types import LLMResponse, ToolCall

from .helpers import push_fixture_input, wrap_fixture_store
from .test_delegation_parity import open_fixture_store, parent_llm, routed, runtime
from .test_tools_control_parity import admit


@pytest.fixture
def store(store):
    wrapped = wrap_fixture_store(store)
    root = getattr(wrapped, "_root", wrapped)
    if hasattr(root, "_rows"):
        root._rows("UPDATE sk_settings SET write_level=2 WHERE write_level<2")
    return wrapped


@pytest.fixture
def database_clock(store, monkeypatch):
    raw_store = store
    while True:
        nested = getattr(raw_store, "_store", None)
        if nested is None or nested is raw_store:
            break
        raw_store = nested
    now = raw_store._now()
    monkeypatch.setattr(type(raw_store), "clock_sql", str(now))

    def advance(milliseconds: int) -> None:
        nonlocal now
        now += milliseconds
        monkeypatch.setattr(type(raw_store), "clock_sql", str(now))

    return advance


@contextmanager
def _worker_heartbeat_store(database, capabilities):
    with open_fixture_store(database) as opened:
        worker_store = getattr(opened, "_root", opened)
        worker_store.configure_worker(level=2, handler_versions=("1",), capabilities=capabilities)
        yield opened


def _parent_with_child(
    store,
    database,
    workspace: Path,
    *,
    wait_for_completion: bool = True,
    wait_for_response: bool = False,
    message: str | None = None,
    status_target: str | None = None,
    child_steps: list[Any] | None = None,
):
    child_id: dict[str, str] = {}

    def status_step(_request):
        arguments: dict[str, Any] = {
            "task_ids": [status_target if status_target is not None else child_id["id"]],
            "wait_for_completion": wait_for_completion,
            "wait_for_response": wait_for_response,
        }
        if wait_for_completion or wait_for_response:
            arguments["max_wait_seconds"] = 60
        if message is not None:
            arguments["message"] = message
        return LLMResponse("", [ToolCall("status", "sub_task_status", arguments)])

    steps = [
        LLMResponse(
            "",
            [
                ToolCall(
                    "delegate",
                    "create_sub_task",
                    {
                        "agent_id": "worker",
                        "task_description": "work",
                        "wait_for_completion": False,
                    },
                )
            ],
        ),
        LLMResponse("parent ready"),
        status_step,
        LLMResponse("parent observed child"),
    ]
    agent = Agent(
        "parent",
        "Parent instructions.",
        model="parent",
        sub_agents={"worker": SubAgentConfig(model="m", description="Worker instructions.")},
    )
    config = RunConfig(
        workspace=workspace,
        model_provider=routed(steps, child_steps or [LLMResponse("child done")]),
    )
    admit(store, config, sid="parent")
    parent_rt = runtime(store, database, agent, config, parent_llm(config), sid="parent")
    worker_store = getattr(store, "_root", store)
    capabilities = worker_store.worker_supports.capabilities
    parent_rt.heartbeat_store = lambda: _worker_heartbeat_store(database, capabilities)
    drive(store, "parent", runtime=parent_rt)

    children = [sid for sid in store.list_sessions() if sid != "parent"]
    assert len(children) == 1
    child_id["id"] = children[0]
    return agent, config, parent_rt, children[0]


def _follow_up(store, input_id: str = "status-input") -> None:
    push_fixture_input(store, "parent", InboxItem(input_id, "user", {"content": "check child"}))


def _complete_child(store, parent_rt, child_id: str) -> None:
    drive(store, child_id, runtime=parent_rt.child_runtime(store, child_id))


def _tool_result(store, session_id: str, tool_call_id: str) -> dict[str, Any]:
    hydration = SessionHydration.from_store(
        store,
        tenant_id=store.tenant_id,
        session_id=session_id,
    )
    seen: list[dict[str, Any]] = []
    for stored in reversed(store.read_state(session_id)[1]):
        if stored.record.kind != "op_completed":
            continue
        record = hydrate_records((stored,), hydration)[0].record
        result = record.payload["result"]
        seen.append(result)
        if result.get("tool_call_id") == tool_call_id:
            return result
    raise AssertionError(f"missing completed tool result {tool_call_id}")


def _logical_parked_handle(store, session_id: str) -> dict[str, Any]:
    hydration = SessionHydration.from_store(store, tenant_id=store.tenant_id, session_id=session_id)
    for record in reversed(store.read_state(session_id)[1]):
        if record.record.kind != "op_parked":
            continue
        hydrated = hydrate_records((record,), hydration)[0].record
        handle = hydrated.payload["handle"]
        if handle["kind"] == "provider":
            return handle
    raise AssertionError("missing provider handle")


@pytest.mark.sqlite_file
def test_nonwait_pending_status_is_definitive_and_does_not_block_parent(store, database, tmp_path):
    _agent, _config, parent_rt, child_id = _parent_with_child(store, database, tmp_path, wait_for_completion=False)
    _follow_up(store, "nonwait-input")
    drive(store, "parent", runtime=parent_rt)

    result = _tool_result(store, "parent", "status")
    assert result["metadata"]["tasks"][0]["task_id"] == child_id
    assert result["metadata"]["tasks"][0]["status"] in {"pending", "running"}
    assert "wait_for_completion" not in result["metadata"]
    assert not any(
        record.record.kind == "op_parked"
        and record.record.turn_id == "parent/turn/nonwait-input"
        and record.record.payload["handle"]["kind"] == "provider"
        for record in store.read_state("parent")[1]
    )
    assert store.read_state("parent")[0].active_turn_id is None


@pytest.mark.sqlite_file
def test_message_unknown_task_is_definitive_not_found(store, database, tmp_path):
    _agent, _config, parent_rt, _child_id = _parent_with_child(
        store,
        database,
        tmp_path,
        wait_for_completion=False,
        message="continue",
        status_target="unknown",
    )
    _follow_up(store, "unknown-message-input")
    drive(store, "parent", runtime=parent_rt)

    result = _tool_result(store, "parent", "status")
    expected = {
        "details": {"task_id": "unknown"},
        "error": "Sub-task unknown not found.",
        "error_code": "sub_task_not_found",
        "ok": False,
    }
    assert result["status_code"] == "ERROR"
    assert result["error_code"] == "sub_task_not_found"
    assert result["metadata"] == expected
    assert json.loads(result["content"]) == expected
    events = project_records(
        store.read_state("parent")[1],
        hydration=hydration_for_session_store(store, "parent"),
    )
    completed = next(event for event in events if event.type == "tool_call_completed" and event.tool_call_id == "status")
    assert completed.error_code == "sub_task_not_found"
    assert not any(
        record.record.kind == "op_parked"
        and record.record.turn_id == "parent/turn/unknown-message-input"
        and record.record.payload["handle"]["kind"] == "provider"
        for record in store.read_state("parent")[1]
    )
    assert store.read_state("parent")[0].active_turn_id is None


@pytest.mark.persistent_store
def test_pending_wait_is_accepted_then_restart_queries_completed_child(store, database, tmp_path, database_clock, monkeypatch):
    def forbidden_wait(*_args, **_kwargs):
        raise AssertionError("accepted sub_task_status must never call ChildTasks.wait")

    monkeypatch.setattr(ChildTasks, "wait", forbidden_wait)
    agent, config, parent_rt, child_id = _parent_with_child(store, database, tmp_path)
    _follow_up(store)
    drive(store, "parent", runtime=parent_rt)

    state, records, _ = store.read_state("parent")
    assert state.active_turn_id is not None
    parked = next(record.record for record in reversed(records) if record.record.kind == "op_parked")
    assert parked.payload["handle"]["kind"] == "provider"
    assert not any(record.record.kind == "turn_ended" and record.record.turn_id == state.active_turn_id for record in records)

    _complete_child(store, parent_rt, child_id)
    database_clock(1_000)
    restarted = runtime(store, database, agent, config, parent_llm(config), sid="parent")
    drive(store, "parent", runtime=restarted)
    result = _tool_result(store, "parent", "status")
    assert result["status_code"] == "SUCCESS"
    assert result["metadata"]["tasks"][0]["task_id"] == child_id
    assert result["metadata"]["tasks"][0]["status"] == "completed"
    assert result["metadata"]["tasks"][0]["final_answer"] == "child done"
    assert result["metadata"]["running_task_ids"] == []
    assert result["metadata"]["wait_exceeded"] is False
    assert store.read_state("parent")[0].active_turn_id is None


@pytest.mark.sqlite_file
def test_message_wait_response_is_accepted_and_queries_continuation(store, database, tmp_path, database_clock):
    _agent, _config, parent_rt, child_id = _parent_with_child(
        store,
        database,
        tmp_path,
        wait_for_completion=False,
        wait_for_response=True,
        message="extra",
        child_steps=[LLMResponse("child done"), LLMResponse("child continued")],
    )
    _follow_up(store, "message-input")
    drive(store, "parent", runtime=parent_rt)
    assert any(record.record.kind == "op_parked" for record in store.read_state("parent")[1])

    _complete_child(store, parent_rt, child_id)
    database_clock(1_000)
    drive(store, "parent", runtime=parent_rt)

    result = _tool_result(store, "parent", "status")
    assert result["metadata"]["interaction"]["task_id"] == child_id
    assert result["metadata"]["interaction"]["action"] == "continued"
    assert result["metadata"]["tasks"][0]["final_answer"] == "child continued"
    assert store.read_state("parent")[0].active_turn_id is None


@pytest.mark.sqlite_file
def test_all_terminal_wait_is_definitive_without_parking(store, database, tmp_path):
    _agent, _config, parent_rt, child_id = _parent_with_child(store, database, tmp_path)
    _complete_child(store, parent_rt, child_id)
    _follow_up(store, "terminal-input")
    drive(store, "parent", runtime=parent_rt)

    result = _tool_result(store, "parent", "status")
    assert result["metadata"]["tasks"][0]["status"] == "completed"
    assert result["metadata"]["wait_exceeded"] is False
    assert not any(
        record.record.kind == "op_parked"
        and record.record.turn_id == "parent/turn/terminal-input"
        and record.record.payload["handle"]["kind"] == "provider"
        for record in store.read_state("parent")[1]
    )


@pytest.mark.sqlite_file
def test_wait_timeout_uses_persisted_database_deadline(store, database, tmp_path, database_clock):
    agent, config, parent_rt, _child_id = _parent_with_child(store, database, tmp_path)
    _follow_up(store)
    drive(store, "parent", runtime=parent_rt)
    assert any(record.record.kind == "op_parked" for record in store.read_state("parent")[1])

    database_clock(60_000)
    restarted = runtime(store, database, agent, config, parent_llm(config), sid="parent")
    drive(store, "parent", runtime=restarted)

    result = _tool_result(store, "parent", "status")
    assert result["metadata"]["wait_exceeded"] is True
    assert result["metadata"]["tasks"][0]["status"] in {"pending", "running"}
    assert result["metadata"]["message"].startswith("Sub-task(s) are still running")
    assert store.read_state("parent")[0].active_turn_id is None


@pytest.mark.sqlite_file
def test_cancel_queries_existing_child_without_resubmit_or_model_call(store, database, tmp_path):
    _agent, _config, parent_rt, child_id = _parent_with_child(store, database, tmp_path)
    _follow_up(store, "cancel-input")
    drive(store, "parent", runtime=parent_rt)
    handle = _logical_parked_handle(store, "parent")

    raw_store = store
    while True:
        nested = getattr(raw_store, "_store", None)
        if nested is None or nested is raw_store:
            break
        raw_store = nested
    opened = 0
    open_store = parent_rt.heartbeat_store

    def observed_heartbeat_store():
        nonlocal opened
        opened += 1
        return open_store()

    parent_rt.heartbeat_store = observed_heartbeat_store
    provider = SubTaskStatusProvider(parent_rt, confirmed_poll=raw_store._now)
    operations = set(store.read_state("parent")[0].operations)

    before = opened
    pending = provider.cancel(handle)
    assert opened == before + 1
    assert isinstance(pending, Accepted)
    assert pending.handle == handle

    _complete_child(store, parent_rt, child_id)
    before = opened
    completed = provider.cancel(handle)
    assert opened == before + 1
    assert isinstance(completed, Definitive)
    assert completed.evidence == (handle["evidence"],)
    assert completed.usage == {}
    assert completed.result["metadata"]["tasks"][0]["final_answer"] == "child done"

    # The accepted attempt is still queryable; cancellation did not dispatch a second model or tool operation.
    assert set(store.read_state("parent")[0].operations) == operations
