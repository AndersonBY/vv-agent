"""Retained child continuation, reply, identity and terminal recovery producers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from support.kernel_runtime import start_runner

from vv_agent import Agent, AgentStatus, RunConfig, Runner, ScriptedModelProvider, SubAgentConfig, function_tool
from vv_agent.session.children import child_delivery, child_handles
from vv_agent.session.kernel import drive
from vv_agent.session.store import Conflict
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse, ToolCall, ToolDirective, ToolExecutionResult


def _contract() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "fixtures/parity/configured_sub_agent.json").read_text())


def _child(driver, parent):
    rows = driver.store.read_state(parent.session_id)[1]
    parked = next(r.record for r in rows if r.record.kind == "op_parked" and r.record.payload["handle"]["kind"] == "child")
    return child_handles(parked.payload["handle"])[0]


def _start(driver, steps, *, waiting=False):
    @function_tool
    def wait_tool():
        return ToolExecutionResult("", "Choose", directive=ToolDirective.WAIT_USER)

    child_args = {"agent_id": "worker", "task_description": "first"}
    provider = ScriptedModelProvider.from_steps(
        "scripted", "m", [LLMResponse("", [ToolCall("delegate", "create_sub_task", child_args)]), *steps]
    )
    return start_runner(
        driver,
        "parent",
        Agent("parent", "Delegate.", tools=[wait_tool], sub_agents={"worker": SubAgentConfig(model="m", description="Work.")}),
        "go",
        run_config=RunConfig(model_provider=provider),
    )


def test_configured_sub_agent_continuation_replays_complete_prior_turn(monkeypatch):
    driver = SessionDriver()
    requests = []

    def continuing(request):
        requests.append(request)
        return LLMResponse("continued")

    try:
        parent = _start(driver, [LLMResponse("first done"), LLMResponse("parent done"), continuing])
        assert parent.result().status is AgentStatus.COMPLETED
        child = _child(driver, parent)
        monkeypatch.setattr(parent, "_schedule_background", lambda: None)
        manager = parent.runtime.child_tasks(driver.store, parent.session_id)
        original = driver.store.read_state(child["session_id"])[1]
        assert manager.message(child["session_id"], "continuation", "more") == "continued"
        rt = parent.runtime.child_runtime(driver.store, child["session_id"])
        drive(driver.store, child["session_id"], runtime=rt)
        state, rows, _ = driver.store.read_state(child["session_id"])
        assert len(state.turns) == 2 and state.active_turn_id is None
        assert rows[: len(original)] == original
        assert any(m.content == "first done" for m in requests[0].messages)
        assert any(m.role == "user" and m.content == "more" for m in requests[0].messages)
        assert _contract()["continuation"]["full_history_from_records"]
        with driver.store.atomic() as tx:
            child_delivery(driver.store, tx, child["session_id"])
        assert not driver.store.peek_inbox("parent")
        assert parent.result().final_output == "parent done"
    finally:
        driver.close()


def test_continuation_admission_is_idempotent_and_conflicts_have_zero_writes(monkeypatch):
    driver = SessionDriver()
    try:
        parent = _start(driver, [LLMResponse("child done"), LLMResponse("parent done")])
        parent.result()
        child = _child(driver, parent)["session_id"]
        monkeypatch.setattr(parent, "_schedule_background", lambda: None)
        manager = parent.runtime.child_tasks(driver.store, "parent")
        assert manager.message(child, "exact/😀", "more") == "continued"
        before = driver.store.read(child).head_seq
        pending = driver.store.peek_inbox(child)
        assert manager.message(child, "exact/😀", "more") == "continued"
        with pytest.raises(Conflict, match="different content"):
            manager.message(child, "exact/😀", "changed")
        assert driver.store.read(child).head_seq == before
        assert driver.store.peek_inbox(child) == pending
        assert manager.get(" " + child) is None
        assert parent.runtime.child_tasks(driver.store, "other-parent").get(child) is None
        with pytest.raises(KeyError):
            parent.runtime.child_tasks(driver.store, "other-parent").message(child, "forged", "more")
    finally:
        driver.close()


def test_wait_user_reply_stays_on_child_and_parent_adopts_only_authenticated_terminal(monkeypatch):
    driver = SessionDriver()
    try:
        parent = _start(
            driver,
            [
                LLMResponse("", [ToolCall("ask", "wait_tool", {})]),
                LLMResponse("child done"),
                LLMResponse("parent done"),
            ],
        )
        first = parent.result()
        assert first.status is AgentStatus.WAIT_USER
        child = _child(driver, parent)
        original_tid = child["turn_id"]
        monkeypatch.setattr(parent, "_schedule_background", lambda: None)
        manager = parent.runtime.child_tasks(driver.store, "parent")
        assert manager.message(child["session_id"], "answer", "choice") == "message_queued"
        assert driver.store.peek_inbox(child["session_id"])[0].item.target_turn_id == original_tid
        assert not driver.store.peek_inbox("parent")
        resumed = Runner.resume(parent.session_id, parent.run_id)
        assert resumed.status is AgentStatus.COMPLETED and resumed.final_output == "parent done"
        state, rows, _ = driver.store.read_state(child["session_id"])
        assert list(state.turns) == [original_tid]
        assert sum(r.record.kind == "turn_ended" for r in rows) == 1
        assert _contract()["continuation"]["same_turn_reply"]
        assert _contract()["continuation"]["terminal_only_parent_adoption"]
    finally:
        driver.close()


def test_retained_child_cancel_does_not_cancel_parent_or_admit_a_new_turn(monkeypatch):
    driver = SessionDriver()
    try:
        parent = _start(driver, [LLMResponse("", [ToolCall("ask", "wait_tool", {})]), LLMResponse("parent done")])
        parent.result()
        child = _child(driver, parent)
        monkeypatch.setattr(parent, "_schedule_background", lambda: None)
        manager = parent.runtime.child_tasks(driver.store, "parent")
        manager.handle(child["session_id"]).cancel()
        resumed = Runner.resume(parent.session_id, parent.run_id)
        assert resumed.status is AgentStatus.COMPLETED
        state, rows, _ = driver.store.read_state(child["session_id"])
        assert len(state.turns) == 1 and state.active_turn_id is None
        assert next(r.record.payload["status"] for r in rows if r.record.kind == "turn_ended") == "cancelled"
        assert _contract()["cancellation"]["child_does_not_cancel_parent"]
    finally:
        driver.close()


def test_child_continuation_schedules_locally_with_host_wake_separate():
    driver = SessionDriver()
    wakes = []
    try:
        parent = _start(driver, [LLMResponse("child done"), LLMResponse("parent done"), LLMResponse("continued")])
        parent.result()
        child = _child(driver, parent)["session_id"]
        parent.runtime.wake = wakes.append
        manager = parent.runtime.child_tasks(driver.store, "parent")
        assert manager.message(child, "continuation", "more") == "continued"
        background = driver._background[child]
        background.join(timeout=5)
        assert background.done() and background._error is None
        state, rows, _ = driver.store.read_state(child)
        assert len(state.turns) == 2 and state.active_turn_id is None
        assert [r.record.payload["result"] for r in rows if r.record.kind == "turn_ended"] == ["child done", "continued"]
        assert wakes == [child]
        assert parent.runtime.wake == wakes.append
    finally:
        driver.close()
