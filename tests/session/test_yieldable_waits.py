"""Yielding completes an observation and applies steering in one real-store commit."""

from dataclasses import replace
from typing import Any

import pytest

from vv_agent.session import children
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.types import LLMResponse, ToolCall

from .test_batch_children import complete, deliver, rows
from .test_children import child_task
from .test_recovery_matrix import runtime, start


def waiting_runtime(database, *, count=1, declared=True, callback=None):
    observed = []

    def next_request(request):
        observed.append(request)
        return LLMResponse("Parent awaiting user")

    rt = runtime(
        database,
        [LLMResponse("", [ToolCall("child-call", "child_task", {})]), next_request],
        [child_task],
    )
    rt.config.no_tool_policy = "wait_user"

    def admit(_plan, *, tx):
        return [
            children.ChildSession(SessionSpec(f"yield-child-{i}", "test", "/tmp"), "work", yield_on_steer=declared)
            for i in range(count)
        ]

    rt.children["child_task"] = admit if callback is None else children.ChildTool(admit, on_yield=callback)
    return rt, observed


def steer(store, *, input_id="steer", generation=0):
    tid = read_state(store, "s")[0].active_turn_id
    with store.atomic() as tx:
        tx.push("s", InboxItem(input_id, "steer", {"content": "Change direction now"}, tid, generation))


@pytest.mark.parametrize("count", [1, 3])
def test_steer_yields_foreground_children_atomically_and_later_notifies(store, database, count):
    start(store)
    rt, observed = waiting_runtime(database, count=count)
    drive(store, "s", runtime=rt)
    park = rows(store, kind="op_parked")[0]
    active_children = []
    for i in range(count):
        sid = f"yield-child-{i}"
        provider = YieldProvider(declared=False)
        child_rt = runtime(
            database,
            [LLMResponse("", [ToolCall("work", "child_task", {})]), LLMResponse(sid + " done")],
            [child_task],
            providers={"child_task": provider},
        )
        drive(store, sid, runtime=child_rt)
        assert read_state(store, sid)[0].active_turn_id is not None
        active_children.append((sid, child_rt, provider))
    steer(store)
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    receipts = [r for r in rows(store, kind="op_completed") if r.record.operation_id == park.record.operation_id]
    assert len(receipts) == 1
    receipt = receipts[0]
    applied = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "steer")
    assert receipt.commit_id == applied.commit_id
    assert receipt.record.payload["yielded_by"] == "steer"
    messages = observed[0].messages
    assert any("Change direction now" in str(m.content) for m in messages)
    assert any("Wait interrupted by new input" in str(m.content) for m in messages)
    for sid, child_rt, provider in active_children:
        assert not rows(store, sid, "turn_ended")
        assert read_state(store, sid)[0].active_turn_id is not None
        assert not any(i.item.kind == "control" for i in store.peek_inbox(sid))
        from vv_agent.types import ToolExecutionResult

        h = provider.handle
        with store.atomic() as tx:
            tx.push(
                sid,
                InboxItem(
                    "child-work-done",
                    "provider_result",
                    {
                        "operation_id": h["operation_id"],
                        "attempt": h["attempt"],
                        "request_digest": h["request_digest"],
                        "provider_binding": h["provider"],
                        "result": ToolExecutionResult("work", "work done").to_dict(),
                        "usage": {},
                        "evidence": [h["evidence"]],
                    },
                    f"{sid}/turn/start",
                    0,
                ),
            )
        drive(store, sid, runtime=child_rt)
        deliver(store, sid)
    drive(store, "s", runtime=rt)
    notices = [r for r in rows(store, kind="input_applied") if r.record.payload["reason"] == "background child notification"]
    assert len(notices) == count
    from vv_agent.session.context import project_context

    state, retained, _ = read_state(store, "s")
    assert sum(m.role == "user" and "Child completed:" in str(m.content) for m in project_context(retained, state)) == count
    assert len([r for r in rows(store, kind="op_completed") if r.record.operation_id == park.record.operation_id]) == 1


def test_undeclared_foreground_wait_keeps_steer_pending(store, database):
    start(store)
    rt, observed = waiting_runtime(database, declared=False)
    drive(store, "s", runtime=rt)
    park = rows(store, kind="op_parked")[0]
    assert "yield_on_steer" not in park.record.payload
    steer(store)
    drive(store, "s", runtime=rt)
    assert not observed
    assert any(i.item.input_id == "steer" for i in store.peek_inbox("s"))
    complete(store, database, "yield-child-0")
    deliver(store, "yield-child-0")
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    assert all("yielded_by" not in r.record.payload for r in rows(store, kind="op_completed"))


@pytest.mark.parametrize("completed_first", [False, True])
def test_same_watermark_child_terminal_precedes_steer(store, database, completed_first):
    start(store)
    rt, observed = waiting_runtime(database)
    drive(store, "s", runtime=rt)
    complete(store, database, "yield-child-0")
    if completed_first:
        deliver(store, "yield-child-0")
        steer(store)
    else:
        steer(store)
        deliver(store, "yield-child-0")
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    assert all("yielded_by" not in r.record.payload for r in rows(store, kind="op_completed"))
    inputs = [r.record.payload["input"]["kind"] for r in rows(store, kind="input_applied")]
    assert inputs.index("child_result") < inputs.index("steer")


@pytest.mark.parametrize("raises", [False, True])
def test_host_callback_or_default_fallback_is_model_visible(store, database, raises):
    from vv_agent.types import ToolExecutionResult

    calls = []

    def render(plan, handles, reason):
        calls.append((plan.operation_id, handles, reason))
        if raises:
            raise RuntimeError("private callback details")
        return ToolExecutionResult(plan.payload["request"]["id"], "Host rendered interruption")

    start(store)
    rt, observed = waiting_runtime(database, callback=render)
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    assert len(calls) == 1
    assert calls[0][2] == "steer"
    expected = "Wait interrupted by new input" if raises else "Host rendered interruption"
    assert any(expected in str(m.content) for m in observed[0].messages)
    assert "private callback details" not in str([r.record.payload for r in rows(store)])


def test_mixed_child_members_wait_for_undeclared_sibling(store, database):
    start(store)
    rt, observed = waiting_runtime(database, count=2)
    admit = rt.children["child_task"]
    rt.children["child_task"] = lambda plan, *, tx: [replace(c, yield_on_steer=i == 0) for i, c in enumerate(admit(plan, tx=tx))]
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    assert not observed
    complete(store, database, "yield-child-1")
    deliver(store, "yield-child-1")
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    result = next(r.record.payload["result"] for r in rows(store, kind="op_completed") if "yielded_by" in r.record.payload)
    assert result["metadata"]["running_children"] == ["yield-child-0"]
    assert not rows(store, "yield-child-0", "turn_ended")


class YieldProvider:
    def __init__(self, *, declared=True, callback=False):
        self.handle: dict[str, Any] = {}
        self.queries = 0
        self.cancels = 0
        self.declared = declared
        self.callback = callback
        self.on_yield = self._render if callback is not None else None

    def submit(self, plan, *, context):
        from vv_agent.session.providers import Accepted

        self.handle = {
            "kind": "provider",
            "provider": plan.payload["provider_binding"],
            "job_id": "job",
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "evidence": "provider-accepted",
            "query_ref": None,
            "cancel_ref": None,
        }
        return Accepted(self.handle, yield_on_steer=self.declared)

    def query(self, handle):
        from vv_agent.session.providers import Accepted

        self.queries += 1
        return Accepted(handle, yield_on_steer=self.declared)

    def cancel(self, handle):
        self.cancels += 1
        raise AssertionError("a yielded provider handle must never be cancelled")

    def authenticate(self, item, plan):
        return item.payload["request_digest"] == self.handle["request_digest"]

    def _render(self, plan, handle, reason):
        from vv_agent.types import ToolExecutionResult

        assert handle == self.handle and reason == "steer"
        if not self.callback:
            raise RuntimeError("renderer unavailable")
        return ToolExecutionResult(plan.payload["request"]["id"], "Provider observation interrupted")


def provider_runtime(database, provider):
    rt, observed = waiting_runtime(database)
    rt.children.clear()
    rt.providers["child_task"] = provider
    return rt, observed


@pytest.mark.parametrize("callback", [None, False, True])
def test_accepted_provider_closes_without_cancel_and_late_result_is_audit(store, database, callback):
    from vv_agent.types import ToolExecutionResult

    start(store)
    provider = YieldProvider(callback=callback)
    rt, observed = provider_runtime(database, provider)
    drive(store, "s", runtime=rt)
    query_count = provider.queries
    steer(store)
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    assert provider.queries == query_count
    receipt = next(r for r in rows(store, kind="op_completed") if "yielded_by" in r.record.payload)
    assert "provider-accepted" in receipt.record.payload["evidence"]
    expected = "Provider observation interrupted" if callback else "Wait interrupted by new input"
    assert any(expected in str(m.content) for m in observed[0].messages)
    h = provider.handle
    tid = receipt.record.turn_id
    late = InboxItem(
        "late",
        "provider_result",
        {
            "operation_id": h["operation_id"],
            "attempt": h["attempt"],
            "request_digest": h["request_digest"],
            "provider_binding": h["provider"],
            "result": ToolExecutionResult("child-call", "Late real output").to_dict(),
            "usage": {},
            "evidence": [h["evidence"]],
        },
        tid,
        0,
    )
    with store.atomic() as tx:
        tx.push("s", late)
        tx.push(
            "s",
            InboxItem(
                "late-accept",
                "provider_evidence",
                {
                    "operation_id": h["operation_id"],
                    "attempt": h["attempt"],
                    "request_digest": h["request_digest"],
                    "handle": h,
                },
                tid,
                0,
            ),
        )
    drive(store, "s", runtime=rt)
    state = read_state(store, "s")[0]
    assert all(
        state.applied_inputs[i].payload["reason"] == "yielded provider handle: audit only" for i in ("late", "late-accept")
    )
    assert next(r for r in rows(store) if r.seq == receipt.seq).record.encode() == receipt.record.encode()
    with store.atomic() as tx:
        tx.push("s", InboxItem("cancel", "control", {"action": "cancel"}, tid, 0))
    drive(store, "s", runtime=rt)
    assert provider.cancels == 0 and provider.queries == query_count


def test_unfinished_undeclared_operation_blocks_whole_batch(store, database):
    start(store)
    rt, observed = waiting_runtime(database)
    provider = YieldProvider(declared=False)
    from vv_agent.tools.function import function_tool

    @function_tool
    def command() -> str:
        raise AssertionError("provider owns command")

    rt.agent.tools.append(command)
    rt.providers["command"] = provider
    rt.llm.steps[0] = LLMResponse("", [ToolCall("child-call", "child_task", {}), ToolCall("command", "command", {})])
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    assert not observed
    h = provider.handle
    from vv_agent.types import ToolExecutionResult

    with store.atomic() as tx:
        tx.push(
            "s",
            InboxItem(
                "command-done",
                "provider_result",
                {
                    "operation_id": h["operation_id"],
                    "attempt": h["attempt"],
                    "request_digest": h["request_digest"],
                    "provider_binding": h["provider"],
                    "result": ToolExecutionResult("command", "Command done").to_dict(),
                    "usage": {},
                    "evidence": [h["evidence"]],
                },
                "s/turn/initial",
                0,
            ),
        )
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    assert not rows(store, "yield-child-0", "turn_ended")


@pytest.mark.parametrize("action", ["finish", "cancel", "abort"])
def test_completion_after_parent_terminal_only_audits(store, database, action):
    start(store)
    rt, _ = waiting_runtime(database)
    if action == "finish":
        rt.config.no_tool_policy = "finish"
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    if action != "finish":
        with store.atomic() as tx:
            tx.push("s", InboxItem(action, "control", {"action": action}, "s/turn/initial", 0))
        drive(store, "s", runtime=rt)
    # Finish or acknowledge the ordinary cancellation on the child.
    drive(store, "yield-child-0", runtime=runtime(database, [LLMResponse("child done")]))
    deliver(store, "yield-child-0")
    before = [r.record.encode() for r in rows(store, kind="op_completed")]
    drive(store, "s", runtime=rt)
    late = [r.record for r in rows(store, kind="input_applied") if r.record.payload["input"]["kind"] == "child_result"]
    assert len(late) == 1
    assert late[0].payload["disposition"] == "noop"
    assert late[0].payload["reason"] == "late background completion: audit only"
    assert before == [r.record.encode() for r in rows(store, kind="op_completed")]


def yield_crash_worker(database, pipe, point):
    from .conftest import open_store

    with open_store(database) as store:
        rt, _ = waiting_runtime(database)
        rt.llm.steps.pop(0)

        def hook(at, record, *, tx):
            if at == point and record and record.kind == "op_completed" and "yielded_by" in record.payload:
                pipe.send("yield-cut")
                pipe.recv()

        rt.hook = hook
        drive(store, "s", runtime=rt)


@pytest.mark.persistent_store
@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
def test_process_kill_around_yield_commit_restores_both_or_neither(store, database, point):
    import multiprocessing

    from .test_recovery_matrix import force_expire

    start(store)
    rt, _ = waiting_runtime(database)
    drive(store, "s", runtime=rt)
    steer(store)
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=yield_crash_worker, args=(database, child, point))
    process.start()
    child.close()
    try:
        assert parent.poll(20)
        assert parent.recv() == "yield-cut"
        process.kill()
        process.join(10)
        state = read_state(store, "s")[0]
        assert ("steer" in state.applied_inputs) == (point == "after_commit")
        assert bool([r for r in rows(store, kind="op_completed") if "yielded_by" in r.record.payload]) == (
            point == "after_commit"
        )
        assert any(i.item.input_id == "steer" for i in store.peek_inbox("s")) == (point == "before_commit")
        force_expire(database)
        callbacks = []

        def render(plan, handles, reason):
            from vv_agent.types import ToolExecutionResult

            callbacks.append(reason)
            return ToolExecutionResult(plan.payload["request"]["id"], "Restored observation")

        resumed, observed = waiting_runtime(database, declared=False, callback=render)
        resumed.llm.steps.pop(0)
        drive(store, "s", runtime=resumed)
        assert len(observed) == 1
        assert callbacks == (["steer"] if point == "before_commit" else [])
        receipts = [r for r in rows(store, kind="op_completed") if "yielded_by" in r.record.payload]
        assert len(receipts) == 1
        applied = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "steer")
        assert applied.commit_id == receipts[0].commit_id
    finally:
        if process.is_alive():
            process.kill()
        process.join(10)
        parent.close()


@pytest.mark.parametrize("generation", [99])
def test_stale_steer_cannot_yield(store, database, generation):
    start(store)
    rt, observed = waiting_runtime(database)
    drive(store, "s", runtime=rt)
    steer(store, generation=generation)
    drive(store, "s", runtime=rt)
    assert not observed
    assert not any("yielded_by" in r.record.payload for r in rows(store))


def test_app_server_completes_yielded_tool_and_fork_inherits_context_only(store, monkeypatch):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider
    from vv_agent.app_server import DefaultAppServerHost

    from .test_app_server_fork import send, server_for

    calls = []

    def model(request):
        calls.append(request)
        return LLMResponse("", [ToolCall("child-call", "child_task", {})]) if len(calls) == 1 else LLMResponse("done")

    provider = YieldProvider(callback=True)
    host = DefaultAppServerHost(
        agent=Agent("yield-server", "Be precise.", tools=[child_task]),
        run_config=RunConfig(
            model_provider=ScriptedModelProvider.from_callback("test", "model", model),
            tool_providers={"child_task": provider},
            session_memory_enabled=False,
        ),
    )
    server = server_for(store, host)
    from threading import Event

    messages = []
    completed_event = Event()
    write = server.transport.write_outbound

    def capture(payload):
        messages.append(payload)
        write(payload)
        if payload.get("method") == "item/completed" and payload["params"].get("type") == "toolCall":
            completed_event.set()

    monkeypatch.setattr(server.transport, "write_outbound", capture)
    try:
        send(server, "create", "thread/start", {"inputId": "source"})
        turn = send(
            server,
            "start",
            "turn/start",
            {
                "threadId": "source",
                "inputId": "initial",
                "expectedTurnId": None,
                "input": [{"type": "text", "text": "go"}],
            },
        )["turnId"]
        server.run_adapter.join()
        assert provider.handle is not None
        send(
            server,
            "steer",
            "turn/steer",
            {
                "threadId": "source",
                "inputId": "new-direction",
                "expectedTurnId": turn,
                "input": [{"type": "text", "text": "Change direction now"}],
            },
        )
        server.wake("source")
        server.run_adapter.join()
        assert completed_event.wait(5), messages
        completed = [w for w in messages if w.get("method") == "item/completed" and w["params"].get("type") == "toolCall"]
        assert completed, messages
        assert len(calls) == 2
        assert any("Change direction now" in str(m.content) for m in calls[-1].messages)
        result = send(server, "read", "thread/read", {"threadId": "source"})
        assert "Provider observation interrupted" in str(result)
        fork = send(server, "fork", "thread/fork", {"threadId": "source", "forkId": "copy", "throughTurnId": turn})
        state, records, _ = read_state(store, fork["threadId"])
        assert not state.operations and state.active_turn_id is None
        seed = records[0].record.payload["attributes"]["seed"]
        assert "Provider observation interrupted" in str(seed["messages"])
        assert "Change direction now" in str(seed["messages"])
        assert all(r.record.kind != "op_parked" for r in records)
    finally:
        server.kernel.close()


def test_suspended_turn_does_not_yield_until_resumed(store, database):
    start(store)
    rt, observed = waiting_runtime(database)
    drive(store, "s", runtime=rt)
    with store.atomic() as tx:
        tx.push("s", InboxItem("suspend", "control", {"action": "suspend"}, "s/turn/initial", 0))
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    assert not observed
    with store.atomic() as tx:
        tx.push("s", InboxItem("resume", "control", {"action": "resume"}, "s/turn/initial", 0))
    drive(store, "s", runtime=rt)
    assert len(observed) == 1


def test_yield_producer_matches_local_contract(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    from support.contract import fixture_root

    produced = tmp_path / "yieldable_waits.json"
    process = subprocess.run(
        [sys.executable, "-m", "scripts.yieldable_wait_fixtures", "--output", str(produced)],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    actual = json.loads(produced.read_text())
    expected = json.loads((fixture_root() / "yieldable_waits.json").read_text())
    assert actual == expected
    for case in actual["cases"]:
        assert case["completed_events"] == ["tool_call_completed"]
        if case["atomic_yield"]:
            assert case["name"] != "terminal_before_steer"
            assert any("yielded_by" in r["payload"] for r in case["records"])


def test_multiple_observations_share_one_steering_commit(store, database):
    start(store)
    rt, observed = waiting_runtime(database)
    rt.llm.steps[0] = LLMResponse("", [ToolCall(f"call-{i}", "child_task", {}) for i in range(2)])
    rt.children["child_task"] = lambda plan, *, tx: children.ChildSession(
        SessionSpec(plan.payload["request"]["id"], "test", "/tmp"),
        "work",
        yield_on_steer=True,
    )
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    receipts = [r for r in rows(store, kind="op_completed") if "yielded_by" in r.record.payload]
    applied = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "steer")
    assert len(receipts) == 2
    assert {r.commit_id for r in [*receipts, applied]} == {applied.commit_id}
    assert [r.record.payload["result"]["tool_call_id"] for r in receipts] == ["call-0", "call-1"]


@pytest.mark.parametrize("invalid", ["call_id", "directive", "output_type"])
def test_invalid_renderer_falls_back_without_stalling_steer(store, database, invalid):
    from vv_agent.types import ToolDirective, ToolExecutionResult

    def render(plan, handles, reason):
        if invalid == "output_type":
            return "not a tool result"
        return ToolExecutionResult(
            "wrong" if invalid == "call_id" else plan.payload["request"]["id"],
            "bad renderer",
            directive=ToolDirective.FINISH if invalid == "directive" else ToolDirective.CONTINUE,
        )

    start(store)
    rt, observed = waiting_runtime(database, callback=render)
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    result = next(r.record.payload["result"] for r in rows(store, kind="op_completed") if "yielded_by" in r.record.payload)
    assert result["metadata"]["yield_callback_failed"] is True
    assert any("Wait interrupted by new input" in str(m.content) for m in observed[0].messages)


def test_implicit_active_turn_steer_uses_the_same_atomic_rule(store, database):
    start(store)
    rt, observed = waiting_runtime(database)
    drive(store, "s", runtime=rt)
    with store.atomic() as tx:
        tx.push("s", InboxItem("implicit", "steer", {"content": "Implicit active turn input"}))
    drive(store, "s", runtime=rt)
    assert len(observed) == 1
    receipt = next(r for r in rows(store, kind="op_completed") if r.record.payload.get("yielded_by") == "implicit")
    applied = next(r for r in rows(store, kind="input_applied") if r.record.payload["input"]["input_id"] == "implicit")
    assert applied.commit_id == receipt.commit_id
    assert any("Implicit active turn input" in str(m.content) for m in observed[0].messages)


def test_store_rejects_yield_completion_without_steer_in_its_commit(store, database):
    from vv_agent.session.store import Conflict

    start(store)
    rt, _ = waiting_runtime(database)
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    receipt = next(r for r in rows(store, kind="op_completed") if "yielded_by" in r.record.payload)
    lease = store.acquire("s", owner="split-commit", ttl_ms=15000)
    assert lease is not None
    try:
        with pytest.raises(Conflict, match="yield completion and steering must commit together"), store.atomic() as tx:
            tx.append("s", lease=lease, expected_seq=store.read("s").head_seq, commit_id="split", records=(receipt.record,))
    finally:
        store.release(lease)


def test_local_handle_keeps_yielded_children_scheduled(store, database, monkeypatch):
    from vv_agent.run_handle import RunHandle
    from vv_agent.session.surfaces import SessionDriver

    start(store)
    rt, _ = waiting_runtime(database)
    drive(store, "s", runtime=rt)
    steer(store)
    drive(store, "s", runtime=rt)
    child_rt = runtime(database, [LLMResponse("child completed locally")])
    monkeypatch.setattr(rt, "child_runtime", lambda _store, _sid: child_rt)
    driver = SessionDriver(store=store)
    parent = RunHandle(driver, "s", "s/turn/initial", rt, None)
    driver.handles.append(parent)
    try:
        parent.start()
        parent.join(10)
        assert parent.done()
        parent.result()
        for handle in driver.handles:
            handle.join(10)
            assert handle.done()
            handle.result()
        assert rows(store, "yield-child-0", "turn_ended")
        drive(store, "s", runtime=rt)
        assert any(r.record.payload.get("reason") == "background child notification" for r in rows(store, kind="input_applied"))
    finally:
        driver.close()


def test_yielded_child_user_wait_remains_visible_to_local_host(store, database, monkeypatch):
    from vv_agent.session.surfaces import SessionDriver

    start(store)
    rt, _ = waiting_runtime(database)
    drive(store, "s", runtime=rt)
    child_rt = runtime(database, [LLMResponse("", [ToolCall("child-question", "ask_user", {"question": "Child question"})])])
    drive(store, "yield-child-0", runtime=child_rt)
    steer(store)
    drive(store, "s", runtime=rt)
    monkeypatch.setattr(rt, "child_runtime", lambda _store, _sid: child_rt)
    driver = SessionDriver(store=store)
    try:
        waits = driver.waits("s", rt)
        assert any(sid == "yield-child-0" and handle.get("question") == "Child question" for sid, _, handle in waits)
    finally:
        driver.close()


@pytest.mark.parametrize("mode", ["steer", "complete"])
def test_local_foreground_child_execution_does_not_block_parent_steering(store, database, monkeypatch, mode):
    from threading import Event

    from vv_agent.run_handle import RunHandle
    from vv_agent.session.surfaces import SessionDriver

    started, release, seen = Event(), Event(), Event()

    def child_model(_request):
        started.set()
        assert release.wait(15)
        return LLMResponse("child finished")

    start(store)
    rt, observed = waiting_runtime(database)
    callback = rt.llm.steps[1]

    def parent_model(request):
        result = callback(request)
        seen.set()
        return result

    rt.llm.steps[1] = parent_model
    child_rt = runtime(database, [child_model])
    monkeypatch.setattr(rt, "child_runtime", lambda _store, _sid: child_rt)
    driver = SessionDriver(store=store)
    parent = RunHandle(driver, "s", "s/turn/initial", rt, None)
    driver.handles.append(parent)
    try:
        parent.start()
        assert started.wait(10)
        assert not parent.done()
        if mode == "steer":
            parent.steer("Change direction now")
            assert seen.wait(2), "parent model remained blocked behind the child execution"
            assert not rows(store, "yield-child-0", "turn_ended")
            assert any("Change direction now" in str(m.content) for m in observed[0].messages)
        else:
            release.set()
            assert seen.wait(10)
        parent.join(5)
        assert parent.done()
        parent.result()
        yielded = [r for r in rows(store, kind="op_completed") if "yielded_by" in r.record.payload]
        assert len(yielded) == (1 if mode == "steer" else 0)
    finally:
        release.set()
        for handle in driver.handles:
            handle.join(10)
        driver.close()
