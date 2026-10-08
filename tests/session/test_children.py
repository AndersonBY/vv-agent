"""Child operations use real PostgreSQL logs, inboxes and consumer transactions."""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from vv_agent.session import children
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.store import Conflict
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall

from .conftest import open_store
from .test_recovery_matrix import runtime, start


@function_tool
def child_task() -> str:
    raise AssertionError("child must use the session kernel")


def parent_runtime(database, *, background=False, steps=None, hook=lambda *_: None):
    return runtime(
        database,
        steps if steps is not None else [LLMResponse("", [ToolCall("child-call", "child_task", {})]), LLMResponse("parent done")],
        [child_task],
        children={
            "child_task": lambda plan: children.ChildSession(
                SessionSpec("child", "test", "/tmp"), "do child", background=background
            )
        },
        hook=hook,
    )


def parked(store, database, **kwargs):
    start(store)
    rt = parent_runtime(database, **kwargs)
    drive(store, "s", runtime=rt)
    state, records, _ = read_state(store, "s")
    handle = next(r.record.payload["handle"] for r in records if r.record.kind == "op_parked")
    return rt, state, handle


def complete_child(store, database):
    drive(store, "child", runtime=runtime(database, [LLMResponse("child done")]))


def delivered(store):
    with store.atomic() as tx:
        return children.child_delivery(store, tx, "child")


def child_result(store):
    return next(i.item for i in store.peek_inbox("s") if i.item.kind == "child_result")


def test_blocking_child_success(store, database):
    rt, state, handle = parked(store, database)
    assert state.phase == "parked"
    assert handle["delivery_target"]["operation_id"] in state.operations
    assert not any(r.record.kind == "turn_ended" for r in read_state(store, "s")[1])
    complete_child(store, database)
    delivered(store)
    drive(store, "s", runtime=rt)
    state, records, _ = read_state(store, "s")
    assert state.active_turn_id is None
    applied = next(r for r in records if r.record.kind == "input_applied" and r.record.payload["input"]["kind"] == "child_result")
    result = next(
        r
        for r in records
        if r.record.kind == "op_completed" and r.record.operation_id == handle["delivery_target"]["operation_id"]
    )
    assert applied.commit_id == result.commit_id
    assert result.record.payload["result"]["content"] == "child done"
    assert not rt.llm.steps


def delivery_worker(database, pipe, point):
    with open_store(database) as store, store.atomic() as tx:

        def hook(at):
            if at == point:
                pipe.send("barrier")
                pipe.recv()

        children.child_delivery(store, tx, "child", hook=hook)


@pytest.mark.parametrize("point", ["before_parent_inbox", "after_parent_inbox", "after_child_ack"])
@pytest.mark.persistent_store
def test_delivery_kill_replay(store, database, point):
    rt, _, _ = parked(store, database)
    complete_child(store, database)
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=delivery_worker, args=(database, child, point))
    process.start()
    child.close()
    try:
        assert parent.poll(15)
        assert parent.recv() == "barrier"
        process.kill()
        process.join(5)
        assert store.peek_inbox("s") == ()
        delivered(store)
        delivered(store)
        assert len(store.peek_inbox("s")) == 1
        drive(store, "s", runtime=rt)
        assert len([r for r in read_state(store, "s")[1] if r.record.kind == "turn_ended"]) == 1
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        parent.close()


def test_twenty_concurrent_completion_duplicates_and_conflict(store, database):
    rt, _, _ = parked(store, database)
    complete_child(store, database)
    delivered(store)
    item = child_result(store)

    def push(_):
        with open_store(database) as other, other.atomic() as tx:
            return tx.push("s", item).replayed

    with ThreadPoolExecutor(max_workers=20) as pool:
        assert all(pool.map(push, range(20)))
    with pytest.raises(Conflict), store.atomic() as tx:
        tx.push("s", replace(item, payload=item.payload | {"result": "forged"}))
    drive(store, "s", runtime=rt)
    with store.atomic() as tx:
        tx.push("s", replace(item, input_id="identical-other-id"))
    drive(store, "s", runtime=rt)
    state, _, _ = read_state(store, "s")
    assert state.applied_inputs["identical-other-id"].payload["disposition"] == "noop"


@pytest.mark.parametrize("child_started", [False, True])
def test_parent_cancel_cascades_and_late_result_audit(store, database, child_started):
    rt, _, handle = parked(store, database)
    if child_started:
        complete_child(store, database)
    target = handle["delivery_target"]
    with store.atomic() as tx:
        tx.push("s", InboxItem("cancel", "control", {"action": "cancel"}, target["turn_id"], target["generation"]))
    drive(store, "s", runtime=rt)
    if not child_started:
        assert any(i.item.payload == {"action": "cancel"} for i in store.peek_inbox("child"))
        drive(store, "child", runtime=runtime(database, []))
    delivered(store)
    drive(store, "s", runtime=rt)
    state, records, _ = read_state(store, "s")
    attempt = state.operations[target["operation_id"]].attempts[1]
    assert attempt.context == "audit"
    assert state.active_turn_id is None
    assert [r.record.payload["status"] for r in records if r.record.kind == "turn_ended"] == ["cancelled"]


def test_background_child_notification_at_safe_point(store, database):
    captured = []

    def next_model(request):
        captured.append(request)
        return LLMResponse("parent done")

    def hook(point, record):
        if point == "after_commit" and record and record.kind == "op_completed" and "/tool/" in (record.operation_id or ""):
            complete_child(store, database)
            delivered(store)

    rt, state, _ = parked(
        store,
        database,
        background=True,
        steps=[LLMResponse("", [ToolCall("child-call", "child_task", {})]), next_model],
        hook=hook,
    )
    assert state.active_turn_id is None
    messages = captured[0].messages
    assert len([m for m in messages if m.role == "tool"]) == 1
    assert any(m.role == "user" and "child done" in m.content for m in messages)
    assert not rt.llm.steps


@pytest.mark.parametrize(
    "field,value",
    [
        ("attempt", 2),
        ("operation_id", "unknown"),
        ("session_id", "wrong-child"),
        ("turn_id", "wrong-turn"),
        ("terminal_seq", 1),
        ("terminal_digest", "0" * 64),
        ("result", "forged"),
        ("generation", 99),
        ("target_turn_id", "wrong-parent-turn"),
    ],
)
def test_child_completion_rejects_untrusted_or_mismatched_identity(store, database, field, value):
    rt, _, handle = parked(store, database)
    complete_child(store, database)
    terminal = next(r for r in read_state(store, "child")[1] if r.record.kind == "turn_ended")
    item = children.completion_input(handle, terminal)
    item = (
        replace(item, **{field: value})
        if field in {"generation", "target_turn_id"}
        else replace(item, payload=item.payload | {field: value})
    )
    with store.atomic() as tx:
        tx.push("s", item)
    drive(store, "s", runtime=rt)
    state, _, _ = read_state(store, "s")
    assert state.phase == "parked"
    assert state.applied_inputs[item.input_id].payload["disposition"] == "rejected"


def test_store_rejects_success_with_waiting_live_child(store, database):
    from vv_agent.session.reducer import TransitionError

    from .helpers import record

    _, state, handle = parked(store, database)
    lease = store.acquire("s", owner="bad-terminal", ttl_ms=15000)
    assert lease is not None
    terminal = record("turn_ended", tid=state.active_turn_id, unconfirmed_operations=[handle["delivery_target"]["operation_id"]])
    try:
        with pytest.raises(TransitionError, match="HasLiveDescendants"), store.atomic() as tx:
            tx.append("s", lease=lease, expected_seq=store.read("s").head_seq, commit_id="bad-terminal", records=(terminal,))
    finally:
        store.release(lease)


def test_child_creation_rolls_back_with_parent_park(store, database):
    start(store)

    def hook(point, record):
        if point == "before_commit" and record and record.kind == "op_parked":
            raise RuntimeError("crash before parent park")

    rt = parent_runtime(database, hook=hook)
    with pytest.raises(RuntimeError, match="crash before parent park"):
        drive(store, "s", runtime=rt)
    assert store.list_sessions() == ("s",)
    drive(store, "s", runtime=parent_runtime(database, steps=[]))
    assert store.list_sessions() == ("child", "s")
    assert read_state(store, "s")[0].phase == "parked"


def test_background_notification_waits_for_frozen_model_then_forces_next_model(store, database):
    captured = []

    def frozen(request):
        captured.append(request)
        with open_store(database) as other:
            complete_child(other, database)
            delivered(other)
        return LLMResponse("candidate before child completion")

    def after(request):
        captured.append(request)
        return LLMResponse("done after notification")

    _, state, _ = parked(
        store,
        database,
        background=True,
        steps=[
            LLMResponse("", [ToolCall("child-call", "child_task", {})]),
            frozen,
            after,
        ],
    )
    assert state.active_turn_id is None
    assert len(captured) == 2
    assert not any(m.role == "user" and "child done" in m.content for m in captured[0].messages)
    assert sum(m.role == "user" and "child done" in m.content for m in captured[1].messages) == 1


def test_late_background_notification_does_not_open_turn(store, database):
    rt, _, _ = parked(store, database, background=True)
    complete_child(store, database)
    delivered(store)
    drive(store, "s", runtime=rt)
    state, records, _ = read_state(store, "s")
    assert state.active_turn_id is None
    assert len([r for r in records if r.record.kind == "turn_ended"]) == 1
    assert (
        next(r.payload for r in state.applied_inputs.values() if r.payload["input"]["kind"] == "child_result")["reason"]
        == "late background completion: audit only"
    )


def test_child_completion_older_generation_cannot_overwrite_successor(store, database):
    rt, _, handle = parked(store, database)
    target = handle["delivery_target"]
    with store.atomic() as tx:
        tx.push("s", InboxItem("cancel", "control", {"action": "cancel"}, target["turn_id"], target["generation"]))
    drive(store, "s", runtime=rt)
    complete_child(store, database)  # The durable cancel wins before the child's first model.
    with store.atomic() as tx:
        tx.push("s", InboxItem("next", "follow_up", {"content": "new generation"}, generation=2))
    drive(store, "s", runtime=runtime(database, [LLMResponse("new result")]))
    delivered(store)
    drive(store, "s", runtime=rt)
    state, records, _ = read_state(store, "s")
    assert state.operations[target["operation_id"]].attempts[1].context == "audit"
    assert [r.record.payload["result"] for r in records if r.record.kind == "turn_ended"] == [None, "new result"]


def test_twenty_concurrent_child_consumers_deliver_once(store, database):
    rt, _, _ = parked(store, database)
    complete_child(store, database)

    def deliver(_):
        with open_store(database) as other:
            return delivered(other)

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(deliver, range(20)))
    assert sum(bool(value) for value in results) == 1
    assert len(store.peek_inbox("s")) == 1
    drive(store, "s", runtime=rt)
    assert read_state(store, "s")[0].active_turn_id is None


def test_store_cannot_forge_child_completion_without_terminal_input(store, database):
    from vv_agent.session.reducer import TransitionError

    from .helpers import record

    _, state, handle = parked(store, database)
    oid = handle["delivery_target"]["operation_id"]
    plan = state.operations[oid].attempts[1].plan
    result = record(
        "op_completed",
        tid=plan.turn_id,
        oid=oid,
        provider_binding=plan.payload["provider_binding"],
        request_digest=plan.payload["request_digest"],
    )
    lease = store.acquire("s", owner="forged-completion", ttl_ms=15000)
    assert lease is not None
    try:
        with pytest.raises(TransitionError, match="child completion needs terminal input"), store.atomic() as tx:
            tx.append("s", lease=lease, expected_seq=store.read("s").head_seq, commit_id="forged", records=(result,))
    finally:
        store.release(lease)


def test_cancel_live_child_recursively_reaches_grandchild(store, database):
    parent_rt, _, handle = parked(store, database)
    child_rt = parent_runtime(database)
    child_rt.children = {
        "child_task": lambda plan: children.ChildSession(SessionSpec("grandchild", "test", "/tmp"), "nested child")
    }
    drive(store, "child", runtime=child_rt)
    assert read_state(store, "child")[0].phase == "parked"
    target = handle["delivery_target"]
    with store.atomic() as tx:
        tx.push("s", InboxItem("cancel", "control", {"action": "cancel"}, target["turn_id"], target["generation"]))
    drive(store, "s", runtime=parent_rt)
    drive(store, "child", runtime=child_rt)
    assert any(i.item.payload == {"action": "cancel"} for i in store.peek_inbox("grandchild"))
    drive(store, "grandchild", runtime=runtime(database, []))
    for sid in ("s", "child", "grandchild"):
        assert [r.record.payload["status"] for r in read_state(store, sid)[1] if r.record.kind == "turn_ended"] == ["cancelled"]


def test_child_input_and_parent_result_rollback_together(store, database):
    rt, _, _ = parked(store, database)
    complete_child(store, database)
    delivered(store)
    item = child_result(store)

    def hook(point, record):
        if point == "before_commit" and record and record.kind == "op_completed":
            raise RuntimeError("parent result transaction lost")

    rt.hook = hook
    with pytest.raises(RuntimeError, match="parent result transaction lost"):
        drive(store, "s", runtime=rt)
    assert item.input_id not in read_state(store, "s")[0].applied_inputs
    assert len(store.peek_inbox("s")) == 1
    rt.hook = lambda *_: None
    drive(store, "s", runtime=rt)
    assert read_state(store, "s")[0].active_turn_id is None


def test_handler_version_stop_with_child_wait_retains_unconfirmed_child(store, database):
    rt, _, handle = parked(store, database)
    rt.handler_version = "replaced"
    drive(store, "s", runtime=rt)
    terminal = next(r.record for r in read_state(store, "s")[1] if r.record.kind == "turn_ended")
    assert terminal.payload["status"] == "failed"
    assert terminal.payload["reason"] == "handler_version_mismatch"
    assert handle["delivery_target"]["operation_id"] in terminal.payload["unconfirmed_operations"]
    assert any(i.item.payload == {"action": "cancel"} for i in store.peek_inbox("child"))


def test_old_child_audit_during_new_model_does_not_force_another_model(store, database):
    rt, _, handle = parked(store, database)
    target = handle["delivery_target"]
    with store.atomic() as tx:
        tx.push("s", InboxItem("cancel", "control", {"action": "cancel"}, target["turn_id"], target["generation"]))
    drive(store, "s", runtime=rt)
    complete_child(store, database)
    with store.atomic() as tx:
        tx.push("s", InboxItem("next", "follow_up", {"content": "new generation"}, generation=2))

    def newer_model(request):
        with open_store(database) as other:
            delivered(other)
        return LLMResponse("new generation result")

    drive(store, "s", runtime=runtime(database, [newer_model]))
    state, records, _ = read_state(store, "s")
    assert state.operations[target["operation_id"]].attempts[1].context == "audit"
    assert [r.record.payload["result"] for r in records if r.record.kind == "turn_ended"] == [None, "new generation result"]
    assert sum(op.kind == "model" and op.turn_id == "s/turn/next" for op in state.operations.values()) == 1
