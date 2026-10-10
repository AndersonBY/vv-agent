"""Custom batch admission shares the delegated sibling transaction and delivery path."""

import json
import multiprocessing
from dataclasses import replace

import pytest

from vv_agent.session import children
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.types import LLMResponse

from .conftest import open_store
from .test_children import parent_runtime
from .test_recovery_matrix import runtime, start

CHILDREN = ("child-0", "child-1", "child-2")


def batch_runtime(database, *, background=False):
    rt = parent_runtime(database)
    rt.children["child_task"] = lambda _plan: tuple(
        children.ChildSession(SessionSpec(sid, "test", "/tmp"), "work", background=background) for sid in CHILDREN
    )
    return rt


def rows(store, sid="s", kind=None):
    return [r for r in read_state(store, sid)[1] if kind is None or r.record.kind == kind]


def deliver(store, sid, hook=lambda _: None):
    with store.atomic() as tx:
        children.child_delivery(store, tx, sid, hook=hook)


def complete(store, database, sid):
    drive(store, sid, runtime=runtime(database, [LLMResponse(sid + " done")]))


def test_batch_admission_waits_for_every_sibling_and_retains_order(store, database):
    start(store)
    rt = batch_runtime(database)
    drive(store, "s", runtime=rt)
    parked = rows(store, kind="op_parked")[0]
    handles = children.child_handles(parked.record.payload["handle"])
    assert [h["session_id"] for h in handles] == list(CHILDREN)
    assert parked.commit_id == rows(store, kind="op_started")[-1].commit_id
    for sid in reversed(CHILDREN):
        assert len(rows(store, sid, "session_created")) == 1
        assert len(store.peek_inbox(sid)) == 1
        complete(store, database, sid)
        deliver(store, sid)
        deliver(store, sid)
        drive(store, "s", runtime=rt)
        assert bool(rows(store, kind="turn_ended")) == (sid == CHILDREN[0])
    result = next(r.record.payload["result"] for r in rows(store, kind="op_completed") if "/tool/" in r.record.operation_id)
    outcomes = json.loads(result["content"])
    assert [o["content"] for o in outcomes] == [sid + " done" for sid in CHILDREN]
    assert result["status_code"] == "SUCCESS"
    for sid in CHILDREN:
        original = next(
            r.record
            for r in rows(store, kind="input_applied")
            if r.record.payload["input"]["kind"] == "child_result" and r.record.payload["input"]["payload"]["session_id"] == sid
        )
        item = InboxItem(**original.payload["input"])
        with store.atomic() as tx:
            tx.push("s", replace(item, input_id=item.input_id + "/duplicate"))
    drive(store, "s", runtime=rt)
    assert len(rows(store, kind="turn_ended")) == 1
    assert all(
        r.record.payload["disposition"] == "noop"
        for r in rows(store, kind="input_applied")
        if r.record.payload["input"]["input_id"].endswith("/duplicate")
    )


@pytest.mark.parametrize("fault", ["host", "create"])
def test_batch_kth_failure_rolls_back_host_rows_children_and_park(store, database, monkeypatch, fault):
    import vv_agent.session.kernel as kernel

    start(store)
    store._rows("CREATE TABLE host_child (id INTEGER PRIMARY KEY)")
    rt = batch_runtime(database)
    callback = rt.children["child_task"]

    def admit(plan):
        for index in range(3):
            store._rows("INSERT INTO host_child VALUES (%s)", (index,))
            if fault == "host" and index == 1:
                raise RuntimeError("host row 2 failed")
        return callback(plan)

    rt.children["child_task"] = admit
    create = kernel.create_child

    def create_child(tx, child, plan, generation):
        handle = create(tx, child, plan, generation)
        if child.spec.session_id == CHILDREN[1]:
            raise RuntimeError("create 2 failed")
        return handle

    if fault == "create":
        monkeypatch.setattr(kernel, "create_child", create_child)
    with pytest.raises(RuntimeError, match="2 failed"):
        drive(store, "s", runtime=rt)
    assert store.list_sessions() == ("s",)
    assert store._rows("SELECT * FROM host_child") == []
    assert rows(store, kind="op_parked") == []
    assert not any("/tool/" in r.record.operation_id for r in rows(store, kind="op_started"))
    rt.children["child_task"] = callback
    monkeypatch.setattr(kernel, "create_child", create)
    drive(store, "s", runtime=rt)
    assert len(store.list_sessions()) == 4


@pytest.mark.parametrize("invalid", ["empty", "mixed_background", "invalid_member"])
def test_invalid_batch_has_no_admission_effect(store, database, invalid):
    start(store)
    store._rows("CREATE TABLE host_child (id INTEGER PRIMARY KEY)")
    rt = batch_runtime(database)
    callback = rt.children["child_task"]

    def bad(plan):
        store._rows("INSERT INTO host_child VALUES (1)")
        batch = list(callback(plan))
        if invalid == "empty":
            return []
        if invalid == "mixed_background":
            batch[1] = replace(batch[1], background=True)
        else:
            batch[1] = object()
        return batch

    rt.children["child_task"] = bad
    with pytest.raises(children.InvalidChildBatch):
        drive(store, "s", runtime=rt)
    assert store.list_sessions() == ("s",)
    assert store._rows("SELECT * FROM host_child") == []
    assert rows(store, kind="op_parked") == []


def test_batch_host_rows_and_child_admission_roll_back_with_parent_append(store, database):
    start(store)
    store._rows("CREATE TABLE host_child (id INTEGER PRIMARY KEY)")
    rt = batch_runtime(database)
    callback = rt.children["child_task"]

    def admit(plan):
        store._rows("INSERT INTO host_child VALUES (1)")
        return callback(plan)

    def fail(point, record):
        if point == "before_commit" and record and record.kind == "op_parked":
            raise RuntimeError("parent append failed")

    rt.children["child_task"], rt.hook = admit, fail
    with pytest.raises(RuntimeError, match="parent append failed"):
        drive(store, "s", runtime=rt)
    assert store.list_sessions() == ("s",)
    assert store._rows("SELECT * FROM host_child") == []
    rt.hook = lambda *_: None
    drive(store, "s", runtime=rt)
    assert store._rows("SELECT * FROM host_child") == [(1,)]


@pytest.mark.parametrize("background", [False, True])
def test_cancel_reaches_every_sibling_and_late_delivery_never_revives(store, database, background):
    start(store)
    rt = batch_runtime(database, background=background)
    if background:
        rt.llm.steps.pop()  # Park parent on its next ask_user instead of completing it.
        from vv_agent.types import ToolCall

        rt.llm.steps.append(LLMResponse("", [ToolCall("wait", "ask_user", {"question": "Wait?"})]))
    drive(store, "s", runtime=rt)
    state = read_state(store, "s")[0]
    tid = state.active_turn_id
    with store.atomic() as tx:
        tx.push("s", InboxItem("cancel", "control", {"action": "cancel"}, tid, 0))
    drive(store, "s", runtime=rt)
    for sid in CHILDREN:
        assert any(i.item.payload == {"action": "cancel"} for i in store.peek_inbox(sid))
        drive(store, sid, runtime=runtime(database, []))
        deliver(store, sid)
    drive(store, "s", runtime=rt)
    assert [r.record.payload["status"] for r in rows(store, kind="turn_ended")] == ["cancelled"]
    assert read_state(store, "s")[0].active_turn_id is None


def test_background_batch_result_lists_every_admitted_child(store, database):
    start(store)
    rt = batch_runtime(database, background=True)
    drive(store, "s", runtime=rt)
    result = next(r.record.payload["result"] for r in rows(store, kind="op_completed") if "/tool/" in r.record.operation_id)
    assert [h["session_id"] for h in json.loads(result["content"])] == list(CHILDREN)
    assert len(result["metadata"]["children"]) == 3
    for sid in CHILDREN:
        complete(store, database, sid)
        deliver(store, sid)
    drive(store, "s", runtime=rt)
    assert len(rows(store, kind="turn_ended")) == 1


def delivery_worker(database, sid, pipe, point):
    with open_store(database) as store, store.atomic() as tx:

        def hook(at):
            if at == point:
                pipe.send("barrier")
                pipe.recv()

        children.child_delivery(store, tx, sid, hook=hook)


@pytest.mark.persistent_store
@pytest.mark.parametrize("point", ["before_parent_inbox", "after_parent_inbox", "after_child_ack"])
def test_batch_delivery_crash_replays_atomically(store, database, point):
    start(store)
    rt = batch_runtime(database)
    drive(store, "s", runtime=rt)
    for sid in CHILDREN:
        complete(store, database, sid)
    deliver(store, CHILDREN[0])
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=delivery_worker, args=(database, CHILDREN[1], child, point))
    process.start()
    child.close()
    try:
        assert parent.poll(15) and parent.recv() == "barrier"
        process.kill()
        process.join(5)
        assert len(store.peek_inbox("s")) == 1
        assert (
            store._one("SELECT last_seq FROM sk_consumer WHERE session_id=%s AND consumer='child_delivery'", (CHILDREN[1],))[0]
            == 0
        )
        deliver(store, CHILDREN[1])
        deliver(store, CHILDREN[1])
        drive(store, "s", runtime=rt)
        assert not rows(store, kind="turn_ended")
        deliver(store, CHILDREN[2])
        drive(store, "s", runtime=rt)
        assert len(rows(store, kind="turn_ended")) == 1
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        parent.close()


def test_background_batch_delivers_all_notifications_at_next_safe_model_boundary(store, database):
    start(store)
    rt = batch_runtime(database, background=True)
    seen = []

    def model(request):
        seen.extend(m.content for m in request.messages if m.role == "user")
        return LLMResponse("done")

    rt.llm.steps[-1] = model

    def hook(point, record):
        if point == "after_commit" and record and record.kind == "op_completed" and "/tool/" in record.operation_id:
            for sid in CHILDREN:
                complete(store, database, sid)
                deliver(store, sid)

    rt.hook = hook
    drive(store, "s", runtime=rt)
    assert all(any(sid + " done" in content for content in seen) for sid in CHILDREN)
    assert len(rows(store, kind="turn_ended")) == 1


def test_batch_error_result_keeps_every_child_outcome(store, database):
    start(store)
    rt = batch_runtime(database)
    drive(store, "s", runtime=rt)
    for index, sid in enumerate(CHILDREN):
        if index == 1:
            with store.atomic() as tx:
                tx.push(sid, InboxItem("cancel", "control", {"action": "cancel"}, f"{sid}/turn/start", 0))
            drive(store, sid, runtime=runtime(database, []))
        else:
            complete(store, database, sid)
        deliver(store, sid)
    drive(store, "s", runtime=rt)
    result = next(r.record.payload["result"] for r in rows(store, kind="op_completed") if "/tool/" in r.record.operation_id)
    assert result["status_code"] == "ERROR"
    outcomes = json.loads(result["content"])
    assert [o["status_code"] for o in outcomes] == ["SUCCESS", "ERROR", "SUCCESS"]
    assert outcomes[1]["error_code"] == "child_cancelled"
