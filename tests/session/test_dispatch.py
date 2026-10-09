"""Host-controlled at-least-once delivery against both durable SQL stores."""

from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from threading import Event

import pytest

from vv_agent.session.children import child_delivery
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.store import LeaseLost
from vv_agent.session.supervisor import tick
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall

from .conftest import open_store
from .test_children import parent_runtime
from .test_recovery_matrix import runtime

pytestmark = pytest.mark.persistent_store


class Transport:
    def __init__(self, store, database):
        self.store, self.database = store, database
        self.queue = []
        self.wakes = []
        self.runtimes = {}
        self.projections = []

    def wake(self, sid):
        self.wakes.append(sid)
        self.queue.append(sid)

    def create(self, sid="s", steps=(), tools=(), consumers=("host",)):
        with self.store.atomic() as tx:
            tx.create(SessionSpec(sid, "test", "/tmp"), consumers=consumers)
        self.runtimes[sid] = runtime(self.database, steps, tools, wake=self.wake)

    def push(self, sid, item):
        with self.store.atomic() as tx:
            tx.push(sid, item)
        self.wake(sid)

    def deliver(self, index=0):
        sid = self.queue.pop(index)
        drive(self.store, sid, runtime=self.runtimes[sid])

    def drain(self):
        for _ in range(20):
            if not self.queue:
                return
            self.deliver()
        pytest.fail("transport did not reach quiescence")

    def project(self, sid, consumer):
        parent = None
        with self.store.atomic() as tx:
            batch = tx.consumer_batch(sid, consumer, limit=256 if consumer == "child_delivery" else 2)
            if batch is None:
                return
            self.projections.append((sid, consumer, batch.from_seq, batch.through_seq))
            if consumer == "child_delivery":
                child_delivery(self.store, tx, sid)
                if any(r.record.kind == "turn_ended" for r in batch.records):
                    parent = self.store.read(sid, limit=1).records[0].record.payload["parent_session_id"]
            else:
                tx.ack(batch)
        if parent:
            self.wake(parent)

    def tick(self, page_size=100):
        return tick(self.store, runtime=self.runtimes.__getitem__, project=self.project, page_size=page_size)


@pytest.fixture
def transport(store, database):
    return Transport(store, database)


@pytest.fixture
def clock(store, monkeypatch):
    now = 1_000_000
    monkeypatch.setattr(type(store), "clock_sql", str(now))

    def advance(ms):
        nonlocal now
        now += ms
        monkeypatch.setattr(type(store), "clock_sql", str(now))
        return now

    return advance


def records(store, sid="s", kind=None):
    return [r.record for r in read_state(store, sid)[1] if kind is None or r.record.kind == kind]


def prompt(input_id="initial", **kw):
    return InboxItem(input_id, "user", {"content": "go"}, **kw)


def test_idle_and_completed_turn_quiesce(transport):
    transport.create(steps=[LLMResponse("done")])
    transport.queue.append("s")
    transport.deliver()
    assert transport.wakes == []
    assert transport.queue == []
    transport.push("s", prompt())
    transport.drain()
    assert transport.wakes == ["s"]  # Only the input writer wakes.
    assert len(records(transport.store, kind="turn_ended")) == 1
    assert not transport.runtimes["s"].llm.steps


def test_duplicate_reordered_and_delayed_delivery(transport):
    calls = []

    @function_tool
    def effect() -> str:
        calls.append("effect")
        return "ok"

    transport.create(steps=[LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")], tools=[effect])
    transport.create("other", steps=[LLMResponse("other done")])
    transport.push("s", prompt())
    transport.push("other", prompt())
    transport.queue.extend(["s", "other", "s"])
    transport.deliver(1)  # Reorder; keep s delayed while the other session finishes.
    assert len(transport.runtimes["s"].llm.steps) == 2
    transport.drain()
    assert calls == ["effect"]
    assert len(records(transport.store, kind="op_started")) == 3
    assert len(records(transport.store, kind="turn_ended")) == 1
    ids = [r.record_id for r in records(transport.store)]
    assert len(ids) == len(set(ids))
    assert transport.wakes == ["s", "other"]


def test_concurrent_duplicate_cannot_dispatch_or_wake(transport):
    entered, finish = Event(), Event()
    calls = []

    def model(_request):
        calls.append("model")
        entered.set()
        assert finish.wait(10)
        return LLMResponse("done")

    transport.create(steps=[model])
    transport.push("s", prompt())
    with ThreadPoolExecutor(max_workers=1) as pool:
        holder = pool.submit(transport.deliver)
        try:
            assert entered.wait(10)
            with open_store(transport.database) as duplicate:
                drive(duplicate, "s", runtime=transport.runtimes["s"])
            assert transport.wakes == ["s"]
            assert calls == ["model"]
        finally:
            finish.set()
        holder.result(timeout=10)
    assert transport.queue == []
    assert len(records(transport.store, kind="op_started")) == 1
    assert len(records(transport.store, kind="turn_ended")) == 1


@pytest.mark.parametrize("window", ["before_release", "after_release"])
def test_input_racing_holder_exit_is_seen_after_release(transport, monkeypatch, window):
    transport.create(steps=[LLMResponse("first"), LLMResponse("second")])
    transport.push("s", prompt())
    store = transport.store
    release = store.release
    check = store._is_runnable
    raced = False

    def push():
        with open_store(transport.database) as pusher, pusher.atomic() as tx:
            tx.push("s", prompt("racing"))
        transport.wake("s")

    def exiting(lease):
        nonlocal raced
        if raced:
            return release(lease)
        raced = True
        if window == "before_release":
            push()
            transport.queue.pop()
            with open_store(transport.database) as duplicate:
                drive(duplicate, "s", runtime=transport.runtimes["s"])
            assert transport.wakes == ["s", "s"]
        released = release(lease)
        if window == "after_release":
            push()
            transport.queue.pop()  # The post-release check must independently see it.
        return released

    def fresh_check(sid):
        with pytest.raises(RuntimeError, match="transaction"):
            store._transaction_id()
        assert store._rows("SELECT lease_owner FROM sk_session WHERE session_id=%s", (sid,)) == [(None,)]
        return check(sid)

    monkeypatch.setattr(store, "release", exiting)
    monkeypatch.setattr(store, "_is_runnable", fresh_check)
    transport.deliver()
    assert transport.queue == ["s"]
    assert transport.wakes == ["s", "s", "s"]
    transport.drain()
    assert len(records(store, kind="turn_ended")) == 2


class WorkerKilled(BaseException):
    pass


@pytest.mark.parametrize("committed", [False, True])
def test_abandoned_worker_recovers_from_committed_log_after_ttl(transport, clock, monkeypatch, committed):
    calls = []

    def model(_request):
        calls.append("model")
        return LLMResponse("", [ToolCall("a", "effect", {})])

    @function_tool
    def effect() -> str:
        calls.append("tool")
        return "ok"

    transport.create(steps=[model, LLMResponse("done")], tools=[effect])
    transport.push("s", prompt())

    def kill(point, record):
        if (
            point == ("after_commit" if committed else "before_commit")
            and record
            and record.kind == "op_planned"
            and record.payload["op_kind"] == "tool"
        ):
            # Both windows have issued the provider call; only one has its durable receipt.
            raise WorkerKilled

    rt = transport.runtimes["s"]
    rt.hook = kill
    with monkeypatch.context() as patch:
        patch.setattr(transport.store, "release", lambda _lease: False)
        with pytest.raises(WorkerKilled):
            transport.deliver()
    assert transport.queue == []
    assert calls == ["model"]
    assert not transport.store._is_runnable("s")
    transport.tick()
    assert calls == ["model"]
    clock(rt.ttl_ms)  # Heartbeat has stopped; equality at expiry is runnable.
    rt.hook = lambda *_: None
    if not committed:
        rt.llm.steps.insert(0, model)
    transport.tick()
    assert calls == (["model", "tool"] if committed else ["model", "model", "tool"])
    unknowns = records(transport.store, kind="op_unknown")
    assert len(unknowns) == (0 if committed else 1)
    if unknowns:
        assert unknowns[0].payload["observation"]["code"] == "duplicate_model_request_and_cost"
    assert len(records(transport.store, kind="turn_ended")) == 1
    assert transport.queue == []


def test_dropped_wake_is_recovered_by_tick_alone(transport):
    transport.create(steps=[LLMResponse("done")])
    transport.push("s", prompt())
    transport.queue.clear()
    transport.tick(page_size=1)
    assert len(records(transport.store, kind="turn_ended")) == 1
    assert transport.queue == []
    while transport.tick(page_size=1):
        pass
    assert not transport.runtimes["s"].llm.steps


@pytest.mark.parametrize("wait_kind", ["approval", "user"])
def test_reply_wakes_only_waiting_session_and_releases_worker(transport, wait_kind):
    calls = []

    @function_tool(needs_approval=wait_kind == "approval")
    def effect() -> str:
        calls.append("tool")
        return "ok"

    first = ToolCall("a", "effect", {}) if wait_kind == "approval" else ToolCall("a", "ask_user", {"question": "Which?"})
    transport.create(steps=[LLMResponse("", [first]), LLMResponse("done")], tools=[effect])
    transport.push("s", prompt())
    transport.deliver()
    assert transport.queue == [] and calls == []
    park = next(r for r in records(transport.store, kind="op_parked") if r.payload["handle"]["kind"] == wait_kind)
    handle = park.payload["handle"]
    assert transport.store._rows("SELECT lease_owner FROM sk_session") == [(None,)]
    assert not transport.store._is_runnable("s")
    transport.create("other", steps=[LLMResponse("worker free")])
    transport.push("other", prompt())
    transport.deliver()
    before = list(transport.wakes)
    if wait_kind == "approval":
        answer = InboxItem(
            "reply",
            "approval_answer",
            {
                "operation_id": park.operation_id,
                "attempt": park.attempt,
                "request_id": handle["request_id"],
                "request_digest": handle["request_digest"],
                "scope": handle["scope"],
                "decision": "approve",
            },
            park.turn_id,
        )
    else:
        answer = InboxItem(
            "reply",
            "user",
            {"content": {"operation_id": park.operation_id, "interaction_id": handle["interaction_id"], "text": "blue"}},
            park.turn_id,
        )
    transport.push("s", answer)
    assert transport.wakes == [*before, "s"] and transport.queue == ["s"]
    transport.drain()
    assert calls == (["tool"] if wait_kind == "approval" else [])
    assert len(records(transport.store, kind="turn_ended")) == 1


def test_child_terminal_projection_wakes_only_parent(transport):
    transport.create()
    rt = parent_runtime(transport.database)
    rt.wake = transport.wake
    transport.runtimes["s"] = rt
    transport.push("s", prompt())
    transport.deliver()
    assert read_state(transport.store, "s")[0].phase == "parked"
    assert transport.queue == []
    assert transport.store._rows("SELECT lease_owner FROM sk_session WHERE session_id='s'") == [(None,)]
    transport.runtimes["child"] = runtime(transport.database, [LLMResponse("child done")], wake=transport.wake)
    transport.wake("child")
    transport.deliver()
    assert transport.queue == []
    before = list(transport.wakes)
    transport.tick()
    assert transport.wakes == [*before, "s"] and transport.queue == ["s"]
    transport.drain()
    assert len(records(transport.store, kind="turn_ended")) == 1
    assert transport.queue == []


def test_delayed_input_becomes_runnable_exactly_when_due(transport, clock):
    transport.create(steps=[LLMResponse("done")])
    transport.push("s", prompt(available_ms=1_000_100))
    transport.deliver()
    assert transport.queue == []
    assert not transport.store._is_runnable("s")
    clock(99)
    transport.tick()
    assert len(transport.runtimes["s"].llm.steps) == 1
    clock(1)
    assert transport.store._is_runnable("s")
    transport.tick()
    assert len(records(transport.store, kind="turn_ended")) == 1
    assert transport.wakes == ["s"]


def test_cancel_from_other_worker_is_observed_at_next_boundary(transport):
    entered, finish = Event(), Event()

    def model(_request):
        entered.set()
        assert finish.wait(10)
        return LLMResponse("discarded")

    transport.create(steps=[model])
    transport.push("s", prompt())
    with ThreadPoolExecutor(max_workers=1) as pool:
        holder = pool.submit(transport.deliver)
        try:
            assert entered.wait(10)
            with open_store(transport.database) as canceller, canceller.atomic() as tx:
                tx.push("s", InboxItem("cancel", "control", {"action": "cancel"}, "s/turn/initial"))
            transport.wake("s")
            transport.queue.pop()
            with open_store(transport.database) as duplicate:
                drive(duplicate, "s", runtime=transport.runtimes["s"])
        finally:
            finish.set()
        holder.result(timeout=10)
    assert [r.payload["status"] for r in records(transport.store, kind="turn_ended")] == ["cancelled"]
    assert transport.queue == []
    assert not transport.store.peek_inbox("s")


def test_tick_projection_pages_and_consumer_cursors_are_monotonic(transport):
    for sid in ("a", "b", "c"):
        transport.create(sid, [LLMResponse("done")], consumers=("billing", "events"))
    transport.tick(page_size=1)
    first = list(transport.projections)
    assert [(sid, consumer) for sid, consumer, _, _ in first] == [
        (sid, consumer) for sid in ("a", "b", "c") for consumer in ("billing", "events")
    ]
    for sid in ("a", "b", "c"):
        transport.push(sid, prompt())
    transport.queue.clear()
    while transport.tick(page_size=1):
        pass
    for sid in ("a", "b", "c"):
        for consumer in ("billing", "events"):
            batches = [(start, end) for s, c, start, end in transport.projections if (s, c) == (sid, consumer)]
            assert batches[0] == (1, 1)
            assert all(start == previous_end + 1 and end >= start for (_, previous_end), (start, end) in pairwise(batches))
            assert batches[-1][1] == transport.store.read(sid).head_seq


def test_post_release_check_and_scan_share_drive_predicate(transport, clock):
    transport.create()
    store = transport.store

    def agree(expected):
        assert store._is_runnable("s") is expected
        assert any(w.kind == "drive" for w in store.list_runnable()) is expected

    agree(False)  # Consumer lag alone must not wake execution.
    transport.push("s", prompt(available_ms=1_000_100))
    agree(False)
    clock(100)
    agree(True)
    lease = store.acquire("s", owner="held", ttl_ms=100)
    assert lease is not None
    agree(False)
    clock(100)
    agree(True)
    with pytest.raises(LeaseLost):
        store.renew(lease, ttl_ms=100)
    with store.atomic():
        store._rows("DELETE FROM sk_inbox WHERE session_id='s'")
        store._rows("UPDATE sk_session SET next_drive_ms=%s WHERE session_id='s'", (1_000_201,))
    agree(False)
    clock(1)
    agree(True)
