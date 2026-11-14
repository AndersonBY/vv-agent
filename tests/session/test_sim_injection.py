"""Exercise injection through the real kernel, SQLite SQL clock and projections."""

from dataclasses import dataclass
from threading import Event, Thread, current_thread
from types import SimpleNamespace

import pytest

from vv_agent import Agent, RunConfig
from vv_agent.budget import RunBudgetLimits
from vv_agent.llm.errors import ModelRequestRejected
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.session import kernel
from vv_agent.session.app_server import ThreadStatus, _KernelThreadStore, thread_status
from vv_agent.session.delegation import ChildTasks
from vv_agent.session.providers import Definitive, Unknown
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import LeaseLost
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.timing import SystemClock, ThreadInvoker, ThreadLeaseKeeper
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall

from .test_recovery_matrix import runtime, start


@dataclass
class ManualClock:
    ms: int = 100_000
    ns: int = 0

    def now_ms(self):
        return self.ms

    def monotonic(self):
        return self.ns / 1_000_000_000

    def monotonic_ns(self):
        return self.ns

    def sleep(self, seconds):
        self.advance(round(seconds * 1000))

    def advance(self, ms):
        self.ms += ms
        self.ns += ms * 1_000_000


class ManualKeeper:
    def __init__(self):
        self.started = []
        self.closed = []

    def start(self, heartbeat):
        self.started.append(heartbeat)
        with heartbeat.heartbeat_store() as store:
            heartbeat.poll(store)

    def close(self, heartbeat):
        self.closed.append(heartbeat)

    def is_alive(self, heartbeat):
        return False


class InlineInvoker:
    def __init__(self):
        self.calls = []

    def invoke(self, callback, **options):
        self.calls.append(options)
        return callback()


@pytest.mark.parametrize("file_store", [False, True])
def test_sql_clock_controls_records_inbox_and_lease_fencing(tmp_path, file_store):
    clock = ManualClock()
    path = str(tmp_path / "world.sqlite") if file_store else ":memory:"
    with SQLiteStore.standalone(path, clock=clock) as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("s", "test", "/tmp"), consumers=())
            tx.push("s", InboxItem("later", "user", {"content": "go"}, available_ms=clock.ms + 100))
        assert store.read("s").records[0].created_ms == clock.ms
        assert not store.peek_inbox("s")
        lease = store.acquire("s", owner="first", ttl_ms=100)
        assert lease is not None
        assert lease.expires_at_ms == clock.ms + 100
        with store.heartbeat_store() as other:
            assert other.clock is clock
            assert (other is store) is not file_store
            clock.advance(99)
            renewed = other.renew(lease, ttl_ms=100)
            assert renewed.db_now_ms == clock.ms
            assert renewed.lease.expires_at_ms == clock.ms + 100
            assert not other.peek_inbox("s")
            clock.advance(1)
            assert [entry.item.input_id for entry in other.peek_inbox("s")] == ["later"]
            assert other.peek_inbox("s")[0].received_ms == 100_000
            clock.advance(100)
            replacement = other.acquire("s", owner="second", ttl_ms=100)
            assert replacement is not None
            assert replacement.epoch == lease.epoch + 1
            with pytest.raises(LeaseLost):
                store.renew(renewed.lease, ttl_ms=100)


@pytest.mark.parametrize("factory", [False, True])
def test_real_drive_uses_scripted_seams_without_starting_threads(tmp_path, monkeypatch, factory):
    clock, keeper, invoker = ManualClock(), ManualKeeper(), InlineInvoker()
    effects = []

    @function_tool
    def effect() -> str:
        effects.append(current_thread())
        clock.advance(7)
        with keeper.started[-1].heartbeat_store() as other:
            keeper.started[-1].poll(other)
        return "ok"

    def no_thread(_self):
        pytest.fail("injected drive started a thread")

    monkeypatch.setattr(Thread, "start", no_thread)
    with SQLiteStore.standalone(str(tmp_path / "world.sqlite"), clock=clock) as store:
        store.install_schema()
        start(store)
        rt = runtime(
            store,
            [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")],
            [effect],
            clock=clock,
            lease_keeper=keeper,
            invoker=invoker,
        )
        built = []

        def build(sid):
            built.append(sid)
            assert keeper.started[0].successful_ready
            return rt

        if factory:
            assert kernel._drive(store, "s", runtime=build, clock=clock, lease_keeper=keeper)
        else:
            assert kernel.drive(store, "s", runtime=rt)
        state, records, _ = store.read_state("s")
        assert state.active_turn_id is None
        assert records[-1].record.kind == "turn_ended"
        assert records[-1].record.payload["status"] == "completed"
        assert [r.record.payload["result"]["content"] for r in records if r.record.kind == "op_completed"][-1] == "done"
        assert effects == [current_thread()]
        assert len(invoker.calls) == 3
        assert all(call["clock"] is clock for call in invoker.calls)
        assert all(call["token"] is keeper.started[-1].token for call in invoker.calls)
        assert all(call["cancellation_grace"] == rt.cancellation_grace for call in invoker.calls)
        assert [call["timeout"] for call in invoker.calls] == [rt.model_timeout, rt.tool_timeout, rt.model_timeout]
        assert len(keeper.started) == 1 + factory
        assert all(h in keeper.closed and h.stop.is_set() and not h.thread.is_alive() for h in keeper.started)
        assert built == (["s"] if factory else [])
        assert {r.created_ms for r in records} == {100_000, 100_007}
        active_ms = [
            r.record.payload["usage"]["elapsed_ms"]
            for r in records
            if r.record.kind == "usage_observed" and r.record.payload["source"] == "kernel-active-scope"
        ]
        assert max(active_ms) == 7
        assert all(op.state == "completed" for op in state.operations.values())


def test_budget_clock_stays_bound_after_runtime_clock_replacement(tmp_path):
    clock = ManualClock(ns=1_000_000_000)
    replacement = ManualClock(ms=clock.ms, ns=clock.ns + 100_000_000_000)

    def answer(_request):
        clock.advance(11)
        return LLMResponse("done")

    with SQLiteStore.standalone(str(tmp_path / "world.sqlite"), clock=clock) as store:
        store.install_schema()
        start(store)
        rt = runtime(store, [answer], clock=clock, lease_keeper=ManualKeeper(), invoker=InlineInvoker())
        rt.config.budget_limits = RunBudgetLimits(max_wall_time_ms=10)
        lease = store.acquire("s", owner="budget-clock", ttl_ms=1000)
        assert lease is not None
        scope = kernel._Scope(lease, rt, now_ms=store._now())
        driver = kernel._Driver(store, "s", rt, scope)
        driver.refresh()

        rt.clock = replacement
        for _ in range(20):
            if not driver.step():
                break

        boundary = next(
            record.record
            for record in driver.records
            if record.record.kind == "boundary_recorded" and record.record.payload["boundary_id"].endswith("/complete/1")
        )
        assert boundary.payload["data"]["usage"]["elapsed_ms"] == 11
        assert boundary.payload["data"]["exhaustion"] == {
            "dimension": "wall_time",
            "tool_name": None,
            "reason": "limit_exceeded",
            "limit": 10,
            "observed": 11,
            "attempted_increment": None,
            "overshoot": 1,
            "unit": "milliseconds",
            "currency": None,
            "enforcement_boundary": "model_call_complete",
            "unavailable_reason": None,
        }


def test_child_runtime_inherits_seams():
    clock, keeper, invoker = ManualClock(), ManualKeeper(), InlineInvoker()
    parent = runtime(None, [], clock=clock, lease_keeper=keeper, invoker=invoker)
    child = parent.for_agent(Agent("child", "Answer."), RunConfig())
    assert (child.clock, child.lease_keeper, child.invoker) == (clock, keeper, invoker)
    assert child.lease_retry_sleep is parent.lease_retry_sleep
    assert child.lease_retry_jitter is parent.lease_retry_jitter


def test_heartbeat_deadline_uses_injected_monotonic_time(tmp_path):
    clock = ManualClock()
    with SQLiteStore.standalone(str(tmp_path / "world.sqlite"), clock=clock) as store:
        store.install_schema()
        start(store)
        lease = store.acquire("s", owner="first", ttl_ms=100)
        assert lease is not None
        heartbeat = kernel._LeaseHeartbeat(lease, store.heartbeat_store, 100, 0.01, clock=clock)
        with heartbeat.committing() as captured:
            assert captured is lease
        clock.advance(100)
        with pytest.raises(LeaseLost, match="expired"), heartbeat.committing():
            pytest.fail("expired lease entered commit")
        assert heartbeat.lost


def test_wait_for_lease_uses_injected_sleep(tmp_path):
    clock, keeper, invoker = ManualClock(), ManualKeeper(), InlineInvoker()
    with SQLiteStore.standalone(str(tmp_path / "world.sqlite"), clock=clock) as store:
        store.install_schema()
        start(store)
        assert store.acquire("s", owner="other", ttl_ms=120) is not None
        rt = runtime(store, [LLMResponse("done")], clock=clock, lease_keeper=keeper, invoker=invoker)
        assert kernel.drive(store, "s", runtime=rt, _wait_for_lease=True)
        assert clock.ms == 100_150


def test_child_wait_uses_runtime_clock(monkeypatch):
    clock = ManualClock()
    with SQLiteStore.standalone(":memory:", clock=clock) as store:
        store.install_schema()
        tasks = ChildTasks(store, "parent", runtime(store, [], clock=clock))
        running = SimpleNamespace(is_running=lambda: True)
        monkeypatch.setattr(tasks, "get", lambda _sid: running)
        assert tasks.wait("child", timeout=0.025) is running
        assert clock.monotonic() == 0.03


def test_app_server_status_uses_thread_store_clock(tmp_path):
    clock, keeper, invoker = ManualClock(), ManualKeeper(), InlineInvoker()
    with SQLiteStore.standalone(str(tmp_path / "world.sqlite"), clock=clock) as store:
        store.install_schema()
        driver = SessionDriver(store=store)
        threads = _KernelThreadStore(driver, clock=clock)
        threads.create_thread_receipt(input_id="s", agent_key="default")
        with store.atomic() as tx:
            tx.push("s", InboxItem("initial", "user", {"content": "go"}))

        def rejected(_request):
            raise ModelRequestRejected(429, "retry later")

        rt = runtime(store, [rejected], clock=clock, lease_keeper=keeper, invoker=invoker)

        class Cut(BaseException):
            pass

        def stop_after_retry_plan(point, record):
            if point == "after_commit" and record and record.kind == "op_planned" and record.attempt == 2:
                raise Cut()

        rt.hook = stop_after_retry_plan
        with pytest.raises(Cut):
            kernel.drive(store, "s", runtime=rt)
        _, records, _ = store.read_state("s")
        plan = next(r.record for r in records if r.record.kind == "op_planned" and r.record.attempt == 2)
        assert plan.payload["not_before_ms"] == clock.ms + 2000
        snapshot = threads.read_thread("s")
        assert snapshot.thread.status == snapshot.state["status"] == ThreadStatus.RETRY_BACKOFF.value
        assert thread_status(driver, "s", clock=clock)["status"] == ThreadStatus.RETRY_BACKOFF.value
        clock.advance(2000)
        snapshot = threads.read_thread("s")
        assert snapshot.thread.status == snapshot.state["status"] == ThreadStatus.RUNNING.value
        assert thread_status(driver, "s", clock=clock)["status"] == ThreadStatus.RUNNING.value
        driver.close()


def test_thread_invoker_preserves_worker_and_exception_receipts():
    invoker, clock = ThreadInvoker(), SystemClock()
    observed = []
    outcome = Definitive({"ok": True})

    def callback():
        observed.append((current_thread().name, current_thread().daemon))
        return outcome

    assert invoker.invoke(callback, timeout=1, token=CancellationToken(), cancellation_grace=0.25, clock=clock) is outcome
    assert observed == [("session-external-call", True)]

    def failing():
        raise ValueError("private provider detail")

    assert invoker.invoke(failing, timeout=1, token=CancellationToken(), cancellation_grace=0.25, clock=clock) == Unknown(
        "ValueError: external outcome unavailable"
    )


@pytest.mark.parametrize("cancelled", [False, True])
def test_thread_invoker_retains_timeout_and_unconfirmed_cancel(cancelled):
    release, stopped = Event(), Event()
    token = CancellationToken()
    if cancelled:
        token.cancel("stop")

    def pending():
        try:
            release.wait(2)
            return Definitive({"late": True})
        finally:
            stopped.set()

    try:
        result = ThreadInvoker().invoke(pending, timeout=0, token=token, cancellation_grace=0, clock=SystemClock())
        assert result == Unknown("cancellation without confirmed external stop" if cancelled else "external call timeout")
    finally:
        release.set()
        assert stopped.wait(1)


def test_runtime_defaults_keep_thread_implementations():
    rt = runtime(None, [])
    assert isinstance(rt.clock, SystemClock)
    assert isinstance(rt.lease_keeper, ThreadLeaseKeeper)
    assert isinstance(rt.invoker, ThreadInvoker)


def test_injected_seams_do_not_enter_runtime_definition_or_digest():
    production = runtime(None, [])
    injected = runtime(None, [], clock=ManualClock(), lease_keeper=ManualKeeper(), invoker=InlineInvoker())
    task = production.compile("go", "t")
    assert injected.definition(task) == production.definition(task)
    assert injected.definition_digest(task) == production.definition_digest(task)
