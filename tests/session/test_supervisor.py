"""Failure isolation and scheduled retry use real SQL clocks, leases and inboxes."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import monotonic

import pytest

from vv_agent.session.kernel import _Driver, drive, read_state
from vv_agent.session.records import InboxItem
from vv_agent.session.runtime import RuntimeNotReady
from vv_agent.session.store import LeaseLost
from vv_agent.session.supervisor import tick
from vv_agent.types import LLMResponse

from .conftest import open_store
from .test_recovery_matrix import runtime, start

CONTRACT = json.loads((Path(__file__).parents[1] / "fixtures/parity/session_supervision.json").read_text())


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


def project(store, sid, consumer):
    with store.atomic() as tx:
        batch = tx.consumer_batch(sid, consumer)
        if batch:
            tx.ack(batch)


def finished(store, sid):
    return any(r.record.kind == "turn_ended" for r in read_state(store, sid)[1])


@pytest.mark.parametrize("failure", ["factory", "drive", "project", "dispatch"])
@pytest.mark.parametrize("page_size", [1, 2, 100])
def test_scan_isolates_failures_across_pages_and_bounds_retries(store, database, monkeypatch, clock, failure, page_size):
    attempts = []
    runtimes = {}
    dispatched = []
    for sid in ("a", "b", "c", "d", "e"):
        start(store, sid)
        runtimes[sid] = runtime(database, [LLMResponse(sid)])
    step = _Driver.step

    def factory(sid):
        if failure == "factory" and sid in ("b", "d"):
            attempts.append(sid)
            raise RuntimeError("poison factory " + sid)
        return runtimes[sid]

    def failing_step(self):
        if failure == "drive" and self.sid in ("b", "d"):
            attempts.append(self.sid)
            raise RuntimeError("poison drive " + self.sid)
        return step(self)

    def projection(sid, consumer):
        if failure == "project" and sid in ("b", "d"):
            attempts.append(sid)
            raise RuntimeError("poison projection " + sid)
        project(store, sid, consumer)

    def dispatch(sid):
        if sid in ("b", "d"):
            attempts.append(sid)
            raise RuntimeError("poison dispatch " + sid)
        dispatched.append(sid)

    monkeypatch.setattr(_Driver, "step", failing_step)
    dispatch_callback = dispatch if failure == "dispatch" else None
    with pytest.raises(ExceptionGroup) as errors:
        tick(store, project=projection, page_size=page_size, dispatch=dispatch_callback, runtime=factory)
    assert len(errors.value.exceptions) == 2
    assert all("tick " in exc.__notes__[0] for exc in errors.value.exceptions)
    assert attempts == ["b", "d"]
    if failure == "dispatch":
        assert dispatched == ["a", "c", "e"]
    else:
        assert all(finished(store, sid) for sid in ("a", "c", "e"))
    for _ in range(3):
        tick(store, project=projection, page_size=page_size, dispatch=dispatch_callback, runtime=factory)
    assert attempts == ["b", "d"]
    clock(CONTRACT["failure_backoff_ms"] - 1)
    tick(store, project=projection, page_size=page_size, dispatch=dispatch_callback, runtime=factory)
    assert attempts == ["b", "d"]
    clock(1)
    with pytest.raises(ExceptionGroup):
        tick(store, project=projection, page_size=page_size, dispatch=dispatch_callback, runtime=factory)
    assert attempts == ["b", "d", "b", "d"]


def test_runtime_not_ready_preserves_input_and_existing_turn_then_succeeds(store, database, clock):
    start(store)
    called = []
    rt = runtime(database, [LLMResponse("done")])

    def not_ready(sid):
        called.append(sid)
        raise RuntimeNotReady(CONTRACT["retry_after_ms"])

    before = store.read("s")
    tick(store, runtime=not_ready, project=lambda sid, c: project(store, sid, c))
    assert store.read("s").records == before.records
    assert store.peek_inbox("s")[0].consumed_seq is None
    assert not store.is_runnable("s")
    drive(store, "s", runtime=not_ready)  # Duplicate wake cannot bypass backoff.
    assert called == ["s"]
    clock(CONTRACT["retry_after_ms"])
    assert store.is_runnable("s")

    # Now admit a real turn, pause it at the model plan, and defer its preparation.
    def stop(point, record):
        if point == "after_commit" and record and record.kind == "op_planned":
            raise RuntimeError("pause existing turn")

    rt.hook = stop
    with pytest.raises(RuntimeError, match="pause existing turn"):
        drive(store, "s", runtime=rt)
    tid = read_state(store, "s")[0].active_turn_id
    drive(store, "s", runtime=not_ready)
    assert read_state(store, "s")[0].active_turn_id == tid
    assert not finished(store, "s")
    clock(CONTRACT["retry_after_ms"])
    rt.hook = lambda *_: None
    tick(store, runtime=lambda _: rt, project=lambda sid, c: project(store, sid, c))
    assert finished(store, "s")
    assert not rt.llm.steps


@pytest.mark.parametrize("failure", ["factory", "not_ready", "drive"])
@pytest.mark.parametrize("takeover", [False, True])
def test_lost_lease_backoff_preserves_original_outcome_and_new_owner(store, database, clock, monkeypatch, failure, takeover):
    start(store)
    before = store.read("s")
    error = RuntimeNotReady(250) if failure == "not_ready" else ValueError(f"original {failure} error")
    rt = runtime(database, [])
    rows = []

    def fail(*_args):
        clock(15000)
        if takeover:
            current = store.acquire("s", owner="new", ttl_ms=30000)
            assert current is not None
            store.defer_drive(current, retry_after_ms=500)
        rows.append(store._one("SELECT * FROM sk_session WHERE session_id=%s", ("s",)))
        raise error

    def factory(_sid):
        if failure != "drive":
            fail()
        return rt

    if failure == "drive":
        monkeypatch.setattr(_Driver, "step", fail)
    if failure == "not_ready":
        drive(store, "s", runtime=factory)
    else:
        with pytest.raises(ValueError) as caught:
            drive(store, "s", runtime=factory)
        assert caught.value is error
    assert store._one("SELECT * FROM sk_session WHERE session_id=%s", ("s",)) == rows[0]
    assert store.read("s").records == before.records
    assert store.peek_inbox("s")[0].consumed_seq is None


@pytest.mark.parametrize("failure", ["factory", "not_ready", "renew", "drive"])
def test_drive_backoff_database_failure_is_chained_to_original(store, database, clock, monkeypatch, failure):
    start(store)
    error = RuntimeNotReady(250) if failure == "not_ready" else ValueError(f"original {failure} error")
    rt = runtime(database, [])
    backoff_errors = []
    query_rows = store._rows

    def broken_write(query, params=()):
        if query.startswith("UPDATE sk_session SET drive_retry_at_ms"):
            try:
                return query_rows("SELECT * FROM missing_backoff_table")
            except Exception as exc:
                backoff_errors.append(exc)
                raise
        return query_rows(query, params)

    def fail(*_args, **_kwargs):
        raise error

    def factory(_sid):
        if failure in {"factory", "not_ready"}:
            fail()
        return rt

    monkeypatch.setattr(store, "_rows", broken_write)
    if failure == "drive":
        monkeypatch.setattr(_Driver, "step", fail)
    elif failure == "renew":
        monkeypatch.setattr(store, "renew", fail)
    with pytest.raises(type(error)) as caught:
        drive(store, "s", runtime=factory)
    assert caught.value is error
    assert len(backoff_errors) == 1 and error.__cause__ is backoff_errors[0]
    assert "missing_backoff_table" in str(error.__cause__)
    assert store.peek_inbox("s")[0].consumed_seq is None


def test_dispatch_backoff_lease_loss_keeps_only_original_failure(store, clock, monkeypatch):
    start(store)
    error = ValueError("original dispatch error")
    defer = store.defer_drive
    rows = []

    def taken_over(lease, *, retry_after_ms):
        clock(15000)
        current = store.acquire("s", owner="new", ttl_ms=30000)
        assert current is not None
        defer(current, retry_after_ms=500)
        rows.append(store._one("SELECT * FROM sk_session WHERE session_id=%s", ("s",)))
        defer(lease, retry_after_ms=retry_after_ms)

    def dispatch(_sid):
        raise error

    monkeypatch.setattr(store, "defer_drive", taken_over)
    with pytest.raises(ExceptionGroup) as caught:
        tick(store, dispatch=dispatch, project=lambda sid, c: project(store, sid, c))
    assert caught.value.exceptions == (error,)
    assert store._one("SELECT * FROM sk_session WHERE session_id=%s", ("s",)) == rows[0]


@pytest.mark.parametrize("failure", ["dispatch", "project"])
def test_tick_backoff_database_failure_is_reported_with_original(store, clock, monkeypatch, failure):
    start(store)
    error = ValueError(f"original {failure} error")

    def fail(*_args):
        raise error

    def broken_write(*_args, **_kwargs):
        store._rows("SELECT * FROM missing_backoff_table")

    monkeypatch.setattr(store, "defer_drive" if failure == "dispatch" else "defer_projection", broken_write)
    with pytest.raises(ExceptionGroup) as caught:
        tick(
            store,
            dispatch=fail if failure == "dispatch" else lambda _sid: None,
            project=fail if failure == "project" else lambda sid, c: project(store, sid, c),
        )
    assert len(caught.value.exceptions) == 2
    assert caught.value.exceptions[0] is error
    assert "missing_backoff_table" in str(caught.value.exceptions[1])
    assert "tick backoff:" in caught.value.exceptions[1].__notes__[0]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_not_ready_validates_retry_delay(value):
    with pytest.raises(ValueError):
        RuntimeNotReady(value)


def test_backoff_fences_expired_and_superseded_lease_preserves_schedule(store, clock):
    start(store)
    old = store.acquire("s", owner="old", ttl_ms=100)
    clock(100)
    with pytest.raises(LeaseLost):
        store.defer_drive(old, retry_after_ms=250)
    current = store.acquire("s", owner="new", ttl_ms=1000)
    with pytest.raises(LeaseLost):
        store.defer_drive(old, retry_after_ms=250)
    store._rows("UPDATE sk_session SET next_drive_ms=%s WHERE session_id=%s", (1_009_000, "s"))
    store.defer_drive(current, retry_after_ms=500)
    store.defer_drive(current, retry_after_ms=100)  # Never shorten a concurrent later retry.
    assert store._one("SELECT next_drive_ms,drive_retry_at_ms FROM sk_session WHERE session_id=%s", ("s",)) == (
        1_009_000,
        1_000_600,
    )
    store.rebuild_schedule("s")  # Schedule reconstruction must also preserve the retry gate.
    store.release(current)
    assert not store.is_runnable("s")
    assert all(w.kind != "drive" for w in store.list_runnable())
    with store.atomic() as tx:
        tx.push("s", InboxItem("during-backoff", "steer", {"content": "wait"}))
    assert store.acquire("s", owner="duplicate", ttl_ms=1000) is None
    clock(499)
    assert not store.is_runnable("s")
    clock(1)
    assert store.is_runnable("s")


def test_projection_backoff_is_independent_of_drives_and_other_consumers(store, clock):
    start(store)
    with store._transaction():
        store._rows("INSERT INTO sk_consumer(session_id,consumer) VALUES ('s','other')")
    store.defer_projection("s", "events", retry_after_ms=500)
    work = store.list_runnable()
    assert [(w.kind, w.consumer) for w in work] == [("drive", ""), ("project", "other")]
    assert store.is_runnable("s")
    clock(500)
    assert ("project", "events") in [(w.kind, w.consumer) for w in store.list_runnable()]


@pytest.mark.persistent_store
def test_dispatch_scan_returns_during_long_drive_without_factory_or_inline_drive(store, database):
    entered, finish = Event(), Event()

    def slow(_request):
        entered.set()
        assert finish.wait(15)
        return LLMResponse("done")

    start(store, "a")
    start(store, "b")
    rt = runtime(database, [slow])

    def worker():
        with open_store(database) as other:
            drive(other, "a", runtime=lambda _: rt)

    def forbidden(_sid):
        raise AssertionError("scan must not build runtimes")

    queued = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(worker)
        try:
            assert entered.wait(10)
            begin = monotonic()
            tick(store, dispatch=queued.append, runtime=forbidden, project=lambda sid, c: project(store, sid, c), page_size=1)
            assert monotonic() - begin < 1
            assert queued == ["b"]
            assert not finished(store, "a") and not finished(store, "b")
        finally:
            finish.set()
        active.result(timeout=10)


def test_failing_factory_is_not_called_for_duplicate_wakes_until_backoff_expires(store, clock):
    start(store)
    calls = []

    def poison(sid):
        calls.append(sid)
        raise RuntimeError("factory")

    with pytest.raises(RuntimeError):
        drive(store, "s", runtime=poison)
    for _ in range(5):
        drive(store, "s", runtime=poison)
    assert calls == ["s"]
    clock(1000)
    with pytest.raises(RuntimeError):
        drive(store, "s", runtime=poison)
    assert calls == ["s", "s"]


def test_scheduled_driver_failure_defers_then_finishes_without_self_wake(store, database, clock, monkeypatch):
    start(store)
    wakes = []
    rt = runtime(database, [LLMResponse("done")], wake=wakes.append)
    step = _Driver.step
    failed = False

    def once(self):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("driver boundary failed")
        return step(self)

    monkeypatch.setattr(_Driver, "step", once)
    with pytest.raises(RuntimeError, match="driver boundary failed"):
        drive(store, "s", runtime=lambda _: rt)
    assert not store.is_runnable("s") and wakes == []
    assert store.peek_inbox("s")
    clock(1000)
    tick(store, runtime=lambda _: rt, project=lambda sid, c: project(store, sid, c))
    assert finished(store, "s") and wakes == []
    assert tick(store, dispatch=wakes.append, project=lambda sid, c: project(store, sid, c)) == 0


@pytest.mark.persistent_store
def test_backoff_waiting_for_concurrent_schedule_writer_preserves_its_schedule(store, database):
    start(store)
    lease = store.acquire("s", owner="holder", ttl_ms=15000)
    locked, release = Event(), Event()
    future = store._now() + 30_000

    def writer():
        with open_store(database) as other, other._transaction():
            other._lock("s")
            other._rows("UPDATE sk_session SET next_drive_ms=%s WHERE session_id=%s", (future, "s"))
            locked.set()
            assert release.wait(10)

    def defer():
        with open_store(database) as other:
            other.defer_drive(lease, retry_after_ms=1000)

    with ThreadPoolExecutor(max_workers=2) as pool:
        changing = pool.submit(writer)
        assert locked.wait(10)
        deferring = pool.submit(defer)
        release.set()
        changing.result(timeout=10)
        deferring.result(timeout=10)
    assert store._one("SELECT next_drive_ms FROM sk_session WHERE session_id=%s", ("s",))[0] == future
    store.release(lease)
    assert not store.is_runnable("s")


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_store_retry_primitives_validate_before_effects(store, value):
    start(store)
    lease = store.acquire("s", owner="holder", ttl_ms=1000)
    with pytest.raises(ValueError):
        store.defer_drive(lease, retry_after_ms=value)
    with pytest.raises(ValueError):
        store.defer_projection("s", "events", retry_after_ms=value)
    assert store._one("SELECT drive_retry_at_ms FROM sk_session WHERE session_id=%s", ("s",))[0] == 0
    assert store._one("SELECT project_retry_at_ms FROM sk_consumer WHERE session_id=%s", ("s",))[0] == 0
    store.release(lease)
