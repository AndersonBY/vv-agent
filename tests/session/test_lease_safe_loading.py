"""Lease liveness while factory preparation and immutable prefix folding are paused."""

import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import pytest

from vv_agent.session import kernel
from vv_agent.session.records import InboxItem
from vv_agent.session.reducer import Fold
from vv_agent.session.store import LeaseLost, SequenceConflict
from vv_agent.types import LLMResponse

from .conftest import open_store
from .helpers import record
from .test_recovery_matrix import runtime, start


class ManualStop:
    """Each heartbeat interval ends only when the test advances database time."""

    def __init__(self):
        self.stopped = threading.Event()
        self.waiting = Queue()
        self.ticks = Queue()

    def is_set(self):
        return self.stopped.is_set()

    def set(self):
        self.stopped.set()
        self.ticks.put(True)

    def wait(self, _timeout):
        self.waiting.put(True)
        return self.ticks.get(timeout=10)


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


@pytest.fixture
def factory_beats(monkeypatch):
    beats = ManualStop()
    init = kernel._LeaseHeartbeat.__init__
    threads = []

    def controlled(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if type(self) is kernel._LeaseHeartbeat:
            self.stop = beats
            threads.append(self.thread)

    monkeypatch.setattr(kernel._LeaseHeartbeat, "__init__", controlled)
    yield beats
    assert threads and all(not thread.is_alive() for thread in threads)


def slow_build(beats, clock):
    # Three renewals keep the acquired object valid past its original 15 s expiry.
    for _ in range(3):
        beats.waiting.get(timeout=10)
        clock(10_000)
        beats.ticks.put(False)
    beats.waiting.get(timeout=10)


def test_factory_outlives_starting_ttl_and_hands_off_runtime_ttl(store, database, clock, factory_beats, monkeypatch):
    start(store)
    rt = runtime(database, [LLMResponse("done")], ttl_ms=42000)
    ttls = []
    renew = store.renew

    def observe(lease, *, ttl_ms):
        ttls.append(ttl_ms)
        return renew(lease, ttl_ms=ttl_ms)

    monkeypatch.setattr(store, "renew", observe)

    def factory(sid):
        slow_build(factory_beats, clock)
        with open_store(database) as competitor:
            assert competitor.acquire(sid, owner="competitor", ttl_ms=15000) is None
        return rt

    assert kernel.drive(store, "s", runtime=factory)
    assert 42000 in ttls
    state, rows, _ = store.read_state("s")
    assert state.active_turn_id is None
    assert rows[-1].record.kind == "turn_ended"
    assert rows[-1].record.payload["status"] == "completed"
    assert not rt.llm.steps


@pytest.mark.parametrize("outcome", ["success", "exception", "not_ready"])
def test_fencing_during_build_never_uses_factory_result(store, database, clock, factory_beats, outcome):
    start(store)
    before = store.read("s")
    rt = runtime(database, [LLMResponse("must not run")])
    error = kernel.RuntimeNotReady(500) if outcome == "not_ready" else ValueError("factory failed")
    retained = []

    def factory(sid):
        slow_build(factory_beats, clock)
        # Advance past even the refreshed lease and give a different epoch authority.
        clock(15000)
        with open_store(database) as competitor:
            lease = competitor.acquire(sid, owner="new-worker", ttl_ms=60000)
            assert lease is not None
            competitor.defer_drive(lease, retry_after_ms=1234)
            retained.append(competitor._one("SELECT * FROM sk_session WHERE session_id=%s", (sid,)))
        factory_beats.ticks.put(False)
        # Wait for the heartbeat to observe fencing before returning or raising.
        build = next(t for t in threading.enumerate() if t.name == "session-heartbeat")
        build.join(timeout=10)
        assert not build.is_alive()
        if outcome != "success":
            raise error
        return rt

    if outcome == "not_ready":
        assert kernel.drive(store, "s", runtime=factory)
    else:
        with pytest.raises(LeaseLost if outcome == "success" else ValueError):
            kernel.drive(store, "s", runtime=factory)
    assert store._one("SELECT * FROM sk_session WHERE session_id=%s", ("s",)) == retained[0]
    assert store.read("s").records == before.records
    assert len(rt.llm.steps) == 1


@pytest.mark.parametrize("not_ready", [False, True])
def test_factory_failure_defers_and_releases_current_lease(store, database, clock, factory_beats, monkeypatch, not_ready):
    start(store)
    before = store.read("s")
    error = kernel.RuntimeNotReady(700) if not_ready else ValueError("factory failed")
    used = []
    defer, release = store.defer_drive, store.release

    def deferred(lease, *, retry_after_ms):
        used.append(lease)
        return defer(lease, retry_after_ms=retry_after_ms)

    def released(lease):
        used.append(lease)
        result = release(lease)
        assert result
        return result

    monkeypatch.setattr(store, "defer_drive", deferred)
    monkeypatch.setattr(store, "release", released)

    def factory(_sid):
        slow_build(factory_beats, clock)
        raise error

    if not_ready:
        assert kernel.drive(store, "s", runtime=factory)
    else:
        with pytest.raises(ValueError) as caught:
            kernel.drive(store, "s", runtime=factory)
        assert caught.value is error
    assert len(used) == 2 and used[0] == used[1]
    assert used[0].expires_at_ms == 1_045_000
    assert store.read("s").records == before.records
    assert store.peek_inbox("s")[0].consumed_seq is None
    assert not store.is_runnable("s")
    clock(700 if not_ready else 1000)
    assert store.acquire("s", owner="next", ttl_ms=15000) is not None


@pytest.mark.parametrize("phase", ["prefix", "fold", "snapshot"])
def test_slow_fold_allows_renew_and_returns_captured_prefix(store, database, clock, monkeypatch, phase):
    start(store)
    lease = store.acquire("s", owner="writer", ttl_ms=15000)
    assert lease is not None
    store._fold_cache = None
    with store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=1, commit_id="turn", records=(record("turn_started"),))
    entered, finish = threading.Event(), threading.Event()
    owner = type(store) if phase == "prefix" else Fold
    method = {"prefix": "_prefix", "fold": "extend", "snapshot": "snapshot"}[phase]
    original = getattr(owner, method)
    reader_thread = []

    def paused(self, *args, **kwargs):
        if threading.get_ident() in reader_thread:
            entered.set()
            assert finish.wait(10), "reader was not released"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(owner, method, paused)

    def read():
        reader_thread.append(threading.get_ident())
        with open_store(database) as reader:
            if phase == "snapshot":
                reader_thread.remove(threading.get_ident())
                reader.read_state("s")
                reader_thread.append(threading.get_ident())
            else:
                reader._fold_cache = None
            return reader.read_state("s")

    # Use the main store's cache only for writer integrity; the reader owns its connection.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(read)
        try:
            assert entered.wait(10)
            clock(10000)
            lease = store.renew(lease, ttl_ms=15000).lease
            # Consume the previously unconsumed input AFTER the captured head.
            incoming = store.read_input("s", "initial").item
            application = record("input_applied", input=incoming.to_dict(), input_digest=incoming.digest)
            with store.atomic() as tx:
                tx.append(
                    "s", lease=lease, expected_seq=2, commit_id="input", records=(application,), consume_input_ids=("initial",)
                )
                tx.push("s", InboxItem("later", "user", {"content": "later"}))
            clock(10000)  # the full read has now exceeded the original lease TTL
            lease = store.renew(lease, ttl_ms=15000).lease
        finally:
            finish.set()
        state, rows, watermark = future.result(timeout=10)
    assert [r.seq for r in rows] == [1, 2]
    assert watermark == 1
    assert state.active_turn_id == "t" and not state.applied_inputs
    assert store.read("s").head_seq == 3 and store.read("s").inbox_seq == 2


@pytest.mark.parametrize("operation", ["rebuild", "idle"])
def test_schedule_preparation_does_not_hold_session_row(store, database, clock, monkeypatch, operation):
    start(store)
    lease = store.acquire("s", owner="writer", ttl_ms=15000)
    assert lease is not None
    with store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=1, commit_id="turn", records=(record("turn_started"),))
    entered, finish = threading.Event(), threading.Event()
    extend = Fold.extend
    thread_ids = []

    def paused(self, *args, **kwargs):
        if threading.get_ident() in thread_ids:
            entered.set()
            assert finish.wait(10)
        return extend(self, *args, **kwargs)

    monkeypatch.setattr(Fold, "extend", paused)

    def prepare():
        thread_ids.append(threading.get_ident())
        with open_store(database) as reader:
            reader._fold_cache = None
            if operation == "rebuild":
                return reader.rebuild_schedule("s")
            return reader.defer_idle_drive(lease, poll_ms=1000)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(prepare)
        try:
            assert entered.wait(10)
            clock(10000)
            current = store.renew(lease, ttl_ms=15000).lease
            with store.atomic() as tx:
                tx.push("s", InboxItem("racing", "user", {"content": "later"}))
        finally:
            finish.set()
        if operation == "idle":
            with pytest.raises(SequenceConflict):
                future.result(timeout=10)
        else:
            assert future.result(timeout=10).active_turn_id == "t"
    assert store.renew(current, ttl_ms=15000).lease.epoch == lease.epoch


@pytest.mark.postgres
def test_append_prepares_cold_history_before_session_row_lock(store, database, clock, monkeypatch):
    start(store)
    lease = store.acquire("s", owner="writer", ttl_ms=15000)
    assert lease is not None
    entered, finish = threading.Event(), threading.Event()
    extend = Fold.extend
    thread_ids = []
    current = [lease]

    def paused(self, *args, **kwargs):
        if threading.get_ident() in thread_ids and not entered.is_set():
            entered.set()
            assert finish.wait(10)
        return extend(self, *args, **kwargs)

    monkeypatch.setattr(Fold, "extend", paused)

    def append():
        thread_ids.append(threading.get_ident())
        with open_store(database) as writer, writer.atomic() as tx:
            return tx.append("s", lease=current[0], expected_seq=1, commit_id="turn", records=(record("turn_started"),))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(append)
        try:
            assert entered.wait(10)
            clock(10000)
            current[0] = store.renew(current[0], ttl_ms=15000).lease
        finally:
            finish.set()
        assert future.result(timeout=10).head_seq == 2


@pytest.mark.parametrize("failure", ["open", "renew"])
def test_factory_heartbeat_failure_prevents_factory_and_execution(store, database, monkeypatch, failure):
    from contextlib import contextmanager

    start(store)
    before = store.read("s")
    called = []
    error = ConnectionError("test heartbeat unavailable")

    @contextmanager
    def heartbeat():
        if failure == "open":
            raise error
        with open_store(database) as other:

            def broken(*_args, **_kwargs):
                raise error

            monkeypatch.setattr(other, "renew", broken)
            yield other

    monkeypatch.setattr(store, "heartbeat_store", heartbeat)

    def unexpected(sid):
        called.append(sid)
        raise AssertionError("factory must not run without renewal")

    with pytest.raises(LeaseLost) as caught:
        kernel.drive(store, "s", runtime=unexpected)
    assert caught.value.__cause__ is error
    assert not called
    assert store.read("s").records == before.records
    assert not any(t.name == "session-heartbeat" for t in threading.enumerate())


@pytest.mark.parametrize("operation", ["submit", "compact"])
def test_app_server_admission_folds_before_locking_session(store, database, clock, monkeypatch, operation):
    from vv_agent.app_server import AppServer

    from .test_app_server_context import server_for

    server, _ = server_for(store, database, monkeypatch)
    server.kernel.close()
    lease = store.acquire("s", owner="writer", ttl_ms=15000)
    assert lease is not None
    entered, finish = threading.Event(), threading.Event()
    extend = Fold.extend
    thread_ids = []

    def paused(self, *args, **kwargs):
        if threading.get_ident() in thread_ids:
            entered.set()
            assert finish.wait(10)
        return extend(self, *args, **kwargs)

    monkeypatch.setattr(Fold, "extend", paused)

    def admit():
        with open_store(database) as reader:
            reader._fold_cache = None
            app = AppServer(store=reader)
            monkeypatch.setattr(app.run_adapter, "recover", lambda *a, **k: None)
            thread_ids.append(threading.get_ident())
            try:
                if operation == "compact":
                    return app.run_adapter.compact_thread(thread_id="s", action_id="compact")
                return app.run_adapter.submit_input(
                    connection_id="test",
                    thread_id="s",
                    input_id="prompt",
                    expected_turn_id=None,
                    kind="start",
                    input=[{"type": "text", "text": "go"}],
                )
            finally:
                app.kernel.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(admit)
        try:
            assert entered.wait(10)
            clock(10000)
            assert store.renew(lease, ttl_ms=15000).lease.epoch == lease.epoch
        finally:
            finish.set()
        result = future.result(timeout=10)
        assert (result[0] if operation == "submit" else result)["accepted"]


def test_factory_interrupt_joins_heartbeat_and_releases_current_lease(store, database, clock, factory_beats):
    start(store)

    class Interrupted(BaseException):
        pass

    def factory(_sid):
        slow_build(factory_beats, clock)
        raise Interrupted()

    with pytest.raises(Interrupted):
        kernel.drive(store, "s", runtime=factory)
    assert store.acquire("s", owner="after-interrupt", ttl_ms=15000) is not None
    assert store.peek_inbox("s")[0].consumed_seq is None


def test_compaction_admission_rechecks_inbox_watermark(store, database, monkeypatch):
    from vv_agent.session.app_server import ContextVersionMismatch

    from .test_app_server_context import server_for

    server, _ = server_for(store, database, monkeypatch)
    adapter = server.run_adapter
    read_context = adapter.read_context
    expected = read_context("s")["contextVersion"]
    raced = []

    def concurrent_input(sid):
        result = read_context(sid)
        if not raced:
            with open_store(database) as other, other.atomic() as tx:
                tx.push(sid, InboxItem("concurrent", "user", {"content": "new context"}))
            raced.append(True)
        return result

    monkeypatch.setattr(adapter, "read_context", concurrent_input)
    try:
        with pytest.raises(ContextVersionMismatch):
            adapter.compact_thread(thread_id="s", action_id="compact", expected_context_version=expected)
        assert [item.item.input_id for item in store.peek_inbox("s")] == ["concurrent"]
    finally:
        server.kernel.close()
