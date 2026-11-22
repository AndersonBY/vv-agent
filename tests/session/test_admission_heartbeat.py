"""Real PostgreSQL transactions and elapsed-time leases during admission and shutdown."""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from time import monotonic, monotonic_ns, sleep

import pytest

from vv_agent.kernel._content_runtime import _ContentSessionTxAdapter
from vv_agent.kernel._postgres import PostgresStore as CorePostgresStore
from vv_agent.session import children, kernel, timing
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.sql import SQLSessionTx
from vv_agent.session.store import LeaseLost
from vv_agent.session.timing import SystemClock
from vv_agent.types import LLMResponse

from .conftest import open_store
from .helpers import fixture_child_spec, fixture_content_reader, read_fixture_state, runtime_turn_record, wrap_fixture_store
from .test_children import parent_runtime
from .test_recovery_matrix import runtime as base_runtime
from .test_recovery_matrix import start
from .test_sim_injection import ManualClock

pytestmark = pytest.mark.postgres
TTL = 400
INTERVAL = 0.025
CHILDREN = tuple(f"child-{i}" for i in range(4))


@pytest.fixture(autouse=True)
def elapsed_lease_clock(monkeypatch):
    # Keep real elapsed TTLs and SQL fencing independent of adjustable wall time.
    from vv_agent.kernel._postgres import PostgresStore as CorePostgresStore

    def elapsed_sql(self):
        del self
        return str(monotonic_ns() // 1_000_000)

    def keep_elapsed_sql(self, value):
        # Heartbeat factories capture and replay this dynamic expression on each
        # independent connection. The expression stays dynamic after replay.
        del self, value

    clock_sql = property(elapsed_sql, keep_elapsed_sql)
    monkeypatch.setattr(PostgresStore, "clock_sql", clock_sql)
    # The S1 adapter heartbeat factory returns the core implementation; both
    # acquisition and renewal stores must share the same elapsed-time domain.
    monkeypatch.setattr(CorePostgresStore, "clock_sql", clock_sql)


@pytest.fixture
def store(store):
    return wrap_fixture_store(store)


def runtime(store, database, steps, tools=(), *, sid="s", **kwargs):
    kwargs.setdefault("content_reader", fixture_content_reader(store, sid))
    return base_runtime(database, steps, tools, **kwargs)


def rows(store, sid="s", kind=None):
    return [r for r in read_fixture_state(store, sid)[1] if kind is None or r.record.kind == kind]


def raw_rows(store, sid="s", kind=None):
    return [r for r in store.read(sid).records if kind is None or r.record.kind == kind]


def bind_lease_guard(tx, lease):
    target = getattr(tx, "_tx", tx)
    target._lease_guard = lambda: lease


@pytest.fixture(scope="module")
def local_counting_ready():
    # Admission timing starts after the real local counting dependencies are ready.
    from vv_agent.memory.token_utils import count_messages_tokens

    count_messages_tokens([{"role": "user", "content": "admission timing readiness"}], model="")


def wait_until(callback, *, timeout=5, future=None):
    deadline = monotonic() + timeout
    while not callback():
        if future is not None and future.done():
            pytest.fail(f"background worker exited before condition: {future.result()!r}")
        assert monotonic() < deadline, "database condition did not become true"
        threading.Event().wait(0.01)


@pytest.mark.parametrize("failed", [False, True])
def test_wait_until_reports_worker_exit(failed):
    future = Future()
    error = RuntimeError("worker failed before admission")
    if failed:
        future.set_exception(error)
        with pytest.raises(RuntimeError) as caught:
            wait_until(lambda: False, future=future)
        assert caught.value is error
    else:
        future.set_result(False)
        with pytest.raises(pytest.fail.Exception, match=r"background worker exited.*False"):
            wait_until(lambda: False, future=future)


def lease_row(store):
    return store._one(
        "SELECT lease_owner,lease_epoch,lease_until_ms,drive_retry_at_ms,next_drive_ms FROM sk_session WHERE session_id='s'"
    )


def run_parent(database, entered, finish, calls, *, phase="prepare", scheduled=False, ttl_ms=TTL):
    with open_store(database) as raw_worker:
        worker = wrap_fixture_store(raw_worker)
        rt = parent_runtime(worker, database)
        rt.ttl_ms, rt.heartbeat_seconds = ttl_ms, INTERVAL
        rt.heartbeat_store = worker.heartbeat_store

        def pause():
            entered.set()
            assert finish.wait(10), "admission was not released"

        def admit(_plan):
            calls.append(1)
            worker._rows("INSERT INTO host_child VALUES (1)")
            if phase == "prepare":
                pause()
            return tuple(children.ChildSession(fixture_child_spec(worker, "s", sid), "work") for sid in CHILDREN)

        def hook(point, record):
            if phase == "before_commit" and point == "before_commit" and record and record.kind == "op_parked":
                pause()

        rt.children["child_task"], rt.hook = admit, hook
        return kernel.drive(worker, "s", runtime=(lambda _: rt) if scheduled else rt)


def observe_three_ttls(store, ttl_ms=TTL):
    initial = lease_row(store)
    begin = store._now()
    expiries = set()

    def observed():
        row = store._one(
            f"SELECT lease_owner,lease_epoch,lease_until_ms,({store.clock_sql}) FROM sk_session WHERE session_id='s'"
        )
        assert row[:2] == initial[:2]
        assert row[2] > row[3], "heartbeat let the real database lease expire"
        expiries.add(row[2])
        return row[3] - begin >= 3 * ttl_ms and len(expiries) >= 4

    wait_until(observed)
    return initial, expiries


@pytest.mark.usefixtures("local_counting_ready")
@pytest.mark.parametrize("competitor", [False, True])
@pytest.mark.parametrize("phase", ["prepare", "before_commit"])
def test_long_admission_hold_renews_or_fails_before_commit(store, database, competitor, phase, caplog):
    start(store)
    store._rows("CREATE TABLE host_child (id INTEGER PRIMARY KEY)")
    entered, finish, calls = threading.Event(), threading.Event(), []
    caplog.set_level(logging.DEBUG, logger=kernel.__name__)
    observed_ttl = 1000
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            run_parent,
            database,
            entered,
            finish,
            calls,
            phase=phase,
            ttl_ms=observed_ttl,
        )
        try:
            wait_until(entered.is_set, future=future)
            if phase == "prepare":
                initial, expiries = observe_three_ttls(store, observed_ttl)
            else:
                initial, expiries = lease_row(store), set()
            if competitor and phase == "prepare":
                assert not kernel.drive(store, "s", runtime=lambda _: pytest.fail("competitor acquired live lease"))
            assert calls == [1]
            if phase == "before_commit":
                # The foreground transaction already holds the session row. The
                # renewal keeper cannot update it, so crossing the safety margin
                # must fence this append even though the process still holds a
                # lease object.
                sleep(observed_ttl / 2000 + 0.1)
        finally:
            finish.set()
            if phase == "before_commit":
                with pytest.raises(LeaseLost):
                    future.result(timeout=10)
            else:
                assert future.result(timeout=10)
    parked = rows(store, kind="op_parked")
    started = [r for r in rows(store, kind="op_started") if "/tool/" in r.record.operation_id]
    assert calls == [1]
    if phase == "prepare":
        assert len(expiries) >= 4
        assert store.list_sessions() == (*CHILDREN, "s")
        assert store._rows("SELECT * FROM host_child") == [(1,)]
        assert len(parked) == len(started) == 1
        assert parked[0].commit_id == started[0].commit_id
        assert parked[0].writer_epoch == started[0].writer_epoch == initial[1]
        assert lease_row(store)[:3] == (None, initial[1], None)
        assert all(len(rows(store, sid)) == 1 and len(store.peek_inbox(sid)) == 1 for sid in CHILDREN)
    else:
        assert not expiries
        assert store.list_sessions() == ("s",)
        assert store._rows("SELECT * FROM host_child") == []
        assert not parked and not started
    diagnostic = [r for r in caplog.records if getattr(r, "phase", None) == phase and r.levelno == logging.WARNING]
    assert len(diagnostic) == 1
    if phase == "prepare":
        assert diagnostic[0].callback_ms >= 3 * observed_ttl
    assert diagnostic[0].kernel_atomic_ms >= diagnostic[0].callback_ms


@pytest.mark.parametrize("external", [False, True])
def test_postgres_external_admission_retains_early_deadline(store, database, monkeypatch, external):
    monkeypatch.setattr(_ContentSessionTxAdapter, "_requires_early_persistence_deadline", external, raising=False)
    start(store)
    store._rows("CREATE TABLE host_child (id INTEGER PRIMARY KEY)")
    clock = ManualClock()
    rt = runtime(store, database, [], clock=clock, ttl_ms=4000, heartbeat_seconds=INTERVAL)
    lease = store.acquire("s", owner="admission", ttl_ms=rt.ttl_ms)
    assert lease is not None
    scope = kernel._Scope(lease, rt, now_ms=store._now())
    driver = kernel._Driver(store, "s", rt, scope)
    initial_deadline = scope.safety_deadline

    with open_store(database) as renewal:

        def prepare(_tx):
            store._rows("INSERT INTO host_child VALUES (1)")
            clock.advance(2000)
            scope.poll(renewal)
            clock.advance(2000)
            assert initial_deadline < clock.monotonic() < scope.safety_deadline
            assert scope.lease.epoch == lease.epoch
            return []

        try:
            if external:
                with pytest.raises(LeaseLost, match="session persistence reached the lease safety deadline"):
                    driver.commit([], prepare=prepare)
            else:
                driver.commit([], prepare=prepare)
        finally:
            scope.close()
            store.release(scope.lease)

    assert store._rows("SELECT * FROM host_child") == ([] if external else [(1,)])


def test_slow_append_encoding_uses_last_lease(store, database, monkeypatch):
    start(store)
    entered, finish = threading.Event(), threading.Event()
    from vv_agent.session.records import Record

    encode = Record.encode

    def slow_encode(self):
        if self.kind == "op_planned" and not entered.is_set() and threading.current_thread().name != "MainThread":
            entered.set()
            assert finish.wait(10)
        return encode(self)

    monkeypatch.setattr(Record, "encode", slow_encode)

    def work():
        with open_store(database) as raw_worker:
            worker = wrap_fixture_store(raw_worker)
            rt = runtime(worker, database, [LLMResponse("done")], ttl_ms=TTL, heartbeat_seconds=INTERVAL)
            rt.heartbeat_store = worker.heartbeat_store
            return kernel.drive(worker, "s", runtime=rt)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(work)
        try:
            wait_until(entered.is_set, future=future)
            initial, _ = observe_three_ttls(store)
        finally:
            finish.set()
            assert future.result(timeout=10)
    assert rows(store)[-1].record.kind == "turn_ended"
    assert {r.writer_epoch for r in rows(store)[1:]} == {initial[1]}


@pytest.mark.usefixtures("local_counting_ready")
def test_real_expiry_fences_old_admission_and_preserves_new_schedule(store, database, monkeypatch):
    start(store)
    store._rows("CREATE TABLE host_child (id INTEGER PRIMARY KEY)")
    entered, finish, blocked, unblock = (threading.Event() for _ in range(4))
    calls = []
    renew = CorePostgresStore.renew

    def fault(self, lease, *, ttl_ms):
        if threading.current_thread().name == "session-heartbeat" and entered.is_set():
            blocked.set()
            assert unblock.wait(10)
        return renew(self, lease, ttl_ms=ttl_ms)

    monkeypatch.setattr(CorePostgresStore, "renew", fault)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_parent, database, entered, finish, calls, scheduled=True)
        try:
            wait_until(entered.is_set, future=future)
            wait_until(blocked.is_set, future=future)
            initial = lease_row(store)
            wait_until(lambda: store._now() >= initial[2])
            # Even the original identity cannot revive a genuinely expired lease.
            from vv_agent.session.store import Lease

            with pytest.raises(LeaseLost):
                renew(store, Lease("s", initial[0], initial[1], initial[2]), ttl_ms=TTL)
            new = store.acquire("s", owner="replacement", ttl_ms=10000)
            assert new is not None and new.epoch == initial[1] + 1
            store.defer_drive(new, retry_after_ms=2345)
            retained = lease_row(store)
        finally:
            unblock.set()
            finish.set()
        with pytest.raises(LeaseLost):
            future.result(timeout=10)
    assert calls == [1]
    assert lease_row(store) == retained
    assert store.list_sessions() == ("s",)
    assert store._rows("SELECT * FROM host_child") == []
    assert not rows(store, kind="op_parked")
    assert not [r for r in rows(store, kind="op_started") if "/tool/" in r.record.operation_id]


@pytest.mark.parametrize("fault", ["row_lock", "statement"])
def test_heartbeat_database_timeout_is_bounded_lease_loss(store, database, monkeypatch, fault):
    start(store)
    lease = store.acquire("s", owner="old", ttl_ms=TTL)
    assert lease is not None
    entered = threading.Event()
    renew = CorePostgresStore.renew

    def paused(self, value, *, ttl_ms):
        entered.set()
        if fault == "statement":
            self._rows("SELECT pg_sleep(10)")
        return renew(self, value, ttl_ms=ttl_ms)

    monkeypatch.setattr(CorePostgresStore, "renew", paused)
    heartbeat = kernel._LeaseHeartbeat(lease, store.heartbeat_store, TTL, INTERVAL, now_ms=store._now())
    with open_store(database) as locker, locker.atomic() as tx:
        if fault == "row_lock":
            tx.lock_session("s")
        begin = monotonic()
        heartbeat.start()
        assert entered.wait(5)
        wait_until(lambda: heartbeat.lost)
        assert not heartbeat.lease_keeper.is_alive(heartbeat)
        assert monotonic() - begin < 3
        assert heartbeat.lost
        assert isinstance(heartbeat.heartbeat_error, LeaseLost)
        expected_cause = "LockNotAvailable" if fault == "row_lock" else "QueryCanceled"
        assert type(heartbeat.heartbeat_error.__cause__).__name__ == expected_cause
        heartbeat.close()
    with pytest.raises(LeaseLost), heartbeat.committing():
        pytest.fail("lost heartbeat admitted an append")


@pytest.mark.parametrize("phase", ["renew", "publication", "exit"])
def test_close_waits_for_inflight_heartbeat_and_retains_committed_snapshot(store, database, monkeypatch, phase):
    start(store)
    lease = store.acquire("s", owner="old", ttl_ms=2000)
    assert lease is not None
    paused, finish = threading.Event(), threading.Event()
    renew = CorePostgresStore.renew

    def pause():
        paused.set()
        assert finish.wait(5)

    def controlled(self, value, *, ttl_ms):
        if phase == "renew":
            pause()
        result = renew(self, value, ttl_ms=ttl_ms)
        if phase == "publication":
            pause()
        return result

    @contextmanager
    def connection():
        with store.heartbeat_store() as other:
            yield other
        if phase == "exit":
            pause()

    monkeypatch.setattr(CorePostgresStore, "renew", controlled)
    heartbeat = kernel._LeaseHeartbeat(lease, connection, 2000, INTERVAL, now_ms=store._now())
    heartbeat.start()
    if phase == "exit":
        assert heartbeat.ready.wait(5)
        heartbeat.stop.set()
    assert paused.wait(5)
    with ThreadPoolExecutor(max_workers=1) as pool:
        closed = pool.submit(heartbeat.close)
        try:
            wait_until(heartbeat.stop.is_set)
            assert not closed.done()
        finally:
            finish.set()
        closed.result(timeout=3)
    assert not heartbeat.lease_keeper.is_alive(heartbeat)
    assert heartbeat.latest_poll is not None
    assert heartbeat.lease is heartbeat.latest_poll.lease
    assert heartbeat.lease.expires_at_ms == lease_row(store)[2]


def test_factory_close_timeout_never_hands_off_or_releases(store, database, monkeypatch):
    start(store)
    entered, finish = threading.Event(), threading.Event()
    threads, releases, model_calls = [], [], []
    monkeypatch.setattr(timing, "HEARTBEAT_CLOSE_SECONDS", 0.1)
    heartbeat_store = store.heartbeat_store

    @contextmanager
    def slow_exit():
        with heartbeat_store() as other:
            yield other
        threads.append(threading.current_thread())
        entered.set()
        assert finish.wait(5)

    monkeypatch.setattr(store, "heartbeat_store_factory", lambda: slow_exit)
    monkeypatch.setattr(store, "release", lambda lease: releases.append(lease))

    def factory(_sid):
        return runtime(store, database, [lambda _: model_calls.append(1) or LLMResponse("done")])

    try:
        begin = monotonic()
        with pytest.raises(LeaseLost, match="did not stop"):
            kernel.drive(store, "s", runtime=factory)
        assert monotonic() - begin < 1
        assert entered.is_set() and not releases and not model_calls
        assert lease_row(store)[0] is not None
    finally:
        finish.set()
        for thread in threads:
            thread.join(timeout=3)
            assert not thread.is_alive()


def test_factory_runtime_heartbeat_contexts_never_overlap(store, database, monkeypatch):
    start(store)
    active, events = [], []
    open_heartbeat = store.heartbeat_store

    @contextmanager
    def connection():
        assert not active
        active.append(threading.current_thread())
        events.append("enter")
        try:
            with open_heartbeat() as other:
                yield other
        finally:
            events.append("exit")
            active.pop()

    monkeypatch.setattr(store, "heartbeat_store_factory", lambda: connection)

    def factory(_sid):
        rt = runtime(store, database, [LLMResponse("done")], ttl_ms=TTL, heartbeat_seconds=INTERVAL)
        rt.heartbeat_store = connection
        return rt

    assert kernel.drive(store, "s", runtime=factory)
    assert len(events) >= 4 and len(events) % 2 == 0
    assert all(events[index : index + 2] == ["enter", "exit"] for index in range(0, len(events), 2))
    assert not active
    assert rows(store)[-1].record.payload["status"] == "completed"


def test_committing_does_not_wait_for_renewal_and_identity_fence_prevents_publication(store, database, monkeypatch):
    start(store)
    lease = store.acquire("s", owner="old", ttl_ms=2000)
    assert lease is not None
    heartbeat = kernel._LeaseHeartbeat(lease, store.heartbeat_store, 2000, INTERVAL, now_ms=None)
    heartbeat.start()
    assert heartbeat.ready.wait(5)
    before = heartbeat.latest_poll
    assert before is not None

    entered, finish = threading.Event(), threading.Event()
    incoming = []
    renew = CorePostgresStore.renew

    def controlled(self, value, *, ttl_ms):
        incoming.append(value)
        entered.set()
        assert finish.wait(5)
        return renew(self, value, ttl_ms=ttl_ms)

    monkeypatch.setattr(CorePostgresStore, "renew", controlled)
    try:
        assert entered.wait(5)
        began = monotonic()
        with heartbeat.committing() as observed:
            assert observed is before.lease
        assert monotonic() - began < 0.5
    finally:
        finish.set()
    wait_until(
        lambda: heartbeat.latest_poll is not None and heartbeat.latest_poll.lease.expires_at_ms > before.lease.expires_at_ms
    )
    latest = heartbeat.latest_poll
    assert latest is not None

    heartbeat.close()
    identity = kernel._LeaseHeartbeat(latest.lease, store.heartbeat_store, 2000, INTERVAL, now_ms=None)

    def changed(self, value, *, ttl_ms):
        result = renew(self, value, ttl_ms=ttl_ms)
        return replace(result, lease=replace(result.lease, owner="other", epoch=result.lease.epoch + 1))

    monkeypatch.setattr(CorePostgresStore, "renew", changed)
    with pytest.raises(LeaseLost, match="identity"), identity.heartbeat_store() as other:
        identity.poll(other)
    assert identity.lost and identity.lease is latest.lease


def test_transaction_logs_follow_rollback_and_hide_exception_contents(store, database, monkeypatch, caplog):
    start(store)
    rt = parent_runtime(store, database)
    rt.heartbeat_store = store.heartbeat_store
    statuses = []
    log = kernel.logger.log

    def logged(*args, **kwargs):
        statuses.append(store.connection.info.transaction_status.name)
        log(*args, **kwargs)

    def fail(_plan):
        raise ValueError("secret callback contents")

    monkeypatch.setattr(kernel.logger, "log", logged)
    rt.children["child_task"] = fail
    caplog.set_level(logging.DEBUG, logger=kernel.__name__)
    with pytest.raises(ValueError, match="secret callback contents"):
        kernel.drive(store, "s", runtime=rt)
    assert statuses and set(statuses) == {"IDLE"}
    diagnostic = [r for r in caplog.records if getattr(r, "phase", None) == "prepare" and getattr(r, "outcome", None) == "error"]
    assert len(diagnostic) == 1 and diagnostic[0].outcome == "error"
    assert diagnostic[0].exception_type == "ValueError"
    assert "secret callback contents" not in caplog.text


def test_locked_append_still_checks_ttl_after_writing(store, database, monkeypatch):
    start(store)
    turn = runtime_turn_record()
    lease = store.acquire("s", owner="old", ttl_ms=TTL)
    assert lease is not None
    commit = SQLSessionTx._commit

    def slow_commit(self, *args):
        commit(self, *args)
        # This fault is inside the final write, so renewal cannot rescue it.
        wait_until(lambda: store._now() >= lease.expires_at_ms)

    monkeypatch.setattr(SQLSessionTx, "_commit", slow_commit)
    with pytest.raises(LeaseLost), store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=1, commit_id="late", records=(turn,))
    assert len(raw_rows(store)) == 1


def test_append_timeouts_use_store_clock_and_reset_after_transaction(store, monkeypatch):
    monkeypatch.setattr(PostgresStore, "clock_sql", "100000")
    start(store)
    lease = store.acquire("s", owner="old", ttl_ms=TTL)
    assert lease is not None
    settings = "SELECT current_setting('lock_timeout'),current_setting('statement_timeout')"
    original = store._one(settings)
    with store.atomic():
        with store._bound_append(lease):
            assert store._one(settings) == ("400ms", "400ms")
        assert store._one(settings) == original
    assert store._one(settings) == original


@pytest.mark.parametrize("failure", [None, "callback", "timeout"])
def test_append_restores_timeouts_inside_outer_host_transaction(store, monkeypatch, failure):
    from psycopg.errors import QueryCanceled

    start(store)
    turn = runtime_turn_record()
    lease = store.acquire("s", owner="host", ttl_ms=200 if failure == "timeout" else 4000)
    assert lease is not None
    settings = "SELECT current_setting('lock_timeout'),current_setting('statement_timeout')"
    commit = SQLSessionTx._commit

    def injected(self, *args):
        if failure == "callback":
            raise RuntimeError("append failed")
        if failure == "timeout":
            self.store._rows("SELECT pg_sleep(2)")
        return commit(self, *args)

    monkeypatch.setattr(SQLSessionTx, "_commit", injected)
    store._rows("CREATE TABLE host_append (id INTEGER PRIMARY KEY)")
    with store.connection.transaction():
        store._rows("SET LOCAL lock_timeout='7s'")
        store._rows("SET LOCAL statement_timeout='8s'")
        store._rows("INSERT INTO host_append VALUES (1)")
        with store.atomic() as tx:
            bind_lease_guard(tx, lease)
            try:
                tx.append("s", lease=lease, expected_seq=1, commit_id="host", records=(turn,))
            except (RuntimeError, QueryCanceled) as exc:
                assert failure is not None
                assert isinstance(exc, QueryCanceled if failure == "timeout" else RuntimeError)
            else:
                assert failure is None
            assert store._one(settings) == ("7s", "8s")
        assert store._one(settings) == ("7s", "8s")
        store._rows("SELECT pg_sleep(0.3)")
        store._rows("INSERT INTO host_append VALUES (2)")
    assert store._rows("SELECT id FROM host_append ORDER BY id") == [(1,), (2,)]
    assert len(raw_rows(store)) == (2 if failure is None else 1)


def test_slow_append_within_lease_has_no_fixed_one_second_cap(store, monkeypatch):
    start(store)
    turn = runtime_turn_record()
    lease = store.acquire("s", owner="host", ttl_ms=4000)
    assert lease is not None
    commit = SQLSessionTx._commit

    def slow(self, *args):
        self.store._rows("SELECT pg_sleep(1.2)")
        return commit(self, *args)

    monkeypatch.setattr(SQLSessionTx, "_commit", slow)
    begin = monotonic()
    with store.atomic() as tx:
        bind_lease_guard(tx, lease)
        tx.append("s", lease=lease, expected_seq=1, commit_id="slow", records=(turn,))
    assert monotonic() - begin >= 1.2
    assert len(raw_rows(store)) == 2


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("host_factory", [False, True])
def test_two_transient_heartbeat_timeouts_keep_epoch_and_commit(store, database, monkeypatch, scheduled, host_factory):
    from psycopg.errors import LockNotAvailable

    start(store)
    recovered = threading.Event()
    attempts, epochs = [], []
    renew = CorePostgresStore.renew

    def intermittent(self, lease, *, ttl_ms):
        if threading.current_thread().name == "session-heartbeat":
            attempts.append(1)
            epochs.append(lease.epoch)
            if len(attempts) <= 2:
                raise LockNotAvailable("injected row lock timeout")
            result = renew(self, lease, ttl_ms=ttl_ms)
            recovered.set()
            return result
        return renew(self, lease, ttl_ms=ttl_ms)

    def response(_request):
        assert recovered.wait(5)
        return LLMResponse("done")

    monkeypatch.setattr(CorePostgresStore, "renew", intermittent)
    factory = (lambda: open_store(database)) if host_factory else store.heartbeat_store
    monkeypatch.setattr(store, "heartbeat_store", factory)
    rt = runtime(store, database, [response], ttl_ms=3000, heartbeat_seconds=INTERVAL)
    rt.heartbeat_store = factory
    assert kernel.drive(store, "s", runtime=(lambda _: rt) if scheduled else rt)
    assert len(attempts) >= 3 and len(set(epochs)) == 1
    assert {r.writer_epoch for r in rows(store)[1:]} == set(epochs)
    assert rows(store)[-1].record.payload["status"] == "completed"


def test_persistent_heartbeat_timeouts_exhaust_margin_and_abort_append(store, database, monkeypatch):
    from psycopg.errors import LockNotAvailable

    start(store)
    attempted = threading.Event()
    attempts, scopes, model_calls = [], [], []
    renew = CorePostgresStore.renew
    init = kernel._Scope.__init__

    def capture(self, *args, **kwargs):
        init(self, *args, **kwargs)
        scopes.append(self)

    def failing(self, lease, *, ttl_ms):
        if threading.current_thread().name == "session-heartbeat":
            attempts.append(1)
            attempted.set()
            raise LockNotAvailable("injected row lock timeout")
        return renew(self, lease, ttl_ms=ttl_ms)

    def response(_request):
        model_calls.append(1)
        return LLMResponse("cannot commit")

    monkeypatch.setattr(kernel._Scope, "__init__", capture)
    monkeypatch.setattr(CorePostgresStore, "renew", failing)
    rt = runtime(store, database, [response], ttl_ms=2000, heartbeat_seconds=INTERVAL)
    rt.heartbeat_store = store.heartbeat_store
    begin = monotonic()
    with pytest.raises(LeaseLost):
        kernel.drive(store, "s", runtime=rt)
    assert attempted.is_set() and len(attempts) > 2
    assert scopes and scopes[0].lost and scopes[0].heartbeat_error is not None
    assert model_calls == []
    assert 0.9 <= monotonic() - begin < 6
    assert not rows(store, kind="turn_ended")
    count = len(attempts)
    scopes[0].close()
    assert len(attempts) == count


def test_committing_refuses_expired_published_lease(store):
    start(store)
    lease = store.acquire("s", owner="old", ttl_ms=100)
    assert lease is not None
    now = [0.0]

    class Clock(SystemClock):
        def monotonic(self):
            return now[0]

    heartbeat = kernel._LeaseHeartbeat(lease, store.heartbeat_store, 100, INTERVAL, now_ms=store._now(), clock=Clock())
    # A database wall-clock jump does not imply the local monotonic deadline elapsed.
    now[0] = heartbeat.expires_deadline
    with pytest.raises(LeaseLost, match="expired"), heartbeat.committing():
        pytest.fail("expired published lease admitted an append")


def test_close_interrupts_transient_retry_wait_without_another_renew(store, database, monkeypatch):
    from psycopg.errors import LockNotAvailable

    start(store)
    lease = store.acquire("s", owner="old", ttl_ms=4000)
    assert lease is not None
    attempts = []
    failed = threading.Event()

    def failing(self, lease, *, ttl_ms):
        attempts.append(1)
        failed.set()
        raise LockNotAvailable("injected row lock timeout")

    monkeypatch.setattr(PostgresStore, "renew", failing)
    heartbeat = kernel._LeaseHeartbeat(lease, lambda: open_store(database), 4000, 1.0, now_ms=store._now())
    heartbeat.start()
    assert failed.wait(5)
    begin = monotonic()
    heartbeat.close()
    assert monotonic() - begin < 6
    assert not heartbeat.lease_keeper.is_alive(heartbeat)
    assert len(attempts) == 1
    with store.heartbeat_store() as other, pytest.raises(LeaseLost):
        heartbeat.poll(other)
    assert len(attempts) == 1


def test_host_heartbeat_retries_two_real_lock_timeouts_then_renews(store, database, monkeypatch):
    from psycopg.errors import LockNotAvailable

    start(store)
    lease = store.acquire("s", owner="old", ttl_ms=3000)
    assert lease is not None
    failures = []
    recovered = threading.Event()
    renew = PostgresStore.renew

    def observed(self, value, *, ttl_ms):
        try:
            result = renew(self, value, ttl_ms=ttl_ms)
        except LockNotAvailable:
            failures.append(1)
            raise
        recovered.set()
        return result

    monkeypatch.setattr(PostgresStore, "renew", observed)
    heartbeat = kernel._LeaseHeartbeat(lease, lambda: open_store(database), 3000, INTERVAL, now_ms=store._now())
    try:
        with open_store(database) as locker, locker.atomic() as tx:
            tx.lock_session("s")
            heartbeat.start()
            wait_until(lambda: len(failures) >= 2)
            assert not heartbeat.lost and not recovered.is_set()
        assert recovered.wait(5)
        assert heartbeat.ready.wait(5)
        assert not heartbeat.lost and heartbeat.lease.epoch == lease.epoch
        assert heartbeat.lease.expires_at_ms > lease.expires_at_ms
    finally:
        begin = monotonic()
        heartbeat.close()
        assert monotonic() - begin < 6


def test_append_lock_wait_over_250ms_succeeds_within_lease(store, database):
    start(store)
    turn = runtime_turn_record()
    lease = store.acquire("s", owner="host", ttl_ms=4000)
    assert lease is not None
    entered = threading.Event()

    def append():
        with open_store(database) as worker, worker.atomic() as tx:
            tx._lease_guard = lambda: lease
            entered.set()
            tx.append("s", lease=lease, expected_seq=1, commit_id="waiting", records=(turn,))

    with ThreadPoolExecutor(max_workers=1) as pool:
        with store.atomic() as tx:
            tx.lock_session("s")
            future = pool.submit(append)
            wait_until(entered.is_set, future=future)
            store._rows("SELECT pg_sleep(0.5)")
        future.result(timeout=5)
    assert len(raw_rows(store)) == 2


def test_public_heartbeat_stopped_distinguishes_normal_stop_from_lease_loss(store):
    from vv_agent import HeartbeatStopped

    start(store)
    lease = store.acquire("s", owner="normal-stop", ttl_ms=2000)
    heartbeat = kernel._LeaseHeartbeat(lease, store.heartbeat_store, 2000, INTERVAL, now_ms=store._now())
    heartbeat.start()
    heartbeat.close()
    with pytest.raises(HeartbeatStopped) as caught:
        heartbeat.poll(store)
    assert type(caught.value) is HeartbeatStopped
    assert isinstance(caught.value, LeaseLost)
