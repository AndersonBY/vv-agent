"""W4 lease scheduling, lock budgeting and handoff-snapshot checks."""

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from typing import cast

import pytest
from sim.assembly import Scope as SimScope
from sim.world import SimWorld

from vv_agent.kernel._lease_scope import LeaseOwnership, apply_controls
from vv_agent.kernel._postgres import PostgresStore as CorePostgresStore
from vv_agent.kernel._store import Lease, LeaseLost, LeasePoll, SessionStore
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.session import kernel as session_kernel
from vv_agent.session import timing as session_timing
from vv_agent.session._driver_scope import _LeaseHeartbeat, _Scope
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.timing import SystemClock, ThreadInvoker, ThreadLeaseKeeper
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall

from .conftest import open_store
from .helpers import create_fixture_session, fixture_content_reader, push_fixture_input, wrap_fixture_store
from .test_recovery_matrix import runtime as base_runtime


@function_tool
def inspect_cycle() -> str:
    return "checked"


def wait_until(predicate, *, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition did not become true"
        time.sleep(0.01)


def test_tenant_scoped_factory_renews_the_requested_session(tmp_path):
    tenant_id = "tenant-scoped-heartbeat"
    sessions = SessionDriver(tenant_id=tenant_id)
    try:
        sessions.create(
            "s",
            str(tmp_path),
            principal={
                "tenant_id": tenant_id,
                "subject_id": "subject",
                "subject_kind": "user",
                "attributes": {"role": "user"},
            },
        )
        with sessions._open_drive_store() as drive:
            lease = drive.acquire("s", owner="tenant-owner", ttl_ms=4_000)
            assert lease is not None and lease.tenant_id == tenant_id
            heartbeat = _LeaseHeartbeat(
                lease,
                drive.heartbeat_store_factory(),
                4_000,
                0.04,
            )
            heartbeat.start()
            try:
                assert heartbeat.ready.wait(5)
                assert heartbeat.successful_ready and not heartbeat.lost
                poll = heartbeat.latest_poll
                assert poll is not None
                assert poll.lease.session_id == "s" and poll.lease.tenant_id == tenant_id
            finally:
                heartbeat.close()
    finally:
        sessions.close()


@pytest.mark.postgres
def test_postgres_tenant_scoped_factory_renews_the_requested_session(store):
    tenant_id = "tenant-scoped-postgres-heartbeat"
    tenant = wrap_fixture_store(store.for_tenant(tenant_id))
    create_fixture_session(tenant, "s", consumers=())
    lease = tenant.acquire("s", owner="tenant-owner", ttl_ms=4_000)
    assert lease is not None and lease.tenant_id == tenant_id
    heartbeat = _LeaseHeartbeat(
        lease,
        tenant.heartbeat_store_factory(),
        4_000,
        0.04,
        now_ms=store._now(),
    )
    heartbeat.start()
    try:
        assert heartbeat.ready.wait(5)
        assert heartbeat.successful_ready and not heartbeat.lost
        poll = heartbeat.latest_poll
        assert poll is not None
        assert poll.lease.session_id == "s" and poll.lease.tenant_id == tenant_id
    finally:
        heartbeat.close()


def create_sessions(store, *session_ids):
    store.configure_input_admission(allow_unsigned=True)
    wrapped = wrap_fixture_store(store)
    for session_id in session_ids:
        create_fixture_session(wrapped, session_id, consumers=())


def heartbeat_for(
    path,
    session_id,
    *,
    ttl_ms=4_000,
    heartbeat_seconds=0.04,
    now_ms=None,
    opened_threads=None,
    connection_ids=None,
    closed_threads=None,
):
    with open_store(path) as store:
        lease = store.acquire(session_id, owner=f"owner-{session_id}", ttl_ms=ttl_ms)
    assert lease is not None

    @contextmanager
    def renewal_store():
        with SQLiteStore.standalone(
            str(path),
            busy_timeout_ms=10,
            heartbeat_busy_timeout_ms=10,
        ) as store:
            store.configure_input_admission(allow_unsigned=True)
            if opened_threads is not None:
                opened_threads.append(threading.get_ident())
            if connection_ids is not None:
                connection_ids.append(id(store.connection))
            try:
                yield store
            finally:
                if closed_threads is not None:
                    closed_threads.append(threading.get_ident())

    return _LeaseHeartbeat(
        lease,
        renewal_store,
        ttl_ms,
        heartbeat_seconds,
        now_ms=now_ms,
        sqlite_heartbeat_retry_budget_seconds=heartbeat_seconds,
        sqlite_heartbeat_retry_delay_seconds=min(heartbeat_seconds / 8, 0.05),
    )


@pytest.mark.sqlite_file
def test_default_thread_keeper_shares_scheduler_and_closes_scopes_independently(tmp_path, monkeypatch):
    path = tmp_path / "shared.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        create_sessions(store, "s1", "s2", "s3")
        driver_connection = id(store.connection)

    opened_threads, opened_connections, renewal_connections, closed_threads = [], [], [], []
    renewals = []
    renew = SQLiteStore.renew

    def observed(self, lease, *, ttl_ms):
        renewals.append(lease.session_id)
        renewal_connections.append(id(self.connection))
        return renew(self, lease, ttl_ms=ttl_ms)

    monkeypatch.setattr(SQLiteStore, "renew", observed)
    scopes = [
        heartbeat_for(
            path,
            session_id,
            now_ms=None,
            opened_threads=opened_threads,
            connection_ids=opened_connections,
            closed_threads=closed_threads,
        )
        for session_id in ("s1", "s2", "s3")
    ]
    keeper = ThreadLeaseKeeper()
    assert keeper is not ThreadLeaseKeeper()
    for scope in scopes:
        assert not any(hasattr(scope, name) for name in ("lock", "poll_lock", "commit_lock", "thread"))
    try:
        for scope in scopes:
            scope.start()
        wait_until(lambda: all(scope.successful_ready and scope.latest_poll is not None for scope in scopes))

        assert len(set(opened_threads)) == 1
        assert threading.get_ident() not in opened_threads
        scheduler_thread = opened_threads[0]
        assert len(opened_connections) == len(scopes)
        assert len(set(opened_connections)) == len(scopes)
        initial_opened_connections = tuple(opened_connections)
        assert driver_connection not in renewal_connections
        assert set(renewal_connections) == set(opened_connections)
        for scope in scopes:
            snapshot = scope.latest_poll
            assert snapshot is not None and snapshot.lease is scope.lease
            assert snapshot.db_now_ms < snapshot.lease.expires_at_ms

        scopes[0].close()
        before = scopes[1].latest_poll.lease.expires_at_ms
        wait_until(lambda: scopes[1].latest_poll.lease.expires_at_ms > before)
        assert not scopes[1].lost and not scopes[2].lost

        latest = scopes[2].latest_poll
        with scopes[2].committing() as lease:
            assert lease is latest.lease
            time.sleep(0.12)
        assert scopes[2].latest_poll.db_now_ms >= latest.db_now_ms

        scopes[1].close()
        scopes[2].close()
        assert opened_connections == list(initial_opened_connections)
        assert closed_threads == [scheduler_thread] * len(scopes)
        wait_until(lambda: scheduler_thread not in {thread.ident for thread in threading.enumerate()})

        with open_store(path) as store:
            assert store.release(scopes[2].lease)
        rebuilt = heartbeat_for(
            path,
            "s3",
            now_ms=None,
            opened_threads=opened_threads,
            connection_ids=opened_connections,
            closed_threads=closed_threads,
        )
        try:
            rebuilt.start()
            wait_until(lambda: rebuilt.successful_ready)
            rebuilt_thread = opened_threads[-1]
            assert rebuilt_thread in {thread.ident for thread in threading.enumerate()}
        finally:
            rebuilt.close()
        assert len(opened_connections) == len(scopes) + 1
        assert len(closed_threads) == len(scopes) + 1
        wait_until(lambda: rebuilt_thread not in {thread.ident for thread in threading.enumerate()})
    finally:
        for scope in scopes:
            scope.close()


@pytest.mark.sqlite_file
def test_runtime_monotonic_schedule_stays_separate_from_sqlite_elapsed_epoch(tmp_path):
    class DatabaseClock:
        def __init__(self):
            self.started = time.monotonic()

        def now_ms(self):
            return 9_000_000 + int((time.monotonic() - self.started) * 1000)

        def monotonic(self):
            return time.monotonic()

        def monotonic_ns(self):
            return time.monotonic_ns()

        def sleep(self, seconds):
            time.sleep(seconds)

    class RuntimeClock(DatabaseClock):
        def monotonic(self):
            return time.monotonic() + 1_000

        def monotonic_ns(self):
            return time.monotonic_ns() + 1_000_000_000_000

    path = tmp_path / "clock-domains.sqlite"
    database_clock, runtime_clock = DatabaseClock(), RuntimeClock()
    with SQLiteStore.standalone(str(path), clock=database_clock) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        create_sessions(store, "s")
        lease = store.acquire("s", owner="owner", ttl_ms=3_000)
        assert lease is not None

    @contextmanager
    def renewal_store():
        with SQLiteStore.standalone(str(path), clock=database_clock) as store:
            store.configure_input_admission(allow_unsigned=True)
            yield store

    heartbeat = _LeaseHeartbeat(
        lease,
        renewal_store,
        3_000,
        0.04,
        clock=runtime_clock,
        now_ms=None,
    )
    local_started = runtime_clock.monotonic()
    try:
        heartbeat.start()
        assert heartbeat.ready.wait(5)
        first = heartbeat.latest_poll
        assert first is not None
        assert heartbeat.clock is runtime_clock
        assert first.db_now_ms >= 9_000_000
        assert first.lease.expires_at_ms > first.db_now_ms
        assert local_started < heartbeat.expires_deadline < local_started + 4
        assert heartbeat.safety_deadline < heartbeat.expires_deadline

        def renewed_after_first() -> bool:
            latest = heartbeat.latest_poll
            return latest is not None and latest.lease.expires_at_ms > first.lease.expires_at_ms

        wait_until(renewed_after_first)
        assert heartbeat.successful_ready and not heartbeat.lost
    finally:
        heartbeat.close()


@pytest.mark.sqlite_file
def test_public_kernel_three_agents_share_keeper_across_model_cycles(tmp_path, monkeypatch):
    database = tmp_path / "public-runtime.sqlite"
    session_ids = ("agent-a", "agent-b", "agent-c")
    model_calls = {session_id: 0 for session_id in session_ids}
    model_barrier = threading.Barrier(len(session_ids), timeout=8)
    lock_held, lock_seen = threading.Event(), threading.Event()
    heartbeat_threads, heartbeat_connections, lock_failures = set(), set(), []
    renew = SQLiteStore.renew

    def observed(self, lease, *, ttl_ms):
        if threading.current_thread().name == "session-heartbeat":
            heartbeat_threads.add(threading.get_ident())
            heartbeat_connections.add(id(self.connection))
        try:
            return renew(self, lease, ttl_ms=ttl_ms)
        except sqlite3.OperationalError as error:
            if threading.current_thread().name == "session-heartbeat" and "database is locked" in str(error):
                lock_failures.append(error)
                lock_seen.set()
            raise

    monkeypatch.setattr(SQLiteStore, "renew", observed)

    def provider(session_id):
        def answer(_request):
            model_calls[session_id] += 1
            cycle = model_calls[session_id]
            if cycle == 1 and session_id == "agent-a":
                with SQLiteStore.standalone(str(database)) as locker:
                    locker.connection.execute("BEGIN IMMEDIATE")
                    lock_held.set()
                    assert lock_seen.wait(5), "heartbeat did not contend with the foreground writer"
                    locker.connection.execute("ROLLBACK")
            model_barrier.wait()
            if cycle == 1:
                return LLMResponse("", [ToolCall(f"{session_id}/tool", "inspect_cycle", {})])
            return LLMResponse(f"{session_id} done")

        return answer

    stores = []
    try:
        stores = [SQLiteStore(str(database)) for _ in session_ids]
        stores[0].install_schema()
        for store in stores:
            store.configure_input_admission(allow_unsigned=True)
        wrapped = [wrap_fixture_store(store) for store in stores]
        for session_id, store in zip(session_ids, wrapped, strict=True):
            create_fixture_session(store, session_id, consumers=())
            push_fixture_input(store, session_id, InboxItem(f"{session_id}/input", "user", {"content": "go"}))
        runtimes = []
        for session_id, store in zip(session_ids, wrapped, strict=True):
            runtime = base_runtime(
                str(database),
                [provider(session_id), provider(session_id)],
                [inspect_cycle],
                content_reader=fixture_content_reader(store, session_id),
            )
            runtime.heartbeat_store = store.heartbeat_store
            runtimes.append(runtime)
        with ThreadPoolExecutor(max_workers=len(session_ids)) as pool:
            results = [
                pool.submit(session_kernel.drive, store, session_id, runtime=runtime)
                for session_id, store, runtime in zip(session_ids, wrapped, runtimes, strict=True)
            ]
            assert [future.result(timeout=20) for future in results] == [True, True, True]
        assert model_calls == {session_id: 2 for session_id in session_ids}
        assert lock_held.is_set() and lock_seen.is_set()
        assert lock_failures
        assert len(heartbeat_threads) == 1
        assert threading.get_ident() not in heartbeat_threads
        assert not ({id(store.connection) for store in stores} & heartbeat_connections)
    finally:
        for store in stores:
            store.connection.close()
        wait_until(
            lambda: "session-heartbeat" not in {thread.name for thread in threading.enumerate()},
            timeout=8,
        )


@pytest.mark.sqlite_file
def test_stop_after_committed_renewal_retains_receipt_and_waits_for_inflight(tmp_path, monkeypatch):
    path = tmp_path / "stop-handoff.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        create_sessions(store, "s")
    heartbeat = heartbeat_for(path, "s", ttl_ms=4_000, heartbeat_seconds=0.04, now_ms=None)
    heartbeat.start()
    assert heartbeat.ready.wait(5)
    committed_at, finish = threading.Event(), threading.Event()
    receipts = []
    renew = SQLiteStore.renew

    def delayed(self, lease, *, ttl_ms):
        receipt = renew(self, lease, ttl_ms=ttl_ms)
        receipts.append(receipt)
        committed_at.set()
        assert finish.wait(5)
        return receipt

    monkeypatch.setattr(SQLiteStore, "renew", delayed)
    assert committed_at.wait(5)
    calls_at_stop = len(receipts)
    with ThreadPoolExecutor(max_workers=1) as pool:
        closing = pool.submit(heartbeat.close)
        try:
            wait_until(heartbeat.stop.is_set)
            assert not closing.done()
        finally:
            finish.set()
        closing.result(timeout=5)
    assert heartbeat.latest_poll is receipts[-1]
    assert len(receipts) == calls_at_stop
    assert not heartbeat.lease_keeper.is_alive(heartbeat)


@pytest.mark.parametrize("error_name", ["LockNotAvailable", "QueryCanceled"])
def test_postgres_heartbeat_retries_explicit_database_timeouts(error_name):
    from psycopg import errors

    failure = getattr(errors, error_name)("injected database timeout")

    class Postgres(CorePostgresStore):
        def __init__(self):
            pass

        def _rows(self, query, params=()):
            del query, params
            return []

        def renew(self, lease, *, ttl_ms):
            del lease, ttl_ms
            raise failure

    heartbeat = _LeaseHeartbeat(
        Lease("s", "owner", 1, 5_000),
        lambda: contextmanager(lambda: (yield))(),
        4_000,
        0.04,
        now_ms=1_000,
    )
    heartbeat.heartbeat_once(Postgres())
    assert heartbeat.heartbeat_error is failure
    assert not heartbeat.lost


@pytest.mark.postgres
@pytest.mark.parametrize("error_name", ["LockNotAvailable", "QueryCanceled"])
@pytest.mark.parametrize("wrapper", ["native", "tenant", "content-tenant"])
def test_postgres_heartbeat_unwraps_backend_before_classifying_timeouts(store, monkeypatch, error_name, wrapper):
    from psycopg import errors

    create_sessions(store, "s")
    lease = store.acquire("s", owner="wrapped-timeout", ttl_ms=5_000)
    assert lease is not None
    failure = getattr(errors, error_name)("injected database timeout")

    candidate = store
    if wrapper == "tenant":
        candidate = store.for_tenant(store.tenant_id)
    elif wrapper == "content-tenant":
        candidate = wrap_fixture_store(store.for_tenant(store.tenant_id))

    def fail_renew(_store, _lease, *, ttl_ms):
        del ttl_ms
        raise failure

    heartbeat = _LeaseHeartbeat(
        lease,
        lambda: contextmanager(lambda: (yield))(),
        4_000,
        0.04,
        now_ms=store._now(),
    )
    monkeypatch.setattr(CorePostgresStore, "renew", fail_renew)
    try:
        heartbeat.heartbeat_once(candidate)
        assert heartbeat.heartbeat_error is failure
        assert not heartbeat.lost
    finally:
        store.release(lease)


def test_postgres_heartbeat_does_not_generalize_storage_errors():
    from psycopg import OperationalError

    failure = OperationalError("plain storage failure")

    class Postgres(CorePostgresStore):
        def __init__(self):
            pass

        def _rows(self, query, params=()):
            del query, params
            return []

        def renew(self, lease, *, ttl_ms):
            del lease, ttl_ms
            raise failure

    heartbeat = _LeaseHeartbeat(
        Lease("s", "owner", 1, 5_000),
        lambda: contextmanager(lambda: (yield))(),
        4_000,
        0.04,
        now_ms=1_000,
    )
    with pytest.raises(OperationalError) as caught:
        heartbeat.heartbeat_once(Postgres())
    assert caught.value is failure


@pytest.mark.sqlite_file
def test_short_foreground_write_lock_retries_without_reinvoking_model_callback(tmp_path, monkeypatch):
    database = tmp_path / "foreground.sqlite"
    with SQLiteStore.standalone(str(database)) as store:
        store.install_schema()
        create_sessions(store, "s")
        lock_held, lock_seen, renewed_after_lock = (threading.Event() for _ in range(3))
        model_calls, lock_failures, renew_durations = [], [], []
        renew = SQLiteStore.renew

        def observed(self, lease, *, ttl_ms):
            started = time.monotonic()
            try:
                try:
                    result = renew(self, lease, ttl_ms=ttl_ms)
                except sqlite3.OperationalError as error:
                    if threading.current_thread().name == "session-heartbeat" and "database is locked" in str(error):
                        lock_failures.append(error)
                        lock_seen.set()
                    raise
                if threading.current_thread().name == "session-heartbeat" and lock_held.is_set():
                    renewed_after_lock.set()
                return result
            finally:
                renew_durations.append(time.monotonic() - started)

        def answer(_request):
            model_calls.append(1)
            return LLMResponse("done")

        monkeypatch.setattr(SQLiteStore, "renew", observed)
        heartbeat = heartbeat_for(database, "s", ttl_ms=3_000, heartbeat_seconds=0.04, now_ms=None)
        heartbeat.start()
        with SQLiteStore.standalone(str(database)) as locker:
            locker.connection.execute("BEGIN IMMEDIATE")
            lock_held.set()
            assert lock_seen.wait(5)
            response = ThreadInvoker().invoke(
                lambda: answer(None),
                timeout=1,
                token=CancellationToken(),
                cancellation_grace=0.1,
                clock=SystemClock(),
            )
            assert response == LLMResponse("done")
            locker.connection.execute("ROLLBACK")
        assert renewed_after_lock.wait(5)
        heartbeat.close()

    assert model_calls == [1]
    assert lock_failures
    assert max(renew_durations) < 0.25
    assert renewed_after_lock.is_set()


@pytest.mark.sqlite_file
def test_sqlite_lock_past_safety_deadline_becomes_lease_lost(tmp_path, monkeypatch):
    path = tmp_path / "safety.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        create_sessions(store, "s")
    heartbeat = heartbeat_for(path, "s", ttl_ms=400, heartbeat_seconds=0.025, now_ms=None)
    heartbeat.start()
    assert heartbeat.ready.wait(5)
    assert heartbeat.successful_ready
    assert heartbeat.safety_deadline == pytest.approx(heartbeat.expires_deadline - 0.2)
    try:
        with SQLiteStore.standalone(str(path)) as locker:
            locker.connection.execute("BEGIN IMMEDIATE")
            time.sleep(0.28)
            locker.connection.execute("ROLLBACK")
        wait_until(lambda: heartbeat.lost)
        assert isinstance(heartbeat.heartbeat_error, LeaseLost)
        assert heartbeat.latest_poll is not None
    finally:
        heartbeat.close()
        assert not heartbeat.lease_keeper.is_alive(heartbeat)


@pytest.mark.sqlite_file
def test_non_lock_sqlite_error_is_storage_failure_without_retry(tmp_path, monkeypatch):
    path = tmp_path / "storage.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        create_sessions(store, "s")
    attempts = []
    failure = sqlite3.OperationalError("disk I/O error")

    def failing(_store, _lease, *, ttl_ms):
        attempts.append(ttl_ms)
        raise failure

    monkeypatch.setattr(SQLiteStore, "renew", failing)
    heartbeat = heartbeat_for(path, "s", now_ms=None)
    heartbeat.start()
    assert heartbeat.ready.wait(5)
    assert attempts == [heartbeat.ttl_ms]
    assert heartbeat.lost and heartbeat.heartbeat_error is failure
    heartbeat.close()


@pytest.mark.sqlite_file
def test_sqlite_lock_attempt_budget_timeout_retains_original_cause(tmp_path, monkeypatch):
    path = tmp_path / "attempt-budget.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        create_sessions(store, "s")
    heartbeat = heartbeat_for(path, "s", ttl_ms=4_000, heartbeat_seconds=0.04, now_ms=None)
    heartbeat.start()
    assert heartbeat.ready.wait(5)
    assert heartbeat.successful_ready

    failure = sqlite3.OperationalError("database table is locked: sk_session")

    def locked(_store, _lease, *, ttl_ms):
        del ttl_ms
        raise failure

    monkeypatch.setattr(SQLiteStore, "renew", locked)
    try:
        wait_until(lambda: isinstance(heartbeat.heartbeat_error, TimeoutError))
        assert heartbeat.heartbeat_error.__cause__ is failure
        assert time.monotonic() < heartbeat.safety_deadline
        assert not heartbeat.lost
    finally:
        heartbeat.close()


@pytest.mark.sqlite_file
def test_close_rejects_inflight_renewal_that_does_not_stop(tmp_path, monkeypatch):
    path = tmp_path / "close.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        create_sessions(store, "s")
    heartbeat = heartbeat_for(path, "s", ttl_ms=4_000, heartbeat_seconds=0.04, now_ms=None)
    heartbeat.start()
    assert heartbeat.ready.wait(5)
    entered, release = threading.Event(), threading.Event()
    renew = SQLiteStore.renew

    def blocked(self, lease, *, ttl_ms):
        entered.set()
        assert release.wait(5)
        return renew(self, lease, ttl_ms=ttl_ms)

    monkeypatch.setattr(SQLiteStore, "renew", blocked)
    monkeypatch.setattr(session_timing, "HEARTBEAT_CLOSE_SECONDS", 0.05)
    assert entered.wait(5)
    try:
        with pytest.raises(LeaseLost, match="heartbeat did not stop"):
            heartbeat.close()
    finally:
        release.set()
    wait_until(lambda: not heartbeat.lease_keeper.is_alive(heartbeat))
    heartbeat.close()


def test_latest_poll_is_one_snapshot_and_controls_cannot_cancel_new_token():
    class Clock:
        def now_ms(self):
            return 1_000

        def monotonic(self):
            return 10.0

        def monotonic_ns(self):
            return 10_000_000_000

        def sleep(self, seconds):
            del seconds

    class Store:
        def __init__(self, receipt):
            self.receipt = receipt

        def renew(self, lease, *, ttl_ms):
            del lease, ttl_ms
            return self.receipt

    class Sink:
        def __init__(self, on_cancel=None):
            self.reasons = []
            self.on_cancel = on_cancel

        def cancel(self, reason):
            self.reasons.append(reason)
            if self.on_cancel is not None:
                self.on_cancel()

    lease = Lease("s", "owner", 1, 2_000)
    control = InboxItem("c", "control", {"action": "cancel"}, target_turn_id="old", generation=1, actor={})
    receipt = LeasePoll(replace(lease, expires_at_ms=2_500), 1_100, 3, (control,))
    initial_token, old, refreshed, next_generation, next_turn = Sink(), Sink(), Sink(), Sink(), Sink()
    tokens = iter((initial_token, old, refreshed, next_generation, next_turn))
    state = LeaseOwnership(
        lease,
        1_500,
        0.04,
        clock=Clock(),
        stop=threading.Event(),
        ready=threading.Event(),
        now_ms=1_000,
        token_factory=lambda: next(tokens),
    )
    assert state.poll(cast(SessionStore, Store(receipt))) is receipt
    assert state.latest_poll is receipt
    assert state.lease is receipt.lease
    adopted = LeaseOwnership(
        lease,
        1_500,
        0.04,
        clock=Clock(),
        stop=threading.Event(),
        ready=threading.Event(),
        now_ms=None,
    )
    adopted.adopt_confirmed(receipt, expires_deadline=10.5)
    assert adopted.latest_poll is receipt
    with adopted.committing() as adopted_lease:
        assert adopted_lease is receipt.lease

    applied = {"c": object()}
    state.bind_turn("old", 1, applied)
    assert state.token is old and not old.reasons
    state.bind_turn("old", 1, applied)
    binding = state.control_binding
    assert binding is not None
    assert state.token is old and binding.applied_inputs is applied
    state.reset_token()
    assert state.token is refreshed and state.turn_id == "old" and state.generation == 1
    assert not refreshed.reasons
    apply_controls(state, receipt)
    assert not refreshed.reasons
    state.bind_turn("old", 2, applied)
    assert state.token is next_generation and not next_generation.reasons
    state.bind_turn("next", 2, {})
    assert state.token is next_turn and not next_turn.reasons
    apply_controls(state, receipt)
    assert not next_turn.reasons


def test_confirmed_poll_is_read_only_and_validates_ownership_state():
    class Clock:
        def __init__(self):
            self.mono = 10.0

        def now_ms(self):
            return 1_000

        def monotonic(self):
            return self.mono

        def monotonic_ns(self):
            return int(self.mono * 1_000_000_000)

        def sleep(self, seconds):
            del seconds

    class Store:
        def __init__(self, receipt):
            self.receipt, self.calls = receipt, 0

        def renew(self, lease, *, ttl_ms):
            del lease, ttl_ms
            self.calls += 1
            return self.receipt

    clock = Clock()
    lease = Lease("s", "owner", 1, 2_000)
    receipt = LeasePoll(replace(lease, expires_at_ms=2_500), 1_100, 3, ())
    store = Store(receipt)
    state = LeaseOwnership(
        lease,
        1_500,
        0.04,
        clock=clock,
        stop=threading.Event(),
        ready=threading.Event(),
        now_ms=1_000,
    )
    store_port = cast(SessionStore, store)
    assert state.poll(store_port) is receipt
    calls = store.calls
    assert state.confirmed_poll() is receipt
    assert store.calls == calls
    clock.mono = state.expires_deadline
    with pytest.raises(LeaseLost, match="expired"):
        state.confirmed_poll()
    assert store.calls == calls

    stopped = LeaseOwnership(
        lease,
        1_500,
        0.04,
        clock=Clock(),
        stop=threading.Event(),
        ready=threading.Event(),
        now_ms=1_000,
    )
    assert stopped.poll(store_port) is receipt
    stopped.request_stop()
    with pytest.raises(LeaseLost, match="heartbeat stopped"):
        stopped.confirmed_poll()
    assert store.calls == calls + 1


def test_factory_scope_without_token_factory_rejects_control_binding():
    class Clock:
        def now_ms(self):
            return 1_000

        def monotonic(self):
            return 10.0

        def monotonic_ns(self):
            return 10_000_000_000

        def sleep(self, seconds):
            del seconds

    state = LeaseOwnership(
        Lease("s", "owner", 1, 5_000),
        4_000,
        0.04,
        clock=Clock(),
        stop=threading.Event(),
        ready=threading.Event(),
        now_ms=1_000,
    )
    assert state.control_binding is None
    with pytest.raises(RuntimeError, match="token factory"):
        state.bind_turn("turn", 1, {})
    with pytest.raises(RuntimeError, match="token factory"):
        state.reset_token()


def test_adopted_controls_wait_until_runtime_binds_the_current_turn(tmp_path):
    class Clock:
        def now_ms(self):
            return 1_000

        def monotonic(self):
            return 10.0

        def monotonic_ns(self):
            return 10_000_000_000

        def sleep(self, seconds):
            del seconds

    clock = Clock()
    lease = Lease("s", "owner", 1, 5_000)
    control = InboxItem("c", "control", {"action": "cancel"}, target_turn_id=None, generation=None, actor={})
    receipt = LeasePoll(lease, 1_000, 2, (control,))
    runtime = base_runtime(tmp_path / "adopted.sqlite", [], clock=clock, cancellation_grace=0.1)
    scope = _Scope(lease, runtime, now_ms=None)
    scope.adopt_confirmed(receipt, expires_deadline=14.0)
    assert scope.latest_poll is receipt and not scope.token.cancelled
    scope.bind_turn("turn", 1, {})
    assert scope.token.reason == "cancel"


def test_scheduled_keeper_first_renew_establishes_anchor_without_wall_clock(tmp_path):
    with SimWorld(tmp_path / "scheduled.sqlite", seed=17) as world:
        world.create()
        selected = world.runtime("s", "owner")
        store = world.connection("owner")
        lease = store.acquire("s", owner="owner", ttl_ms=selected.ttl_ms)
        assert lease is not None
        scope = SimScope(lease, selected, now_ms=None)
        assert scope.latest_poll is None
        scope.start()
        try:
            assert scope.successful_ready and scope.latest_poll is not None
            assert scope.latest_poll.db_now_ms < scope.latest_poll.lease.expires_at_ms
            before = scope.latest_poll
            world.advance(300)
            world.step()
            assert scope.latest_poll.lease.expires_at_ms > before.lease.expires_at_ms
        finally:
            scope.close()
