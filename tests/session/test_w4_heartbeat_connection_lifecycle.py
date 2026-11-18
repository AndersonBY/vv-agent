"""LeaseKeeper owns one real heartbeat connection per registration."""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager

import pytest

from vv_agent.kernel._sqlite import SQLiteStore as CoreSQLiteStore
from vv_agent.kernel._store import LeaseLost
from vv_agent.session import timing as session_timing
from vv_agent.session._driver_scope import _lease_backend, _LeaseHeartbeat, _Scope
from vv_agent.session.reducer import Fold
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.timing import SystemClock, ThreadLeaseKeeper, _SharedLeaseKeeper

from .helpers import create_fixture_session, wrap_fixture_store
from .test_recovery_matrix import runtime


def _wait_until(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition did not become true"
        time.sleep(0.005)


def _lease_for(path, session_id: str, *, ttl_ms: int = 4_000):
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        create_fixture_session(wrap_fixture_store(store), session_id, consumers=())
        lease = store.acquire(session_id, owner=f"owner-{session_id}", ttl_ms=ttl_ms)
    assert lease is not None
    return lease


def _assert_current_lease(store, lease) -> None:
    status = store.lease_status(lease.session_id)
    assert status[1] == lease.epoch
    assert status[2] > 0
    owner, epoch, _until = store._one(
        "SELECT lease_owner,lease_epoch,lease_until_ms FROM sk_session WHERE tenant_id=%s AND session_id=%s",
        (lease.tenant_id, lease.session_id),
    )
    assert (owner, epoch) == (lease.owner, lease.epoch)


@pytest.mark.sqlite_file
def test_core_drive_store_factory_tenant_facade_is_supported(tmp_path):
    database = tmp_path / "core-tenant-heartbeat.sqlite"
    with CoreSQLiteStore.standalone(str(database), fold_factory=Fold, clock=SystemClock()) as core:
        core.install_schema()
        core.configure_input_admission(allow_unsigned=True)
        create_fixture_session(wrap_fixture_store(core), "s", consumers=())
        lease = core.acquire("s", owner="core-owner", ttl_ms=4_000)
        assert lease is not None

        with core.drive_store_factory()() as drive:
            backend = _lease_backend(drive)
            assert type(backend) is CoreSQLiteStore
            assert backend.path == str(database)
            heartbeat = _LeaseHeartbeat(lease, drive.heartbeat_store_factory(), 4_000, 0.04)
            heartbeat.start()
            try:
                assert heartbeat.ready.wait(5)
                assert heartbeat.successful_ready and not heartbeat.lost
                poll = heartbeat.latest_poll
                assert poll is not None and poll.lease.tenant_id == lease.tenant_id
            finally:
                heartbeat.close()


@pytest.mark.sqlite_file
def test_runtime_sqlite_heartbeat_retry_budget_controls_lock_retries(tmp_path, monkeypatch):
    database = tmp_path / "runtime-retry-budget.sqlite"
    lease = _lease_for(database, "s")
    with SQLiteStore.standalone(str(database), heartbeat_busy_timeout_ms=10) as store:
        rt = runtime(
            database,
            [],
            ttl_ms=4_000,
            heartbeat_seconds=0.01,
            sqlite_heartbeat_retry_budget_seconds=0.12,
            sqlite_heartbeat_retry_delay_seconds=0.02,
        )
        scope = _Scope(lease, rt, now_ms=store._now())
        attempts = []

        def locked(_lease, *, ttl_ms):
            del ttl_ms
            attempts.append(time.monotonic())
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(store, "renew", locked)
        started = time.monotonic()
        scope.heartbeat_once(store)
        elapsed = time.monotonic() - started

    assert len(attempts) >= 4
    assert 0.08 <= elapsed < 0.8
    assert isinstance(scope.heartbeat_error, TimeoutError)
    assert isinstance(scope.heartbeat_error.__cause__, sqlite3.OperationalError)
    assert not scope.lost


@pytest.mark.sqlite_file
def test_runtime_sqlite_retry_budget_cannot_exceed_lease_safety_deadline(tmp_path, monkeypatch):
    database = tmp_path / "runtime-safe-deadline.sqlite"
    lease = _lease_for(database, "s", ttl_ms=600)
    with SQLiteStore.standalone(str(database), heartbeat_busy_timeout_ms=10) as store:
        rt = runtime(
            database,
            [],
            ttl_ms=600,
            heartbeat_seconds=0.01,
            sqlite_heartbeat_retry_budget_seconds=10,
            sqlite_heartbeat_retry_delay_seconds=0.01,
        )
        scope = _Scope(lease, rt, now_ms=store._now())

        def locked(_lease, *, ttl_ms):
            del ttl_ms
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(store, "renew", locked)
        started = time.monotonic()
        with pytest.raises(LeaseLost, match="safety deadline"):
            scope.heartbeat_once(store)
        elapsed = time.monotonic() - started

    assert elapsed < 1
    assert scope.lost


@pytest.mark.sqlite_file
def test_real_sqlite_write_lock_caps_each_renew_to_retry_window(tmp_path, monkeypatch):
    database = tmp_path / "real-lock-retry-window.sqlite"
    lease = _lease_for(database, "s")
    retry_budget_seconds = 0.08
    with SQLiteStore.standalone(str(database), heartbeat_busy_timeout_ms=300) as store:
        store.configure_input_admission(allow_unsigned=True)
        rt = runtime(
            database,
            [],
            ttl_ms=4_000,
            heartbeat_seconds=0.01,
            sqlite_heartbeat_retry_budget_seconds=retry_budget_seconds,
            sqlite_heartbeat_retry_delay_seconds=0.01,
        )
        scope = _Scope(lease, rt, now_ms=store._now())
        busy_timeouts = []
        renew_durations = []
        renew = SQLiteStore.renew

        def observed(current, *, ttl_ms):
            busy_timeouts.append(heartbeat_store.connection.execute("PRAGMA busy_timeout").fetchone()[0])
            started = time.monotonic()
            try:
                return renew(heartbeat_store, current, ttl_ms=ttl_ms)
            finally:
                renew_durations.append(time.monotonic() - started)

        with store.heartbeat_store() as heartbeat_store:
            monkeypatch.setattr(heartbeat_store, "renew", observed)
            with SQLiteStore.standalone(str(database), busy_timeout_ms=30_000) as locker, locker.atomic():
                started = time.monotonic()
                scope.heartbeat_once(heartbeat_store)
                elapsed = time.monotonic() - started
                restored_busy_timeout = heartbeat_store.connection.execute("PRAGMA busy_timeout").fetchone()

    assert isinstance(scope.heartbeat_error, TimeoutError)
    assert isinstance(scope.heartbeat_error.__cause__, sqlite3.OperationalError)
    assert not scope.lost
    assert busy_timeouts and max(busy_timeouts) <= int(retry_budget_seconds * 1000)
    assert restored_busy_timeout == (300,)
    assert renew_durations and max(renew_durations) < 0.25
    assert elapsed < 0.25


@pytest.mark.sqlite_file
def test_successful_sqlite_renew_after_safety_deadline_is_not_confirmed(tmp_path, monkeypatch):
    # Keep unrelated host wall-clock steps out of this real SQLite renewal window.
    wall_origin_ns = time.time_ns()
    mono_origin_ns = time.monotonic_ns()
    monkeypatch.setattr(session_timing.time, "time_ns", lambda: wall_origin_ns + time.monotonic_ns() - mono_origin_ns)
    database = tmp_path / "late-success.sqlite"
    lease = _lease_for(database, "s", ttl_ms=10_000)
    with SQLiteStore.standalone(str(database), heartbeat_busy_timeout_ms=300) as store:
        store.configure_input_admission(allow_unsigned=True)
        rt = runtime(
            database,
            [],
            ttl_ms=10_000,
            heartbeat_seconds=1.0,
            sqlite_heartbeat_retry_budget_seconds=2,
        )
        scope = _Scope(lease, rt, now_ms=store._now())
        renew = SQLiteStore.renew

        def delayed(current, *, ttl_ms):
            delay = scope.safety_deadline - time.monotonic() + 0.02
            if delay > 0:
                time.sleep(delay)
            return renew(store, current, ttl_ms=ttl_ms)

        monkeypatch.setattr(store, "renew", delayed)
        with pytest.raises(LeaseLost, match="safety deadline"):
            scope.heartbeat_once(store)

    assert scope.lost
    assert scope.latest_poll is None
    assert not scope.ready.is_set()


@pytest.mark.sqlite_file
def test_keeper_reuses_one_heartbeat_connection_for_repeated_renewals(tmp_path, monkeypatch):
    database = tmp_path / "reused-heartbeat.sqlite"
    lease = _lease_for(database, "s")
    with SQLiteStore.standalone(str(database)) as store:
        store.configure_input_admission(allow_unsigned=True)
        heartbeat_store = store.heartbeat_store
        foreground_thread = threading.get_ident()
        foreground_connection = id(store.connection)

    opened_threads: list[int] = []
    used_threads: list[int] = []
    closed_threads: list[int] = []
    opened_connections: list[int] = []
    used_connections: list[int] = []
    opened_stores = []
    renew_barrier = threading.Barrier(2, timeout=2)

    @contextmanager
    def tracked_store():
        with heartbeat_store() as store:
            store.configure_input_admission(allow_unsigned=True)
            opened_threads.append(threading.get_ident())
            opened_connections.append(id(store.connection))
            opened_stores.append(store)
            try:
                yield store
            finally:
                closed_threads.append(threading.get_ident())

    heartbeat_once = _LeaseHeartbeat.heartbeat_once

    def observed_heartbeat_once(self, store):
        used_threads.append(threading.get_ident())
        used_connections.append(id(store.connection))
        if len(used_threads) <= 3:
            renew_barrier.wait()
        heartbeat_once(self, store)

    monkeypatch.setattr(_LeaseHeartbeat, "heartbeat_once", observed_heartbeat_once)
    heartbeat = _LeaseHeartbeat(lease, tracked_store, 4_000, 0.02, lease_keeper=ThreadLeaseKeeper())
    heartbeat.start()
    for _ in range(3):
        renew_barrier.wait()
    _wait_until(lambda: heartbeat.successful_ready and len(used_threads) >= 3)

    assert len(opened_threads) == 1
    assert len(opened_stores) == 1
    assert len(set(opened_threads + used_threads)) == 1
    assert opened_threads[0] != foreground_thread
    assert set(opened_connections) == set(used_connections)
    assert foreground_connection not in used_connections
    keeper_thread = opened_threads[0]
    connection = opened_stores[0].connection

    heartbeat.close()

    assert closed_threads == [keeper_thread]
    assert not heartbeat.lease_keeper.is_alive(heartbeat)
    _wait_until(lambda: keeper_thread not in {thread.ident for thread in threading.enumerate()})
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


@pytest.mark.sqlite_file
def test_blocked_heartbeat_resource_close_keeps_lease_unreleased(tmp_path, monkeypatch):
    database = tmp_path / "blocked-close.sqlite"
    lease = _lease_for(database, "s")
    keeper = _SharedLeaseKeeper()
    close_entered = threading.Event()
    release_close = threading.Event()
    opened_threads: list[int] = []
    closed_threads: list[int] = []
    connection = None

    @contextmanager
    def blocked_store():
        nonlocal connection
        manager = SQLiteStore.standalone(str(database))
        store = manager.__enter__()
        store.configure_input_admission(allow_unsigned=True)
        connection = store.connection
        opened_threads.append(threading.get_ident())
        try:
            yield store
        finally:
            close_entered.set()
            try:
                assert release_close.wait(3)
            finally:
                manager.__exit__(None, None, None)
                closed_threads.append(threading.get_ident())

    heartbeat = _LeaseHeartbeat(lease, blocked_store, 4_000, 0.02, lease_keeper=keeper)
    heartbeat.start()
    assert heartbeat.ready.wait(3) and heartbeat.successful_ready
    assert len(opened_threads) == 1
    keeper_thread = opened_threads[0]

    monkeypatch.setattr(session_timing, "HEARTBEAT_CLOSE_SECONDS", 0.05)
    try:
        with pytest.raises(LeaseLost, match="heartbeat did not stop"):
            heartbeat.close()
        assert close_entered.is_set()
        assert keeper.is_alive(heartbeat)
        with SQLiteStore.standalone(str(database)) as verifier:
            verifier.configure_input_admission(allow_unsigned=True)
            _assert_current_lease(verifier, lease)
    finally:
        release_close.set()
        stopped = False
        try:
            _wait_until(lambda: not keeper.is_alive(heartbeat))
            stopped = True
            assert closed_threads == [keeper_thread]
            assert connection is not None
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
            with SQLiteStore.standalone(str(database)) as verifier:
                verifier.configure_input_admission(allow_unsigned=True)
                _assert_current_lease(verifier, lease)
        finally:
            if stopped:
                with SQLiteStore.standalone(str(database)) as verifier:
                    verifier.configure_input_admission(allow_unsigned=True)
                    assert verifier.release(lease)
                    assert not verifier.release(lease)


@pytest.mark.persistent_store
def test_persistent_keeper_reuses_one_heartbeat_connection(store, monkeypatch):
    wrapped = wrap_fixture_store(store)
    create_fixture_session(wrapped, "persistent", consumers=())
    lease = wrapped.acquire("persistent", owner="persistent-owner", ttl_ms=4_000)
    assert lease is not None
    heartbeat_store = store.heartbeat_store_factory()
    foreground_thread = threading.get_ident()
    foreground_connection = id(store.connection)
    keeper = _SharedLeaseKeeper()
    board = threading.Barrier(2, timeout=3)
    opened_threads: list[int] = []
    used_threads: list[int] = []
    closed_threads: list[int] = []
    opened_connections: list[int] = []
    used_connections: list[int] = []
    opened_stores = []

    @contextmanager
    def tracked_store():
        with heartbeat_store() as actual:
            actual.configure_input_admission(allow_unsigned=True)
            opened_threads.append(threading.get_ident())
            opened_connections.append(id(actual.connection))
            opened_stores.append(actual)
            try:
                yield actual
            finally:
                closed_threads.append(threading.get_ident())

    heartbeat_once = _LeaseHeartbeat.heartbeat_once

    def observed_heartbeat_once(self, actual):
        used_threads.append(threading.get_ident())
        used_connections.append(id(actual.connection))
        if len(used_threads) <= 3:
            board.wait()
        heartbeat_once(self, actual)

    monkeypatch.setattr(_LeaseHeartbeat, "heartbeat_once", observed_heartbeat_once)
    heartbeat = _LeaseHeartbeat(
        lease,
        tracked_store,
        4_000,
        0.02,
        now_ms=store._now(),
        lease_keeper=keeper,
    )
    try:
        heartbeat.start()
        for _ in range(3):
            board.wait()
        _wait_until(lambda: heartbeat.successful_ready and len(used_threads) >= 3)
        assert len(opened_threads) == 1
        assert len(opened_stores) == 1
        assert len(set(opened_threads + used_threads)) == 1
        assert opened_threads[0] != foreground_thread
        assert set(opened_connections) == set(used_connections)
        assert foreground_connection not in used_connections
        latest = heartbeat.latest_poll
        assert latest is not None
        assert (latest.lease.owner, latest.lease.epoch) == (lease.owner, lease.epoch)
        keeper_thread = opened_threads[0]
    finally:
        board.abort()
        try:
            heartbeat.close()
        finally:
            if not keeper.is_alive(heartbeat):
                store.release(heartbeat.lease)
    assert closed_threads == [keeper_thread]
    assert not keeper.is_alive(heartbeat)
    closed_connection = opened_stores[0].connection
    if isinstance(opened_stores[0], SQLiteStore):
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            closed_connection.execute("SELECT 1")
    else:
        assert closed_connection.closed


@pytest.mark.sqlite_file
def test_heartbeat_resource_close_error_propagates_from_close(tmp_path):
    database = tmp_path / "failed-close.sqlite"
    lease = _lease_for(database, "s")
    keeper = _SharedLeaseKeeper()
    failure = RuntimeError("heartbeat resource close failed")

    @contextmanager
    def failing_store():
        with SQLiteStore.standalone(str(database)) as store:
            store.configure_input_admission(allow_unsigned=True)
            yield store
        raise failure

    heartbeat = _LeaseHeartbeat(lease, failing_store, 4_000, 0.02, lease_keeper=keeper)
    heartbeat.start()
    assert heartbeat.ready.wait(3) and heartbeat.successful_ready

    with pytest.raises(RuntimeError) as caught:
        heartbeat.close()

    assert caught.value is failure
    assert heartbeat.lost and heartbeat.heartbeat_error is failure
    _wait_until(lambda: not keeper.is_alive(heartbeat))
