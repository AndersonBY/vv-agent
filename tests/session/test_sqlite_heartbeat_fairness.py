"""SQLite lease scheduling must leave a window for foreground writers."""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager

import pytest

from vv_agent.kernel._sqlite import SQLiteStore as CoreSQLiteStore
from vv_agent.session import timing as session_timing
from vv_agent.session._driver_scope import _LeaseHeartbeat
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import Lease, SessionStore
from vv_agent.session.timing import _SharedLeaseKeeper

from .helpers import create_fixture_session, wrap_fixture_store


def _wait_until(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition did not become true"
        time.sleep(0.005)


@pytest.mark.sqlite_file
def test_slow_renewal_barrier_case_does_not_starve_foreground_writer(tmp_path, monkeypatch):
    """A deterministic barrier places renewal work inside the interval."""

    database = tmp_path / "heartbeat-fairness.sqlite"
    heartbeat_seconds = 0.02
    with SQLiteStore.standalone(str(database)) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        create_fixture_session(wrap_fixture_store(store), "s", consumers=())
        lease = store.acquire("s", owner="heartbeat-owner", ttl_ms=4_000)
        assert lease is not None
        heartbeat_store = store.heartbeat_store_factory()

    armed = threading.Event()
    slow_started = threading.Event()
    release_slow = threading.Event()
    foreground_attempted = threading.Event()
    renew_starts: list[float] = []
    renew_ends: list[float] = []
    foreground_errors: list[BaseException] = []
    foreground_elapsed: list[float] = []

    original_transaction = CoreSQLiteStore._transaction

    @contextmanager
    def paused_renewal_transaction(requested_store):
        started = time.monotonic()
        if threading.current_thread().name == "session-heartbeat":
            renew_starts.append(started)
        try:
            with original_transaction(requested_store):
                if threading.current_thread().name == "session-heartbeat" and armed.is_set() and not slow_started.is_set():
                    slow_started.set()
                    assert release_slow.wait(3), "foreground test did not release the renewal"
                yield
        finally:
            if threading.current_thread().name == "session-heartbeat":
                renew_ends.append(time.monotonic())

    monkeypatch.setattr(CoreSQLiteStore, "_transaction", paused_renewal_transaction)

    heartbeat = _LeaseHeartbeat(
        lease,
        heartbeat_store,
        4_000,
        heartbeat_seconds,
    )

    def foreground_write() -> None:
        connection = sqlite3.connect(database, isolation_level=None, timeout=2)
        started = time.monotonic()
        try:
            connection.execute("PRAGMA busy_timeout=2000")
            foreground_attempted.set()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE sk_session SET next_drive_ms=next_drive_ms WHERE tenant_id='default' AND session_id='s'")
            connection.execute("COMMIT")
            foreground_elapsed.append(time.monotonic() - started)
        except BaseException as exc:
            foreground_errors.append(exc)
        finally:
            connection.close()

    worker: threading.Thread | None = None
    try:
        heartbeat.start()
        assert heartbeat.ready.wait(3) and heartbeat.successful_ready, repr(heartbeat.heartbeat_error)

        armed.set()
        assert slow_started.wait(3), "scheduled renewal did not enter its write transaction"
        worker = threading.Thread(target=foreground_write, name="foreground-writer")
        worker.start()
        assert foreground_attempted.wait(3)

        begin_count = len(renew_starts)
        release_slow.set()
        worker.join(2)
        assert not worker.is_alive(), "foreground writer remained blocked behind renewal catch-up"
        assert not foreground_errors

        _wait_until(lambda: len(renew_starts) > begin_count)
        assert foreground_elapsed and foreground_elapsed[0] < 1.0
        assert renew_starts[begin_count] - renew_ends[begin_count - 1] >= heartbeat_seconds * 0.75
        assert heartbeat.successful_ready and not heartbeat.lost
    finally:
        release_slow.set()
        if worker is not None:
            worker.join(2)
        heartbeat.close()


def test_finished_callback_can_reenter_close_without_waiting_on_itself(tmp_path, monkeypatch):
    keeper = _SharedLeaseKeeper()
    callback_done = threading.Event()
    callback_elapsed: list[float] = []

    database = tmp_path / "reentrant-close.sqlite"
    with SQLiteStore.standalone(str(database)) as store:
        heartbeat_store = store.heartbeat_store_factory()
        lease = Lease("s", "owner", 1, 2_000)

        class ReentrantHeartbeat(_LeaseHeartbeat):
            def __init__(self) -> None:
                self.finished = 0
                super().__init__(lease, heartbeat_store, 2_000, 0.01, lease_keeper=keeper)

            def heartbeat_once(self, store: SessionStore) -> None:
                del store

            def heartbeat_finished(self) -> None:
                super().heartbeat_finished()
                self.finished += 1
                if self.finished != 1:
                    return
                started = time.monotonic()
                keeper.close(self)
                callback_elapsed.append(time.monotonic() - started)
                callback_done.set()

        monkeypatch.setattr(session_timing, "HEARTBEAT_CLOSE_SECONDS", 0.5)
        heartbeat = ReentrantHeartbeat()
        try:
            keeper.start(heartbeat)

            assert callback_done.wait(2)
            _wait_until(lambda: not keeper.is_alive(heartbeat))
            assert heartbeat.finished == 1
            assert callback_elapsed[0] < 0.2
        finally:
            heartbeat.request_stop()
            keeper.close(heartbeat)
