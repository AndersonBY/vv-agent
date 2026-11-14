import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, Event, current_thread
from time import monotonic

import pytest

from vv_agent.session import kernel
from vv_agent.session.kernel import drive
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse

from .test_recovery_matrix import records_of, runtime, start


@contextmanager
def no_wait_heartbeat_store(database):
    with SQLiteStore.standalone(str(database)) as store:
        # Keep lock-contention tests within the heartbeat retry budget.
        store.connection.execute("PRAGMA busy_timeout=0")
        yield store


@pytest.mark.sqlite_file
def test_existing_database_heartbeat_connections_skip_journal_configuration_under_write_lock(store, database, monkeypatch):
    start(store)
    statements = []
    connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    with SQLiteStore.standalone(str(database)) as writer:
        writer.connection.execute("BEGIN IMMEDIATE")
        statements.clear()
        connected = Barrier(3)

        def open_heartbeat(_):
            with SQLiteStore.standalone(str(database)) as heartbeat:
                connected.wait(timeout=5)
                return (
                    heartbeat.connection.execute("PRAGMA journal_mode").fetchone()[0],
                    heartbeat.connection.execute("PRAGMA synchronous").fetchone()[0],
                    heartbeat.connection.execute("PRAGMA busy_timeout").fetchone()[0],
                )

        with ThreadPoolExecutor(max_workers=3) as pool:
            results = [future.result(timeout=10) for future in [pool.submit(open_heartbeat, index) for index in range(3)]]
        writer.connection.execute("ROLLBACK")

    assert results == [("wal", 2, 5000)] * 3
    assert not [statement for statement in statements if statement.strip().lower() == "pragma journal_mode=wal"]


@pytest.mark.sqlite_file
def test_new_database_initialization_still_configures_wal_once(tmp_path):
    database = tmp_path / "new.sqlite"
    statements = []
    connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sqlite3, "connect", traced_connect)
        with SQLiteStore.standalone(str(database)) as store:
            store.install_schema()
            assert store.connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
            assert store.connection.execute("PRAGMA synchronous").fetchone() == (2,)
            assert store.connection.execute("PRAGMA busy_timeout").fetchone() == (5000,)

    assert sum(statement.strip().lower() == "pragma journal_mode=wal" for statement in statements) == 1


@pytest.mark.sqlite_file
def test_file_heartbeat_recovers_from_sqlite_lock_without_duplicate_model_call(store, database, monkeypatch):
    start(store)
    write_lock_held, release_write_lock = Event(), Event()
    retried, renewed_after_lock = Event(), Event()
    model_calls, lock_failures = [], []
    renew = SQLiteStore.renew

    def observed(value, lease, *, ttl_ms):
        try:
            result = renew(value, lease, ttl_ms=ttl_ms)
        except sqlite3.OperationalError as error:
            if current_thread().name == "session-heartbeat" and "database is locked" in str(error):
                lock_failures.append(1)
                if len(lock_failures) >= 2:
                    retried.set()
            raise
        if current_thread().name == "session-heartbeat" and write_lock_held.is_set():
            renewed_after_lock.set()
        return result

    def answer(_request):
        model_calls.append(1)
        with SQLiteStore.standalone(str(database)) as locker, locker.atomic():
            write_lock_held.set()
            assert release_write_lock.wait(5)
        assert renewed_after_lock.wait(5)
        return LLMResponse("done")

    monkeypatch.setattr(SQLiteStore, "renew", observed)
    rt = runtime(database, [answer], ttl_ms=2000, heartbeat_seconds=0.04)
    rt.heartbeat_store = lambda: no_wait_heartbeat_store(database)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(drive, store, "s", runtime=rt)
        try:
            assert write_lock_held.wait(5)
            assert retried.wait(5)
            assert model_calls == [1]
        finally:
            release_write_lock.set()
        future.result(timeout=10)

    assert renewed_after_lock.is_set()
    assert len(lock_failures) >= 2
    assert model_calls == [1] and not rt.llm.steps
    assert [record.payload["status"] for record in records_of(store, "turn_ended")] == ["completed"]


@pytest.mark.sqlite_file
def test_persistent_sqlite_lock_exhausts_retry_budget_as_transient_timeout(store, database, monkeypatch):
    start(store)
    lease = store.acquire("s", owner="heartbeat", ttl_ms=500)
    assert lease is not None
    lock_failures = []
    renew = SQLiteStore.renew

    def observed(value, current, *, ttl_ms):
        try:
            return renew(value, current, ttl_ms=ttl_ms)
        except sqlite3.OperationalError as error:
            if current_thread().name == "session-heartbeat" and "database is locked" in str(error):
                lock_failures.append(1)
            raise

    monkeypatch.setattr(SQLiteStore, "renew", observed)
    heartbeat = kernel._LeaseHeartbeat(
        lease,
        lambda: no_wait_heartbeat_store(database),
        300,
        0.04,
        now_ms=store._now(),
    )
    with SQLiteStore.standalone(str(database)) as locker, locker.atomic():
        heartbeat.thread.start()
        assert heartbeat.ready.wait(5)

    assert len(lock_failures) >= 2
    assert heartbeat.lost
    assert isinstance(heartbeat.heartbeat_error, TimeoutError)
    assert isinstance(heartbeat.heartbeat_error.__cause__, sqlite3.OperationalError)
    assert "database is locked" in str(heartbeat.heartbeat_error.__cause__)
    heartbeat.close()


@pytest.mark.sqlite_file
def test_sqlite_table_lock_message_is_retried(store, database, monkeypatch):
    start(store)
    lease = store.acquire("s", owner="heartbeat", ttl_ms=4000)
    assert lease is not None
    attempts, renewed = [], Event()
    renew = SQLiteStore.renew

    def observed(value, current, *, ttl_ms):
        attempts.append(1)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("database table is locked: sk_session")
        result = renew(value, current, ttl_ms=ttl_ms)
        renewed.set()
        return result

    monkeypatch.setattr(SQLiteStore, "renew", observed)
    heartbeat = kernel._LeaseHeartbeat(
        lease,
        lambda: no_wait_heartbeat_store(database),
        4000,
        0.04,
        now_ms=store._now(),
    )
    heartbeat.thread.start()
    assert renewed.wait(5)
    heartbeat.close()

    assert len(attempts) >= 2
    assert not heartbeat.lost


@pytest.mark.sqlite_file
@pytest.mark.parametrize(
    "message",
    ["disk I/O error", "no such table: sk_session", "database disk image is malformed"],
)
def test_non_lock_sqlite_operational_error_fails_without_retry(store, database, monkeypatch, message):
    start(store)
    lease = store.acquire("s", owner="heartbeat", ttl_ms=4000)
    assert lease is not None
    attempts = []

    def failing(_store, _lease, *, ttl_ms):
        attempts.append(ttl_ms)
        raise sqlite3.OperationalError(message)

    monkeypatch.setattr(SQLiteStore, "renew", failing)
    heartbeat = kernel._LeaseHeartbeat(
        lease,
        lambda: no_wait_heartbeat_store(database),
        4000,
        0.04,
        now_ms=store._now(),
    )
    heartbeat.thread.start()
    assert heartbeat.ready.wait(5)

    assert attempts == [4000]
    assert heartbeat.lost
    assert isinstance(heartbeat.heartbeat_error, sqlite3.OperationalError)
    assert str(heartbeat.heartbeat_error) == message
    heartbeat.close()


@pytest.mark.sqlite_file
def test_close_interrupts_sqlite_lock_retry_wait(store, database, monkeypatch):
    start(store)
    lease = store.acquire("s", owner="heartbeat", ttl_ms=4000)
    assert lease is not None
    attempts, retried = [], Event()
    renew = SQLiteStore.renew

    def observed(value, current, *, ttl_ms):
        try:
            return renew(value, current, ttl_ms=ttl_ms)
        except sqlite3.OperationalError as error:
            if current_thread().name == "session-heartbeat" and "database is locked" in str(error):
                attempts.append(1)
                if len(attempts) >= 2:
                    retried.set()
            raise

    monkeypatch.setattr(SQLiteStore, "renew", observed)
    heartbeat = kernel._LeaseHeartbeat(
        lease,
        lambda: no_wait_heartbeat_store(database),
        4000,
        0.04,
        now_ms=store._now(),
    )
    with SQLiteStore.standalone(str(database)) as locker, locker.atomic():
        heartbeat.thread.start()
        assert retried.wait(5)
        started = monotonic()
        heartbeat.close()
        assert monotonic() - started < 2

    assert not heartbeat.thread.is_alive()
    assert not heartbeat.lost
    assert len(attempts) >= 2


def test_memory_heartbeat_borrows_owner_connection_and_owner_closes_it(monkeypatch):
    renewed = Event()
    heartbeat_threads = set()
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        start(store)
        renew = store.renew

        def observe_renew(lease, *, ttl_ms):
            result = renew(lease, ttl_ms=ttl_ms)
            if current_thread().name == "session-heartbeat":
                heartbeat_threads.add(current_thread())
                renewed.set()
            return result

        def answer(_request):
            assert renewed.wait(2)
            return LLMResponse("done")

        monkeypatch.setattr(store, "renew", observe_renew)
        rt = runtime(store, [answer], heartbeat_seconds=0.01)
        drive(store, "s", runtime=rt)
        assert renewed.is_set()
        assert records_of(store, "turn_ended")[0].payload["status"] == "completed"
        assert all(not thread.is_alive() for thread in heartbeat_threads)
        assert store.connection.execute("SELECT 1").fetchone() == (1,)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        store.connection.execute("SELECT 1")


@pytest.mark.parametrize("owner", ["standalone", "driver"])
def test_close_waits_for_inflight_read_in_another_thread(owner):
    import subprocess
    import sys
    import textwrap

    # Isolate the native SQLite shutdown race so a regression cannot kill pytest.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import sys
            from threading import Event, Thread
            from vv_agent.session.sqlite import SQLiteStore
            from vv_agent.session.surfaces import SessionDriver

            if sys.argv[1] == "standalone":
                manager = SQLiteStore.standalone(":memory:")
                store = manager.__enter__()
                close_owner = lambda: manager.__exit__(None, None, None)
            else:
                driver = SessionDriver()
                store = driver.store
                close_owner = driver.close
            entered, release, closing, closed = (Event() for _ in range(4))
            rows, errors = [], []
            def slow():
                entered.set()
                assert release.wait(3)
                return 7
            store.connection.create_function("slow", 0, slow)
            def read():
                try:
                    rows.extend(store._rows("SELECT slow()"))
                except BaseException as error:
                    errors.append(error)
            def close():
                closing.set()
                close_owner()
                closed.set()
            reader = Thread(target=read)
            closer = Thread(target=close)
            reader.start()
            assert entered.wait(2)
            closer.start()
            assert closing.wait(2)
            assert not closed.wait(0.05)
            release.set()
            reader.join(3)
            closer.join(3)
            assert not reader.is_alive() and not closer.is_alive()
            assert rows == [(7,)] and not errors
            assert closed.is_set()
        """),
            owner,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
