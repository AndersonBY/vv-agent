import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, current_thread
from time import monotonic

import pytest

from vv_agent.session import children, kernel
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.records import SessionSpec
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall

from .test_recovery_matrix import records_of, runtime, start


@function_tool
def child_task() -> str:
    raise AssertionError("the child task must run through its session")


@contextmanager
def no_wait_heartbeat_store(database):
    with SQLiteStore.standalone(str(database)) as store:
        # Keep lock-contention tests within the heartbeat retry budget.
        store.connection.execute("PRAGMA busy_timeout=0")
        yield store


def test_new_sqlite_database_uses_wal(tmp_path):
    database = tmp_path / "new.sqlite"

    with SQLiteStore.standalone(str(database)) as store:
        assert store.connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        store.install_schema()


def test_sqlite_store_uses_distinct_configured_heartbeat_busy_timeout(tmp_path):
    database = tmp_path / "timeouts.sqlite"

    with SQLiteStore.standalone(str(database), busy_timeout_ms=8000, heartbeat_busy_timeout_ms=125) as store:
        assert store.connection.execute("PRAGMA busy_timeout").fetchone() == (8000,)
        with store.heartbeat_store() as heartbeat:
            assert heartbeat.connection.execute("PRAGMA busy_timeout").fetchone() == (125,)


@pytest.mark.sqlite_file
def test_three_concurrent_long_child_tasks_survive_shared_sqlite_write_lock(store, database):
    parent_ids = [f"parent-{index}" for index in range(3)]
    child_ids = []

    def parent_runtime(prefix):
        return runtime(
            database,
            [LLMResponse("", [ToolCall("child-call", "child_task", {})]), LLMResponse("parent done")],
            [child_task],
            children={
                "child_task": lambda _plan: tuple(
                    children.ChildSession(SessionSpec(f"{prefix}-child-{index}", "test", "/tmp"), "work") for index in range(2)
                )
            },
        )

    for parent_id in parent_ids:
        start(store, parent_id)
        drive(store, parent_id, runtime=parent_runtime(parent_id))
        parked = next(row.record for row in read_state(store, parent_id)[1] if row.record.kind == "op_parked")
        child_ids.extend(handle["session_id"] for handle in children.child_handles(parked.payload["handle"]))

    started = [Event() for _ in child_ids]
    release_models = Event()

    def run_child(index, sid):
        def response(_request):
            started[index].set()
            assert release_models.wait(30)
            return LLMResponse("child done")

        with SQLiteStore.standalone(str(database)) as child_store:
            drive(child_store, sid, runtime=runtime(database, [response]))

    with ThreadPoolExecutor(max_workers=len(child_ids)) as pool:
        futures = [pool.submit(run_child, index, sid) for index, sid in enumerate(child_ids)]
        assert all(event.wait(5) for event in started)
        with SQLiteStore.standalone(str(database), busy_timeout_ms=30000) as locker, locker.atomic():
            locker._rows("UPDATE sk_session SET next_drive_ms=next_drive_ms WHERE session_id=%s", (child_ids[0],))
            Event().wait(16)
        release_models.set()
        for future in futures:
            future.result(timeout=15)

    assert all(any(row.record.kind == "turn_ended" for row in read_state(store, sid)[1]) for sid in child_ids)


@pytest.mark.sqlite_file
def test_existing_sqlite_heartbeat_connections_open_during_write_transaction(store, database):
    start(store)
    # A non-WAL journal makes an accidental journal_mode rewrite fail immediately
    # while the foreground connection holds SQLite's write lock.
    assert store.connection.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)

    def open_heartbeat(_):
        with SQLiteStore.standalone(str(database)) as heartbeat:
            return heartbeat.connection.execute("SELECT COUNT(*) FROM sk_session").fetchone()

    locker = sqlite3.connect(str(database), isolation_level=None)
    try:
        locker.execute("BEGIN IMMEDIATE")
        locker.execute("UPDATE sk_session SET next_drive_ms=next_drive_ms WHERE session_id='s'")
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = [future.result(timeout=10) for future in [pool.submit(open_heartbeat, index) for index in range(3)]]
    finally:
        locker.rollback()
        locker.close()

    assert results == [(1,), (1,), (1,)]
    assert store.connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)


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
