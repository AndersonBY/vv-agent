import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, Event, current_thread
from time import monotonic

import pytest

from vv_agent.session import children, kernel
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import LeaseLost
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall

from .helpers import fixture_child_spec, fixture_content_reader, wrap_fixture_store
from .test_recovery_matrix import records_of, runtime, start


@function_tool
def child_task() -> str:
    raise AssertionError("the child task must run through its session")


@contextmanager
def no_wait_heartbeat_store(database, *, clock=None):
    with SQLiteStore.standalone(str(database), clock=clock) as store:
        # Keep lock-contention tests within the heartbeat retry budget.
        store.configure_input_admission(allow_unsigned=True)
        store.connection.execute("PRAGMA busy_timeout=0")
        yield store


def test_sqlite_store_uses_default_and_configured_busy_timeouts(tmp_path):
    default_database = tmp_path / "default-timeouts.sqlite"
    configured_database = tmp_path / "configured-timeouts.sqlite"

    with SQLiteStore.standalone(str(default_database)) as store:
        assert store.connection.execute("PRAGMA busy_timeout").fetchone() == (5000,)
        with store.heartbeat_store() as heartbeat:
            assert heartbeat.connection.execute("PRAGMA busy_timeout").fetchone() == (250,)

    with SQLiteStore.standalone(
        str(configured_database),
        busy_timeout_ms=8000,
        heartbeat_busy_timeout_ms=125,
    ) as store:
        assert store.connection.execute("PRAGMA busy_timeout").fetchone() == (8000,)
        with store.heartbeat_store() as heartbeat:
            assert heartbeat.connection.execute("PRAGMA busy_timeout").fetchone() == (125,)


def test_drive_store_factory_preserves_sqlite_timeouts_worker_and_admission(tmp_path):
    database = tmp_path / "drive-factory.sqlite"
    with SQLiteStore.standalone(str(database), busy_timeout_ms=8000, heartbeat_busy_timeout_ms=125) as store:
        store.install_schema()
        store.configure_worker(
            level=3,
            handler_versions=("1", "2", "3"),
            capabilities=("child-admission",),
            all_tenants=True,
        )
        store.configure_input_admission(allow_unsigned=True)

        with store.drive_store_factory()() as drive:
            assert drive.connection.execute("PRAGMA busy_timeout").fetchone() == (8000,)
            assert drive.heartbeat_busy_timeout_ms == 125
            with drive.heartbeat_store() as heartbeat:
                assert heartbeat.connection.execute("PRAGMA busy_timeout").fetchone() == (125,)
            assert drive.worker_supports.max_level == 3
            assert drive.worker_supports.handler_versions == ("1", "2", "3")
            assert drive.worker_supports.capabilities == ("child-admission",)
            assert drive._ledger_integrity_mode() == "hash-only"


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
            with writer.heartbeat_store() as heartbeat:
                connected.wait(timeout=5)
                return (
                    heartbeat.connection.execute("PRAGMA journal_mode").fetchone()[0],
                    heartbeat.connection.execute("PRAGMA synchronous").fetchone()[0],
                    heartbeat.connection.execute("PRAGMA busy_timeout").fetchone()[0],
                )

        with ThreadPoolExecutor(max_workers=3) as pool:
            results = [future.result(timeout=10) for future in [pool.submit(open_heartbeat, index) for index in range(3)]]
        writer.connection.execute("ROLLBACK")

    assert results == [("wal", 2, 250)] * 3
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
            assert store.connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
            store.install_schema()
            store.install_schema()
            assert store.connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
            assert store.connection.execute("PRAGMA synchronous").fetchone() == (2,)
            assert store.connection.execute("PRAGMA busy_timeout").fetchone() == (5000,)

    assert sum(statement.strip().lower() == "pragma journal_mode=wal" for statement in statements) == 1


@pytest.mark.sqlite_file
def test_three_concurrent_long_child_tasks_survive_shared_sqlite_write_lock(store, database):
    parent_ids = [f"parent-{index}" for index in range(3)]
    child_ids = []

    def parent_runtime(prefix):
        child_specs = tuple(fixture_child_spec(store, prefix, f"{prefix}-child-{index}") for index in range(2))
        return runtime(
            database,
            [LLMResponse("", [ToolCall("child-call", "child_task", {})]), LLMResponse("parent done")],
            [child_task],
            children={"child_task": lambda _plan: tuple(children.ChildSession(spec, "work") for spec in child_specs)},
        )

    for parent_id in parent_ids:
        start(store, parent_id)
        drive(store, parent_id, runtime=parent_runtime(parent_id))
        parked = next(row.record for row in read_state(store, parent_id)[1] if row.record.kind == "op_parked")
        child_ids.extend(handle["session_id"] for handle in children.child_handles(parked.payload["handle"]))

    assert len(parent_ids) == 3 and len(child_ids) == 6
    started = [Event() for _ in child_ids]
    release_models = Event()
    model_calls = {session_id: 0 for session_id in child_ids}

    def run_child(index, session_id):
        def response(_request):
            model_calls[session_id] += 1
            started[index].set()
            assert release_models.wait(30)
            return LLMResponse("child done")

        with SQLiteStore.standalone(str(database)) as child_store:
            child_store.configure_input_admission(allow_unsigned=True)
            drive(wrap_fixture_store(child_store), session_id, runtime=runtime(database, [response]))

    with ThreadPoolExecutor(max_workers=len(child_ids)) as pool:
        futures = [pool.submit(run_child, index, session_id) for index, session_id in enumerate(child_ids)]
        deadline = monotonic() + 5
        while not all(event.is_set() for event in started) and monotonic() < deadline:
            Event().wait(0.01)
        assert all(event.is_set() for event in started), [
            future.exception() if future.done() else "running" for future in futures
        ]
        with SQLiteStore.standalone(str(database), busy_timeout_ms=30000) as locker, locker.atomic():
            locker._rows("UPDATE sk_session SET next_drive_ms=next_drive_ms WHERE session_id=%s", (child_ids[0],))
            Event().wait(16)
        release_models.set()
        for future in futures:
            future.result(timeout=15)

    assert model_calls == {session_id: 1 for session_id in child_ids}
    assert all(any(row.record.kind == "turn_ended" for row in read_state(store, session_id)[1]) for session_id in child_ids)


@pytest.mark.sqlite_file
def test_file_heartbeat_recovers_from_sqlite_lock_without_duplicate_model_call(store, database, monkeypatch):
    wrapped = wrap_fixture_store(store)
    start(wrapped)
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
    rt = runtime(
        database,
        [answer],
        ttl_ms=2000,
        heartbeat_seconds=0.04,
        content_reader=fixture_content_reader(wrapped, "s"),
    )
    rt.heartbeat_store = lambda: no_wait_heartbeat_store(database, clock=store.clock)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(drive, wrapped, "s", runtime=rt)
        try:
            if not write_lock_held.wait(5):
                future.result(timeout=0)
            assert write_lock_held.is_set()
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
def test_persistent_sqlite_lock_exhausts_retry_budget_as_lease_lost(store, database, monkeypatch):
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
        lambda: no_wait_heartbeat_store(database, clock=store.clock),
        300,
        0.04,
        now_ms=store._now(),
    )
    with SQLiteStore.standalone(str(database)) as locker, locker.atomic():
        heartbeat.start()
        assert heartbeat.ready.wait(5)

    assert len(lock_failures) >= 2
    assert heartbeat.lost
    assert isinstance(heartbeat.heartbeat_error, LeaseLost)
    cause = heartbeat.heartbeat_error
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert isinstance(cause, sqlite3.OperationalError)
    assert "database is locked" in str(cause)
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
        lambda: no_wait_heartbeat_store(database, clock=store.clock),
        4000,
        0.04,
        now_ms=store._now(),
    )
    heartbeat.start()
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
        lambda: no_wait_heartbeat_store(database, clock=store.clock),
        4000,
        0.04,
        now_ms=store._now(),
    )
    heartbeat.start()
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
        lambda: no_wait_heartbeat_store(database, clock=store.clock),
        4000,
        0.04,
        now_ms=store._now(),
    )
    with SQLiteStore.standalone(str(database)) as locker, locker.atomic():
        heartbeat.start()
        assert retried.wait(5)
        started = monotonic()
        heartbeat.close()
        assert monotonic() - started < 2

    assert not heartbeat.lease_keeper.is_alive(heartbeat)
    assert not heartbeat.lost
    assert len(attempts) >= 2


def test_memory_database_role_is_rejected():
    with pytest.raises(ValueError, match=r"requires a file path.*:memory:.*not supported") as caught:
        SQLiteStore(":memory:")
    assert type(caught.value) is ValueError


@pytest.mark.parametrize("owner", ["standalone", "driver"])
def test_close_waits_for_inflight_read_in_another_thread(owner, tmp_path):
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
                manager = SQLiteStore.standalone(sys.argv[2])
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
            str(tmp_path / "closing.sqlite"),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
