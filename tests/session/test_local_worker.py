"""Local workers own scans, drives, and ephemeral SQLite lifecycles."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Event
from threading import enumerate as thread_enumerate
from time import monotonic
from typing import cast

import pytest

from vv_agent import Agent, RunConfig
from vv_agent.model import ScriptedModelProvider
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import Lease, SessionStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.worker import DriveCallback, LocalWorker, WorkerSupport


def _wait_for(predicate, timeout=3.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        Event().wait(0.01)
    raise AssertionError("condition was not met before timeout")


def _driver(tmp_path: Path) -> tuple[SessionDriver, Path]:
    database = tmp_path / "worker.sqlite3"
    provider = LocalKeyProvider(b"w" * 32, state_path=tmp_path / "keys.sqlite3")
    driver = SessionDriver(str(database), content_key_provider=provider)
    driver.store.configure_input_admission(allow_unsigned=True)
    return driver, database


def _start_session(driver: SessionDriver, session_id: str) -> None:
    driver.create(session_id, ".")
    driver.push(session_id, InboxItem("initial", "user", {"content": "go"}))


def _defer_for_test(processed: set[str]) -> DriveCallback:
    def drive(store: SessionStore, session_id: str, *, lease: Lease) -> None:
        try:
            store.defer_drive(lease, retry_after_ms=60_000)
        finally:
            assert store.release(lease)
        processed.add(session_id)

    return drive


def _support(database: Path, drive: DriveCallback, opened: list[SQLiteStore] | None = None) -> WorkerSupport:
    @contextmanager
    def open_store():
        with SQLiteStore.standalone(str(database)) as store:
            store.configure_input_admission(allow_unsigned=True)
            if opened is not None:
                opened.append(store)
            yield store

    return WorkerSupport(open_store=open_store, drive=drive)


def test_default_session_driver_owns_temporary_sqlite_file():
    driver = SessionDriver()
    assert driver.path is not None
    path = Path(driver.path)
    assert path.name == "sessions.sqlite3"
    assert path.is_file()
    driver.create("temporary", ".")
    driver.close()
    assert not path.exists()
    assert not path.parent.exists()


def test_explicit_memory_store_is_rejected():
    with pytest.raises(ValueError, match="file path"):
        SessionDriver(":memory:")


def test_one_worker_handles_two_runnable_sessions(tmp_path):
    driver, database = _driver(tmp_path)
    processed: set[str] = set()
    try:
        _start_session(driver, "a")
        _start_session(driver, "b")
        worker = LocalWorker(_support(database, _defer_for_test(processed)), scan_interval=0.02)
        try:
            worker.start()
            _wait_for(lambda: processed == {"a", "b"})
            assert worker.last_error is None
        finally:
            assert worker.close(timeout=2)
    finally:
        driver.close()


def test_scan_and_drive_use_different_store_instances(tmp_path):
    driver, database = _driver(tmp_path)
    opened: list[SQLiteStore] = []
    drive_stores: list[SessionStore] = []
    processed: set[str] = set()
    defer = _defer_for_test(processed)

    def drive(store: SessionStore, session_id: str, *, lease: Lease) -> None:
        drive_stores.append(store)
        defer(store, session_id, lease=lease)

    try:
        _start_session(driver, "isolated")
        worker = LocalWorker(_support(database, drive, opened), scan_interval=0.02)
        try:
            worker.start()
            _wait_for(lambda: processed == {"isolated"})
        finally:
            assert worker.close(timeout=2)
        assert len(opened) >= 2
        assert drive_stores[0] is not opened[0]
    finally:
        driver.close()


def test_periodic_scan_recovers_after_wake_is_not_sent(tmp_path):
    driver, database = _driver(tmp_path)
    processed: set[str] = set()
    try:
        worker = LocalWorker(_support(database, _defer_for_test(processed)), scan_interval=0.02)
        try:
            worker.start()
            Event().wait(0.08)  # Let the initial wake-only scan observe an empty queue.
            _start_session(driver, "lost-wake")
            _wait_for(lambda: processed == {"lost-wake"})
        finally:
            assert worker.close(timeout=2)
    finally:
        driver.close()


def test_worker_stop_and_join_leave_no_background_thread(tmp_path):
    driver, database = _driver(tmp_path)
    try:
        worker = LocalWorker(_support(database, _defer_for_test(set())), scan_interval=0.02, name="test-session-worker")
        worker.start()
        _wait_for(lambda: worker.running)
        worker.stop()
        assert worker.join(2)
        assert not worker.running
        assert not any(thread.name == "test-session-worker" for thread in thread_enumerate())
    finally:
        driver.close()


def test_session_driver_schedules_run_handle_on_worker(tmp_path, monkeypatch):
    driver, _database = _driver(tmp_path)
    called = Event()
    try:
        driver.create("scheduled", ".")
        provider = ScriptedModelProvider.new("test", "m", [])
        handle = driver.start(
            "scheduled",
            Agent("test", "Answer.", model="m"),
            RunConfig(model_provider=provider, session_memory_enabled=False),
            "go",
            autostart=False,
        )

        def run_on_worker(*_args):
            called.set()

        monkeypatch.setattr(driver._worker, "_periodic_scan", run_on_worker)
        handle.start()
        assert called.wait(3)
        assert driver._worker.started
        assert not any(thread.name == "session-surface-driver" for thread in thread_enumerate())
    finally:
        driver.close()


def test_local_worker_declares_compatibility_support(tmp_path):
    driver, database = _driver(tmp_path)
    seen = []

    def drive(store: SessionStore, session_id: str, *, lease: Lease) -> None:
        sqlite_store = cast(SQLiteStore, store)
        seen.append(
            (
                sqlite_store.worker_level,
                sqlite_store.worker_supports.handler_versions,
                sqlite_store.worker_supports.capabilities,
            )
        )
        store.defer_drive(lease, retry_after_ms=60_000)
        assert store.release(lease)

    support = replace(
        _support(database, drive),
        level=2,
        handler_versions=("handler/v2",),
        capabilities=("env:remote",),
    )
    try:
        _start_session(driver, "support")
        worker = LocalWorker(support, scan_interval=0.02)
        try:
            worker.start()
            _wait_for(lambda: bool(seen))
        finally:
            assert worker.close(timeout=2)
        assert seen == [(2, ("handler/v2",), ("env:remote",))]
    finally:
        driver.close()
