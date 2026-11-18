"""LocalWorker coordinator/drive role separation and work ownership filters."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event, current_thread
from threading import enumerate as thread_enumerate
from time import monotonic, sleep
from typing import Any, cast

import pytest

from vv_agent.kernel._requirements import SessionRequirements
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.records import SessionSpec
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import Lease, SessionStore
from vv_agent.session.worker import LocalWorker, WorkerSupport, _DriveCompletion

from .helpers import content_ref, item, principal, record


def _wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        Event().wait(0.01)
    raise AssertionError("condition was not met before timeout")


@contextmanager
def _open_store(database: str | Path) -> Iterator[SQLiteStore | PostgresStore]:
    manager = SQLiteStore.standalone(str(database)) if isinstance(database, Path) else PostgresStore.standalone(database)
    with manager as store:
        store.configure_input_admission(allow_unsigned=True)
        yield store


@contextmanager
def _installed_database(path: Path) -> Iterator[Path]:
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
    yield path


def _start(store: SessionStore, session_id: str, *, consumers: tuple[str, ...] = (), push: bool = False) -> None:
    with store.atomic() as tx:
        tx.create(
            SessionSpec(
                session_id,
                principal(),
                content_ref("workspace"),
                attributes=content_ref(f"attributes/{session_id}"),
            ),
            consumers=consumers,
        )
        if push:
            tx.push(session_id, item("initial", actor=principal("test")))


def _lease_row(store: Any, session_id: str) -> tuple[int, str | None, int | None, int]:
    return store._one(
        "SELECT lease_epoch,lease_owner,lease_until_ms,drive_retry_at_ms FROM sk_session WHERE tenant_id=%s AND session_id=%s",
        (store.tenant_id, session_id),
    )


def _consumer_rows(store: Any, session_id: str) -> dict[str, int]:
    return dict(
        store._rows(
            "SELECT consumer,last_seq FROM sk_consumer WHERE tenant_id=%s AND session_id=%s ORDER BY consumer",
            (store.tenant_id, session_id),
        )
    )


@pytest.mark.sqlite_file
def test_observer_only_starts_without_drive_thread_or_projection_claim(tmp_path: Path) -> None:
    database = tmp_path / "observer.sqlite"
    with _installed_database(database) as database, _open_store(database) as store:
        _start(store, "observer", consumers=("events", "traces"))

        opens = 0
        closes = 0
        observed = 0
        observed_event = Event()

        @contextmanager
        def open_store():
            nonlocal opens, closes
            opens += 1
            try:
                with _open_store(database) as worker_store:
                    worker_store.claim_runnable = lambda *_args, **_kwargs: pytest.fail("observer-only worker must not claim")
                    yield worker_store
            finally:
                closes += 1

        def observe(_store: SessionStore) -> None:
            nonlocal observed
            observed += 1
            observed_event.set()

        worker = LocalWorker(
            WorkerSupport(
                open_store=open_store,
                drive=None,
                project=None,
                observe=observe,
                projection_consumers=(),
            ),
            scan_interval=0.01,
            name="observer-only",
        )
        try:
            worker.start()
            assert observed_event.wait(3)
            _wait_for(lambda: observed >= 3)
            worker.wait_idle(1)
            assert worker.started
            assert not any(thread.name.startswith("observer-only-drive") for thread in thread_enumerate())
            assert _consumer_rows(store, "observer") == {"events": 0, "traces": 0}
        finally:
            assert worker.close(timeout=2)
        assert opens == closes


@pytest.mark.sqlite_file
def test_post_observation_starts_worker_and_runs_fifo_on_coordinator(tmp_path: Path) -> None:
    database = tmp_path / "commands.sqlite"
    with _installed_database(database):
        entries: list[tuple[int, int, int, str]] = []

        @contextmanager
        def open_store():
            with _open_store(database) as worker_store:
                yield worker_store

        worker = LocalWorker(
            WorkerSupport(open_store=open_store, drive=None, project=None, projection_consumers=()),
            scan_interval=0.2,
            name="observation-commands",
        )

        def first(store: SessionStore) -> None:
            entries.append((1, current_thread().ident or 0, id(store), current_thread().name))
            sleep(0.03)

        def second(store: SessionStore) -> None:
            entries.append((2, current_thread().ident or 0, id(store), current_thread().name))

        try:
            worker.post_observation(first)
            worker.post_observation(second)
            worker.wait_idle(2)
            assert [entry[0] for entry in entries] == [1, 2]
            assert len({entry[1] for entry in entries}) == 1
            assert {entry[3] for entry in entries} == {"observation-commands"}
            assert worker.started
        finally:
            assert worker.close(timeout=2)
        assert not any(thread.name.startswith("observation-commands-drive") for thread in thread_enumerate())


@pytest.mark.sqlite_file
def test_blocked_drive_does_not_stop_observer_or_share_its_store(tmp_path: Path) -> None:
    database = tmp_path / "blocked-drive.sqlite"
    with _installed_database(database) as database, _open_store(database) as store:
        _start(store, "blocked", push=True)
        _start(store, "late-record")
        entered = Event()
        release = Event()
        processed = Event()
        late_record_seen = Event()
        observe_calls = 0
        observer_roles: list[tuple[str, int, int]] = []
        drive_roles: list[tuple[str, int, int]] = []

        @contextmanager
        def open_store():
            with _open_store(database) as worker_store:
                worker_store._test_open_thread = current_thread().ident
                worker_store._test_open_role = (
                    "drive" if current_thread().name.startswith("observer-progress-drive") else "coordinator"
                )
                yield worker_store

        def observe(worker_store: SessionStore) -> None:
            nonlocal observe_calls
            observed_store = cast(Any, worker_store)
            observe_calls += 1
            if worker_store.read("late-record", limit=1).head_seq > 1:
                late_record_seen.set()
            observer_roles.append(
                (
                    observed_store._test_open_role,
                    observed_store._test_open_thread,
                    id(worker_store),
                )
            )

        def drive(store: SessionStore, session_id: str, *, lease) -> None:
            drive_store = cast(Any, store)
            drive_roles.append(
                (
                    drive_store._test_open_role,
                    drive_store._test_open_thread,
                    id(store),
                )
            )
            entered.set()
            assert release.wait(3)
            try:
                store.defer_drive(lease, retry_after_ms=60_000)
            finally:
                assert store.release(lease)
            processed.set()

        worker = LocalWorker(
            WorkerSupport(open_store=open_store, drive=drive, observe=observe, projection_consumers=()),
            scan_interval=0.01,
            name="observer-progress",
        )
        try:
            worker.start()
            assert entered.wait(3)
            late_lease = store.acquire("late-record", owner="host-approval", ttl_ms=10_000)
            assert late_lease is not None
            try:
                with store.atomic() as tx:
                    tx.append(
                        "late-record",
                        lease=late_lease,
                        expected_seq=store.read("late-record", limit=1).head_seq,
                        commit_id="late-approval",
                        records=(record("turn_started", sid="late-record"),),
                    )
            finally:
                assert store.release(late_lease)
            assert late_record_seen.wait(3)
            _wait_for(lambda: observe_calls >= 3)
            assert drive_roles[0][2] not in {role[2] for role in observer_roles}
            assert drive_roles[0][0] == "drive"
            assert {role[0] for role in observer_roles} == {"coordinator"}
            assert all(role[1] == drive_roles[0][1] for role in drive_roles)
            assert all(role[1] != drive_roles[0][1] for role in observer_roles)
        finally:
            release.set()
            assert processed.wait(3)
            assert worker.wait_idle(3) is None
            assert worker.close(timeout=2)
        assert not any(thread.name.startswith("observer-progress-drive") for thread in thread_enumerate())


@pytest.mark.sqlite_file
def test_close_timeout_does_not_force_release_or_abandon_running_drive(tmp_path: Path) -> None:
    database = tmp_path / "close-timeout.sqlite"
    with _installed_database(database) as database, _open_store(database) as store:
        _start(store, "close-timeout", push=True)
        entered = Event()
        finish_drive = Event()
        released = Event()

        @contextmanager
        def open_store():
            with _open_store(database) as worker_store:
                yield worker_store

        def drive(store: SessionStore, session_id: str, *, lease) -> None:
            entered.set()
            assert finish_drive.wait(3)
            try:
                store.defer_drive(lease, retry_after_ms=60_000)
            finally:
                assert store.release(lease)
            released.set()

        worker = LocalWorker(
            WorkerSupport(open_store=open_store, drive=drive, projection_consumers=()),
            scan_interval=0.01,
            name="close-timeout",
        )
        try:
            worker.start()
            assert entered.wait(3)
            held = _lease_row(store, "close-timeout")
            assert held[1] is not None
            assert worker.close(timeout=0.05) is False
            assert _lease_row(store, "close-timeout") == held
        finally:
            finish_drive.set()
            assert released.wait(3)
            assert worker.close(timeout=3)
        assert _lease_row(store, "close-timeout")[:3] == (1, None, None)
        assert not any(
            thread.name == "close-timeout" or thread.name.startswith("close-timeout-drive") for thread in thread_enumerate()
        )


@pytest.mark.sqlite_file
def test_stop_rejects_new_observation_while_blocked_drive_finishes(tmp_path: Path) -> None:
    database = tmp_path / "stop-command-race.sqlite"
    with _installed_database(database) as database, _open_store(database) as store:
        _start(store, "stop-command-race", push=True)
        drive_blocked = Barrier(2)
        finish_drive = Event()
        released = Event()
        accepted_called = Event()
        callback_called = Event()

        @contextmanager
        def open_store():
            with _open_store(database) as worker_store:
                yield worker_store

        def drive(store: SessionStore, session_id: str, *, lease) -> None:
            drive_blocked.wait(3)
            assert finish_drive.wait(3)
            try:
                store.defer_drive(lease, retry_after_ms=60_000)
            finally:
                assert store.release(lease)
            released.set()

        worker = LocalWorker(
            WorkerSupport(open_store=open_store, drive=drive, projection_consumers=()),
            scan_interval=0.01,
            name="stop-command-race",
        )
        try:
            worker.start()
            drive_blocked.wait(3)
            worker.post_observation(lambda _store: accepted_called.set())
            worker.stop()
            assert any(thread.name == "stop-command-race" for thread in thread_enumerate())
            with pytest.raises(RuntimeError, match="stopped"):
                worker.post_observation(lambda _store: callback_called.set())
        finally:
            finish_drive.set()
            assert released.wait(3)
            assert accepted_called.wait(3)
            assert worker.wait_idle(3) is None
            assert worker.close(timeout=3)
        assert not callback_called.is_set()
        assert not any(
            thread.name == "stop-command-race" or thread.name.startswith("stop-command-race-drive")
            for thread in thread_enumerate()
        )


@pytest.mark.sqlite_file
def test_stop_fails_accepted_observation_when_coordinator_store_cannot_open() -> None:
    callback_called = Event()

    @contextmanager
    def open_store():
        raise RuntimeError("store unavailable")
        yield  # pragma: no cover

    worker = LocalWorker(
        WorkerSupport(open_store=open_store, drive=None, project=None, projection_consumers=()),
        scan_interval=0.01,
        name="failed-store-observation",
    )
    worker.post_observation(lambda _store: callback_called.set())
    worker.stop()
    assert worker.join(2)
    assert worker.wait_idle(0.5) is None
    assert not callback_called.is_set()
    last_error = worker.last_error
    assert last_error is not None
    errors = last_error.exceptions if isinstance(last_error, ExceptionGroup) else (last_error,)
    assert errors and all(isinstance(error, RuntimeError) for error in errors)


@pytest.mark.sqlite_file
def test_final_cleanup_opener_failure_clears_completed_future_state() -> None:
    future_started = Barrier(2)
    future_completed = Event()

    @contextmanager
    def open_store():
        raise RuntimeError("final cleanup opener failed")
        yield  # pragma: no cover

    worker = LocalWorker(
        WorkerSupport(open_store=open_store, drive=None, project=None, projection_consumers=()),
        scan_interval=0.01,
        name="failed-final-cleanup",
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="failed-final-cleanup-drive")

    def completed_job():
        future_started.wait(3)
        future_completed.set()
        return _DriveCompletion()

    future = executor.submit(completed_job)
    future_started.wait(3)
    assert future_completed.wait(3)
    worker._executor = executor
    worker._inflight = future
    worker._inflight_lease = Lease("session", "local-worker/test", 1, 60_000)
    worker.stop()
    worker._run()

    assert worker.wait_idle(0.5) is None
    assert worker.close(0.5)
    assert worker.last_error is not None
    assert "final cleanup opener failed" in str(worker.last_error)


@pytest.mark.sqlite_file
def test_running_worker_uses_support_extension_without_rewriting_inflight_drive(tmp_path: Path) -> None:
    database = tmp_path / "support-extension.sqlite"
    requirements = SessionRequirements.normalized(
        level=2,
        handler_version="1",
        capabilities=("runtime:new",),
    )
    with _installed_database(database) as database, _open_store(database) as store:
        _start(store, "old-drive", push=True)
        store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        with store.atomic() as tx:
            tx.create(
                SessionSpec(
                    "new-capability",
                    principal(),
                    content_ref("workspace"),
                    attributes=content_ref("attributes/new-capability"),
                    feature_level=2,
                    bootstrap=requirements,
                ),
                consumers=(),
            )
            tx.push(
                "new-capability",
                replace(
                    item("new-input", actor=principal("test")),
                    feature_level=2,
                    requirements=requirements,
                ),
            )

        entered = Event()
        release_old = Event()
        support_seen = Event()
        processed: set[str] = set()
        support_at_drive: dict[str, tuple[tuple[int, tuple[str, ...]], tuple[int, tuple[str, ...]]]] = {}

        @contextmanager
        def open_store():
            with _open_store(database) as worker_store:
                yield worker_store

        worker = LocalWorker(
            WorkerSupport(open_store=open_store, projection_consumers=()),
            scan_interval=0.01,
            name="support-extension",
        )

        def observe(worker_store: SessionStore) -> None:
            worker_store_state = cast(Any, worker_store)
            if worker.support.level == 2 and worker_store_state.worker_level == 2:
                support_seen.set()

        def drive(store: SessionStore, session_id: str, *, lease) -> None:
            drive_store = cast(Any, store)
            before = (drive_store.worker_level, drive_store.worker_supports.capabilities)
            if session_id == "old-drive":
                entered.set()
                assert release_old.wait(3)
            after = (drive_store.worker_level, drive_store.worker_supports.capabilities)
            support_at_drive[session_id] = (before, after)
            try:
                store.defer_drive(lease, retry_after_ms=60_000)
            finally:
                assert store.release(lease)
            processed.add(session_id)

        worker.support = replace(worker.support, observe=observe, drive=drive)
        try:
            worker.start()
            assert entered.wait(3)
            worker.support = replace(
                worker.support,
                level=2,
                capabilities=("runtime:new",),
            )
            assert support_seen.wait(3)
        finally:
            release_old.set()
            _wait_for(lambda: processed == {"old-drive", "new-capability"})
            assert worker.close(timeout=3)

        assert support_at_drive["old-drive"] == ((1, ()), (1, ()))
        assert support_at_drive["new-capability"] == ((2, ("runtime:new",)), (2, ("runtime:new",)))


@pytest.mark.sqlite_file
def test_busy_drive_slot_does_not_claim_another_drive(tmp_path: Path) -> None:
    database = tmp_path / "single-slot.sqlite"
    with _installed_database(database) as database, _open_store(database) as store:
        for session_id in ("a-blocked", "b-waiting", "c-waiting"):
            _start(store, session_id, push=True)
        entered = Event()
        release = Event()
        processed: set[str] = set()
        observe_count = 0
        observed_busy_slot = Event()

        @contextmanager
        def open_store():
            with _open_store(database) as worker_store:
                yield worker_store

        def drive(store: SessionStore, session_id: str, *, lease) -> None:
            if session_id == "a-blocked":
                entered.set()
                assert release.wait(3)
            try:
                store.defer_drive(lease, retry_after_ms=60_000)
            finally:
                assert store.release(lease)
            processed.add(session_id)

        def observe(_store: SessionStore) -> None:
            nonlocal observe_count
            observe_count += 1
            if observe_count >= 3:
                observed_busy_slot.set()

        worker = LocalWorker(
            WorkerSupport(open_store=open_store, drive=drive, observe=observe, projection_consumers=()),
            scan_interval=0.01,
            name="single-slot",
        )
        try:
            worker.start()
            assert entered.wait(3)
            assert observed_busy_slot.wait(3)
            assert _lease_row(store, "a-blocked")[0] == 1
            assert _lease_row(store, "a-blocked")[1] is not None
            assert _lease_row(store, "b-waiting")[:2] == (0, None)
            assert _lease_row(store, "c-waiting")[:2] == (0, None)
        finally:
            release.set()
            _wait_for(lambda: processed == {"a-blocked", "b-waiting", "c-waiting"})
            assert worker.close(timeout=3)


@pytest.mark.sqlite_file
def test_drive_store_open_failure_returns_unhanded_claim_once(tmp_path: Path) -> None:
    database = tmp_path / "open-failure.sqlite"
    with _installed_database(database) as database, _open_store(database) as store:
        _start(store, "open-failure", push=True)
        failed = Event()
        called = Event()

        @contextmanager
        def open_store():
            if current_thread().name.startswith("open-failure-drive"):
                failed.set()
                raise RuntimeError("drive store open failed")
            with _open_store(database) as worker_store:
                yield worker_store

        def drive(*_args, **_kwargs) -> None:
            called.set()

        worker = LocalWorker(
            WorkerSupport(open_store=open_store, drive=drive, projection_consumers=()),
            scan_interval=1,
            name="open-failure",
        )
        try:
            worker.start()
            assert failed.wait(3)
            worker.wait_idle(3)
        finally:
            assert worker.close(timeout=2)
        assert not called.is_set()
        epoch, owner, expires_at_ms, retry_at_ms = _lease_row(store, "open-failure")
        assert epoch == 1 and owner is None and expires_at_ms is None
        assert retry_at_ms > 0


@pytest.mark.sqlite_file
def test_drive_store_close_failure_after_handoff_does_not_release_again(tmp_path: Path) -> None:
    database = tmp_path / "close-failure.sqlite"
    with _installed_database(database) as database, _open_store(database) as store:
        _start(store, "close-failure", push=True)
        released = Event()
        finish_drive = Event()

        @contextmanager
        def open_store():
            with _open_store(database) as worker_store:
                yield worker_store
            if current_thread().name.startswith("close-failure-drive"):
                raise RuntimeError("keeper close failed")

        def drive(store: SessionStore, session_id: str, *, lease) -> None:
            assert store.release(lease)
            released.set()
            assert finish_drive.wait(3)

        worker = LocalWorker(
            WorkerSupport(open_store=open_store, drive=drive, projection_consumers=()),
            scan_interval=1,
            name="close-failure",
        )
        try:
            worker.start()
            assert released.wait(3)
            worker.stop()
            finish_drive.set()
            assert worker.wait_idle(3) is None
        finally:
            assert worker.close(timeout=2)
        assert _lease_row(store, "close-failure") == (1, None, None, 0)


@pytest.mark.sqlite_file
def test_wait_idle_does_not_wait_for_a_remote_lease(tmp_path: Path) -> None:
    database = tmp_path / "remote-lease.sqlite"
    with _installed_database(database) as database, _open_store(database) as store:
        _start(store, "remote", push=True)
        remote = store.acquire("remote", owner="remote-worker", ttl_ms=60_000)
        assert remote is not None
        observed = Event()

        @contextmanager
        def open_store():
            with _open_store(database) as worker_store:
                yield worker_store

        worker = LocalWorker(
            WorkerSupport(open_store=open_store, drive=None, project=None, observe=lambda _store: observed.set()),
            scan_interval=0.01,
            name="remote-independent",
        )
        try:
            worker.start()
            assert observed.wait(3)
            started = monotonic()
            worker.wait_idle(0.5)
            assert monotonic() - started < 0.2
        finally:
            assert worker.close(timeout=2)
            assert store.release(remote)


@pytest.mark.sqlite_file
def test_wake_session_remains_visible_until_its_coordinate_pass_finishes(tmp_path: Path) -> None:
    database = tmp_path / "wake-handoff.sqlite"
    first_enter, first_release, second_enter, second_release = Event(), Event(), Event(), Event()
    scan_calls = 0
    with _installed_database(database) as database:
        worker = LocalWorker(
            WorkerSupport(
                open_store=lambda: _open_store(database),
                drive=None,
                project=None,
                projection_consumers=(),
            ),
            scan_interval=1,
            name="wake-handoff",
        )

        def blocked_scan(_store: SessionStore) -> None:
            nonlocal scan_calls
            scan_calls += 1
            if scan_calls == 1:
                first_enter.set()
                assert first_release.wait(3)
            elif scan_calls == 2:
                second_enter.set()
                assert second_release.wait(3)

        worker._periodic_scan = blocked_scan
        try:
            worker.start()
            assert first_enter.wait(3)
            previous = worker.coordinate_pass
            worker.wake("thread")

            assert worker.active_wake_sessions == frozenset({"thread"})
            first_release.set()
            assert worker.wait_for_coordinate_pass(previous, 3)
            assert second_enter.wait(3)

            assert worker.active_wake_sessions == frozenset({"thread"})
            previous = worker.coordinate_pass
            second_release.set()
            assert worker.wait_for_coordinate_pass(previous, 3)
            assert not worker.active_wake_sessions
        finally:
            first_release.set()
            second_release.set()
            assert worker.close(timeout=2)


@pytest.mark.persistent_store
def test_claim_filters_work_kind_and_projection_consumers_before_claim(store: SessionStore, database: Any) -> None:
    del database
    _start(
        store,
        "owned-projection",
        consumers=("app_server", "child_delivery", "events", "traces"),
    )

    all_projection = store.list_runnable(work_kinds=("project",), projection_consumers=None)
    assert {(work.kind, work.consumer) for work in all_projection} == {
        ("project", "app_server"),
        ("project", "child_delivery"),
        ("project", "events"),
        ("project", "traces"),
    }

    owned = store.claim_runnable(
        owner="filtered-worker",
        ttl_ms=10_000,
        limit=10,
        work_kinds=("project",),
        projection_consumers=("child_delivery",),
    )
    assert [(entry.work.kind, entry.work.consumer) for entry in owned] == [("project", "child_delivery")]
    assert all(entry.lease is None for entry in owned)

    assert (
        store.claim_runnable(
            owner="observer-only",
            ttl_ms=10_000,
            work_kinds=("project",),
            projection_consumers=(),
        )
        == ()
    )
    assert _consumer_rows(store, "owned-projection") == {
        "app_server": 0,
        "child_delivery": 0,
        "events": 0,
        "traces": 0,
    }
