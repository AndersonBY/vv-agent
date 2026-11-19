"""W4 atomic claim and claimed-lease drive coverage."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from threading import Barrier, Event
from time import monotonic
from uuid import uuid4

import pytest

from vv_agent.kernel._drive import DriveBinding, drive_core
from vv_agent.kernel._postgres import RuntimeBinding
from vv_agent.kernel._sql import (
    LedgerIntegrityError,
    SQLStore,
    UnsupportedFeatureLevel,
    UnsupportedWorkerCapabilities,
)
from vv_agent.session import _drive_phase as drive_phase_module
from vv_agent.session.kernel import _Driver, drive
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.records import SessionSpec
from vv_agent.session.runtime import RuntimeNotReady
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import Lease, LeaseLost, LeasePoll, SessionStore
from vv_agent.session.supervisor import _tick, tick
from vv_agent.session.worker import LocalWorker, WorkerSupport

from .helpers import content_ref, item, principal, record, wrap_fixture_store
from .test_recovery_matrix import runtime


class _ManualClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def now_ms(self) -> int:
        return int(self.seconds * 1000)

    def monotonic(self) -> float:
        return self.seconds

    def monotonic_ns(self) -> int:
        return int(self.seconds * 1_000_000_000)

    def sleep(self, seconds: float) -> None:
        self.seconds += seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


class _RuntimeStub:
    ttl_ms = 100
    poll_ms = 10


def _binding(clock, *, make_factory_heartbeat=None, make_runtime=None, runtime=None, make_scope=None):
    return DriveBinding(
        clock=clock,
        ttl_ms=100,
        runtime=runtime,
        make_factory_heartbeat=make_factory_heartbeat,
        make_runtime=make_runtime,
        factory_retry_after_ms=lambda error: error.retry_after_ms if isinstance(error, RuntimeNotReady) else None,
        make_scope=make_scope or (lambda _lease, _runtime: pytest.fail("scope must not be created")),
        make_driver=lambda _store, _sid, _runtime, _scope: pytest.fail("driver must not be created"),
        after_drive=lambda _runtime: None,
        wake_if_runnable=lambda _runtime: None,
    )


def _start(store, session_id: str, *, consumers: tuple[str, ...] = (), push: bool = True) -> None:
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


@contextmanager
def _open_independent_store(database):
    manager = SQLiteStore.standalone(str(database)) if isinstance(database, Path) else PostgresStore.standalone(database)
    with manager as store:
        store.configure_input_admission(allow_unsigned=True)
        yield store


def _lease_row(store, session_id: str) -> tuple[int, str | None, int | None, int]:
    return store._one(
        "SELECT lease_epoch,lease_owner,lease_until_ms,drive_retry_at_ms FROM sk_session WHERE tenant_id=%s AND session_id=%s",
        (store.tenant_id, session_id),
    )


def _race_high_input_before_inbox_read(store, database, session_id: str, monkeypatch) -> list[bool]:
    original_rows = store._rows
    injected = [False]

    def rows(query: str, params=()):
        if not injected[0] and "FROM sk_inbox" in query:
            injected[0] = True
            high = item("concurrent-high")
            body = high.encode()
            with SQLiteStore.standalone(str(database)) as writer, writer._transaction():
                row = writer._lock(session_id)
                now, input_seq = writer._now(), row[1] + 1
                writer._rows(
                    "INSERT INTO sk_inbox(tenant_id,session_id,input_id,input_seq,kind,body,digest,"
                    "admission_key_id,admission_signature,admission_signed_ms,available_ms,received_ms) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,NULL,NULL,NULL,%s,%s)",
                    (
                        writer.tenant_id,
                        session_id,
                        high.input_id,
                        input_seq,
                        high.kind,
                        body,
                        sha256(body).hexdigest(),
                        now,
                        now,
                    ),
                )
                writer._rows(
                    "UPDATE sk_session SET inbox_seq=%s,level=2 WHERE tenant_id=%s AND session_id=%s",
                    (input_seq, writer.tenant_id, session_id),
                )
        return original_rows(query, params)

    monkeypatch.setattr(store, "_rows", rows)
    return injected


@pytest.mark.sqlite_file
def test_sqlite_file_workers_do_not_duplicate_claim(store, database) -> None:
    _start(store, "shared")
    barrier = Barrier(2)

    def claim(owner: str) -> str | None:
        with SQLiteStore.standalone(str(database)) as worker:
            worker.configure_input_admission(allow_unsigned=True)
            worker.configure_worker(level=1, handler_versions=("1",), capabilities=())
            barrier.wait(timeout=10)
            claimed = worker.claim_runnable(owner=owner, ttl_ms=10_000, limit=1)
            return None if not claimed else claimed[0].lease.owner if claimed[0].lease is not None else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        owners = list(executor.map(claim, ("sqlite-a", "sqlite-b")))

    assert sum(owner is not None for owner in owners) == 1
    owner = next(owner for owner in owners if owner is not None)
    epoch, actual_owner, expires_at_ms, _ = _lease_row(store, "shared")
    assert (epoch, actual_owner) == (1, owner)
    assert expires_at_ms is not None


@pytest.mark.persistent_store
def test_read_verification_is_bounded_by_captured_head(store, database, monkeypatch) -> None:
    _start(store, "captured-head")
    captured = store.read("captured-head")
    reader_manager = _open_independent_store(database)
    write_done = Event()
    captured_before_write = [False]

    write_started = Barrier(2)

    def append_with_second_connection() -> None:
        write_started.wait(timeout=10)
        try:
            with _open_independent_store(database) as writer:
                lease = writer.acquire("captured-head", owner="concurrent-writer", ttl_ms=10_000)
                assert lease is not None
                with writer.atomic() as tx:
                    tx.append(
                        "captured-head",
                        lease=lease,
                        expected_seq=writer.read("captured-head").head_seq,
                        commit_id="concurrent-turn",
                        records=(record("turn_started", sid="captured-head"),),
                    )
                assert writer.release(lease)
        finally:
            write_done.set()

    with reader_manager as reader, ThreadPoolExecutor(max_workers=1) as executor:
        original_rows = reader._rows

        def rows(query, params=()):
            if (
                not captured_before_write[0]
                and "FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq>%s AND seq<=%s" in query
            ):
                captured_before_write[0] = True
                write_started.wait(timeout=10)
                assert write_done.wait(timeout=10)
            return original_rows(query, params)

        monkeypatch.setattr(reader, "_rows", rows)
        appended = executor.submit(append_with_second_connection)
        page = reader.read("captured-head")
        appended.result(timeout=10)

    assert captured_before_write == [True]
    assert page.head_seq == captured.head_seq
    assert tuple(stored.seq for stored in page.records) == tuple(stored.seq for stored in captured.records)
    assert store.read("captured-head").head_seq == captured.head_seq + 1


@pytest.mark.persistent_store
@pytest.mark.parametrize("orphan", ["checkpoint", "record", "commit"])
def test_verify_ledger_rejects_append_only_rows_beyond_session_head(store, orphan: str) -> None:
    session_id = f"orphan-{orphan}"

    class Signer:
        key_id = "orphan-test-key"

        def bind_key(self):
            return self

        def sign(self, message: bytes) -> bytes:
            return sha256(message).digest()

        def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
            return key_id == self.key_id and signature == sha256(message).digest()

    store.configure_input_admission(Signer())
    _start(store, session_id)
    lease = store.acquire(session_id, owner="orphan-writer", ttl_ms=10_000)
    assert lease is not None
    with store.atomic() as tx:
        tx.append(
            session_id,
            lease=lease,
            expected_seq=store.read(session_id).head_seq,
            commit_id=f"orphan-{orphan}-turn",
            records=(record("turn_started", sid=session_id),),
        )
    assert store.release(lease)

    if orphan == "commit":
        store._rows("DELETE FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s AND seq=2", (store.tenant_id, session_id))
        store._rows("DELETE FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=2", (store.tenant_id, session_id))
    elif orphan == "record":
        store._rows("DELETE FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s AND seq=2", (store.tenant_id, session_id))
        store._rows(
            "DELETE FROM sk_commit WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
            (store.tenant_id, session_id, f"orphan-{orphan}-turn"),
        )
    store._rows(
        "UPDATE sk_session SET head_seq=1 WHERE tenant_id=%s AND session_id=%s",
        (store.tenant_id, session_id),
    )

    with pytest.raises(LedgerIntegrityError, match=rf"{orphan} outlives the session head"):
        store.verify_ledger(session_id)


@pytest.mark.sqlite_file
def test_claim_rejects_invalid_page_size_before_claiming(store) -> None:
    _start(store, "invalid-page")

    with pytest.raises(ValueError, match="limit"):
        store.claim_runnable(owner="invalid-page", ttl_ms=10_000, limit=0)

    assert _lease_row(store, "invalid-page")[:3] == (0, None, None)


@pytest.mark.sqlite_file
def test_claim_rejects_store_atomic_transaction(store) -> None:
    _start(store, "nested")

    with store.atomic(), pytest.raises(RuntimeError, match="outside a host transaction"):
        store.claim_runnable(owner="nested", ttl_ms=10_000)

    assert _lease_row(store, "nested")[:3] == (0, None, None)


@pytest.mark.postgres
def test_claim_rejects_postgres_nonautocommit_transaction(store, database: str) -> None:
    import psycopg

    _start(store, "pg-nested")
    with psycopg.connect(database, autocommit=False) as connection:
        worker = PostgresStore(connection)
        worker.configure_input_admission(allow_unsigned=True)
        worker._runtime_binding = RuntimeBinding("worker")
        worker.configure_worker(level=1, handler_versions=("1",), capabilities=())
        connection.execute("SELECT 1")
        with pytest.raises(RuntimeError, match="outside a host transaction"):
            worker.claim_runnable(owner="nested", ttl_ms=10_000)
        connection.rollback()

    assert _lease_row(store, "pg-nested")[:3] == (0, None, None)


@pytest.mark.sqlite_file
def test_local_worker_passes_claimed_lease_to_modern_callback(store, database) -> None:
    _start(store, "local-claim")
    seen: list[tuple[str, Lease]] = []

    @contextmanager
    def open_store():
        with SQLiteStore.standalone(str(database)) as worker_store:
            worker_store.configure_input_admission(allow_unsigned=True)
            yield worker_store

    def drive(store: SessionStore, session_id: str, *, lease: Lease) -> None:
        seen.append((session_id, lease))
        store.defer_drive(lease, retry_after_ms=250)
        store.release(lease)

    worker = LocalWorker(WorkerSupport(open_store=open_store, drive=drive), scan_interval=1)
    worker._scan()

    assert len(seen) == 1
    session_id, lease = seen[0]
    assert session_id == "local-claim"
    assert lease.owner.startswith("local-worker/") and lease.epoch == 1


@pytest.mark.persistent_store
def test_public_supervisor_preserves_claim_when_runtime_scope_close_fails(store, database, monkeypatch) -> None:
    session_id = "supervisor-close-failure"
    _start(store, session_id)
    rt = runtime(database, [])
    claimed_leases = []
    original_claim = store.claim_runnable

    def observe_claim(*args, **kwargs):
        page = original_claim(*args, **kwargs)
        claimed_leases.extend(entry.lease for entry in page if entry.lease is not None)
        return page

    monkeypatch.setattr(_Driver, "step", lambda _self: False)
    monkeypatch.setattr(store, "claim_runnable", observe_claim)
    rows_at_close = []
    scopes = []
    original_scope_close = drive_phase_module._Scope.close

    def blocked_close(scope):
        scopes.append(scope)
        rows_at_close.append(_lease_row(store, session_id))
        raise LeaseLost("runtime keeper is still renewing")

    monkeypatch.setattr(drive_phase_module._Scope, "close", blocked_close)
    try:
        with pytest.raises(ExceptionGroup) as raised:
            tick(store, project=lambda _sid, _consumer: None, runtime=lambda _sid: rt, page_size=1)

        assert len(raised.value.exceptions) == 1
        assert len(claimed_leases) == 1
        claimed = claimed_leases[0]
        assert (claimed.epoch, claimed.owner) == (rows_at_close[0][0], rows_at_close[0][1])
        assert claimed.expires_at_ms <= (rows_at_close[0][2] or 0)
        assert _lease_row(store, session_id) == rows_at_close[0]
    finally:
        assert len(scopes) == 1
        original_scope_close(scopes[0])
        assert not scopes[0].lease_keeper.is_alive(scopes[0])
        assert store.release(scopes[0].lease)


@pytest.mark.sqlite_file
def test_public_supervisor_blocked_renew_keeps_claim_until_bounded_close_timeout(store, database, monkeypatch) -> None:
    session_id = "supervisor-blocked-renew"
    _start(store, session_id)
    rt = runtime(database, [], ttl_ms=400, heartbeat_seconds=0.02)
    renew_entered = Event()
    unblock_renew = Event()
    renew_calls = 0
    scopes = []
    rows_at_close = []
    original_renew = SQLStore.renew
    original_scope_close = drive_phase_module._Scope.close

    def observed_renew(self, lease, *, ttl_ms):
        nonlocal renew_calls
        if lease.session_id == session_id:
            renew_calls += 1
            if renew_calls == 3:
                renew_entered.set()
                assert unblock_renew.wait(5), "blocked renewal was never released"
        return original_renew(self, lease, ttl_ms=ttl_ms)

    def wait_for_blocked_renew(_self):
        assert renew_entered.wait(2), "runtime never reached its second periodic renewal"
        return False

    def bounded_close(scope):
        scopes.append(scope)
        scope.request_stop()
        deadline = monotonic() + 0.1
        while scope.lease_keeper.is_alive(scope) and monotonic() < deadline:
            Event().wait(0.005)
        if scope.lease_keeper.is_alive(scope):
            rows_at_close.append(_lease_row(store, session_id))
            raise LeaseLost("runtime keeper is still blocked in renew")
        pytest.fail("close was expected to time out while renew remained blocked")

    monkeypatch.setattr(SQLStore, "renew", observed_renew)
    monkeypatch.setattr(_Driver, "step", wait_for_blocked_renew)
    monkeypatch.setattr(drive_phase_module._Scope, "close", bounded_close)

    try:
        with pytest.raises(ExceptionGroup) as raised:
            tick(store, project=lambda _sid, _consumer: None, runtime=lambda _sid: rt, page_size=1)
        assert len(raised.value.exceptions) == 1
        assert renew_entered.is_set()
        assert _lease_row(store, session_id) == rows_at_close[0]
    finally:
        unblock_renew.set()
        if scopes:
            original_scope_close(scopes[0])
            assert not scopes[0].lease_keeper.is_alive(scopes[0])
            assert store.release(scopes[0].lease)


@pytest.mark.sqlite_file
def test_local_worker_preserves_claim_after_core_owns_it(store, database, monkeypatch) -> None:
    session_id = "local-worker-close-failure"
    _start(store, session_id)
    rt = runtime(database, [])

    @contextmanager
    def open_store():
        with SQLiteStore.standalone(str(database)) as worker_store:
            worker_store.configure_input_admission(allow_unsigned=True)
            yield worker_store

    claimed_leases = []

    def drive_owned(store: SessionStore, session_id: str, *, lease: Lease) -> None:
        claimed_leases.append(lease)
        drive(store, session_id, runtime=lambda _sid: rt, _lease=lease)

    monkeypatch.setattr(_Driver, "step", lambda _self: False)
    rows_at_close = []
    scopes = []
    original_scope_close = drive_phase_module._Scope.close

    def blocked_close(scope):
        scopes.append(scope)
        rows_at_close.append(_lease_row(store, session_id))
        raise LeaseLost("runtime keeper is still renewing")

    monkeypatch.setattr(drive_phase_module._Scope, "close", blocked_close)
    worker = LocalWorker(WorkerSupport(open_store=open_store, drive=drive_owned), scan_interval=1)
    try:
        with pytest.raises(ExceptionGroup):
            worker._scan()

        assert len(claimed_leases) == 1
        claimed = claimed_leases[0]
        assert (claimed.epoch, claimed.owner) == (rows_at_close[0][0], rows_at_close[0][1])
        assert claimed.expires_at_ms <= (rows_at_close[0][2] or 0)
        assert _lease_row(store, session_id) == rows_at_close[0]
    finally:
        assert len(scopes) == 1
        original_scope_close(scopes[0])
        assert not scopes[0].lease_keeper.is_alive(scopes[0])
        assert store.release(scopes[0].lease)


@pytest.mark.sqlite_file
def test_local_worker_returns_claim_when_drive_store_open_fails(store, database) -> None:
    session_id = "local-worker-open-failure"
    _start(store, session_id)
    opens = 0

    @contextmanager
    def open_store():
        nonlocal opens
        opens += 1
        if opens > 1:
            raise RuntimeError("drive store open failed")
        with SQLiteStore.standalone(str(database)) as worker_store:
            worker_store.configure_input_admission(allow_unsigned=True)
            yield worker_store

    worker = LocalWorker(
        WorkerSupport(
            open_store=open_store,
            drive=lambda *_args, **_kwargs: pytest.fail("claim must not reach drive"),
            failure_backoff_ms=321,
        ),
        scan_interval=1,
    )
    with pytest.raises(ExceptionGroup):
        worker._scan()

    epoch, owner, expires_at_ms, retry_at_ms = _lease_row(store, session_id)
    assert epoch == 1 and owner is None and expires_at_ms is None
    assert retry_at_ms > 0


@pytest.mark.sqlite_file
def test_local_worker_returns_claim_when_stopped_before_handoff(store, database) -> None:
    session_id = "local-worker-stopped"
    _start(store, session_id)
    opens = 0
    worker = None

    @contextmanager
    def open_store():
        nonlocal opens
        opens += 1
        with SQLiteStore.standalone(str(database)) as worker_store:
            worker_store.configure_input_admission(allow_unsigned=True)
            if opens == 2:
                assert worker is not None
                worker.stop()
            yield worker_store

    worker = LocalWorker(
        WorkerSupport(
            open_store=open_store,
            drive=lambda *_args, **_kwargs: pytest.fail("stopped work must not reach drive"),
            failure_backoff_ms=321,
        ),
        scan_interval=1,
    )
    worker._scan()

    epoch, owner, expires_at_ms, retry_at_ms = _lease_row(store, session_id)
    assert epoch == 1 and owner is None and expires_at_ms is None
    assert retry_at_ms > 0


@pytest.mark.sqlite_file
def test_supervisor_claims_next_drive_only_after_previous_lease_is_finished(store) -> None:
    _start(store, "page-a")
    _start(store, "page-b")
    seen = []

    observed_epochs = []

    def dispatch_claimed(session_id, lease):
        seen.append(session_id)
        observed_epochs.append((_lease_row(store, "page-a")[0], _lease_row(store, "page-b")[0]))
        store.defer_drive(lease, retry_after_ms=250)
        store.release(lease)

    assert _tick(store, project=lambda _sid, _consumer: None, dispatch_claimed=dispatch_claimed, page_size=2) == 2
    assert seen == ["page-a", "page-b"]
    assert observed_epochs == [(1, 0), (1, 1)]


@pytest.mark.sqlite_file
def test_claim_runnable_never_preclaims_multiple_drive_leases(store) -> None:
    _start(store, "claim-a")
    _start(store, "claim-b")

    claimed = store.claim_runnable(owner="single", ttl_ms=10_000, limit=100)

    assert [entry.work.session_id for entry in claimed] == ["claim-a"]
    assert _lease_row(store, "claim-a")[:2] == (1, "single")
    assert _lease_row(store, "claim-b")[:2] == (0, None)
    assert store.release(claimed[0].lease)


@pytest.mark.persistent_store
@pytest.mark.parametrize("tenant_bound", [False, True])
def test_factory_heartbeat_opens_a_fresh_store_for_each_renewal(store, database, monkeypatch, tenant_bound) -> None:
    _start(store, "factory-heartbeat")
    lease = store.acquire("factory-heartbeat", owner="factory", ttl_ms=10_000)
    assert lease is not None
    opened = []

    def drive_core_stub(drive_store, session_id, *, make_binding, lease, **_kwargs):
        binding = make_binding()
        factory = binding.make_factory_heartbeat
        assert factory is not None
        heartbeat = factory(lease, binding.ttl_ms)
        for _ in range(2):
            with heartbeat.heartbeat_store() as heartbeat_store:
                opened.append(heartbeat_store)
                heartbeat.poll(heartbeat_store)
        return True

    monkeypatch.setattr(drive_phase_module, "drive_core", drive_core_stub)
    drive_store = store.for_tenant(store.tenant_id) if tenant_bound else store
    assert drive_phase_module.drive_with_runtime(
        drive_store,
        "factory-heartbeat",
        runtime=lambda session_id: runtime(database, []),
        lease=lease,
    )

    assert len(opened) == 2
    assert opened[0] is not opened[1]
    assert store.release(lease)


@pytest.mark.sqlite_file
def test_legacy_dispatch_uses_read_only_hint_without_epoch_or_lease(store) -> None:
    _start(store, "hint")
    interrupted = []

    def dispatch(session_id: str) -> None:
        interrupted.append(session_id)
        raise ValueError("legacy dispatch interrupted")

    with pytest.raises(ExceptionGroup) as raised:
        tick(store, project=lambda _sid, _consumer: None, dispatch=dispatch)

    assert interrupted == ["hint"]
    assert len(raised.value.exceptions) == 1
    assert isinstance(raised.value.exceptions[0], ValueError)
    assert _lease_row(store, "hint")[:3] == (0, None, None)


@pytest.mark.sqlite_file
def test_claim_returns_drive_lease_and_projection_without_lease(store) -> None:
    _start(store, "projected", consumers=("events",))
    lease = store.acquire("projected", owner="seed", ttl_ms=10_000)
    assert lease is not None
    requirements = {"level": 1, "handler_version": "1", "capabilities": []}
    with store.atomic() as tx:
        tx.append(
            "projected",
            lease=lease,
            expected_seq=store.read("projected").head_seq,
            commit_id="seed-turn",
            records=(record("turn_started", sid="projected", handler_version="1", requirements=requirements),),
        )
    assert store.release(lease)

    drive_claim = store.claim_runnable(owner="projection-claim", ttl_ms=10_000, limit=10)
    assert [item.work.kind for item in drive_claim] == ["drive"]
    assert drive_claim[0].lease is not None

    projection_claim = store.claim_runnable(owner="projection-claim", ttl_ms=10_000, limit=10)
    assert [(item.work.kind, item.work.consumer, item.lease) for item in projection_claim] == [("project", "events", None)]
    assert store.release(drive_claim[0].lease)


@pytest.mark.sqlite_file
def test_input_reads_use_captured_inbox_watermark_during_level_raising_push(store, database, monkeypatch) -> None:
    store.configure_worker(level=1, handler_versions=("1",), capabilities=())
    store.configure_input_admission(allow_unsigned=True)
    _start(store, "captured-input")
    injected = _race_high_input_before_inbox_read(store, database, "captured-input", monkeypatch)

    assert store.read_input("captured-input", "concurrent-high") is None
    assert injected == [True]


@pytest.mark.sqlite_file
def test_pending_inputs_use_captured_inbox_watermark_during_level_raising_push(store, database, monkeypatch) -> None:
    store.configure_worker(level=1, handler_versions=("1",), capabilities=())
    store.configure_input_admission(allow_unsigned=True)
    _start(store, "captured-pending")
    injected = _race_high_input_before_inbox_read(store, database, "captured-pending", monkeypatch)

    assert tuple(stored.item.input_id for stored in store.peek_inbox("captured-pending", ready_only=False)) == ("initial",)
    assert injected == [True]


@pytest.mark.sqlite_file
def test_read_records_use_captured_inbox_watermark_during_level_raising_push(store, database, monkeypatch) -> None:
    store.configure_worker(level=1, handler_versions=("1",), capabilities=())
    store.configure_input_admission(allow_unsigned=True)
    _start(store, "captured-read")
    decoded: list[int] = []
    original_stored_input = store._stored_input

    def stored_input(session_id, row, **kwargs):
        decoded.append(row[1])
        return original_stored_input(session_id, row, **kwargs)

    monkeypatch.setattr(store, "_stored_input", stored_input)
    injected = _race_high_input_before_inbox_read(store, database, "captured-read", monkeypatch)

    assert store.read("captured-read").head_seq == 1
    assert decoded == [1]
    assert injected == [True]


@pytest.mark.sqlite_file
def test_control_records_use_captured_inbox_watermark_during_level_raising_push(store, database, monkeypatch) -> None:
    store.configure_worker(level=1, handler_versions=("1",), capabilities=())
    store.configure_input_admission(allow_unsigned=True)
    _start(store, "captured-control")
    decoded: list[int] = []
    original_stored_input = store._stored_input

    def stored_input(session_id, row, **kwargs):
        decoded.append(row[1])
        return original_stored_input(session_id, row, **kwargs)

    monkeypatch.setattr(store, "_stored_input", stored_input)
    injected = _race_high_input_before_inbox_read(store, database, "captured-control", monkeypatch)

    assert store.control_records("captured-control", through_seq=1) == ()
    assert decoded == [1]
    assert injected == [True]


@pytest.mark.persistent_store
def test_idle_deferral_compares_matching_head_snapshot(store) -> None:
    _start(store, "idle")
    lease = store.acquire("idle", owner="idle", ttl_ms=10_000)
    assert lease is not None
    requirements = {"level": 1, "handler_version": "1", "capabilities": []}
    with store.atomic() as tx:
        tx.append(
            "idle",
            lease=lease,
            expected_seq=store.read("idle").head_seq,
            commit_id="idle-turn",
            records=(record("turn_started", sid="idle", handler_version="1", requirements=requirements),),
        )

    before_ms = store._now()
    assert store.defer_idle_drive(lease, poll_ms=250)
    scheduled_ms = store._one(
        "SELECT next_drive_ms FROM sk_session WHERE tenant_id=%s AND session_id=%s",
        (store.tenant_id, "idle"),
    )[0]
    after_ms = store._now()
    assert before_ms + 250 <= scheduled_ms <= after_ms + 250
    assert store.release(lease)


@pytest.mark.postgres
def test_postgres_workers_claim_one_session_once(store, database: str) -> None:
    _start(store, "shared-pg")
    barrier = Barrier(2)

    def claim(owner: str) -> str | None:
        with PostgresStore.standalone(database) as worker:
            worker.configure_input_admission(allow_unsigned=True)
            worker._runtime_binding = RuntimeBinding("worker")
            worker.configure_worker(level=1, handler_versions=("1",), capabilities=())
            barrier.wait(timeout=10)
            claimed = worker.claim_runnable(owner=owner, ttl_ms=10_000, limit=1)
            return None if not claimed else claimed[0].lease.owner if claimed[0].lease is not None else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        owners = list(executor.map(claim, ("postgres-a", "postgres-b")))

    assert sum(owner is not None for owner in owners) == 1
    owner = next(owner for owner in owners if owner is not None)
    assert _lease_row(store, "shared-pg")[:2] == (1, owner)


@pytest.mark.postgres
def test_postgres_claim_commits_before_returning(store, database: str) -> None:
    _start(store, "unlocked")

    with PostgresStore.standalone(database) as claimant:
        claimant.configure_input_admission(allow_unsigned=True)
        claimant._runtime_binding = RuntimeBinding("worker")
        claimant.configure_worker(level=1, handler_versions=("1",), capabilities=())
        claimed = claimant.claim_runnable(owner="claimant", ttl_ms=10_000, limit=1)
        assert len(claimed) == 1 and claimed[0].lease is not None

        with PostgresStore.standalone(database) as probe:
            row = probe._one(
                "SELECT session_id FROM sk_session WHERE tenant_id=%s AND session_id=%s FOR UPDATE NOWAIT",
                (probe.tenant_id, "unlocked"),
            )
            assert row == ("unlocked",)
        assert claimant.release(claimed[0].lease)


@pytest.mark.sqlite_file
def test_public_drive_uses_claimed_runtime_not_ready_lease_without_acquire(store, database) -> None:
    _start(store, "not-ready")
    claimed = store.claim_runnable(owner="preclaimed", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    lease = claimed[0].lease

    def forbidden_acquire(*_args, **_kwargs):
        raise AssertionError("claimed drive must not acquire a second lease")

    store.acquire = forbidden_acquire  # type: ignore[method-assign]

    def not_ready(_session_id: str):
        raise RuntimeNotReady(250)

    assert drive(store, "not-ready", runtime=not_ready, _lease=lease)
    epoch, owner, expires_at_ms, retry_at_ms = _lease_row(store, "not-ready")
    assert epoch == lease.epoch
    assert owner is None and expires_at_ms is None
    assert retry_at_ms > 0


@pytest.mark.sqlite_file
def test_wrapped_store_first_renewal_recovers_short_lock_before_dispatch(store, database, monkeypatch) -> None:
    session_id = "wrapped-anchor"
    wrapped = wrap_fixture_store(store)
    _start(wrapped, session_id)
    claimed = wrapped.claim_runnable(owner="wrapped", ttl_ms=5_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None

    rt = runtime(database, [], ttl_ms=400, heartbeat_seconds=0.02)
    lock_until = monotonic() + 0.08
    lock_attempts = []
    runtime_confirmations = []
    anchor_stores = []
    original_renew = SQLStore.renew
    original_heartbeat_confirmed = drive_phase_module._Scope.heartbeat_confirmed
    original_anchor = drive_phase_module._lease_anchor_now_ms

    def delayed_renew(self, lease, *, ttl_ms):
        if lease.session_id == session_id and monotonic() < lock_until:
            lock_attempts.append(monotonic())
            raise sqlite3.OperationalError("database is locked")
        return original_renew(self, lease, ttl_ms=ttl_ms)

    def observe_runtime_ready(self, poll):
        if self.lease.session_id == session_id:
            runtime_confirmations.append(monotonic())
        return original_heartbeat_confirmed(self, poll)

    def observe_step(_self):
        assert runtime_confirmations, "driver dispatched before runtime heartbeat confirmation"
        return False

    def observe_anchor(selected_store, selected_session_id, candidate, clock):
        anchor_stores.append(selected_store)
        return original_anchor(selected_store, selected_session_id, candidate, clock)

    monkeypatch.setattr(SQLStore, "renew", delayed_renew)
    monkeypatch.setattr(drive_phase_module._Scope, "heartbeat_confirmed", observe_runtime_ready)
    monkeypatch.setattr(drive_phase_module, "_lease_anchor_now_ms", observe_anchor)
    monkeypatch.setattr(_Driver, "step", observe_step)

    assert drive(wrapped, session_id, runtime=lambda _session_id: rt, _lease=claimed[0].lease)
    assert anchor_stores == [wrapped]
    assert lock_attempts
    assert runtime_confirmations


@pytest.mark.sqlite_file
def test_claimed_drive_failure_defers_and_releases(store, database, monkeypatch) -> None:
    _start(store, "failure")
    claimed = store.claim_runnable(owner="failing", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    rt = runtime(database, [])
    hooks = []
    rt.hook = lambda point, _record: hooks.append(point)

    def fail_step(_self):
        raise ValueError("driver failed")

    monkeypatch.setattr(_Driver, "step", fail_step)
    with pytest.raises(ValueError, match="driver failed"):
        drive(store, "failure", runtime=lambda _session_id: rt, _lease=claimed[0].lease)

    epoch, owner, expires_at_ms, retry_at_ms = _lease_row(store, "failure")
    assert epoch == claimed[0].lease.epoch
    assert owner is None and expires_at_ms is None
    assert retry_at_ms > 0
    assert hooks == ["after_drive"]


@pytest.mark.sqlite_file
def test_claimed_scope_start_failure_still_defers_and_releases(store, database, monkeypatch) -> None:
    _start(store, "scope-failure")
    claimed = store.claim_runnable(owner="scope-failing", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    rt = runtime(database, [])

    def fail_start(_self):
        raise ValueError("scope start failed")

    def forbidden_renew(*_args, **_kwargs):
        raise AssertionError("factory handoff must adopt its confirmed poll without renewing")

    monkeypatch.setattr(drive_phase_module._Scope, "start", fail_start)
    monkeypatch.setattr(store, "renew", forbidden_renew)
    with pytest.raises(ValueError, match="scope start failed"):
        drive(store, "scope-failure", runtime=lambda _session_id: rt, _lease=claimed[0].lease)

    epoch, owner, expires_at_ms, retry_at_ms = _lease_row(store, "scope-failure")
    assert epoch == claimed[0].lease.epoch
    assert owner is None and expires_at_ms is None
    assert retry_at_ms > 0


@pytest.mark.sqlite_file
def test_claimed_make_binding_failure_releases_lease(store) -> None:
    _start(store, "binding-failure")
    claimed = store.claim_runnable(owner="binding", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None

    def fail_binding():
        raise ValueError("binding failed")

    with pytest.raises(ValueError, match="binding failed"):
        drive_core(store, "binding-failure", lease=claimed[0].lease, make_binding=fail_binding)

    assert _lease_row(store, "binding-failure")[:3] == (claimed[0].lease.epoch, None, None)


@pytest.mark.persistent_store
def test_claimed_scope_close_failure_preserves_ownership_and_skips_backoff(store, database, monkeypatch) -> None:
    _start(store, "close-failure")
    claimed = store.claim_runnable(owner="closing", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    rt = runtime(database, [])
    scopes = []
    original_scope_close = drive_phase_module._Scope.close

    def fail_step(_self):
        raise ValueError("driver failed")

    def fail_close(scope):
        scopes.append(scope)
        raise RuntimeError("scope close failed")

    monkeypatch.setattr(_Driver, "step", fail_step)
    monkeypatch.setattr(drive_phase_module._Scope, "close", fail_close)
    try:
        with pytest.raises(ValueError, match="driver failed"):
            drive(store, "close-failure", runtime=lambda _session_id: rt, _lease=claimed[0].lease)

        epoch, owner, expires_at_ms, retry_at_ms = _lease_row(store, "close-failure")
        assert epoch == claimed[0].lease.epoch
        assert owner == claimed[0].lease.owner and expires_at_ms is not None
        assert retry_at_ms == 0
    finally:
        assert len(scopes) == 1
        original_scope_close(scopes[0])
        assert not scopes[0].lease_keeper.is_alive(scopes[0])
        assert store.release(scopes[0].lease)


@pytest.mark.sqlite_file
def test_factory_runtime_not_ready_with_alive_close_keeps_owner_and_does_not_defer(store, monkeypatch) -> None:
    _start(store, "factory-close-alive")
    lease = store.acquire("factory-close-alive", owner="factory-owner", ttl_ms=10_000)
    assert lease is not None
    releases = []
    original_release = store.release

    def release(candidate):
        releases.append(candidate)
        return original_release(candidate)

    class Build:
        def __init__(self):
            self.clock = _ManualClock()
            self.ready = Event()
            self.ready.set()
            self.lost = False
            self.successful_ready = True
            self.heartbeat_error = None
            self.lease = lease
            self.latest_poll = None
            self.expires_deadline = 15.0

        def start(self):
            pass

        def close(self):
            raise LeaseLost("factory heartbeat is still renewing")

        def adopt_confirmed(self, poll, *, expires_deadline):
            pytest.fail("factory heartbeat must not adopt")

    build = Build()

    def make_runtime():
        raise RuntimeNotReady(250)

    binding = _binding(
        build.clock,
        make_factory_heartbeat=lambda _lease, _ttl: build,
        make_runtime=make_runtime,
    )
    monkeypatch.setattr(store, "release", release)

    with pytest.raises(RuntimeNotReady):
        drive_core(store, "factory-close-alive", lease=lease, make_binding=lambda: binding)

    assert releases == []
    assert _lease_row(store, "factory-close-alive")[:4] == (lease.epoch, "factory-owner", lease.expires_at_ms, 0)


@pytest.mark.sqlite_file
@pytest.mark.parametrize("error_type", [UnsupportedFeatureLevel, UnsupportedWorkerCapabilities])
def test_factory_heartbeat_preserves_typed_mismatch_and_releases_after_close(store, error_type) -> None:
    session_id = f"factory-typed-{error_type.__name__}"
    _start(store, session_id)
    lease = store.acquire(session_id, owner="factory-typed-owner", ttl_ms=10_000)
    assert lease is not None
    closed = []

    class Build:
        def __init__(self):
            self.clock = _ManualClock()
            self.ready = Event()
            self.ready.set()
            self.lost = True
            self.successful_ready = False
            self.heartbeat_error = error_type("typed factory capability mismatch")
            self.lease = lease
            self.latest_poll = None
            self.expires_deadline = 15.0

        def start(self):
            pass

        def close(self):
            closed.append(True)

        def adopt_confirmed(self, poll, *, expires_deadline):
            pytest.fail("factory heartbeat must not adopt")

    binding = _binding(
        Build().clock,
        make_factory_heartbeat=lambda _lease, _ttl: Build(),
        make_runtime=lambda: _RuntimeStub(),
    )

    with pytest.raises(error_type, match="typed factory capability mismatch"):
        drive_core(store, session_id, lease=lease, make_binding=lambda: binding)

    assert closed == [True]
    assert _lease_row(store, session_id)[:4] == (lease.epoch, None, None, 0)


@pytest.mark.sqlite_file
def test_factory_typed_mismatch_with_alive_close_keeps_owner_without_defer(store) -> None:
    session_id = "factory-typed-close-alive"
    _start(store, session_id)
    lease = store.acquire(session_id, owner="factory-typed-close-owner", ttl_ms=10_000)
    assert lease is not None
    typed_error = UnsupportedWorkerCapabilities("factory capability mismatch while closing")

    class Build:
        def __init__(self):
            self.clock = _ManualClock()
            self.ready = Event()
            self.ready.set()
            self.lost = True
            self.successful_ready = False
            self.heartbeat_error = typed_error
            self.lease = lease
            self.latest_poll = None
            self.expires_deadline = 15.0

        def start(self):
            pass

        def close(self):
            raise LeaseLost("factory keeper is still active")

        def adopt_confirmed(self, poll, *, expires_deadline):
            pytest.fail("factory heartbeat must not adopt")

    binding = _binding(
        Build().clock,
        make_factory_heartbeat=lambda _lease, _ttl: Build(),
        make_runtime=lambda: _RuntimeStub(),
    )
    try:
        with pytest.raises(UnsupportedWorkerCapabilities) as raised:
            drive_core(store, session_id, lease=lease, make_binding=lambda: binding)
        assert raised.value is typed_error
        assert _lease_row(store, session_id)[:4] == (lease.epoch, "factory-typed-close-owner", lease.expires_at_ms, 0)
    finally:
        assert store.release(lease)


@pytest.mark.sqlite_file
def test_runtime_scope_preserves_typed_mismatch_and_releases_after_close(store) -> None:
    session_id = "runtime-typed-mismatch"
    _start(store, session_id)
    lease = store.acquire(session_id, owner="runtime-typed-owner", ttl_ms=10_000)
    assert lease is not None
    factory_closed = []
    scope_closed = []
    typed_error = UnsupportedWorkerCapabilities("runtime capability mismatch")
    clock = _ManualClock()
    poll = LeasePoll(lease, db_now_ms=lease.expires_at_ms - 15_000, inbox_seq=0, controls=())

    class Build:
        def __init__(self):
            self.clock = clock
            self.ready = Event()
            self.ready.set()
            self.lost = False
            self.successful_ready = True
            self.heartbeat_error = None
            self.lease = lease
            self.latest_poll = poll
            self.expires_deadline = 15.0

        def start(self):
            pass

        def close(self):
            factory_closed.append(True)

        def adopt_confirmed(self, poll, *, expires_deadline):
            pytest.fail("factory heartbeat must not adopt")

    class Scope:
        def __init__(self):
            self.clock = clock
            self.ready = Event()
            self.lost = True
            self.successful_ready = False
            self.heartbeat_error = typed_error
            self.lease = lease
            self.latest_poll = None
            self.expires_deadline = 15.0

        def start(self):
            self.ready.set()

        def close(self):
            scope_closed.append(True)

        def adopt_confirmed(self, poll, *, expires_deadline):
            self.latest_poll = poll
            self.expires_deadline = expires_deadline

        @contextmanager
        def committing(self):
            yield self.lease

    scope = Scope()
    binding = replace(
        _binding(
            clock,
            make_factory_heartbeat=lambda _lease, _ttl: Build(),
            make_runtime=lambda: _RuntimeStub(),
            make_scope=lambda _lease, _runtime: scope,
        ),
        make_driver=lambda _store, _sid, _runtime, _scope: pytest.fail("driver must not be created"),
    )

    with pytest.raises(UnsupportedWorkerCapabilities) as raised:
        drive_core(store, session_id, lease=lease, make_binding=lambda: binding)

    assert raised.value is typed_error
    assert factory_closed == [True]
    assert scope_closed == [True]
    assert _lease_row(store, session_id)[:4] == (lease.epoch, None, None, 0)


@pytest.mark.sqlite_file
def test_runtime_scope_close_failure_keeps_owner(store, monkeypatch) -> None:
    _start(store, "scope-close-alive")
    lease = store.acquire("scope-close-alive", owner="scope-owner", ttl_ms=10_000)
    assert lease is not None
    releases = []
    original_release = store.release

    def release(candidate):
        releases.append(candidate)
        return original_release(candidate)

    clock = _ManualClock()
    poll = LeasePoll(lease, db_now_ms=lease.expires_at_ms - 15_000, inbox_seq=0, controls=())

    class Build:
        def __init__(self):
            self.clock = clock
            self.ready = Event()
            self.ready.set()
            self.lost = False
            self.successful_ready = True
            self.heartbeat_error = None
            self.lease = lease
            self.latest_poll = poll
            self.expires_deadline = 15.0

        def start(self):
            pass

        def close(self):
            pass

        def adopt_confirmed(self, poll, *, expires_deadline):
            pytest.fail("factory heartbeat must not adopt")

    class Scope:
        def __init__(self):
            self.clock = clock
            self.ready = Event()
            self.lost = False
            self.successful_ready = True
            self.heartbeat_error = None
            self.lease = lease
            self.latest_poll = None
            self.expires_deadline = 15.0

        def start(self):
            self.ready.set()

        def close(self):
            raise LeaseLost("runtime heartbeat is still renewing")

        def adopt_confirmed(self, poll, *, expires_deadline):
            self.latest_poll = poll
            self.expires_deadline = expires_deadline
            self.successful_ready = True

        @contextmanager
        def committing(self):
            yield self.lease

    scope = Scope()

    class Step:
        records = ()

        def step(self):
            raise ValueError("driver failed while scope still owns lease")

    binding = replace(
        _binding(
            scope.clock,
            make_factory_heartbeat=lambda _lease, _ttl: Build(),
            make_runtime=lambda: _RuntimeStub(),
            make_scope=lambda _lease, _runtime: scope,
        ),
        make_driver=lambda _store, _sid, _runtime, _scope: Step(),
    )
    monkeypatch.setattr(store, "release", release)

    with pytest.raises(ValueError, match="scope still owns"):
        drive_core(store, "scope-close-alive", lease=lease, make_binding=lambda: binding)

    assert releases == []
    assert _lease_row(store, "scope-close-alive")[:4] == (lease.epoch, "scope-owner", lease.expires_at_ms, 0)


@pytest.mark.sqlite_file
def test_factory_close_adopts_latest_receipt_without_resetting_deadline(store) -> None:
    _start(store, "factory-latest")
    lease = store.acquire("factory-latest", owner="latest-owner", ttl_ms=10_000)
    assert lease is not None
    factory_clock = _ManualClock()
    runtime_clock = _ManualClock()
    factory_clock.seconds = 100
    runtime_clock.seconds = 1_000
    initial = LeasePoll(lease, db_now_ms=lease.expires_at_ms - 15_000, inbox_seq=0, controls=())
    latest = LeasePoll(
        lease,
        db_now_ms=lease.expires_at_ms - 8_000,
        inbox_seq=3,
        controls=(item("close-inflight-control", kind="control", action="cancel"),),
    )

    class Build:
        def __init__(self):
            self.clock = factory_clock
            self.ready = Event()
            self.ready.set()
            self.lost = False
            self.successful_ready = True
            self.heartbeat_error = None
            self.lease = lease
            self.latest_poll = initial
            self.expires_deadline = 15.0

        def start(self):
            pass

        def close(self):
            factory_clock.advance(10)
            self.latest_poll = latest
            self.expires_deadline = factory_clock.monotonic() + 8

        def adopt_confirmed(self, poll, *, expires_deadline):
            pytest.fail("factory heartbeat must not adopt")

    class Scope:
        def __init__(self):
            self.clock = runtime_clock
            self.ready = Event()
            self.lost = False
            self.successful_ready = True
            self.heartbeat_error = None
            self.lease = lease
            self.latest_poll = None
            self.expires_deadline = 0.0
            self.adopted_poll = None
            self.adopted_deadline = None

        def start(self):
            self.ready.set()

        def close(self):
            pass

        def adopt_confirmed(self, poll, *, expires_deadline):
            self.adopted_poll = poll
            self.adopted_deadline = expires_deadline
            self.latest_poll = poll
            self.expires_deadline = expires_deadline
            self.successful_ready = True

        @contextmanager
        def committing(self):
            yield self.lease

    scope = Scope()

    class Step:
        records = ()

        def step(self):
            raise ValueError("stop after handoff")

    binding = replace(
        _binding(
            factory_clock,
            make_factory_heartbeat=lambda _lease, _ttl: Build(),
            make_runtime=lambda: _RuntimeStub(),
            make_scope=lambda _lease, _runtime: scope,
        ),
        make_driver=lambda _store, _sid, _runtime, _scope: Step(),
    )

    with pytest.raises(ValueError, match="stop after handoff"):
        drive_core(store, "factory-latest", lease=lease, make_binding=lambda: binding)

    assert scope.adopted_poll is latest
    assert scope.adopted_poll.controls[0].input_id == "close-inflight-control"
    assert scope.adopted_deadline == 1_008.0
    epoch, owner, expires_at_ms, retry_at_ms = _lease_row(store, "factory-latest")
    assert epoch == lease.epoch
    assert owner is None and expires_at_ms is None
    assert retry_at_ms > 0


@pytest.mark.sqlite_file
def test_runtime_scope_must_be_ready_before_driver_step(store) -> None:
    _start(store, "scope-not-ready")
    lease = store.acquire("scope-not-ready", owner="not-ready-owner", ttl_ms=10_000)
    assert lease is not None
    step_called = []

    class Scope:
        def __init__(self):
            self.clock = _ManualClock()
            self.ready = Event()
            self.lost = False
            self.successful_ready = True
            self.heartbeat_error = None
            self.lease = lease
            self.latest_poll = None
            self.expires_deadline = 15.0

        def start(self):
            pass

        def close(self):
            pass

        def adopt_confirmed(self, poll, *, expires_deadline):
            pass

        @contextmanager
        def committing(self):
            yield self.lease

    scope = Scope()

    class Step:
        records = ()

        def step(self):
            step_called.append(True)
            return False

    binding = replace(
        _binding(
            scope.clock,
            runtime=_RuntimeStub(),
            make_scope=lambda _lease, _runtime: scope,
        ),
        make_driver=lambda _store, _sid, _runtime, _scope: Step(),
    )

    with pytest.raises(LeaseLost, match="did not become ready"):
        drive_core(store, "scope-not-ready", lease=lease, make_binding=lambda: binding)

    assert step_called == []


@pytest.mark.persistent_store
def test_existing_lease_admission_mismatch_releases_pending_work_for_compatible_worker(store) -> None:
    session_id = "admission-mismatch-release"
    _start(store, session_id)
    store.configure_worker(level=1, handler_versions=("1",), capabilities=())
    lease = store.acquire(session_id, owner="old-worker", ttl_ms=10_000)
    assert lease is not None

    store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    with store.atomic() as tx:
        tx.push(
            session_id,
            item(
                "high-level",
                actor=principal("test"),
                feature_level=2,
                requirements={"level": 2, "handler_version": "1", "capabilities": []},
            ),
        )

    def incompatible_binding():
        store._require_worker_accepts_row(store._head(session_id))
        pytest.fail("admission should reject the old worker")

    with pytest.raises(UnsupportedFeatureLevel, match="feature level 2"):
        drive_core(store, session_id, lease=lease, make_binding=incompatible_binding)

    assert _lease_row(store, session_id)[:3] == (lease.epoch, None, None)
    assert [(work.kind, work.session_id) for work in store.list_runnable(work_kinds=("drive",))] == [("drive", session_id)]
    assert not store.is_runnable(session_id)
    assert store.claim_runnable(owner="still-old", ttl_ms=10_000, limit=1) == ()
    assert (
        store._one(
            "SELECT consumed_seq FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND input_id='high-level'",
            (store.tenant_id, session_id),
        )[0]
        is None
    )

    store.configure_worker(level=2, handler_versions=("1",), capabilities=())
    claimed = store.claim_runnable(owner="compatible", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    assert claimed[0].lease.epoch == lease.epoch + 1
    assert store.release(claimed[0].lease)


@pytest.mark.postgres
def test_supervisor_claim_path_passes_original_lease(store, database: str) -> None:
    session_id = f"dispatch-{uuid4().hex}"
    _start(store, session_id)
    seen = []

    def dispatch_claimed(selected_session_id, lease):
        seen.append((selected_session_id, lease))
        store.defer_drive(lease, retry_after_ms=250)
        store.release(lease)

    assert _tick(store, project=lambda _sid, _consumer: None, dispatch_claimed=dispatch_claimed, page_size=1) == 1
    assert len(seen) == 1
    selected_session_id, lease = seen[0]
    assert selected_session_id == session_id
    assert lease.owner and lease.epoch == 1 and lease.expires_at_ms > 0
