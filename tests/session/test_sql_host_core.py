"""Caller-owned PostgreSQL stores and business admission share the real SQL transaction."""

from contextlib import nullcontext
from threading import RLock

import pytest

from vv_agent.kernel import _driver_dispatch
from vv_agent.session import kernel
from vv_agent.session.children import ChildSession, ChildTool, InvalidChildBatch
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.sql import SQLSessionTx
from vv_agent.session.store import LeaseLost, SequenceConflict
from vv_agent.session.supervisor import tick

from .conftest import open_store
from .helpers import fixture_child_spec, read_fixture_state, wrap_fixture_store
from .test_children import parent_runtime
from .test_recovery_matrix import start


class HostPostgresStore(PostgresStore):
    """Django-style raw connection property, without superclass initialization."""

    def __init__(self, raw):
        self.raw = raw
        self._connection_lock = RLock()

    @property
    def connection(self):
        return self.raw


@pytest.fixture
def store(store):
    return wrap_fixture_store(store)


def rows(store, sid="s", kind=None):
    return [r for r in read_fixture_state(store, sid)[1] if kind is None or r.record.kind == kind]


@pytest.mark.postgres
def test_public_drive_and_dispatch_tick_preserve_host_instance(store, database, monkeypatch):
    raw_host = HostPostgresStore(store.connection)
    host = wrap_fixture_store(raw_host)
    assert "connection" not in vars(host)
    epochs, claimed_epochs, peeks, observations = [], [], [], []
    acquire, claim, peek = host.acquire, host.claim_runnable, host.peek_inbox

    def capture_acquire(*args, **kwargs):
        observations.append((host, raw_host, host.connection, raw_host.connection))
        lease = acquire(*args, **kwargs)
        epochs.append(None if lease is None else lease.epoch)
        return lease

    def capture_claim(*args, **kwargs):
        observations.append((host, raw_host, host.connection, raw_host.connection))
        claimed = claim(*args, **kwargs)
        claimed_epochs.extend(None if item.lease is None else item.lease.epoch for item in claimed)
        return claimed

    def capture_peek(*args, **kwargs):
        observations.append((host, raw_host, host.connection, raw_host.connection))
        peeks.append(args[0])
        return peek(*args, **kwargs)

    monkeypatch.setattr(host, "acquire", capture_acquire)
    monkeypatch.setattr(host, "claim_runnable", capture_claim)
    monkeypatch.setattr(host, "peek_inbox", capture_peek)
    start(host)
    rt = parent_runtime(host, database)
    assert kernel.drive(host, "s", runtime=lambda _sid: rt)
    assert epochs == [1] and peeks and set(peeks) == {"s"}
    assert {r.writer_epoch for r in rows(host) if r.writer_epoch} == {epochs[0]}
    start(host, "queued")
    dispatched, projected = [], []

    def forbidden(_sid):
        raise AssertionError("dispatch-only tick must not build a runtime")

    def project(sid, consumer):
        projected.append((sid, consumer))
        with host.atomic() as tx:
            batch = tx.consumer_batch(sid, consumer)
            if batch:
                tx.ack(batch)

    assert (
        tick(
            host,
            dispatch_claimed=lambda sid, lease: dispatched.append(sid),
            runtime=forbidden,
            project=project,
            page_size=1,
        )
        >= 1
    )
    assert claimed_epochs and "queued" in dispatched and epochs == [1]
    assert projected
    assert all(
        adapter is host and root is raw_host and connection is raw_connection is root.connection
        for adapter, root, connection, raw_connection in observations
    )
    assert host._store is raw_host and host._root is raw_host
    print(
        "HOST_INSTANCE",
        {"epochs": epochs, "claim_epochs": claimed_epochs, "peek_calls": len(peeks), "dispatch": dispatched},
    )


class PlannedCut(BaseException):
    pass


@pytest.mark.postgres
@pytest.mark.parametrize("fault", ["invalid", "child_2", "cas", "lease", "commit", None])
def test_child_admission_business_and_session_increment_is_atomic(store, database, monkeypatch, fault):
    import psycopg

    raw_host = HostPostgresStore(store.connection)
    host = wrap_fixture_store(raw_host)
    host._rows("CREATE TABLE allowed (id integer PRIMARY KEY)")
    host._rows("INSERT INTO allowed VALUES (1)")
    host._rows(
        "CREATE TABLE business_child (id integer PRIMARY KEY, marker integer REFERENCES allowed(id) "
        "DEFERRABLE INITIALLY DEFERRED)"
    )
    start(host)
    rt = parent_runtime(host, database)
    admissions = []

    def admit(_plan):
        admissions.append(1)
        assert host.connection is store.connection
        for i in range(2):
            host._rows("INSERT INTO business_child VALUES (%s,%s)", (i, 2 if fault == "commit" else 1))
        specs = [
            ChildSession(
                fixture_child_spec(host, "s", f"business-{i}"),
                "go",
                yield_on_steer=True,
            )
            for i in range(2)
        ]
        return [*specs, object()] if fault == "invalid" else specs

    rt.children["child_task"] = ChildTool(admit)

    def cut(point, record):
        if point == "after_commit" and record and record.kind == "op_planned" and record.payload["op_kind"] == "tool":
            raise PlannedCut()

    rt.hook = cut
    with pytest.raises(PlannedCut):
        kernel.drive(host, "s", runtime=rt)
    rt.hook = lambda *_: None
    prefix = [r.record.encode() for r in rows(host)]
    create = _driver_dispatch.create_child
    append = SQLSessionTx.append
    append_returned = []

    def create_child(tx, child, plan, generation):
        result = create(tx, child, plan, generation)
        if fault == "child_2" and child.spec.session_id == "business-1":
            raise RuntimeError("second child failed")
        return result

    def fault_append(tx, sid, **kwargs):
        if any(r.kind == "op_parked" for r in kwargs["records"]):
            if fault == "cas":
                kwargs["expected_seq"] -= 1
            if fault == "lease":
                # Invalidate the database row after the driver's late guard sample.
                guard = tx._lease_guard
                assert guard is not None
                kwargs["lease"] = guard()
                host._rows(
                    "UPDATE sk_session SET lease_epoch=lease_epoch+1,lease_until_ms=0 WHERE tenant_id=%s AND session_id=%s",
                    (host.tenant_id, sid),
                )
        receipt = append(tx, sid, **kwargs)
        append_returned.append(receipt)
        return receipt

    monkeypatch.setattr(_driver_dispatch, "create_child", create_child)
    monkeypatch.setattr(SQLSessionTx, "append", fault_append)
    lease = host.acquire("s", owner="one-admission", ttl_ms=15000)
    assert lease is not None
    scope = kernel._Scope(lease, rt, now_ms=host._now())
    driver = kernel._Driver(host, "s", rt, scope)
    driver.bind_turn()
    scope.poll(host)
    attempt = next(op.attempts[1] for op in driver.state.operations.values() if op.kind == "tool")
    errors = {
        "invalid": InvalidChildBatch,
        "child_2": RuntimeError,
        "cas": SequenceConflict,
        "lease": LeaseLost,
        "commit": psycopg.errors.ForeignKeyViolation,
    }
    try:
        with pytest.raises(errors[fault]) if fault else nullcontext():
            driver.dispatch(attempt)
    finally:
        host.release(scope.lease)
    assert admissions == [1]
    if fault:
        assert host._rows("SELECT * FROM business_child") == []
        assert host.list_sessions() == ("s",)
        assert [r.record.encode() for r in rows(host)] == prefix
        assert host._one("SELECT count(*) FROM sk_inbox WHERE session_id != 's'")[0] == 0
        assert not rows(host, kind="op_parked")
        if fault == "commit":
            # append returned normally; only the real outer COMMIT rejected the FK.
            assert len(append_returned) == 1
    else:
        assert host._rows("SELECT * FROM business_child ORDER BY id") == [(0, 1), (1, 1)]
        assert host.list_sessions() == ("business-0", "business-1", "s")
        assert len(rows(host, kind="op_parked")) == 1
        for sid in ("business-0", "business-1"):
            assert len(rows(host, sid, "session_created")) == 1
            assert len(host.peek_inbox(sid)) == 1
    # Independent read proves this is committed state, not the writer's view.
    with open_store(database) as raw_reader:
        reader = wrap_fixture_store(raw_reader)
        assert reader._one("SELECT count(*) FROM business_child")[0] == (0 if fault else 2)
        assert reader.list_sessions() == host.list_sessions()
    print("BUSINESS_ATOMIC", {"fault": fault, "admissions": len(admissions), "append_returned": len(append_returned)})
