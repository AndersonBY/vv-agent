"""Confirmed commits survive host callback/real factory-exit errors; original error wins."""

from contextlib import contextmanager

import pytest

from vv_agent.session.children import ChildSession
from vv_agent.session.kernel import drive
from vv_agent.session.ownership import checkout
from vv_agent.session.records import SessionSpec
from vv_agent.session.sql import SQLSessionTx

from .test_children import parent_runtime
from .test_recovery_matrix import start


@pytest.mark.parametrize("cut", ["after", "close", "both", "append_close"])
def test_commit_and_real_exit(store, database, monkeypatch, cut):
    start(store)
    store._rows("CREATE TABLE host_effect (id integer PRIMARY KEY)")
    store._rows("CREATE TABLE host_before (id integer PRIMARY KEY)")
    factory = store.execution_store
    admitted = []
    after = []
    wakes = []
    exits = []
    primary = ValueError("host callback failure")
    close_error = OSError("real checkout exit")
    append_error = RuntimeError("append failed")

    @contextmanager
    def closing():
        try:
            with factory() as execution:
                yield execution
        finally:
            exits.append(1)
            if cut in {"close", "both", "append_close"}:
                raise close_error

    monkeypatch.setattr(store, "execution_store", closing)
    append = SQLSessionTx.append

    def append_cut(tx, *a, **kw):
        if cut == "append_close" and any(r.kind == "op_parked" for r in kw["records"]):
            raise append_error
        return append(tx, *a, **kw)

    monkeypatch.setattr(SQLSessionTx, "append", append_cut)

    def admit(plan, *, tx):
        admitted.append(tx)
        tx.connection.execute("INSERT INTO host_effect VALUES (1)")
        return [ChildSession(SessionSpec(f"c{i}", "test", "/tmp"), "go") for i in range(2)]

    def hook(point, record, *, tx):
        if point == "before_commit" and record and record.kind == "op_parked":
            assert tx is admitted[0]
            tx.connection.execute("INSERT INTO host_before VALUES (1)")
        if point == "after_commit" and record and record.kind == "op_parked":
            after.append(1)
            if cut in {"after", "both"}:
                raise primary

    rt = parent_runtime(database)
    rt.children["child_task"] = admit
    rt.hook = hook
    rt.wake = wakes.append
    expected = append_error if cut == "append_close" else primary if cut in {"after", "both"} else close_error
    with pytest.raises(type(expected)) as caught:
        drive(store, "s", runtime=rt)
    assert caught.value is expected
    if cut in {"both", "append_close"}:
        assert any("checkout exit failed" in n for n in caught.value.__notes__)
    assert len(admitted) == len(exits) == 1
    durable = cut != "append_close"
    assert after == ([1] if durable else [])
    assert store._one("SELECT count(*) FROM host_effect")[0] == int(durable)
    assert store._one("SELECT count(*) FROM host_before")[0] == int(durable)
    assert store._one("SELECT count(*) FROM sk_session WHERE parent_session_id='s'")[0] == (2 if durable else 0)
    if cut in {"after", "both"}:
        assert {"c0", "c1"} <= set(wakes)
    monkeypatch.setattr(store, "execution_store", factory)
    if durable:
        rt.hook = lambda point, record, *, tx: None
        assert drive(store, "s", runtime=rt)
        assert len(admitted) == 1
    assert store._one("SELECT 1")[0] == 1
    print(
        "CO2-D01/L01",
        type(store).__name__,
        cut,
        "durable",
        durable,
        "admit",
        len(admitted),
        "after",
        len(after),
        "exit",
        len(exits),
        flush=True,
    )


@pytest.mark.persistent_store
def test_commit_ack_unknown_requires_identity_recovery(store, database, monkeypatch):
    start(store)
    calls = []
    unknown = ConnectionError("COMMIT acknowledgement unavailable")
    cls = type(store)
    original = cls._transaction

    @contextmanager
    def disconnect(candidate):
        depth = getattr(candidate, "_co2_depth", 0)
        candidate._co2_depth = depth + 1
        try:
            with original(candidate):
                yield
            if depth == 0 and getattr(candidate, "_co2_unknown", False):
                candidate._co2_unknown = False
                candidate.connection.close()
                raise unknown
        finally:
            candidate._co2_depth = depth

    monkeypatch.setattr(cls, "_transaction", disconnect)

    def admit(plan, *, tx):
        calls.append(1)
        tx.store._co2_unknown = True
        return [ChildSession(SessionSpec(f"unknown{i}", "test", "/tmp"), "go") for i in range(2)]

    rt = parent_runtime(database)
    rt.children["child_task"] = admit
    with pytest.raises(ConnectionError) as caught:
        drive(store, "s", runtime=rt)
    assert caught.value is unknown and calls == [1]
    # Resolve using original durable session/operation identity, never the provisional receipt.
    with checkout(store, role="projection") as reader:
        state, records, _ = reader.read_state("s")
        assert len(state.waits) == 1
        assert sum(r.record.kind == "op_parked" for r in records) == 1
    # The disconnected E cannot release its lease. Honor the original TTL;
    # no forced expiry, rescue wake or alternate admission is used.
    from threading import Event

    remaining = store.lease_status("s")[2]
    assert remaining > 0
    assert not drive(store, "s", runtime=rt)
    Event().wait(remaining / 1000 + 0.05)
    assert drive(store, "s", runtime=rt)
    assert calls == [1]


@pytest.mark.persistent_store
def test_host_close_does_not_close_owned_execution_and_reopen_uses_cache(store):
    from vv_agent.session.ownership import checkout

    start(store)
    with checkout(store) as execution:
        host = store.connection
        store.connection.close()
        assert execution.connection is not host
        assert execution.read_state("s")[0].phase == "idle"
        with checkout(store) as reopened:
            assert reopened.connection is not execution.connection
            assert reopened.read_state("s")[0].phase == "idle"
