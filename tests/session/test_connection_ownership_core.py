"""Early vetoes: actual host SQL and two-child admission share the owned transaction."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from vv_agent.session.callbacks import CallbackContractError
from vv_agent.session.children import ChildSession, InvalidChildBatch
from vv_agent.session.kernel import drive
from vv_agent.session.ownership import OwnershipError
from vv_agent.session.records import SessionSpec
from vv_agent.session.sql import SQLSessionTx

from .test_children import parent_runtime
from .test_recovery_matrix import start


@pytest.mark.parametrize("cut", ["commit", "invalid", "append", "outer_commit"])
@pytest.mark.parametrize("archive_parameter", ["raw_connection", "routed_sql"])
def test_hostrows_atomic(store, database, monkeypatch, cut, archive_parameter):
    start(store)
    store._rows("CREATE TABLE host_effect (id integer PRIMARY KEY)")
    store._rows("CREATE TABLE host_before (id integer PRIMARY KEY)")
    store._rows("CREATE TABLE host_reference (id integer PRIMARY KEY)")
    store._rows("CREATE TABLE host_deferred (id integer REFERENCES host_reference(id) DEFERRABLE INITIALLY DEFERRED)")
    admitted = []
    hooks = []
    wakes = []
    error = RuntimeError("append cut")
    append = SQLSessionTx.append

    def fail_append(tx, *args, **kw):
        if cut == "append" and any(r.kind == "op_parked" for r in kw["records"]):
            raise error
        return append(tx, *args, **kw)

    monkeypatch.setattr(SQLSessionTx, "append", fail_append)

    def admit(plan, *, tx):
        assert tx.store.connection is tx.connection
        assert (tx.connection is store.connection) == (getattr(store, "path", None) == ":memory:")
        tx.connection.execute("INSERT INTO host_effect VALUES (1)")
        admitted.append(tx)
        if cut == "outer_commit":
            tx.connection.execute("INSERT INTO host_deferred VALUES (99)")
        return [] if cut == "invalid" else [ChildSession(SessionSpec(f"owned-{i}", "test", "/tmp"), "go") for i in range(2)]

    def hook(point, record, *, tx):
        if point == "before_commit" and record and record.kind == "op_parked":
            assert tx is admitted[0]
            tx.connection.execute("INSERT INTO host_before VALUES (1)")
        if point == "after_commit" and record and record.kind == "op_parked":
            assert tx is None
            hooks.append(1)

    rt = parent_runtime(database)
    rt.children["child_task"] = admit
    rt.hook = hook
    rt.wake = wakes.append
    if cut == "commit":
        assert drive(store, "s", runtime=rt)
    elif cut == "invalid":
        with pytest.raises(InvalidChildBatch):
            drive(store, "s", runtime=rt)
    elif cut == "append":
        with pytest.raises(RuntimeError) as caught:
            drive(store, "s", runtime=rt)
        assert caught.value is error
    else:
        if hasattr(store, "path"):
            import sqlite3

            failure = sqlite3.IntegrityError
        else:
            import psycopg

            failure = psycopg.errors.ForeignKeyViolation
        with pytest.raises(failure):
            drive(store, "s", runtime=rt)
    assert len(admitted) == 1

    def observe(reader):
        return (
            reader._one("SELECT count(*) FROM host_effect")[0],
            reader._one("SELECT count(*) FROM host_before")[0],
            reader._one("SELECT count(*) FROM sk_session WHERE parent_session_id='s'")[0],
            reader._one("SELECT count(*) FROM sk_inbox WHERE session_id IN ('owned-0','owned-1')")[0],
            reader._one("SELECT count(*) FROM sk_record WHERE session_id='s' AND kind='op_parked'")[0],
        )

    if getattr(store, "path", None) == ":memory:":
        with ThreadPoolExecutor(1) as pool:
            counts = pool.submit(observe, store).result(5)
    else:
        with store.execution_store() as reader:
            counts = observe(reader)
    assert counts == ((1, 1, 2, 2, 1) if cut == "commit" else (0, 0, 0, 0, 0))
    assert hooks == ([1] if cut == "commit" else [])
    records = store.read_state("s")[1]
    assert any(r.record.kind == "op_completed" and r.record.payload.get("result", {}).get("kind") != "child" for r in records)
    with pytest.raises(OwnershipError, match="transaction_scope"):
        _ = admitted[0].connection
    print(
        "CO2-A01/A02/A03",
        type(store).__name__,
        archive_parameter,
        cut,
        counts,
        "admit",
        len(admitted),
        "aftercommit",
        len(hooks),
        flush=True,
    )


@pytest.mark.parametrize("archive_parameter", ["raw_connection", "routed_sql"])
def test_old_arity_rejected_without_host_sql(store, database, archive_parameter):
    start(store)
    calls = []

    store._rows("CREATE TABLE old_host_effect (id integer PRIMARY KEY)")
    original = store.connection

    def old(plan):
        calls.append(plan)
        with store.atomic():
            if archive_parameter == "raw_connection":
                original.execute("INSERT INTO old_host_effect VALUES (1)")
            else:
                store._rows("INSERT INTO old_host_effect VALUES (1)")
        return []

    rt = parent_runtime(database)
    rt.children["child_task"] = old
    with pytest.raises(CallbackContractError, match="transaction_callback_contract"):
        drive(store, "s", runtime=rt)
    assert calls == []
    assert store._one("SELECT count(*) FROM old_host_effect")[0] == 0
    assert store._one("SELECT count(*) FROM sk_record WHERE kind='op_planned'")[0] == 0
