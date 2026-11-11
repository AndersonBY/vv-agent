"""Independent committed readers cannot observe a partial two-child admission."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from vv_agent.session.children import ChildSession
from vv_agent.session.kernel import drive
from vv_agent.session.records import SessionSpec

from .test_children import parent_runtime
from .test_recovery_matrix import start


@pytest.mark.parametrize("background", [False, True])
def test_reader_before_and_after_outer_commit(store, database, background):
    start(store)
    store._rows("CREATE TABLE business_admit (id integer PRIMARY KEY)")
    store._rows("CREATE TABLE business_before (id integer PRIMARY KEY)")
    admitted = []
    started = Event()
    finished = Event()
    future = []

    def counts(reader):
        return (
            reader._one("SELECT count(*) FROM business_admit")[0],
            reader._one("SELECT count(*) FROM business_before")[0],
            reader._one("SELECT count(*) FROM sk_session WHERE parent_session_id='s'")[0],
            reader._one("SELECT count(*) FROM sk_inbox WHERE session_id IN ('visible-0','visible-1')")[0],
            reader._one("SELECT count(*) FROM sk_record WHERE session_id='s' AND kind='op_parked'")[0],
        )

    def observer():
        started.set()
        result = counts(store)
        finished.set()
        return result

    def admit(plan, *, tx):
        admitted.append(tx)
        tx.connection.execute("INSERT INTO business_admit VALUES (1)")
        return [ChildSession(SessionSpec(f"visible-{i}", "test", "/tmp"), "go", background=background) for i in range(2)]

    with ThreadPoolExecutor(1) as pool:

        def hook(point, record, *, tx):
            if (
                point == "before_commit"
                and record
                and (
                    record.kind == "op_parked"
                    or (record.kind == "op_completed" and record.operation_id and "/tool/" in record.operation_id)
                )
            ):
                assert tx is admitted[0]
                tx.connection.execute("INSERT INTO business_before VALUES (1)")
                if getattr(store, "path", None) == ":memory:":
                    future.append(pool.submit(observer))
                    assert started.wait(2)
                    assert not finished.is_set()
                else:
                    with store.execution_store() as reader:
                        assert reader.connection is not tx.connection
                        assert counts(reader) == (0, 0, 0, 0, 0)

        rt = parent_runtime(database)
        rt.children["child_task"] = admit
        rt.hook = hook
        assert drive(store, "s", runtime=rt)
        if future:
            assert future[0].result(5) == (1, 1, 2, 2, 1)
    assert counts(store) == (1, 1, 2, 2, 1)
    assert len(admitted) == 1
    rows = store.read_state("s")[1]
    parked = next(r for r in rows if r.record.kind == "op_parked")
    started_record = next(
        r for r in rows if r.record.kind == "op_started" and r.record.operation_id == parked.record.operation_id
    )
    assert parked.commit_id == started_record.commit_id
    assert len(rt.llm.steps) == (0 if background else 1)
    print(
        "CO2-A01-COMMITTED-READER",
        type(store).__name__,
        background,
        "before",
        (0, 0, 0, 0, 0),
        "after",
        (1, 1, 2, 2, 1),
        flush=True,
    )
