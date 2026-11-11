"""Real append predicates reject after host SQL, without driver retry hiding the cut."""

import pytest

from vv_agent.session.callbacks import noop_hook
from vv_agent.session.children import ChildSession
from vv_agent.session.kernel import _Driver, _Scope, drive
from vv_agent.session.ownership import checkout
from vv_agent.session.records import SessionSpec
from vv_agent.session.store import LeaseLost, SequenceConflict

from .test_children import parent_runtime
from .test_recovery_matrix import start


@pytest.mark.parametrize("fault", ["head", "inbox", "epoch", "expired"])
def test_actual_append_predicates(store, database, fault):
    start(store)
    store._rows("CREATE TABLE fenced_host (id integer PRIMARY KEY)")

    class AdmissionCut(Exception):
        pass

    rt = parent_runtime(database)

    def pause(plan, *, tx):
        raise AdmissionCut()

    rt.children["child_task"] = pause
    with pytest.raises(AdmissionCut):
        drive(store, "s", runtime=rt)
    baseline = tuple(r.record.encode() for r in store.read_state("s")[1])
    calls = []
    after = []
    assert any(r.record.kind == "op_completed" for r in store.read_state("s")[1])

    def admit(plan, *, tx):
        calls.append(tx)
        tx.connection.execute("INSERT INTO fenced_host VALUES (1)")
        return [ChildSession(SessionSpec(f"fenced-{i}", "test", "/tmp"), "go") for i in range(2)]

    def hook(point, record, *, tx):
        if point == "before_commit" and record and record.kind == "op_parked":
            changes = {
                "head": "head_seq=head_seq+1",
                "inbox": "inbox_seq=inbox_seq+1",
                "epoch": "lease_epoch=lease_epoch+1",
                "expired": "lease_until_ms=0",
            }
            tx.store._rows("UPDATE sk_session SET " + changes[fault] + " WHERE session_id='s'")
        if point == "after_commit":
            after.append(record)

    rt.children["child_task"] = admit
    rt.hook = hook
    # Exercise the same actual producer once, exposing its real CAS/lease error
    # instead of allowing public drive's normal SequenceConflict retry loop.
    with checkout(store) as execution:
        lease = execution.acquire("s", owner="fence-probe", ttl_ms=rt.ttl_ms)
        assert lease is not None
        scope = _Scope(lease, rt, source=execution)
        scope.thread.start()
        try:
            assert scope.ready.wait(5) and scope.successful_ready and not scope.lost
            driver = _Driver(execution, "s", rt, scope)
            op = next(op for op in driver.state.operations.values() if op.kind != "model")
            expected = SequenceConflict if fault in {"head", "inbox"} else LeaseLost
            with pytest.raises(expected) as caught:
                driver.dispatch(op.attempts[max(op.attempts)])
            assert type(caught.value) is expected
        finally:
            scope.close()
            execution.release(scope.lease)
    assert len(calls) == 1 and after == []
    assert store._one("SELECT count(*) FROM fenced_host")[0] == 0
    assert store.list_sessions() == ("s",)
    assert tuple(r.record.encode() for r in store.read_state("s")[1]) == baseline
    rt.hook = noop_hook
    assert drive(store, "s", runtime=rt)
    assert len(calls) == 2
    assert store._one("SELECT count(*) FROM fenced_host")[0] == 1
    assert store.list_sessions() == ("fenced-0", "fenced-1", "s")
    print("CO2-A03", type(store).__name__, fault, "failed_delta_zero", True, "successful_admission", 1, flush=True)
