"""Execute yielded receipts against real SQL before any SDK imports are possible."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from session.test_batch_children import complete, deliver
from session.test_core_yield_outcomes import yield_fixture
from session.test_recovery_matrix import start
from session.test_yieldable_waits import waiting_runtime
from sim.world import SimWorld

from vv_agent.session import kernel
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
@pytest.mark.parametrize("kind", ["provider", "children"])
def test_core_yield_receipt_matches_sdk_with_import_blocker(tmp_path, flags, kind):
    database = tmp_path / "prefix.sqlite"
    if kind == "provider":
        with SimWorld(database, seed=17) as world:
            worker, attempt, _, item = yield_fixture(world)
            expected = worker.yield_results([attempt], item)
            source = yield_facts(worker, attempt, item, expected, kind)
    else:
        with SQLiteStore.standalone(str(database)) as store:
            store.install_schema()
            start(store)
            runtime, _ = waiting_runtime(database, count=2)
            kernel.drive(store, "s", runtime=runtime)
            complete(store, database, "yield-child-0")
            deliver(store, "yield-child-0")
            kernel.drive(store, "s", runtime=runtime)
            lease = store.acquire("s", owner="yield-reader", ttl_ms=runtime.ttl_ms)
            assert lease is not None
            scope = kernel._Scope(lease, runtime, now_ms=store._now())
            worker = kernel._Driver(store, "s", runtime, scope)
            worker.refresh()
            attempt = next(a for op in worker.state.operations.values() for a in op.attempts.values() if a.state == "parked")
            item = InboxItem("steer-proof", "steer", {"content": "change"}, attempt.plan.turn_id, scope.generation)
            expected = worker.yield_results([attempt], item)
            receipt = next(r for r in expected if r.kind == "op_completed")
            metadata = receipt.payload["result"]["metadata"]
            assert metadata["running_children"] == ["yield-child-1"]
            assert metadata["children"] == [
                {"session_id": "yield-child-0", "status": "completed"},
                {"session_id": "yield-child-1", "status": "running"},
            ]
            source = yield_facts(worker, attempt, item, expected, kind)
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(source))
    script = r"""
import importlib.abc
import json
import sys
from pathlib import Path
from threading import Event, Lock, local, get_ident
sys.path.insert(0, sys.argv[1])
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {'vv_agent','vv_agent.kernel'} or fullname.startswith('vv_agent.kernel.')
                or fullname.split('.')[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_outcomes import yield_results_steps, LookupYieldCallback, PrepareToolResources
from vv_agent.kernel._lease_scope import LeaseOwnership
from vv_agent.kernel._records import InboxItem
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._sqlite import SQLiteStore
from vv_agent.kernel._store import Conflict
source=json.loads(Path(sys.argv[3]).read_text())
class Clock:
    def now_ms(self): return source['clock_ms']
    def monotonic(self): return self.now_ms()/1000
    def monotonic_ns(self): return self.now_ms()*1000000
    def sleep(self,seconds): raise AssertionError('no wait requested')
clock=Clock()
def unused(*args,**kwargs): raise AssertionError('no compaction in produced prefix')
def factory(): return Fold(normalize_memory_settings=unused,count_messages=unused)
class Scope(LeaseOwnership):
    turn_id=None
    generation=None
with SQLiteStore.standalone(sys.argv[2],clock=clock,fold_factory=factory) as store:
    lease=store.acquire('s',owner='core-yield',ttl_ms=15000)
    assert lease is not None
    scope=Scope(lease,15000,.25,clock=clock,stop=Event(),ready=Event(),lock=Lock(),poll_lock=Lock(),
                commit_lock=Lock(),latch_entry=local(),actor_identity=get_ident)
    owner=CoreDriver(store,'s',scope,fold_factory=factory,clock=clock,observe_cost=lambda:None,
                     hook=lambda *a:None,emit_phase=lambda *a:None)
    owner.bind_turn()
    attempt=owner.state.operations[source['operation_id']].attempts[source['attempt']]
    steps=yield_results_steps(owner,[attempt],InboxItem(**source['item']))
    response=None
    seen=[]
    while True:
        try: request=steps.send(response)
        except StopIteration as done:
            result=done.value
            break
        seen.append(type(request).__name__)
        if isinstance(request,LookupYieldCallback):
            assert request.child==(source['kind']=='children')
            response=None
        elif isinstance(request,PrepareToolResources):
            # This produced text result has no resource needing a host backend.
            assert request.result==source['resource_result']
            response=request.result
        else: raise AssertionError(request)
    assert seen==['LookupYieldCallback','PrepareToolResources']
    assert [record.to_dict() for record in result]==source['records']
    # A receipt alone must still fail the actual storage atomic-steering fence.
    before=store._head('s')[:2]
    try: owner.commit(result)
    except Conflict as error: assert str(error)=='yield completion and steering must commit together'
    else: raise AssertionError('yield receipt committed without steering')
    assert store._head('s')[:2]==before
    state,rows,_=store.read_state('s')
    receipts=[row.record for row in rows if row.record.kind=='op_completed'
              and row.record.operation_id==source['operation_id']]
    assert receipts==[]
assert all(n in {'vv_agent','vv_agent.kernel'} or n.startswith('vv_agent.kernel.')
           for n in sys.modules if n.startswith('vv_agent'))
"""
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(Path(__file__).parents[1] / "src"), str(database), str(facts)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def yield_facts(worker, attempt, item, records, kind):
    receipt = next(r for r in records if r.kind == "op_completed")
    result = {
        "kind": kind,
        "operation_id": attempt.plan.operation_id,
        "attempt": attempt.plan.attempt,
        "clock_ms": worker.store._now(),
        "item": item.to_dict(),
        "records": [r.to_dict() for r in records],
        "resource_result": receipt.payload["result"],
    }
    assert worker.store.release(worker.scope.lease)
    return result
