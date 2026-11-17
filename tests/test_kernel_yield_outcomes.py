"""Execute yielded receipts against real SQL before any SDK imports are possible."""

import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from session.test_batch_children import complete, deliver
from session.test_core_yield_outcomes import yield_fixture
from session.test_recovery_matrix import start
from session.test_yieldable_waits import waiting_runtime
from support.s1_fixture_values import wrap_fixture_store
from support.sdk_runtime import SDKWorld

from vv_agent.kernel._content import ContentRef, KeyDestroyOutcome, KeyDestroyStatus, KeyUnavailable, WrappedDataKey
from vv_agent.session import kernel
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore


class TestKeyProvider:
    provider_id = "sim-scripted-v1"

    @staticmethod
    def _require_binding(tenant_id, session_id, wrapped_key):
        if (wrapped_key.tenant_id, wrapped_key.session_id) != (tenant_id, session_id):
            raise KeyUnavailable("captured session key binding mismatch")

    def generate_data_key(self, tenant_id, session_id):
        return WrappedDataKey(tenant_id, session_id, f"key/{tenant_id}/{session_id}", b"captured-core-key")

    def restore_data_key(self, wrapped_key):
        self._require_binding(wrapped_key.tenant_id, wrapped_key.session_id, wrapped_key)

    def unwrap(self, tenant_id, session_id, wrapped_key):
        self._require_binding(tenant_id, session_id, wrapped_key)
        return b"k" * 32

    def destroy(self, tenant_id, key_id):
        return KeyDestroyOutcome(False, KeyDestroyStatus.KEY_UNAVAILABLE, None, "destroy is outside this test")


class _CapturingReader:
    def __init__(self, reader):
        self.reader = reader
        self.binding = reader.binding
        self.links: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    def read(self, tenant_id: str, session_id: str, owner: str, slot: str, ref: ContentRef) -> bytes:
        body = self.reader.read(tenant_id, session_id, owner, slot, ref)
        key = (tenant_id, session_id, owner, slot, ref.content_id)
        entry = {
            "tenant_id": tenant_id,
            "session_id": session_id,
            "owner": owner,
            "slot": slot,
            "ref": ref.to_dict(),
            "body": base64.b64encode(body).decode("ascii"),
        }
        previous = self.links.setdefault(key, entry)
        assert previous == entry
        return body

    def facts(self):
        return {
            "tenant_id": self.binding[0],
            "session_id": self.binding[1],
            "links": list(self.links.values()),
        }


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
@pytest.mark.parametrize("kind", ["provider", "children"])
def test_core_yield_receipt_matches_sdk_with_import_blocker(tmp_path, flags, kind):
    database = tmp_path / "prefix.sqlite"
    if kind == "provider":
        with SDKWorld(database, seed=17) as world:
            worker, attempt, _, item = yield_fixture(world)
            _, source = capture_yield_records(worker, attempt, item, kind)
    else:
        with SQLiteStore.standalone(str(database)) as raw_store:
            store = wrap_fixture_store(raw_store, TestKeyProvider())
            store.install_schema()
            start(store)
            runtime, _ = waiting_runtime(store, database, count=2)
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
            expected, source = capture_yield_records(worker, attempt, item, kind)
            receipt = next(r for r in expected if r.kind == "op_completed")
            metadata = receipt.payload["result"]["metadata"]
            assert metadata["running_children"] == ["yield-child-1"]
            assert metadata["children"] == [
                {"session_id": "yield-child-0", "status": "completed"},
                {"session_id": "yield-child-1", "status": "running"},
            ]
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(source))
    script = r"""
import base64
import importlib.abc
import json
import sys
from pathlib import Path
from threading import Event
sys.path.insert(0, sys.argv[1])
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {'vv_agent','vv_agent.kernel'} or fullname.startswith('vv_agent.kernel.')
                or fullname.split('.')[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_outcomes import yield_results_steps, LookupYieldCallback, PrepareToolResources
from vv_agent.kernel._content import ContentRef, KeyDestroyOutcome, KeyUnavailable, WrappedDataKey
from vv_agent.kernel._content_runtime import bind_content_runtime, wrap_session_content_store
from vv_agent.kernel._hydration import MissingContentLink
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
class Token:
    def cancel(self, reason): pass
class FactsReader:
    def __init__(self, facts):
        self.binding=(facts['tenant_id'],facts['session_id'])
        self.links={}
        for entry in facts['links']:
            ref=ContentRef.from_dict(entry['ref'])
            key=(entry['tenant_id'],entry['session_id'],entry['owner'],entry['slot'],ref.content_id)
            body=base64.b64decode(entry['body'],validate=True)
            if key in self.links and self.links[key] != (ref,body):
                raise MissingContentLink('conflicting captured content link')
            self.links[key]=(ref,body)
    def read(self,tenant_id,session_id,owner,slot,ref):
        if (tenant_id,session_id)!=self.binding:
            raise MissingContentLink('captured content session binding mismatch')
        value=self.links.get((tenant_id,session_id,owner,slot,ref.content_id))
        if value is None or value[0] != ref:
            raise MissingContentLink('captured content owner, slot, or reference mismatch')
        return value[1]
class PlainCipher:
    def __init__(self): self.counter=0
    def encrypt(self,key,nonce,plaintext,aad): return plaintext
    def decrypt(self,key,nonce,ciphertext,aad): return ciphertext
    def random_bytes(self,count):
        self.counter+=1
        return self.counter.to_bytes(count,'big')
class TestKeyProvider:
    provider_id='sim-scripted-v1'
    @staticmethod
    def _require_binding(tenant_id,session_id,wrapped_key):
        if (wrapped_key.tenant_id,wrapped_key.session_id)!=(tenant_id,session_id):
            raise KeyUnavailable('captured session key binding mismatch')
    def generate_data_key(self,tenant_id,session_id):
        return WrappedDataKey(tenant_id,session_id,f'key/{tenant_id}/{session_id}',b'captured-core-key')
    def restore_data_key(self,wrapped_key):
        self._require_binding(wrapped_key.tenant_id,wrapped_key.session_id,wrapped_key)
    def unwrap(self,tenant_id,session_id,wrapped_key):
        self._require_binding(tenant_id,session_id,wrapped_key)
        return b'k'*32
    def destroy(self,tenant_id,key_id):
        return KeyDestroyOutcome(False,'key_unavailable',None,'destroy is outside this test')
with SQLiteStore.standalone(sys.argv[2],clock=clock,fold_factory=factory) as store:
    store.configure_worker(handler_versions=(source['handler_version'],))
    binding=bind_content_runtime(store,key_provider=TestKeyProvider(),cipher=PlainCipher())
    store=wrap_session_content_store(store,binding)
    lease=store.acquire('s',owner='core-yield',ttl_ms=15000)
    assert lease is not None
    scope=Scope(lease,15000,.25,clock=clock,stop=Event(),ready=Event(),token_factory=Token)
    owner=CoreDriver(store,'s',scope,fold_factory=factory,clock=clock,observe_cost=lambda:None,
                     hook=lambda *a:None,emit_phase=lambda *a:None,
                     content_reader=FactsReader(source['content_reader']))
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
    now_ms = worker.store._now()
    delay_ms = worker.store.next_drive_delay_ms("s", lease=worker.scope.lease)
    result = {
        "kind": kind,
        "operation_id": attempt.plan.operation_id,
        "attempt": attempt.plan.attempt,
        "clock_ms": now_ms + (delay_ms or 0),
        "handler_version": worker.runtime.handler_version,
        "item": item.to_dict(),
        "records": [r.to_dict() for r in records],
        "resource_result": receipt.payload["result"],
    }
    assert worker.store.release(worker.scope.lease)
    return result


def capture_yield_records(worker, attempt, item, kind):
    capture = _CapturingReader(worker.content_reader)
    worker.content_reader = capture
    try:
        records = worker.yield_results([attempt], item)
        source = yield_facts(worker, attempt, item, records, kind)
        source["content_reader"] = capture.facts()
        return records, source
    finally:
        worker.content_reader = capture.reader
