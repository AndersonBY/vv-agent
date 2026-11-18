"""CoreDriver imports and owns actual SQL commits without any SDK module."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_core_driver_commits_real_sqlite_without_sdk(tmp_path, flags):
    script = r"""
import importlib.abc
import sys
from pathlib import Path
from threading import Event
root=Path(sys.argv[1]).parent
sys.path.insert(0,str(root / 'src'))
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {'vv_agent','vv_agent.kernel'} or fullname.startswith('vv_agent.kernel.')
                or fullname.split('.')[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel._content import KeyDestroyOutcome, KeyDestroyStatus, Principal, WrappedDataKey
from vv_agent.kernel._content_publication import declare_input_content, link_declared_input_content
from vv_agent.kernel._content_runtime import SessionContentWriter, bind_content_runtime, wrap_session_content_store
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._lease_scope import LeaseOwnership
from vv_agent.kernel._sql import stored_input_proof
from vv_agent.kernel._sqlite import SQLiteStore
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._records import InboxItem, SessionSpec, digest
class Clock:
    def now_ms(self): return 1000
    def monotonic(self): return 1.0
    def monotonic_ns(self): return 1000000000
    def sleep(self, seconds): raise AssertionError('no scheduler')
clock=Clock()
def unused(*args,**kwargs): raise AssertionError('no compaction requested')
def factory(): return Fold(normalize_memory_settings=unused, count_messages=unused)
class Scope(LeaseOwnership):
    turn_id=None
    generation=None
class Token:
    def cancel(self,reason): raise AssertionError(f'unexpected cancellation: {reason}')
class KeyProvider:
    provider_id='core-fixture'
    def generate_data_key(self,tenant_id,session_id):
        return WrappedDataKey(tenant_id,session_id,'key/core',b'k'*32)
    def unwrap(self,tenant_id,session_id,wrapped_key):
        return wrapped_key.wrapped_key
    def destroy(self,tenant_id,key_id):
        return KeyDestroyOutcome(True,KeyDestroyStatus.DESTROYED,f'destroy/{tenant_id}/{key_id}')
class Cipher:
    def __init__(self): self.counter=0
    def encrypt(self,key,nonce,plaintext,aad): return b'encrypted:'+plaintext
    def decrypt(self,key,nonce,ciphertext,aad): return ciphertext.removeprefix(b'encrypted:')
    def random_bytes(self,count):
        self.counter+=1
        seed=str(self.counter).encode()
        return (seed*((count+len(seed)-1)//len(seed)))[:count]
def create_session(store):
    with store.atomic() as tx:
        writer=SessionContentWriter.bind(tx,tenant_id=store.tenant_id,session_id='s')
        def spec(content):
            actor,_=content.service_principal('fixture/host')
            return SessionSpec('s',actor,content.put_json('/tmp').to_dict(),attributes={'seed':{'messages':[],'shared_state':{}}})
        writer.create_session(spec,consumers=())
def push_input(store):
    with store.atomic() as tx:
        writer=SessionContentWriter.bind(tx,tenant_id=store.tenant_id,session_id='s')
        actor=Principal(store.tenant_id,'fixture/user','user',writer.put_json({'role':'fixture/user'})).to_dict()
        published=declare_input_content(writer,InboxItem('i','user',{'content':'go'},actor=actor))
        tx.push('s',published.item)
        link_declared_input_content(writer,published.item)
        return published.item
with SQLiteStore.standalone(sys.argv[2],clock=clock,fold_factory=factory) as raw_store:
    raw_store.install_schema()
    binding=bind_content_runtime(raw_store,key_provider=KeyProvider(),cipher=Cipher())
    store=wrap_session_content_store(raw_store,binding)
    create_session(store)
    item=push_input(store)
    stored=store.read_input('s','i')
    assert stored is not None
    lease=store.acquire('s',owner='o',ttl_ms=15000)
    assert lease is not None
    scope=Scope(lease,15000,.25,clock=clock,stop=Event(),ready=Event(),now_ms=clock.now_ms(),token_factory=Token)
    events=[]
    driver=CoreDriver(store,'s',scope,fold_factory=factory,clock=clock,
        observe_cost=lambda:None,hook=lambda point,r:events.append(point),
        emit_phase=lambda duration,fields:events.append(fields['phase']))
    applied=driver.record('input_applied',{'input':item.to_dict(),'input_digest':digest(item.to_dict()),
        'input_proof':stored_input_proof(stored),
        'disposition':'queued','reason':None,'target_operation_id':None,'target_wait_id':None,'position':'before_dispatch'})
    driver.commit([applied],consumed=('i',),guarded=True)
    assert store.read_input('s','i').consumed_seq==2 and len(driver.records)==2
    assert driver.fold.consumed['i'].encode()==item.encode()
    driver.commit([applied],consumed=('i',))
    assert len(driver.records)==2 and not store.peek_inbox('s')
    assert events==[
        'before_commit','publish','before_commit','append','logical_preflight','finalize','after_commit',
        'before_commit','publish','before_commit','append','finalize','after_commit',
    ]
    assert driver.store is store and driver.scope is scope
assert all(n in {'vv_agent','vv_agent.kernel'} or n.startswith('vv_agent.kernel.')
           for n in sys.modules if n.startswith('vv_agent'))
"""
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(Path(__file__).parents[1] / "src"), str(tmp_path / "core.sqlite")],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
