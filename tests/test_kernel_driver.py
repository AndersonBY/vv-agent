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
from threading import Event, Lock, local, get_ident
sys.path.insert(0, sys.argv[1])
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {'vv_agent','vv_agent.kernel'} or fullname.startswith('vv_agent.kernel.')
                or fullname.split('.')[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._lease_scope import LeaseOwnership
from vv_agent.kernel._sqlite import SQLiteStore
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._records import SessionSpec, InboxItem, digest
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
with SQLiteStore.standalone(sys.argv[2],clock=clock,fold_factory=factory) as store:
    store.install_schema()
    with store.atomic() as tx:
        tx.create(SessionSpec('s','p','w'), consumers=())
        item=InboxItem('i','user',{'content':'go'})
        tx.push('s',item)
    scope=Scope(store.acquire('s',owner='o',ttl_ms=15000),15000,.25,clock=clock,
        stop=Event(),ready=Event(),lock=Lock(),poll_lock=Lock(),commit_lock=Lock(),
        latch_entry=local(),actor_identity=get_ident)
    events=[]
    driver=CoreDriver(store,'s',scope,fold_factory=factory,clock=clock,
        observe_cost=lambda:None,hook=lambda point,r:events.append(point),
        emit_phase=lambda duration,fields:events.append(fields['phase']))
    applied=driver.record('input_applied',{'input':item.to_dict(),'input_digest':digest(item.to_dict()),
        'disposition':'queued','reason':None,'target_operation_id':None,'target_wait_id':None,'position':'before_dispatch'})
    driver.commit([applied],consumed=('i',),guarded=True)
    assert store.read_input('s','i').consumed_seq==2 and len(driver.records)==2
    assert driver.fold.consumed['i'].encode()==item.encode()
    driver.commit([applied],consumed=('i',))
    assert len(driver.records)==2 and not store.peek_inbox('s')
    assert events==['before_commit','before_commit','append','after_commit']*2
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
