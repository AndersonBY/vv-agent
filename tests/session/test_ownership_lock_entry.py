"""Same-thread insertion callbacks must be rejected before waiting on the ledger."""

import json
import subprocess
import sys
from pathlib import Path
from threading import Event, Thread

import pytest

from vv_agent.session import ownership


def test_trace_reentry_after_acquire_is_rejected_before_waiting(tmp_path):
    source = str(Path(ownership.__file__).resolve().parents[2])
    code = """import ast,inspect,json,os,sys,textwrap,threading,traceback
sys.path.insert(0,SOURCE)
from vv_agent.session import ownership
fn=ownership._ledger_lock.__wrapped__
lines,start=inspect.getsourcelines(fn)
tree=ast.parse(textwrap.dedent(''.join(lines)))
def is_ledger(n):
 return isinstance(n,ast.With) and any(isinstance(i.context_expr,ast.Name) and i.context_expr.id=='_lock' for i in n.items)
held=next(n for n in ast.walk(tree) if is_ledger(n))
target=start+held.body[0].lineno-1
facts=[];errors=[]
def trace(frame,event,arg):
 if frame.f_code is fn.__code__ and event=='line' and frame.f_lineno==target:
  sys.settrace(None)
  facts.append({'locked':ownership._lock.locked(),'owner':ownership._lock_owner})
  try:
   with ownership._ledger_lock():raise AssertionError('nested entry acquired ledger')
  except ownership.OwnershipError as error:
   assert error.stage=='lock-reentry';facts.append({'stage':error.stage})
 return trace
def worker():
 try:
  sys.settrace(trace)
  with ownership._ledger_lock():pass
  sys.settrace(None)
  with ownership._ledger_lock():pass
 except BaseException:errors.append(traceback.format_exc())
t=threading.Thread(target=worker,daemon=True);t.start();t.join(1)
print('LEDGER_ENTRY '+json.dumps({'facts':facts,'alive':t.is_alive(),'errors':errors}),flush=True)
if t.is_alive():os._exit(70)
assert not errors and facts[0]['locked'] and facts[0]['owner'] is None
assert facts[1]=={'stage':'lock-reentry'} and not ownership._lock.locked() and ownership._lock_owner is None
""".replace("SOURCE", repr(source))
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "TMPDIR": str(tmp_path), "LANG": "C.UTF-8"}
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-W", "error", "-c", code],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=5,
    )
    print(result.stdout, end="")
    assert result.returncode == 0, "same-thread ledger insertion blocked or failed: " + result.stdout
    records = [json.loads(line.split(" ", 1)[1]) for line in result.stdout.splitlines() if line.startswith("LEDGER_ENTRY ")]
    assert len(records) == 1 and not records[0]["alive"]


def test_ledger_entry_mark_clears_on_error_and_is_thread_local():
    checked = Event()
    errors = []

    def check_other_thread():
        try:
            ownership._check_lock_reentry()
            checked.set()
            with ownership._ledger_lock():
                assert ownership._lock_owner is not None
        except BaseException as error:
            errors.append(error)

    primary = ValueError("ledger body")
    with pytest.raises(ValueError) as caught, ownership._ledger_lock():
        with pytest.raises(ownership.OwnershipError) as nested:
            ownership._check_lock_reentry()
        assert nested.value.stage == "lock-reentry"
        worker = Thread(target=check_other_thread, daemon=True)
        worker.start()
        assert checked.wait(3)
        raise primary
    worker.join(3)
    assert caught.value is primary and not worker.is_alive() and not errors
    with ownership._ledger_lock():
        assert ownership._lock_owner is not None
