"""Core child-ledger ownership and dependency isolation."""

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from vv_agent.kernel import _child_store as core
from vv_agent.kernel._records import SessionSpec
from vv_agent.session import children


@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_child_ledger_imports_without_sdk(tmp_path, site_flags):
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {'vv_agent', 'vv_agent.kernel'}
                or fullname.startswith('vv_agent.kernel.')
                or fullname.split('.')[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel._child_store import ChildSession, completion_input
from vv_agent.kernel._records import SessionSpec, make_record
from vv_agent.kernel._store import StoredRecord
child = ChildSession(SessionSpec('child', 'p', 'w'), 'work')
assert child.spec.session_id == 'child'
terminal = make_record('turn_ended', session_id='child', turn_id='child/turn/start',
    payload={'status':'completed','reason':None,'result':{'text':'雪','values':[True,1]},
             'budget':{},'adopted_results':[],'unconfirmed_operations':[]})
handle = {'session_id':'child','turn_id':'child/turn/start',
          'delivery_target':{'session_id':'parent','turn_id':'parent/turn/start',
                            'generation':0,'operation_id':'op','attempt':1}}
item = completion_input(handle, StoredRecord(terminal, 8, 'commit', 1, 0))
assert item.target_turn_id == 'parent/turn/start' and item.generation == 0
assert item.payload['terminal_digest'] == terminal.digest
assert item.payload['result'] == {'text':'雪','values':[True,1]}
assert all(name in {'vv_agent', 'vv_agent.kernel'} or name.startswith('vv_agent.kernel.')
           for name in sys.modules if name.startswith('vv_agent'))
"""
    result = subprocess.run(
        [sys.executable, "-I", *site_flags, "-B", "-c", script, str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_session_child_exports_are_actual_core_objects():
    for name in (
        "ChildSession",
        "ChildTool",
        "SummaryChildTool",
        "InvalidChildBatch",
        "cancel_children",
        "create_child",
        "child_delivery",
        "verify_completion",
        "child_handles",
    ):
        assert getattr(children, name) is getattr(core, name)
    child = children.ChildSession(SessionSpec("child", "p", "w"), "work")
    with pytest.raises(FrozenInstanceError):
        child.__setattr__("content", "changed")
