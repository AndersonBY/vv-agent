"""Core child-ledger ownership and dependency isolation."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from vv_agent.kernel import _child_store as core
from vv_agent.kernel._records import Record, SessionSpec, make_record
from vv_agent.kernel._state import ExecutionState
from vv_agent.kernel._store import StoredRecord
from vv_agent.session import children
from vv_agent.session.sqlite import SQLiteStore


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
child = ChildSession(SessionSpec('child', {}, {}), 'work')
assert child.spec.session_id == 'child'
ref = {'content_id':'content/test/result','mac':'0'*64,'size':1,'key_id':'key',
       'labels':{'source':[],'classification':'internal'}}
terminal = make_record('turn_ended', session_id='child', turn_id='child/turn/start',
    payload={'status':'completed','reason':None,'result':ref,
             'budget':ref,'adopted_results':[],'unconfirmed_operations':[]})
handle = {'session_id':'child','turn_id':'child/turn/start',
          'delivery_target':{'session_id':'parent','turn_id':'parent/turn/start',
                            'generation':0,'operation_id':'op','attempt':1}}
item = completion_input(handle, StoredRecord(terminal, 8, 'commit', 1, 0))
assert item.target_turn_id == 'parent/turn/start' and item.generation == 0
assert item.payload['terminal_digest'] == terminal.digest
assert item.payload['result'] == ref
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
    child = children.ChildSession(SessionSpec("child", {}, {}), "work")
    with pytest.raises(FrozenInstanceError):
        child.__setattr__("content", "changed")


def test_child_outcome_rejects_an_unhydrated_content_ref() -> None:
    ref = {
        "content_id": "content/test/result",
        "mac": "0" * 64,
        "size": 1,
        "key_id": "key",
        "labels": {"source": [], "classification": "internal"},
    }
    terminal = make_record(
        "turn_ended",
        session_id="child",
        turn_id="child/turn/start",
        payload={
            "status": "completed",
            "reason": ref,
            "result": ref,
            "adopted_results": [],
            "budget": ref,
            "unconfirmed_operations": [],
        },
    )
    plan = Record(
        "plan",
        "op_planned",
        "parent",
        "parent/turn/start",
        "operation",
        1,
        {"request": {"id": "child-call"}},
    )

    with pytest.raises(ValueError, match="requires content hydration"):
        core.child_outcome(plan, StoredRecord(terminal, 1, "commit", 1, 0))


def test_ordinary_children_hydrate_only_each_terminal_result(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    ref = {
        "content_id": "content/test/result",
        "mac": "0" * 64,
        "size": 1,
        "key_id": "key",
        "labels": {"source": [], "classification": "internal"},
    }
    plan = Record(
        "plan",
        "op_planned",
        "parent",
        "parent/turn/start",
        "operation",
        1,
        {"request": {"id": "child-call"}},
    )

    def terminal(name: str, seq: int) -> StoredRecord:
        record = make_record(
            "turn_ended",
            session_id=name,
            turn_id=f"{name}/turn/start",
            payload={
                "status": "completed",
                "reason": ref,
                "result": ref,
                "adopted_results": [],
                "budget": ref,
                "unconfirmed_operations": [],
            },
        )
        return StoredRecord(record, seq, f"commit-{name}", 1, 0)

    first_terminal = terminal("child-a", 1)
    second_terminal = terminal("child-b", 2)
    first = (ExecutionState(session_id="child-a"), (first_terminal,), 0)
    second = (ExecutionState(session_id="child-b"), (second_terminal,), 0)
    state_reads: list[str] = []
    hydration_calls: list[tuple[tuple[str, str], str, str, str]] = []

    with SQLiteStore.standalone(str(tmp_path / "child.sqlite")) as store:

        def read_state(session_id: str):
            state_reads.append(session_id)
            return second

        class Hydration:
            def __init__(self, *, tenant_id: str, session_id: str) -> None:
                self.binding = (tenant_id, session_id)

            def read(
                self,
                owner: str,
                slot: str,
                content_ref: Any,
                *,
                record_kind: str | None = None,
                record_payload: dict[str, Any] | None = None,
            ) -> str:
                assert content_ref.content_id == ref["content_id"]
                assert record_kind == "turn_ended"
                assert record_payload is not None and record_payload["status"] == "completed"
                hydration_calls.append((self.binding, owner, slot, content_ref.content_id))
                return "hydrated-result"

        def from_store(cls, candidate, *, tenant_id: str, session_id: str):
            assert candidate is store
            return Hydration(session_id=session_id, tenant_id=tenant_id)

        monkeypatch.setattr(store, "read_state", read_state)
        monkeypatch.setattr(core.SessionHydration, "from_store", classmethod(from_store))
        result = core.ordinary_result_for_children(
            store,
            plan,
            [
                {"session_id": "child-a", "turn_id": "child-a/turn/start"},
                {"session_id": "child-b", "turn_id": "child-b/turn/start"},
            ],
            first_snapshot=first,
        )

    assert state_reads == ["child-b"]
    assert hydration_calls == [
        (("default", "child-a"), "record:1", "turn_ended_result", ref["content_id"]),
        (("default", "child-b"), "record:2", "turn_ended_result", ref["content_id"]),
    ]
    assert result.status_code is core.ToolResultStatus.SUCCESS
    assert json.loads(result.content) == [
        {
            "tool_call_id": "child-call",
            "content": "hydrated-result",
            "status_code": "SUCCESS",
            "directive": "continue",
            "metadata": {
                "child_session_id": "child-a",
                "child_turn_id": "child-a/turn/start",
                "status": "completed",
            },
        },
        {
            "tool_call_id": "child-call",
            "content": "hydrated-result",
            "status_code": "SUCCESS",
            "directive": "continue",
            "metadata": {
                "child_session_id": "child-b",
                "child_turn_id": "child-b/turn/start",
                "status": "completed",
            },
        },
    ]
