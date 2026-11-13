"""Recovery control flow stays in the stdlib core and runs on the real driver."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from session.test_yieldable_waits import YieldProvider
from sim.scenarios import Provider
from support.sdk_runtime import SDKWorld, sdk_driver, until

from vv_agent.kernel import _driver_outcomes as outcomes
from vv_agent.kernel import _driver_recovery as core
from vv_agent.session import _recovery_phase
from vv_agent.session.records import InboxItem


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_core_recovery_step_imports_without_sdk(tmp_path, flags):
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
from vv_agent.kernel._driver_recovery import Read, step
from vv_agent.kernel._lease_scope import LeaseOwnership
from vv_agent.kernel._records import SessionSpec
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._sqlite import SQLiteStore
class Clock:
    def now_ms(self): return 1000
    def monotonic(self): return 1.0
    def monotonic_ns(self): return 1000000000
    def sleep(self, seconds): raise AssertionError('no scheduler')
clock=Clock()
def unused(*args,**kwargs): raise AssertionError('unexpected compaction')
def factory(): return Fold(normalize_memory_settings=unused,count_messages=unused)
class Scope(LeaseOwnership):
    turn_id=None
    generation=None
with SQLiteStore.standalone(sys.argv[2],clock=clock,fold_factory=factory) as store:
    store.install_schema()
    with store.atomic() as tx: tx.create(SessionSpec('s','p','w'),consumers=())
    scope=Scope(store.acquire('s',owner='core',ttl_ms=15000),15000,.25,clock=clock,
        stop=Event(),ready=Event(),lock=Lock(),poll_lock=Lock(),commit_lock=Lock(),
        latch_entry=local(),actor_identity=get_ident)
    owner=CoreDriver(store,'s',scope,fold_factory=factory,clock=clock,observe_cost=lambda:None,
        hook=lambda *args:None,emit_phase=lambda *args:None)
    steps=step(owner,one_turn=True)
    assert next(steps) is Read.REFRESH
    steps.close()
assert not any(name.startswith('vv_agent.session') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(Path(__file__).parents[1] / "src"), str(tmp_path / "core.sqlite")],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_recovery_has_no_sdk_or_transport_effects():
    tree = ast.parse(Path(core.__file__).read_text())
    imported = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(module.startswith("vv_agent.session") for module in imported)
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
            continue
        if node.value.id in {"driver", "self"} and node.attr in {"runtime", "scope", "store"}:
            forbidden.append(node.attr)
    assert forbidden == []


def test_extracted_recovery_runs_complete_real_sql_turn(tmp_path):
    with SDKWorld(tmp_path / "recovery.sqlite", seed=17) as world:
        world.create()
        assert world.drive(provider=Provider(world))
        state, rows, _ = world.connection("observer").read_state("s")
        records = [row.record for row in rows]
        assert state.active_turn_id is None
        assert [r._payload["status"] for r in records if r.kind == "turn_ended"] == ["completed"]
        completed = [r.operation_id or "" for r in records if r.kind == "op_completed"]
        assert any(operation_id.endswith("/model/primary/1") for operation_id in completed)
        assert any(operation_id.endswith("/model/primary/2") for operation_id in completed)
        assert any("/tool/" in operation_id for operation_id in completed)
        assert not any(r.kind == "op_unknown" for r in records)
        world.check(quiescent=True, every_prefix=True)


def test_core_composes_actual_yield_results_owner(tmp_path, monkeypatch):
    class AtReceipt(BaseException):
        pass

    with SDKWorld(tmp_path / "yield.sqlite", seed=17) as world:
        world.create()
        worker = sdk_driver(world)
        worker.runtime.providers["effect"] = Provider(world)
        captured = []
        original = outcomes.completed_steps

        def cut(owner, plan, outcome):
            if plan._payload["op_kind"] == "tool":
                captured.append((plan, outcome))
                raise AtReceipt
            return (yield from original(owner, plan, outcome))

        monkeypatch.setattr(outcomes, "completed_steps", cut)
        with pytest.raises(AtReceipt):
            until(worker, lambda: False)
        monkeypatch.setattr(outcomes, "completed_steps", original)

        plan, _ = captured[0]
        provider = YieldProvider(callback=None)
        worker.runtime.providers["effect"] = provider
        worker.commit([worker.accepted(plan, provider.submit(plan, context=None))])
        attempt = worker.state.operations[plan.operation_id].attempts[plan.attempt]
        assert attempt.state == "parked" and attempt.wait["yield_on_steer"]
        item = InboxItem("steer-core", "steer", {"content": "change direction"}, plan.turn_id, worker.scope.generation)

        steps = core._compose(worker, outcomes.yield_results_steps(worker, [attempt], item))
        records = _recovery_phase.run(worker, steps)
        receipt = next(record for record in records if record.kind == "op_completed")
        assert receipt._payload["yielded_by"] == item.input_id
        assert receipt._payload["evidence"] == ["steer/steer-core", "provider-accepted"]
