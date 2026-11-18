"""Directed checks for fail-closed host observations at admission boundaries."""

from __future__ import annotations

import pytest

from sim.assembly import Scope, SimDriver
from sim.world import SimWorld
from vv_agent.kernel import _tool_admission as admission
from vv_agent.kernel._cancellation import CancelledError
from vv_agent.kernel._driver_dispatch import AdmissionException, FinishPreflight, FormatAdmissionException
from vv_agent.kernel._tool_values import ToolCall, ToolExecutionResult


def _worker(world: SimWorld):
    world.create()
    runtime = world.runtime("s", "owner")
    store = world.connection("owner")
    lease = store.acquire("s", owner="owner", ttl_ms=runtime.ttl_ms)
    assert lease is not None
    scope = Scope(lease, runtime, now_ms=store._now())
    worker = SimDriver(store, "s", runtime, scope)
    worker.refresh()
    return worker


def test_policy_predicate_false_denies_with_real_admission_result(tmp_path):
    with SimWorld(tmp_path / "predicate.sqlite", seed=17) as world:
        worker = _worker(world)
        observed = []

        def deny(name, arguments):
            observed.append((name, arguments))
            return False

        task = worker.runtime.compile_task("go", "s/turn/predicate")
        task.metadata["_vv_agent_tool_policy_can_use_tool"] = deny
        worker.context_task = task
        call = ToolCall("call", "effect", {})
        result = worker.run(
            admission.admit_steps(
                call,
                original_call=call,
                executor_present=True,
                allowed_tool_names=["effect"],
            )
        )
        assert observed == [("effect", {})]
        assert isinstance(result, ToolExecutionResult)
        assert result.error_code == "tool_not_allowed"
        assert not any(call[3] == "tool" for call in world.calls)


def test_format_and_finish_preflight_boundaries_do_not_continue_dispatch(tmp_path):
    with SimWorld(tmp_path / "preflight-boundaries.sqlite", seed=17) as world:
        worker = _worker(world)
        call = ToolCall("call", "effect", {})
        result = worker.handle(FormatAdmissionException(call, RuntimeError("denied")))
        assert isinstance(result, ToolExecutionResult)
        assert result.error_code == "tool_execution_failed"
        assert "Tool execution failed (effect): denied" in result.content
        assert worker.handle(FinishPreflight(call, result)) is result
        with pytest.raises(ValueError, match="preflight handler bypassed the dispatch boundary"):
            worker.handle(FinishPreflight(call, object()))


def test_admission_exception_only_preserves_exact_core_cancellation(tmp_path):
    with SimWorld(tmp_path / "admission-exception.sqlite", seed=17) as world:
        worker = _worker(world)
        assert worker.handle(AdmissionException(CancelledError("cancelled"))) is True
        assert worker.handle(AdmissionException(RuntimeError("format me"))) is False
