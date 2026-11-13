"""Drive after-cycle host requests against the real session driver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vv_agent.kernel import _after_cycle as phase
from vv_agent.kernel._cycles import project_cycles, turn_result_prefix
from vv_agent.kernel._driver_requests import CommitRecords
from vv_agent.kernel._usage import project_model_usage
from vv_agent.runtime.lifecycle import AfterCycleHookError, AfterCycleHookManager

from .projection import project_records
from .providers import Definitive

if TYPE_CHECKING:
    from .kernel import _Driver


def after_cycle(driver: _Driver) -> bool:
    manager = AfterCycleHookManager(driver.runtime.config.after_cycle_hooks)
    steps = phase.after_cycle(driver.state, has_hooks=manager.has_hooks())
    pending_error: AfterCycleHookError | None = None
    reply = None
    try:
        while True:
            try:
                request = steps.send(reply) if pending_error is None else steps.throw(pending_error)
            except StopIteration as done:
                return done.value
            try:
                reply = _perform(driver, manager, request)
            except AfterCycleHookError as exc:
                pending_error = exc
            else:
                pending_error = None
    finally:
        steps.close()


def _perform(driver: _Driver, manager: AfterCycleHookManager, request: phase.AfterCycleRequest) -> Any:
    if isinstance(request, phase.CompleteSkippedTool):
        return driver.completed(request.plan, Definitive(request.result))
    if isinstance(request, CommitRecords):
        return driver.commit(request.records, guarded=True)
    if isinstance(request, phase.CloseAfterCycle):
        return driver.close("failed", "agent_failed", request.error)
    if isinstance(request, phase.ReadCycleProjection):
        state, records, _ = driver.store.read_state(driver.sid)
        _, records = turn_result_prefix(records, request.turn_id)
        usage, _ = project_model_usage(state, request.turn_id)
        cycles = project_cycles(state, records, request.turn_id, usage)
        # Retained event errors must stop before callbacks or another dispatch.
        project_records(records)
        return cycles, usage
    if isinstance(request, phase.ReadSharedState):
        return driver.shared_state()
    if isinstance(request, phase.ReadTranscript):
        return driver.transcript()
    if isinstance(request, phase.ApplyAfterCycleHooks):
        return manager.apply(request.snapshot)
    if isinstance(request, phase.RecordAfterCycle):
        data = request.data | {"shared_state": driver.runtime.durable_state(request.data["shared_state"])}
        return driver.commit(
            [driver.boundary_record("after_cycle", request.operation_id, data, source_operation_id=request.operation_id)],
            guarded=True,
        )
    raise AssertionError(f"Unexpected after-cycle request: {request!r}")
