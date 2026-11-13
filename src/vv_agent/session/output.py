"""SDK observations and guarded ledger operations for the core output phase."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from vv_agent.kernel import _output_phase as phase
from vv_agent.kernel._driver_requests import CommitRecords, CreateBoundary, PlanModel
from vv_agent.kernel._output_phase import serializable_output as serializable_output
from vv_agent.output_validation import (
    OutputValidationContext,
    OutputValidationResult,
    OutputValidator,
    output_validation_error,
    postprocess_output,
    run_output_validator,
)
from vv_agent.types import AgentResult, AgentStatus, CompletionReason

if TYPE_CHECKING:
    from .kernel import _Driver


def prepare_output(driver: _Driver, value: Any) -> tuple[str, str | None, Any] | None:
    assert driver.state.active_turn_id is not None
    agent = driver.runtime.agent
    steps = phase.prepare_output(driver.state, value)
    response = None
    try:
        while True:
            try:
                request = steps.send(response)
            except StopIteration as done:
                return done.value
            response = None
            if request is phase.OutputSetting.HAS_GUARDRAILS:
                response = bool(agent.output_guardrails)
            elif request is phase.OutputSetting.HAS_OUTPUT_TYPE:
                response = agent.output_type is not None
            elif request is phase.OutputSetting.VALIDATION_ENABLED:
                response = agent.output_validation_enabled
            elif request is phase.OutputSetting.HAS_VALIDATOR:
                response = agent.output_validator is not None
            elif request is phase.OutputSetting.HAS_REPAIR:
                response = agent.output_repair is not None
            elif request is phase.OutputSetting.MAX_REPAIRS:
                response = agent.output_validation_max_repairs
            elif request is phase.OutputSetting.REPAIR_MODEL:
                response = agent.output_repair_model
            elif isinstance(request, phase.PostprocessOutput):
                raw = AgentResult(
                    status=AgentStatus.COMPLETED,
                    messages=driver.transcript(),
                    cycles=[],
                    final_answer=request.value,
                    completion_reason=CompletionReason.NO_TOOL_FINISH,
                )
                checked, error = postprocess_output(
                    agent=agent,
                    run_context=driver.runtime.run_context(request.turn_id),
                    raw_result=raw,
                    final_output=request.value,
                    cancellation_token=driver.scope.token,
                )
                response = phase.PostprocessedOutput(checked, error, raw.status == AgentStatus.FAILED)
            elif isinstance(request, phase.ValidateOutput):
                response = run_output_validator(
                    agent=agent,
                    validator=cast(OutputValidator, agent.output_validator),
                    validation_context=OutputValidationContext(request.turn_id, agent.name, agent.output_type),
                    candidate=request.value,
                    output_coercion_error=request.coercion_error,
                    coerce=request.coerce,
                )
            elif isinstance(request, phase.ValidationError):
                response = output_validation_error(cast(OutputValidationResult, request.validation))
            elif isinstance(request, phase.SerializeOutput):
                response = serializable_output(request.value)
            elif isinstance(request, CreateBoundary):
                response = driver.boundary_record(request.stage, request.identity, request.data)
            elif isinstance(request, PlanModel):
                response = driver.plan(request.oid, request.request, "model", purpose=request.purpose)
            elif isinstance(request, CommitRecords):
                driver.commit(request.records, guarded=True)
            else:
                raise TypeError(f"unsupported output observation: {type(request).__name__}")
    finally:
        steps.close()
