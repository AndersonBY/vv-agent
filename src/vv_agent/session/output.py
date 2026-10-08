"""Output checks and one logged, tools-free repair through the ordinary driver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vv_agent.output_validation import OutputValidationContext
from vv_agent.runner import Runner
from vv_agent.types import AgentResult, AgentStatus, CompletionReason

if TYPE_CHECKING:
    from .kernel import _Driver


def prepare_output(driver: _Driver, value: Any) -> tuple[str, str | None, Any] | None:
    """None means a repair operation was committed; the same drive dispatches it."""
    tid = driver.state.active_turn_id
    assert tid is not None
    agent = driver.runtime.agent
    raw = AgentResult(
        status=AgentStatus.COMPLETED,
        messages=driver.transcript(),
        cycles=[],
        final_answer=value,
        completion_reason=CompletionReason.NO_TOOL_FINISH,
    )
    value, error = Runner._postprocess_output(
        agent=agent,
        run_context=driver.runtime.run_context(tid),
        raw_result=raw,
        final_output=value,
        cancellation_token=driver.scope.token,
    )
    if raw.status == AgentStatus.FAILED:
        return "failed", "agent_failed", value
    if not agent.output_validation_enabled or agent.output_validator is None:
        return ("failed", "output_type_invalid", str(error)) if error else ("completed", None, value)
    repair_id = f"{tid}/model/output_repair/1"
    repair = driver.state.operations.get(repair_id)
    if repair is not None:
        receipt = repair.attempts[repair.selected_attempt or max(repair.attempts)].result
        if receipt is None:
            return "failed", "output_validation_failed", "output_validation_failed: repair_provider_error"
        value, error = receipt.payload["result"]["content"], None
    value, validation, repairable = Runner._run_output_validator(
        agent=agent,
        validator=agent.output_validator,
        validation_context=OutputValidationContext(tid, agent.name, agent.output_type),
        candidate=value,
        output_coercion_error=error,
        coerce=repair is not None,
    )
    if validation.valid:
        return "completed", None, value
    if repair is None and repairable and agent.output_repair is not None and agent.output_validation_max_repairs == 1:
        task = driver.task()
        request = {
            "model": task.model,
            "messages": [],
            "tools": [],
            "prompt_bundle": None,
            "model_settings": task.model_settings.to_dict() if task.model_settings else None,
            "metadata": {"invalid_output": value, "validation_code": validation.code, "validation_message": validation.message},
        }
        driver.commit([driver.plan(repair_id, request, "model", purpose="output_repair")], guarded=True)
        return None
    return "failed", "output_validation_failed", Runner._output_validation_error(validation)
