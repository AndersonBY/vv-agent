"""Output checks and one logged, tools-free repair through the ordinary driver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vv_agent.output_validation import OutputValidationContext, output_validation_error, postprocess_output, run_output_validator
from vv_agent.result import RunResult
from vv_agent.types import AgentResult, AgentStatus, CompletionReason

if TYPE_CHECKING:
    from .kernel import _Driver


def serializable_output(value: Any) -> Any:
    return RunResult._serializable_output(value)


def prepare_output(driver: _Driver, value: Any) -> tuple[str, str | None, Any] | None:
    tid = driver.state.active_turn_id
    assert tid is not None
    agent = driver.runtime.agent
    final = driver.boundary("output_checked", "final")
    if final:
        data = final._payload["data"]
        return data["status"], data["reason"], data["value"]
    if not agent.output_guardrails and agent.output_type is None and not agent.output_validation_enabled:
        return "completed", None, value
    repair_id = f"{tid}/model/output_repair/1"
    repair = driver.state.operations.get(repair_id)
    initial = driver.boundary("output_checked", "initial")
    error = None
    if initial is None:
        raw = AgentResult(
            status=AgentStatus.COMPLETED,
            messages=driver.transcript(),
            cycles=[],
            final_answer=value,
            completion_reason=CompletionReason.NO_TOOL_FINISH,
        )
        value, error = postprocess_output(
            agent=agent,
            run_context=driver.runtime.run_context(tid),
            raw_result=raw,
            final_output=value,
            cancellation_token=driver.scope.token,
        )
        if raw.status == AgentStatus.FAILED:
            outcome = ("failed", "agent_failed", value)
        elif not agent.output_validation_enabled or agent.output_validator is None:
            outcome = ("failed", "output_type_invalid", str(error)) if error else ("completed", None, value)
        else:
            outcome = None
        if outcome:
            driver.commit(
                [
                    driver.boundary_record(
                        "output_checked",
                        "final",
                        {"status": outcome[0], "reason": outcome[1], "value": serializable_output(outcome[2])},
                    )
                ],
                guarded=True,
            )
            return outcome
    elif repair is None:
        raise ValueError("output repair boundary has no logged operation")
    if repair is not None:
        receipt = repair.attempts[repair.selected_attempt or max(repair.attempts)].result
        if receipt is None:
            outcome = ("failed", "output_validation_failed", "output_validation_failed: repair_provider_error")
        elif receipt._payload["result"].get("error_code"):
            outcome = (
                "failed",
                "output_validation_failed",
                f"output_validation_failed: repair_provider_error: {receipt._payload['result']['content']}",
            )
        else:
            value = receipt.payload["result"]["content"]
            outcome = None
        if outcome:
            driver.commit(
                [
                    driver.boundary_record(
                        "output_checked",
                        "final",
                        {
                            "status": outcome[0],
                            "reason": outcome[1],
                            "value": outcome[2],
                            "partial_output": initial._payload["data"]["value"] if initial else value,
                        },
                    )
                ],
                guarded=True,
            )
            return outcome
    assert agent.output_validator is not None
    value, validation, repairable = run_output_validator(
        agent=agent,
        validator=agent.output_validator,
        validation_context=OutputValidationContext(tid, agent.name, agent.output_type),
        candidate=value,
        output_coercion_error=error,
        coerce=repair is not None,
    )
    if (
        not validation.valid
        and repair is None
        and repairable
        and agent.output_repair is not None
        and agent.output_validation_max_repairs == 1
    ):
        task = driver.task()
        request = {
            "model": agent.output_repair_model or task.model,
            "messages": [],
            "tools": [],
            "prompt_bundle": None,
            "model_settings": task.model_settings.to_dict() if task.model_settings else None,
            "metadata": {
                "invalid_output": serializable_output(value),
                "validation_code": validation.code,
                "validation_message": validation.message,
            },
        }
        driver.commit(
            [
                driver.boundary_record(
                    "output_checked",
                    "initial",
                    {"status": "repair", "reason": validation.code, "value": serializable_output(value)},
                ),
                driver.plan(repair_id, request, "model", purpose="output_repair"),
            ],
            guarded=True,
        )
        return None
    outcome = (
        ("completed", None, value)
        if validation.valid
        else ("failed", "output_validation_failed", output_validation_error(validation))
    )
    driver.commit(
        [
            driver.boundary_record(
                "output_checked",
                "final",
                {
                    "status": outcome[0],
                    "reason": outcome[1],
                    "value": serializable_output(outcome[2]),
                    "partial_output": serializable_output(value),
                },
            )
        ],
        guarded=True,
    )
    return outcome
