from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, cast

from vv_agent.types import AgentResult, AgentStatus, CompletionReason, _last_assistant_output

if TYPE_CHECKING:
    from vv_agent.agent import Agent, RunContext

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

OUTPUT_VALIDATION_FAILED = "output_validation_failed"


@dataclass(frozen=True, slots=True)
class OutputValidationContext:
    """Task-neutral context supplied to a host-owned output validator."""

    run_id: str
    agent_name: str
    output_type: Any | None = None


@dataclass(frozen=True, slots=True)
class OutputValidationResult:
    """Typed result returned by an explicitly registered output validator."""

    valid: bool
    code: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("OutputValidationResult.valid must be a boolean")
        if self.valid:
            if self.code is not None or self.message is not None:
                raise ValueError("a valid output result cannot contain an error")
            return
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("an invalid output result requires a non-empty code")
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("OutputValidationResult.message must be a string or None")

    @classmethod
    def accept(cls) -> OutputValidationResult:
        return cls(valid=True)

    @classmethod
    def reject(cls, code: str, message: str | None = None) -> OutputValidationResult:
        return cls(valid=False, code=code, message=message)


@dataclass(frozen=True, slots=True)
class OutputRepairRequest:
    """Tools-free request supplied to a host-owned repair callback."""

    invalid_output: Any
    validation_code: str
    validation_message: str | None
    model: Any | None = None
    model_settings: Any | None = None
    tools: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.validation_code, str) or not self.validation_code.strip():
            raise ValueError("validation_code cannot be empty")
        if self.validation_message is not None and not isinstance(self.validation_message, str):
            raise TypeError("validation_message must be a string or None")
        if self.tools != ():
            raise ValueError("output repair requests cannot contain tools")


OutputValidator = Callable[[Any, OutputValidationContext], OutputValidationResult]
OutputRepair = Callable[[OutputRepairRequest], Any]


def output_validator(func: OutputValidator) -> OutputValidator:
    """Mark a host callback as an output validator without changing it."""

    return func


def output_repair(func: OutputRepair) -> OutputRepair:
    """Mark a host callback as an output repair provider without changing it."""

    return func


__all__ = [
    "OUTPUT_VALIDATION_FAILED",
    "OutputRepair",
    "OutputRepairRequest",
    "OutputValidationContext",
    "OutputValidationResult",
    "OutputValidator",
    "output_repair",
    "output_validator",
]


def postprocess_output(
    *,
    agent: Agent,
    run_context: RunContext[Any],
    raw_result: AgentResult,
    final_output: Any,
    cancellation_token: Any | None,
) -> tuple[Any, Exception | None]:
    from vv_agent.guardrails import apply_output_guardrails

    normalize_completion_observation(raw_result, cancellation_token=cancellation_token)
    if raw_result.completion_reason in {CompletionReason.CANCELLED, CompletionReason.BUDGET_EXHAUSTED}:
        return final_output, None
    output_result = apply_output_guardrails(
        agent=agent,
        run_context=run_context,
        final_output=final_output,
    )
    if output_result.outcome == "rewrite":
        final_output = output_result.value
        replace_raw_result_output(raw_result, final_output)
    elif output_result.outcome in {"block", "require_approval"}:
        final_output = output_result.message or "Output blocked by guardrail."
        raw_result.status = AgentStatus.FAILED
        raw_result.completion_reason = CompletionReason.FAILED
        raw_result.completion_tool_name = None
        raw_result.partial_output = raw_result.partial_output or _last_assistant_output(raw_result.cycles)
        raw_result.final_answer = None
        raw_result.wait_reason = None
        raw_result.error = {
            "code": "agent_failed",
            "message": str(final_output),
            "retryable": False,
        }

    normalize_completion_observation(raw_result, cancellation_token=cancellation_token)
    output_coercion_error: Exception | None = None
    if raw_result.status == AgentStatus.COMPLETED:
        try:
            final_output = coerce_output_type(agent=agent, final_output=final_output)
        except Exception as exc:
            output_coercion_error = ValueError(f"failed to validate final output: {exc}")
    return final_output, output_coercion_error


def run_output_validator(
    *,
    agent: Agent,
    validator: Callable[[Any, OutputValidationContext], OutputValidationResult],
    validation_context: OutputValidationContext,
    candidate: Any,
    output_coercion_error: Exception | None,
    coerce: bool,
) -> tuple[Any, OutputValidationResult, bool]:
    if output_coercion_error is not None:
        return candidate, OutputValidationResult.reject("output_type_invalid", str(output_coercion_error)), True
    if coerce:
        try:
            candidate = coerce_output_type(agent=agent, final_output=candidate)
        except Exception as exc:
            return candidate, OutputValidationResult.reject("output_type_invalid", str(exc)), True
    try:
        result = validator(candidate, validation_context)
    except Exception as exc:
        return candidate, OutputValidationResult.reject("output_validator_error", str(exc)), False
    if not isinstance(result, OutputValidationResult):
        return (
            candidate,
            OutputValidationResult.reject(
                "output_validator_contract_invalid",
                "output_validator must return OutputValidationResult",
            ),
            False,
        )
    return candidate, result, True


def output_validation_error(result: OutputValidationResult) -> str:
    detail = cast(str, result.code)
    if result.message:
        detail = f"{detail}: {result.message}"
    return f"{OUTPUT_VALIDATION_FAILED}: {detail}"


def replace_raw_result_output(result: AgentResult, output: Any) -> None:
    value = output if isinstance(output, str) or output is None else json.dumps(output, ensure_ascii=False)
    if result.status == AgentStatus.COMPLETED:
        result.final_answer = value
    elif result.status == AgentStatus.WAIT_USER:
        result.wait_reason = value
    else:
        result.error = {
            "code": result.error_code or "agent_failed",
            "message": value or "operation failed",
            "retryable": False,
        }


def normalize_completion_observation(
    result: AgentResult,
    *,
    cancellation_token: Any | None,
) -> None:
    cancelled = bool(result.status == AgentStatus.FAILED and cancellation_token is not None and cancellation_token.cancelled)
    if cancelled:
        result.completion_reason = CompletionReason.CANCELLED
        result.completion_tool_name = None
    elif result.status == AgentStatus.WAIT_USER:
        result.completion_reason = result.completion_reason or CompletionReason.WAIT_USER
    elif result.status == AgentStatus.MAX_CYCLES:
        result.completion_reason = CompletionReason.MAX_CYCLES
        result.completion_tool_name = None
    elif result.status == AgentStatus.FAILED:
        if result.completion_reason != CompletionReason.BUDGET_EXHAUSTED:
            result.completion_reason = CompletionReason.FAILED
        result.completion_tool_name = None

    if result.status == AgentStatus.COMPLETED:
        result.partial_output = None
    else:
        result.partial_output = result.partial_output or _last_assistant_output(result.cycles)


def coerce_output_type(*, agent: Agent, final_output: Any) -> Any:
    output_type = agent.output_type
    if output_type is None or final_output is None:
        return final_output
    if output_type is str:
        return str(final_output)

    payload = final_output
    if isinstance(final_output, str):
        payload = json.loads(final_output)

    if output_type is dict:
        if not isinstance(payload, dict):
            raise ValueError("Expected final output JSON object for output_type=dict.")
        return payload
    if output_type is list:
        if not isinstance(payload, list):
            raise ValueError("Expected final output JSON array for output_type=list.")
        return payload
    if isinstance(output_type, type) and is_dataclass(output_type):
        if not isinstance(payload, dict):
            raise ValueError("Expected final output JSON object for dataclass output_type.")
        field_names = {item.name for item in fields(output_type)}
        return output_type(**{key: value for key, value in payload.items() if key in field_names})

    model_validate = getattr(output_type, "model_validate", None)
    if callable(model_validate):
        return model_validate(payload)
    return final_output
