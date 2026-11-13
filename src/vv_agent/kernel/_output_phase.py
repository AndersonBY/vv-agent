"""Output decisions, receipts and one repair plan, suspended at SDK and ledger work."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Protocol

from vv_agent.kernel._records import Record
from vv_agent.kernel._state import ExecutionState


class OutputSetting(Enum):
    HAS_GUARDRAILS = "has_guardrails"
    HAS_OUTPUT_TYPE = "has_output_type"
    VALIDATION_ENABLED = "validation_enabled"
    HAS_VALIDATOR = "has_validator"
    HAS_REPAIR = "has_repair"
    MAX_REPAIRS = "max_repairs"
    REPAIR_MODEL = "repair_model"


@dataclass(frozen=True, slots=True)
class PostprocessOutput:
    turn_id: str
    value: Any


@dataclass(frozen=True, slots=True)
class PostprocessedOutput:
    value: Any
    error: Exception | None
    failed: bool


class ValidationResult(Protocol):
    @property
    def valid(self) -> bool: ...

    @property
    def code(self) -> str | None: ...

    @property
    def message(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class ValidateOutput:
    turn_id: str
    value: Any
    coercion_error: Exception | None
    coerce: bool


@dataclass(frozen=True, slots=True)
class ValidationError:
    validation: ValidationResult


@dataclass(frozen=True, slots=True)
class SerializeOutput:
    value: Any


@dataclass(frozen=True, slots=True)
class CreateBoundary:
    stage: str
    identity: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PlanModel:
    oid: str
    request: dict[str, Any]
    purpose: str


@dataclass(frozen=True, slots=True)
class CommitRecords:
    records: list[Record]


type OutputRequest = (
    OutputSetting
    | PostprocessOutput
    | ValidateOutput
    | ValidationError
    | SerializeOutput
    | CreateBoundary
    | PlanModel
    | CommitRecords
)

type OutputOutcome = tuple[str, str | None, Any]


def serializable_output(value: Any) -> Any:
    """Shared SDK/session serialization; custom model_dump executes at the caller."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return value


def prepare_output(state: ExecutionState, value: Any) -> Generator[OutputRequest, Any, OutputOutcome | None]:
    tid = state.active_turn_id
    assert tid is not None
    final = state.boundaries.get((tid, "output_checked", "final"))
    if final:
        data = final._payload["data"]
        return data["status"], data["reason"], data["value"]
    if (
        not (yield OutputSetting.HAS_GUARDRAILS)
        and not (yield OutputSetting.HAS_OUTPUT_TYPE)
        and not (yield OutputSetting.VALIDATION_ENABLED)
    ):
        return "completed", None, value
    repair_id = f"{tid}/model/output_repair/1"
    repair = state.operations.get(repair_id)
    initial = state.boundaries.get((tid, "output_checked", "initial"))
    error = None
    if initial is None:
        checked: PostprocessedOutput = yield PostprocessOutput(tid, value)
        value, error = checked.value, checked.error
        if checked.failed:
            outcome = ("failed", "agent_failed", value)
        elif not (yield OutputSetting.VALIDATION_ENABLED) or not (yield OutputSetting.HAS_VALIDATOR):
            outcome = ("failed", "output_type_invalid", str(error)) if error else ("completed", None, value)
        else:
            outcome = None
        if outcome:
            record = yield CreateBoundary(
                "output_checked",
                "final",
                {"status": outcome[0], "reason": outcome[1], "value": (yield SerializeOutput(outcome[2]))},
            )
            yield CommitRecords([record])
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
            record = yield CreateBoundary(
                "output_checked",
                "final",
                {
                    "status": outcome[0],
                    "reason": outcome[1],
                    "value": outcome[2],
                    "partial_output": initial._payload["data"]["value"] if initial else value,
                },
            )
            yield CommitRecords([record])
            return outcome
    assert (yield OutputSetting.HAS_VALIDATOR)
    validation: ValidationResult
    value, validation, repairable = yield ValidateOutput(tid, value, error, coerce=repair is not None)
    if (
        not validation.valid
        and repair is None
        and repairable
        and (yield OutputSetting.HAS_REPAIR)
        and (yield OutputSetting.MAX_REPAIRS) == 1
    ):
        task = state.turns[tid].start.task()
        request = {
            "model": (yield OutputSetting.REPAIR_MODEL) or task.model,
            "messages": [],
            "tools": [],
            "prompt_bundle": None,
            "model_settings": task.model_settings.to_dict() if task.model_settings else None,
            "metadata": {
                "invalid_output": (yield SerializeOutput(value)),
                "validation_code": validation.code,
                "validation_message": validation.message,
            },
        }
        initial_record = yield CreateBoundary(
            "output_checked",
            "initial",
            {"status": "repair", "reason": validation.code, "value": (yield SerializeOutput(value))},
        )
        plan = yield PlanModel(repair_id, request, "output_repair")
        yield CommitRecords([initial_record, plan])
        return None
    outcome = (
        ("completed", None, value)
        if validation.valid
        else ("failed", "output_validation_failed", (yield ValidationError(validation)))
    )
    record = yield CreateBoundary(
        "output_checked",
        "final",
        {
            "status": outcome[0],
            "reason": outcome[1],
            "value": (yield SerializeOutput(outcome[2])),
            "partial_output": (yield SerializeOutput(value)),
        },
    )
    yield CommitRecords([record])
    return outcome
