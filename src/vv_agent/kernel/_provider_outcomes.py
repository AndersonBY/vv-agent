"""Provider outcome values and pure validation of durable receipt identities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._canonical_json import canonical_json_bytes
from ._record_codec import validate
from ._record_schema import HANDLE
from ._records import Record
from ._tool_result_checks import validate_definitive_result
from ._tool_values import ToolExecutionResult


class ProviderResultInvalid(ValueError):
    code = "provider_result_invalid"
    _query_operation_key: tuple[str, int] | None = None

    def __init__(self) -> None:
        super().__init__(self.code)


def _validate_provider_identity(plan: Record, handle: dict[str, Any]) -> None:
    if handle["kind"] != "provider" or any(
        handle[key] != expected
        for key, expected in (
            ("provider", plan._payload["provider_binding"]),
            ("operation_id", plan.operation_id),
            ("attempt", plan.attempt),
            ("request_digest", plan._payload["request_digest"]),
        )
    ):
        raise ValueError("provider handle mismatch")


def validate_reference_provider_handle(plan: Record, handle: dict[str, Any]) -> None:
    """Validate the S1 record-boundary shape of a retained provider handle."""
    try:
        validate(handle, HANDLE)
        _validate_provider_identity(plan, handle)
    except (ValueError, TypeError, KeyError) as exc:
        raise ProviderResultInvalid() from exc


def _provider_evidence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        values = tuple(value)
    else:
        raise ValueError("provider handle evidence must be a string or string list")
    if not values or any(not item.strip() for item in values):
        raise ValueError("provider handle evidence must be non-empty")
    return values


def validate_provider_handle(plan: Record, handle: dict[str, Any]) -> None:
    """Validate a provider-owned logical handle without the S1 content schema."""
    try:
        if type(handle) is not dict:
            raise TypeError("provider handle must be an object")
        if set(handle) != {
            "kind",
            "provider",
            "job_id",
            "operation_id",
            "attempt",
            "request_digest",
            "evidence",
            "query_ref",
            "cancel_ref",
        }:
            raise ValueError("provider handle members mismatch")
        _validate_provider_identity(plan, handle)
        _provider_evidence(handle["evidence"])
        canonical_json_bytes(handle)
    except (ValueError, TypeError, KeyError) as exc:
        raise ProviderResultInvalid() from exc


def validate_provider_result(plan: Record, outcome: Definitive, handle: dict[str, Any] | None = None) -> None:
    """Validate external returns before hooks, receipt creation or state mutation."""
    try:
        canonical_json_bytes(outcome.result)
        if plan._payload["op_kind"] != "model":
            from ._tool_result_checks import validate_output_body

            result = ToolExecutionResult.from_dict(outcome.result)
            validate_definitive_result(result)
            validate_output_body(result)
            if result.tool_call_id != plan._payload["request"]["id"]:
                raise ValueError("provider result call ID mismatch")
        if (
            not isinstance(outcome.evidence, tuple)
            or not outcome.evidence
            or any(not isinstance(value, str) or not value.strip() for value in outcome.evidence)
            or not isinstance(outcome.usage, dict)
            or (outcome.shared_state is not None and not isinstance(outcome.shared_state, dict))
        ):
            raise ValueError("provider result evidence or usage is incomplete")
        canonical_json_bytes([list(outcome.evidence), outcome.usage, outcome.shared_state])
        if handle is not None:
            validate_provider_handle(plan, handle)
            if not set(_provider_evidence(handle["evidence"])).issubset(outcome.evidence):
                raise ValueError("provider result lost acceptance evidence")
    except (ValueError, TypeError, KeyError) as exc:
        raise ProviderResultInvalid() from exc


@dataclass(frozen=True)
class Definitive:
    result: dict[str, Any]
    evidence: tuple[str, ...] = ("synchronous-return",)
    usage: dict[str, Any] = field(default_factory=dict)
    shared_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class Accepted:
    handle: dict[str, Any]
    yield_on_steer: bool = False


def validate_acceptance(plan: Record, outcome: Accepted) -> None:
    validate_provider_handle(plan, outcome.handle)
    if type(outcome.yield_on_steer) is not bool:
        raise ProviderResultInvalid()


@dataclass(frozen=True)
class Unknown:
    reason: str


Outcome = Definitive | Accepted | Unknown
