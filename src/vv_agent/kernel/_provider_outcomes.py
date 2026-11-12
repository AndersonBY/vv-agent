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

    def __init__(self) -> None:
        super().__init__(self.code)


def validate_provider_handle(plan: Record, handle: dict[str, Any]) -> None:
    try:
        validate(handle, HANDLE)
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
            if handle["evidence"] not in outcome.evidence:
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
