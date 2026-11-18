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

_PROVIDER_HANDLE_FIELDS = {
    "kind",
    "provider",
    "job_id",
    "operation_id",
    "attempt",
    "request_digest",
    "evidence",
    "query_ref",
    "cancel_ref",
}
_W3_PROVIDER_HANDLE_FIELDS = _PROVIDER_HANDLE_FIELDS | {
    "execution_id",
    "command_digest",
    "environment_id",
    "generation",
}


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


def _validate_provider_handle_binding_fields(
    handle: dict[str, Any],
    *,
    operation_id: str,
    attempt: int,
    request_digest: str,
) -> None:
    if type(handle) is not dict:
        raise ValueError("provider handle must be an object")
    for field_name, expected in (
        ("operation_id", operation_id),
        ("attempt", attempt),
        ("request_digest", request_digest),
    ):
        if handle.get(field_name) != expected:
            raise ValueError(f"provider handle {field_name} mismatch")


def _validate_provider_handle_identity(
    handle: dict[str, Any],
    *,
    provider: str,
    operation_id: str,
    attempt: int,
    request_digest: str,
    execution_id: str | None = None,
    command_digest: str | None = None,
    environment_id: str | None = None,
    generation: int | None = None,
) -> None:
    """Check identity fields shared by ordinary and W3 provider handles."""
    _validate_provider_handle_binding_fields(
        handle,
        operation_id=operation_id,
        attempt=attempt,
        request_digest=request_digest,
    )
    if any(handle.get(key) != expected for key, expected in (("kind", "provider"), ("provider", provider))):
        raise ValueError("provider handle mismatch")
    w3_fields = _W3_PROVIDER_HANDLE_FIELDS - _PROVIDER_HANDLE_FIELDS
    if not set(handle).intersection(w3_fields):
        return
    if not w3_fields.issubset(handle) or any(
        expected is not None and handle[key] != expected
        for key, expected in (
            ("execution_id", execution_id),
            ("command_digest", command_digest),
            ("environment_id", environment_id),
            ("generation", generation),
        )
    ):
        raise ValueError("provider handle execution identity mismatch")
    if generation is None and (type(handle["generation"]) is not int or handle["generation"] < 0):
        raise ValueError("provider handle generation is invalid")


def _validate_provider_handle_refs(handle: dict[str, Any]) -> None:
    job_id = handle["job_id"]
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("provider handle job_id must be a non-empty string")
    for field_name in ("query_ref", "cancel_ref"):
        value = handle[field_name]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"provider handle {field_name} must be a non-empty string or null")


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
        if set(handle) != _PROVIDER_HANDLE_FIELDS:
            raise ValueError("provider handle members mismatch")
        _validate_provider_handle_identity(
            handle,
            provider=plan._payload["provider_binding"],
            operation_id=plan.operation_id,
            attempt=plan.attempt,
            request_digest=plan._payload["request_digest"],
        )
        _provider_evidence(handle["evidence"])
        _validate_provider_handle_refs(handle)
        canonical_json_bytes(handle)
    except (ValueError, TypeError, KeyError) as exc:
        raise ProviderResultInvalid() from exc


def validate_w3_provider_handle(binding: Any, handle: dict[str, Any]) -> None:
    """Validate the level-3 lookup variant against its authenticated binding."""
    try:
        if type(handle) is not dict or set(handle) != _W3_PROVIDER_HANDLE_FIELDS:
            raise ValueError("provider handle members mismatch")
        _validate_provider_handle_identity(
            handle,
            provider=binding.provider_id,
            operation_id=binding.operation_id,
            attempt=binding.attempt,
            request_digest=binding.request_digest,
            execution_id=binding.execution_id,
            command_digest=binding.command_digest,
            environment_id=binding.environment_id,
            generation=binding.generation,
        )
        _provider_evidence(handle["evidence"])
        _validate_provider_handle_refs(handle)
        canonical_json_bytes(handle)
    except (ValueError, TypeError, KeyError) as exc:
        raise ProviderResultInvalid() from exc


def validate_ordinary_provider_handle(binding: Any, handle: dict[str, Any]) -> None:
    """Validate the Contract 40 lookup handle against its authenticated binding."""
    try:
        if type(handle) is not dict or set(handle) != _PROVIDER_HANDLE_FIELDS:
            raise ValueError("provider handle members mismatch")
        _validate_provider_handle_identity(
            handle,
            provider=binding.provider_id,
            operation_id=binding.operation_id,
            attempt=binding.attempt,
            request_digest=binding.request_digest,
        )
        _provider_evidence(handle["evidence"])
        _validate_provider_handle_refs(handle)
        canonical_json_bytes(handle)
    except (ValueError, TypeError, KeyError) as exc:
        raise ProviderResultInvalid() from exc


def validate_lookup_provider_handle(binding: Any, handle: dict[str, Any], *, feature_level: int) -> None:
    """Validate one lookup handle through the level-selected shape."""
    if feature_level == 3:
        validate_w3_provider_handle(binding, handle)
        return
    if feature_level in {1, 2}:
        validate_ordinary_provider_handle(binding, handle)
        return
    raise ProviderResultInvalid()


def _validate_definitive_value(
    outcome: Definitive,
    plan: Record | None = None,
    *,
    require_tool_result: bool = False,
) -> None:
    canonical_json_bytes(outcome.result)
    if require_tool_result or (plan is not None and plan._payload["op_kind"] != "model"):
        from ._tool_result_checks import validate_output_body

        result = ToolExecutionResult.from_dict(outcome.result)
        validate_definitive_result(result)
        validate_output_body(result)
        if plan is not None and result.tool_call_id != plan._payload["request"]["id"]:
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


def validate_provider_result(plan: Record, outcome: Definitive, handle: dict[str, Any] | None = None) -> None:
    """Validate external returns before hooks, receipt creation or state mutation."""
    try:
        _validate_definitive_value(outcome, plan)
        if handle is not None:
            validate_provider_handle(plan, handle)
            if not set(_provider_evidence(handle["evidence"])).issubset(outcome.evidence):
                raise ValueError("provider result lost acceptance evidence")
    except (ValueError, TypeError, KeyError) as exc:
        raise ProviderResultInvalid() from exc


def validate_lookup_provider_result(
    binding: Any,
    status: str,
    outcome: Definitive | Accepted | Unknown | None,
    evidence: tuple[str, ...],
    *,
    feature_level: int = 1,
) -> None:
    """Validate a W3 lookup result at the authenticated provider boundary."""
    try:
        if feature_level not in {1, 2, 3}:
            raise ValueError("lookup feature level is not supported")
        if not evidence or any(not isinstance(value, str) or not value.strip() for value in evidence):
            raise ValueError("provider lookup evidence is incomplete")
        if status == "not_started":
            if outcome is not None:
                raise ValueError("not_started lookup cannot carry an execution outcome")
        elif status == "running":
            if not isinstance(outcome, Accepted):
                raise ValueError("running lookup requires Accepted")
            validate_lookup_provider_handle(binding, outcome.handle, feature_level=feature_level)
            if type(outcome.yield_on_steer) is not bool:
                raise ValueError("running lookup declaration is invalid")
            if not set(_provider_evidence(outcome.handle["evidence"])).issubset(evidence):
                raise ValueError("provider lookup lost acceptance evidence")
        elif status == "exited":
            if not isinstance(outcome, Definitive):
                raise ValueError("exited lookup requires Definitive")
            _validate_definitive_value(outcome, require_tool_result=True)
            if not set(outcome.evidence).issubset(evidence):
                raise ValueError("provider lookup lost terminal evidence")
        elif status == "unknown":
            if not isinstance(outcome, Unknown):
                raise ValueError("unknown lookup requires Unknown")
            if not isinstance(outcome.reason, str) or not outcome.reason.strip():
                raise ValueError("unknown lookup reason is invalid")
        else:
            raise ValueError("lookup status is not recognized")
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
