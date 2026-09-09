from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from vv_agent.budget import BudgetUsageSnapshot
from vv_agent.checkpoint import (
    RUN_DEFINITION_SCHEMA,
    CheckpointError,
    EventCursor,
    OperationKind,
    OperationState,
    ResumeObservation,
    ToolIdempotency,
    canonical_json_bytes,
    canonical_json_sha256,
    compute_event_payload_digest,
    compute_operation_request_digest,
    compute_run_definition_digest,
    validate_extension_namespace,
    validate_sha256,
)
from vv_agent.deferred import (
    DeferredCheckpointClaimed,
    DeferredResolutionConflict,
    DeferredResolutionReceipt,
    DeferredResolutionStale,
    DeferredResolveDecision,
    DeferredToolHandle,
    ToolCallOutcome,
    validate_definitive_result,
)
from vv_agent.types import (
    AgentResult,
    AgentStatus,
    CycleRecord,
    Message,
    ModelCallOperation,
    ModelCallRecord,
    ModelCallStatus,
    ToolExecutionResult,
    ToolResultStatus,
)

if TYPE_CHECKING:
    from vv_agent.runtime.controller import HostInteractionAdmissionContext

CHECKPOINT_SCHEMA = "vv-agent.checkpoint.v10"
HOST_INTERACTION_REQUEST_SCHEMA = "vv-agent.host-interaction-request.v1"
_HOST_INTERACTION_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "interaction_id",
        "logical_cycle",
        "operation_id",
        "tool_call_id",
        "request_digest",
        "prompt",
    }
)
_SUSPENDED_ORIGIN_FIELDS = frozenset({"status", "active_host_interaction"})
MAX_WIRE_INTEGER = (1 << 53) - 1
ClaimMode = Literal["continue", "recovery"]


class RenewOutcome(StrEnum):
    RENEWED = "renewed"
    CANCEL_REQUESTED = "cancel_requested"
    CLAIM_LOST = "claim_lost"


@dataclass(frozen=True, slots=True)
class CheckpointRenewal:
    outcome: RenewOutcome
    lease_expires_at_ms: int | None = None
    revision: int | None = None
    schema_version: str = "vv-agent.checkpoint-renewal.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, RenewOutcome):
            object.__setattr__(self, "outcome", RenewOutcome(self.outcome))
        if self.outcome is RenewOutcome.CLAIM_LOST:
            if (
                self.revision is None
                or isinstance(self.revision, bool)
                or not isinstance(self.revision, int)
                or self.revision < 0
            ):
                raise ValueError("claim_lost renewal requires a non-negative revision")
            if self.lease_expires_at_ms is not None:
                raise ValueError("claim_lost renewal cannot include a lease")
        else:
            if (
                self.lease_expires_at_ms is None
                or isinstance(self.lease_expires_at_ms, bool)
                or not isinstance(self.lease_expires_at_ms, int)
                or self.lease_expires_at_ms < 0
            ):
                raise ValueError("renewal requires a non-negative lease expiry")
            if self.revision is not None:
                raise ValueError("successful renewal cannot include a revision")
        if self.schema_version != "vv-agent.checkpoint-renewal.v1":
            raise ValueError("unsupported checkpoint renewal schema_version")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"schema_version": self.schema_version, "outcome": self.outcome.value}
        if self.outcome is RenewOutcome.CLAIM_LOST:
            payload["revision"] = self.revision
        else:
            payload["lease_expires_at_ms"] = self.lease_expires_at_ms
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> CheckpointRenewal:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint renewal must be an object")
        outcome = RenewOutcome(payload.get("outcome"))
        expected = (
            {"schema_version", "outcome", "revision"}
            if outcome is RenewOutcome.CLAIM_LOST
            else {"schema_version", "outcome", "lease_expires_at_ms"}
        )
        if set(payload) != expected:
            raise ValueError("checkpoint renewal has missing or unknown fields")
        return cls(
            outcome=outcome,
            lease_expires_at_ms=payload.get("lease_expires_at_ms"),
            revision=payload.get("revision"),
            schema_version=payload.get("schema_version", ""),
        )


class CheckpointConflictError(RuntimeError):
    """The requested checkpoint transition lost its compare-and-swap race."""


class _LeaseOperationClock:
    def __init__(self, now_ms: int) -> None:
        self._now_ms = now_ms
        self._started_ns = time.monotonic_ns()

    def now_ms(self) -> int:
        elapsed_ms = max(0, time.monotonic_ns() - self._started_ns) // 1_000_000
        return min((1 << 64) - 1, self._now_ms + elapsed_ms)


def _validate_claim(cycle_index: int, claim_token: str, lease_expires_at_ms: int, now_ms: int) -> None:
    _positive_wire_integer(cycle_index, "claimed cycle_index")
    if not isinstance(claim_token, str) or not claim_token:
        raise ValueError("claim_token must be a non-empty string")
    for value, name in ((lease_expires_at_ms, "lease_expires_at_ms"), (now_ms, "now_ms")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if lease_expires_at_ms <= now_ms:
        raise ValueError("lease_expires_at_ms must be greater than now_ms")


def _validate_renew(claim_token: str, lease_expires_at_ms: int, now_ms: int) -> None:
    if not isinstance(claim_token, str) or not claim_token:
        raise ValueError("claim_token must be a non-empty string")
    for value, name in ((lease_expires_at_ms, "lease_expires_at_ms"), (now_ms, "now_ms")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if lease_expires_at_ms <= now_ms:
        raise ValueError("lease_expires_at_ms must be greater than now_ms")


@dataclass(frozen=True, slots=True)
class OperationError:
    code: str
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("operation error code must be non-empty")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("operation error message must be non-empty")
        if not isinstance(self.retryable, bool):
            raise TypeError("operation error retryable must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OperationError:
        if not isinstance(payload, dict):
            raise ValueError("operation error must be an object")
        if set(payload) != {"code", "message", "retryable"}:
            raise ValueError("operation error has missing or unknown fields")
        return cls(
            code=_required_string(payload, "code"),
            message=_required_string(payload, "message"),
            retryable=_required_boolean(payload, "retryable", default=False),
        )


def operation_error_from_tool_result(result: ToolExecutionResult) -> OperationError:
    """Project a failed tool result into the journal's diagnostic error."""
    retryable = result.metadata.get("retryable")
    return OperationError(
        code=result.error_code or "tool_operation_failed",
        message=result.content or "tool operation failed",
        retryable=retryable if isinstance(retryable, bool) else False,
    )


@dataclass(slots=True)
class OperationJournalEntry:
    kind: OperationKind
    operation_id: str
    cycle_index: int
    attempt: int
    state: OperationState
    request_digest: str
    idempotency_key: str | None = None
    response: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: OperationError | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    idempotency_support: ToolIdempotency | None = None
    model_operation: ModelCallOperation | None = None
    backend: str | None = None
    model: str | None = None
    call_id: str | None = None
    deferred_handle: DeferredToolHandle | None = None
    identity_key: str | None = None
    result_digest: str | None = None
    resume_observation: ResumeObservation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OperationKind):
            try:
                self.kind = OperationKind(self.kind)
            except (TypeError, ValueError) as exc:
                raise CheckpointError(
                    "operation kind is invalid",
                    code="operation_kind_fields_invalid",
                ) from exc
        if not isinstance(self.state, OperationState):
            try:
                self.state = OperationState(self.state)
            except (TypeError, ValueError) as exc:
                raise CheckpointError(
                    "operation state is invalid",
                    code="operation_state_invalid",
                ) from exc
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise CheckpointError(
                "operation_id must be non-empty",
                code="operation_id_invalid",
            )
        try:
            _positive_wire_integer(self.cycle_index, "operation cycle_index")
        except ValueError as exc:
            raise CheckpointError(str(exc), code="operation_cycle_invalid") from exc
        try:
            _positive_wire_integer(self.attempt, "operation attempt")
        except ValueError as exc:
            raise CheckpointError(str(exc), code="operation_attempt_invalid") from exc
        try:
            validate_sha256(self.request_digest, "operation request_digest")
        except ValueError as exc:
            raise CheckpointError(
                str(exc),
                code="operation_request_digest_invalid",
            ) from exc
        if self.idempotency_key is not None and (not isinstance(self.idempotency_key, str) or not self.idempotency_key):
            raise CheckpointError(
                "operation idempotency_key must be a non-empty string or null",
                code="operation_idempotency_key_invalid",
            )
        if self.kind is OperationKind.MODEL:
            if any(
                value is not None
                for value in (
                    self.tool_call_id,
                    self.tool_name,
                    self.arguments,
                    self.idempotency_support,
                    self.result,
                )
            ):
                raise CheckpointError(
                    "model journal entry cannot contain tool fields",
                    code="operation_kind_fields_invalid",
                )
            try:
                self.model_operation = ModelCallOperation(self.model_operation)
            except (TypeError, ValueError) as exc:
                raise CheckpointError(
                    "model journal entry requires a valid model_operation",
                    code="model_identity_invalid",
                ) from exc
            for value, name in (
                (self.backend, "backend"),
                (self.model, "model"),
                (self.call_id, "call_id"),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise CheckpointError(
                        f"model journal entry requires non-empty {name}",
                        code="model_identity_invalid",
                    )
        else:
            if any(value is not None for value in (self.model_operation, self.backend, self.model, self.call_id)):
                raise CheckpointError(
                    "tool journal entry cannot contain model identity fields",
                    code="operation_kind_fields_invalid",
                )
            if not isinstance(self.tool_call_id, str) or not self.tool_call_id:
                raise CheckpointError(
                    "tool journal entry requires tool_call_id",
                    code="operation_kind_fields_invalid",
                )
            if not isinstance(self.tool_name, str) or not self.tool_name:
                raise CheckpointError(
                    "tool journal entry requires tool_name",
                    code="operation_kind_fields_invalid",
                )
            if not isinstance(self.arguments, dict):
                raise CheckpointError(
                    "tool journal entry requires object arguments",
                    code="operation_kind_fields_invalid",
                )
            canonical_json_bytes(self.arguments, "tool journal arguments")
            if not isinstance(self.idempotency_support, ToolIdempotency):
                try:
                    self.idempotency_support = ToolIdempotency(self.idempotency_support)
                except (TypeError, ValueError) as exc:
                    raise CheckpointError(
                        "tool idempotency support is invalid",
                        code="operation_kind_fields_invalid",
                    ) from exc
            if self.idempotency_support is ToolIdempotency.UNSUPPORTED and self.idempotency_key is not None:
                raise CheckpointError(
                    "unsupported tools must not carry an idempotency key",
                    code="tool_idempotency_key_invalid",
                )
            if self.idempotency_support is not ToolIdempotency.UNSUPPORTED and (
                not isinstance(self.idempotency_key, str) or not self.idempotency_key
            ):
                raise CheckpointError(
                    "supported or unknown tools require an idempotency key",
                    code="tool_idempotency_key_required",
                )
            if self.response is not None:
                raise CheckpointError(
                    "tool journal entry cannot contain a model response",
                    code="operation_kind_fields_invalid",
                )
            if self.deferred_handle is not None and not isinstance(self.deferred_handle, DeferredToolHandle):
                raise CheckpointError(
                    "deferred_handle must be a DeferredToolHandle",
                    code="deferred_handle_invalid",
                )
            if self.state is OperationState.DEFERRED and (
                self.deferred_handle is None
                or self.result is not None
                or self.error is not None
                or self.identity_key is not None
                or self.result_digest is not None
                or self.resume_observation is not None
            ):
                raise CheckpointError(
                    "deferred operation requires an exact handle and no receipt",
                    code="operation_deferred_fields_invalid",
                )
            if self.identity_key is not None:
                try:
                    validate_sha256(self.identity_key, "tool identity_key")
                except ValueError as exc:
                    raise CheckpointError(str(exc), code="tool_receipt_identity_invalid") from exc
            if self.result_digest is not None:
                try:
                    validate_sha256(self.result_digest, "tool result_digest")
                except ValueError as exc:
                    raise CheckpointError(str(exc), code="tool_receipt_digest_invalid") from exc
            if self.resume_observation is not None and not isinstance(self.resume_observation, ResumeObservation):
                raise CheckpointError(
                    "tool resume_observation must be a ResumeObservation",
                    code="tool_resume_observation_invalid",
                )
            if self.deferred_handle is not None and self.state is not OperationState.DEFERRED:
                raise CheckpointError(
                    "only deferred operation entries may contain deferred_handle",
                    code="operation_deferred_fields_invalid",
                )
        if self.state is OperationState.SUCCEEDED:
            receipt = self.response if self.kind is OperationKind.MODEL else self.result
            if receipt is None or self.error is not None:
                raise CheckpointError(
                    "succeeded operation requires exactly one success receipt",
                    code="operation_receipt_required",
                )
            canonical_json_bytes(receipt, "operation success receipt")
            if self.kind is OperationKind.TOOL:
                if self.identity_key is None or self.result_digest is None or self.resume_observation is not None:
                    raise CheckpointError(
                        "successful tool receipt requires identity_key and result_digest",
                        code=(
                            "operation_receipt_identity_required"
                            if self.identity_key is None
                            else "operation_result_digest_required"
                        ),
                    )
                try:
                    parsed_result = ToolExecutionResult.from_dict(receipt)
                except (TypeError, ValueError) as exc:
                    raise CheckpointError(
                        "tool operation receipt is not a current ToolExecutionResult",
                        code="operation_receipt_invalid",
                    ) from exc
                if parsed_result.status_code is not ToolResultStatus.SUCCESS:
                    raise CheckpointError(
                        "succeeded tool receipt must contain a SUCCESS result",
                        code="operation_receipt_invalid",
                    )
                if parsed_result.to_dict() != receipt:
                    raise CheckpointError(
                        "tool operation receipt is not canonical",
                        code="operation_receipt_invalid",
                    )
                if canonical_json_sha256(parsed_result.to_dict(), "tool result") != self.result_digest:
                    raise CheckpointError(
                        "tool result_digest does not match result",
                        code="tool_receipt_digest_invalid",
                    )
        elif self.state is OperationState.FAILED:
            synthetic_cancelled = (
                self.kind is OperationKind.TOOL and self.error is not None and self.error.code == "tool_cancelled"
            )
            if synthetic_cancelled:
                if self.resume_observation is None:
                    raise CheckpointError(
                        "synthetic tool outcome requires a resume observation",
                        code="operation_resume_observation_required",
                    )
                if self.result is not None or self.result_digest is not None:
                    raise CheckpointError(
                        "synthetic tool outcome cannot contain a definitive receipt",
                        code="operation_closure_receipt_forbidden",
                    )
            if self.error is None or self.response is not None:
                raise CheckpointError(
                    "failed operation requires exactly one typed error",
                    code="operation_error_required",
                )
            if self.kind is OperationKind.TOOL:
                if self.identity_key is None:
                    raise CheckpointError(
                        "failed tool receipt requires identity_key",
                        code="operation_receipt_identity_required",
                    )
                if synthetic_cancelled:
                    return
                if self.result is None:
                    raise CheckpointError(
                        "failed tool receipt requires result",
                        code="operation_result_required",
                    )
                if self.result_digest is None:
                    raise CheckpointError(
                        "failed tool receipt requires result_digest",
                        code="operation_result_digest_required",
                    )
                try:
                    parsed_result = ToolExecutionResult.from_dict(self.result)
                except (TypeError, ValueError) as exc:
                    raise CheckpointError(
                        "failed tool receipt result is invalid",
                        code="operation_failed_result_status_invalid",
                    ) from exc
                if parsed_result.status_code.value != "ERROR":
                    raise CheckpointError(
                        "failed tool receipt must contain an ERROR result",
                        code="operation_failed_result_status_invalid",
                    )
                if parsed_result.tool_call_id != self.tool_call_id:
                    raise CheckpointError(
                        "failed tool receipt tool_call_id does not match its journal entry",
                        code="tool_receipt_identity_invalid",
                    )
                if parsed_result.to_dict() != self.result:
                    raise CheckpointError(
                        "failed tool receipt is not canonical",
                        code="operation_receipt_invalid",
                    )
                if canonical_json_sha256(parsed_result.to_dict(), "tool result") != self.result_digest:
                    raise CheckpointError(
                        "failed tool result_digest does not match result",
                        code="operation_result_digest_mismatch",
                    )
                expected_error = operation_error_from_tool_result(parsed_result)
                if self.error != expected_error:
                    raise CheckpointError(
                        "failed tool error projection does not match result",
                        code="operation_error_projection_mismatch",
                    )
                if self.error.code == "tool_outcome_unknown":
                    if self.resume_observation is None:
                        raise CheckpointError(
                            "unknown tool outcome requires a resume observation",
                            code="operation_resume_observation_required",
                        )
                elif self.resume_observation is not None:
                    raise CheckpointError(
                        "ordinary failed tool receipt cannot contain resume_observation",
                        code="operation_resume_observation_forbidden",
                    )
        elif self.state is OperationState.DEFERRED:
            # The tool-specific branch above enforces the closed deferred
            # shape; model entries can never be deferred.
            if self.kind is OperationKind.MODEL:
                raise CheckpointError(
                    "model operation entries cannot be deferred",
                    code="operation_deferred_fields_invalid",
                )
        elif (
            self.response is not None
            or self.result is not None
            or self.error is not None
            or self.identity_key is not None
            or self.result_digest is not None
            or self.resume_observation is not None
            or self.deferred_handle is not None
        ):
            raise CheckpointError(
                "non-terminal operation cannot contain a receipt",
                code="operation_receipt_unexpected",
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind.value,
            "operation_id": self.operation_id,
            "cycle_index": self.cycle_index,
            "attempt": self.attempt,
            "state": self.state.value,
            "request_digest": self.request_digest,
            "idempotency_key": self.idempotency_key,
        }
        if self.kind is OperationKind.MODEL:
            model_operation = self.model_operation
            assert model_operation is not None
            payload.update(
                {
                    "response": self.response,
                    "model_operation": model_operation.value,
                    "backend": self.backend,
                    "model": self.model,
                    "call_id": self.call_id,
                }
            )
        else:
            idempotency_support = self.idempotency_support
            assert idempotency_support is not None
            payload.update(
                {
                    "tool_call_id": self.tool_call_id,
                    "tool_name": self.tool_name,
                    "arguments": self.arguments,
                    "idempotency_support": idempotency_support.value,
                    "result": self.result,
                }
            )
            if self.deferred_handle is not None:
                payload["deferred_handle"] = self.deferred_handle.to_dict()
            if self.identity_key is not None:
                payload["identity_key"] = self.identity_key
            if self.result_digest is not None:
                payload["result_digest"] = self.result_digest
            if self.resume_observation is not None:
                payload["resume_observation"] = self.resume_observation.to_dict()
        payload["error"] = self.error.to_dict() if self.error is not None else None
        return payload

    def verify_request(self, request: dict[str, Any]) -> None:
        if compute_operation_request_digest(request) != self.request_digest:
            raise CheckpointError(
                "operation request digest does not match the durable journal",
                code="checkpoint_journal_integrity_mismatch",
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OperationJournalEntry:
        if not isinstance(payload, dict):
            raise CheckpointError(
                "operation journal entry must be an object",
                code="operation_entry_invalid",
            )
        try:
            kind = OperationKind(payload["kind"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(
                "operation kind is invalid",
                code="operation_kind_fields_invalid",
            ) from exc
        try:
            state = OperationState(_required_string(payload, "state"))
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                "operation state is invalid",
                code="operation_state_invalid",
            ) from exc
        common_fields = {
            "kind",
            "operation_id",
            "cycle_index",
            "attempt",
            "request_digest",
            "idempotency_key",
            "state",
        }
        model_fields = common_fields | {
            "response",
            "error",
            "model_operation",
            "backend",
            "model",
            "call_id",
        }
        tool_fields = common_fields | {
            "tool_call_id",
            "tool_name",
            "arguments",
            "idempotency_support",
            "result",
            "error",
        }
        try:
            error_payload = payload["error"]
        except KeyError:
            error_payload = None
        is_synthetic_closure = (
            kind is OperationKind.TOOL
            and state is OperationState.FAILED
            and isinstance(error_payload, dict)
            and error_payload.get("code") == "tool_cancelled"
        )
        is_unknown_outcome = (
            kind is OperationKind.TOOL
            and state is OperationState.FAILED
            and isinstance(error_payload, dict)
            and error_payload.get("code") == "tool_outcome_unknown"
        )
        if kind is OperationKind.MODEL and not {
            "model_operation",
            "backend",
            "model",
            "call_id",
        }.issubset(payload):
            raise CheckpointError(
                "model journal entry is missing model identity fields",
                code="model_identity_invalid",
            )
        if kind is OperationKind.MODEL:
            expected_fields = model_fields
        elif state is OperationState.DEFERRED:
            expected_fields = tool_fields | {"deferred_handle"}
        elif state is OperationState.SUCCEEDED:
            expected_fields = tool_fields | {"identity_key", "result_digest"}
        elif state is OperationState.FAILED and is_synthetic_closure:
            expected_fields = tool_fields | {"identity_key", "resume_observation"}
        elif state is OperationState.FAILED:
            expected_fields = tool_fields | {"identity_key", "result_digest"}
            if is_unknown_outcome:
                expected_fields.add("resume_observation")
        else:
            expected_fields = tool_fields

        missing_fields = expected_fields - set(payload)
        if missing_fields:
            if kind is OperationKind.MODEL and missing_fields & {
                "model_operation",
                "backend",
                "model",
                "call_id",
            }:
                raise CheckpointError(
                    "model journal entry is missing model identity fields",
                    code="model_identity_invalid",
                )
            if kind is OperationKind.MODEL and state is OperationState.FAILED and "error" in missing_fields:
                raise CheckpointError(
                    "failed operation requires exactly one typed error",
                    code="operation_error_required",
                )
            if kind is OperationKind.MODEL and state is OperationState.SUCCEEDED and "response" in missing_fields:
                raise CheckpointError(
                    "succeeded operation requires exactly one success receipt",
                    code="operation_receipt_required",
                )
            if kind is OperationKind.TOOL:
                if state is OperationState.SUCCEEDED and ("result" in missing_fields or payload["result"] is None):
                    raise CheckpointError(
                        "succeeded operation requires exactly one success receipt",
                        code="operation_receipt_required",
                    )
                if "identity_key" in missing_fields:
                    raise CheckpointError(
                        "tool receipt requires identity_key",
                        code="operation_receipt_identity_required",
                    )
                if "result_digest" in missing_fields:
                    raise CheckpointError(
                        "tool receipt requires result_digest",
                        code="operation_result_digest_required",
                    )
                if "result" in missing_fields and state is OperationState.FAILED:
                    raise CheckpointError(
                        "failed tool receipt requires result",
                        code="operation_result_required",
                    )
                if "error" in missing_fields and state is OperationState.FAILED:
                    raise CheckpointError(
                        "failed operation requires exactly one typed error",
                        code="operation_error_required",
                    )
                if "resume_observation" in missing_fields and (is_synthetic_closure or is_unknown_outcome):
                    raise CheckpointError(
                        "unknown tool outcome requires a resume observation",
                        code="operation_resume_observation_required",
                    )
                if "deferred_handle" in missing_fields:
                    raise CheckpointError(
                        "deferred operation requires an exact handle and no receipt",
                        code="operation_deferred_fields_invalid",
                    )
            raise CheckpointError(
                "operation journal entry is missing required fields",
                code="operation_kind_fields_invalid",
            )
        if kind is OperationKind.TOOL and state is OperationState.FAILED:
            if is_synthetic_closure and "result_digest" in payload:
                raise CheckpointError(
                    "synthetic tool outcome cannot contain a definitive receipt",
                    code="operation_closure_receipt_forbidden",
                )
            if not is_synthetic_closure and not is_unknown_outcome and "resume_observation" in payload:
                raise CheckpointError(
                    "ordinary failed tool receipt cannot contain resume_observation",
                    code="operation_resume_observation_forbidden",
                )
        if (
            kind is OperationKind.TOOL
            and state
            in {
                OperationState.PLANNED,
                OperationState.STARTED,
                OperationState.DEFERRED,
                OperationState.AMBIGUOUS,
            }
            and "resume_observation" in payload
        ):
            raise CheckpointError(
                "non-terminal operation cannot contain a receipt",
                code=(
                    "operation_deferred_fields_invalid" if state is OperationState.DEFERRED else "operation_receipt_unexpected"
                ),
            )
        extra_fields = set(payload) - expected_fields
        cross_kind_fields = (tool_fields if kind is OperationKind.MODEL else model_fields) & extra_fields
        if extra_fields:
            raise CheckpointError(
                "operation journal contains unknown fields",
                code=("operation_kind_fields_invalid" if cross_kind_fields else "operation_entry_unknown_field"),
            )
        return cls(
            kind=kind,
            operation_id=_required_string(payload, "operation_id"),
            cycle_index=_required_integer(payload, "cycle_index"),
            attempt=_required_integer(payload, "attempt"),
            state=state,
            request_digest=_required_string(payload, "request_digest"),
            idempotency_key=payload["idempotency_key"],
            response=payload["response"] if kind is OperationKind.MODEL else None,
            result=payload["result"] if kind is OperationKind.TOOL else None,
            error=OperationError.from_dict(payload["error"]) if payload["error"] is not None else None,
            tool_call_id=payload["tool_call_id"] if kind is OperationKind.TOOL else None,
            tool_name=payload["tool_name"] if kind is OperationKind.TOOL else None,
            arguments=payload["arguments"] if kind is OperationKind.TOOL else None,
            idempotency_support=payload["idempotency_support"] if kind is OperationKind.TOOL else None,
            model_operation=payload["model_operation"] if kind is OperationKind.MODEL else None,
            backend=payload["backend"] if kind is OperationKind.MODEL else None,
            model=payload["model"] if kind is OperationKind.MODEL else None,
            call_id=payload["call_id"] if kind is OperationKind.MODEL else None,
            deferred_handle=(
                DeferredToolHandle.from_dict(payload["deferred_handle"]) if state is OperationState.DEFERRED else None
            ),
            identity_key=(
                payload["identity_key"]
                if kind is OperationKind.TOOL and state in {OperationState.SUCCEEDED, OperationState.FAILED}
                else None
            ),
            result_digest=(
                payload["result_digest"]
                if kind is OperationKind.TOOL
                and state in {OperationState.SUCCEEDED, OperationState.FAILED}
                and not is_synthetic_closure
                else None
            ),
            resume_observation=(
                ResumeObservation.from_dict(payload["resume_observation"])
                if kind is OperationKind.TOOL and state is OperationState.FAILED and (is_synthetic_closure or is_unknown_outcome)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ExtensionStateEntry:
    version: str
    required: bool
    state: Any

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("checkpoint extension version must be non-empty")
        if not isinstance(self.required, bool):
            raise TypeError("checkpoint extension required must be a boolean")
        canonical_json_bytes(self.state, "checkpoint extension state")

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "required": self.required, "state": self.state}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExtensionStateEntry:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint extension state entry must be an object")
        if set(payload) != {"version", "required", "state"}:
            raise ValueError("checkpoint extension state entry has missing or unknown fields")
        return cls(
            version=_required_string(payload, "version"),
            required=_required_boolean(payload, "required"),
            state=payload.get("state"),
        )


@dataclass(slots=True)
class EventOutboxEntry:
    event_id: str
    payload_digest: str
    state: str
    event: dict[str, Any]
    cursor: EventCursor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("outbox event_id must be non-empty")
        validate_sha256(self.payload_digest, "outbox payload_digest")
        if self.state not in {"pending", "delivered"}:
            raise ValueError("outbox state must be pending or delivered")
        if not isinstance(self.event, dict):
            raise ValueError("outbox event must be an object")
        from vv_agent.events import event_from_dict

        try:
            event = event_from_dict(self.event)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                "outbox event must match the current RunEvent wire contract",
                code="checkpoint_event_invalid",
            ) from exc
        if event.event_id != self.event_id:
            raise CheckpointError(
                "outbox event_id must match the embedded RunEvent event_id",
                code="event_identity_conflict",
            )
        if event.to_dict() != self.event:
            raise CheckpointError(
                "outbox event must use the canonical current RunEvent shape",
                code="checkpoint_event_invalid",
            )
        canonical_json_bytes(self.event, "outbox event")
        if self.state == "pending" and self.cursor is not None:
            raise ValueError("pending outbox event cannot have a cursor")
        if self.state == "delivered" and self.cursor is None:
            raise ValueError("delivered outbox event requires a cursor")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "payload_digest": self.payload_digest,
            "state": self.state,
            "event": dict(self.event),
            "cursor": self.cursor.to_dict() if self.cursor is not None else None,
        }

    def verify_payload(self) -> None:
        if compute_event_payload_digest(self.event) != self.payload_digest:
            raise CheckpointError(
                "event payload digest does not match the durable outbox entry",
                code="event_identity_conflict",
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EventOutboxEntry:
        if not isinstance(payload, dict):
            raise ValueError("event outbox entry must be an object")
        if set(payload) != {"event_id", "payload_digest", "state", "event", "cursor"}:
            raise ValueError("event outbox entry has missing or unknown fields")
        cursor_raw = payload.get("cursor")
        return cls(
            event_id=_required_string(payload, "event_id"),
            payload_digest=_required_string(payload, "payload_digest"),
            state=_required_string(payload, "state"),
            event=_required_object(payload, "event"),
            cursor=EventCursor.from_dict(cursor_raw) if cursor_raw is not None else None,
        )

    @classmethod
    def pending(cls, event_id: str, event: dict[str, Any]) -> EventOutboxEntry:
        return cls(
            event_id=event_id,
            payload_digest=compute_event_payload_digest(event),
            state="pending",
            event=event,
        )


@dataclass(slots=True)
class Checkpoint:
    checkpoint_key: str
    task_id: str
    root_run_id: str
    trace_id: str
    run_definition: dict[str, Any]
    run_definition_digest: str
    resume_attempt: int
    cycle_index: int
    status: AgentStatus
    messages: list[Message]
    cycles: list[CycleRecord]
    cancel_requested: bool = False
    active_host_interaction: dict[str, Any] | None = None
    suspended_origin: dict[str, Any] | None = None
    model_calls: list[ModelCallRecord] = field(default_factory=list)
    shared_state: dict[str, Any] = field(default_factory=dict)
    budget_usage: BudgetUsageSnapshot | None = None
    event_cursor: EventCursor | None = None
    event_outbox: list[EventOutboxEntry] = field(default_factory=list)
    extension_state: dict[str, ExtensionStateEntry] = field(default_factory=dict)
    model_call_journal: list[OperationJournalEntry] = field(default_factory=list)
    tool_journal: list[OperationJournalEntry] = field(default_factory=list)
    revision: int = 0
    claim_token: str | None = None
    claimed_cycle: int | None = None
    lease_expires_at_ms: int | None = None
    terminal_result: AgentResult | None = None
    terminal_acknowledged: bool = False
    schema_version: str = CHECKPOINT_SCHEMA
    run_definition_schema: str = RUN_DEFINITION_SCHEMA


@runtime_checkable
class CheckpointStore(Protocol):
    def create_checkpoint(self, checkpoint: Checkpoint) -> bool: ...

    def load_checkpoint(self, checkpoint_key: str) -> Checkpoint | None: ...

    def claim_checkpoint(
        self,
        checkpoint_key: str,
        cycle_index: int,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
        claim_mode: ClaimMode,
    ) -> Checkpoint | None: ...

    def progress_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool: ...

    def suspend_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool: ...

    def commit_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool: ...

    def finalize_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        expected_revision: int,
    ) -> bool: ...

    def finalize_claimed_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool: ...

    def record_event_delivery(
        self,
        checkpoint_key: str,
        *,
        event_id: str,
        payload_digest: str,
        cursor: EventCursor,
        expected_revision: int,
        claim_token: str | None,
    ) -> bool: ...

    def renew_checkpoint_claim(
        self,
        checkpoint_key: str,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> CheckpointRenewal: ...

    def record_tool_receipt(
        self,
        checkpoint: Checkpoint,
        *,
        operation_id: str,
        attempt: int,
        tool_call_id: str,
        request_digest: str,
        result: ToolExecutionResult,
        claim_token: str,
        expected_revision: int,
        claimed_cycle: int,
    ) -> bool: ...

    def acknowledge_terminal(self, checkpoint_key: str, *, expected_revision: int) -> bool: ...

    def delete_checkpoint(self, checkpoint_key: str) -> None: ...

    def preflight_tool_batch(
        self,
        checkpoint: Checkpoint,
        *,
        tool_call_count: int,
        expected_revision: int,
        claim_token: str,
        claimed_cycle: int,
    ) -> bool: ...

    def admit_deferred_batch(
        self,
        checkpoint: Checkpoint,
        *,
        outcomes: list[Any],
        claim_token: str,
        expected_revision: int,
        claimed_cycle: int,
    ) -> bool: ...

    def resolve_deferred(self, handle: DeferredToolHandle, result: ToolExecutionResult) -> DeferredResolveDecision: ...

    def accept_deferred_batch(
        self,
        checkpoint: Checkpoint,
        *,
        decisions: list[Any],
        claim_token: str,
        expected_revision: int,
        claimed_cycle: int,
    ) -> bool: ...

    def admit_controller_command(self, command: Any) -> Any: ...

    def get_controller_command_receipt(self, command_id: str) -> Any: ...

    def get_controller_command(self, command_id: str) -> Any: ...

    def resolve_controller_command(self, command: Any) -> Any: ...

    def claim_controller_command_wake(
        self,
        *,
        command_id: str,
        command_digest: str,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> Any: ...

    def complete_controller_command_wake(
        self,
        *,
        command_id: str,
        command_digest: str,
        claim_token: str,
        attempt: int,
        outcome: str,
        now_ms: int,
        error: str | None = None,
    ) -> Any: ...

    def reconcile_controller_command_wake(
        self,
        *,
        command_id: str,
        command_digest: str,
        outcome: str,
        now_ms: int,
    ) -> Any: ...

    def reap_controller_command_wakes(self, checkpoint_key: str, now_ms: int) -> list[Any]: ...

    def produce_host_interaction(
        self,
        request: Any,
        *,
        admission_context: HostInteractionAdmissionContext,
    ) -> Any: ...

    def claim_and_consume_host_interaction_response(self, envelope: Any) -> Any: ...

    def reap_host_interaction_record(self, *, record_id: str, checkpoint_key: str, now_ms: int) -> Any: ...

    def get_host_interaction_notification(self, notification_id: str) -> dict[str, Any] | None: ...

    def claim_host_interaction_notification(
        self,
        *,
        notification_id: str,
        payload_digest: str,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> Any: ...

    def complete_host_interaction_notification(
        self,
        *,
        notification_id: str,
        payload_digest: str,
        claim_token: str,
        attempt: int,
        outcome: str,
        now_ms: int,
        error: str | None = None,
    ) -> Any: ...

    def reconcile_host_interaction_notification(
        self,
        *,
        notification_id: str,
        payload_digest: str,
        outcome: str,
        now_ms: int,
        abort_reason: str | None = None,
    ) -> Any: ...


def validate_checkpoint(checkpoint: Checkpoint) -> None:
    if not isinstance(checkpoint, Checkpoint):
        raise TypeError("checkpoint must be a Checkpoint")
    if checkpoint.schema_version != CHECKPOINT_SCHEMA:
        raise CheckpointError(
            f"unsupported checkpoint schema_version; expected {CHECKPOINT_SCHEMA}",
            code="checkpoint_schema_unsupported",
        )
    if checkpoint.run_definition_schema != RUN_DEFINITION_SCHEMA:
        raise CheckpointError(
            "unsupported checkpoint run_definition_schema",
            code="checkpoint_definition_schema_unsupported",
        )
    for value, name in (
        (checkpoint.checkpoint_key, "checkpoint_key"),
        (checkpoint.task_id, "task_id"),
        (checkpoint.root_run_id, "root_run_id"),
        (checkpoint.trace_id, "trace_id"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise CheckpointError(
                f"checkpoint {name} must be non-empty",
                code=("checkpoint_key_invalid" if name == "checkpoint_key" else "checkpoint_identity_invalid"),
            )
    if len(checkpoint.checkpoint_key.encode("utf-8")) > 512:
        raise CheckpointError(
            "checkpoint_key must be at most 512 UTF-8 bytes",
            code="checkpoint_key_invalid",
        )
    expected_definition_digest = compute_run_definition_digest(checkpoint.run_definition)
    try:
        validate_sha256(checkpoint.run_definition_digest, "run_definition_digest")
    except ValueError as exc:
        raise CheckpointError(
            str(exc),
            code="checkpoint_definition_digest_invalid",
        ) from exc
    if checkpoint.run_definition_digest != expected_definition_digest:
        raise CheckpointError(
            "run_definition_digest does not match the embedded run_definition",
            code="checkpoint_definition_mismatch",
        )
    try:
        _positive_wire_integer(checkpoint.resume_attempt, "resume_attempt")
    except ValueError as exc:
        raise CheckpointError(str(exc), code="checkpoint_resume_attempt_invalid") from exc
    try:
        _wire_integer(checkpoint.cycle_index, "cycle_index")
        _wire_integer(checkpoint.revision, "revision")
    except ValueError as exc:
        raise CheckpointError(str(exc), code="checkpoint_revision_invalid") from exc
    if not isinstance(checkpoint.status, AgentStatus):
        raise TypeError("checkpoint status must be an AgentStatus")
    if not isinstance(checkpoint.cancel_requested, bool):
        raise CheckpointError(
            "checkpoint cancel_requested must be a boolean",
            code="checkpoint_status_invalid",
        )
    _validate_host_interaction_request(checkpoint.active_host_interaction, "active_host_interaction")
    _validate_suspended_origin(checkpoint.suspended_origin)
    if checkpoint.status is AgentStatus.HOST_INTERACTION:
        if checkpoint.active_host_interaction is None or checkpoint.suspended_origin is not None:
            raise CheckpointError(
                "host_interaction status requires active_host_interaction and no suspended_origin",
                code="checkpoint_status_invalid",
            )
    elif checkpoint.status is AgentStatus.SUSPENDED:
        if checkpoint.active_host_interaction is not None or checkpoint.suspended_origin is None:
            raise CheckpointError(
                "suspended status requires suspended_origin and no active_host_interaction",
                code="checkpoint_status_invalid",
            )
    elif checkpoint.active_host_interaction is not None or checkpoint.suspended_origin is not None:
        raise CheckpointError(
            "active_host_interaction and suspended_origin are only valid for waiting statuses",
            code="checkpoint_status_invalid",
        )
    if not isinstance(checkpoint.messages, list) or not all(isinstance(item, Message) for item in checkpoint.messages):
        raise TypeError("checkpoint messages must contain Message values")
    if not isinstance(checkpoint.cycles, list) or not all(isinstance(item, CycleRecord) for item in checkpoint.cycles):
        raise TypeError("checkpoint cycles must contain CycleRecord values")
    if not isinstance(checkpoint.model_calls, list) or not all(
        isinstance(item, ModelCallRecord) for item in checkpoint.model_calls
    ):
        raise TypeError("checkpoint model_calls must contain ModelCallRecord values")
    call_ids = [record.call_id for record in checkpoint.model_calls]
    if len(call_ids) != len(set(call_ids)):
        raise CheckpointError(
            "checkpoint model_calls contains duplicate call ids",
            code="checkpoint_status_invalid",
        )
    canonical_json_bytes(checkpoint.shared_state, "checkpoint shared_state")
    if checkpoint.budget_usage is not None and not isinstance(
        checkpoint.budget_usage,
        BudgetUsageSnapshot,
    ):
        raise TypeError("checkpoint budget_usage must be a BudgetUsageSnapshot or None")
    if checkpoint.event_cursor is not None and not isinstance(checkpoint.event_cursor, EventCursor):
        raise TypeError("checkpoint event_cursor must be an EventCursor or None")
    if not isinstance(checkpoint.event_outbox, list) or not all(
        isinstance(item, EventOutboxEntry) for item in checkpoint.event_outbox
    ):
        raise TypeError("checkpoint event_outbox must contain EventOutboxEntry values")
    event_ids: set[str] = set()
    for entry in checkpoint.event_outbox:
        entry.verify_payload()
        if entry.event_id in event_ids:
            raise CheckpointError(
                f"checkpoint event_outbox contains duplicate event id {entry.event_id!r}",
                code="event_identity_conflict",
            )
        event_ids.add(entry.event_id)
    if not isinstance(checkpoint.extension_state, dict) or not all(
        isinstance(entry, ExtensionStateEntry) for entry in checkpoint.extension_state.values()
    ):
        raise TypeError("checkpoint extension_state must contain ExtensionStateEntry values")
    if not isinstance(checkpoint.model_call_journal, list) or not isinstance(
        checkpoint.tool_journal,
        list,
    ):
        raise TypeError("checkpoint journals must be arrays")
    claim_values = (
        checkpoint.claim_token,
        checkpoint.claimed_cycle,
        checkpoint.lease_expires_at_ms,
    )
    if any(value is None for value in claim_values) != all(value is None for value in claim_values):
        raise CheckpointError(
            "checkpoint claim fields must be all present or all null",
            code="checkpoint_claim_invalid",
        )
    if checkpoint.claim_token is not None:
        if not checkpoint.claim_token:
            raise CheckpointError(
                "checkpoint claim_token must be non-empty",
                code="checkpoint_claim_invalid",
            )
        if checkpoint.claimed_cycle != checkpoint.cycle_index + 1:
            raise CheckpointError(
                "checkpoint claimed_cycle must equal cycle_index + 1",
                code="checkpoint_claim_invalid",
            )
        try:
            _positive_wire_integer(checkpoint.lease_expires_at_ms, "lease_expires_at_ms")
        except ValueError as exc:
            raise CheckpointError(str(exc), code="checkpoint_claim_invalid") from exc
    if checkpoint.terminal_result is not None and checkpoint.claim_token is not None:
        raise CheckpointError(
            "terminal checkpoint cannot have an active claim",
            code="checkpoint_status_invalid",
        )
    if checkpoint.terminal_acknowledged and checkpoint.terminal_result is None:
        raise CheckpointError(
            "terminal acknowledgement requires a terminal result",
            code="checkpoint_status_invalid",
        )
    if checkpoint.terminal_result is None and checkpoint.status not in {
        AgentStatus.RUNNING,
        AgentStatus.HOST_INTERACTION,
        AgentStatus.SUSPENDED,
        AgentStatus.DEFERRED,
        AgentStatus.RECONCILIATION_REQUIRED,
    }:
        raise CheckpointError(
            "non-terminal checkpoint status must be running or reconciliation_required",
            code="checkpoint_status_invalid",
        )
    active_cycle = checkpoint.claimed_cycle or (checkpoint.cycle_index + 1)
    journals = [*checkpoint.model_call_journal, *checkpoint.tool_journal]
    if not all(isinstance(entry, OperationJournalEntry) for entry in journals):
        raise TypeError("checkpoint journals must contain OperationJournalEntry values")
    for entry in journals:
        if entry.cycle_index != active_cycle:
            raise CheckpointError(
                "journal cycle_index must equal active cycle",
                code="checkpoint_journal_cycle_invalid",
            )
        entry.__post_init__()
    validate_model_journal_accounting(checkpoint)
    for entry in checkpoint.tool_journal:
        if entry.kind is not OperationKind.TOOL:
            raise CheckpointError(
                "tool_journal contains a non-tool entry",
                code="operation_kind_fields_invalid",
            )
        if entry.state is OperationState.DEFERRED:
            handle = entry.deferred_handle
            if handle is None or (
                handle.checkpoint_key != checkpoint.checkpoint_key
                or handle.operation_id != entry.operation_id
                or handle.attempt != entry.attempt
                or handle.request_digest != entry.request_digest
            ):
                raise CheckpointError(
                    "deferred handle identity does not match its journal entry",
                    code="deferred_handle_invalid",
                )
            continue
        if entry.state not in {OperationState.SUCCEEDED, OperationState.FAILED}:
            continue
        expected_identity = compute_tool_identity_key(
            checkpoint.checkpoint_key,
            entry.operation_id,
            entry.attempt,
            entry.tool_call_id or "",
            entry.request_digest,
        )
        if entry.identity_key != expected_identity:
            raise CheckpointError(
                "terminal tool journal identity_key does not match its identity fields",
                code="tool_receipt_identity_invalid",
            )
        if entry.result is not None:
            try:
                parsed_result = ToolExecutionResult.from_dict(entry.result)
            except (TypeError, ValueError) as exc:
                raise CheckpointError(
                    "terminal tool journal result is invalid",
                    code="tool_receipt_identity_invalid",
                ) from exc
            if parsed_result.tool_call_id != entry.tool_call_id:
                raise CheckpointError(
                    "terminal tool result tool_call_id does not match its journal entry",
                    code="tool_receipt_identity_invalid",
                )
        if entry.result_digest is None:
            continue
        expected_event_id = f"evt_receipt_{entry.identity_key}"
        matching_events = [item for item in checkpoint.event_outbox if item.event_id == expected_event_id]
        if len(matching_events) != 1 or matching_events[0].event.get("type") != "tool_call_completed":
            raise CheckpointError(
                "terminal tool journal has no canonical receipt completion event",
                code="tool_receipt_identity_invalid",
            )
        completed_event = matching_events[0].event
        if any(
            completed_event.get(field) != expected
            for field, expected in (
                ("operation_id", entry.operation_id),
                ("attempt", entry.attempt),
                ("tool_call_id", entry.tool_call_id),
            )
        ):
            raise CheckpointError(
                "receipt completion event does not match its terminal tool journal",
                code="tool_receipt_identity_invalid",
            )
        if "checkpoint_key" in completed_event and completed_event["checkpoint_key"] != checkpoint.checkpoint_key:
            raise CheckpointError(
                "receipt completion event checkpoint_key does not match its checkpoint",
                code="tool_receipt_identity_invalid",
            )
    ambiguous = [entry for entry in journals if entry.state is OperationState.AMBIGUOUS]
    deferred_entries = [entry for entry in checkpoint.tool_journal if entry.state is OperationState.DEFERRED]
    if checkpoint.status is AgentStatus.DEFERRED:
        if checkpoint.claim_token is not None or not deferred_entries or checkpoint.terminal_result is not None:
            raise CheckpointError(
                "deferred status requires deferred journal entries and no claim",
                code="checkpoint_status_invalid",
            )
        if any(entry.cycle_index != active_cycle for entry in deferred_entries):
            raise CheckpointError(
                "deferred journal cycle is not active",
                code="checkpoint_journal_cycle_invalid",
            )
    elif checkpoint.status is AgentStatus.RECONCILIATION_REQUIRED:
        if checkpoint.claim_token is not None or not ambiguous or checkpoint.terminal_result is not None:
            raise CheckpointError(
                "reconciliation_required requires ambiguity and no claim or terminal result",
                code="checkpoint_status_invalid",
            )
    elif checkpoint.status is AgentStatus.RUNNING and (ambiguous or deferred_entries) and checkpoint.claim_token is None:
        raise CheckpointError(
            "running checkpoint ambiguity or deferred barrier requires an active claim",
            code="checkpoint_status_invalid",
        )
    if checkpoint.terminal_result is not None:
        if checkpoint.terminal_result.status is not checkpoint.status:
            raise CheckpointError(
                "checkpoint terminal status must match terminal_result status",
                code="checkpoint_status_invalid",
            )
        if checkpoint.terminal_result.checkpoint_key not in {None, checkpoint.checkpoint_key}:
            raise CheckpointError(
                "terminal_result checkpoint_key does not match checkpoint",
                code="checkpoint_status_invalid",
            )
        if checkpoint.terminal_result.token_usage.model_calls != checkpoint.model_calls:
            raise CheckpointError(
                "terminal result model-call ledger does not match checkpoint",
                code="checkpoint_status_invalid",
            )
        if any(
            entry.state in {OperationState.PLANNED, OperationState.STARTED, OperationState.DEFERRED, OperationState.AMBIGUOUS}
            for entry in journals
        ):
            raise CheckpointError(
                "terminal checkpoint cannot retain active journals",
                code="checkpoint_status_invalid",
            )
    for namespace in checkpoint.extension_state:
        try:
            validate_extension_namespace(namespace)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                str(exc),
                code="checkpoint_extension_namespace_invalid",
            ) from exc


def validate_checkpoint_creation(checkpoint: Checkpoint) -> None:
    validate_checkpoint(checkpoint)

    def invalid(message: str) -> CheckpointError:
        return CheckpointError(message, code="checkpoint_initial_invalid")

    if checkpoint.revision != 0:
        raise invalid("new checkpoints must start at revision zero")
    if checkpoint.resume_attempt != 1:
        raise invalid("new checkpoints must start at resume_attempt one")
    if any(value is not None for value in (checkpoint.claim_token, checkpoint.claimed_cycle, checkpoint.lease_expires_at_ms)):
        raise invalid("new checkpoints must not carry an execution claim")
    if checkpoint.status in {AgentStatus.WAIT_USER, AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.MAX_CYCLES}:
        raise invalid("new checkpoints must be non-terminal")
    if checkpoint.terminal_result is not None or checkpoint.terminal_acknowledged:
        raise invalid("new checkpoints must be non-terminal")
    has_only_creation_event = not checkpoint.event_outbox
    if len(checkpoint.event_outbox) == 1:
        entry = checkpoint.event_outbox[0]
        event = entry.event
        has_only_creation_event = (
            entry.state == "pending"
            and entry.cursor is None
            and event.get("type") == "checkpoint_created"
            and event.get("cycle_index") == 0
            and event.get("checkpoint_key") == checkpoint.checkpoint_key
            and event.get("resume_attempt") == 1
        )
    if (
        checkpoint.cancel_requested
        or checkpoint.active_host_interaction is not None
        or checkpoint.suspended_origin is not None
        or checkpoint.event_cursor is not None
        or checkpoint.cycles
        or checkpoint.model_calls
        or not has_only_creation_event
        or checkpoint.model_call_journal
        or checkpoint.tool_journal
    ):
        raise CheckpointError(
            "new checkpoints must not carry persisted lifecycle state",
            code="checkpoint_initial_invalid",
        )


def _validate_host_interaction_request(value: Any, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != _HOST_INTERACTION_REQUEST_FIELDS:
        raise CheckpointError(
            f"{field_name} must be a closed HostInteractionRequest object",
            code="host_interaction_fields_invalid",
        )
    if value.get("schema_version") != HOST_INTERACTION_REQUEST_SCHEMA:
        raise CheckpointError(
            f"{field_name} schema_version is unsupported",
            code="host_interaction_fields_invalid",
        )
    for key in ("interaction_id", "operation_id", "tool_call_id"):
        text = value.get(key)
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 512:
            raise CheckpointError(
                f"{field_name}.{key} is invalid",
                code="host_interaction_fields_invalid",
            )
    logical_cycle = value.get("logical_cycle")
    if isinstance(logical_cycle, bool) or not isinstance(logical_cycle, int) or logical_cycle < 1:
        raise CheckpointError(
            f"{field_name}.logical_cycle is invalid",
            code="host_interaction_fields_invalid",
        )
    prompt = value.get("prompt")
    if not isinstance(prompt, str) or not prompt or len(prompt.encode("utf-8")) > 65_536:
        raise CheckpointError(
            f"{field_name}.prompt is invalid",
            code="host_interaction_content_too_large" if isinstance(prompt, str) else "host_interaction_fields_invalid",
        )
    request_digest = value.get("request_digest")
    if not isinstance(request_digest, str) or len(request_digest.encode("utf-8")) != 64:
        raise CheckpointError(
            f"{field_name}.request_digest is invalid",
            code="host_interaction_fields_invalid",
        )
    try:
        validate_sha256(request_digest, f"{field_name}.request_digest")
        expected = canonical_json_sha256(
            {key: value[key] for key in sorted(_HOST_INTERACTION_REQUEST_FIELDS - {"request_digest"})},
            f"{field_name} request",
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointError(str(exc), code="host_interaction_fields_invalid") from exc
    if request_digest != expected:
        raise CheckpointError(
            f"{field_name}.request_digest does not match request",
            code="host_interaction_request_digest_invalid",
        )


def _validate_suspended_origin(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != _SUSPENDED_ORIGIN_FIELDS:
        raise CheckpointError(
            "suspended_origin must be a closed object",
            code="checkpoint_status_invalid",
        )
    status = value.get("status")
    if status not in {AgentStatus.RUNNING.value, AgentStatus.HOST_INTERACTION.value}:
        raise CheckpointError(
            "suspended_origin.status is invalid",
            code="checkpoint_status_invalid",
        )
    _validate_host_interaction_request(value.get("active_host_interaction"), "suspended_origin.active_host_interaction")
    if status == AgentStatus.RUNNING.value and value.get("active_host_interaction") is not None:
        raise CheckpointError(
            "running suspended_origin cannot contain active_host_interaction",
            code="checkpoint_status_invalid",
        )
    if status == AgentStatus.HOST_INTERACTION.value and value.get("active_host_interaction") is None:
        raise CheckpointError(
            "host_interaction suspended_origin requires active_host_interaction",
            code="checkpoint_status_invalid",
        )


def validate_model_journal_accounting(checkpoint: Checkpoint) -> None:
    for entry in checkpoint.model_call_journal:
        if entry.kind is not OperationKind.MODEL:
            raise CheckpointError(
                "model_call_journal contains a non-model entry",
                code="operation_kind_fields_invalid",
            )
        _validate_model_journal_entry_accounting(checkpoint, entry)


def _validate_model_journal_entry_accounting(checkpoint: Checkpoint, journal: OperationJournalEntry) -> None:
    identity = _model_journal_identity(journal)
    record_candidates = [
        record
        for record in checkpoint.model_calls
        if record.call_id == journal.call_id
        or (record.operation_id == journal.operation_id and record.attempt == journal.attempt)
    ]
    event_candidates = [
        entry.event
        for entry in checkpoint.event_outbox
        if entry.event.get("type") in {"model_call_started", "model_call_completed", "model_call_failed"}
        and (
            entry.event.get("call_id") == journal.call_id
            or (entry.event.get("operation_id") == journal.operation_id and entry.event.get("attempt") == journal.attempt)
        )
    ]
    started_events = [event for event in event_candidates if event["type"] == "model_call_started"]
    terminal_events = [event for event in event_candidates if event["type"] in {"model_call_completed", "model_call_failed"}]

    if len(record_candidates) > 1 or len(started_events) > 1 or len(terminal_events) > 1:
        raise CheckpointError(
            "model journal attempt has duplicate accounting evidence",
            code="checkpoint_status_invalid",
        )

    if journal.state is OperationState.PLANNED:
        _require_model_evidence_counts(record_candidates, started_events, terminal_events, expected=(0, 0, 0))
        return
    if journal.state is OperationState.STARTED:
        _require_model_evidence_counts(record_candidates, started_events, terminal_events, expected=(0, 1, 0))
        _require_model_identity(identity, _model_event_identity(started_events[0]))
        return

    evidence_present = bool(record_candidates or started_events or terminal_events)
    if journal.state is OperationState.FAILED and not evidence_present:
        return
    _require_model_evidence_counts(record_candidates, started_events, terminal_events, expected=(1, 1, 1))

    record = record_candidates[0]
    started_event = started_events[0]
    terminal_event = terminal_events[0]
    _require_model_identity(identity, _model_record_identity(record))
    _require_model_identity(identity, _model_event_identity(started_event))
    _require_model_identity(identity, _model_event_identity(terminal_event))

    expected_statuses = {
        OperationState.SUCCEEDED: {ModelCallStatus.COMPLETED, ModelCallStatus.AMBIGUOUS},
        OperationState.FAILED: {ModelCallStatus.FAILED, ModelCallStatus.AMBIGUOUS},
        OperationState.AMBIGUOUS: {ModelCallStatus.AMBIGUOUS},
    }.get(journal.state)
    expected_event_type = "model_call_completed" if record.status is ModelCallStatus.COMPLETED else "model_call_failed"
    if expected_statuses is None or record.status not in expected_statuses or terminal_event["type"] != expected_event_type:
        raise CheckpointError(
            "model journal terminal state does not match its accounting evidence",
            code="checkpoint_status_invalid",
        )
    if terminal_event.get("usage") != record.usage.to_dict():
        raise CheckpointError(
            "model terminal event usage does not match its ledger record",
            code="checkpoint_status_invalid",
        )
    if record.status is not ModelCallStatus.COMPLETED:
        expected_outcome = "ambiguous" if record.status is ModelCallStatus.AMBIGUOUS else "definitive"
        if terminal_event.get("outcome") != expected_outcome or terminal_event.get("error_code") != record.error_code:
            raise CheckpointError(
                "model failed event does not match its ledger record",
                code="checkpoint_status_invalid",
            )


def _require_model_evidence_counts(
    records: list[ModelCallRecord],
    started_events: list[dict[str, Any]],
    terminal_events: list[dict[str, Any]],
    *,
    expected: tuple[int, int, int],
) -> None:
    observed = (len(records), len(started_events), len(terminal_events))
    if observed != expected:
        raise CheckpointError(
            "model journal attempt is missing atomic accounting evidence",
            code="checkpoint_status_invalid",
        )


def _model_journal_identity(journal: OperationJournalEntry) -> tuple[Any, ...]:
    operation = journal.model_operation
    assert operation is not None
    return (
        journal.call_id,
        journal.operation_id,
        journal.attempt,
        operation.value,
        journal.cycle_index,
        journal.backend,
        journal.model,
    )


def _model_record_identity(record: ModelCallRecord) -> tuple[Any, ...]:
    return (
        record.call_id,
        record.operation_id,
        record.attempt,
        record.operation.value,
        record.cycle_index,
        record.backend,
        record.model,
    )


def _model_event_identity(event: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        event.get(field) for field in ("call_id", "operation_id", "attempt", "operation", "cycle_index", "backend", "model")
    )


def _require_model_identity(expected: tuple[Any, ...], observed: tuple[Any, ...]) -> None:
    if observed != expected:
        raise CheckpointError(
            "model journal, event, and ledger identities do not match",
            code="checkpoint_status_invalid",
        )


def check_claim(
    checkpoint: Checkpoint,
    cycle_index: int,
    now_ms: int,
    claim_mode: ClaimMode,
) -> None:
    validate_claim_mode(claim_mode)
    _positive_wire_integer(cycle_index, "claimed cycle_index")
    if checkpoint.terminal_result is not None:
        raise CheckpointError(
            "checkpoint is terminal",
            code="checkpoint_terminal_immutable",
        )
    if checkpoint.status not in {
        AgentStatus.RUNNING,
        AgentStatus.RECONCILIATION_REQUIRED,
    }:
        raise CheckpointError(
            "checkpoint is not claimable",
            code="checkpoint_not_claimable" if checkpoint.status is AgentStatus.DEFERRED else "checkpoint_status_invalid",
        )
    if checkpoint.cycle_index != cycle_index - 1:
        raise CheckpointError(
            "checkpoint cycle conflict",
            code="checkpoint_cycle_conflict",
        )
    if checkpoint.claim_token is not None and (checkpoint.lease_expires_at_ms or 0) > now_ms:
        raise CheckpointError(
            "checkpoint claim is active",
            code="checkpoint_claim_active",
        )


def validate_claim_mode(claim_mode: str) -> ClaimMode:
    if claim_mode not in {"continue", "recovery"}:
        raise CheckpointError(
            "checkpoint claim_mode must be continue or recovery",
            code="checkpoint_claim_mode_invalid",
        )
    return cast(ClaimMode, claim_mode)


def claim_matches(
    current: Checkpoint | None,
    snapshot: Checkpoint,
    claim_token: str,
    expected_revision: int,
) -> bool:
    return bool(
        current is not None
        and current.revision == expected_revision
        and snapshot.revision == expected_revision
        and current.claim_token == claim_token
        and current.claimed_cycle == snapshot.claimed_cycle
        and snapshot.checkpoint_key == current.checkpoint_key
        and checkpoint_definition_matches(current, snapshot)
    )


def checkpoint_definition_matches(current: Checkpoint, snapshot: Checkpoint) -> bool:
    return bool(
        current.schema_version == snapshot.schema_version
        and current.checkpoint_key == snapshot.checkpoint_key
        and current.task_id == snapshot.task_id
        and current.root_run_id == snapshot.root_run_id
        and current.trace_id == snapshot.trace_id
        and current.run_definition_schema == snapshot.run_definition_schema
        and current.run_definition_digest == snapshot.run_definition_digest
        and canonical_json_bytes(current.run_definition) == canonical_json_bytes(snapshot.run_definition)
        and current.resume_attempt == snapshot.resume_attempt
        and current.terminal_acknowledged is snapshot.terminal_acknowledged
    )


def merge_event_outbox(
    authoritative: list[EventOutboxEntry],
    candidate: list[EventOutboxEntry],
) -> list[EventOutboxEntry]:
    merged = deepcopy(authoritative)
    by_id = {entry.event_id: entry for entry in merged}
    for entry in candidate:
        entry.verify_payload()
        current = by_id.get(entry.event_id)
        if current is None:
            current = deepcopy(entry)
            merged.append(current)
            by_id[entry.event_id] = current
        elif current.payload_digest != entry.payload_digest or current.event != entry.event:
            raise CheckpointError(
                f"checkpoint event id {entry.event_id!r} has conflicting payload bytes",
                code="event_identity_conflict",
            )
    return merged


TOOL_CANCELLED_MESSAGE = "Tool execution ended before a definitive receipt; external effect remains unknown."
TOOL_UNKNOWN_OUTCOME_MESSAGE = "The tool outcome is unknown."


def compute_tool_identity_key(
    checkpoint_key: str,
    operation_id: str,
    attempt: int,
    tool_call_id: str,
    request_digest: str,
) -> str:
    return canonical_json_sha256(
        {
            "attempt": attempt,
            "checkpoint_key": checkpoint_key,
            "operation_id": operation_id,
            "request_digest": request_digest,
            "tool_call_id": tool_call_id,
        },
        "tool identity",
    )


def _terminal_abort_reason(terminal: AgentResult) -> str | None:
    error_code = terminal.error_code
    if error_code is None and isinstance(terminal.error, dict):
        error_code = terminal.error.get("code")
    return {
        "cancelled_with_unknown_outcome": "cancelled",
        "operator_abort_with_unknown_outcome": "operator_abort",
        "lease_lost_with_unknown_outcome": "lease_lost",
    }.get(error_code or "")


def _close_unclosed_tools(
    checkpoint: Checkpoint,
) -> list[ResumeObservation]:
    observations: list[ResumeObservation] = []
    for entry in checkpoint.tool_journal:
        if entry.state not in {
            OperationState.PLANNED,
            OperationState.STARTED,
            OperationState.DEFERRED,
            OperationState.AMBIGUOUS,
        }:
            continue
        observation = ResumeObservation(
            operation_id=entry.operation_id,
            operation_kind=entry.kind,
            cycle_index=entry.cycle_index,
            risk="unknown_tool_side_effect",
            idempotency_support=entry.idempotency_support,
        )
        entry.state = OperationState.FAILED
        entry.deferred_handle = None
        entry.result = None
        entry.error = OperationError(
            code="tool_cancelled",
            message=TOOL_CANCELLED_MESSAGE,
            retryable=False,
        )
        entry.identity_key = compute_tool_identity_key(
            checkpoint.checkpoint_key,
            entry.operation_id,
            entry.attempt,
            entry.tool_call_id or "",
            entry.request_digest,
        )
        entry.result_digest = None
        entry.resume_observation = observation
        observations.append(observation)
    observations.sort(key=lambda item: (item.operation_id, item.operation_kind.value, item.cycle_index))
    return observations


def _merge_resume_observations(
    existing: list[ResumeObservation],
    additions: list[ResumeObservation],
) -> list[ResumeObservation]:
    merged: dict[tuple[str, str, int], ResumeObservation] = {}
    for observation in [*existing, *additions]:
        key = (
            observation.operation_id,
            observation.operation_kind.value,
            observation.cycle_index,
        )
        merged[key] = observation
    return [merged[key] for key in sorted(merged)]


def _append_cycle_aborted_event(
    checkpoint: Checkpoint,
    *,
    logical_cycle: int,
    reason: str,
    created_at: float | None = None,
) -> None:
    from vv_agent.events import CycleAbortedEvent

    event = CycleAbortedEvent(
        run_id=checkpoint.root_run_id,
        trace_id=checkpoint.trace_id,
        cycle_index=logical_cycle - 1,
        logical_cycle=logical_cycle,
        reason=reason,
        event_id=f"evt_cycle_aborted_{reason}",
        created_at=created_at,
    ).to_dict()
    existing = next((entry for entry in checkpoint.event_outbox if entry.event_id == event["event_id"]), None)
    if existing is not None:
        existing.verify_payload()
        durable_identity = {key: value for key, value in existing.event.items() if key != "created_at"}
        requested_identity = {key: value for key, value in event.items() if key != "created_at"}
        if durable_identity != requested_identity:
            raise CheckpointError("cycle_aborted event identity conflicts", code="event_identity_conflict")
        return
    entry = EventOutboxEntry.pending(event["event_id"], event)
    terminal_index = next(
        (
            index
            for index, existing_entry in enumerate(checkpoint.event_outbox)
            if existing_entry.event.get("type") in {"run_completed", "run_failed", "run_cancelled"}
            or (existing_entry.event.get("type") == "run_state_changed" and existing_entry.event.get("state") != "running")
        ),
        len(checkpoint.event_outbox),
    )
    checkpoint.event_outbox.insert(terminal_index, entry)


def prepare_tool_receipt(
    current: Checkpoint | None,
    checkpoint: Checkpoint,
    *,
    operation_id: str,
    attempt: int,
    tool_call_id: str,
    request_digest: str,
    result: ToolExecutionResult,
    claim_token: str,
    expected_revision: int,
    claimed_cycle: int,
    created_at: float,
) -> Checkpoint | None:
    from vv_agent.deferred import validate_definitive_result

    validate_definitive_result(result)
    if result.tool_call_id != tool_call_id:
        raise CheckpointError(
            "tool receipt result tool_call_id does not match the journal identity",
            code="tool_receipt_identity_invalid",
        )
    from vv_agent.events import ToolCallCompletedEvent
    from vv_agent.runtime.checkpoint_codec import clone_checkpoint

    identity_key = compute_tool_identity_key(
        checkpoint.checkpoint_key,
        operation_id,
        attempt,
        tool_call_id,
        request_digest,
    )
    result_digest = canonical_json_sha256(result.to_dict(), "tool result")
    existing = (
        next(
            (entry for entry in current.tool_journal if entry.identity_key == identity_key),
            None,
        )
        if current is not None
        else None
    )
    if existing is not None:
        if existing.result_digest == result_digest:
            return current
        raise CheckpointError(
            "tool receipt conflicts with the retained identity",
            code="tool_receipt_conflict",
        )
    if current is None:
        return None
    if current.claim_token is None or not claim_token:
        raise CheckpointError(
            "tool receipt requires an active claim",
            code="checkpoint_claim_required",
        )
    if current.claim_token != claim_token or current.claimed_cycle != claimed_cycle:
        raise CheckpointError(
            "tool receipt claim does not match the checkpoint claim",
            code="checkpoint_claim_conflict",
        )
    if current.revision != expected_revision or checkpoint.revision != expected_revision:
        raise CheckpointError(
            "tool receipt revision does not match the checkpoint revision",
            code="checkpoint_revision_conflict",
        )
    if (
        current.status is not AgentStatus.RUNNING
        or current.terminal_result is not None
        or not checkpoint_definition_matches(current, checkpoint)
    ):
        return None
    entry = next(
        (
            item
            for item in current.tool_journal
            if item.operation_id == operation_id
            and item.attempt == attempt
            and item.tool_call_id == tool_call_id
            and item.request_digest == request_digest
            and item.cycle_index == claimed_cycle
        ),
        None,
    )
    if entry is None or entry.state not in {OperationState.STARTED, OperationState.AMBIGUOUS}:
        return None
    execution_started = entry.state in {OperationState.STARTED, OperationState.AMBIGUOUS}
    snapshot = clone_checkpoint(current)
    target = next(
        item
        for item in snapshot.tool_journal
        if item.cycle_index == entry.cycle_index
        and item.operation_id == entry.operation_id
        and item.attempt == entry.attempt
        and item.tool_call_id == entry.tool_call_id
        and item.request_digest == entry.request_digest
    )
    target.identity_key = identity_key
    target.result_digest = result_digest
    source = next(
        (
            item
            for item in checkpoint.tool_journal
            if item.operation_id == entry.operation_id
            and item.attempt == entry.attempt
            and item.tool_call_id == entry.tool_call_id
            and item.request_digest == entry.request_digest
            and item.cycle_index == entry.cycle_index
        ),
        None,
    )
    if result.error_code == "tool_outcome_unknown":
        observation = (
            ResumeObservation(
                operation_id=entry.operation_id,
                operation_kind=entry.kind,
                cycle_index=entry.cycle_index,
                risk="unknown_tool_side_effect",
                idempotency_support=entry.idempotency_support,
            )
            if source is not None
            and source.kind is entry.kind
            and source.state is OperationState.AMBIGUOUS
            and source.resume_observation is not None
            else None
        )
        if observation is None or source is None or source.resume_observation != observation:
            raise CheckpointError(
                "tool outcome observation does not match the authoritative operation",
                code="checkpoint_journal_integrity_mismatch",
            )
        target.resume_observation = observation
    else:
        target.resume_observation = None
    target.deferred_handle = None
    if result.status_code.value == "SUCCESS":
        target.state = OperationState.SUCCEEDED
        target.result = result.to_dict()
        target.error = None
    else:
        target.state = OperationState.FAILED
        target.result = result.to_dict()
        target.error = operation_error_from_tool_result(result)
    event = ToolCallCompletedEvent(
        run_id=snapshot.root_run_id,
        trace_id=snapshot.trace_id,
        cycle_index=target.cycle_index,
        tool_call_id=target.tool_call_id or tool_call_id,
        tool_name=target.tool_name or "tool",
        operation_id=target.operation_id,
        attempt=target.attempt,
        status=result.status_code.value.lower(),
        directive=result.directive.value,
        error_code=result.error_code,
        execution_started=execution_started,
        duration_ms=None,
        checkpoint_key=snapshot.checkpoint_key,
        event_id=f"evt_receipt_{identity_key}",
        created_at=created_at,
    ).to_dict()
    snapshot.event_outbox = merge_event_outbox(
        snapshot.event_outbox,
        [EventOutboxEntry.pending(event["event_id"], event)],
    )
    snapshot.revision = expected_revision + 1
    validate_checkpoint(snapshot)
    return snapshot


def prepare_deferred_admission(
    current: Checkpoint | None,
    checkpoint: Checkpoint,
    *,
    outcomes: list[Any],
    claim_token: str,
    expected_revision: int,
    claimed_cycle: int,
    created_at: float,
) -> Checkpoint | None:
    """Atomically admit one ordered model-tool batch.

    The store owns the only claim release for a deferred batch.  All
    journal and lifecycle outbox mutations are prepared on a clone and
    validated before replacing the authoritative record.
    """
    from vv_agent.events import ToolCallDeferredEvent
    from vv_agent.runtime.checkpoint_codec import clone_checkpoint

    if (
        current is None
        or current.revision != expected_revision
        or current.claim_token != claim_token
        or current.claimed_cycle != claimed_cycle
        or checkpoint.revision != expected_revision
    ):
        return None
    normalized = _normalize_batch_outcomes(outcomes)
    if not normalized:
        return None
    if any(outcome.kind == "completed" for _call_id, outcome in normalized):
        raise CheckpointError(
            "deferred admission accepts deferred outcomes only",
            code="deferred_admission_completed_outcome_invalid",
        )
    snapshot = clone_checkpoint(current)
    covered: set[tuple[str, int, str | None, str, int]] = set()
    for call_id, outcome in normalized:
        handle = outcome.handle
        assert handle is not None
        entry = next(
            (
                item
                for item in snapshot.tool_journal
                if item.cycle_index == claimed_cycle
                and item.operation_id == handle.operation_id
                and item.attempt == handle.attempt
                and item.request_digest == handle.request_digest
            ),
            None,
        )
        if entry is None or entry.cycle_index != claimed_cycle or entry.state is not OperationState.STARTED:
            return None
        identity = (
            entry.operation_id,
            entry.attempt,
            entry.tool_call_id,
            entry.request_digest,
            entry.cycle_index,
        )
        if identity in covered:
            return None
        covered.add(identity)
        if handle.checkpoint_key != snapshot.checkpoint_key:
            return None
        entry.state = OperationState.DEFERRED
        entry.deferred_handle = handle
        entry.result = None
        entry.error = None
        event = ToolCallDeferredEvent(
            run_id=snapshot.root_run_id,
            trace_id=snapshot.trace_id,
            cycle_index=entry.cycle_index,
            tool_call_id=entry.tool_call_id or call_id,
            tool_name=entry.tool_name or "tool",
            operation_id=entry.operation_id,
            attempt=entry.attempt,
            handle=handle,
            execution_started=True,
            duration_ms=None,
            checkpoint_key=snapshot.checkpoint_key,
            operation_kind="tool",
            event_id=_stable_deferred_event_id(entry, "deferred"),
            created_at=created_at,
        ).to_dict()
        snapshot.event_outbox = merge_event_outbox(snapshot.event_outbox, [EventOutboxEntry.pending(event["event_id"], event)])
    # Admission is the all-or-none boundary for the complete started
    # model-tool batch.  A missing started slot would otherwise leave
    # an unclassified external operation behind a released claim.
    if any(
        entry.cycle_index == claimed_cycle
        and entry.state is OperationState.STARTED
        and (
            entry.operation_id,
            entry.attempt,
            entry.tool_call_id,
            entry.request_digest,
            entry.cycle_index,
        )
        not in covered
        for entry in [*snapshot.model_call_journal, *snapshot.tool_journal]
    ):
        raise CheckpointError(
            "deferred batch must cover every started tool in the claimed cycle",
            code="deferred_batch_incomplete",
        )
    snapshot.status = AgentStatus.DEFERRED
    snapshot.claim_token = None
    snapshot.claimed_cycle = None
    snapshot.lease_expires_at_ms = None
    snapshot.revision = expected_revision + 1
    try:
        validate_checkpoint(snapshot)
    except Exception:
        return None
    return snapshot


def prepare_deferred_resolution(
    checkpoint: Checkpoint | None,
    existing: DeferredResolutionReceipt | None,
    handle: DeferredToolHandle,
    result: ToolExecutionResult,
    *,
    created_at: float,
) -> tuple[Checkpoint | None, DeferredResolveDecision]:
    from vv_agent.events import ToolCallCompletedEvent
    from vv_agent.runtime.checkpoint_codec import clone_checkpoint

    validate_definitive_result(result)
    if existing is not None:
        if existing.handle.key != handle.key or existing.handle_key != handle.key:
            raise ValueError("deferred_receipt_identity_invalid")
        if existing.result.to_dict() != result.to_dict():
            raise DeferredResolutionConflict()
        return None, DeferredResolveDecision.Replayed(existing)
    if checkpoint is None:
        raise DeferredResolutionStale()
    entry = next(
        (
            item
            for item in checkpoint.tool_journal
            if item.operation_id == handle.operation_id
            and item.attempt == handle.attempt
            and item.request_digest == handle.request_digest
        ),
        None,
    )
    if entry is None:
        raise DeferredResolutionStale()
    if entry.state is OperationState.STARTED:
        return None, DeferredResolveDecision.NotAdmitted()
    if entry.state is OperationState.AMBIGUOUS:
        return None, DeferredResolveDecision.ReconciliationRequired()
    if entry.state is not OperationState.DEFERRED or entry.deferred_handle != handle:
        raise DeferredResolutionStale()
    if checkpoint.claim_token is not None:
        raise DeferredCheckpointClaimed()
    if result.tool_call_id != entry.tool_call_id:
        raise DeferredResolutionStale("deferred result tool_call_id does not match handle")
    snapshot = clone_checkpoint(checkpoint)
    target = next(
        item
        for item in snapshot.tool_journal
        if item.cycle_index == entry.cycle_index
        and item.operation_id == entry.operation_id
        and item.attempt == entry.attempt
        and item.tool_call_id == entry.tool_call_id
        and item.request_digest == entry.request_digest
    )
    from vv_agent.checkpoint import canonical_json_sha256

    identity_key = compute_tool_identity_key(
        snapshot.checkpoint_key,
        target.operation_id,
        target.attempt,
        target.tool_call_id or result.tool_call_id,
        target.request_digest,
    )
    target.identity_key = identity_key
    target.result_digest = canonical_json_sha256(result.to_dict(), "deferred result")

    if result.status_code.value == "SUCCESS":
        target.state = OperationState.SUCCEEDED
        target.deferred_handle = None
        target.result = result.to_dict()
        target.error = None
        receipt_status = "succeeded"
    else:
        target.state = OperationState.FAILED
        target.deferred_handle = None
        target.result = result.to_dict()
        target.error = operation_error_from_tool_result(result)
        receipt_status = "failed"
    event = ToolCallCompletedEvent(
        run_id=snapshot.root_run_id,
        trace_id=snapshot.trace_id,
        cycle_index=target.cycle_index,
        tool_call_id=target.tool_call_id or result.tool_call_id,
        tool_name=target.tool_name or "tool",
        operation_id=target.operation_id,
        attempt=target.attempt,
        status=result.status_code.value.lower(),
        directive=result.directive.value,
        error_code=result.error_code,
        execution_started=True,
        duration_ms=None,
        event_id=f"evt_receipt_{identity_key}",
        created_at=created_at,
    ).to_dict()
    snapshot.event_outbox = merge_event_outbox(snapshot.event_outbox, [EventOutboxEntry.pending(event["event_id"], event)])
    remaining = [item for item in snapshot.tool_journal if item.state is OperationState.DEFERRED]
    snapshot.status = AgentStatus.DEFERRED if remaining else AgentStatus.RUNNING
    snapshot.revision = checkpoint.revision + 1
    receipt = DeferredResolutionReceipt(
        handle=handle,
        result=result,
        result_digest=canonical_json_sha256(result.to_dict(), "deferred result"),
        event_id=event["event_id"],
        event_payload_digest=compute_event_payload_digest(event),
        receipt_status=receipt_status,
    )
    validate_checkpoint(snapshot)
    return snapshot, (
        DeferredResolveDecision.AppliedReady(receipt) if not remaining else DeferredResolveDecision.AppliedWaiting(receipt)
    )


def prepare_deferred_acceptance(
    current: Checkpoint | None,
    checkpoint: Checkpoint,
    *,
    decisions: list[Any],
    claim_token: str,
    expected_revision: int,
    claimed_cycle: int,
    created_at: float,
) -> Checkpoint | None:
    from vv_agent.deferred import AcceptDeferredDecision
    from vv_agent.events import RUN_EVENT_VERSION, ToolCallDeferredEvent
    from vv_agent.runtime.checkpoint_codec import clone_checkpoint

    if current is None:
        return None
    parsed = [d if isinstance(d, AcceptDeferredDecision) else AcceptDeferredDecision.from_dict(d) for d in decisions]
    if not parsed:
        return None

    # This is a batch boundary, not a per-operation convenience
    # method.  Reject duplicate handles and any decision set which
    # does not cover the complete current-cycle ambiguity.  The
    # controller performs the same aggregation before calling the
    # store, but keeping the invariant here is essential for direct
    # SQLite/Redis callers and for the all-or-none CAS contract.
    decision_keys = [(item.handle.operation_id, item.handle.attempt, item.handle.request_digest) for item in parsed]
    if len(decision_keys) != len(set(decision_keys)):
        return None

    # A repeated acceptance is an idempotent replay of the durable
    # reconciliation/deferred identities.  It must not require a new
    # claim or revision.  A mixed replay/new batch still needs the
    # active recovery claim and is validated all-or-none below.
    snapshot = clone_checkpoint(current)
    if current.claimed_cycle is not None:
        current_cycle = current.claimed_cycle
    else:
        # Admission leaves the checkpoint's committed cycle index at
        # the prior completed cycle while the deferred barrier owns
        # the just-executed cycle.  Replayed acceptance has no claim
        # from which to recover that cycle, so derive it only from
        # the exact deferred handles supplied by the caller.
        decision_keys_set = set(decision_keys)
        decision_cycles = {
            item.cycle_index
            for item in snapshot.tool_journal
            if item.state is OperationState.DEFERRED
            and (item.operation_id, item.attempt, item.request_digest) in decision_keys_set
        }
        deferred_cycles = {item.cycle_index for item in snapshot.tool_journal if item.state is OperationState.DEFERRED}
        if len(decision_cycles) == 1:
            current_cycle = next(iter(decision_cycles))
        elif snapshot.status is AgentStatus.DEFERRED and len(deferred_cycles) == 1:
            current_cycle = next(iter(deferred_cycles))
        else:
            current_cycle = snapshot.cycle_index
    all_cycle_entries = [
        item for item in [*snapshot.model_call_journal, *snapshot.tool_journal] if item.cycle_index == current_cycle
    ]
    replay_entries = [
        item for item in snapshot.tool_journal if item.cycle_index == current_cycle and item.state is OperationState.DEFERRED
    ]
    replayed = (
        bool(replay_entries)
        and len(parsed) == len(replay_entries)
        and all(item.kind.value == "tool" for item in replay_entries)
        and not any(
            item.state in {OperationState.AMBIGUOUS, OperationState.STARTED, OperationState.PLANNED} for item in all_cycle_entries
        )
    )
    for decision in parsed:
        entry = next(
            (
                item
                for item in snapshot.tool_journal
                if item.cycle_index == current_cycle
                and item.operation_id == decision.handle.operation_id
                and item.attempt == decision.handle.attempt
                and item.request_digest == decision.handle.request_digest
            ),
            None,
        )
        if entry is None or entry.state is not OperationState.DEFERRED or entry.deferred_handle != decision.handle:
            replayed = False
            break
    if replayed:
        return current
    if (
        current.revision != expected_revision
        or checkpoint.revision != expected_revision
        or not checkpoint_definition_matches(current, checkpoint)
    ):
        return None
    if current.resume_attempt <= 1 or current.claim_token != claim_token or current.claimed_cycle != claimed_cycle:
        return None

    batch_entries = [item for item in [*snapshot.model_call_journal, *snapshot.tool_journal] if item.cycle_index == current_cycle]
    # The store, not only the recovery controller, owns the complete
    # batch invariant. Any model entry, STARTED entry, or omitted
    # current-cycle operation makes partial acceptance unsafe.
    if not batch_entries or any(item.state in {OperationState.STARTED, OperationState.PLANNED} for item in batch_entries):
        return None
    ambiguous_entries = [item for item in batch_entries if item.state is OperationState.AMBIGUOUS]
    if not ambiguous_entries or any(item.kind.value != "tool" for item in ambiguous_entries):
        return None
    # A recovery acceptance may include already-adopted entries when
    # a caller retries after a partial transport failure, but every
    # current-cycle entry must be represented exactly once and the
    # batch must contain tools only.  Any omission or extra handle
    # therefore leaves the checkpoint untouched.
    current_entries = list(ambiguous_entries)
    if not ambiguous_entries or len(parsed) != len(current_entries):
        return None
    current_keys = {(item.operation_id, item.attempt, item.request_digest) for item in current_entries}
    if set(decision_keys) != current_keys:
        return None
    # Provider response order is not the model-call order.  Apply
    # accepted entries in journal order so event/outbox ordering is
    # deterministic and independent of reconciliation delivery order.
    decisions_by_key = dict(zip(decision_keys, parsed, strict=True))
    ordered_decisions = [decisions_by_key[(item.operation_id, item.attempt, item.request_digest)] for item in current_entries]
    for decision in ordered_decisions:
        entry = next(
            (
                item
                for item in snapshot.tool_journal
                if item.cycle_index == current_cycle
                and item.operation_id == decision.handle.operation_id
                and item.attempt == decision.handle.attempt
                and item.request_digest == decision.handle.request_digest
            ),
            None,
        )
        if entry is None:
            return None
        if entry.state is OperationState.DEFERRED and entry.deferred_handle == decision.handle:
            # Already accepted item in a mixed retry; preserve its
            # existing audit and deferred event identity.
            continue
        if (
            entry.kind.value != "tool"
            or entry.state is not OperationState.AMBIGUOUS
            or decision.handle.checkpoint_key != snapshot.checkpoint_key
            or decision.handle.operation_id != entry.operation_id
            or decision.handle.attempt != entry.attempt
            or decision.handle.request_digest != entry.request_digest
        ):
            return None
        entry.state = OperationState.DEFERRED
        entry.deferred_handle = decision.handle
        entry.result = None
        entry.error = None
        entry.resume_observation = None
        audit = {
            "version": RUN_EVENT_VERSION,
            "type": "reconciliation_resolved",
            "event_id": _stable_deferred_event_id(entry, "reconciliation"),
            "run_id": snapshot.root_run_id,
            "trace_id": snapshot.trace_id,
            "created_at": 0.0,
            "cycle_index": entry.cycle_index,
            "checkpoint_key": snapshot.checkpoint_key,
            "operation_id": entry.operation_id,
            "operation_kind": "tool",
            "decision": "accept_deferred",
            "claim_mode": "recovery",
        }
        snapshot.event_outbox = merge_event_outbox(snapshot.event_outbox, [EventOutboxEntry.pending(audit["event_id"], audit)])
        deferred = ToolCallDeferredEvent(
            run_id=snapshot.root_run_id,
            trace_id=snapshot.trace_id,
            cycle_index=entry.cycle_index,
            tool_call_id=entry.tool_call_id or "tool_call",
            tool_name=entry.tool_name or "tool",
            operation_id=entry.operation_id,
            attempt=entry.attempt,
            handle=decision.handle,
            execution_started=True,
            duration_ms=None,
            checkpoint_key=snapshot.checkpoint_key,
            operation_kind="tool",
            event_id=_stable_deferred_event_id(entry, "deferred"),
            created_at=created_at,
        ).to_dict()
        snapshot.event_outbox = merge_event_outbox(
            snapshot.event_outbox, [EventOutboxEntry.pending(deferred["event_id"], deferred)]
        )
    snapshot.status = AgentStatus.DEFERRED
    snapshot.claim_token = None
    snapshot.claimed_cycle = None
    snapshot.lease_expires_at_ms = None
    snapshot.revision = expected_revision + 1
    validate_checkpoint(snapshot)
    return snapshot


def _normalize_batch_outcomes(outcomes: list[Any]) -> list[tuple[str, ToolCallOutcome]]:
    normalized: list[tuple[str, ToolCallOutcome]] = []
    for item in outcomes:
        call_id: str | None = None
        outcome: Any = item
        if isinstance(item, tuple) and len(item) == 2:
            call_id, outcome = item
            # The runner keeps the original ToolCall beside its outcome so
            # admission can verify the exact invocation.  Store APIs also
            # accept a plain call-id for recovery/tests; normalize both
            # shapes without stringifying a ToolCall's repr.
            if not isinstance(call_id, str):
                call_id = getattr(call_id, "id", None)
            if call_id is not None:
                call_id = str(call_id)
        if isinstance(outcome, ToolExecutionResult):
            call_id = call_id or outcome.tool_call_id
            outcome = ToolCallOutcome.Completed(outcome)
        if not isinstance(outcome, ToolCallOutcome):
            raise ValueError("deferred batch outcomes must be ToolCallOutcome values")
        if outcome.kind == "completed":
            result = outcome.result
            if not isinstance(result, ToolExecutionResult):
                raise ValueError("deferred batch completed outcome has no tool result")
            call_id = call_id or result.tool_call_id
        else:
            call_id = call_id or (outcome.handle.operation_id if outcome.handle else "")
        if not call_id:
            raise ValueError("deferred batch outcome is missing tool call identity")
        normalized.append((call_id, outcome))
    return normalized


def _stable_deferred_event_id(entry: Any, suffix: str) -> str:
    # Reconciliation and admission replays retain the model tool-call identity.
    call_id = entry.tool_call_id or entry.operation_id
    return f"evt_deferred_{call_id}_{suffix}"


def prepare_claimed_terminal(
    current: Checkpoint,
    checkpoint: Checkpoint,
    *,
    claim_token: str,
    expected_revision: int,
    created_at: float | None = None,
) -> Checkpoint | None:
    if (
        current.revision != expected_revision
        or checkpoint.revision != expected_revision
        or current.claim_token != claim_token
        or checkpoint.claim_token != claim_token
        or current.claimed_cycle != checkpoint.claimed_cycle
        or current.terminal_result is not None
        or checkpoint.terminal_result is None
        or not checkpoint_definition_matches(current, checkpoint)
    ):
        return None
    checkpoint.cancel_requested = current.cancel_requested
    checkpoint.event_outbox = merge_event_outbox(current.event_outbox, checkpoint.event_outbox)
    checkpoint.event_cursor = deepcopy(current.event_cursor)
    validate_model_journal_accounting(checkpoint)
    terminal_result = deepcopy(checkpoint.terminal_result)
    assert terminal_result is not None
    reason = _terminal_abort_reason(terminal_result)
    logical_cycle = checkpoint.cycle_index + 1
    existing_observations = list(terminal_result.resume_observations)
    observations: list[ResumeObservation] = []
    if reason is not None:
        observations = _close_unclosed_tools(checkpoint)
        terminal_result.resume_observations = _merge_resume_observations(existing_observations, observations)
        has_unclosed_cycle = any(
            entry.state in {OperationState.PLANNED, OperationState.STARTED, OperationState.DEFERRED, OperationState.AMBIGUOUS}
            for entry in [*checkpoint.model_call_journal, *checkpoint.tool_journal]
        )
        if terminal_result.resume_observations or has_unclosed_cycle:
            _append_cycle_aborted_event(checkpoint, logical_cycle=logical_cycle, reason=reason, created_at=created_at)
    checkpoint.terminal_result = terminal_result
    preserve_model_journal = reason is not None
    preserve_journals = bool(reason is not None and observations)
    terminal = replace(
        deepcopy(checkpoint),
        revision=expected_revision + 1,
        claim_token=None,
        claimed_cycle=None,
        lease_expires_at_ms=None,
        model_call_journal=(
            [
                deepcopy(entry)
                for entry in checkpoint.model_call_journal
                if entry.state
                not in {
                    OperationState.PLANNED,
                    OperationState.STARTED,
                    OperationState.DEFERRED,
                    OperationState.AMBIGUOUS,
                }
            ]
            if preserve_model_journal
            else []
        ),
        tool_journal=(deepcopy(checkpoint.tool_journal) if preserve_journals else []),
    )
    validate_checkpoint(terminal)
    return terminal


def prepare_unclaimed_terminal(checkpoint: Checkpoint, *, created_at: float | None = None) -> Checkpoint:
    if checkpoint.terminal_result is None or checkpoint.claim_token is not None:
        raise ValueError("unclaimed terminal preparation requires a terminal result and no claim")
    terminal = deepcopy(checkpoint)
    result = terminal.terminal_result
    assert result is not None
    reason = _terminal_abort_reason(result)
    logical_cycle = terminal.cycle_index + 1
    existing_observations = list(result.resume_observations)
    observations: list[ResumeObservation] = []
    if reason is not None:
        observations = _close_unclosed_tools(terminal)
        result.resume_observations = _merge_resume_observations(existing_observations, observations)
        has_unclosed_cycle = any(
            entry.state in {OperationState.PLANNED, OperationState.STARTED, OperationState.DEFERRED, OperationState.AMBIGUOUS}
            for entry in [*terminal.model_call_journal, *terminal.tool_journal]
        )
        if result.resume_observations or has_unclosed_cycle:
            _append_cycle_aborted_event(terminal, logical_cycle=logical_cycle, reason=reason, created_at=created_at)
    terminal.model_call_journal = (
        [
            entry
            for entry in terminal.model_call_journal
            if entry.state
            not in {
                OperationState.PLANNED,
                OperationState.STARTED,
                OperationState.DEFERRED,
                OperationState.AMBIGUOUS,
            }
        ]
        if reason is not None
        else []
    )
    terminal.tool_journal = deepcopy(terminal.tool_journal) if observations else []
    validate_checkpoint(terminal)
    return terminal


def prepare_event_delivery(
    current: Checkpoint,
    *,
    event_id: str,
    payload_digest: str,
    cursor: EventCursor,
    expected_revision: int,
    claim_token: str | None,
) -> Checkpoint | None:
    if current.revision != expected_revision or current.claim_token != claim_token:
        return None
    if cursor.last_event_id != event_id:
        raise CheckpointError(
            "event cursor last_event_id does not match the delivered event",
            code="event_cursor_invalid",
        )
    matches = [entry for entry in current.event_outbox if entry.event_id == event_id]
    if len(matches) != 1:
        raise CheckpointError(
            "checkpoint outbox does not contain exactly one matching event",
            code="event_identity_conflict",
        )
    entry = matches[0]
    entry.verify_payload()
    if entry.payload_digest != payload_digest:
        raise CheckpointError(
            "event payload digest conflicts with the durable outbox entry",
            code="event_identity_conflict",
        )
    if entry.state != "pending":
        return None
    delivered = deepcopy(current)
    delivered_entry = next(item for item in delivered.event_outbox if item.event_id == event_id)
    delivered_entry.state = "delivered"
    delivered_entry.cursor = deepcopy(cursor)
    delivered.event_cursor = deepcopy(cursor)
    delivered.revision = expected_revision + 1
    validate_checkpoint(delivered)
    return delivered


def _wire_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_WIRE_INTEGER:
        raise ValueError(f"{field_name} must be between 0 and {MAX_WIRE_INTEGER}")
    return value


def _positive_wire_integer(value: Any, field_name: str) -> int:
    result = _wire_integer(value, field_name)
    if result == 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _required_string(payload: dict[str, Any], field_name: str) -> str:
    if field_name not in payload:
        raise ValueError(f"{field_name} is required")
    value = payload[field_name]
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _required_integer(payload: dict[str, Any], field_name: str) -> int:
    if field_name not in payload:
        raise ValueError(f"{field_name} is required")
    value = payload[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _required_boolean(
    payload: dict[str, Any],
    field_name: str,
    *,
    default: bool | None = None,
) -> bool:
    if field_name in payload:
        value = payload[field_name]
    elif default is not None:
        value = default
    else:
        raise ValueError(f"{field_name} is required")
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _required_object(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    if field_name not in payload:
        raise ValueError(f"{field_name} is required")
    value = payload[field_name]
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value
