"""Closed, framework-owned durable deferred-tool wires.

The provider which accepts an external operation never needs to be represented
by these values.  A deferred handle is only the framework identity required to
deliver the eventual definitive tool result back to the owning checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from vv_agent.checkpoint import MAX_WIRE_INTEGER, canonical_json_sha256, validate_sha256
from vv_agent.types import ToolExecutionResult, ToolResultStatus

DEFERRED_HANDLE_SCHEMA = "vv-agent.deferred-tool-handle.v2"
TOOL_CALL_OUTCOME_SCHEMA = "vv-agent.tool-call-outcome.v2"
DEFERRED_RESOLVE_DECISION_SCHEMA = "vv-agent.deferred-resolve-decision.v1"
RECONCILIATION_DECISION_SCHEMA = "vv-agent.reconciliation-decision.v1"


def _is_ambiguous_tool_error(result: Any) -> bool:
    """Return whether a tool error lacks an adapter-proven definitive outcome."""
    return (
        isinstance(result, ToolExecutionResult)
        and result.status_code is ToolResultStatus.ERROR
        and result.error_code
        in {
            "tool_timeout",
            "tool_cancelled",
            "tool_connection_lost",
            "tool_execution_failed",
            "tool_orchestrator_error",
        }
        and result.metadata.get("definitive_outcome") is not True
    )


class DeferredWireError(ValueError):
    """Base error carrying the stable contract error code."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class DeferredHandleError(DeferredWireError):
    pass


class DeferredResolutionError(DeferredWireError):
    pass


class DeferredResolutionConflict(DeferredResolutionError):
    def __init__(self, message: str = "deferred resolution conflicts with the retained receipt") -> None:
        super().__init__(message, code="deferred_resolution_conflict")


class DeferredResolutionStale(DeferredResolutionError):
    def __init__(self, message: str = "deferred handle is stale") -> None:
        super().__init__(message, code="deferred_resolution_stale")


class DeferredCheckpointClaimed(DeferredResolutionError):
    """A resolution cannot mutate a checkpoint while a worker owns its claim."""

    def __init__(self, message: str = "deferred checkpoint is currently claimed") -> None:
        super().__init__(message, code="deferred_checkpoint_claimed")


class DeferredResolutionResultInvalid(DeferredResolutionError):
    def __init__(
        self,
        message: str = "deferred resolution result is not definitive",
        *,
        code: str = "deferred_resolution_result_invalid",
    ) -> None:
        super().__init__(message, code=code)


def _non_empty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeferredHandleError(f"{field} must be a non-empty string", code="deferred_handle_invalid")
    return value


@dataclass(frozen=True, slots=True)
class DeferredToolHandle:
    """Exact identity of one admitted deferred tool invocation."""

    checkpoint_key: str
    operation_id: str
    attempt: int
    request_digest: str
    schema_version: str = DEFERRED_HANDLE_SCHEMA

    SCHEMA_VERSION: ClassVar[str] = DEFERRED_HANDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DEFERRED_HANDLE_SCHEMA:
            raise DeferredHandleError(
                f"unsupported deferred handle schema: {self.schema_version!r}",
                code="deferred_handle_schema_unsupported",
            )
        _non_empty(self.checkpoint_key, "checkpoint_key")
        _non_empty(self.operation_id, "operation_id")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or not 1 <= self.attempt <= MAX_WIRE_INTEGER:
            raise DeferredHandleError("attempt must be a positive JSON-safe integer", code="deferred_handle_invalid")
        try:
            validate_sha256(self.request_digest, "request_digest")
        except ValueError as exc:
            raise DeferredHandleError(
                "request_digest must be a lowercase SHA-256 digest",
                code="deferred_handle_invalid",
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_key": self.checkpoint_key,
            "operation_id": self.operation_id,
            "attempt": self.attempt,
            "request_digest": self.request_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeferredToolHandle:
        if not isinstance(payload, Mapping):
            raise DeferredHandleError("deferred handle must be an object", code="deferred_handle_invalid")
        expected = {"schema_version", "checkpoint_key", "operation_id", "attempt", "request_digest"}
        if "schema_version" not in payload or payload.get("schema_version") != DEFERRED_HANDLE_SCHEMA:
            raise DeferredHandleError(
                "deferred handle schema discriminator is unsupported",
                code="deferred_handle_schema_unsupported",
            )
        unknown = set(payload) - expected
        missing = expected - set(payload)
        if unknown:
            raise DeferredHandleError(
                f"deferred handle contains unknown field(s): {sorted(unknown)}",
                code="deferred_handle_unknown_field",
            )
        if missing:
            raise DeferredHandleError(
                f"deferred handle is missing field(s): {sorted(missing)}",
                code="deferred_handle_invalid",
            )
        try:
            return cls(
                checkpoint_key=payload["checkpoint_key"],
                operation_id=payload["operation_id"],
                attempt=payload["attempt"],
                request_digest=payload["request_digest"],
                schema_version=payload["schema_version"],
            )
        except DeferredWireError:
            raise
        except (TypeError, ValueError) as exc:
            raise DeferredHandleError("deferred handle fields are invalid", code="deferred_handle_invalid") from exc

    @property
    def key(self) -> str:
        return canonical_json_sha256(self.to_dict(), "deferred handle")

    def __hash__(self) -> int:
        return hash(self.key)


@dataclass(frozen=True, slots=True)
class ToolCallOutcome:
    """Closed two-variant tool outcome.

    ``ToolCallOutcome.Completed(result)`` and
    ``ToolCallOutcome.Deferred(handle)`` are the intentionally compact Python
    spelling of the language-neutral variants.
    """

    kind: str
    result: ToolExecutionResult | None = None
    handle: DeferredToolHandle | None = None
    schema_version: str = TOOL_CALL_OUTCOME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TOOL_CALL_OUTCOME_SCHEMA:
            raise ValueError("tool_call_outcome_invalid: unsupported schema_version")
        if self.kind == "completed":
            if self.result is None or self.handle is not None:
                raise ValueError("tool_call_outcome_invalid: completed requires only result")
            from vv_agent.types import ToolExecutionResult, ToolResultStatus

            if not isinstance(self.result, ToolExecutionResult) or self.result.status_code not in {
                ToolResultStatus.SUCCESS,
                ToolResultStatus.ERROR,
                ToolResultStatus.WAIT_RESPONSE,
                ToolResultStatus.RUNNING,
                ToolResultStatus.PENDING_COMPRESS,
            }:
                raise ValueError("tool_call_outcome_invalid: completed result is invalid")
        elif self.kind == "deferred":
            if self.result is not None or not isinstance(self.handle, DeferredToolHandle):
                raise ValueError("tool_call_outcome_invalid: deferred requires only handle")
        else:
            raise ValueError("tool_call_outcome_invalid: unknown outcome kind")

    @classmethod
    def Completed(cls, result: ToolExecutionResult) -> ToolCallOutcome:
        return cls(kind="completed", result=result)

    @classmethod
    def completed(cls, result: ToolExecutionResult) -> ToolCallOutcome:
        return cls.Completed(result)

    @classmethod
    def Deferred(cls, handle: DeferredToolHandle) -> ToolCallOutcome:
        return cls(kind="deferred", handle=handle)

    @classmethod
    def deferred(cls, handle: DeferredToolHandle) -> ToolCallOutcome:
        return cls.Deferred(handle)

    @property
    def is_deferred(self) -> bool:
        return self.kind == "deferred"

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "completed":
            result = self.result
            assert result is not None
            return {"schema_version": TOOL_CALL_OUTCOME_SCHEMA, "kind": "completed", "result": result.to_dict()}
        assert self.handle is not None
        return {"schema_version": TOOL_CALL_OUTCOME_SCHEMA, "kind": "deferred", "handle": self.handle.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ToolCallOutcome:
        if not isinstance(payload, Mapping):
            raise ValueError("tool_call_outcome_invalid: payload must be an object")
        if payload.get("schema_version") != TOOL_CALL_OUTCOME_SCHEMA:
            raise ValueError("tool_call_outcome_invalid: unsupported schema_version")
        kind = payload.get("kind")
        if kind == "completed":
            if set(payload) != {"schema_version", "kind", "result"}:
                raise ValueError("tool_call_outcome_invalid: completed fields are not closed")
            from vv_agent.types import ToolExecutionResult

            return cls.Completed(ToolExecutionResult.from_dict(payload["result"]))
        if kind == "deferred":
            if set(payload) != {"schema_version", "kind", "handle"}:
                raise ValueError("tool_call_outcome_invalid: deferred fields are not closed")
            return cls.Deferred(DeferredToolHandle.from_dict(payload["handle"]))
        raise ValueError("tool_call_outcome_invalid: unknown outcome kind")


@dataclass(frozen=True, slots=True)
class DeferredResolutionReceipt:
    """Durable tombstone retained independently from checkpoint payload."""

    handle: DeferredToolHandle
    result: Any
    result_digest: str
    event_id: str
    event_payload_digest: str
    receipt_status: str
    handle_key: str | None = None

    def __post_init__(self) -> None:
        from vv_agent.types import ToolExecutionResult, ToolResultStatus

        if not isinstance(self.handle, DeferredToolHandle):
            raise ValueError("deferred_receipt_identity_invalid: handle is invalid")
        if not isinstance(self.result, ToolExecutionResult) or self.result.status_code not in {
            ToolResultStatus.SUCCESS,
            ToolResultStatus.ERROR,
        }:
            raise ValueError("deferred_resolution_result_invalid")
        if self.result.tool_call_id.strip() == "":
            raise ValueError("deferred_resolution_result_invalid")
        expected_digest = canonical_json_sha256(self.result.to_dict(), "deferred result")
        if self.result_digest != expected_digest:
            raise ValueError("deferred_receipt_result_digest_invalid")
        for value, field in ((self.event_id, "event_id"), (self.event_payload_digest, "event_payload_digest")):
            if not isinstance(value, str) or not value:
                raise ValueError(f"deferred_receipt_{field}_invalid")
        validate_sha256(self.event_payload_digest, "event_payload_digest")
        if self.receipt_status not in {"succeeded", "failed"}:
            raise ValueError("deferred_receipt_status_invalid")
        expected_status = "succeeded" if self.result.status_code is ToolResultStatus.SUCCESS else "failed"
        if self.receipt_status != expected_status:
            raise ValueError("deferred_receipt_status_invalid")
        if self.handle_key is not None and self.handle_key != self.handle.key:
            raise ValueError("deferred_receipt_identity_invalid")
        object.__setattr__(self, "handle_key", self.handle.key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle_key": self.handle.key,
            "handle": self.handle.to_dict(),
            "result": self.result.to_dict(),
            "result_digest": self.result_digest,
            "event_id": self.event_id,
            "event_payload_digest": self.event_payload_digest,
            "receipt_status": self.receipt_status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeferredResolutionReceipt:
        if not isinstance(payload, Mapping):
            raise ValueError("deferred_receipt_invalid")
        required = {"handle_key", "handle", "result", "result_digest", "event_id", "event_payload_digest", "receipt_status"}
        if set(payload) != required:
            unknown = sorted(set(payload) - required)
            raise ValueError("deferred_receipt_unknown_field" if unknown else "deferred_receipt_invalid")
        from vv_agent.types import ToolExecutionResult

        return cls(
            handle=DeferredToolHandle.from_dict(payload["handle"]),
            result=ToolExecutionResult.from_dict(payload["result"]),
            result_digest=payload["result_digest"],
            event_id=payload["event_id"],
            event_payload_digest=payload["event_payload_digest"],
            receipt_status=payload["receipt_status"],
            handle_key=payload["handle_key"],
        )


@dataclass(frozen=True, slots=True)
class DeferredResolveDecision:
    kind: str
    receipt: DeferredResolutionReceipt | None = None
    retryable_error: str | None = None
    schema_version: str = DEFERRED_RESOLVE_DECISION_SCHEMA

    _KINDS: ClassVar[frozenset[str]] = frozenset(
        {"applied_ready", "applied_waiting", "replayed", "not_admitted", "reconciliation_required"}
    )

    def __post_init__(self) -> None:
        if self.schema_version != DEFERRED_RESOLVE_DECISION_SCHEMA or self.kind not in self._KINDS:
            raise ValueError("deferred_resolve_decision_invalid")
        if self.kind in {"applied_ready", "applied_waiting", "replayed"}:
            if not isinstance(self.receipt, DeferredResolutionReceipt) or self.retryable_error is not None:
                raise ValueError("deferred_resolve_decision_invalid")
        elif self.kind == "not_admitted":
            if self.receipt is not None or self.retryable_error != "deferred_resolution_not_admitted":
                raise ValueError("deferred_resolve_decision_invalid")
        elif self.receipt is not None or self.retryable_error is not None:
            raise ValueError("deferred_resolve_decision_invalid")

    @classmethod
    def AppliedReady(cls, receipt: DeferredResolutionReceipt) -> DeferredResolveDecision:
        return cls("applied_ready", receipt=receipt)

    @classmethod
    def AppliedWaiting(cls, receipt: DeferredResolutionReceipt) -> DeferredResolveDecision:
        return cls("applied_waiting", receipt=receipt)

    @classmethod
    def Replayed(cls, receipt: DeferredResolutionReceipt) -> DeferredResolveDecision:
        return cls("replayed", receipt=receipt)

    @classmethod
    def NotAdmitted(cls) -> DeferredResolveDecision:
        return cls("not_admitted", retryable_error="deferred_resolution_not_admitted")

    @classmethod
    def ReconciliationRequired(cls) -> DeferredResolveDecision:
        return cls("reconciliation_required")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"schema_version": self.schema_version, "kind": self.kind}
        if self.receipt is not None:
            payload["receipt"] = self.receipt.to_dict()
        if self.retryable_error is not None:
            payload["retryable_error"] = self.retryable_error
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeferredResolveDecision:
        if not isinstance(payload, Mapping) or payload.get("schema_version") != DEFERRED_RESOLVE_DECISION_SCHEMA:
            raise ValueError("deferred_resolve_decision_invalid")
        kind = payload.get("kind")
        if kind in {"applied_ready", "applied_waiting", "replayed"}:
            if set(payload) != {"schema_version", "kind", "receipt"}:
                raise ValueError("deferred_resolve_decision_invalid")
            receipt = DeferredResolutionReceipt.from_dict(payload["receipt"])
            return cls(kind, receipt=receipt)
        if kind == "not_admitted":
            if set(payload) != {"schema_version", "kind", "retryable_error"}:
                raise ValueError("deferred_resolve_decision_invalid")
            return (
                cls.NotAdmitted()
                if payload.get("retryable_error") == "deferred_resolution_not_admitted"
                else cls(kind, retryable_error=payload.get("retryable_error"))
            )
        if kind == "reconciliation_required":
            if set(payload) != {"schema_version", "kind"}:
                raise ValueError("deferred_resolve_decision_invalid")
            return cls.ReconciliationRequired()
        raise ValueError("deferred_resolve_decision_invalid")


@dataclass(frozen=True, slots=True)
class AcceptDeferredDecision:
    """Trusted, out-of-band authority evidence for recovery adoption."""

    handle: DeferredToolHandle
    schema_version: str = RECONCILIATION_DECISION_SCHEMA
    kind: str = "accept_deferred"

    def __post_init__(self) -> None:
        if self.schema_version != RECONCILIATION_DECISION_SCHEMA or self.kind != "accept_deferred":
            raise ValueError("reconciliation_decision_invalid")
        if not isinstance(self.handle, DeferredToolHandle):
            raise ValueError("reconciliation_decision_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "kind": self.kind, "handle": self.handle.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AcceptDeferredDecision:
        if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "kind", "handle"}:
            raise ValueError("reconciliation_decision_invalid")
        return cls(
            handle=DeferredToolHandle.from_dict(payload["handle"]), schema_version=payload["schema_version"], kind=payload["kind"]
        )


def validate_definitive_result(result: Any) -> None:
    if not isinstance(result, ToolExecutionResult) or result.status_code not in {
        ToolResultStatus.SUCCESS,
        ToolResultStatus.ERROR,
    }:
        raise DeferredResolutionResultInvalid()
    if result.status_code is ToolResultStatus.SUCCESS and result.error_code is not None:
        raise DeferredResolutionResultInvalid("tool_result_invalid", code="tool_result_invalid")
    if result.status_code is ToolResultStatus.ERROR and _is_ambiguous_tool_error(result):
        raise DeferredResolutionResultInvalid()


__all__ = [
    "DEFERRED_HANDLE_SCHEMA",
    "DEFERRED_RESOLVE_DECISION_SCHEMA",
    "RECONCILIATION_DECISION_SCHEMA",
    "TOOL_CALL_OUTCOME_SCHEMA",
    "AcceptDeferredDecision",
    "DeferredCheckpointClaimed",
    "DeferredHandleError",
    "DeferredResolutionConflict",
    "DeferredResolutionError",
    "DeferredResolutionReceipt",
    "DeferredResolutionResultInvalid",
    "DeferredResolutionStale",
    "DeferredResolveDecision",
    "DeferredToolHandle",
    "DeferredWireError",
    "ToolCallOutcome",
    "validate_definitive_result",
]
