"""Closed, framework-owned durable deferred-tool wires.

The provider which accepts an external operation never needs to be represented
by these values.  A deferred handle is only the framework identity required to
deliver the eventual definitive tool result back to the owning checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from vv_agent.canonical_json import canonical_json_sha256, validate_sha256
from vv_agent.tools import outcomes
from vv_agent.types import ToolExecutionResult, ToolResultStatus

DEFERRED_RESOLVE_DECISION_SCHEMA = "vv-agent.deferred-resolve-decision.v1"
RECONCILIATION_DECISION_SCHEMA = "vv-agent.reconciliation-decision.v1"


class DeferredResolutionConflict(outcomes.DeferredResolutionError):
    def __init__(self, message: str = "deferred resolution conflicts with the retained receipt") -> None:
        super().__init__(message, code="deferred_resolution_conflict")


class DeferredResolutionStale(outcomes.DeferredResolutionError):
    def __init__(self, message: str = "deferred handle is stale") -> None:
        super().__init__(message, code="deferred_resolution_stale")


class DeferredCheckpointClaimed(outcomes.DeferredResolutionError):
    """A resolution cannot mutate a checkpoint while a worker owns its claim."""

    def __init__(self, message: str = "deferred checkpoint is currently claimed") -> None:
        super().__init__(message, code="deferred_checkpoint_claimed")


@dataclass(frozen=True, slots=True)
class DeferredResolutionReceipt:
    """Durable tombstone retained independently from checkpoint payload."""

    handle: outcomes.DeferredToolHandle
    result: Any
    result_digest: str
    event_id: str
    event_payload_digest: str
    receipt_status: str
    handle_key: str | None = None

    def __post_init__(self) -> None:
        from vv_agent.runtime.state import compute_tool_identity_key

        if not isinstance(self.handle, outcomes.DeferredToolHandle):
            raise ValueError("deferred_receipt_identity_invalid: handle is invalid")
        if not isinstance(self.result, ToolExecutionResult) or self.result.status_code not in {
            ToolResultStatus.SUCCESS,
            ToolResultStatus.ERROR,
        }:
            raise ValueError("deferred_resolution_result_invalid")
        if self.result.tool_call_id.strip() == "":
            raise ValueError("deferred_resolution_result_invalid")
        expected_event_id = "evt_receipt_" + compute_tool_identity_key(
            self.handle.checkpoint_key,
            self.handle.operation_id,
            self.handle.attempt,
            self.result.tool_call_id,
            self.handle.request_digest,
        )
        if self.event_id != expected_event_id:
            raise ValueError("deferred_receipt_identity_invalid")
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

        return cls(
            handle=outcomes.DeferredToolHandle.from_dict(payload["handle"]),
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

    handle: outcomes.DeferredToolHandle
    schema_version: str = RECONCILIATION_DECISION_SCHEMA
    kind: str = "accept_deferred"

    def __post_init__(self) -> None:
        if self.schema_version != RECONCILIATION_DECISION_SCHEMA or self.kind != "accept_deferred":
            raise ValueError("reconciliation_decision_invalid")
        if not isinstance(self.handle, outcomes.DeferredToolHandle):
            raise ValueError("reconciliation_decision_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "kind": self.kind, "handle": self.handle.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AcceptDeferredDecision:
        if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "kind", "handle"}:
            raise ValueError("reconciliation_decision_invalid")
        return cls(
            handle=outcomes.DeferredToolHandle.from_dict(payload["handle"]),
            schema_version=payload["schema_version"],
            kind=payload["kind"],
        )


__all__ = [
    "DEFERRED_RESOLVE_DECISION_SCHEMA",
    "RECONCILIATION_DECISION_SCHEMA",
    "AcceptDeferredDecision",
    "DeferredCheckpointClaimed",
    "DeferredResolutionConflict",
    "DeferredResolutionReceipt",
    "DeferredResolutionStale",
    "DeferredResolveDecision",
]
