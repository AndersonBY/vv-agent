"""Durable at-least-once enqueue receipts for distributed cycle delivery.

The controller command wake outbox is intentionally scoped to the closed
``ControllerCommand`` union.  Cycle delivery has a different identity and is
kept in this task-neutral storage protocol instead of manufacturing a
controller command.  A successful broker call is at-least-once: the worker's
checkpoint claim/CAS remains the exactly-once state-transition boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, runtime_checkable

from vv_agent.checkpoint import CheckpointError, canonical_json_sha256

DISPATCH_OUTBOX_SCHEMA = "vv-agent.distributed-dispatch.v1"
DispatchState = Literal["pending", "claimed", "delivered", "ambiguous"]
_MAX_WIRE_INTEGER = (1 << 53) - 1
_MAX_ID_BYTES = 512
_MAX_ERROR_BYTES = 65_536
_DISPATCH_FIELDS = {
    "schema_version",
    "dispatch_id",
    "checkpoint_key",
    "cycle_index",
    "envelope",
    "envelope_digest",
    "state",
    "attempt",
    "claim_token",
    "lease_expires_at_ms",
    "delivered_at_ms",
    "last_error",
}


def _error(message: str, code: str) -> CheckpointError:
    return CheckpointError(message, code=code)


def _text(value: Any, field_name: str, *, max_bytes: int = _MAX_ID_BYTES) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > max_bytes:
        raise _error(f"dispatch outbox {field_name} is invalid", "dispatch_outbox_conflict")
    return value


def _integer(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > _MAX_WIRE_INTEGER:
        raise _error(f"dispatch outbox {field_name} is invalid", "dispatch_outbox_conflict")
    return value


def _digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise _error(f"dispatch outbox {field_name} is invalid", "dispatch_outbox_conflict")
    try:
        int(value, 16)
    except ValueError as exc:
        raise _error(f"dispatch outbox {field_name} is invalid", "dispatch_outbox_conflict") from exc
    return value


def _validate_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(envelope, Mapping):
        raise _error("dispatch outbox envelope must be an object", "dispatch_outbox_conflict")
    # Import lazily: distributed.py is also used by the Celery backend and
    # imports no storage implementation at module import time.
    from vv_agent.runtime.backends.distributed import DistributedRunEnvelope

    try:
        decoded = DistributedRunEnvelope.from_dict(dict(envelope))
    except Exception as exc:
        raise _error("dispatch outbox envelope is not the current distributed wire", "dispatch_outbox_conflict") from exc
    canonical = decoded.to_dict()
    if canonical != dict(envelope):
        raise _error("dispatch outbox envelope is not canonical", "dispatch_outbox_conflict")
    return canonical


def _identity_payload(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable delivery identity, excluding scheduling metadata."""
    payload = deepcopy(dict(envelope))
    # Deadline and claim/recovery metadata may legitimately change when a
    # broker response is pending or ambiguous.  They do not change the cycle
    # side effect identity represented by the stable job_id.
    for field_name in ("deadline_unix_ms", "claim_mode", "resume_attempt"):
        payload.pop(field_name, None)
    return payload


def dispatch_envelope_digest(envelope: Mapping[str, Any]) -> str:
    canonical = _validate_envelope(envelope)
    return canonical_json_sha256(_identity_payload(canonical), "distributed dispatch identity")


@dataclass(frozen=True, slots=True)
class DispatchOutboxRecord:
    schema_version: str
    dispatch_id: str
    checkpoint_key: str
    cycle_index: int
    envelope: dict[str, Any]
    envelope_digest: str
    state: DispatchState
    attempt: int = 0
    claim_token: str | None = None
    lease_expires_at_ms: int | None = None
    delivered_at_ms: int | None = None
    last_error: str | None = None

    @classmethod
    def pending(cls, envelope: Mapping[str, Any]) -> DispatchOutboxRecord:
        canonical = _validate_envelope(envelope)
        return cls(
            schema_version=DISPATCH_OUTBOX_SCHEMA,
            dispatch_id=_text(canonical["job_id"], "dispatch_id"),
            checkpoint_key=_text(canonical["checkpoint_config"]["key"], "checkpoint_key"),
            cycle_index=_integer(canonical["cycle_index"], "cycle_index", minimum=1),
            envelope=canonical,
            envelope_digest=dispatch_envelope_digest(canonical),
            state="pending",
        )

    def __post_init__(self) -> None:
        if self.schema_version != DISPATCH_OUTBOX_SCHEMA:
            raise _error("dispatch outbox schema_version is unsupported", "dispatch_outbox_conflict")
        envelope = _validate_envelope(self.envelope)
        dispatch_id = _text(self.dispatch_id, "dispatch_id")
        checkpoint_key = _text(self.checkpoint_key, "checkpoint_key")
        cycle_index = _integer(self.cycle_index, "cycle_index", minimum=1)
        if dispatch_id != envelope["job_id"] or checkpoint_key != envelope["checkpoint_config"]["key"]:
            raise _error("dispatch outbox identity conflicts with envelope", "dispatch_outbox_conflict")
        if cycle_index != envelope["cycle_index"]:
            raise _error("dispatch outbox cycle conflicts with envelope", "dispatch_outbox_conflict")
        digest = _digest(self.envelope_digest, "envelope_digest")
        if digest != dispatch_envelope_digest(envelope):
            raise _error("dispatch outbox envelope digest conflicts", "dispatch_outbox_conflict")
        if self.state not in {"pending", "claimed", "delivered", "ambiguous"}:
            raise _error("dispatch outbox state is invalid", "dispatch_outbox_conflict")
        attempt = _integer(self.attempt, "attempt")
        if self.state != "pending" and attempt < 1:
            raise _error("dispatch outbox non-pending state requires an attempt", "dispatch_outbox_conflict")
        claim = self.claim_token
        lease = self.lease_expires_at_ms
        if (claim is None) != (lease is None):
            raise _error("dispatch outbox claim and lease are inconsistent", "dispatch_outbox_conflict")
        if claim is not None:
            _text(claim, "claim_token")
            _integer(lease, "lease_expires_at_ms", minimum=1)
        if self.state == "claimed" and claim is None:
            raise _error("claimed dispatch outbox has no owner", "dispatch_outbox_conflict")
        if self.state != "claimed" and claim is not None:
            raise _error("unclaimed dispatch outbox has an owner", "dispatch_outbox_conflict")
        delivered_at = self.delivered_at_ms
        if delivered_at is not None:
            _integer(delivered_at, "delivered_at_ms")
        if self.state == "delivered" and delivered_at is None:
            raise _error("delivered dispatch outbox has no timestamp", "dispatch_outbox_conflict")
        if self.state != "delivered" and delivered_at is not None:
            raise _error("non-delivered dispatch outbox has a delivered timestamp", "dispatch_outbox_conflict")
        if self.last_error is not None:
            _text(self.last_error, "last_error", max_bytes=_MAX_ERROR_BYTES)
        object.__setattr__(self, "dispatch_id", dispatch_id)
        object.__setattr__(self, "checkpoint_key", checkpoint_key)
        object.__setattr__(self, "cycle_index", cycle_index)
        object.__setattr__(self, "envelope", envelope)
        object.__setattr__(self, "envelope_digest", digest)
        object.__setattr__(self, "attempt", attempt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dispatch_id": self.dispatch_id,
            "checkpoint_key": self.checkpoint_key,
            "cycle_index": self.cycle_index,
            "envelope": deepcopy(self.envelope),
            "envelope_digest": self.envelope_digest,
            "state": self.state,
            "attempt": self.attempt,
            "claim_token": self.claim_token,
            "lease_expires_at_ms": self.lease_expires_at_ms,
            "delivered_at_ms": self.delivered_at_ms,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> DispatchOutboxRecord:
        if not isinstance(payload, Mapping) or set(payload) != _DISPATCH_FIELDS:
            raise _error("dispatch outbox wire fields are invalid", "dispatch_outbox_conflict")
        return cls(
            schema_version=payload["schema_version"],
            dispatch_id=payload["dispatch_id"],
            checkpoint_key=payload["checkpoint_key"],
            cycle_index=payload["cycle_index"],
            envelope=dict(payload["envelope"]),
            envelope_digest=payload["envelope_digest"],
            state=payload["state"],
            attempt=payload["attempt"],
            claim_token=payload["claim_token"],
            lease_expires_at_ms=payload["lease_expires_at_ms"],
            delivered_at_ms=payload["delivered_at_ms"],
            last_error=payload["last_error"],
        )


@dataclass(frozen=True, slots=True)
class DispatchOutboxClaim:
    record: DispatchOutboxRecord
    should_enqueue: bool


@runtime_checkable
class DispatchOutboxStore(Protocol):
    """Transport-owned receipt storage for brokered cycle delivery.

    This protocol is deliberately separate from :class:`CheckpointStore`.
    Dispatch receipts describe a Celery/Apalis transport observation, not the
    language-neutral checkpoint contract.  A host may inject an implementation
    when it needs durable claim/lease/reconciliation semantics; the core
    checkpoint store remains usable without this transport capability.
    """

    def claim_distributed_dispatch(
        self,
        envelope: Mapping[str, Any],
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> DispatchOutboxClaim: ...

    def complete_distributed_dispatch(
        self,
        *,
        dispatch_id: str,
        envelope_digest: str,
        claim_token: str,
        attempt: int,
        outcome: str,
        now_ms: int,
        error: str | None = None,
    ) -> DispatchOutboxRecord | None: ...

    def reconcile_distributed_dispatch(
        self,
        *,
        dispatch_id: str,
        envelope_digest: str,
        outcome: str,
        now_ms: int,
        error: str | None = None,
    ) -> DispatchOutboxRecord | None: ...

    def get_distributed_dispatch(self, dispatch_id: str) -> DispatchOutboxRecord | None: ...

    def reap_distributed_dispatches(
        self,
        *,
        checkpoint_key: str | None = None,
        now_ms: int,
    ) -> list[DispatchOutboxRecord]: ...


def claim_dispatch(
    record: DispatchOutboxRecord,
    *,
    claim_token: str,
    lease_expires_at_ms: int,
    now_ms: int,
) -> DispatchOutboxClaim:
    _text(claim_token, "claim_token")
    _integer(lease_expires_at_ms, "lease_expires_at_ms", minimum=1)
    _integer(now_ms, "now_ms")
    if lease_expires_at_ms <= now_ms:
        raise _error("dispatch outbox lease must be in the future", "dispatch_outbox_conflict")
    if record.state == "delivered":
        return DispatchOutboxClaim(record=record, should_enqueue=False)
    if record.state == "ambiguous":
        raise _error("dispatch outbox requires reconciliation", "dispatch_outbox_reconciliation_required")
    if record.state == "claimed":
        if record.claim_token == claim_token:
            return DispatchOutboxClaim(record=record, should_enqueue=False)
        if (record.lease_expires_at_ms or 0) > now_ms:
            # An in-flight receipt is already owned by another scheduler.
            # Replays observe it but never steal the lease or enqueue a second
            # broker task. Only an expired claim may enter the explicit
            # reap/reconcile path.
            return DispatchOutboxClaim(record=record, should_enqueue=False)
        raise _error("dispatch outbox claim expired; reap before retry", "dispatch_outbox_stale")
    return DispatchOutboxClaim(
        record=replace(
            record,
            state="claimed",
            attempt=record.attempt + 1,
            claim_token=claim_token,
            lease_expires_at_ms=lease_expires_at_ms,
            delivered_at_ms=None,
            last_error=None,
        ),
        should_enqueue=True,
    )


def complete_dispatch(
    record: DispatchOutboxRecord,
    *,
    claim_token: str,
    attempt: int,
    outcome: Literal["delivered", "ambiguous"],
    now_ms: int,
    error: str | None = None,
) -> DispatchOutboxRecord:
    _integer(attempt, "attempt", minimum=1)
    _integer(now_ms, "now_ms")
    if outcome not in {"delivered", "ambiguous"}:
        raise _error("dispatch outbox completion outcome is invalid", "dispatch_outbox_conflict")
    if record.state != "claimed" or record.claim_token != claim_token or record.attempt != attempt:
        raise _error("dispatch outbox owner or attempt is stale", "dispatch_outbox_stale")
    if error is not None:
        _text(error, "last_error", max_bytes=_MAX_ERROR_BYTES)
    return replace(
        record,
        state=outcome,
        claim_token=None,
        lease_expires_at_ms=None,
        delivered_at_ms=now_ms if outcome == "delivered" else None,
        last_error=error,
    )


def reconcile_dispatch(
    record: DispatchOutboxRecord,
    *,
    outcome: Literal["retry", "delivered"],
    now_ms: int,
    error: str | None = None,
) -> DispatchOutboxRecord:
    _integer(now_ms, "now_ms")
    if outcome not in {"retry", "delivered"}:
        raise _error("dispatch outbox reconciliation outcome is invalid", "dispatch_outbox_conflict")
    if error is not None:
        _text(error, "last_error", max_bytes=_MAX_ERROR_BYTES)
    # A nonblocking scheduler may receive an explicit transport-failure
    # observation after ``send_task`` returned and the receipt was marked
    # delivered. That observation is the only legal path from delivered back
    # to pending; a replay without an error remains a no-op/conflict and cannot
    # accidentally enqueue the same task twice.
    if record.state == "delivered" and outcome == "retry" and error is not None:
        return replace(record, state="pending", delivered_at_ms=None, last_error=error)
    if record.state != "ambiguous":
        raise _error("dispatch outbox is not ambiguous", "dispatch_outbox_stale")
    if outcome == "retry":
        return replace(record, state="pending", last_error=error)
    return replace(record, state="delivered", delivered_at_ms=now_ms, last_error=error)


def reap_dispatch(record: DispatchOutboxRecord, *, now_ms: int) -> DispatchOutboxRecord | None:
    _integer(now_ms, "now_ms")
    if record.state != "claimed" or (record.lease_expires_at_ms or 0) > now_ms:
        return None
    return replace(
        record,
        state="ambiguous",
        claim_token=None,
        lease_expires_at_ms=None,
        delivered_at_ms=None,
        last_error="enqueue lease expired before broker outcome",
    )
