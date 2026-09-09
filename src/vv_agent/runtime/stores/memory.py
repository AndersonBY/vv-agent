from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from threading import RLock
from typing import Any, Literal, cast

import vv_agent.events as run_events
from vv_agent.checkpoint import CheckpointError, EventCursor
from vv_agent.deferred import (
    DeferredResolutionReceipt,
    DeferredResolveDecision,
    DeferredToolHandle,
)
from vv_agent.runtime.checkpoint_codec import clone_checkpoint
from vv_agent.runtime.dispatch_outbox import (
    DispatchOutboxClaim,
    DispatchOutboxRecord,
    claim_dispatch,
    complete_dispatch,
    reap_dispatch,
    reconcile_dispatch,
)
from vv_agent.runtime.state import (
    Checkpoint,
    CheckpointConflictError,
    CheckpointRenewal,
    ClaimMode,
    OperationState,
    RenewOutcome,
    _LeaseOperationClock,
    _validate_claim,
    _validate_renew,
    check_claim,
    checkpoint_definition_matches,
    claim_matches,
    merge_event_outbox,
    prepare_claimed_terminal,
    prepare_deferred_acceptance,
    prepare_deferred_admission,
    prepare_deferred_resolution,
    prepare_event_delivery,
    prepare_tool_receipt,
    prepare_unclaimed_terminal,
    validate_checkpoint_creation,
    validate_model_journal_accounting,
)
from vv_agent.runtime.stores.controller_store import ControllerStoreMixin
from vv_agent.types import AgentStatus, ToolExecutionResult


class InMemoryCheckpointStore(ControllerStoreMixin):
    """Thread-safe process-local checkpoint store."""

    def __init__(self) -> None:
        self._store: dict[str, Checkpoint] = {}
        self._deferred_receipts: dict[str, DeferredResolutionReceipt] = {}
        self._lock = RLock()
        self._init_controller_indexes()
        self._distributed_dispatch_outboxes: dict[str, DispatchOutboxRecord] = {}

    def create_checkpoint(self, checkpoint: Checkpoint) -> bool:
        snapshot = clone_checkpoint(checkpoint)
        validate_checkpoint_creation(snapshot)
        with self._lock:
            if snapshot.checkpoint_key in self._store:
                return False
            self._store[snapshot.checkpoint_key] = snapshot
            return True

    def load_checkpoint(self, checkpoint_key: str) -> Checkpoint | None:
        with self._lock:
            checkpoint = self._store.get(checkpoint_key)
            return clone_checkpoint(checkpoint) if checkpoint is not None else None

    def claim_checkpoint(
        self,
        checkpoint_key: str,
        cycle_index: int,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
        claim_mode: ClaimMode,
    ) -> Checkpoint | None:
        _validate_claim(cycle_index, claim_token, lease_expires_at_ms, now_ms)
        with self._lock:
            checkpoint = self._store.get(checkpoint_key)
            if checkpoint is None:
                return None
            try:
                check_claim(checkpoint, cycle_index, now_ms, claim_mode)
            except ValueError as exc:
                raise CheckpointConflictError(str(exc)) from exc
            if checkpoint.claim_token is not None and claim_mode != "recovery":
                raise CheckpointConflictError("expired checkpoint claims require recovery mode")
            if checkpoint.status is AgentStatus.RECONCILIATION_REQUIRED and claim_mode != "recovery":
                raise CheckpointConflictError("reconciliation checkpoints require recovery mode")
            if any(
                record["checkpoint_key"] == checkpoint_key and record["state"] in {"resolved_pending", "resolved_claimed"}
                for record in self._host_interaction_records.values()
            ):
                raise CheckpointError(
                    "host interaction response requires dedicated recovery",
                    code="host_interaction_recovery_required",
                )
            checkpoint.revision += 1
            if claim_mode == "recovery":
                checkpoint.resume_attempt += 1
            checkpoint.status = AgentStatus.RUNNING
            checkpoint.claim_token = claim_token
            checkpoint.claimed_cycle = cycle_index
            checkpoint.lease_expires_at_ms = lease_expires_at_ms
            return clone_checkpoint(checkpoint)

    def progress_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        with self._lock:
            current = self._store.get(checkpoint.checkpoint_key)
            if not claim_matches(current, checkpoint, claim_token, expected_revision):
                return False
            assert current is not None
            if (
                current.terminal_result is not None
                or current.status is not AgentStatus.RUNNING
                or checkpoint.status is not AgentStatus.RUNNING
            ):
                return False
            snapshot = clone_checkpoint(checkpoint)
            snapshot.event_outbox = merge_event_outbox(current.event_outbox, snapshot.event_outbox)
            snapshot.cancel_requested = snapshot.cancel_requested or current.cancel_requested
            snapshot.event_cursor = deepcopy(current.event_cursor)
            snapshot.revision = expected_revision + 1
            snapshot.claim_token = current.claim_token
            snapshot.claimed_cycle = current.claimed_cycle
            snapshot.lease_expires_at_ms = current.lease_expires_at_ms
            self._store[snapshot.checkpoint_key] = snapshot
            return True

    def suspend_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        with self._lock:
            current = self._store.get(checkpoint.checkpoint_key)
            if not claim_matches(current, checkpoint, claim_token, expected_revision):
                return False
            assert current is not None
            if (
                current.terminal_result is not None
                or checkpoint.cycle_index != current.cycle_index
                or checkpoint.status is not AgentStatus.RECONCILIATION_REQUIRED
            ):
                return False
            snapshot = clone_checkpoint(
                replace(
                    checkpoint,
                    revision=expected_revision + 1,
                    claim_token=None,
                    claimed_cycle=None,
                    lease_expires_at_ms=None,
                )
            )
            snapshot.event_outbox = merge_event_outbox(current.event_outbox, snapshot.event_outbox)
            snapshot.event_cursor = deepcopy(current.event_cursor)
            snapshot.cancel_requested = current.cancel_requested
            self._store[snapshot.checkpoint_key] = snapshot
            return True

    def commit_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        with self._lock:
            current = self._store.get(checkpoint.checkpoint_key)
            if not claim_matches(current, checkpoint, claim_token, expected_revision):
                return False
            assert current is not None
            if (
                current.terminal_result is not None
                or checkpoint.terminal_result is not None
                or checkpoint.status is not AgentStatus.RUNNING
                or checkpoint.cycle_index != current.claimed_cycle
                or current.cancel_requested
                or checkpoint.cancel_requested
                or any(
                    entry.state
                    in {OperationState.PLANNED, OperationState.STARTED, OperationState.DEFERRED, OperationState.AMBIGUOUS}
                    for entry in [*checkpoint.model_call_journal, *checkpoint.tool_journal]
                )
            ):
                return False
            checkpoint.event_outbox = merge_event_outbox(current.event_outbox, checkpoint.event_outbox)
            checkpoint.event_cursor = deepcopy(current.event_cursor)
            validate_model_journal_accounting(checkpoint)
            snapshot = clone_checkpoint(
                replace(
                    checkpoint,
                    revision=expected_revision + 1,
                    claim_token=None,
                    claimed_cycle=None,
                    lease_expires_at_ms=None,
                    event_outbox=[entry for entry in checkpoint.event_outbox if entry.state == "pending"],
                    model_call_journal=[],
                    tool_journal=[],
                )
            )
            self._store[snapshot.checkpoint_key] = snapshot
            return True

    def finalize_checkpoint(self, checkpoint: Checkpoint, *, expected_revision: int) -> bool:
        snapshot = prepare_unclaimed_terminal(checkpoint)
        snapshot = clone_checkpoint(snapshot)
        if snapshot.terminal_result is None or snapshot.claim_token is not None:
            raise ValueError("finalized checkpoint must be terminal and unclaimed")
        with self._lock:
            current = self._store.get(snapshot.checkpoint_key)
            if (
                current is None
                or current.revision != expected_revision
                or snapshot.revision != expected_revision
                or current.claim_token is not None
                or current.terminal_result is not None
                or not checkpoint_definition_matches(current, snapshot)
            ):
                return False
            snapshot.cancel_requested = current.cancel_requested
            snapshot.event_outbox = merge_event_outbox(current.event_outbox, snapshot.event_outbox)
            snapshot.event_cursor = deepcopy(current.event_cursor)
            snapshot.revision = expected_revision + 1
            self._store[snapshot.checkpoint_key] = snapshot
            return True

    def finalize_claimed_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        with self._lock:
            current = self._store.get(checkpoint.checkpoint_key)
            if current is None:
                return False
            terminal = prepare_claimed_terminal(
                current,
                checkpoint,
                claim_token=claim_token,
                expected_revision=expected_revision,
            )
            if terminal is None:
                return False
            self._store[terminal.checkpoint_key] = terminal
            return True

    def record_event_delivery(
        self,
        checkpoint_key: str,
        *,
        event_id: str,
        payload_digest: str,
        cursor: EventCursor,
        expected_revision: int,
        claim_token: str | None,
    ) -> bool:
        with self._lock:
            current = self._store.get(checkpoint_key)
            if current is None:
                return False
            delivered = prepare_event_delivery(
                current,
                event_id=event_id,
                payload_digest=payload_digest,
                cursor=cursor,
                expected_revision=expected_revision,
                claim_token=claim_token,
            )
            if delivered is None:
                return False
            self._store[checkpoint_key] = delivered
            return True

    def renew_checkpoint_claim(
        self,
        checkpoint_key: str,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> CheckpointRenewal:
        _validate_renew(claim_token, lease_expires_at_ms, now_ms)
        clock = _LeaseOperationClock(now_ms)
        with self._lock:
            current_now_ms = clock.now_ms()
            checkpoint = self._store.get(checkpoint_key)
            if checkpoint is None:
                return CheckpointRenewal(outcome=RenewOutcome.CLAIM_LOST, revision=0)
            if (
                checkpoint.claim_token != claim_token
                or (checkpoint.lease_expires_at_ms or 0) <= current_now_ms
                or lease_expires_at_ms <= current_now_ms
            ):
                return CheckpointRenewal(outcome=RenewOutcome.CLAIM_LOST, revision=checkpoint.revision)
            checkpoint.lease_expires_at_ms = lease_expires_at_ms
            return CheckpointRenewal(
                outcome=(RenewOutcome.CANCEL_REQUESTED if checkpoint.cancel_requested else RenewOutcome.RENEWED),
                lease_expires_at_ms=lease_expires_at_ms,
            )

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
    ) -> bool:
        with self._lock:
            current = self._store.get(checkpoint.checkpoint_key)
            updated = prepare_tool_receipt(
                current,
                checkpoint,
                operation_id=operation_id,
                attempt=attempt,
                tool_call_id=tool_call_id,
                request_digest=request_digest,
                result=result,
                claim_token=claim_token,
                expected_revision=expected_revision,
                claimed_cycle=claimed_cycle,
                created_at=run_events.event_created_at(),
            )
            if updated is None:
                return False
            if updated is not current:
                self._store[updated.checkpoint_key] = updated
            return True

    def acknowledge_terminal(self, checkpoint_key: str, *, expected_revision: int) -> bool:
        with self._lock:
            checkpoint = self._store.get(checkpoint_key)
            if (
                checkpoint is None
                or checkpoint.revision != expected_revision
                or checkpoint.terminal_result is None
                or checkpoint.claim_token is not None
                or checkpoint.terminal_acknowledged
            ):
                return False
            checkpoint.revision += 1
            checkpoint.terminal_acknowledged = True
            return True

    def delete_checkpoint(self, checkpoint_key: str) -> None:
        with self._lock:
            self._store.pop(checkpoint_key, None)
            self._deferred_receipts = {
                key: receipt
                for key, receipt in self._deferred_receipts.items()
                if receipt.handle.checkpoint_key != checkpoint_key
            }
            self._host_interaction_records = {
                key: record for key, record in self._host_interaction_records.items() if key[0] != checkpoint_key
            }
            self._controller_command_receipts = {
                key: receipt
                for key, receipt in self._controller_command_receipts.items()
                if receipt.handle.checkpoint_key != checkpoint_key
            }
            self._controller_command_outboxes = {
                key: row
                for key, row in self._controller_command_outboxes.items()
                if self._controller_command_receipts.get(key) is not None
            }
            self._host_interaction_notifications = {
                key: row for key, row in self._host_interaction_notifications.items() if row["checkpoint_key"] != checkpoint_key
            }
            self._distributed_dispatch_outboxes = {
                key: row for key, row in self._distributed_dispatch_outboxes.items() if row.checkpoint_key != checkpoint_key
            }

    def claim_distributed_dispatch(
        self,
        envelope: Mapping[str, Any],
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> DispatchOutboxClaim:
        candidate = DispatchOutboxRecord.pending(envelope)
        with self._lock:
            if candidate.checkpoint_key not in self._store:
                raise CheckpointError("dispatch checkpoint was not found", code="checkpoint_not_found")
            current = self._distributed_dispatch_outboxes.get(candidate.dispatch_id)
            if current is None:
                current = candidate
            elif current.envelope_digest != candidate.envelope_digest:
                raise CheckpointError(
                    "dispatch id was reused with a different immutable envelope",
                    code="dispatch_outbox_conflict",
                )
            claim = claim_dispatch(
                current,
                claim_token=claim_token,
                lease_expires_at_ms=lease_expires_at_ms,
                now_ms=now_ms,
            )
            if claim.record != current:
                self._distributed_dispatch_outboxes[candidate.dispatch_id] = claim.record
            return claim

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
    ) -> DispatchOutboxRecord | None:
        with self._lock:
            current = self._distributed_dispatch_outboxes.get(dispatch_id)
            if current is None:
                return None
            if current.envelope_digest != envelope_digest:
                raise CheckpointError("dispatch envelope digest conflicts", code="dispatch_outbox_conflict")
            if outcome not in {"delivered", "ambiguous"}:
                raise ValueError("dispatch completion outcome must be delivered or ambiguous")
            typed_outcome = cast(Literal["delivered", "ambiguous"], outcome)
            if current.state == typed_outcome:
                return current
            updated = complete_dispatch(
                current,
                claim_token=claim_token,
                attempt=attempt,
                outcome=typed_outcome,
                now_ms=now_ms,
                error=error,
            )
            self._distributed_dispatch_outboxes[dispatch_id] = updated
            return updated

    def reconcile_distributed_dispatch(
        self,
        *,
        dispatch_id: str,
        envelope_digest: str,
        outcome: str,
        now_ms: int,
        error: str | None = None,
    ) -> DispatchOutboxRecord | None:
        with self._lock:
            current = self._distributed_dispatch_outboxes.get(dispatch_id)
            if current is None:
                return None
            if current.envelope_digest != envelope_digest:
                raise CheckpointError("dispatch envelope digest conflicts", code="dispatch_outbox_conflict")
            if outcome not in {"retry", "delivered"}:
                raise ValueError("dispatch reconciliation outcome must be retry or delivered")
            typed_outcome = cast(Literal["retry", "delivered"], outcome)
            if (typed_outcome == "retry" and current.state == "pending") or (
                typed_outcome == "delivered" and current.state == "delivered"
            ):
                return current
            updated = reconcile_dispatch(
                current,
                outcome=typed_outcome,
                now_ms=now_ms,
                error=error,
            )
            self._distributed_dispatch_outboxes[dispatch_id] = updated
            return updated

    def get_distributed_dispatch(self, dispatch_id: str) -> DispatchOutboxRecord | None:
        with self._lock:
            return self._distributed_dispatch_outboxes.get(dispatch_id)

    def reap_distributed_dispatches(
        self,
        *,
        checkpoint_key: str | None = None,
        now_ms: int,
    ) -> list[DispatchOutboxRecord]:
        with self._lock:
            rows: list[DispatchOutboxRecord] = []
            for dispatch_id, current in tuple(self._distributed_dispatch_outboxes.items()):
                if checkpoint_key is not None and current.checkpoint_key != checkpoint_key:
                    continue
                updated = reap_dispatch(current, now_ms=now_ms)
                if updated is not None:
                    self._distributed_dispatch_outboxes[dispatch_id] = updated
                    rows.append(updated)
            return rows

    def preflight_tool_batch(
        self,
        checkpoint: Checkpoint,
        *,
        tool_call_count: int,
        expected_revision: int,
        claim_token: str,
        claimed_cycle: int,
    ) -> bool:
        """Prove lifecycle-outbox writability before the first tool effect.

        The current wire has no arbitrary outbox cardinality or byte cap.  An
        explicit store boundary is still required: a bounded or unavailable
        implementation can return ``False`` and the runner will refuse to
        dispatch the provider.  The in-memory store validates the
        authoritative claim/revision and serializes a complete snapshot.
        """
        if isinstance(tool_call_count, bool) or not isinstance(tool_call_count, int) or tool_call_count <= 0:
            return False
        with self._lock:
            current = self._store.get(checkpoint.checkpoint_key)
            if (
                current is None
                or current.revision != expected_revision
                or checkpoint.revision != expected_revision
                or current.claim_token != claim_token
                or current.claimed_cycle != claimed_cycle
                or current.status is not AgentStatus.RUNNING
                or current.terminal_result is not None
            ):
                return False
            try:
                from vv_agent.runtime.state import validate_checkpoint

                validate_checkpoint(current)
                clone_checkpoint(current)
            except Exception:
                return False
            return True

    def admit_deferred_batch(
        self,
        checkpoint: Checkpoint,
        *,
        outcomes: list[Any],
        claim_token: str,
        expected_revision: int,
        claimed_cycle: int,
    ) -> bool:
        with self._lock:
            current = self._store.get(checkpoint.checkpoint_key)
            updated = prepare_deferred_admission(
                current,
                checkpoint,
                outcomes=outcomes,
                claim_token=claim_token,
                expected_revision=expected_revision,
                claimed_cycle=claimed_cycle,
                created_at=run_events.event_created_at(),
            )
            if updated is None:
                return False
            if updated is not current:
                self._store[updated.checkpoint_key] = updated
            return True

    def resolve_deferred(self, handle: DeferredToolHandle, result: Any) -> DeferredResolveDecision:
        with self._lock:
            updated, decision = prepare_deferred_resolution(
                self._store.get(handle.checkpoint_key),
                self._deferred_receipts.get(handle.key),
                handle,
                result,
                created_at=run_events.event_created_at(),
            )
            if updated is not None:
                self._store[updated.checkpoint_key] = updated
                assert decision.receipt is not None
                self._deferred_receipts[handle.key] = decision.receipt
            return decision

    def accept_deferred_batch(
        self,
        checkpoint: Checkpoint,
        *,
        decisions: list[Any],
        claim_token: str,
        expected_revision: int,
        claimed_cycle: int,
    ) -> bool:
        with self._lock:
            current = self._store.get(checkpoint.checkpoint_key)
            updated = prepare_deferred_acceptance(
                current,
                checkpoint,
                decisions=decisions,
                claim_token=claim_token,
                expected_revision=expected_revision,
                claimed_cycle=claimed_cycle,
                created_at=run_events.event_created_at(),
            )
            if updated is None:
                return False
            if updated is not current:
                self._store[updated.checkpoint_key] = updated
            return True
