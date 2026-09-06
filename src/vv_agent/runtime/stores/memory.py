from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from threading import RLock
from typing import Any, Literal, cast

from vv_agent.checkpoint import CheckpointError, EventCursor, ResumeObservation
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
from vv_agent.events import RUN_EVENT_VERSION, ToolCallCompletedEvent, ToolCallDeferredEvent
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
    operation_error_from_tool_result,
    prepare_claimed_terminal,
    prepare_event_delivery,
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
        validate_definitive_result(result)
        if result.tool_call_id != tool_call_id:
            raise CheckpointError(
                "tool receipt result tool_call_id does not match the journal identity",
                code="tool_receipt_identity_invalid",
            )
        from vv_agent.checkpoint import canonical_json_sha256
        from vv_agent.runtime.state import compute_tool_identity_key

        identity_key = compute_tool_identity_key(
            checkpoint.checkpoint_key,
            operation_id,
            attempt,
            tool_call_id,
            request_digest,
        )
        result_digest = canonical_json_sha256(result.to_dict(), "tool result")
        with self._lock:
            current = self._store.get(checkpoint.checkpoint_key)
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
                    return True
                raise CheckpointError(
                    "tool receipt conflicts with the retained identity",
                    code="tool_receipt_conflict",
                )
            if current is None:
                return False
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
                return False
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
                return False
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
            ).to_dict()
            _enqueue_event(snapshot, event)
            snapshot.revision = expected_revision + 1
            from vv_agent.runtime.state import validate_checkpoint

            validate_checkpoint(snapshot)
            self._store[snapshot.checkpoint_key] = snapshot
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
        """Atomically admit one ordered model-tool batch.

        The store owns the only claim release for a deferred batch.  All
        journal and lifecycle outbox mutations are prepared on a clone and
        validated before replacing the authoritative record.
        """
        with self._lock:
            current = self._store.get(checkpoint.checkpoint_key)
            if (
                current is None
                or current.revision != expected_revision
                or current.claim_token != claim_token
                or current.claimed_cycle != claimed_cycle
                or checkpoint.revision != expected_revision
            ):
                return False
            normalized = _normalize_batch_outcomes(outcomes)
            if not normalized:
                return False
            if any(outcome.kind == "completed" for _call_id, outcome in normalized):
                raise CheckpointError(
                    "deferred admission accepts deferred outcomes only",
                    code="deferred_admission_completed_outcome_invalid",
                )
            snapshot = clone_checkpoint(current)
            covered: set[tuple[str, int, str | None, str, int]] = set()
            deferred_seen = False
            for call_id, outcome in normalized:
                # Completed outcomes identify a journal slot by tool_call_id;
                # deferred outcomes carry the complete operation identity in
                # their handle.  Select by that identity before mutating.
                handle = outcome.handle if outcome.kind == "deferred" else None
                entry = next(
                    (
                        item
                        for item in snapshot.tool_journal
                        if item.cycle_index == claimed_cycle
                        and (
                            (
                                handle is not None
                                and item.operation_id == handle.operation_id
                                and item.attempt == handle.attempt
                                and item.request_digest == handle.request_digest
                            )
                            or (handle is None and item.tool_call_id == call_id)
                        )
                    ),
                    None,
                )
                if entry is None or entry.cycle_index != claimed_cycle or entry.state is not OperationState.STARTED:
                    return False
                identity = (
                    entry.operation_id,
                    entry.attempt,
                    entry.tool_call_id,
                    entry.request_digest,
                    entry.cycle_index,
                )
                if identity in covered:
                    return False
                covered.add(identity)
                if outcome.kind == "deferred":
                    handle = outcome.handle
                    assert handle is not None
                    if (
                        handle.checkpoint_key != snapshot.checkpoint_key
                        or handle.operation_id != entry.operation_id
                        or handle.attempt != entry.attempt
                        or handle.request_digest != entry.request_digest
                    ):
                        return False
                    entry.state = OperationState.DEFERRED
                    entry.deferred_handle = handle
                    entry.result = None
                    entry.error = None
                    deferred_seen = True
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
                    ).to_dict()
                else:
                    # Definitive outcomes are admitted by record_tool_receipt
                    # before this barrier.  Re-admitting one would duplicate
                    # the receipt event and violate the one-claim release.
                    raise CheckpointError(
                        "deferred admission accepts deferred outcomes only",
                        code="deferred_admission_completed_outcome_invalid",
                    )
                _enqueue_event(snapshot, event)
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
            if not deferred_seen:
                return False
            snapshot.status = AgentStatus.DEFERRED
            snapshot.claim_token = None
            snapshot.claimed_cycle = None
            snapshot.lease_expires_at_ms = None
            snapshot.revision = expected_revision + 1
            try:
                from vv_agent.runtime.state import validate_checkpoint

                validate_checkpoint(snapshot)
            except Exception:
                return False
            self._store[snapshot.checkpoint_key] = snapshot
            return True

    def resolve_deferred(self, handle: DeferredToolHandle, result: Any) -> DeferredResolveDecision:
        validate_definitive_result(result)
        with self._lock:
            existing = self._deferred_receipts.get(handle.key)
            if existing is not None:
                if existing.handle.key != handle.key or existing.handle_key != handle.key:
                    raise ValueError("deferred_receipt_identity_invalid")
                if existing.result.to_dict() != result.to_dict():
                    raise DeferredResolutionConflict()
                return DeferredResolveDecision.Replayed(existing)
            checkpoint = self._store.get(handle.checkpoint_key)
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
                return DeferredResolveDecision.NotAdmitted()
            if entry.state is OperationState.AMBIGUOUS:
                return DeferredResolveDecision.ReconciliationRequired()
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
            from vv_agent.runtime.state import compute_tool_identity_key

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
            ).to_dict()
            _enqueue_event(snapshot, event)
            remaining = [item for item in snapshot.tool_journal if item.state is OperationState.DEFERRED]
            snapshot.status = AgentStatus.DEFERRED if remaining else AgentStatus.RUNNING
            snapshot.revision = checkpoint.revision + 1
            receipt = DeferredResolutionReceipt(
                handle=handle,
                result=result,
                result_digest=__import__("vv_agent.checkpoint", fromlist=["canonical_json_sha256"]).canonical_json_sha256(
                    result.to_dict(), "deferred result"
                ),
                event_id=event["event_id"],
                event_payload_digest=__import__(
                    "vv_agent.checkpoint", fromlist=["compute_event_payload_digest"]
                ).compute_event_payload_digest(event),
                receipt_status=receipt_status,
            )
            from vv_agent.runtime.state import validate_checkpoint

            validate_checkpoint(snapshot)
            self._store[snapshot.checkpoint_key] = snapshot
            self._deferred_receipts[handle.key] = receipt
            return (
                DeferredResolveDecision.AppliedReady(receipt)
                if not remaining
                else DeferredResolveDecision.AppliedWaiting(receipt)
            )

    def accept_deferred_batch(
        self,
        checkpoint: Checkpoint,
        *,
        decisions: list[Any],
        claim_token: str,
        expected_revision: int,
        claimed_cycle: int,
    ) -> bool:
        from vv_agent.deferred import AcceptDeferredDecision

        with self._lock:
            current = self._store.get(checkpoint.checkpoint_key)
            if current is None:
                return False
            parsed = [d if isinstance(d, AcceptDeferredDecision) else AcceptDeferredDecision.from_dict(d) for d in decisions]
            if not parsed:
                return False

            # This is a batch boundary, not a per-operation convenience
            # method.  Reject duplicate handles and any decision set which
            # does not cover the complete current-cycle ambiguity.  The
            # controller performs the same aggregation before calling the
            # store, but keeping the invariant here is essential for direct
            # SQLite/Redis callers and for the all-or-none CAS contract.
            decision_keys = [(item.handle.operation_id, item.handle.attempt, item.handle.request_digest) for item in parsed]
            if len(decision_keys) != len(set(decision_keys)):
                return False

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
                item
                for item in snapshot.tool_journal
                if item.cycle_index == current_cycle and item.state is OperationState.DEFERRED
            ]
            replayed = (
                bool(replay_entries)
                and len(parsed) == len(replay_entries)
                and all(item.kind.value == "tool" for item in replay_entries)
                and not any(
                    item.state in {OperationState.AMBIGUOUS, OperationState.STARTED, OperationState.PLANNED}
                    for item in all_cycle_entries
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
                return True
            if (
                current.revision != expected_revision
                or checkpoint.revision != expected_revision
                or not checkpoint_definition_matches(current, checkpoint)
            ):
                return False
            if current.resume_attempt <= 1 or current.claim_token != claim_token or current.claimed_cycle != claimed_cycle:
                return False

            batch_entries = [
                item for item in [*snapshot.model_call_journal, *snapshot.tool_journal] if item.cycle_index == current_cycle
            ]
            # The store, not only the recovery controller, owns the complete
            # batch invariant. Any model entry, STARTED entry, or omitted
            # current-cycle operation makes partial acceptance unsafe.
            if not batch_entries or any(item.state in {OperationState.STARTED, OperationState.PLANNED} for item in batch_entries):
                return False
            ambiguous_entries = [item for item in batch_entries if item.state is OperationState.AMBIGUOUS]
            if not ambiguous_entries or any(item.kind.value != "tool" for item in ambiguous_entries):
                return False
            # A recovery acceptance may include already-adopted entries when
            # a caller retries after a partial transport failure, but every
            # current-cycle entry must be represented exactly once and the
            # batch must contain tools only.  Any omission or extra handle
            # therefore leaves the checkpoint untouched.
            current_entries = list(ambiguous_entries)
            if not ambiguous_entries or len(parsed) != len(current_entries):
                return False
            current_keys = {(item.operation_id, item.attempt, item.request_digest) for item in current_entries}
            if set(decision_keys) != current_keys:
                return False
            # Provider response order is not the model-call order.  Apply
            # accepted entries in journal order so event/outbox ordering is
            # deterministic and independent of reconciliation delivery order.
            decisions_by_key = dict(zip(decision_keys, parsed, strict=True))
            ordered_decisions = [
                decisions_by_key[(item.operation_id, item.attempt, item.request_digest)] for item in current_entries
            ]
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
                    return False
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
                    return False
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
                _enqueue_event(snapshot, audit)
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
                ).to_dict()
                _enqueue_event(snapshot, deferred)
            snapshot.status = AgentStatus.DEFERRED
            snapshot.claim_token = None
            snapshot.claimed_cycle = None
            snapshot.lease_expires_at_ms = None
            snapshot.revision = expected_revision + 1
            from vv_agent.runtime.state import validate_checkpoint

            validate_checkpoint(snapshot)
            self._store[snapshot.checkpoint_key] = snapshot
            return True


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
    # The callback receipt identity is intentionally derived from the stable
    # model tool-call id, not a process-local timestamp or attempt counter.
    # This is what lets a resolution CAS and every replay expose the same
    # tool_call_completed event id.  Older operation-shaped ids are not
    # accepted as a second wire shape at the current revision.
    call_id = entry.tool_call_id or entry.operation_id
    return f"evt_deferred_{call_id}_{suffix}"


def _enqueue_event(checkpoint: Checkpoint, event: dict[str, Any]) -> None:
    from vv_agent.runtime.state import EventOutboxEntry

    event_id = event["event_id"]
    for existing in checkpoint.event_outbox:
        if existing.event_id == event_id:
            if existing.event != event:
                raise CheckpointError("event_identity_conflict", code="event_identity_conflict")
            return
    checkpoint.event_outbox.append(EventOutboxEntry.pending(event_id, event))
