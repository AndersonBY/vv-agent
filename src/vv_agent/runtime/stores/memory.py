from __future__ import annotations

from dataclasses import replace
from threading import RLock
from typing import Any

from vv_agent.checkpoint import EventCursor
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
from vv_agent.events import ToolCallCompletedEvent, ToolCallDeferredEvent
from vv_agent.runtime.checkpoint_codec import clone_checkpoint
from vv_agent.runtime.state import (
    Checkpoint,
    CheckpointConflictError,
    ClaimMode,
    OperationState,
    _LeaseOperationClock,
    _validate_claim,
    _validate_renew,
    check_claim,
    checkpoint_definition_matches,
    claim_matches,
    prepare_claimed_terminal,
    prepare_event_delivery,
    validate_model_journal_accounting,
)
from vv_agent.types import AgentStatus, ToolExecutionResult


class InMemoryCheckpointStore:
    """Thread-safe process-local checkpoint store."""

    def __init__(self) -> None:
        self._store: dict[str, Checkpoint] = {}
        self._deferred_receipts: dict[str, DeferredResolutionReceipt] = {}
        self._lock = RLock()

    def create_checkpoint(self, checkpoint: Checkpoint) -> bool:
        snapshot = clone_checkpoint(checkpoint)
        if snapshot.revision != 0 or snapshot.resume_attempt != 1 or snapshot.claim_token is not None:
            raise ValueError("new checkpoint records must be unclaimed at revision zero")
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
            ):
                return False
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
        snapshot = clone_checkpoint(checkpoint)
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
    ) -> bool:
        _validate_renew(claim_token, lease_expires_at_ms, now_ms)
        clock = _LeaseOperationClock(now_ms)
        with self._lock:
            current_now_ms = clock.now_ms()
            checkpoint = self._store.get(checkpoint_key)
            if (
                checkpoint is None
                or checkpoint.claim_token != claim_token
                or (checkpoint.lease_expires_at_ms or 0) <= current_now_ms
                or lease_expires_at_ms <= current_now_ms
            ):
                return False
            checkpoint.lease_expires_at_ms = lease_expires_at_ms
            return True

    def acknowledge_terminal(self, checkpoint_key: str, *, expected_revision: int) -> bool:
        with self._lock:
            checkpoint = self._store.get(checkpoint_key)
            if (
                checkpoint is None
                or checkpoint.revision != expected_revision
                or checkpoint.terminal_result is None
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
            snapshot = clone_checkpoint(current)
            covered: set[str] = set()
            deferred_seen = False
            for call_id, outcome in normalized:
                # Completed outcomes identify a journal slot by tool_call_id;
                # a deferred outcome only carries the framework operation_id
                # in its opaque handle.  Accept both identities, but still
                # verify the complete handle below before mutating anything.
                entry = next(
                    (item for item in snapshot.tool_journal if item.tool_call_id == call_id or item.operation_id == call_id),
                    None,
                )
                if entry is None or entry.cycle_index != claimed_cycle or entry.state is not OperationState.STARTED:
                    return False
                identity = entry.operation_id
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
                    result = outcome.result
                    validate_definitive_result(result)
                    assert result is not None
                    if result.tool_call_id != entry.tool_call_id:
                        return False
                    entry.deferred_handle = None
                    if result.status_code.value == "SUCCESS":
                        entry.state = OperationState.SUCCEEDED
                        entry.result = result.to_dict()
                        entry.error = None
                    else:
                        from vv_agent.runtime.state import OperationError

                        entry.state = OperationState.FAILED
                        entry.result = None
                        entry.error = OperationError(
                            code=result.error_code or "tool_operation_failed",
                            message=result.content or "tool operation failed",
                            retryable=bool(result.metadata.get("retryable")),
                        )
                    event = ToolCallCompletedEvent(
                        run_id=snapshot.root_run_id,
                        trace_id=snapshot.trace_id,
                        cycle_index=entry.cycle_index,
                        tool_call_id=entry.tool_call_id or call_id,
                        tool_name=entry.tool_name or "tool",
                        operation_id=entry.operation_id,
                        attempt=entry.attempt,
                        status=result.status_code.value.lower(),
                        directive=result.directive.value,
                        error_code=result.error_code,
                        execution_started=True,
                        duration_ms=None,
                        checkpoint_key=snapshot.checkpoint_key,
                        event_id=_stable_deferred_event_id(
                            entry,
                            "completed" if result.status_code.value == "SUCCESS" else "failed",
                        ),
                    ).to_dict()
                _enqueue_event(snapshot, event)
            # Admission is the all-or-none boundary for the complete started
            # model-tool batch.  A missing started slot would otherwise leave
            # an unclassified external operation behind a released claim.
            if any(
                entry.cycle_index == claimed_cycle and entry.state is OperationState.STARTED and entry.operation_id not in covered
                for entry in [*snapshot.model_call_journal, *snapshot.tool_journal]
            ):
                return False
            if deferred_seen:
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
                if item.operation_id == handle.operation_id and item.attempt == handle.attempt
            )
            from vv_agent.runtime.state import OperationError

            if result.status_code.value == "SUCCESS":
                target.state = OperationState.SUCCEEDED
                target.deferred_handle = None
                target.result = result.to_dict()
                target.error = None
                receipt_status = "succeeded"
            else:
                target.state = OperationState.FAILED
                target.deferred_handle = None
                target.result = None
                target.error = OperationError(
                    code=result.error_code or "tool_operation_failed",
                    message=result.content or "tool operation failed",
                    retryable=bool(result.metadata.get("retryable")),
                )
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
                event_id=_stable_deferred_event_id(
                    target,
                    "completed" if result.status_code.value == "SUCCESS" else "failed",
                ),
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
            if (
                current is None
                or current.revision != expected_revision
                or checkpoint.revision != expected_revision
                or not checkpoint_definition_matches(current, checkpoint)
            ):
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
            decision_keys = [(item.handle.operation_id, item.handle.attempt) for item in parsed]
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
                    if item.state is OperationState.DEFERRED and (item.operation_id, item.attempt) in decision_keys_set
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
                    ),
                    None,
                )
                if entry is None or entry.state is not OperationState.DEFERRED or entry.deferred_handle != decision.handle:
                    replayed = False
                    break
            if replayed:
                return True
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
            current_keys = {(item.operation_id, item.attempt) for item in current_entries}
            if set(decision_keys) != current_keys:
                return False
            # Provider response order is not the model-call order.  Apply
            # accepted entries in journal order so event/outbox ordering is
            # deterministic and independent of reconciliation delivery order.
            decisions_by_key = dict(zip(decision_keys, parsed, strict=True))
            ordered_decisions = [decisions_by_key[(item.operation_id, item.attempt)] for item in current_entries]
            for decision in ordered_decisions:
                entry = next(
                    (
                        item
                        for item in snapshot.tool_journal
                        if item.cycle_index == current_cycle
                        and item.operation_id == decision.handle.operation_id
                        and item.attempt == decision.handle.attempt
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
                audit = {
                    "version": "v4",
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
                raise ValueError("event_identity_conflict")
            return
    checkpoint.event_outbox.append(EventOutboxEntry.pending(event_id, event))
