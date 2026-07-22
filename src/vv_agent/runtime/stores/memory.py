from __future__ import annotations

from dataclasses import replace
from threading import RLock

from vv_agent.checkpoint import EventCursor
from vv_agent.runtime.checkpoint_codec import clone_checkpoint
from vv_agent.runtime.state import (
    Checkpoint,
    CheckpointConflictError,
    ClaimMode,
    _LeaseOperationClock,
    _validate_claim,
    _validate_renew,
    check_claim,
    checkpoint_definition_matches,
    claim_matches,
    prepare_claimed_terminal,
    prepare_event_delivery,
)
from vv_agent.types import AgentStatus


class InMemoryCheckpointStore:
    """Thread-safe process-local checkpoint store."""

    def __init__(self) -> None:
        self._store: dict[str, Checkpoint] = {}
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
            snapshot = clone_checkpoint(
                replace(
                    checkpoint,
                    revision=expected_revision + 1,
                    claim_token=None,
                    claimed_cycle=None,
                    lease_expires_at_ms=None,
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
