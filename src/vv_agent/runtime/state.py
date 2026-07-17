from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Any, Literal, Protocol, runtime_checkable

from vv_agent.budget import BudgetUsageSnapshot
from vv_agent.checkpoint import EventCursor
from vv_agent.runtime.state_v2 import (
    CheckpointV2,
    ClaimMode,
    check_claim_v2,
    checkpoint_definition_matches_v2,
    claim_matches_v2,
    prepare_claimed_terminal_v2,
    prepare_event_delivery_v2,
)
from vv_agent.types import AgentResult, AgentStatus, CycleRecord, Message


@dataclass(slots=True)
class Checkpoint:
    task_id: str
    cycle_index: int
    status: AgentStatus
    messages: list[Message]
    cycles: list[CycleRecord]
    shared_state: dict[str, Any] = field(default_factory=dict)
    revision: int = 0
    claim_token: str | None = None
    claimed_cycle: int | None = None
    lease_expires_at_ms: int | None = None
    terminal_result: AgentResult | None = None
    budget_usage: BudgetUsageSnapshot | None = None


class CheckpointConflictError(RuntimeError):
    """The requested worker cycle no longer matches the durable checkpoint."""


@dataclass(frozen=True, slots=True)
class StateStoreSpec:
    kind: Literal["sqlite", "redis"]
    location: str

    def __post_init__(self) -> None:
        if self.kind not in {"sqlite", "redis"}:
            raise ValueError(f"unsupported state store kind: {self.kind}")
        if not isinstance(self.location, str) or not self.location.strip():
            raise ValueError("state store location must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "location": self.location}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StateStoreSpec:
        if not isinstance(payload, Mapping):
            raise ValueError("state_store must be an object")
        kind = payload.get("kind")
        location = payload.get("location")
        if kind not in {"sqlite", "redis"}:
            raise ValueError("state_store.kind must be 'sqlite' or 'redis'")
        if not isinstance(location, str) or not location.strip():
            raise ValueError("state_store.location must be a non-empty string")
        return cls(kind=kind, location=location)


@runtime_checkable
class StateStore(Protocol):
    def create_checkpoint(self, checkpoint: Checkpoint) -> bool: ...
    def save_checkpoint(self, checkpoint: Checkpoint) -> None: ...
    def load_checkpoint(self, task_id: str) -> Checkpoint | None: ...
    def claim_checkpoint(
        self,
        task_id: str,
        cycle_index: int,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> Checkpoint | None: ...
    def commit_checkpoint(self, checkpoint: Checkpoint, *, claim_token: str, expected_revision: int) -> bool: ...
    def renew_checkpoint_claim(
        self,
        task_id: str,
        *,
        claim_token: str,
        expected_revision: int,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> bool: ...
    def finalize_checkpoint(self, checkpoint: Checkpoint, *, expected_revision: int) -> bool: ...
    def delete_checkpoint(self, task_id: str) -> None: ...
    def acknowledge_terminal(self, task_id: str, *, expected_revision: int) -> bool: ...
    def list_checkpoints(self) -> list[str]: ...
    def state_store_spec(self) -> StateStoreSpec | None: ...


class _LeaseOperationClock:
    """Advance a caller-provided wall-clock snapshot with monotonic elapsed time."""

    def __init__(self, now_ms: int) -> None:
        self._now_ms = now_ms
        self._started_ns = time.monotonic_ns()

    def now_ms(self) -> int:
        elapsed_ms = max(0, time.monotonic_ns() - self._started_ns) // 1_000_000
        return min((1 << 64) - 1, self._now_ms + elapsed_ms)


class InMemoryStateStore:
    """Simple in-memory state store for testing and single-process use."""

    def __init__(self) -> None:
        self._store: dict[str, Checkpoint] = {}
        self._store_v2: dict[str, CheckpointV2] = {}
        self._lock = RLock()

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        from vv_agent.runtime.checkpoint_codec import clone_checkpoint

        snapshot = clone_checkpoint(checkpoint)
        with self._lock:
            self._store[snapshot.task_id] = snapshot

    def create_checkpoint(self, checkpoint: Checkpoint) -> bool:
        from vv_agent.runtime.checkpoint_codec import clone_checkpoint

        snapshot = clone_checkpoint(checkpoint)
        with self._lock:
            if snapshot.task_id in self._store:
                return False
            self._store[snapshot.task_id] = snapshot
            return True

    def load_checkpoint(self, task_id: str) -> Checkpoint | None:
        from vv_agent.runtime.checkpoint_codec import clone_checkpoint

        with self._lock:
            checkpoint = self._store.get(task_id)
            return clone_checkpoint(checkpoint) if checkpoint is not None else None

    def claim_checkpoint(
        self,
        task_id: str,
        cycle_index: int,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> Checkpoint | None:
        from vv_agent.runtime.checkpoint_codec import clone_checkpoint

        _validate_claim(cycle_index, claim_token, lease_expires_at_ms, now_ms)
        with self._lock:
            checkpoint = self._store.get(task_id)
            if checkpoint is None:
                return None
            _check_claim(checkpoint, cycle_index, now_ms)
            checkpoint.revision += 1
            checkpoint.claim_token = claim_token
            checkpoint.claimed_cycle = cycle_index
            checkpoint.lease_expires_at_ms = lease_expires_at_ms
            return clone_checkpoint(checkpoint)

    def commit_checkpoint(self, checkpoint: Checkpoint, *, claim_token: str, expected_revision: int) -> bool:
        from vv_agent.runtime.checkpoint_codec import clone_checkpoint

        with self._lock:
            current = self._store.get(checkpoint.task_id)
            if not _claim_matches(current, checkpoint, claim_token, expected_revision):
                return False
            snapshot = clone_checkpoint(
                replace(
                    checkpoint,
                    claim_token=None,
                    claimed_cycle=None,
                    lease_expires_at_ms=None,
                )
            )
            snapshot.revision = expected_revision + 1
            self._store[snapshot.task_id] = snapshot
            return True

    def renew_checkpoint_claim(
        self,
        task_id: str,
        *,
        claim_token: str,
        expected_revision: int,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> bool:
        _validate_renew(claim_token, expected_revision, lease_expires_at_ms, now_ms)
        clock = _LeaseOperationClock(now_ms)
        with self._lock:
            current_now_ms = clock.now_ms()
            checkpoint = self._store.get(task_id)
            if (
                checkpoint is None
                or checkpoint.revision != expected_revision
                or checkpoint.claim_token != claim_token
                or (checkpoint.lease_expires_at_ms or 0) <= current_now_ms
                or lease_expires_at_ms <= current_now_ms
            ):
                return False
            checkpoint.lease_expires_at_ms = lease_expires_at_ms
            return True

    def finalize_checkpoint(self, checkpoint: Checkpoint, *, expected_revision: int) -> bool:
        from vv_agent.runtime.checkpoint_codec import clone_checkpoint

        snapshot = clone_checkpoint(checkpoint)
        if snapshot.terminal_result is None:
            raise ValueError("finalized checkpoint must include terminal_result")
        with self._lock:
            current = self._store.get(snapshot.task_id)
            if (
                current is None
                or current.revision != expected_revision
                or current.claim_token is not None
                or current.terminal_result is not None
            ):
                return False
            snapshot.revision = expected_revision + 1
            _clear_claim(snapshot)
            self._store[snapshot.task_id] = snapshot
            return True

    def delete_checkpoint(self, task_id: str) -> None:
        with self._lock:
            self._store.pop(task_id, None)

    def acknowledge_terminal(self, task_id: str, *, expected_revision: int) -> bool:
        with self._lock:
            checkpoint = self._store.get(task_id)
            if checkpoint is None or checkpoint.revision != expected_revision or checkpoint.terminal_result is None:
                return False
            del self._store[task_id]
            return True

    def list_checkpoints(self) -> list[str]:
        with self._lock:
            return sorted(self._store.keys())

    def state_store_spec(self) -> StateStoreSpec | None:
        return None

    def save_checkpoint_v2(self, checkpoint: CheckpointV2) -> None:
        from vv_agent.runtime.checkpoint_codec_v2 import clone_checkpoint_v2

        snapshot = clone_checkpoint_v2(checkpoint)
        with self._lock:
            self._store_v2[snapshot.checkpoint_key] = snapshot

    def create_checkpoint_v2(self, checkpoint: CheckpointV2) -> bool:
        from vv_agent.runtime.checkpoint_codec_v2 import clone_checkpoint_v2

        snapshot = clone_checkpoint_v2(checkpoint)
        if (
            snapshot.revision != 0
            or snapshot.resume_attempt != 1
            or snapshot.claim_token is not None
        ):
            raise ValueError("new checkpoint v2 records must be unclaimed at revision zero")
        with self._lock:
            if snapshot.checkpoint_key in self._store_v2:
                return False
            self._store_v2[snapshot.checkpoint_key] = snapshot
            return True

    def load_checkpoint_v2(self, checkpoint_key: str) -> CheckpointV2 | None:
        from vv_agent.runtime.checkpoint_codec_v2 import clone_checkpoint_v2

        with self._lock:
            checkpoint = self._store_v2.get(checkpoint_key)
            return clone_checkpoint_v2(checkpoint) if checkpoint is not None else None

    def claim_checkpoint_v2(
        self,
        checkpoint_key: str,
        cycle_index: int,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
        claim_mode: ClaimMode,
    ) -> CheckpointV2 | None:
        from vv_agent.runtime.checkpoint_codec_v2 import clone_checkpoint_v2

        _validate_claim(cycle_index, claim_token, lease_expires_at_ms, now_ms)
        with self._lock:
            checkpoint = self._store_v2.get(checkpoint_key)
            if checkpoint is None:
                return None
            try:
                check_claim_v2(checkpoint, cycle_index, now_ms, claim_mode)
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
            return clone_checkpoint_v2(checkpoint)

    def progress_checkpoint_v2(
        self,
        checkpoint: CheckpointV2,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        from vv_agent.runtime.checkpoint_codec_v2 import clone_checkpoint_v2

        with self._lock:
            current = self._store_v2.get(checkpoint.checkpoint_key)
            if not claim_matches_v2(current, checkpoint, claim_token, expected_revision):
                return False
            assert current is not None
            if (
                current.terminal_result is not None
                or current.status is not AgentStatus.RUNNING
                or checkpoint.status is not AgentStatus.RUNNING
            ):
                return False
            snapshot = clone_checkpoint_v2(checkpoint)
            snapshot.revision = expected_revision + 1
            snapshot.claim_token = current.claim_token
            snapshot.claimed_cycle = current.claimed_cycle
            snapshot.lease_expires_at_ms = current.lease_expires_at_ms
            self._store_v2[snapshot.checkpoint_key] = snapshot
            return True

    def suspend_checkpoint_v2(
        self,
        checkpoint: CheckpointV2,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        from vv_agent.runtime.checkpoint_codec_v2 import clone_checkpoint_v2

        with self._lock:
            current = self._store_v2.get(checkpoint.checkpoint_key)
            if not claim_matches_v2(current, checkpoint, claim_token, expected_revision):
                return False
            assert current is not None
            if (
                current.terminal_result is not None
                or checkpoint.cycle_index != current.cycle_index
                or checkpoint.status is not AgentStatus.RECONCILIATION_REQUIRED
            ):
                return False
            snapshot = replace(
                checkpoint,
                revision=expected_revision + 1,
                claim_token=None,
                claimed_cycle=None,
                lease_expires_at_ms=None,
            )
            snapshot = clone_checkpoint_v2(snapshot)
            self._store_v2[snapshot.checkpoint_key] = snapshot
            return True

    def commit_checkpoint_v2(
        self,
        checkpoint: CheckpointV2,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        from vv_agent.runtime.checkpoint_codec_v2 import clone_checkpoint_v2

        with self._lock:
            current = self._store_v2.get(checkpoint.checkpoint_key)
            if not claim_matches_v2(current, checkpoint, claim_token, expected_revision):
                return False
            assert current is not None
            if (
                current.terminal_result is not None
                or checkpoint.terminal_result is not None
                or checkpoint.status is not AgentStatus.RUNNING
                or checkpoint.cycle_index != current.claimed_cycle
            ):
                return False
            snapshot = replace(
                checkpoint,
                revision=expected_revision + 1,
                claim_token=None,
                claimed_cycle=None,
                lease_expires_at_ms=None,
                model_call_journal=[],
                tool_journal=[],
            )
            snapshot = clone_checkpoint_v2(snapshot)
            self._store_v2[snapshot.checkpoint_key] = snapshot
            return True

    def finalize_checkpoint_v2(
        self,
        checkpoint: CheckpointV2,
        *,
        expected_revision: int,
    ) -> bool:
        from vv_agent.runtime.checkpoint_codec_v2 import clone_checkpoint_v2

        snapshot = clone_checkpoint_v2(checkpoint)
        if snapshot.terminal_result is None or snapshot.claim_token is not None:
            raise ValueError("finalized checkpoint v2 must be terminal and unclaimed")
        with self._lock:
            current = self._store_v2.get(snapshot.checkpoint_key)
            if (
                current is None
                or current.revision != expected_revision
                or snapshot.revision != expected_revision
                or current.claim_token is not None
                or current.terminal_result is not None
                or not checkpoint_definition_matches_v2(current, snapshot)
            ):
                return False
            snapshot.revision = expected_revision + 1
            self._store_v2[snapshot.checkpoint_key] = snapshot
            return True

    def finalize_claimed_checkpoint_v2(
        self,
        checkpoint: CheckpointV2,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        with self._lock:
            current = self._store_v2.get(checkpoint.checkpoint_key)
            if current is None:
                return False
            terminal = prepare_claimed_terminal_v2(
                current,
                checkpoint,
                claim_token=claim_token,
                expected_revision=expected_revision,
            )
            if terminal is None:
                return False
            self._store_v2[terminal.checkpoint_key] = terminal
            return True

    def record_event_delivery_v2(
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
            current = self._store_v2.get(checkpoint_key)
            if current is None:
                return False
            delivered = prepare_event_delivery_v2(
                current,
                event_id=event_id,
                payload_digest=payload_digest,
                cursor=cursor,
                expected_revision=expected_revision,
                claim_token=claim_token,
            )
            if delivered is None:
                return False
            self._store_v2[checkpoint_key] = delivered
            return True

    def renew_checkpoint_claim_v2(
        self,
        checkpoint_key: str,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> bool:
        _validate_renew(claim_token, 0, lease_expires_at_ms, now_ms)
        clock = _LeaseOperationClock(now_ms)
        with self._lock:
            current_now_ms = clock.now_ms()
            checkpoint = self._store_v2.get(checkpoint_key)
            if (
                checkpoint is None
                or checkpoint.claim_token != claim_token
                or (checkpoint.lease_expires_at_ms or 0) <= current_now_ms
                or lease_expires_at_ms <= current_now_ms
            ):
                return False
            checkpoint.lease_expires_at_ms = lease_expires_at_ms
            return True

    def acknowledge_terminal_v2(self, checkpoint_key: str, *, expected_revision: int) -> bool:
        with self._lock:
            checkpoint = self._store_v2.get(checkpoint_key)
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

    def delete_checkpoint_v2(self, checkpoint_key: str) -> None:
        with self._lock:
            self._store_v2.pop(checkpoint_key, None)

    def list_checkpoints_v2(self) -> list[str]:
        with self._lock:
            return sorted(self._store_v2)


def _validate_claim(cycle_index: int, claim_token: str, lease_expires_at_ms: int, now_ms: int) -> None:
    if isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or not 1 <= cycle_index <= (1 << 32) - 1:
        raise ValueError("claimed cycle_index must be between 1 and 4294967295")
    if not isinstance(claim_token, str) or not claim_token:
        raise ValueError("claim_token must be a non-empty string")
    for value, name in ((lease_expires_at_ms, "lease_expires_at_ms"), (now_ms, "now_ms")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if lease_expires_at_ms <= now_ms:
        raise ValueError("lease_expires_at_ms must be greater than now_ms")


def _validate_renew(claim_token: str, expected_revision: int, lease_expires_at_ms: int, now_ms: int) -> None:
    if not isinstance(claim_token, str) or not claim_token:
        raise ValueError("claim_token must be a non-empty string")
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
        raise ValueError("expected_revision must be a non-negative integer")
    for value, name in ((lease_expires_at_ms, "lease_expires_at_ms"), (now_ms, "now_ms")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if lease_expires_at_ms <= now_ms:
        raise ValueError("lease_expires_at_ms must be greater than now_ms")


def _check_claim(checkpoint: Checkpoint, cycle_index: int, now_ms: int) -> None:
    expected = cycle_index - 1
    if checkpoint.terminal_result is not None or checkpoint.status is not AgentStatus.RUNNING:
        raise CheckpointConflictError(f"checkpoint for task {checkpoint.task_id} is terminal")
    if checkpoint.cycle_index != expected:
        raise CheckpointConflictError(
            f"checkpoint cycle conflict for task {checkpoint.task_id}: expected {expected}, found {checkpoint.cycle_index}"
        )
    if checkpoint.claim_token is not None and (checkpoint.lease_expires_at_ms or 0) > now_ms:
        raise CheckpointConflictError(f"checkpoint cycle {cycle_index} for task {checkpoint.task_id} is already claimed")


def _claim_matches(
    current: Checkpoint | None,
    snapshot: Checkpoint,
    claim_token: str,
    expected_revision: int,
) -> bool:
    return bool(
        current is not None
        and current.revision == expected_revision
        and current.claim_token == claim_token
        and current.claimed_cycle is not None
        and snapshot.cycle_index == current.claimed_cycle
    )


def _clear_claim(checkpoint: Checkpoint) -> None:
    checkpoint.claim_token = None
    checkpoint.claimed_cycle = None
    checkpoint.lease_expires_at_ms = None


def build_state_store(spec: StateStoreSpec) -> StateStore:
    if spec.kind == "sqlite":
        from vv_agent.runtime.stores.sqlite import SqliteStateStore

        return SqliteStateStore(spec.location)
    from vv_agent.runtime.stores.redis import RedisStateStore

    return RedisStateStore(spec.location)
