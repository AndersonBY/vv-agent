from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Any, Literal, Protocol, runtime_checkable

from vv_agent.budget import BudgetUsageSnapshot
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
