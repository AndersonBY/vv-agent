from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from v_agent.types import AgentStatus, CycleRecord, Message


@dataclass(slots=True)
class Checkpoint:
    task_id: str
    cycle_index: int
    status: AgentStatus
    messages: list[Message]
    cycles: list[CycleRecord]
    shared_state: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class StateStore(Protocol):
    def save_checkpoint(self, checkpoint: Checkpoint) -> None: ...
    def load_checkpoint(self, task_id: str) -> Checkpoint | None: ...
    def delete_checkpoint(self, task_id: str) -> None: ...
    def list_checkpoints(self) -> list[str]: ...


class InMemoryStateStore:
    """Simple in-memory state store for testing and single-process use."""

    def __init__(self) -> None:
        self._store: dict[str, Checkpoint] = {}

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._store[checkpoint.task_id] = checkpoint

    def load_checkpoint(self, task_id: str) -> Checkpoint | None:
        return self._store.get(task_id)

    def delete_checkpoint(self, task_id: str) -> None:
        self._store.pop(task_id, None)

    def list_checkpoints(self) -> list[str]:
        return sorted(self._store.keys())
