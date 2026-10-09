from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vv_agent.app_server.protocol import ThreadItem

if TYPE_CHECKING:
    from vv_agent.session.app_server import _KernelThreadStore


@dataclass(frozen=True, slots=True)
class ThreadRecord:
    thread_id: str
    agent_key: str
    cwd: str | None = None
    created_at: float = 0
    updated_at: float = 0
    archived_at: float | None = None
    status: str = "idle"
    active_turn_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TurnRecord:
    turn_id: str
    thread_id: str
    run_id: str | None = None
    status: str = "running"
    started_at: float = 0
    completed_at: float | None = None
    input: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ThreadSnapshot:
    thread: ThreadRecord
    turns: list[TurnRecord]
    items: list[ThreadItem]


class ThreadStore:
    def __new__(cls, store=None) -> _KernelThreadStore:
        from vv_agent.session.app_server import _KernelThreadStore
        from vv_agent.session.surfaces import SessionDriver

        return _KernelThreadStore(SessionDriver(store=store))
