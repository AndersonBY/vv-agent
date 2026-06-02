from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass(slots=True)
class ActiveTurn:
    thread_id: str
    turn_id: str
    handle: Any


@dataclass(slots=True)
class ThreadState:
    thread_id: str
    active_turn: ActiveTurn | None = None
    subscribers: set[str] = field(default_factory=set)
    pending_steering: list[list[dict[str, Any]]] = field(default_factory=list)
    pending_follow_ups: list[list[dict[str, Any]]] = field(default_factory=list)


class ThreadStateManager:
    def __init__(self) -> None:
        self._states: dict[str, ThreadState] = {}
        self._lock = RLock()

    def load(self, thread_id: str) -> ThreadState:
        with self._lock:
            state = self._states.get(thread_id)
            if state is None:
                state = ThreadState(thread_id=thread_id)
                self._states[thread_id] = state
            return state

    def subscribe(self, thread_id: str, connection_id: str) -> None:
        with self._lock:
            self.load(thread_id).subscribers.add(connection_id)

    def unsubscribe(self, thread_id: str, connection_id: str) -> None:
        with self._lock:
            self.load(thread_id).subscribers.discard(connection_id)

    def set_active_turn(self, *, thread_id: str, turn_id: str, handle: Any) -> None:
        with self._lock:
            self.load(thread_id).active_turn = ActiveTurn(thread_id=thread_id, turn_id=turn_id, handle=handle)

    def clear_active_turn(self, thread_id: str, turn_id: str) -> None:
        with self._lock:
            state = self.load(thread_id)
            if state.active_turn is not None and state.active_turn.turn_id == turn_id:
                state.active_turn = None

    def active_turn(self, thread_id: str) -> ActiveTurn | None:
        with self._lock:
            return self.load(thread_id).active_turn

    def queue_steering(self, thread_id: str, input: list[dict[str, Any]]) -> None:
        with self._lock:
            self.load(thread_id).pending_steering.append([dict(item) for item in input])

    def drain_steering(self, thread_id: str) -> list[list[dict[str, Any]]]:
        with self._lock:
            state = self.load(thread_id)
            queued = list(state.pending_steering)
            state.pending_steering.clear()
            return queued

    def queue_follow_up(self, thread_id: str, input: list[dict[str, Any]]) -> None:
        with self._lock:
            self.load(thread_id).pending_follow_ups.append([dict(item) for item in input])

    def pop_next_follow_up(self, thread_id: str) -> list[dict[str, Any]] | None:
        with self._lock:
            state = self.load(thread_id)
            if not state.pending_follow_ups:
                return None
            return state.pending_follow_ups.pop(0)
