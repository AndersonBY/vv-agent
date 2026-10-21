from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class ActiveTurn:
    thread_id: str
    turn_id: str
    handle: Any
    run_id: str | None = None


@dataclass(slots=True)
class ThreadState:
    thread_id: str
    status: str = "idle"
    active_turn: ActiveTurn | None = None
    subscribers: set[str] = field(default_factory=set)
    listener_generation: int = 0
    turn_lock: Any = field(default_factory=RLock, repr=False)


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
            state = self.load(thread_id)
            state.subscribers.add(connection_id)

    def unsubscribe(self, thread_id: str, connection_id: str) -> None:
        with self._lock:
            self.load(thread_id).subscribers.discard(connection_id)

    def unsubscribe_connection(self, connection_id: str) -> None:
        with self._lock:
            for state in self._states.values():
                state.subscribers.discard(connection_id)

    def subscribers(self, thread_id: str) -> set[str]:
        with self._lock:
            return set(self.load(thread_id).subscribers)

    def is_subscribed(self, thread_id: str, connection_id: str) -> bool:
        with self._lock:
            return connection_id in self.load(thread_id).subscribers

    def set_status(self, thread_id: str, status: str) -> None:
        with self._lock:
            self.load(thread_id).status = status

    def subscribe_and_snapshot(self, thread_id: str, connection_id: str, snapshot_fn: Callable[[], T]) -> T:
        with self._lock:
            state = self.load(thread_id)
            previous_status = state.status
            inserted = connection_id not in state.subscribers
            state.subscribers.add(connection_id)
            try:
                return snapshot_fn()
            except BaseException:
                if inserted:
                    state.subscribers.discard(connection_id)
                state.status = previous_status
                raise

    def set_active_turn(
        self,
        *,
        thread_id: str,
        turn_id: str,
        handle: Any,
        run_id: str | None = None,
    ) -> None:
        with self._lock:
            if self.load(thread_id).active_turn is not None:
                raise RuntimeError("Thread already has a local handle")
            self.load(thread_id).active_turn = ActiveTurn(
                thread_id=thread_id,
                turn_id=turn_id,
                handle=handle,
                run_id=run_id,
            )

    def clear_active_turn(self, thread_id: str, turn_id: str, *, handle: Any) -> bool:
        with self._lock:
            state = self.load(thread_id)
            if state.active_turn is not None and state.active_turn.turn_id == turn_id and state.active_turn.handle is handle:
                state.active_turn = None
                return True
            return False

    def turn_lock(self, thread_id: str):
        return self.load(thread_id).turn_lock

    def active_turn(self, thread_id: str) -> ActiveTurn | None:
        with self._lock:
            return self.load(thread_id).active_turn
