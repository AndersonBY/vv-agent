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
            self._reopen(state)
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

    def close_if_idle(self, thread_id: str) -> bool:
        with self._lock:
            state = self.load(thread_id)
            if state.subscribers or state.active_turn is not None:
                return False
            state.status = "closed"
            return True

    def subscribe_and_snapshot(self, thread_id: str, connection_id: str, snapshot_fn: Callable[[], T]) -> T:
        with self._lock:
            state = self.load(thread_id)
            previous_status = state.status
            inserted = connection_id not in state.subscribers
            state.subscribers.add(connection_id)
            self._reopen(state)
            try:
                return snapshot_fn()
            except BaseException:
                if inserted:
                    state.subscribers.discard(connection_id)
                state.status = previous_status
                raise

    def reopen(self, thread_id: str) -> None:
        with self._lock:
            self._reopen(self.load(thread_id))

    def set_active_turn(
        self,
        *,
        thread_id: str,
        turn_id: str,
        handle: Any,
        run_id: str | None = None,
    ) -> None:
        with self._lock:
            self.load(thread_id).active_turn = ActiveTurn(
                thread_id=thread_id,
                turn_id=turn_id,
                handle=handle,
                run_id=run_id,
            )

    def clear_active_turn(self, thread_id: str, turn_id: str) -> None:
        with self._lock:
            state = self.load(thread_id)
            if state.active_turn is not None and state.active_turn.turn_id == turn_id:
                state.active_turn = None

    def active_turn(self, thread_id: str) -> ActiveTurn | None:
        with self._lock:
            return self.load(thread_id).active_turn

    @staticmethod
    def _reopen(state: ThreadState) -> None:
        if state.status == "closed":
            state.status = "running" if state.active_turn is not None else "idle"
