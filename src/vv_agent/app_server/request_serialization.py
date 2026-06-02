from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

RequestAccess = Literal["exclusive", "shared_read"]
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RequestScope:
    key: str

    @classmethod
    def thread(cls, thread_id: str) -> RequestScope:
        return cls(key=f"thread:{thread_id}")

    @classmethod
    def thread_read(cls, thread_id: str) -> RequestScope:
        return cls(key=f"thread:{thread_id}")

    @classmethod
    def global_scope(cls, name: str) -> RequestScope:
        return cls(key=f"global:{name}")

    @classmethod
    def global_read(cls, name: str) -> RequestScope:
        return cls(key=f"global:{name}")

    @classmethod
    def server_request(cls, request_id: str) -> RequestScope:
        return cls(key=f"server-request:{request_id}")


@dataclass(slots=True)
class _QueuedWork:
    access: RequestAccess
    fn: Callable[[], Any]
    future: Future[Any]


@dataclass(slots=True)
class _ScopeState:
    queue: deque[_QueuedWork] = field(default_factory=deque)
    running_exclusive: bool = False
    running_shared_reads: int = 0


class RequestSerializationQueues:
    def __init__(self) -> None:
        self._states: dict[str, _ScopeState] = {}
        self._lock = threading.Lock()

    def enqueue(self, *, key: RequestScope, access: RequestAccess, fn: Callable[[], T]) -> Future[T]:
        if access not in {"exclusive", "shared_read"}:
            raise ValueError(f"Unknown request access: {access}")
        future: Future[T] = Future()
        work = _QueuedWork(access=access, fn=fn, future=future)
        with self._lock:
            state = self._states.setdefault(key.key, _ScopeState())
            state.queue.append(work)
            self._drain_locked(key.key, state)
        return future

    def _drain_locked(self, scope_key: str, state: _ScopeState) -> None:
        if state.running_exclusive:
            return
        if state.running_shared_reads > 0:
            while state.queue and state.queue[0].access == "shared_read":
                self._start_shared_read_locked(scope_key, state, state.queue.popleft())
            return
        if not state.queue:
            self._states.pop(scope_key, None)
            return
        if state.queue[0].access == "exclusive":
            self._start_exclusive_locked(scope_key, state, state.queue.popleft())
            return
        while state.queue and state.queue[0].access == "shared_read":
            self._start_shared_read_locked(scope_key, state, state.queue.popleft())

    def _start_exclusive_locked(self, scope_key: str, state: _ScopeState, work: _QueuedWork) -> None:
        state.running_exclusive = True
        self._start_worker(scope_key, work)

    def _start_shared_read_locked(self, scope_key: str, state: _ScopeState, work: _QueuedWork) -> None:
        state.running_shared_reads += 1
        self._start_worker(scope_key, work)

    def _start_worker(self, scope_key: str, work: _QueuedWork) -> None:
        thread = threading.Thread(target=self._run_work, args=(scope_key, work), daemon=True)
        thread.start()

    def _run_work(self, scope_key: str, work: _QueuedWork) -> None:
        try:
            result = work.fn()
        except BaseException as exc:
            work.future.set_exception(exc)
        else:
            work.future.set_result(result)
        finally:
            with self._lock:
                state = self._states.get(scope_key)
                if state is not None:
                    if work.access == "exclusive":
                        state.running_exclusive = False
                    else:
                        state.running_shared_reads -= 1
                    self._drain_locked(scope_key, state)
