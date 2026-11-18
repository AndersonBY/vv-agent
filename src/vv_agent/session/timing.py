"""Injectable session time, renewal scheduling and external invocation."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from vv_agent.kernel._lease_scope import Clock as Clock

if TYPE_CHECKING:
    from vv_agent.runtime.cancellation import CancellationToken

    from .kernel import _LeaseHeartbeat
    from .providers import Outcome
    from .store import SessionStore

HEARTBEAT_CLOSE_SECONDS = 6.0


class SystemClock:
    def now_ms(self) -> int:
        return time.time_ns() // 1_000_000

    def monotonic(self) -> float:
        return time.monotonic()

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class LeaseKeeper(Protocol):
    """Schedule heartbeat.poll(store); start must publish readiness for admission.

    close is idempotent. Ownership, controls and fencing stay in the kernel.
    """

    def start(self, heartbeat: _LeaseHeartbeat) -> None: ...

    def close(self, heartbeat: _LeaseHeartbeat) -> None: ...

    def is_alive(self, heartbeat: _LeaseHeartbeat) -> bool: ...


@dataclass(slots=True)
class _LeaseRegistration:
    heartbeat: _LeaseHeartbeat
    next_due: float
    in_flight: bool = False
    closing: bool = False
    finished: bool = False
    worker: threading.Thread | None = None
    store_context: AbstractContextManager[SessionStore] | None = None
    store: SessionStore | None = None
    close_error: Exception | None = None


class _SharedLeaseKeeper:
    """Multiplex lease renewals on one process-wide scheduler thread."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._registrations: dict[int, _LeaseRegistration] = {}
        self._thread: threading.Thread | None = None

    def start(self, heartbeat: _LeaseHeartbeat) -> None:
        key = id(heartbeat)
        with self._condition:
            registration = self._registrations.get(key)
            if registration is not None:
                return
            registration = _LeaseRegistration(heartbeat, heartbeat.clock.monotonic())
            self._registrations[key] = registration
            registration.worker = threading.Thread(
                target=self._run_worker,
                args=(registration,),
                daemon=True,
                name="session-heartbeat",
            )
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run,
                    daemon=True,
                    name="session-heartbeat-scheduler",
                )
                self._thread.start()
            if heartbeat.stop.is_set():
                registration.closing = True
            registration.worker.start()
            self._condition.notify_all()

    def close(self, heartbeat: _LeaseHeartbeat) -> None:
        key = id(heartbeat)
        registration: _LeaseRegistration | None
        with self._condition:
            registration = self._registrations.get(key)
            if registration is None:
                return
            reentrant = threading.current_thread() is registration.worker
            registration.closing = True
            registration.next_due = float("inf")
            self._condition.notify_all()
        if reentrant:
            return
        deadline = time.monotonic() + HEARTBEAT_CLOSE_SECONDS
        with self._condition:
            while not registration.finished:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._condition.wait(remaining)
        if registration.close_error is not None:
            raise registration.close_error

    def is_alive(self, heartbeat: _LeaseHeartbeat) -> bool:
        with self._condition:
            registration = self._registrations.get(id(heartbeat))
            return registration is not None and (registration.in_flight or registration.closing)

    def _run(self) -> None:
        while True:
            with self._condition:
                if not self._registrations:
                    self._thread = None
                    self._condition.notify_all()
                    return
                active = [registration for registration in self._registrations.values() if not registration.closing]
                if not active:
                    self._thread = None
                    self._condition.notify_all()
                    return
                next_wait: float | None = None
                dispatched = False
                stopped = False
                for registration in active:
                    if registration.in_flight:
                        continue
                    if registration.heartbeat.stop.is_set():
                        registration.closing = True
                        stopped = True
                        continue
                    wait = registration.next_due - registration.heartbeat.clock.monotonic()
                    if wait <= 0:
                        registration.in_flight = True
                        dispatched = True
                    elif next_wait is None or wait < next_wait:
                        next_wait = wait
                if dispatched or stopped:
                    self._condition.notify_all()
                    continue
                self._condition.wait(next_wait)

    def _run_worker(self, registration: _LeaseRegistration) -> None:
        while True:
            with self._condition:
                if registration.finished:
                    return
                while not registration.closing and not registration.in_flight:
                    self._condition.wait()
                finishing = registration.closing
            if finishing:
                self._finish(registration, notify_finished=True)
                return
            try:
                self._renew(registration)
            except Exception as exc:
                with self._condition:
                    if registration.close_error is None:
                        registration.close_error = exc
                    registration.closing = True
                self._finish(registration, notify_finished=False)
                return

    def _renew(self, registration: _LeaseRegistration) -> None:
        heartbeat = registration.heartbeat
        error: Exception | None = None
        try:
            store = registration.store
            if store is None:
                store_context = heartbeat.heartbeat_store()
                store = store_context.__enter__()
                registration.store_context = store_context
                registration.store = store
            heartbeat.heartbeat_once(store)
        except Exception as exc:
            error = exc
        with self._condition:
            if registration.closing or heartbeat.stop.is_set() or heartbeat.lost or error is not None:
                registration.closing = True
        if error is not None:
            heartbeat.heartbeat_failed(error)
        heartbeat.heartbeat_finished()
        finish = False
        with self._condition:
            if registration.closing:
                finish = True
            elif self._registrations.get(id(heartbeat)) is registration:
                registration.next_due = heartbeat.clock.monotonic() + heartbeat.heartbeat_seconds
                registration.in_flight = False
            self._condition.notify_all()
        if finish:
            self._finish(registration, notify_finished=False)

    def _finish(self, registration: _LeaseRegistration, *, notify_finished: bool) -> None:
        store_context = registration.store_context
        registration.store_context = None
        registration.store = None
        error = registration.close_error
        if store_context is not None:
            try:
                store_context.__exit__(None, None, None)
            except Exception as exc:
                error = exc
        try:
            if error is not None:
                registration.heartbeat.heartbeat_failed(error)
            if notify_finished:
                registration.heartbeat.heartbeat_finished()
        finally:
            with self._condition:
                registration.close_error = error
                registration.in_flight = False
                registration.finished = True
                if self._registrations.get(id(registration.heartbeat)) is registration:
                    self._registrations.pop(id(registration.heartbeat), None)
                self._condition.notify_all()


_PROCESS_LEASE_KEEPER = _SharedLeaseKeeper()


class ThreadLeaseKeeper:
    """Share one daemon scheduler across every default runtime in this process."""

    def __init__(self) -> None:
        self._shared = _PROCESS_LEASE_KEEPER

    def start(self, heartbeat: _LeaseHeartbeat) -> None:
        self._shared.start(heartbeat)

    def close(self, heartbeat: _LeaseHeartbeat) -> None:
        self._shared.close(heartbeat)

    def is_alive(self, heartbeat: _LeaseHeartbeat) -> bool:
        return self._shared.is_alive(heartbeat)


class Invoker(Protocol):
    def invoke(
        self,
        callback: Callable[[], Outcome],
        *,
        timeout: float,
        token: CancellationToken,
        cancellation_grace: float,
        clock: Clock,
    ) -> Outcome: ...


class ThreadInvoker:
    def invoke(
        self,
        callback: Callable[[], Outcome],
        *,
        timeout: float,
        token: CancellationToken,
        cancellation_grace: float,
        clock: Clock,
    ) -> Outcome:
        from .providers import Unknown

        done = threading.Event()
        result: list[Outcome] = []

        def run() -> None:
            try:
                result.append(callback())
            except Exception as exc:
                result.append(Unknown(f"{type(exc).__name__}: external outcome unavailable"))
            finally:
                done.set()

        worker = threading.Thread(target=run, daemon=True, name="session-external-call")
        worker.start()
        deadline = clock.monotonic() + timeout
        while not done.wait(0.02):
            if token.cancelled:
                if done.wait(cancellation_grace):
                    break
                return Unknown("cancellation without confirmed external stop")
            if clock.monotonic() >= deadline:
                return Unknown("external call timeout")
        worker.join()
        return result[0] if result else Unknown("external worker exited without a receipt")
