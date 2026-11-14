"""Injectable session time, renewal scheduling and external invocation."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from vv_agent.kernel._lease_scope import Clock as Clock

if TYPE_CHECKING:
    from vv_agent.runtime.cancellation import CancellationToken

    from .kernel import _LeaseHeartbeat
    from .providers import Outcome

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


class ThreadLeaseKeeper:
    def start(self, heartbeat: _LeaseHeartbeat) -> None:
        heartbeat.thread.start()

    def close(self, heartbeat: _LeaseHeartbeat) -> None:
        heartbeat.thread.join(timeout=HEARTBEAT_CLOSE_SECONDS)

    def is_alive(self, heartbeat: _LeaseHeartbeat) -> bool:
        return heartbeat.thread.is_alive()


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
