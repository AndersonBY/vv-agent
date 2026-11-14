"""SDK construction for the shared lease-fenced drive lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from vv_agent.kernel._drive import (
    FACTORY_HEARTBEAT_SECONDS,
    FACTORY_LEASE_TTL_MS,
    DriveBinding,
    DriveHeartbeat,
    DriveLeaseScope,
    DriveRuntime,
    DriveStep,
    drive_core,
)
from vv_agent.kernel._sql import SQLStore

from ._driver_scope import _LeaseHeartbeat, _Scope
from .runtime import Runtime, RuntimeNotReady
from .store import Lease, SessionStore
from .timing import Clock, LeaseKeeper, SystemClock, ThreadLeaseKeeper


def drive_with_runtime(
    store: SessionStore,
    session_id: str,
    *,
    runtime: Runtime | Callable[[str], Runtime],
    failure_backoff_ms: int = 1000,
    one_turn: bool = False,
    wait_for_lease: bool = False,
    clock: Clock | None = None,
    lease_keeper: LeaseKeeper | None = None,
) -> bool:
    """Bind SDK Runtime, synchronization and the real Driver to the core lifecycle."""

    def make_binding() -> DriveBinding:
        direct = isinstance(runtime, Runtime)
        selected_clock = clock if clock is not None else runtime.clock if direct else SystemClock()
        selected_keeper = lease_keeper if lease_keeper is not None else runtime.lease_keeper if direct else ThreadLeaseKeeper()

        def make_factory_heartbeat(lease: Lease, ttl_ms: int) -> _LeaseHeartbeat:
            # Resolve store setup on the driving thread (host connections can be thread-bound).
            heartbeat = store.heartbeat_store()
            return _LeaseHeartbeat(
                lease,
                lambda: heartbeat,
                ttl_ms,
                FACTORY_HEARTBEAT_SECONDS,
                clock=selected_clock,
                lease_keeper=selected_keeper,
                now_ms=store._now() if isinstance(store, SQLStore) else None,
            )

        def make_scope(lease: Lease, selected: DriveRuntime) -> _Scope:
            return _Scope(lease, cast(Runtime, selected), now_ms=store._now() if isinstance(store, SQLStore) else None)

        def make_driver(
            selected_store: SessionStore, selected_session_id: str, selected: DriveRuntime, scope: DriveLeaseScope
        ) -> DriveStep:
            from .kernel import _Driver

            return cast(
                DriveStep,
                _Driver(
                    selected_store,
                    selected_session_id,
                    cast(Runtime, selected),
                    cast(_Scope, scope),
                    one_turn=one_turn,
                ),
            )

        def heartbeat_alive(heartbeat: DriveHeartbeat) -> bool:
            return selected_keeper.is_alive(cast(_LeaseHeartbeat, heartbeat))

        def after_drive(selected: DriveRuntime) -> None:
            cast(Runtime, selected).hook("after_drive", None)

        def wake_if_runnable(selected: DriveRuntime) -> None:
            cast(Runtime, selected).wake(session_id)

        def factory_retry_after_ms(error: Exception) -> int | None:
            return error.retry_after_ms if isinstance(error, RuntimeNotReady) else None

        return DriveBinding(
            clock=selected_clock,
            ttl_ms=runtime.ttl_ms if direct else FACTORY_LEASE_TTL_MS,
            runtime=runtime if direct else None,
            make_factory_heartbeat=None if direct else make_factory_heartbeat,
            make_runtime=None if direct else lambda: runtime(session_id),
            factory_retry_after_ms=factory_retry_after_ms,
            heartbeat_alive=heartbeat_alive,
            make_scope=make_scope,
            make_driver=make_driver,
            after_drive=after_drive,
            wake_if_runnable=wake_if_runnable,
        )

    return drive_core(
        store,
        session_id,
        failure_backoff_ms=failure_backoff_ms,
        one_turn=one_turn,
        wait_for_lease=wait_for_lease,
        make_binding=make_binding,
    )
