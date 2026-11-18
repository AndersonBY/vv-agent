"""SDK construction for the shared lease-fenced drive lifecycle."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

from vv_agent.kernel._content import KeyUnavailable, SessionHead
from vv_agent.kernel._content_runtime import (
    _ContentSessionStoreAdapter,
    content_runtime_binding_for,
    durable_content_reader,
    wrap_session_content_store,
)
from vv_agent.kernel._drive import (
    FACTORY_HEARTBEAT_SECONDS,
    FACTORY_LEASE_TTL_MS,
    DriveBinding,
    DriveLeaseScope,
    DriveRuntimeCore,
    DriveStep,
    drive_core,
)

from ._driver_scope import _LeaseHeartbeat, _Scope
from .runtime import Runtime, RuntimeNotReady
from .store import Lease, LeaseLost, SessionStore
from .timing import Clock, LeaseKeeper, SystemClock, ThreadLeaseKeeper


class _KeyUnavailableTrackingReader:
    def __init__(self, reader, runtime: Runtime) -> None:
        self.reader = reader
        self.runtime = runtime

    def read(self, tenant_id: str, session_id: str, owner: str, slot: str, ref):
        try:
            return self.reader.read(tenant_id, session_id, owner, slot, ref)
        except KeyUnavailable:
            self.runtime._key_unavailable = True
            raise


def _lease_anchor_now_ms(store: SessionStore, session_id: str, lease: Lease, clock: Clock) -> int:
    """Return an inferred database now whose local deadline charges the status read."""

    started = clock.monotonic()
    _head, epoch, remaining_ms = store.lease_status(session_id)
    finished = clock.monotonic()
    if epoch != lease.epoch:
        raise LeaseLost("lease epoch changed while anchoring heartbeat budget")
    remaining_ms -= max(0, math.ceil((finished - started) * 1000))
    if remaining_ms <= 0:
        raise LeaseLost("lease budget expired while anchoring heartbeat")
    # LeaseOwnership converts database-now into its local monotonic domain. Using
    # the DB remaining budget, reduced by the measured anchor round trip, keeps
    # transient failures fail-closed without granting another full TTL.
    return lease.expires_at_ms - remaining_ms


def drive_with_runtime(
    store: SessionStore,
    session_id: str,
    *,
    runtime: Runtime | Callable[[str], Runtime],
    failure_backoff_ms: int = 1000,
    expected_head: SessionHead | tuple[int, str] | None = None,
    one_turn: bool = False,
    wait_for_lease: bool = False,
    lease: Lease | None = None,
    initial_polled: set[tuple[str, int]] | None = None,
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
            heartbeat_factory = store.heartbeat_store_factory()
            now_ms = _lease_anchor_now_ms(store, session_id, lease, selected_clock)
            return _LeaseHeartbeat(
                lease,
                heartbeat_factory,
                ttl_ms,
                FACTORY_HEARTBEAT_SECONDS,
                clock=selected_clock,
                lease_keeper=selected_keeper,
                now_ms=now_ms,
            )

        def make_scope(lease: Lease, selected: DriveRuntimeCore) -> _Scope:
            selected_runtime = cast(Runtime, selected)
            return _Scope(
                lease,
                selected_runtime,
                now_ms=(_lease_anchor_now_ms(store, session_id, lease, selected_runtime.clock) if direct else None),
            )

        def make_driver(
            selected_store: SessionStore, selected_session_id: str, selected: DriveRuntimeCore, scope: DriveLeaseScope
        ) -> DriveStep:
            from .kernel import _Driver

            selected_runtime = cast(Runtime, selected)
            selected_runtime._key_unavailable = False
            binding = content_runtime_binding_for(selected_store)
            if binding is not None:
                if not isinstance(selected_store, _ContentSessionStoreAdapter):
                    selected_store = wrap_session_content_store(selected_store, binding)
                if selected_runtime.content_reader is None:
                    tenant_id = getattr(selected_store, "tenant_id", None)
                    if not isinstance(tenant_id, str) or not tenant_id:
                        raise ValueError("content-bound store requires a non-empty tenant binding")
                    selected_runtime.content_reader = durable_content_reader(
                        selected_store,
                        tenant_id=tenant_id,
                        session_id=selected_session_id,
                    )
            if selected_runtime.content_reader is not None and not isinstance(
                selected_runtime.content_reader,
                _KeyUnavailableTrackingReader,
            ):
                selected_runtime.content_reader = _KeyUnavailableTrackingReader(
                    selected_runtime.content_reader,
                    selected_runtime,
                )
            driver = _Driver(
                selected_store,
                selected_session_id,
                selected_runtime,
                cast(_Scope, scope),
                one_turn=one_turn,
                expected_head=expected_head,
            )
            driver.polled.update(initial_polled or ())
            return cast(DriveStep, driver)

        def after_drive(selected: DriveRuntimeCore) -> None:
            cast(Runtime, selected).hook("after_drive", None)

        def wake_if_runnable(selected: DriveRuntimeCore) -> None:
            cast(Runtime, selected).wake(session_id)

        def factory_retry_after_ms(error: Exception) -> int | None:
            if isinstance(error, RuntimeNotReady):
                return error.retry_after_ms
            return failure_backoff_ms if isinstance(error, KeyUnavailable) else None

        return DriveBinding(
            clock=selected_clock,
            ttl_ms=runtime.ttl_ms if direct else FACTORY_LEASE_TTL_MS,
            runtime=runtime if direct else None,
            make_factory_heartbeat=None if direct else make_factory_heartbeat,
            make_runtime=None if direct else lambda: runtime(session_id),
            factory_retry_after_ms=factory_retry_after_ms,
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
        lease=lease,
        make_binding=make_binding,
    )
