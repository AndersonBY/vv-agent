"""Lease-fenced drive lifecycle with host-supplied construction and effects."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, ExitStack, suppress
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from vv_agent.kernel._lease_scope import Clock, Signal
from vv_agent.kernel._sql import SQLStore
from vv_agent.kernel._store import Lease, LeaseLost, SequenceConflict, SessionStore, StoredRecord

FACTORY_LEASE_TTL_MS = 15000
FACTORY_HEARTBEAT_SECONDS = 0.25


class DriveRuntime(Protocol):
    ttl_ms: int
    poll_ms: int


class DriveHeartbeat(Protocol):
    ready: Signal
    lost: bool
    successful_ready: bool
    heartbeat_error: Exception | None
    lease: Lease

    def start(self) -> None: ...

    def close(self) -> None: ...


class DriveLeaseScope(DriveHeartbeat, Protocol):
    def committing(self) -> AbstractContextManager[Lease]: ...


class DriveStep(Protocol):
    records: Sequence[StoredRecord]

    def step(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class DriveBinding:
    """Objects and callbacks external to the core algorithm."""

    clock: Clock
    ttl_ms: int
    runtime: DriveRuntime | None
    make_factory_heartbeat: Callable[[Lease, int], DriveHeartbeat] | None
    make_runtime: Callable[[], DriveRuntime] | None
    factory_retry_after_ms: Callable[[Exception], int | None]
    heartbeat_alive: Callable[[DriveHeartbeat], bool]
    make_scope: Callable[[Lease, DriveRuntime], DriveLeaseScope]
    make_driver: Callable[[SessionStore, str, DriveRuntime, DriveLeaseScope], DriveStep]
    after_drive: Callable[[DriveRuntime], None]
    wake_if_runnable: Callable[[DriveRuntime], None]


def _defer_after_error(store: SessionStore, lease: Lease, retry_after_ms: int, error: Exception) -> None:
    try:
        store.defer_drive(lease, retry_after_ms=retry_after_ms)
    except LeaseLost:
        pass
    except Exception as backoff_error:
        raise error from backoff_error


def drive_core(
    store: SessionStore,
    session_id: str,
    *,
    failure_backoff_ms: int = 1000,
    one_turn: bool = False,
    wait_for_lease: bool = False,
    make_binding: Callable[[], DriveBinding],
) -> bool:
    """Acquire, drive and release one session using an explicitly bound host."""
    if type(failure_backoff_ms) is not int or failure_backoff_ms <= 0:
        raise ValueError("failure_backoff_ms must be a positive integer")
    binding = make_binding()
    ttl_ms = binding.ttl_ms
    lease = store.acquire(session_id, owner=uuid4().hex, ttl_ms=ttl_ms)
    delay = 0.05
    while lease is None and wait_for_lease:
        binding.clock.sleep(delay)
        delay = min(1.0, delay * 2)
        lease = store.acquire(session_id, owner=uuid4().hex, ttl_ms=ttl_ms)
    if lease is None:
        return False

    runtime = binding.runtime
    scheduled = binding.make_runtime is not None
    if scheduled:
        build = None
        try:
            make_heartbeat = binding.make_factory_heartbeat
            make_runtime = binding.make_runtime
            assert make_heartbeat is not None and make_runtime is not None and runtime is None
            build = make_heartbeat(lease, ttl_ms)
            build.start()
            try:
                if not build.ready.wait(ttl_ms / 1000):
                    raise LeaseLost("factory heartbeat did not become ready") from build.heartbeat_error
                if build.lost or not build.successful_ready:
                    raise LeaseLost("factory heartbeat lost ownership") from build.heartbeat_error
                runtime = make_runtime()
            finally:
                build.close()
                lease = build.lease
            if build.lost:
                raise LeaseLost("factory heartbeat lost ownership") from build.heartbeat_error
            lease = store.renew(lease, ttl_ms=runtime.ttl_ms).lease
        except Exception as exc:
            retry_after_ms = binding.factory_retry_after_ms(exc)
            if retry_after_ms is not None:
                try:
                    _defer_after_error(store, lease, retry_after_ms, exc)
                finally:
                    store.release(lease)
                return True
            if build is not None and binding.heartbeat_alive(build):
                raise
            try:
                _defer_after_error(store, lease, failure_backoff_ms, exc)
            finally:
                store.release(lease)
            raise
        except BaseException:
            if build is None or not binding.heartbeat_alive(build):
                with suppress(Exception):
                    store.release(lease)
            raise
    assert runtime is not None

    scope = binding.make_scope(lease, runtime)
    scope.start()
    try:
        driver = binding.make_driver(store, session_id, runtime, scope)
        while not scope.lost:
            try:
                if not driver.step():
                    with ExitStack() as latch:
                        if isinstance(store, SQLStore):
                            store._defer_idle_drive(
                                scope.lease,
                                poll_ms=runtime.poll_ms,
                                lease_guard=lambda: latch.enter_context(scope.committing()),
                            )
                        else:
                            store.defer_idle_drive(latch.enter_context(scope.committing()), poll_ms=runtime.poll_ms)
                    break
                if one_turn and driver.records[-1].record.kind == "turn_ended":
                    break
            except SequenceConflict:
                continue
        if scope.lost:
            raise LeaseLost("heartbeat lost ownership") from scope.heartbeat_error
    except Exception as exc:
        scope.close()
        if scheduled:
            _defer_after_error(store, scope.lease, failure_backoff_ms, exc)
        raise
    finally:
        scope.close()
        with suppress(Exception):
            store.release(scope.lease)
        with suppress(Exception):
            binding.after_drive(runtime)
        with suppress(Exception):
            if store.is_runnable(session_id):
                binding.wake_if_runnable(runtime)
    return True
