"""Lease-fenced drive lifecycle with host-supplied construction and effects."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, ExitStack, suppress
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from vv_agent.kernel._content import KeyUnavailable
from vv_agent.kernel._lease_scope import Clock, Signal
from vv_agent.kernel._sql import SQLStore, UnsupportedFeatureLevel, UnsupportedWorkerCapabilities
from vv_agent.kernel._store import (
    Lease,
    LeaseLost,
    LeasePoll,
    ObjectPublicationBusy,
    SequenceConflict,
    SessionStore,
    StoredRecord,
)

FACTORY_LEASE_TTL_MS = 60000
FACTORY_HEARTBEAT_SECONDS = 0.25
_TYPED_HEARTBEAT_FAILURES = (UnsupportedFeatureLevel, UnsupportedWorkerCapabilities)


class DriveRuntimeCore(Protocol):
    ttl_ms: int
    poll_ms: int


class DriveRuntime(DriveRuntimeCore, Protocol):
    _key_unavailable: bool


class DriveHeartbeat(Protocol):
    clock: Clock
    ready: Signal
    lost: bool
    successful_ready: bool
    heartbeat_error: Exception | None
    lease: Lease
    latest_poll: LeasePoll | None
    expires_deadline: float

    def start(self) -> None: ...

    def close(self) -> None: ...

    def adopt_confirmed(self, poll: LeasePoll, *, expires_deadline: float) -> None: ...


class DriveLeaseScope(DriveHeartbeat, Protocol):
    def committing(self) -> AbstractContextManager[Lease]: ...


class DriveStep(Protocol):
    records: Sequence[StoredRecord]

    def step(self) -> bool: ...


def _set_key_unavailable(runtime: DriveRuntimeCore, unavailable: bool) -> None:
    with suppress(AttributeError):
        setattr(runtime, "_key_unavailable", unavailable)  # noqa: B010 - third-party runtimes may omit writable state.


def _consume_key_unavailable(runtime: DriveRuntimeCore) -> bool:
    unavailable = bool(getattr(runtime, "_key_unavailable", False))
    if unavailable:
        _set_key_unavailable(runtime, False)
    return unavailable


@dataclass(frozen=True, slots=True)
class DriveBinding:
    """Objects and callbacks external to the core algorithm."""

    clock: Clock
    ttl_ms: int
    runtime: DriveRuntimeCore | None
    make_factory_heartbeat: Callable[[Lease, int], DriveHeartbeat] | None
    make_runtime: Callable[[], DriveRuntimeCore] | None
    factory_retry_after_ms: Callable[[Exception], int | None]
    make_scope: Callable[[Lease, DriveRuntimeCore], DriveLeaseScope]
    make_driver: Callable[[SessionStore, str, DriveRuntimeCore, DriveLeaseScope], DriveStep]
    after_drive: Callable[[DriveRuntimeCore], None]
    wake_if_runnable: Callable[[DriveRuntimeCore], None]


def _defer_after_error(store: SessionStore, lease: Lease, retry_after_ms: int, error: Exception) -> None:
    try:
        store.defer_drive(lease, retry_after_ms=retry_after_ms)
    except LeaseLost:
        pass
    except Exception as backoff_error:
        raise error from backoff_error


def _typed_heartbeat_failure(heartbeat: DriveHeartbeat) -> Exception | None:
    error = heartbeat.heartbeat_error
    return error if isinstance(error, _TYPED_HEARTBEAT_FAILURES) else None


def _is_typed_heartbeat_failure(error: BaseException | None) -> bool:
    return isinstance(error, _TYPED_HEARTBEAT_FAILURES)


def drive_core(
    store: SessionStore,
    session_id: str,
    *,
    failure_backoff_ms: int = 1000,
    one_turn: bool = False,
    wait_for_lease: bool = False,
    lease: Lease | None = None,
    make_binding: Callable[[], DriveBinding],
) -> bool:
    """Drive one session with an existing claim or a directly acquired lease."""
    if type(failure_backoff_ms) is not int or failure_backoff_ms <= 0:
        raise ValueError("failure_backoff_ms must be a positive integer")
    store_tenant_id = getattr(store, "tenant_id", lease.tenant_id if lease is not None else None)
    if lease is not None and (lease.session_id != session_id or lease.tenant_id != store_tenant_id):
        raise ValueError("claimed lease does not belong to the requested session and tenant")
    try:
        binding = make_binding()
    except BaseException:
        if lease is not None:
            with suppress(LeaseLost):
                store.release(lease)
        raise
    ttl_ms = binding.ttl_ms
    if lease is None:
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
    handoff_poll: LeasePoll | None = None
    handoff_clock: Clock | None = None
    handoff_deadline: float | None = None
    if scheduled:
        build = None
        scheduled_error: BaseException | None = None
        scheduled_error_cause: BaseException | None = None
        try:
            make_heartbeat = binding.make_factory_heartbeat
            make_runtime = binding.make_runtime
            assert make_heartbeat is not None and make_runtime is not None and runtime is None
            build = make_heartbeat(lease, ttl_ms)
            build.start()
            if not build.ready.wait(ttl_ms / 1000):
                typed_failure = _typed_heartbeat_failure(build)
                if typed_failure is not None:
                    raise typed_failure
                raise LeaseLost("factory heartbeat did not become ready") from build.heartbeat_error
            if build.lost or not build.successful_ready:
                typed_failure = _typed_heartbeat_failure(build)
                if typed_failure is not None:
                    raise typed_failure
                raise LeaseLost("factory heartbeat lost ownership") from build.heartbeat_error
            runtime = make_runtime()
        except BaseException as exc:
            scheduled_error = exc
        finally:
            if build is not None:
                try:
                    build.close()
                except BaseException:
                    if scheduled_error is not None:
                        raise scheduled_error from None
                    raise
                lease = build.lease

        if build is not None and build.lost and scheduled_error is None:
            scheduled_error = _typed_heartbeat_failure(build) or LeaseLost("factory heartbeat lost ownership")
            scheduled_error_cause = build.heartbeat_error
        if scheduled_error is not None:
            retry_after_ms = binding.factory_retry_after_ms(scheduled_error) if isinstance(scheduled_error, Exception) else None
            try:
                if isinstance(scheduled_error, Exception) and not _is_typed_heartbeat_failure(scheduled_error):
                    _defer_after_error(
                        store,
                        lease,
                        retry_after_ms if retry_after_ms is not None else failure_backoff_ms,
                        scheduled_error,
                    )
            finally:
                store.release(lease)
            if retry_after_ms is not None:
                return True
            raise scheduled_error from scheduled_error_cause

        if build is None:
            raise LeaseLost("factory heartbeat was not constructed")
        handoff_poll = build.latest_poll
        if handoff_poll is None:
            try:
                _defer_after_error(
                    store,
                    lease,
                    failure_backoff_ms,
                    LeaseLost("factory heartbeat did not publish a confirmed lease"),
                )
            finally:
                store.release(lease)
            raise LeaseLost("factory heartbeat did not publish a confirmed lease")
        handoff_clock = build.clock
        handoff_deadline = build.expires_deadline
        lease = handoff_poll.lease
    assert runtime is not None

    scope = None
    scope_close_attempted = False
    scope_close_succeeded = False
    try:
        scope = binding.make_scope(lease, runtime)
        if handoff_poll is not None and handoff_clock is not None and handoff_deadline is not None:
            # Read the destination clock first. Any time spent converting is then
            # charged by the second, source-clock read rather than extending the lease.
            runtime_now = scope.clock.monotonic()
            factory_now = handoff_clock.monotonic()
            scope.adopt_confirmed(
                handoff_poll,
                expires_deadline=runtime_now + (handoff_deadline - factory_now),
            )
        scope.start()
        if not scope.ready.wait(ttl_ms / 1000):
            typed_failure = _typed_heartbeat_failure(scope)
            if typed_failure is not None:
                raise typed_failure
            raise LeaseLost("runtime heartbeat did not become ready") from scope.heartbeat_error
        if scope.lost or not scope.successful_ready:
            typed_failure = _typed_heartbeat_failure(scope)
            if typed_failure is not None:
                raise typed_failure
            raise LeaseLost("runtime heartbeat lost ownership") from scope.heartbeat_error
        driver = binding.make_driver(store, session_id, runtime, scope)
        while not scope.lost:
            _set_key_unavailable(runtime, False)
            try:
                if isinstance(store, SQLStore):
                    store.next_drive_delay_ms(session_id, lease=scope.lease)
                if not driver.step():
                    key_unavailable = _consume_key_unavailable(runtime)
                    with ExitStack() as latch:
                        if key_unavailable:
                            store.defer_drive(
                                latch.enter_context(scope.committing()),
                                retry_after_ms=failure_backoff_ms,
                            )
                        elif isinstance(store, SQLStore):
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
            typed_failure = _typed_heartbeat_failure(scope)
            if typed_failure is not None:
                raise typed_failure
            raise LeaseLost("heartbeat lost ownership") from scope.heartbeat_error
        scope_close_attempted = True
        scope.close()
        scope_close_succeeded = True
    except BaseException as exc:
        if scope is not None and not scope_close_attempted:
            scope_close_attempted = True
            try:
                scope.close()
            except BaseException:
                pass
            else:
                scope_close_succeeded = True
        key_unavailable = isinstance(exc, KeyUnavailable)
        if (
            (scheduled or key_unavailable or isinstance(exc, ObjectPublicationBusy))
            and (scope is None or scope_close_succeeded)
            and isinstance(exc, Exception)
            and not _is_typed_heartbeat_failure(exc)
        ):
            _defer_after_error(store, scope.lease if scope is not None else lease, failure_backoff_ms, exc)
            if key_unavailable:
                return True
        raise
    finally:
        if scope is None or scope_close_succeeded:
            with suppress(LeaseLost):
                store.release(scope.lease if scope is not None else lease)
        with suppress(Exception):
            binding.after_drive(runtime)
        with suppress(Exception):
            if store.is_runnable(session_id):
                binding.wake_if_runnable(runtime)
    return True
