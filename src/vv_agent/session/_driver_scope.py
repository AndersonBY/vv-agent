"""Lease ownership, heartbeat scheduling and the shared invocation boundary."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

from vv_agent.runtime.cancellation import CancellationToken

from .postgres import PostgresStore
from .providers import Outcome
from .runtime import Runtime
from .store import Lease, LeaseLost, SessionStore
from .timing import Clock, LeaseKeeper, SystemClock, ThreadLeaseKeeper


class HeartbeatStopped(LeaseLost):
    """Normal stop of a lease-renewal worker; grants no execution ownership."""


class _LeaseHeartbeat:
    def __init__(
        self,
        lease: Lease,
        heartbeat_store: Callable[[], AbstractContextManager[SessionStore]],
        ttl_ms: int,
        heartbeat_seconds: float,
        *,
        now_ms: int | None = None,
        clock: Clock | None = None,
        lease_keeper: LeaseKeeper | None = None,
    ):
        self.clock = clock if clock is not None else SystemClock()
        self.lease_keeper = lease_keeper if lease_keeper is not None else ThreadLeaseKeeper()
        self.successful_ready = False
        self.lease = lease
        self.heartbeat_store = heartbeat_store
        self.ttl_ms, self.heartbeat_seconds = ttl_ms, heartbeat_seconds
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.poll_lock = threading.Lock()
        self.commit_lock = threading.Lock()
        self._latch_entry = threading.local()
        self._poll_thread: int | None = None
        self._commit_thread: int | None = None
        self.lost = False
        self.heartbeat_error: Exception | None = None
        now_ms = self.clock.now_ms() if now_ms is None else now_ms
        self.expires_deadline = self.clock.monotonic() + (lease.expires_at_ms - now_ms) / 1000
        self.thread = threading.Thread(target=self._heartbeat, daemon=True, name="session-heartbeat")

    def poll(self, store: SessionStore):
        thread = threading.get_ident()
        if getattr(self._latch_entry, "commit", False) or getattr(self._latch_entry, "poll", False):
            raise RuntimeError("poll while holding commit/poll latch")
        self._latch_entry.poll = True
        try:
            with self.poll_lock:
                self._poll_thread = thread
                try:
                    with self.lock:
                        if self.lost:
                            raise LeaseLost("heartbeat lost ownership") from self.heartbeat_error
                        if self.stop.is_set():
                            raise HeartbeatStopped("heartbeat stopped")
                        lease = self.lease
                    started = self.clock.monotonic()
                    poll = store.renew(lease, ttl_ms=self.ttl_ms)
                    with self.lock:
                        if (poll.lease.session_id, poll.lease.owner, poll.lease.epoch) != (
                            lease.session_id,
                            lease.owner,
                            lease.epoch,
                        ):
                            self.lost = True
                            raise LeaseLost("heartbeat changed lease identity")
                        if not self.stop.is_set() and not self.lost:
                            self.lease = poll.lease
                            self.expires_deadline = started + (poll.lease.expires_at_ms - poll.db_now_ms) / 1000
                            self.heartbeat_error = None
                            self.successful_ready = True
                            self.ready.set()
                finally:
                    self._poll_thread = None
        finally:
            self._latch_entry.poll = False
        return poll

    @contextmanager
    def committing(self) -> Iterator[Lease]:
        self._latch_entry.commit = True
        try:
            with self.commit_lock:
                self._commit_thread = threading.get_ident()
                try:
                    with self.lock:
                        if self.lost:
                            raise LeaseLost("heartbeat lost ownership") from self.heartbeat_error
                        if self.stop.is_set():
                            raise HeartbeatStopped("heartbeat stopped")
                        if self.clock.monotonic() >= self.expires_deadline:
                            self.lost = True
                            raise LeaseLost("heartbeat lease expired")
                        lease = self.lease
                    yield lease
                finally:
                    self._commit_thread = None
        finally:
            self._latch_entry.commit = False

    def _heartbeat(self) -> None:
        try:
            with self.heartbeat_store() as store:
                transient_errors: tuple[type[Exception], ...] = (TimeoutError,)
                if isinstance(store, PostgresStore):
                    from psycopg import OperationalError

                    transient_errors = (TimeoutError, OperationalError)
                    # Host factories may bypass PostgresStore.heartbeat_store().
                    store._rows("SELECT set_config('lock_timeout','250ms',false),set_config('statement_timeout','1000ms',false)")
                while not self.stop.is_set():
                    with self.lock:
                        if self.heartbeat_error is not None and self.expires_deadline - self.clock.monotonic() <= max(
                            2 * self.heartbeat_seconds, 1.0
                        ):
                            raise self.heartbeat_error
                    try:
                        self.poll(store)
                    except Exception as exc:
                        if not isinstance(exc, transient_errors):
                            raise
                        with self.lock:
                            self.heartbeat_error = exc
                            if self.expires_deadline - self.clock.monotonic() <= max(2 * self.heartbeat_seconds, 1.0):
                                raise
                    if self.stop.wait(self.heartbeat_seconds):
                        break
        except HeartbeatStopped:
            pass
        except Exception as exc:
            with self.lock:
                self.heartbeat_error = exc
                self.lost = True
        finally:
            self.ready.set()

    def start(self) -> None:
        self.lease_keeper.start(self)

    def close(self) -> None:
        with self.lock:
            self.stop.set()
        self.lease_keeper.close(self)
        if self.lease_keeper.is_alive(self):
            with self.lock:
                self.lost = True
            raise LeaseLost("heartbeat did not stop; lease cannot be released or handed off")


class _Scope(_LeaseHeartbeat):
    def __init__(self, lease: Lease, runtime: Runtime, *, now_ms: int | None = None):
        super().__init__(
            lease,
            runtime.heartbeat_store,
            runtime.ttl_ms,
            runtime.heartbeat_seconds,
            now_ms=now_ms,
            clock=runtime.clock,
            lease_keeper=runtime.lease_keeper,
        )
        self.runtime = runtime
        self.token = CancellationToken()
        self.turn_id: str | None = None
        self.generation: int | None = None

    def poll(self, store: SessionStore):
        poll = super().poll(store)
        for item in poll.controls:
            if (
                item.target_turn_id in {None, self.turn_id}
                and item.generation in {None, self.generation}
                and item.payload["action"] in {"cancel", "abort", "close", "suspend"}
            ):
                self.token.cancel(item.payload["action"])
        return poll

    def _heartbeat(self) -> None:
        super()._heartbeat()
        if self.lost:
            self.token.cancel("lease_lost")


def _invoke(scope: _Scope, callback: Callable[[], Outcome], timeout: float) -> Outcome:
    return scope.runtime.invoker.invoke(
        callback,
        timeout=timeout,
        token=scope.token,
        cancellation_grace=scope.runtime.cancellation_grace,
        clock=scope.runtime.clock,
    )
