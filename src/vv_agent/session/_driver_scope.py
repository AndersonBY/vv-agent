"""Outer construction and scheduling for the shared core lease rules."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import cast

from vv_agent.kernel._lease_scope import HeartbeatStopped as HeartbeatStopped
from vv_agent.kernel._lease_scope import LatchEntry, LeaseOwnership, apply_controls, cancel_if_lost
from vv_agent.kernel._postgres import PostgresStore
from vv_agent.runtime.cancellation import CancellationToken

from .providers import Outcome
from .runtime import Runtime
from .sqlite import SQLiteStore
from .store import Lease, SessionStore
from .timing import Clock, LeaseKeeper, SystemClock, ThreadLeaseKeeper

_SQLITE_LOCK_MESSAGES = ("database is locked", "database table is locked")


class _SQLiteLockRetryingStore:
    """Retry only SQLite file-heartbeat write-lock contention."""

    def __init__(
        self,
        store: SessionStore,
        *,
        clock: Clock,
        stop: threading.Event,
        retry_budget_seconds: float,
        retry_delay_seconds: float,
    ) -> None:
        self.store = store
        self.clock = clock
        self.stop = stop
        self.retry_budget_seconds = retry_budget_seconds
        self.retry_delay_seconds = retry_delay_seconds

    def renew(self, lease: Lease, *, ttl_ms: int):
        deadline = self.clock.monotonic() + self.retry_budget_seconds
        while True:
            try:
                return self.store.renew(lease, ttl_ms=ttl_ms)
            except sqlite3.OperationalError as exc:
                if not any(message in str(exc) for message in _SQLITE_LOCK_MESSAGES):
                    raise
                remaining = deadline - self.clock.monotonic()
                if remaining <= 0:
                    raise TimeoutError("SQLite heartbeat renewal lock retry budget exhausted") from exc
                if self.stop.wait(min(self.retry_delay_seconds, remaining)):
                    raise HeartbeatStopped("heartbeat stopped") from exc


class _LeaseHeartbeat(LeaseOwnership):
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
        self.heartbeat_store = heartbeat_store
        super().__init__(
            lease,
            ttl_ms,
            heartbeat_seconds,
            clock=self.clock,
            stop=threading.Event(),
            ready=threading.Event(),
            lock=threading.Lock(),
            poll_lock=threading.Lock(),
            commit_lock=threading.Lock(),
            latch_entry=cast(LatchEntry, threading.local()),
            actor_identity=threading.get_ident,
            now_ms=now_ms,
        )
        self.thread = threading.Thread(target=self._heartbeat, daemon=True, name="session-heartbeat")

    def _heartbeat(self) -> None:
        try:
            with self.heartbeat_store() as store:
                transient_errors: tuple[type[Exception], ...] = (TimeoutError,)
                heartbeat_store = store
                if isinstance(store, PostgresStore):
                    from psycopg import OperationalError

                    transient_errors = (TimeoutError, OperationalError)
                    # Host factories may bypass PostgresStore.heartbeat_store().
                    store._rows("SELECT set_config('lock_timeout','250ms',false),set_config('statement_timeout','1000ms',false)")
                elif isinstance(store, SQLiteStore) and store.path != ":memory:":
                    heartbeat_store = cast(
                        SessionStore,
                        _SQLiteLockRetryingStore(
                            store,
                            clock=self.clock,
                            stop=self.stop,
                            retry_budget_seconds=min(self.heartbeat_seconds / 2, 0.25),
                            retry_delay_seconds=min(self.heartbeat_seconds / 8, 0.05),
                        ),
                    )
                while not self.stop.is_set():
                    self.heartbeat_step(heartbeat_store, transient_errors)
                    if self.stop.wait(self.heartbeat_seconds):
                        break
        except Exception as exc:
            self.heartbeat_failed(exc)
        finally:
            self.heartbeat_finished()

    def start(self) -> None:
        self.lease_keeper.start(self)

    def close(self) -> None:
        self.request_stop()
        self.lease_keeper.close(self)
        self.check_closed(self.lease_keeper.is_alive(self))


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
        apply_controls(self, poll)
        return poll

    def _heartbeat(self) -> None:
        super()._heartbeat()
        cancel_if_lost(self)


def _invoke(scope: _Scope, callback: Callable[[], Outcome], timeout: float) -> Outcome:
    return scope.runtime.invoker.invoke(
        callback,
        timeout=timeout,
        token=scope.token,
        cancellation_grace=scope.runtime.cancellation_grace,
        clock=scope.runtime.clock,
    )
