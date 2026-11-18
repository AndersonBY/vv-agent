"""Outer construction and scheduling for the shared core lease rules."""

from __future__ import annotations

import math
import sqlite3
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from typing import cast

from vv_agent.kernel._content_runtime import _root_store
from vv_agent.kernel._lease_scope import CancellationSink, LeaseOwnership, Signal, apply_controls, cancel_if_lost
from vv_agent.kernel._lease_scope import HeartbeatStopped as HeartbeatStopped
from vv_agent.kernel._postgres import PostgresStore as CorePostgresStore
from vv_agent.kernel._sqlite import SQLiteStore as CoreSQLiteStore
from vv_agent.runtime.cancellation import CancellationToken

from .providers import Outcome
from .runtime import (
    DEFAULT_SQLITE_HEARTBEAT_RETRY_BUDGET_SECONDS,
    DEFAULT_SQLITE_HEARTBEAT_RETRY_DELAY_SECONDS,
    Runtime,
)
from .store import Lease, LeaseLost, LeasePoll, SessionStore
from .timing import Clock, LeaseKeeper, SystemClock, ThreadLeaseKeeper

_SQLITE_BUSY = 5
_SQLITE_LOCKED = 6


def _sqlite_lock_error(error: sqlite3.OperationalError) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    if isinstance(code, int):
        return code & 0xFF in {_SQLITE_BUSY, _SQLITE_LOCKED}
    message = str(error).lower()
    return message == "database is locked" or message.startswith("database table is locked")


def _lease_backend(store: SessionStore) -> CoreSQLiteStore | CorePostgresStore:
    """Cross the explicit S1 content adapter boundary to the SQL backend."""

    backend = _root_store(store)
    if not isinstance(backend, CoreSQLiteStore | CorePostgresStore):
        raise RuntimeError("W4 lease renewal requires a supported SQL backend")
    return backend


class _SQLiteLockRetryingStore:
    """Retry only SQLite file-heartbeat write-lock contention."""

    def __init__(
        self,
        store: SessionStore,
        *,
        clock: Clock,
        stop: Signal,
        heartbeat: LeaseOwnership,
        retry_budget_seconds: float,
        retry_delay_seconds: float,
    ) -> None:
        self.store = store
        self.clock = clock
        self.stop = stop
        self.heartbeat = heartbeat
        self.retry_budget_seconds = retry_budget_seconds
        self.retry_delay_seconds = retry_delay_seconds

    def renew(self, lease: Lease, *, ttl_ms: int):
        connection = getattr(self.store, "connection", None)
        if not isinstance(connection, sqlite3.Connection):
            raise TypeError("SQLite heartbeat store does not expose a connection")
        started = self.clock.monotonic()
        safety_deadline = self.heartbeat.safety_deadline
        deadline = min(started + self.retry_budget_seconds, safety_deadline)
        last_lock: sqlite3.OperationalError | None = None
        while True:
            now = self.clock.monotonic()
            if now >= safety_deadline:
                self.heartbeat.lost = True
                raise LeaseLost("heartbeat reached the lease safety deadline") from last_lock
            remaining = deadline - now
            if remaining <= 0:
                raise TimeoutError("SQLite heartbeat renewal lock retry budget exhausted") from last_lock
            busy_timeout_ms = int(remaining * 1000)
            if busy_timeout_ms < 1:
                if deadline == safety_deadline:
                    self.heartbeat.lost = True
                    raise LeaseLost("heartbeat reached the lease safety deadline") from last_lock
                raise TimeoutError("SQLite heartbeat renewal lock retry budget exhausted") from last_lock
            configured_busy_timeout_ms = connection.execute("PRAGMA busy_timeout").fetchone()[0]
            connection.execute(f"PRAGMA busy_timeout={min(configured_busy_timeout_ms, busy_timeout_ms)}")
            try:
                try:
                    poll = self.store.renew(lease, ttl_ms=ttl_ms)
                except sqlite3.OperationalError as exc:
                    if not _sqlite_lock_error(exc):
                        raise
                    last_lock = exc
                    now = self.clock.monotonic()
                    if now >= safety_deadline:
                        self.heartbeat.lost = True
                        raise LeaseLost("heartbeat reached the lease safety deadline") from exc
                    remaining = deadline - now
                    if remaining <= 0:
                        raise TimeoutError("SQLite heartbeat renewal lock retry budget exhausted") from exc
                    if self.stop.wait(min(self.retry_delay_seconds, remaining)):
                        raise HeartbeatStopped("heartbeat stopped") from exc
                    continue
            finally:
                with suppress(sqlite3.Error):
                    connection.execute(f"PRAGMA busy_timeout={configured_busy_timeout_ms}")
            return poll


class _LeaseHeartbeat[TokenT: CancellationSink](LeaseOwnership[TokenT]):
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
        token_factory: Callable[[], TokenT] | None = None,
        sqlite_heartbeat_retry_budget_seconds: float = DEFAULT_SQLITE_HEARTBEAT_RETRY_BUDGET_SECONDS,
        sqlite_heartbeat_retry_delay_seconds: float = DEFAULT_SQLITE_HEARTBEAT_RETRY_DELAY_SECONDS,
    ):
        self.clock = clock if clock is not None else SystemClock()
        self.lease_keeper = lease_keeper if lease_keeper is not None else ThreadLeaseKeeper()
        self.heartbeat_store = heartbeat_store
        self.sqlite_heartbeat_retry_budget_seconds = sqlite_heartbeat_retry_budget_seconds
        self.sqlite_heartbeat_retry_delay_seconds = sqlite_heartbeat_retry_delay_seconds
        super().__init__(
            lease,
            ttl_ms,
            heartbeat_seconds,
            clock=self.clock,
            stop=threading.Event(),
            ready=threading.Event(),
            now_ms=now_ms,
            token_factory=token_factory,
        )

    def heartbeat_once(self, store: SessionStore) -> None:
        backend = _lease_backend(store)
        heartbeat_store = store
        transient_errors: tuple[type[Exception], ...] = (TimeoutError,)
        if isinstance(backend, CorePostgresStore):
            from psycopg.errors import LockNotAvailable, QueryCanceled

            transient_errors = (TimeoutError, LockNotAvailable, QueryCanceled)
            # Host factories may bypass PostgresStore.heartbeat_store(). Keep a
            # single statement inside the remaining local safety budget.
            if self.latest_poll is None and self._db_anchor_deadline is None:
                self.anchor_from_database(backend._now())
            remaining_ms = max(1, math.ceil((self.safety_deadline - self.clock.monotonic()) * 1000))
            # Leave a small ordering gap so lock waits report LockNotAvailable
            # rather than racing the enclosing statement deadline.
            lock_timeout_ms = min(250, max(1, remaining_ms - 10))
            statement_timeout_ms = min(1000, remaining_ms)
            backend._rows(
                "SELECT set_config('lock_timeout',%s,false),set_config('statement_timeout',%s,false)",
                (f"{lock_timeout_ms}ms", f"{statement_timeout_ms}ms"),
            )
        elif isinstance(backend, CoreSQLiteStore):
            if backend.path == ":memory:":
                raise RuntimeError("W4 lease renewal requires a file-backed SQLite store")
            heartbeat_store = cast(
                SessionStore,
                _SQLiteLockRetryingStore(
                    store,
                    clock=self.clock,
                    stop=self.stop,
                    heartbeat=self,
                    retry_budget_seconds=self.sqlite_heartbeat_retry_budget_seconds,
                    retry_delay_seconds=self.sqlite_heartbeat_retry_delay_seconds,
                ),
            )
        if not isinstance(backend, CorePostgresStore) and self.latest_poll is None and self._db_anchor_deadline is None:
            self.anchor_from_database(backend._now())
        self.heartbeat_step(heartbeat_store, transient_errors)

    def start(self) -> None:
        self.lease_keeper.start(self)

    def close(self) -> None:
        self.request_stop()
        self.lease_keeper.close(self)
        self.check_closed(self.lease_keeper.is_alive(self))


class _Scope(_LeaseHeartbeat[CancellationToken]):
    def __init__(self, lease: Lease, runtime: Runtime, *, now_ms: int | None = None):
        super().__init__(
            lease,
            runtime.heartbeat_store,
            runtime.ttl_ms,
            runtime.heartbeat_seconds,
            now_ms=now_ms,
            clock=runtime.clock,
            lease_keeper=runtime.lease_keeper,
            token_factory=CancellationToken,
            sqlite_heartbeat_retry_budget_seconds=runtime.sqlite_heartbeat_retry_budget_seconds,
            sqlite_heartbeat_retry_delay_seconds=runtime.sqlite_heartbeat_retry_delay_seconds,
        )
        self.runtime = runtime

    def adopt_confirmed(self, poll: LeasePoll, *, expires_deadline: float) -> None:
        super().adopt_confirmed(poll, expires_deadline=expires_deadline)
        if self.latest_poll is poll:
            apply_controls(self, poll)

    def heartbeat_confirmed(self, poll: LeasePoll) -> None:
        apply_controls(self, poll)
        super().heartbeat_confirmed(poll)

    def heartbeat_finished(self) -> None:
        super().heartbeat_finished()
        cancel_if_lost(self)


def _invoke(scope: _Scope, callback: Callable[[], Outcome], timeout: float) -> Outcome:
    return scope.runtime.invoker.invoke(
        callback,
        timeout=timeout,
        token=scope.token,
        cancellation_grace=scope.runtime.cancellation_grace,
        clock=scope.runtime.clock,
    )
