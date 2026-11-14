"""Lease ownership and latch rules using supplied time and synchronization ports."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Protocol

from vv_agent.kernel._store import Lease, LeaseLost, LeasePoll, SessionStore


class Clock(Protocol):
    def now_ms(self) -> int: ...

    def monotonic(self) -> float: ...

    def monotonic_ns(self) -> int: ...

    def sleep(self, seconds: float) -> None: ...


class Signal(Protocol):
    def is_set(self) -> bool: ...

    def set(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...


class Latch(Protocol):
    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool: ...

    def release(self) -> None: ...

    def locked(self) -> bool: ...

    def __enter__(self) -> object: ...

    def __exit__(
        self, kind: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None, /
    ) -> bool | None: ...


class LatchEntry(Protocol):
    commit: bool
    poll: bool


class CancellationSink(Protocol):
    def cancel(self, reason: str) -> None: ...


class ControlScope(Protocol):
    @property
    def token(self) -> CancellationSink: ...

    turn_id: str | None
    generation: int | None
    lost: bool


class HeartbeatStopped(LeaseLost):
    """Normal stop of a lease-renewal worker; grants no execution ownership."""


class LeaseOwnership:
    def __init__(
        self,
        lease: Lease,
        ttl_ms: int,
        heartbeat_seconds: float,
        *,
        clock: Clock,
        stop: Signal,
        ready: Signal,
        lock: Latch,
        poll_lock: Latch,
        commit_lock: Latch,
        latch_entry: LatchEntry,
        actor_identity: Callable[[], int],
        now_ms: int | None = None,
    ) -> None:
        self.clock = clock
        self.successful_ready = False
        self.lease = lease
        self.ttl_ms, self.heartbeat_seconds = ttl_ms, heartbeat_seconds
        self.stop, self.ready = stop, ready
        self.lock, self.poll_lock, self.commit_lock = lock, poll_lock, commit_lock
        self._latch_entry = latch_entry
        self._actor_identity = actor_identity
        self._poll_thread: int | None = None
        self._commit_thread: int | None = None
        self.lost = False
        self.heartbeat_error: Exception | None = None
        now_ms = self.clock.now_ms() if now_ms is None else now_ms
        self.expires_deadline = self.clock.monotonic() + (lease.expires_at_ms - now_ms) / 1000

    def poll(self, store: SessionStore):
        thread = self._actor_identity()
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
                self._commit_thread = self._actor_identity()
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

    def heartbeat_step(self, store: SessionStore, transient_errors: tuple[type[Exception], ...]) -> None:
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

    def heartbeat_failed(self, error: Exception) -> None:
        if isinstance(error, HeartbeatStopped):
            return
        with self.lock:
            self.heartbeat_error = error
            self.lost = True

    def heartbeat_finished(self) -> None:
        self.ready.set()

    def request_stop(self) -> None:
        with self.lock:
            self.stop.set()

    def check_closed(self, alive: bool) -> None:
        if alive:
            with self.lock:
                self.lost = True
            raise LeaseLost("heartbeat did not stop; lease cannot be released or handed off")


def apply_controls(self: ControlScope, poll: LeasePoll) -> None:
    for item in poll.controls:
        if (
            item.target_turn_id in {None, self.turn_id}
            and item.generation in {None, self.generation}
            and item.payload["action"] in {"cancel", "abort", "close", "suspend"}
        ):
            self.token.cancel(item.payload["action"])


def cancel_if_lost(self: ControlScope) -> None:
    if self.lost:
        self.token.cancel("lease_lost")
