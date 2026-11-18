"""Lease ownership and renewal rules using supplied time and signaling ports."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
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
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> bool | None: ...


class CancellationSink(Protocol):
    def cancel(self, reason: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ControlBinding[TokenT: CancellationSink]:
    """One token, turn identity and consumed-input view for control delivery."""

    token: TokenT
    turn_id: str | None
    generation: int | None
    applied_inputs: Mapping[str, object]


class ControlScope[TokenT: CancellationSink](Protocol):
    control_binding: ControlBinding[TokenT] | None
    lost: bool


class HeartbeatStopped(LeaseLost):
    """Normal stop of a lease-renewal worker; grants no execution ownership."""


@dataclass(frozen=True, slots=True)
class _ConfirmedLease:
    poll: LeasePoll
    expires_deadline: float


class LeaseOwnership[TokenT: CancellationSink]:
    def __init__(
        self,
        lease: Lease,
        ttl_ms: int,
        heartbeat_seconds: float,
        *,
        clock: Clock,
        stop: Signal,
        ready: Signal,
        now_ms: int | None = None,
        token_factory: Callable[[], TokenT] | None = None,
    ) -> None:
        self.clock = clock
        self.successful_ready = False
        self._initial_lease = lease
        self._confirmed: _ConfirmedLease | None = None
        self._db_anchor_deadline: float | None = None
        self.ttl_ms, self.heartbeat_seconds = ttl_ms, heartbeat_seconds
        self.stop, self.ready = stop, ready
        self.lost = False
        self.heartbeat_error: Exception | None = None
        self._token_factory = token_factory
        self.control_binding: ControlBinding[TokenT] | None = (
            ControlBinding(token_factory(), None, None, {}) if token_factory is not None else None
        )
        if now_ms is not None:
            self._db_anchor_deadline = self.clock.monotonic() + (lease.expires_at_ms - now_ms) / 1000

    @property
    def latest_poll(self) -> LeasePoll | None:
        """Return the latest immutable renewal receipt as one confirmed snapshot."""

        confirmed = self._confirmed
        return None if confirmed is None else confirmed.poll

    def confirmed_poll(self) -> LeasePoll:
        """Return the current complete receipt without opening a renewal connection."""

        if self.lost:
            raise LeaseLost("heartbeat lost ownership") from self.heartbeat_error
        if self.stop.is_set():
            raise HeartbeatStopped("heartbeat stopped")
        confirmed = self._confirmed
        if confirmed is None:
            raise LeaseLost("heartbeat lease has no confirmed poll")
        if self.clock.monotonic() >= confirmed.expires_deadline:
            self.lost = True
            raise LeaseLost("heartbeat lease expired")
        return confirmed.poll

    @property
    def token(self) -> TokenT:
        binding = self.control_binding
        if binding is None:
            raise RuntimeError("lease scope has no cancellation token")
        return binding.token

    @property
    def turn_id(self) -> str | None:
        binding = self.control_binding
        return None if binding is None else binding.turn_id

    @property
    def generation(self) -> int | None:
        binding = self.control_binding
        return None if binding is None else binding.generation

    def bind_turn(
        self,
        turn_id: str | None,
        generation: int | None,
        applied_inputs: Mapping[str, object],
    ) -> None:
        """Atomically publish token and principal turn identity for controls."""

        previous = self.control_binding
        same_turn = previous is not None and (previous.turn_id, previous.generation) == (turn_id, generation)
        token = previous.token if same_turn else self._new_token()
        self.control_binding = ControlBinding(token, turn_id, generation, applied_inputs)
        self._apply_latest_controls()

    def reset_token(self) -> None:
        """Replace only the cancellation token while preserving the current binding."""

        previous = self.control_binding
        token = self._new_token()
        self.control_binding = ControlBinding(
            token,
            None if previous is None else previous.turn_id,
            None if previous is None else previous.generation,
            {} if previous is None else previous.applied_inputs,
        )
        self._apply_latest_controls()

    def _new_token(self) -> TokenT:
        if self._token_factory is None:
            raise RuntimeError("lease scope has no cancellation token factory")
        return self._token_factory()

    def _apply_latest_controls(self) -> None:
        poll = self.latest_poll
        if poll is not None:
            apply_controls(self, poll)

    @property
    def lease(self) -> Lease:
        confirmed = self._confirmed
        return self._initial_lease if confirmed is None else confirmed.poll.lease

    @property
    def expires_deadline(self) -> float:
        confirmed = self._confirmed
        if confirmed is not None:
            return confirmed.expires_deadline
        # A database read may seed only the local deadline. It is not published
        # as latest_poll because no renewal/inbox receipt has been confirmed.
        return self._db_anchor_deadline if self._db_anchor_deadline is not None else self.clock.monotonic()

    @property
    def safety_deadline(self) -> float:
        margin = min(self.ttl_ms / 2000, max(2 * self.heartbeat_seconds, 1.0))
        return self.expires_deadline - margin

    def poll(self, store: SessionStore):
        if self.lost:
            raise LeaseLost("heartbeat lost ownership") from self.heartbeat_error
        if self.stop.is_set():
            raise HeartbeatStopped("heartbeat stopped")
        lease = self.lease
        safety_deadline = self.safety_deadline if self._confirmed is not None or self._db_anchor_deadline is not None else None
        started = self.clock.monotonic()
        poll = store.renew(lease, ttl_ms=self.ttl_ms)
        # A response held past the pre-renew safety bound cannot restore local
        # ownership confidence, even when its database receipt remains valid.
        if safety_deadline is not None and self.clock.monotonic() >= safety_deadline:
            self.lost = True
            raise LeaseLost("heartbeat reached the lease safety deadline")
        if (poll.lease.session_id, poll.lease.owner, poll.lease.epoch, poll.lease.tenant_id) != (
            lease.session_id,
            lease.owner,
            lease.epoch,
            lease.tenant_id,
        ):
            self.lost = True
            raise LeaseLost("heartbeat changed lease identity")
        if poll.db_now_ms >= poll.lease.expires_at_ms:
            self.lost = True
            raise LeaseLost("heartbeat renewal returned an expired lease")
        confirmed = _ConfirmedLease(
            poll,
            started + (poll.lease.expires_at_ms - poll.db_now_ms) / 1000,
        )
        current = self._confirmed
        if current is None or (poll.db_now_ms, poll.lease.expires_at_ms) >= (
            current.poll.db_now_ms,
            current.poll.lease.expires_at_ms,
        ):
            self._confirmed = confirmed
            self.heartbeat_error = None
            self.successful_ready = True
            self.heartbeat_confirmed(poll)
        return poll

    def heartbeat_confirmed(self, poll: LeasePoll) -> None:
        """Publish readiness after subclass-specific confirmed-receipt handling."""

        del poll
        self.ready.set()

    def anchor_from_database(self, now_ms: int) -> None:
        """Seed the first renewal deadline from the renewal connection's clock."""

        if self.lost:
            raise LeaseLost("heartbeat lost ownership") from self.heartbeat_error
        if self.stop.is_set():
            raise HeartbeatStopped("heartbeat stopped")
        lease = self.lease
        if now_ms >= lease.expires_at_ms:
            self.lost = True
            raise LeaseLost("heartbeat lease expired before its first renewal")
        self._db_anchor_deadline = self.clock.monotonic() + (lease.expires_at_ms - now_ms) / 1000

    def adopt_confirmed(self, poll: LeasePoll, *, expires_deadline: float) -> None:
        """Seed a runtime scope from a stopped factory's confirmed handoff."""

        if self.lost:
            raise LeaseLost("cannot adopt a lease snapshot after ownership loss")
        if self.stop.is_set():
            raise HeartbeatStopped("cannot adopt a lease snapshot after stop")
        lease = self.lease
        if (poll.lease.session_id, poll.lease.owner, poll.lease.epoch, poll.lease.tenant_id) != (
            lease.session_id,
            lease.owner,
            lease.epoch,
            lease.tenant_id,
        ):
            raise LeaseLost("handoff snapshot belongs to a different lease")
        if self.clock.monotonic() >= expires_deadline:
            raise LeaseLost("handoff snapshot is already outside its lease deadline")
        current = self._confirmed
        if current is None or (poll.db_now_ms, poll.lease.expires_at_ms) >= (
            current.poll.db_now_ms,
            current.poll.lease.expires_at_ms,
        ):
            self._confirmed = _ConfirmedLease(poll, expires_deadline)
            self.successful_ready = True
            self.heartbeat_error = None

    @contextmanager
    def committing(self) -> Iterator[Lease]:
        if self.lost:
            raise LeaseLost("heartbeat lost ownership") from self.heartbeat_error
        if self.stop.is_set():
            raise HeartbeatStopped("heartbeat stopped")
        confirmed = self._confirmed
        if confirmed is not None:
            lease, expires_deadline = confirmed.poll.lease, confirmed.expires_deadline
        elif self._db_anchor_deadline is not None:
            lease, expires_deadline = self._initial_lease, self._db_anchor_deadline
        else:
            raise LeaseLost("heartbeat lease has not been confirmed by the database")
        if self.clock.monotonic() >= expires_deadline:
            self.lost = True
            raise LeaseLost("heartbeat lease expired")
        # This is a non-blocking local precheck. The transaction's owner, epoch
        # and expiry fences remain authoritative after any concurrent renewal.
        yield lease

    def heartbeat_once(self, store: SessionStore) -> None:
        """Perform one renewal using the core state transition and receipt checks."""

        self.heartbeat_step(store, (TimeoutError,))

    def heartbeat_step(self, store: SessionStore, transient_errors: tuple[type[Exception], ...]) -> None:
        prior_error = self.heartbeat_error
        if prior_error is not None and self.clock.monotonic() >= self.safety_deadline:
            self.lost = True
            raise LeaseLost("heartbeat reached the lease safety deadline") from prior_error
        try:
            self.poll(store)
        except Exception as exc:
            if isinstance(exc, LeaseLost):
                raise
            if not isinstance(exc, transient_errors):
                raise
            self.heartbeat_error = exc
            if self._confirmed is None and self._db_anchor_deadline is None:
                self.lost = True
                raise LeaseLost("heartbeat lease has no database-time safety bound") from exc
            if self.clock.monotonic() >= self.safety_deadline:
                self.lost = True
                raise LeaseLost("heartbeat reached the lease safety deadline") from exc

    def heartbeat_failed(self, error: Exception) -> None:
        if isinstance(error, HeartbeatStopped):
            return
        self.heartbeat_error = error
        self.lost = True

    def heartbeat_finished(self) -> None:
        if self.successful_ready or self.lost or self.stop.is_set():
            self.ready.set()

    def request_stop(self) -> None:
        self.stop.set()

    def check_closed(self, alive: bool) -> None:
        if alive:
            self.lost = True
            raise LeaseLost("heartbeat did not stop; lease cannot be released or handed off")


def apply_controls[TokenT: CancellationSink](self: ControlScope[TokenT], poll: LeasePoll) -> None:
    binding = self.control_binding
    if binding is None or binding.turn_id is None:
        return
    for item in poll.controls:
        if item.input_id in binding.applied_inputs:
            continue
        if (
            item.target_turn_id in {None, binding.turn_id}
            and item.generation in {None, binding.generation}
            and item.payload.get("action") in {"cancel", "abort", "close", "suspend"}
        ):
            binding.token.cancel(item.payload["action"])


def cancel_if_lost[TokenT: CancellationSink](self: ControlScope[TokenT]) -> None:
    binding = self.control_binding
    if self.lost and binding is not None:
        binding.token.cancel("lease_lost")
