"""Lease cleanup failures remain visible and do not skip release."""

from __future__ import annotations

from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vv_agent.kernel._drive import DriveBinding, DriveLeaseScope, DriveStep, drive_core
from vv_agent.kernel._store import Lease
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.timing import SystemClock


class _Runtime:
    ttl_ms = 1_000
    poll_ms = 10


class _Scope:
    def __init__(self, lease: Lease) -> None:
        self.clock = SystemClock()
        self.ready = Event()
        self.ready.set()
        self.lost = False
        self.successful_ready = True
        self.heartbeat_error: Exception | None = None
        self.lease = lease
        self.latest_poll = None
        self.expires_deadline = 1.0

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def adopt_confirmed(self, poll, *, expires_deadline: float) -> None:
        raise AssertionError((poll, expires_deadline))

    def committing(self):
        return nullcontext(self.lease)


class _OneTurn:
    records = (SimpleNamespace(record=SimpleNamespace(kind="turn_ended")),)

    def step(self) -> bool:
        return True


def test_drive_core_propagates_release_error_after_successful_drive() -> None:
    lease = Lease("release-error", "owner", 1, 10_000)
    release_error = OSError("release failed")

    class Store:
        tenant_id = lease.tenant_id

        def __init__(self) -> None:
            self.released: list[Lease] = []

        def release(self, candidate: Lease) -> bool:
            self.released.append(candidate)
            raise release_error

    store = Store()
    runtime = _Runtime()
    scope = _Scope(lease)
    binding = DriveBinding(
        clock=scope.clock,
        ttl_ms=runtime.ttl_ms,
        runtime=runtime,
        make_factory_heartbeat=None,
        make_runtime=None,
        factory_retry_after_ms=lambda _error: None,
        make_scope=lambda _lease, _runtime: cast(DriveLeaseScope, scope),
        make_driver=lambda _store, _session_id, _runtime, _scope: cast(DriveStep, _OneTurn()),
        after_drive=lambda _runtime: None,
        wake_if_runnable=lambda _runtime: None,
    )

    with pytest.raises(OSError) as caught:
        drive_core(
            cast(Any, store),
            lease.session_id,
            one_turn=True,
            lease=lease,
            make_binding=lambda: binding,
        )

    assert caught.value is release_error
    assert store.released == [lease]


def test_worker_read_error_releases_once_when_defer_also_fails() -> None:
    lease = Lease("defer-error", "owner", 1, 10_000)
    read_error = ValueError("read failed")
    defer_error = OSError("defer failed")

    class Store:
        def __init__(self) -> None:
            self.deferred: list[Lease] = []
            self.released: list[Lease] = []

        def read(self, *_args, **_kwargs):
            raise read_error

        def defer_drive(self, candidate: Lease, *, retry_after_ms: int) -> None:
            assert retry_after_ms == 25
            self.deferred.append(candidate)
            raise defer_error

        def release(self, candidate: Lease) -> bool:
            self.released.append(candidate)
            return True

    store = Store()
    worker = cast(
        SessionDriver,
        SimpleNamespace(
            _runtime_lineage=lambda _store, _session_id: (SimpleNamespace(armed=True), ()),
            _failure_after_progress=lambda *_args: None,
            _errors={},
            _worker=SimpleNamespace(support=SimpleNamespace(failure_backoff_ms=25)),
        ),
    )

    with pytest.raises(OSError) as caught:
        SessionDriver._drive_worker_session(worker, cast(Any, store), lease.session_id, lease=lease)

    assert caught.value is defer_error
    assert caught.value.__context__ is read_error
    assert store.deferred == [lease]
    assert store.released == [lease]
