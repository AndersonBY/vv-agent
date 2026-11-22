"""Local child discovery wake and deleted-session observer regressions."""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, cast

import pytest

import vv_agent.session.surfaces as surface_module
from vv_agent import RunConfig
from vv_agent.kernel._sql import SessionNotLive
from vv_agent.session.store import Conflict, SessionStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.worker import LocalWorker, _DriveCompletion


class _FakeStore:
    def __init__(self) -> None:
        self.states: dict[str, Any] = {}
        self.heads: dict[str, int] = {}
        self.deleted: set[str] = set()
        self.read_errors: dict[str, Conflict] = {}
        self.runnable: set[str] = set()

    def read_state(self, session_id: str):
        if session_id in self.read_errors:
            raise self.read_errors[session_id]
        if session_id in self.deleted:
            raise SessionNotLive("session is fenced for deletion")
        return self.states[session_id], (), 0

    def read(self, session_id: str, *, limit: int = 1024):
        del limit
        if session_id in self.read_errors:
            raise self.read_errors[session_id]
        if session_id in self.deleted:
            raise SessionNotLive("session is fenced for deletion")
        return SimpleNamespace(head_seq=self.heads[session_id], inbox_seq=0)

    def is_runnable(self, session_id: str) -> bool:
        return session_id in self.runnable

    def lease_status(self, session_id: str) -> tuple[int, int, int]:
        return self.heads[session_id], 0, 0


def _registration() -> Any:
    return SimpleNamespace(
        agent=None,
        config=RunConfig(stream=lambda _event: None),
        resolved=None,
        handler_version="1",
    )


def _child_handle(session_id: str) -> dict[str, Any]:
    return {
        "kind": "child",
        "session_id": session_id,
        "turn_id": f"{session_id}/turn/start",
        "generation": 1,
        "background": True,
        "delivery_target": {
            "session_id": "root",
            "turn_id": "root/turn/main",
            "generation": 1,
            "operation_id": "op",
            "attempt": 1,
        },
    }


def _state(*child_sessions: str) -> Any:
    return SimpleNamespace(
        operations={
            f"op-{child_session}": SimpleNamespace(attempts={1: SimpleNamespace(child_handle=_child_handle(child_session))})
            for child_session in child_sessions
        }
    )


def test_observer_wakes_known_child_only_for_new_head_or_first_discovery(monkeypatch) -> None:
    driver = SessionDriver()
    store = _FakeStore()
    root = object()
    wakes: list[str] = []
    try:
        driver._runtimes["root"] = _registration()
        store.states["root"] = _state("child-one")
        store.heads["child-one"] = 4
        monkeypatch.setattr(driver, "_runtime_lineage", lambda _store, _sid: (root, ()))
        monkeypatch.setattr(driver, "_registration_for_cursor", lambda _sid, store=None: None)
        worker = driver._worker
        assert isinstance(worker, LocalWorker)
        monkeypatch.setattr(worker, "wake", wakes.append)

        driver._observe_registered_handles(cast(SessionStore, store))
        assert wakes == ["child-one"]
        assert driver._observer_cursors["child-one"] == 4

        driver._observe_registered_handles(cast(SessionStore, store))
        assert wakes == ["child-one"]

        store.runnable.add("child-one")
        worker._inflight = Future[_DriveCompletion]()
        driver._observe_registered_handles(cast(SessionStore, store))
        assert wakes == ["child-one"]

        worker._inflight = None
        driver._observe_registered_handles(cast(SessionStore, store))
        assert wakes == ["child-one", "child-one"]

        store.runnable.remove("child-one")  # the scan claimed the child drive
        store.heads["child-one"] = 5
        driver._observe_registered_handles(cast(SessionStore, store))
        assert wakes == ["child-one", "child-one", "child-one"]

        driver._observer_cursors["child-one"] = 5  # projection drained the new prefix
        store.states["root"] = _state("child-one", "child-two")
        store.heads["child-two"] = 2
        driver._observe_registered_handles(cast(SessionStore, store))
        assert wakes == ["child-one", "child-one", "child-one", "child-two"]
    finally:
        driver.close()


@pytest.mark.parametrize(
    "deletion_error",
    [
        SessionNotLive("session is fenced for deletion"),
        Conflict("session or cursor not found"),
    ],
)
def test_deleted_child_is_pruned_without_stopping_live_child_observation(monkeypatch, deletion_error) -> None:
    driver = SessionDriver()
    store = _FakeStore()
    root = object()
    observed: list[str] = []
    try:
        driver._runtimes.update(
            {
                "root": _registration(),
                "other-root": _registration(),
                "deleted-child": _registration(),
            }
        )
        store.states["root"] = _state("deleted-child", "live-child")
        store.states["other-root"] = _state()
        store.states["live-child"] = _state()
        store.heads["live-child"] = 3
        store.read_errors["deleted-child"] = deletion_error
        driver._observer_cursors.update({"deleted-child": 2, "live-child": 2})
        surface_module._DRIVERS[(driver.tenant_id, "deleted-child")] = driver
        surface_module._DRIVERS[(driver.tenant_id, "live-child")] = driver

        def registration(session_id: str, store=None):
            del store
            if session_id != "live-child":
                return None
            return _registration()

        monkeypatch.setattr(driver, "_runtime_lineage", lambda _store, _sid: (root, ()))
        monkeypatch.setattr(driver, "_registration_for_cursor", registration)
        monkeypatch.setattr(driver, "_deliver_projected_events", lambda _store, sid, _runtime: observed.append(sid))
        monkeypatch.setattr(driver._worker, "wake", lambda _sid: None)

        driver._observe_registered_handles(cast(SessionStore, store))

        assert observed == ["live-child"]
        assert "deleted-child" not in driver._runtimes
        assert "deleted-child" not in driver._observer_cursors
        assert (driver.tenant_id, "deleted-child") not in surface_module._DRIVERS
        assert {"root", "other-root"} <= set(driver._runtimes)
        assert driver._observer_cursors["live-child"] == 2
        assert surface_module._DRIVERS[(driver.tenant_id, "live-child")] is driver
    finally:
        driver.close()


def test_observer_propagates_non_deletion_conflicts(monkeypatch) -> None:
    driver = SessionDriver()
    store = _FakeStore()
    root = object()
    try:
        driver._runtimes["root"] = _registration()
        store.states["root"] = _state("broken-child")
        store.heads["broken-child"] = 1
        driver._observer_cursors["broken-child"] = 1
        surface_module._DRIVERS[(driver.tenant_id, "broken-child")] = driver
        monkeypatch.setattr(driver, "_runtime_lineage", lambda _store, _sid: (root, ()))
        monkeypatch.setattr(driver, "_registration_for_cursor", lambda _sid, store=None: None)
        monkeypatch.setattr(
            store,
            "read",
            lambda session_id, **_kwargs: (_ for _ in ()).throw(Conflict("database invariant failed")),
        )

        with pytest.raises(Conflict, match="database invariant failed"):
            driver._observe_registered_handles(cast(SessionStore, store))
        assert "root" in driver._runtimes
        assert driver._observer_cursors["broken-child"] == 1
        assert surface_module._DRIVERS[(driver.tenant_id, "broken-child")] is driver
    finally:
        driver.close()
