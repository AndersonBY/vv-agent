"""Scheduled normal-stop classification against the actual SDK/core lease rules."""

import sqlite3

import pytest

from vv_agent.kernel._lease_scope import HeartbeatStopped, LeaseOwnership
from vv_agent.session._driver_scope import _LeaseHeartbeat, _Scope
from vv_agent.session.store import LeaseLost

from .world import SimWorld


def make_scope(world):
    world.create()
    runtime = world.runtime("s", "owner")
    store = world.connection("owner")
    lease = store.acquire("s", owner="owner", ttl_ms=runtime.ttl_ms)
    assert lease is not None
    return _Scope(lease, runtime, now_ms=store._now())


def outcome(scope):
    return {
        "lost": scope.lost,
        "error": type(scope.heartbeat_error).__name__ if scope.heartbeat_error else None,
        "cancelled": scope.token.cancelled,
        "reason": scope.token.reason,
        "ready": scope.ready.is_set(),
        "successful_ready": scope.successful_ready,
    }


def test_seeded_stop_drains_renewal_and_matches_sdk(tmp_path):
    outcomes, traces = [], []
    for name, seed, replay in (("first", 17, None), ("replay", 999, [{"at": "participant#0", "value": "stop/owner"}])):
        with SimWorld(tmp_path / f"{name}.sqlite", seed=seed, replay=replay) as world:
            scope = make_scope(world)
            keeper = world.keepers["owner"]
            # Both decisions use LeaseOwnership methods; there is no synthetic lease verdict.
            assert _LeaseHeartbeat.poll is LeaseOwnership.poll
            world.enqueue("renew/owner", lambda keeper=keeper, scope=scope: keeper.poll(scope))
            world.enqueue("stop/owner", scope.request_stop)
            assert world.step() and scope.stop.is_set()
            assert world.step()
            assert not world.step() and not world.tasks
            outcomes.append(outcome(scope))
            traces.append((world.choices.trace, world.events))
    assert traces[0] == traces[1]
    assert traces[0][0] == [
        {"at": "participant#0", "value": "stop/owner"},
        {"at": "participant#1", "value": "renew/owner"},
    ]
    with SimWorld(tmp_path / "sdk.sqlite") as world:
        scope = make_scope(world)
        scope.request_stop()
        scope._heartbeat()  # Actual production loop, synchronously; no thread is started.
        assert outcomes == [outcome(scope), outcome(scope)]
        assert outcome(scope) == {
            "lost": False,
            "error": None,
            "cancelled": False,
            "reason": None,
            "ready": True,
            "successful_ready": False,
        }


@pytest.mark.parametrize("adapter", ["sim", "sdk"])
def test_stop_between_loop_check_and_core_poll_is_normal(tmp_path, monkeypatch, adapter):
    with SimWorld(tmp_path / f"{adapter}.sqlite") as world:
        scope = make_scope(world)
        original_poll = scope.poll
        caught = []

        def stop_then_poll(store):
            scope.request_stop()
            try:
                return original_poll(store)
            except HeartbeatStopped as exc:
                caught.append(exc)
                raise

        monkeypatch.setattr(scope, "poll", stop_then_poll)
        if adapter == "sdk":
            scope._heartbeat()
        else:
            world.keepers["owner"].start(scope)
        assert len(caught) == 1 and type(caught[0]) is HeartbeatStopped
        assert not scope.lost and scope.heartbeat_error is None and not scope.token.cancelled
        assert scope.ready.is_set() and not world.tasks
        assert not scope.thread.is_alive()


@pytest.mark.parametrize("scoped", [True, False])
def test_start_after_stop_never_enqueues(tmp_path, scoped):
    with SimWorld(tmp_path / "stopped.sqlite") as world:
        scope = make_scope(world)
        heartbeat = (
            scope
            if scoped
            else _LeaseHeartbeat(
                scope.lease,
                scope.heartbeat_store,
                scope.ttl_ms,
                scope.heartbeat_seconds,
                clock=world.clock,
                lease_keeper=world.keepers["owner"],
            )
        )
        world.enqueue("renew/owner", lambda: world.keepers["owner"].poll(heartbeat))
        heartbeat.request_stop()
        heartbeat.start()
        assert heartbeat.ready.is_set() and not heartbeat.lost
        assert heartbeat.heartbeat_error is None
        assert not world.tasks and not world.step()


def test_stop_during_committed_renewal_keeps_l2_publication_and_drains(tmp_path, monkeypatch):
    with SimWorld(tmp_path / "during.sqlite") as world:
        scope = make_scope(world)
        initial = scope.lease
        store = world.connection("owner/heartbeat")
        renew = store.renew
        receipts = []

        def stop_after_renew(lease, *, ttl_ms):
            receipt = renew(lease, ttl_ms=ttl_ms)
            receipts.append(receipt)
            scope.request_stop()
            return receipt

        monkeypatch.setattr(store, "renew", stop_after_renew)
        world.advance(1000)
        scope.start()
        assert len(receipts) == 1 and receipts[0].lease.expires_at_ms > initial.expires_at_ms
        assert scope.lease is initial and not scope.successful_ready
        assert not scope.lost and not scope.token.cancelled and scope.ready.is_set()
        assert not world.tasks and not world.step()


@pytest.mark.parametrize("adapter", ["sim", "sdk"])
def test_real_sql_expiry_keeps_loss_evidence_and_cancellation(tmp_path, adapter):
    with SimWorld(tmp_path / f"lost-{adapter}.sqlite") as world:
        scope = make_scope(world)
        world.expire()
        successor = world.connection("successor").acquire("s", owner="successor", ttl_ms=scope.ttl_ms)
        assert successor is not None and successor.epoch > scope.lease.epoch
        keeper = world.keepers["owner"]
        if adapter == "sim":
            world.enqueue("renew/owner", lambda: keeper.poll(scope))
            with pytest.raises(LeaseLost) as raised:
                keeper.poll(scope)
            assert scope.heartbeat_error is raised.value
        else:
            scope._heartbeat()
        assert type(scope.heartbeat_error) is LeaseLost
        assert scope.lost and scope.token.cancelled and scope.token.reason == "lease_lost"
        assert scope.ready.is_set() and not world.tasks
        # Normal stop must not erase already established ownership loss.
        scope.request_stop()
        with pytest.raises(LeaseLost) as raised:
            keeper.poll(scope)
        assert type(raised.value) is LeaseLost and raised.value.__cause__ is not None
        assert not world.tasks


def test_enqueue_failure_is_not_swallowed_or_reclassified(tmp_path, monkeypatch):
    with SimWorld(tmp_path / "enqueue.sqlite") as world:
        scope = make_scope(world)
        failure = RuntimeError("scheduler enqueue failed")

        def reject(label, callback):
            raise failure

        monkeypatch.setattr(world, "enqueue", reject)
        with pytest.raises(RuntimeError) as raised:
            scope.start()
        assert raised.value is failure
        assert scope.successful_ready and not scope.lost and not scope.token.cancelled
        assert scope.heartbeat_error is None and not world.tasks


def test_blocked_renew_then_stop_drains_existing_retry(tmp_path, monkeypatch):
    with SimWorld(tmp_path / "blocked.sqlite") as world:
        scope = make_scope(world)
        store = world.connection("owner/heartbeat")
        calls = []

        def busy(*args, **kwargs):
            calls.append(1)
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(store, "renew", busy)
        world.enqueue("renew/owner", lambda: world.keepers["owner"].poll(scope))
        assert world.step()
        assert {"blocked": "renew/owner"} in world.events and "renew/owner" in world.tasks
        scope.request_stop()
        assert world.step()
        assert calls == [1] and not world.tasks and not world.step()
        assert not scope.lost and not scope.token.cancelled and scope.heartbeat_error is None
