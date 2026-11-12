"""Known lease cuts: reproduce robustness symptoms, then check real SQL recovery."""

from __future__ import annotations

import inspect
import sys

import pytest

from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.session import kernel
from vv_agent.session.children import ChildSession
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.store import LeaseLost
from vv_agent.types import LLMResponse

from .world import SimWorld

# Replay identifiers for the cut schedules documented in lease-gap-evidence.md.
L2_SEED, L3_SEED, L4_SEED = 2002, 3003, 4004


def driver(world, sid="s", participant="manual"):
    store = world.connection(participant)
    runtime = world.runtime(sid, participant)
    lease = store.acquire(sid, owner=participant, ttl_ms=runtime.ttl_ms)
    assert lease is not None
    scope = kernel._Scope(lease, runtime, now_ms=store._now())
    worker = kernel._Driver(store, sid, runtime, scope)
    worker.refresh()
    return worker


def terminal_status(rows, tid):
    return next(r.record.payload["status"] for r in rows if r.record.kind == "turn_ended" and r.record.turn_id == tid)


def until(worker, predicate):
    for _ in range(40):
        if predicate():
            return
        assert worker.step(), "driver became idle before the scheduled cut"
    pytest.fail("bounded driver schedule did not reach its cut")


def test_l2_stop_after_committed_renewal_preserves_fencing_and_control(tmp_path, monkeypatch):
    with SimWorld(tmp_path / "l2.sqlite", seed=L2_SEED) as world:
        world.install_write_observer()
        world.create()
        worker = driver(world)
        until(worker, lambda: worker.state.active_turn_id is not None)
        tid = worker.state.active_turn_id
        control = InboxItem("l2-cancel", "control", {"action": "cancel"}, tid, 0)
        world.push("s", control)
        initial = worker.scope.lease
        build = kernel._LeaseHeartbeat(
            initial,
            worker.runtime.heartbeat_store,
            15000,
            0.25,
            now_ms=world.clock.now_ms(),
            clock=world.clock,
        )
        heartbeat_store = world.connection("manual/heartbeat")
        renew = heartbeat_store.renew
        returned = []

        def stop_after_commit(lease, *, ttl_ms):
            poll = renew(lease, ttl_ms=ttl_ms)
            # The SQL transaction has committed; close wins before publication.
            build.stop.set()
            returned.append(poll)
            return poll

        monkeypatch.setattr(heartbeat_store, "renew", stop_after_commit)
        world.advance(world.choices.choose("l2/renew-after-ms", [1000, 2000, 3000]))
        world.enqueue("l2/committed-renewal-then-stop", lambda: build.poll(heartbeat_store))
        assert world.step()
        poll = returned[0]
        assert poll.controls == (control,)
        assert build.lease == initial and not build.successful_ready
        assert heartbeat_store._one("SELECT lease_until_ms FROM sk_session WHERE session_id=%s", ("s",))[0] == (
            poll.lease.expires_at_ms
        )
        world.advance(initial.expires_at_ms - world.clock.now_ms() + 1)
        before = world.connection("observer").read("s").records
        assert not worker.store.release(build.lease)
        with pytest.raises(LeaseLost):
            worker.store.renew(build.lease, ttl_ms=15000)
        # Neither stale cleanup nor a new claimant can erase the committed renewal.
        assert worker.store.acquire("s", owner="too-early", ttl_ms=15000) is None
        assert worker.store.read("s").records == before
        assert worker.store.read_input("s", control.input_id).consumed_seq is None
        world.advance(poll.lease.expires_at_ms - world.clock.now_ms() + 1)
        world.enqueue("l2/recover-after-db-expiry", lambda: world.drive(participant="recovery"))
        assert world.step()
        _state, rows, _ = worker.store.read_state("s")
        assert terminal_status(rows, tid) == "cancelled"
        assert len([r for r in rows if r.record.kind == "input_applied" and r.record.payload["input"] == control.to_dict()]) == 1
        assert worker.store.read_input("s", control.input_id).consumed_seq is not None
        assert not world.calls
        assert max(r.writer_epoch for r in rows) == initial.epoch + 1
        world.check(quiescent=True, every_prefix=True)


def test_l2_factory_handoff_poll_leaves_controls_for_formal_driver(tmp_path, monkeypatch):
    with SimWorld(tmp_path / "l2-handoff.sqlite", seed=L2_SEED) as world:
        world.install_write_observer()
        world.create()
        worker = driver(world)
        until(worker, lambda: worker.state.active_turn_id is not None)
        tid = worker.state.active_turn_id
        assert worker.store.release(worker.scope.lease)
        control = InboxItem("handoff-cancel", "control", {"action": "cancel"}, tid, 0)
        factory_store = world.connection("factory-worker")
        renew, polls = factory_store.renew, []

        def observe_handoff(lease, *, ttl_ms):
            poll = renew(lease, ttl_ms=ttl_ms)
            polls.append(poll)
            return poll

        monkeypatch.setattr(factory_store, "renew", observe_handoff)

        def factory(_sid):
            # The build keeper has already polled; this input reaches the handoff.
            world.push("s", control)
            return world.runtime("s", "factory-worker")

        world.enqueue(
            "l2/formal-driver-consumes-retained-control",
            lambda: world.drive(participant="factory-worker", factory=factory),
        )
        assert world.step()
        assert polls and polls[0].controls == (control,)
        assert terminal_status(worker.store.read("s").records, tid) == "cancelled"
        assert worker.store.read_input("s", "handoff-cancel").consumed_seq is not None
        assert not world.calls
        world.check(quiescent=True, every_prefix=True)


def test_l3_late_control_hits_new_token_but_durable_cancel_reaches_child(tmp_path):
    with SimWorld(tmp_path / "l3.sqlite", seed=L3_SEED) as world:
        world.install_write_observer()
        world.create()
        children = {"effect": lambda _plan: ChildSession(SessionSpec("child", "sim", "/tmp"), "go")}
        world.drive(children=children)
        worker = driver(world)
        old_tid = worker.state.active_turn_id
        assert old_tid is not None
        assert worker.store.list_sessions() == ("child", "s")
        old_token = worker.scope.token
        world.push("s", InboxItem("l3-cancel", "control", {"action": "cancel"}, old_tid, 0))
        world.push("s", InboxItem("next", "follow_up", {"content": "continue"}))
        # Pause after the control matched old_tid, before the token attribute is read.
        # Source lookup locates a scheduling cut; all verdicts below use SQL/driver outcomes.
        source, start_line = inspect.getsourcelines(kernel._Scope.poll)
        cut_line = start_line + next(i for i, line in enumerate(source) if "self.token.cancel(item.payload" in line)
        reached = []

        def switch_turn(frame, event, _arg):
            if event == "line" and frame.f_code is kernel._Scope.poll.__code__ and frame.f_lineno == cut_line:
                sys.settrace(None)
                reached.append(old_tid)
                until(worker, lambda: worker.state.active_turn_id not in {None, old_tid})
                assert terminal_status(worker.records, old_tid) == "cancelled"
                assert not worker.scope.token.cancelled
            return switch_turn

        def late_poll():
            previous = sys.gettrace()
            try:
                sys.settrace(switch_turn)
                worker.scope.poll(world.connection("manual/heartbeat"))
            finally:
                sys.settrace(previous)

        world.enqueue("l3/target-matched-switch-turn-resume-control", late_poll)
        assert world.step()
        assert reached == [old_tid]
        new_token = worker.scope.token
        assert new_token is not old_token and new_token.cancelled and new_token.reason == "cancel"
        assert not old_token.cancelled
        new_tid = worker.state.active_turn_id
        # Durable parent cancellation has already queued the child's targeted control.
        child_controls = [i.item for i in worker.store.peek_inbox("child") if i.item.kind == "control"]
        assert len(child_controls) == 1 and child_controls[0].payload["action"] == "cancel"
        world.drive("child", participant="child-worker")
        child_state, child_rows, _ = worker.store.read_state("child")
        assert child_state.active_turn_id is None
        assert child_rows[-1].record.payload["status"] == "cancelled"

        def finish_next(request):
            world.calls.append(("s", request.metadata["operation_id"], request.metadata["attempt"], "model", None))
            return LLMResponse("next completed")

        worker.runtime.llm = ScriptedLLM([finish_next])
        until(worker, lambda: worker.state.turns[new_tid].ended)
        assert terminal_status(worker.records, new_tid) == "completed"
        assert worker.scope.token is not new_token and not worker.scope.token.cancelled
        world.check(quiescent=True, every_prefix=True)


@pytest.mark.parametrize("skew_ms", [-60000, 60000], ids=["wall-behind", "wall-ahead"])
def test_l4_opaque_wall_deadline_cannot_bypass_real_sql_fence(tmp_path, skew_ms):
    with SimWorld(tmp_path / "l4.sqlite", seed=L4_SEED) as world:
        world.install_write_observer()
        world.create("deadline")
        worker = driver(world, "deadline")
        until(worker, lambda: worker.state.active_turn_id is not None)
        lease = worker.scope.lease

        class SkewClock:
            def now_ms(self):
                return world.clock.now_ms() + skew_ms

            def __getattr__(self, name):
                return getattr(world.clock, name)

        worker.runtime.clock = SkewClock()
        # Omit now_ms to exercise the opaque-store fallback, without a fake store.
        worker.scope = kernel._Scope(lease, worker.runtime)
        before = worker.store.read("deadline").records
        if skew_ms > 0:
            with pytest.raises(LeaseLost):
                worker.commit([])
            assert worker.store.read("deadline").records == before
            assert worker.store.release(lease)
        else:
            world.advance(15000 + world.choices.choose("l4/after-expiry-ms", [1, 2, 3]))
            replacement = world.connection("competitor").acquire("deadline", owner="replacement", ttl_ms=15000)
            assert replacement is not None and replacement.epoch == lease.epoch + 1
            with worker.scope.committing():
                pass  # The local deadline is wrong and still admits entry.
            with pytest.raises(LeaseLost):
                worker.commit([])  # The actual SQLite append must reject it.
            assert worker.store.read("deadline").records == before
            assert world.connection("competitor").release(replacement)
        assert not world.calls
        world.enqueue("l4/recover-with-db-time", lambda: world.drive("deadline", participant="recovery"))
        assert world.step()
        state, rows, _ = worker.store.read_state("deadline")
        assert state.active_turn_id is None and rows[-1].record.payload["status"] == "completed"
        world.check(quiescent=True, every_prefix=True)
