"""Bounded W1 exploration and negative controls on real durable storage."""

import json
from dataclasses import replace

import pytest

from vv_agent.kernel import _child_store, _driver_outcomes
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._definition import DefinitionCache
from vv_agent.kernel._records import Record
from vv_agent.kernel._reducer import TransitionError
from vv_agent.kernel._usage import project_model_usage

from .scenarios import Provider, run_scenario
from .world import InvariantViolation, SimWorld


@pytest.mark.parametrize("scenario", ["tools", "accepted_lost"])
def test_scanner_is_only_recovery_after_ready_active_hint_loss(sim_database, scenario):
    with SimWorld(sim_database, seed=17) as world:
        result = run_scenario(world, scenario)
        lost = next(i for i, event in enumerate(result["events"]) if "ready_active_head" in event)
        assert result["events"][lost]["lost_hints"] > 0
        assert any(event.get("scan_candidates") for event in result["events"][lost + 1 :])
        assert sum(call[3] == "tool" for call in world.calls) == 1
        assert world.connection("provider")._one("SELECT count(*) FROM sim_effect")[0] == 1
        if result["backend"] == "sqlite":
            connection = world.connection("observer").connection
            assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
            assert connection.execute("PRAGMA synchronous").fetchone() == (2,)
        world.check(quiescent=True, every_prefix=True)


@pytest.mark.parametrize("scenario", ["tools", "accepted_lost"])
def test_disabling_scanner_fails_i4_with_ready_active_work(sim_database, monkeypatch, scenario):
    with SimWorld(sim_database, seed=17) as world:
        monkeypatch.setattr(world.connection("scanner"), "list_runnable", lambda **_kwargs: ())
        with pytest.raises(InvariantViolation, match="I4: fair SQL scan"):
            run_scenario(world, scenario)
        state, _, _ = world.connection("observer").read_state("s")
        assert state.active_turn_id is not None
        assert world.connection("observer").is_runnable("s")
        assert not world.queue


@pytest.mark.parametrize("seed,cut", [(0, "before_send"), (1, "after_model"), (2, "planned")])
def test_competing_workers_renew_expire_and_take_over_at_driver_checkpoints(sim_database, seed, cut):
    with SimWorld(sim_database, seed=seed) as world:
        result = run_scenario(world, "lease_takeover")
        assert [event["takeover_cut"] for event in result["events"] if "takeover_cut" in event] == [cut]
        assert len([event for event in result["events"] if "contender_refused" in event]) == 1
        assert len([event for event in result["events"] if "renewed_owner" in event]) == 1
        assert len([event for event in result["events"] if "stale_renewal_rejected" in event]) == 1
        assert sum(call[3] == "tool" for call in world.calls) == 1
        world.check(quiescent=True, every_prefix=True)


def test_frozen_definition_mismatch_fails_without_more_external_sends(sim_database):
    with SimWorld(sim_database, seed=17) as world:
        run_scenario(world, "definition_mismatch")
        rows = world.connection("observer").read("s").records
        terminal = next(row.record for row in rows if row.record.kind == "turn_ended")
        assert terminal.payload["reason"] == "handler_schema_or_capability_mismatch"
        assert sum(call[3] == "model" for call in world.calls) == 1
        assert sum(call[3] == "tool" for call in world.calls) == 1


def test_definition_evidence_detects_bypassed_runtime_digest_check(sim_database, monkeypatch):
    actual = DefinitionCache.definition_digest

    def trust_retained_digest(self, task, bindings, catalog):
        return task.payload["definition_digest"] if isinstance(task, Record) else actual(self, task, bindings, catalog)

    monkeypatch.setattr(DefinitionCache, "definition_digest", trust_retained_digest)
    with SimWorld(sim_database, seed=17) as world:
        with pytest.raises(InvariantViolation, match="definition:"):
            run_scenario(world, "definition_mismatch")
        assert sum(call[3] == "model" for call in world.calls) == 2


def test_precommit_model_result_recovers_as_meter_missing_then_one_retry(sim_database):
    with SimWorld(sim_database, seed=17) as world:
        run_scenario(world, "model_result_precommit")
        store = world.connection("observer")
        state, rows, _ = store.read_state("s")
        model = next(operation for operation in state.operations.values() if operation.kind == "model")
        oid = model.attempts[1].plan.operation_id
        usage, _ledger = project_model_usage(state, model.turn_id)
        calls = usage.model_calls
        attempts = {call.attempt: call for call in calls if call.operation_id == oid}
        assert set(attempts) == {1, 2}
        assert attempts[1].usage.usage_source.value == "accounting_missing"
        assert attempts[2].usage.input_tokens == 11 and attempts[2].usage.output_tokens == 3
        assert not any(
            row.record.kind == "op_completed" and row.record.operation_id == oid and row.record.attempt == 1 for row in rows
        )
        assert model.attempts[1].unknown.payload["measurement_missing"]
        assert model.attempts[1].unknown.payload["duplicate_cost_risk"]
        world.check_accounting()


def test_metered_tool_authenticates_usage_and_retains_original_accounting(sim_database):
    with SimWorld(sim_database, seed=17) as world:
        run_scenario(world, "metered_tool")
        state, rows, _ = world.connection("observer").read_state("s")
        completion = next(
            row.record for row in rows if row.record.kind == "op_completed" and "/tool/" in (row.record.operation_id or "")
        )
        assert completion.payload["usage"]["prompt_tokens"] == 12000
        service_usage = json.loads(world.connection("provider")._one("SELECT usage FROM sim_effect")[0])
        assert service_usage["prompt_tokens"] == 12001  # Authenticated conflicting service revision was refused.
        assert state.applied_inputs["usage-identical"].payload["disposition"] == "noop"
        assert state.applied_inputs["usage-conflict"].payload["reason"] == "result conflict"
        assert all(state.applied_inputs[key].payload["disposition"] == "rejected" for key in ("usage-forged", "result-forged"))
        world.check(quiescent=True, every_prefix=True)


def test_metered_evidence_detects_authentication_that_ignores_usage(sim_database, monkeypatch):
    actual = Provider.authenticate

    def ignore_supplied_usage(self, item, plan):
        if item.kind == "provider_result":
            usage = json.loads(self.store._one("SELECT usage FROM sim_effect")[0])
            item = replace(item, payload=item.payload | {"usage": usage})
        return actual(self, item, plan)

    monkeypatch.setattr(Provider, "authenticate", ignore_supplied_usage)
    with SimWorld(sim_database, seed=17) as world, pytest.raises(InvariantViolation, match=r"I10:.*forged"):
        run_scenario(world, "metered_tool")


def test_seeded_takeover_replays_the_same_interleaving(tmp_path):
    with SimWorld(tmp_path / "first.sqlite", seed=0) as original:
        first = run_scenario(original, "lease_takeover")
    with SimWorld(tmp_path / "replay.sqlite", seed=999, replay=first["decisions"]) as replay:
        second = run_scenario(replay, "lease_takeover")
        assert replay.choices.used == set(replay.choices.overrides)
    # SQLite's clock is injected; PostgreSQL intentionally keeps its real clock.
    assert canonical_json_bytes(first["events"]) == canonical_json_bytes(second["events"])
    assert first["calls"] == second["calls"]


def test_approval_binds_request_digest_scope_and_generation(sim_database):
    with SimWorld(sim_database, seed=17) as world:
        run_scenario(world, "approval_reply")
        state, rows, _ = world.connection("observer").read_state("s")
        for suffix in ("request", "digest", "scope", "turn", "generation"):
            assert state.applied_inputs[f"approval-{suffix}"].payload["disposition"] == "rejected"
        assert state.applied_inputs["approval-valid"].payload["disposition"] == "applied"
        assert all(state.applied_inputs[key].payload["disposition"] == "noop" for key in ("approval-copy", "approval-late-copy"))
        assert len([row for row in rows if row.record.kind == "op_started" and "/tool/" in row.record.operation_id]) == 1
        assert sum(call[3] == "tool" for call in world.calls) == 1
        world.check(quiescent=True, every_prefix=True)


def test_future_input_is_immutable_and_only_due_scanner_resumes(sim_database):
    with SimWorld(sim_database, seed=17) as world:
        summary = run_scenario(world, "future_input")
        held = next(event for event in summary["events"] if "future_input_held" in event)
        applied = next(event for event in summary["events"] if "future_input_applied" in event)
        assert held["observed_ms"] < held["available_ms"] == applied["available_ms"] <= applied["created_ms"]
        item = world.connection("observer").read_input("s", "future")
        assert item.item.available_ms == held["available_ms"] and item.consumed_seq is not None
        assert "I4" in summary["covered"]
        world.check(quiescent=True, every_prefix=True)


def test_future_input_oracle_detects_removed_sql_readiness_filter(sim_database, monkeypatch):
    from vv_agent.kernel._sql import SQLStore

    from .test_mutations import mutate

    monkeypatch.setattr(
        SQLStore,
        "peek_inbox",
        mutate(
            SQLStore.peek_inbox,
            'readiness = f"AND available_ms<={self.clock_sql} " if ready_only else ""',
            'readiness = ""',
        ),
    )
    with SimWorld(sim_database, seed=17) as world, pytest.raises(InvariantViolation, match="I3: future input appeared"):
        run_scenario(world, "future_input")


def _assert_i7_cancel_tree(world, result):
    active = next(event["active_cancellation_tree"] for event in result["events"] if "active_cancellation_tree" in event)
    assert active == {
        "s": "s/turn/initial",
        "child": "child/turn/start",
        "grandchild": "grandchild/turn/start",
    }
    assert [(sid, attempt) for sid, _op, attempt, kind, _key in world.calls if kind == "model"] == [
        ("s", 1),
        ("child", 1),
        ("grandchild", 1),
    ]
    for sid in active:
        state, rows, _ = world.connection("observer").read_state(sid)
        assert state.active_turn_id is None
        assert [row.record.payload["status"] for row in rows if row.record.kind == "turn_ended"] == ["cancelled"]
    assert world.connection("provider")._one("SELECT count(*) FROM sim_effect")[0] == 0
    assert "I7" in result["covered"]


def test_root_cancel_terminates_started_child_and_grandchild(sim_database):
    with SimWorld(sim_database, seed=17) as world:
        result = run_scenario(world, "cancel_descendants")
        _assert_i7_cancel_tree(world, result)
        world.check(quiescent=True, every_prefix=True)


@pytest.mark.parametrize("seed", [250229, 875459])
def test_final_evidence_seed_redrives_transient_executor_stop_before_cancel(tmp_path, seed):
    with SimWorld(tmp_path / f"seed-{seed}.sqlite", seed=seed) as world:
        actual_drive_runtime = world._drive_runtime
        interrupted = False

        def stop_grandchild_once(store, sid, *, runtime, factory):
            nonlocal interrupted
            if not interrupted and sid == "grandchild" and runtime.participant == "grandchild-worker":
                interrupted = True
                raise TimeoutError("executor stopped before claiming grandchild")
            return actual_drive_runtime(store, sid, runtime=runtime, factory=factory)

        world._drive_runtime = stop_grandchild_once
        result = run_scenario(world, "cancel_descendants")

        assert interrupted
        assert result["seed"] == seed
        assert result["decisions"][0] == {"at": "scenario#0", "value": "cancel_descendants"}
        assert [event["worker_stopped"] for event in result["events"] if "worker_stopped" in event] == ["grandchild-worker"]
        assert [event["executor_stop"] for event in result["events"] if "executor_stop" in event] == [
            {"session_id": "grandchild", "participant": "grandchild-worker", "exception_type": "TimeoutError"}
        ]
        assert [event["precondition_redrive"] for event in result["events"] if "precondition_redrive" in event] == [
            {
                "session_id": "grandchild",
                "participant": "grandchild-worker",
                "redrive": 1,
                "exception_type": "TimeoutError",
            }
        ]
        _assert_i7_cancel_tree(world, result)
        world.check(quiescent=True, every_prefix=True)


@pytest.mark.parametrize("seed", [250228, 250229, 250230, 875458, 875459, 875460])
def test_cancel_descendants_seed_range_preserves_three_generation_tree(tmp_path, seed):
    with SimWorld(tmp_path / f"seed-{seed}.sqlite", seed=seed) as world:
        result = run_scenario(world, "cancel_descendants")
        _assert_i7_cancel_tree(world, result)


def test_precondition_rejects_unrelated_participant_stop(tmp_path):
    with SimWorld(tmp_path / "unrelated-stop.sqlite", seed=250229) as world:
        actual_drive = world.drive

        def stop_unrelated_worker(store_sid="s", *, participant="worker-0", **kwargs):
            if store_sid == "grandchild" and participant == "grandchild-worker":
                world.events.append({"worker_stopped": "other-worker", "dead": False})
                return None
            return actual_drive(store_sid, participant=participant, **kwargs)

        world.drive = stop_unrelated_worker
        with pytest.raises(InvariantViolation, match="no matching executor stop for grandchild/grandchild-worker"):
            run_scenario(world, "cancel_descendants")
        assert not any("precondition_redrive" in event for event in world.events)


def test_precondition_does_not_swallow_non_executor_failure(tmp_path):
    with SimWorld(tmp_path / "non-executor.sqlite", seed=250229) as world:
        actual_drive_runtime = world._drive_runtime

        def fail_non_executor(store, sid, *, runtime, factory):
            if sid != "grandchild":
                return actual_drive_runtime(store, sid, runtime=runtime, factory=factory)
            raise InvariantViolation("synthetic", "non-executor failure")

        world._drive_runtime = fail_non_executor
        with pytest.raises(InvariantViolation, match="synthetic: non-executor failure"):
            run_scenario(world, "cancel_descendants")
        assert not any("executor_stop" in event or "precondition_redrive" in event for event in world.events)


def test_persistent_executor_stop_fails_after_three_bounded_drives(tmp_path):
    with SimWorld(tmp_path / "persistent-stop.sqlite", seed=875459) as world:
        actual_drive_runtime = world._drive_runtime
        attempts = []

        def always_stop(store, sid, *, runtime, factory):
            if sid != "grandchild":
                return actual_drive_runtime(store, sid, runtime=runtime, factory=factory)
            attempts.append((sid, runtime.participant))
            raise TimeoutError("persistent executor stop")

        world._drive_runtime = always_stop
        with pytest.raises(InvariantViolation, match="grandchild never became active after one drive and three bounded redrives"):
            run_scenario(world, "cancel_descendants")
        assert attempts == [("grandchild", "grandchild-worker")] * 4
        stops = [event["executor_stop"] for event in world.events if "executor_stop" in event]
        assert stops == [{"session_id": "grandchild", "participant": "grandchild-worker", "exception_type": "TimeoutError"}] * 4
        assert len([event for event in world.events if "precondition_redrive" in event]) == 3


def test_descendant_oracle_detects_missing_second_generation_cancel(sim_database, monkeypatch):
    actual = _child_store.cancel_children

    def omit_grandchild(store, tx, state, sid, tid):
        return [] if sid == "child" else actual(store, tx, state, sid, tid)

    monkeypatch.setattr(_child_store, "cancel_children", omit_grandchild)
    with SimWorld(sim_database, seed=17) as world:
        with pytest.raises(InvariantViolation, match="I7: cancellation did not propagate to grandchild"):
            run_scenario(world, "cancel_descendants")
        assert world.connection("observer").read_state("child")[0].active_turn_id is None
        assert world.connection("observer").read_state("grandchild")[0].active_turn_id == "grandchild/turn/start"


@pytest.mark.parametrize("scenario", ["approval_reply", "future_input", "cancel_descendants"])
def test_participant_cases_replay_deterministically(tmp_path, scenario):
    with SimWorld(tmp_path / "first.sqlite", seed=17) as original:
        first = run_scenario(original, scenario)
    with SimWorld(tmp_path / "replay.sqlite", seed=999, replay=first["decisions"]) as replay:
        second = run_scenario(replay, scenario)
        assert replay.choices.used == set(replay.choices.overrides)
    assert first["events"] == second["events"]
    assert first["calls"] == second["calls"]


def test_descendant_settlement_requires_real_terminal_receipt(sim_database, monkeypatch):
    actual = _driver_outcomes.completed_steps

    def false_terminal_receipt(driver, plan, outcome):
        attempt = driver.state.operations[plan.operation_id].attempts[plan.attempt]
        if driver.sid == "child" and attempt.child_handle:
            outcome = replace(outcome, evidence=("child/0/" + "0" * 64,))
        return (yield from actual(driver, plan, outcome))

    monkeypatch.setattr(_driver_outcomes, "completed_steps", false_terminal_receipt)
    with SimWorld(sim_database, seed=17) as world:
        with pytest.raises(TransitionError, match="child completion needs terminal input"):
            run_scenario(world, "cancel_descendants")
        observer = world.connection("observer")
        _, rows, _ = observer.read_state("child")
        assert not any(row.record.kind == "op_completed" and "/tool/" in row.record.operation_id for row in rows)
        assert any(item.item.kind == "child_result" and item.consumed_seq is None for item in observer.peek_inbox("child"))
        world.check()
