"""Bounded W1 exploration and negative controls on real durable storage."""

import json
from dataclasses import replace

import pytest

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.session.records import Record
from vv_agent.session.result import project_result
from vv_agent.session.runtime import Runtime

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
    actual = Runtime.definition_digest

    def trust_retained_digest(self, task):
        return task.payload["definition_digest"] if isinstance(task, Record) else actual(self, task)

    monkeypatch.setattr(Runtime, "definition_digest", trust_retained_digest)
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
        calls = project_result(store, "s", model.turn_id).token_usage.model_calls
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
