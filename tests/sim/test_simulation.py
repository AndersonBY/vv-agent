"""Bounded real-driver recovery, reproducibility and replay oracles."""

import os

import pytest

from .report import DEFAULT_WORKERS, run_campaign, seed_results, versions
from .scenarios import SCENARIOS, run_scenario
from .world import SimWorld


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_real_driver_recovery(sim_database, scenario):
    with SimWorld(sim_database, seed=17) as world:
        result = run_scenario(world, scenario)
        assert result["calls"]
        world.check(quiescent=True, every_prefix=True)


def test_seed_replay(tmp_path):
    with SimWorld(tmp_path / "first.sqlite", seed=17) as first:
        original = run_scenario(first, "tools")
    with SimWorld(tmp_path / "replay.sqlite", seed=999, replay=original["decisions"]) as second:
        replayed = run_scenario(second, "tools")
        assert second.choices.used == set(second.choices.overrides)
    assert original["events"] == replayed["events"]
    assert original["calls"] == replayed["calls"]


def test_simulation_smoke():
    # The PR gate is 1000; explicit smaller overrides are development samples.
    count = int(os.environ.get("VV_AGENT_SIM_SEEDS", "1000"))
    assert count > 0
    result = run_campaign(seeds=count, workers=int(os.environ.get("VV_AGENT_SIM_WORKERS", str(DEFAULT_WORKERS))))
    assert "error" not in result, result
    print(result)
    if count >= 1000:
        assert result["elapsed_seconds"] < count * 0.06, "W1 throughput acceptance remains unmet"


@pytest.mark.parametrize("fault,invariant", [("external", "I1"), ("inbox", "I3")])
def test_step_oracles_detect_changes_without_new_records(sim_database, fault, invariant):
    from vv_agent.runtime.cancellation import CancellationToken

    from .world import InvariantViolation

    with SimWorld(sim_database, seed=17) as world:
        run_scenario(world, "tools")
        store = world.connection("observer")
        before = store._head("s")[:2]
        if fault == "inbox":
            host = world.connection("host")
            with host.atomic():
                host._rows("UPDATE sk_inbox SET consumed_seq=NULL WHERE session_id='s' AND input_id='initial'")
        else:
            rt = world.runtimes[("s", "worker-0")]
            state, _, _ = store.read_state("s")
            operation = next(op for op in state.operations.values() if op.kind == "tool")
            plan = operation.attempts[1].execution_plan
            task = state.turns[operation.turn_id].start._task()
            context = rt.context(task, plan, CancellationToken(), store=store, session_id="s")
            rt.providers["effect"].submit(plan, context=context)
        assert store._head("s")[:2] == before
        with pytest.raises(InvariantViolation, match=invariant):
            world.check()


def test_empty_scan_cannot_claim_i4(sim_database):
    with SimWorld(sim_database, seed=17) as world:
        run_scenario(world, "tools")
        world.covered.discard("I4")
        assert not world.scan()
        world.check(quiescent=True)
        assert "I4" not in world.summary()["covered"]


def test_accounting_oracle_rejects_lost_committed_measurement(sim_database, monkeypatch):
    from vv_agent.session import result

    from .world import InvariantViolation

    with SimWorld(sim_database, seed=17) as world:
        run_scenario(world, "model_result_precommit")
        normalize = result.model_usage
        monkeypatch.setattr(result, "model_usage", lambda _usage: normalize(None))
        with pytest.raises(InvariantViolation, match="I5: provider usage"):
            world.check_accounting()


def test_successful_world_summary_does_not_hash_source(tmp_path, monkeypatch):
    from . import report

    def forbidden():
        raise AssertionError("per-world source hashing is forbidden")

    monkeypatch.setattr(report, "versions", forbidden)
    with SimWorld(tmp_path / "summary.sqlite", seed=17) as world:
        assert "code_digest" not in run_scenario(world, "tools")


def test_spawned_seeds_match_serial_real_driver_traces():
    kwargs = {"seed": 17, "seeds": 2, "scenario": "model_result_precommit"}
    serial = list(seed_results(**kwargs, workers=1))
    spawned = list(seed_results(**kwargs, workers=2))
    assert all("error" not in result for result in serial + spawned), serial + spawned
    assert serial == spawned


@pytest.mark.parametrize(
    "field,value",
    [
        ("code_digest", "different"),
        ("contract_version", "0"),
        ("contract_revision", "different"),
        ("in_memory_mutations", [{"mutation": "retry_key"}]),
    ],
)
def test_campaign_rejects_unmatched_replay_before_execution(field, value):
    identity = versions()
    result = run_campaign(replay=identity | {field: value}, workers=1)
    assert "Replay provenance mismatch" in result["error"]
    assert field in result["replay_provenance"]["differences"]


def test_campaign_identity_once_and_matching_replay(tmp_path, monkeypatch):
    from . import report

    identity, calls = versions(), []

    def identify_once():
        calls.append(True)
        return identity

    monkeypatch.setattr(report, "versions", identify_once)
    with SimWorld(tmp_path / "original.sqlite", seed=17) as world:
        original = run_scenario(world, "model_result_precommit") | identity
    result = run_campaign(replay=original, workers=1)
    assert "error" not in result, result
    assert calls == [True]
    assert result["replay_provenance"]["matched"]
    assert result["code_digest"] == identity["code_digest"]
    assert result["decisions"] == len(original["decisions"])


@pytest.mark.parametrize("workers", [0, 9])
def test_worker_cap_rejects_invalid_counts_before_any_seed(workers):
    with pytest.raises(ValueError, match="workers"):
        next(seed_results(workers=workers))


def test_empty_accounting_cannot_claim_i5(tmp_path):
    with SimWorld(tmp_path / "unstarted.sqlite") as world:
        world.create()
        world.check_accounting()
        assert "I5" not in world.summary()["covered"]
