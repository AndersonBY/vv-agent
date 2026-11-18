"""SQL worker compatibility, preclaimed core drive and tenant-page acceptance."""

from __future__ import annotations

import pytest

from .scenarios import run_scenario
from .world import SimWorld


def _assert_worker_compatibility_evidence(world, summary):
    assert summary["scenario"] == "worker_compatibility"
    assert summary["pending"] == {}
    assert "I11" in summary["covered"]
    assert summary["calls"] and all(call[3] == "model" for call in summary["calls"])

    incompatible = {"level-1", "handler-other", "capability-missing", "tool-schema-other", "tool-metadata-other", "model-other"}
    claims = {item["owner"]: item for item in summary["worker_claims"] if item.get("owner") in incompatible}
    assert set(claims) == incompatible
    for claim in claims.values():
        assert claim["matched"] is False
        assert claim["lease_epoch_after"] == claim["lease_epoch_before"]
        assert claim["decrypts_before"] == claim["decrypts_after"]

    extra = next(item for item in summary["worker_claims"] if item.get("owner") == "extra-capability")
    exact = next(item for item in summary["worker_claims"] if item.get("owner") == "compatible")
    assert extra["matched"] is True and exact["matched"] is True
    assert any(event.get("preclaimed_core_drive") == "worker-compatible" for event in summary["events"])
    root = next(item for item in summary["worker_claims"] if item.get("owner") == "root")
    assert root["same_session_id"] is True and root["paged"] is True and root["tenant_fenced"] is True
    assert root["tenant_ids"] == ["tenant-a", "tenant-b"]
    assert any(item.get("owner") == "cache-tamper" and item.get("cache_rejected") for item in summary["worker_claims"])

    store = world.connection("worker/assertions")
    store.configure_reader(level=2)
    target = store.read("worker-compatible")
    assert target.records[0].record.feature_level == 2
    assert target.records[0].record._payload["bootstrap"]["level"] == 2
    assert store.read_input("worker-compatible", "start").consumed_seq is not None
    target_terminal = [row.record for row in target.records if row.record.kind == "turn_ended"]
    assert len(target_terminal) == 1 and target_terminal[0].payload["status"] == "completed"
    pending = store.read_input("worker-unmatched", "start")
    assert pending is not None and pending.consumed_seq is None
    unmatched = store.read("worker-unmatched")
    assert not any(row.record.kind in {"turn_started", "turn_ended"} for row in unmatched.records)
    assert "worker-unmatched" in {work.session_id for work in store.list_runnable()}
    for tenant_id in ("tenant-a", "tenant-b"):
        scoped = store.for_tenant(tenant_id)
        shared = scoped.read("worker-shared")
        assert shared.records[0].record.feature_level == 2
        assert scoped.read_input("worker-shared", "start").consumed_seq is None


@pytest.mark.parametrize(("seed", "signed"), [(17, False), (43, True)])
def test_worker_compatibility_real_sql_claims(tmp_path, seed, signed):
    with SimWorld(tmp_path / f"worker-compatibility-{seed}.sqlite", seed=seed, signed=signed) as world:
        summary = run_scenario(world, "worker_compatibility")
        world.check(every_prefix=True)
        _assert_worker_compatibility_evidence(world, summary)
        if signed:
            assert "I9" in summary["covered"]


def test_worker_compatibility_replay_is_fixed_by_seed(tmp_path):
    with SimWorld(tmp_path / "worker-compatibility-original.sqlite", seed=17) as world:
        original = run_scenario(world, "worker_compatibility")
    with SimWorld(
        tmp_path / "worker-compatibility-replay.sqlite",
        seed=999,
        replay=original["decisions"],
    ) as world:
        replayed = run_scenario(world, "worker_compatibility")
        assert world.choices.used == set(world.choices.overrides)
    assert original["events"] == replayed["events"]
    assert original["calls"] == replayed["calls"]
    assert original["worker_claims"] == replayed["worker_claims"]
