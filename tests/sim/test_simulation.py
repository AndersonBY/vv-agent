"""Bounded real-driver recovery, reproducibility and replay oracles."""

import hashlib
import json
import os
from concurrent.futures import Executor, Future

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
    from .assembly import CallContext
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
            context = CallContext(plan._payload["idempotency_key"], plan)
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
    from vv_agent.kernel import _usage

    from .world import InvariantViolation

    with SimWorld(sim_database, seed=17) as world:
        run_scenario(world, "model_result_precommit")
        normalize = _usage.model_usage
        monkeypatch.setattr(_usage, "model_usage", lambda _usage: normalize(None))
        with pytest.raises(InvariantViolation, match="I5: provider usage"):
            world.check_accounting()


def test_successful_world_summary_does_not_hash_source(tmp_path, monkeypatch):
    from . import report

    def forbidden():
        raise AssertionError("per-world source hashing is forbidden")

    monkeypatch.setattr(report, "versions", forbidden)
    with SimWorld(tmp_path / "summary.sqlite", seed=17) as world:
        assert "code_digest" not in run_scenario(world, "tools")


def test_report_identity_carries_canonical_scenario_inventory():
    from . import report

    identity = report.versions()
    inventory = sorted(SCENARIOS)
    assert identity["scenario_inventory"] == inventory
    assert (
        identity["scenario_inventory_sha256"]
        == hashlib.sha256(json.dumps(inventory, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    )


def test_context_compaction_calls_real_normalizer_and_token_counter(sim_database, monkeypatch):
    from . import assembly

    normalized, counted = [], []
    actual_normalize = assembly.normalize_memory_settings
    actual_count = assembly.count_messages_tokens

    def normalize(settings, protected_start):
        result = actual_normalize(settings, protected_start)
        normalized.append(result)
        return result

    def count(messages, *, model):
        result = actual_count(messages, model=model)
        counted.append((len(messages), model, result))
        return result

    monkeypatch.setattr(assembly, "normalize_memory_settings", normalize)
    monkeypatch.setattr(assembly, "count_messages_tokens", count)
    with SimWorld(sim_database, seed=17) as world:
        run_scenario(world, "context_compacted")
    assert normalized and all(value["compact_threshold"] == 1 for value in normalized)
    assert counted and all(result > 0 for _messages, _model, result in counted)


def test_spawned_seeds_match_serial_real_driver_traces():
    kwargs = {"seed": 17, "seeds": 5, "scenario": "model_result_precommit"}
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
    storage = report.temporary_storage()
    assert result["environment"]["sqlite_temporary_storage"] == {
        "verified": storage["filesystem"] is not None,
        "observations": [storage],
    }


@pytest.mark.parametrize("workers", [0, 9])
def test_worker_cap_rejects_invalid_counts_before_any_seed(workers):
    with pytest.raises(ValueError, match="workers"):
        next(seed_results(workers=workers))


def test_empty_accounting_cannot_claim_i5(tmp_path):
    with SimWorld(tmp_path / "unstarted.sqlite") as world:
        world.create()
        world.check_accounting()
        assert "I5" not in world.summary()["covered"]


@pytest.fixture
def controlled_seed_pool(monkeypatch):
    from . import report

    submitted, completed, closed = [], [], []
    failures = set()

    class Pool(Executor):
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 2
            assert mp_context.get_start_method() == "spawn"

        def __enter__(self):
            return self

        def submit(self, fn, /, *args, **kwargs):
            assert fn is report._run_seed and len(args) == 1 and not kwargs
            number = args[0][1]
            future = Future()
            submitted.append((number, future))
            # Complete the first window backwards, so result order cannot rely
            # on completion order. Later jobs complete before older yields too.
            ready = list(reversed(submitted)) if len(submitted) == 4 else [(number, future)] if len(submitted) > 4 else []
            for seed, pending in ready:
                result = {"seed": seed, "scenario": "controlled", "decisions": [], "covered": []}
                if seed in failures:
                    result["error"] = f"seed {seed} failed"
                pending.set_result(result)
                completed.append(seed)
            return future

        def __exit__(self, exc_type, exc_val, exc_tb):
            closed.append(True)

    monkeypatch.setattr(report, "ProcessPoolExecutor", Pool)
    return submitted, completed, closed, failures


def test_seed_submission_is_bounded_and_preserves_input_order(controlled_seed_pool):
    submitted, completed, closed, _failures = controlled_seed_pool
    results = seed_results(seeds=9, workers=2)
    for expected in range(9):
        assert next(results)["seed"] == expected
        assert [seed for seed, _future in submitted] == list(range(min(9, 4 * (expected // 4 + 1))))
        assert len(submitted) - expected <= 4
    assert completed[:4] == [3, 2, 1, 0]
    with pytest.raises(StopIteration):
        next(results)
    assert closed == [True]


def test_campaign_stops_at_first_ordered_failure(controlled_seed_pool):
    submitted, completed, closed, failures = controlled_seed_pool
    failures.update({1, 2})
    result = run_campaign(seeds=9, workers=2)
    assert completed.index(2) < completed.index(1)
    assert result["seed"] == 1 and result["error"] == "seed 1 failed"
    assert result["environment"]["sqlite_temporary_storage"] == {"verified": False, "observations": []}
    assert [seed for seed, _future in submitted] == [0, 1, 2, 3]
    assert closed == [True]


@pytest.fixture
def local_contract(tmp_path):
    from .report import ROOT

    lock = json.loads((ROOT / "contract.lock.json").read_text())
    source = tmp_path / "canonical"
    fixtures = source / "fixtures"
    fixtures.mkdir(parents=True)
    body = b'{"vector": 1}\n'
    (fixtures / "sample.json").write_bytes(body)
    manifest = f"{hashlib.sha256(body).hexdigest()}  sample.json\n".encode()
    (fixtures / "SHA256SUMS").write_bytes(manifest)
    metadata = {
        "schema_version": 1,
        "name": "vv-agent-contract",
        "version": f"{int(lock['contract_version'].split('.')[0]) + 1}.0.0",
        "fixtures": {"path": "fixtures", "manifest_sha256": hashlib.sha256(manifest).hexdigest()},
    }
    (source / "contract.json").write_text(json.dumps(metadata))
    return source, metadata, lock


def test_released_contract_identity_matches_validated_pin(monkeypatch):
    from .report import ROOT

    monkeypatch.delenv("VV_AGENT_CONTRACT_SOURCE", raising=False)
    lock = json.loads((ROOT / "contract.lock.json").read_text())
    identity = versions()
    assert identity["contract_selection"] == "released"
    assert identity["contract_version"] == lock["contract_version"]
    assert identity["contract_revision"] == lock["contract_revision"]
    assert identity["contract_manifest_sha256"] == lock["fixture_manifest_sha256"]
    assert identity["pinned_contract"] == {key: lock[key] for key in identity["pinned_contract"]}


def test_local_contract_identity_keeps_effective_and_pinned_separate(local_contract, monkeypatch):
    from . import report

    source, metadata, lock = local_contract
    monkeypatch.setenv("VV_AGENT_CONTRACT_SOURCE", str(source))
    actual_git = report._git

    def local_git(root, *args):
        if root == source:
            return {
                ("rev-parse", "--show-toplevel"): str(source),
                ("rev-parse", "HEAD"): "a" * 40,
                ("status", "--porcelain"): " M contract.json",
            }[args]
        return actual_git(root, *args)

    monkeypatch.setattr(report, "_git", local_git)
    identity = versions()
    assert identity["contract_selection"] == "local"
    assert identity["contract_version"] == metadata["version"] != lock["contract_version"]
    assert identity["contract_revision"] == "a" * 40
    assert identity["contract_dirty"] is True
    assert identity["contract_manifest_sha256"] == metadata["fixtures"]["manifest_sha256"]
    assert identity["pinned_contract"]["contract_version"] == lock["contract_version"]
    assert identity["pinned_contract"]["contract_revision"] == lock["contract_revision"]


def test_local_archive_does_not_inherit_parent_git_revision(local_contract, monkeypatch):
    from . import report

    source, metadata, _lock = local_contract
    monkeypatch.setenv("VV_AGENT_CONTRACT_SOURCE", str(source))
    actual_git = report._git

    def parent_git(root, *args):
        if root == source:
            assert args == ("rev-parse", "--show-toplevel"), "must not read the parent repository's revision"
            return str(source.parent)
        return actual_git(root, *args)

    monkeypatch.setattr(report, "_git", parent_git)
    identity = versions()
    assert identity["contract_selection"] == "local" and identity["contract_version"] == metadata["version"]
    assert identity["contract_revision"] is None and identity["contract_dirty"] is None


def test_replay_rejects_effective_contract_selection_change(local_contract, monkeypatch):
    source, _metadata, _lock = local_contract
    monkeypatch.delenv("VV_AGENT_CONTRACT_SOURCE", raising=False)
    released = versions()
    monkeypatch.setenv("VV_AGENT_CONTRACT_SOURCE", str(source))
    result = run_campaign(replay=released, workers=1)
    assert "Replay provenance mismatch" in result["error"]
    assert "contract_selection" in result["replay_provenance"]["differences"]
    assert result["pinned_contract"] == released["pinned_contract"]


def test_replay_rejects_changed_local_manifest(local_contract, monkeypatch):
    source, metadata, _lock = local_contract
    monkeypatch.setenv("VV_AGENT_CONTRACT_SOURCE", str(source))
    recorded = versions()
    body = b'{"vector": 2}\n'
    (source / "fixtures/sample.json").write_bytes(body)
    manifest = f"{hashlib.sha256(body).hexdigest()}  sample.json\n".encode()
    (source / "fixtures/SHA256SUMS").write_bytes(manifest)
    metadata["fixtures"]["manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
    (source / "contract.json").write_text(json.dumps(metadata))
    result = run_campaign(replay=recorded, workers=1)
    assert "Replay provenance mismatch" in result["error"]
    assert "contract_manifest_sha256" in result["replay_provenance"]["differences"]


@pytest.mark.parametrize(
    "invalid", ["missing", "empty", "bad_json", "bad_version", "bad_manifest", "missing_vector", "changed_vector"]
)
def test_invalid_explicit_contract_source_never_falls_back(local_contract, monkeypatch, invalid):
    from . import report

    SnapshotError = report._snapshot_tools().SnapshotError
    source, metadata, _lock = local_contract
    selected = str(source)
    if invalid == "missing":
        selected = str(source / "absent")
    elif invalid == "empty":
        selected = ""
    elif invalid == "bad_json":
        (source / "contract.json").write_text("not JSON")
    elif invalid in {"bad_version", "bad_manifest"}:
        if invalid == "bad_version":
            metadata["version"] = "unknown"
        else:
            metadata["fixtures"]["manifest_sha256"] = "0" * 64
        (source / "contract.json").write_text(json.dumps(metadata))
    elif invalid == "missing_vector":
        (source / "fixtures/sample.json").unlink()
    else:
        (source / "fixtures/sample.json").write_text("tampered")
    monkeypatch.setenv("VV_AGENT_CONTRACT_SOURCE", selected)

    def forbidden(_job):
        pytest.fail("invalid contract source reached seed execution")

    monkeypatch.setattr(report, "_run_seed", forbidden)
    with pytest.raises(SnapshotError):
        run_campaign(seeds=1, workers=1)


@pytest.mark.parametrize("directory", ["/tmp", "/dev/shm"])
def test_report_identifies_actual_sqlite_file_filesystem(directory, monkeypatch):
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    from vv_agent.kernel._sqlite import SQLiteStore

    from .assembly import fold_factory
    from .report import database, temporary_storage
    from .world import SystemClock

    if not Path(directory).is_dir() or shutil.which("stat") is None:
        pytest.skip("filesystem probe requires this temporary directory and stat")
    monkeypatch.setenv("TMPDIR", directory)
    monkeypatch.setattr(tempfile, "tempdir", None)
    metadata = temporary_storage()
    assert metadata["directory"] == str(Path(directory).resolve())
    assert metadata["requested_TMPDIR"] == directory
    with database("sqlite") as path, SQLiteStore.standalone(str(path), fold_factory=fold_factory, clock=SystemClock()) as store:
        store.install_schema()
        assert path.is_file() and path.parent.parent.resolve() == Path(metadata["directory"])
        actual = subprocess.run(["stat", "-f", "-c", "%T", "--", str(path)], capture_output=True, text=True, check=False)
        assert metadata["filesystem"] == (actual.stdout.strip() if actual.returncode == 0 else None)
        if directory == "/dev/shm" and actual.returncode == 0:
            assert metadata["filesystem"] == "tmpfs"
        assert store.connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert store.connection.execute("PRAGMA synchronous").fetchone() == (2,)


def test_unavailable_filesystem_probe_reports_unknown(monkeypatch):
    from . import report

    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("stat is unavailable")

    monkeypatch.setattr(report.subprocess, "run", unavailable)
    report._filesystem.cache_clear()
    try:
        assert report.temporary_storage()["filesystem"] is None
    finally:
        report._filesystem.cache_clear()


def test_campaign_reports_spawned_worker_storage_not_parent_tempfile_cache(monkeypatch):
    import tempfile
    from pathlib import Path

    if not Path("/dev/shm").is_dir():
        pytest.skip("requires Linux shared-memory filesystem")
    monkeypatch.setattr(tempfile, "tempdir", "/tmp")
    monkeypatch.setenv("TMPDIR", "/dev/shm")
    assert tempfile.gettempdir() == "/tmp"
    result = run_campaign(seeds=1, workers=2, scenario="tools")
    assert "error" not in result, result
    assert tempfile.gettempdir() == "/tmp"
    assert result["environment"]["sqlite_temporary_storage"] == {
        "verified": True,
        "observations": [{"directory": "/dev/shm", "filesystem": "tmpfs", "requested_TMPDIR": "/dev/shm"}],
    }


def test_same_content_local_archive_replays_without_git_metadata(local_contract, tmp_path, monkeypatch):
    import shutil

    from . import report

    source, _metadata, _lock = local_contract
    actual_git = report._git

    def source_git(root, *args):
        if root == source:
            return {
                ("rev-parse", "--show-toplevel"): str(source),
                ("rev-parse", "HEAD"): "a" * 40,
                ("status", "--porcelain"): "",
            }[args]
        return actual_git(root, *args)

    monkeypatch.setattr(report, "_git", source_git)
    monkeypatch.setenv("VV_AGENT_CONTRACT_SOURCE", str(source))
    with SimWorld(tmp_path / "original.sqlite", seed=17) as world:
        recorded = run_scenario(world, "tools") | versions()
    archive = tmp_path / "archive"
    shutil.copytree(source, archive)
    monkeypatch.setenv("VV_AGENT_CONTRACT_SOURCE", str(archive))
    current = versions()
    assert recorded["contract_revision"] == "a" * 40 and current["contract_revision"] is None
    assert current["contract_manifest_sha256"] == recorded["contract_manifest_sha256"]
    replayed = run_campaign(replay=recorded, workers=1)
    assert "error" not in replayed, replayed
    assert replayed["replay_provenance"]["matched"]
    assert replayed["replay_provenance"]["differences"]["contract_revision"] == {"recorded": "a" * 40, "current": None}
    for field in (
        "code_digest",
        "scenario_inventory",
        "scenario_inventory_sha256",
        "contract_selection",
        "contract_version",
        "contract_manifest_sha256",
        "in_memory_mutations",
    ):
        altered = recorded | {field: "changed"}
        assert not report.replay_provenance(altered, current)["matched"], field
    for field in ("scenario_inventory", "scenario_inventory_sha256"):
        missing = dict(recorded)
        missing.pop(field)
        assert not report.replay_provenance(missing, current)["matched"], field
    assert not report.replay_provenance(recorded, current | {"contract_revision": "b" * 40})["matched"]
