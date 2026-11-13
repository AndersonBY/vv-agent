from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("colab_campaign", ROOT / "scripts/colab_campaign.py")
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)

CODE = "a" * 40
CONTRACT = "b" * 40
CHILD = {"pid": 4321, "pgid": 4321, "start_ticks": 98765, "live": True}


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def completed(arguments, code=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(arguments, code, stdout=stdout, stderr=stderr)


def part(backend: str, start: int, count: int, coverage=()) -> dict:
    value = {
        "backend": backend,
        "seed": start,
        "seeds": count,
        "code_revision": CODE,
        "contract_revision": CONTRACT,
        "dirty": False,
        "contract_dirty": False,
        "code_digest": "d" * 64,
        "contract_manifest_sha256": "e" * 64,
        "in_memory_mutations": [],
        "pending": {"I9": "future", "I11": "future"},
        "scenarios": {"precommit": count},
        "covered": list(coverage),
        "environment": {},
    }
    if backend == "sqlite":
        value["environment"]["sqlite_temporary_storage"] = {
            "verified": True,
            "observations": [{"directory": "/tmp/run", "filesystem": "ext4"}],
        }
    return value


def payloads(sqlite_seeds=2, postgres_seeds=1) -> dict[str, bytes]:
    parts = {
        f"reports/sqlite-{0:010d}-{sqlite_seeds:010d}.json": part("sqlite", 0, sqlite_seeds),
        f"reports/postgres-{0:010d}-{postgres_seeds:010d}.json": part("postgres", 0, postgres_seeds),
    }
    result = {
        "status": "complete",
        "full_acceptance": False,
        "identity": {"vv_agent_revision": CODE, "contract_revision": CONTRACT},
        "campaigns": {},
        "runtime": {
            "postgres": {
                "temporary_role_removed": True,
                "temporary_databases_removed": True,
            }
        },
    }
    for backend, seeds in (("sqlite", sqlite_seeds), ("postgres", postgres_seeds)):
        names = sorted(Path(name).name for name in parts if Path(name).name.startswith(backend + "-"))
        result["campaigns"][backend] = {
            "seed": 0,
            "seeds": seeds,
            "scenarios": {"precommit": seeds},
            "covered": [],
            "pending": {"I9": "future", "I11": "future"},
            "reports": names,
        }
    value = {name: json.dumps(body).encode() for name, body in parts.items()}
    value["reports/result.json"] = json.dumps(result).encode()
    value["process-status.json"] = json.dumps({"state": "exited", "exit_code": 0, "child_identity": CHILD}).encode()
    value["child.stdout.log"] = b"complete\n"
    value["child.stderr.log"] = b""
    return value


def observation(files: dict[str, bytes], *, state="terminal_complete", wrong_hash=False) -> dict:
    artifacts = [{"relative_path": name, "sha256": "0" * 64 if wrong_hash else sha(body)} for name, body in sorted(files.items())]
    terminal = state != "running"
    status: dict[str, object] = {"state": "exited" if terminal else "running", "child_identity": CHILD}
    if terminal:
        status["exit_code"] = 0
    result = {"status": "complete", "full_acceptance": False, "has_error": False} if terminal else None
    return {
        "session": "session-1",
        "paths": {
            "job_dir": "/content/vv-agent-campaigns/job-1",
            "reports_dir": "/content/vv-agent-campaigns/job-1/reports",
        },
        "status": status,
        "recorded": CHILD,
        "current": CHILD if not terminal else {"pid": CHILD["pid"], "live": False},
        "identity_matches": not terminal,
        "launch_matches": True,
        "result": result,
        "artifacts": artifacts,
    }


def write_state(job_dir: Path, sqlite_seeds=2, postgres_seeds=1) -> dict:
    job_dir.mkdir()
    state = {
        "job_dir": str(job_dir.resolve()),
        "session": "session-1",
        "vv_agent_revision": CODE,
        "contract_revision": CONTRACT,
        "script_sha256": "c" * 64,
        "sqlite_seeds": sqlite_seeds,
        "postgres_seeds": postgres_seeds,
        "sqlite_chunk_size": campaign.SQLITE_CHUNK,
        "postgres_chunk_size": campaign.POSTGRES_CHUNK,
        "remote_job_dir": "/content/vv-agent-campaigns/job-1",
        "remote_reports_dir": "/content/vv-agent-campaigns/job-1/reports",
        "command": ["python3", "/content/vv-agent-campaigns/job-1/colab_simulation.py"],
        "launch_identity": CHILD,
    }
    campaign._write_json(job_dir / "state.json", state)
    return state


def write_evidence(evidence: Path, files: dict[str, bytes]) -> None:
    for name, body in files.items():
        path = evidence / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)


def terminal_cli(job_dir: Path, files: dict[str, bytes], observed: dict):
    calls = []

    def run(arguments):
        calls.append(list(arguments))
        if arguments[0] == "status":
            return completed(arguments, stdout="[session-1] Status: IDLE\n")
        if arguments[0] == "exec":
            return completed(arguments, stdout="OBSERVE_JSON=" + json.dumps(observed) + "\n")
        if arguments[0] == "download":
            remote = arguments[arguments.index("--session") + 2]
            relative = remote.removeprefix("/content/vv-agent-campaigns/job-1/")
            Path(arguments[-1]).write_bytes(files[relative])
            return completed(arguments)
        if arguments[0] == "stop":
            return completed(arguments)
        raise AssertionError(arguments)

    return run, calls


def test_public_run_waits_from_running_to_terminal_and_stops_after_collect(tmp_path: Path):
    job_dir = tmp_path / "job-1"
    files = payloads()
    running_files = {name: body for name, body in files.items() if name.startswith("reports/sqlite-")}
    running = observation(running_files, state="running")
    terminal = observation(files)
    observations = 0
    calls = []

    def run(arguments):
        nonlocal observations
        calls.append(list(arguments))
        if arguments[0] in {"new", "upload", "stop"}:
            return completed(arguments)
        if arguments[0] == "exec" and Path(arguments[arguments.index("--file") + 1]).name == "launch_cell.py":
            marker = {"status": {"state": "running", "child_identity": CHILD}}
            return completed(arguments, stdout="LAUNCH_JSON=" + json.dumps(marker) + "\n")
        if arguments[0] == "status":
            return completed(arguments, stdout="[session-1] Status: IDLE\n")
        if arguments[0] == "exec":
            observations += 1
            current = running if observations == 1 else terminal
            return completed(arguments, stdout="OBSERVE_JSON=" + json.dumps(current) + "\n")
        if arguments[0] == "download":
            remote = arguments[arguments.index("--session") + 2]
            relative = remote.removeprefix("/content/vv-agent-campaigns/job-1/")
            Path(arguments[-1]).write_bytes(files[relative])
            return completed(arguments)
        raise AssertionError(arguments)

    code = campaign.main(
        [
            "run",
            "--job-dir",
            str(job_dir),
            "--session",
            "session-1",
            "--vv-agent-revision",
            CODE,
            "--contract-revision",
            CONTRACT,
            "--sqlite-seeds",
            "2",
            "--postgres-seeds",
            "1",
            "--interval",
            "0.001",
            "--max-observations",
            "4",
        ],
        cli=run,
    )
    assert code == 0
    assert observations == 2
    names = [call[0] for call in calls]
    assert names[-1] == "stop"
    assert (job_dir / "evidence/reports/result.json").is_file()
    assert campaign._json(job_dir / "released.json")["acceptance"]["accepted"] is True


def test_wait_unknown_never_reuses_previous_live_observation_or_stops(tmp_path: Path):
    job_dir = tmp_path / "job-1"
    write_state(job_dir)
    running_files = {name: body for name, body in payloads().items() if name.startswith("reports/sqlite-")}
    running = observation(running_files, state="running")
    status_calls = 0
    calls = []

    def run(arguments):
        nonlocal status_calls
        calls.append(list(arguments))
        if arguments[0] == "status":
            status_calls += 1
            if status_calls == 1:
                return completed(arguments, stdout="[session-1] Status: IDLE\n")
            return completed(arguments, code=1, stderr="Connection was lost")
        if arguments[0] == "exec":
            return completed(arguments, stdout="OBSERVE_JSON=" + json.dumps(running) + "\n")
        raise AssertionError(arguments)

    observed = campaign.observe(job_dir, cli=run)
    assert observed["state"] == "running"
    assert "observation" not in campaign._json(job_dir / "state.json")
    result = campaign.wait(job_dir, interval=0, max_observations=2, cli=run)
    assert result["state"] == "unknown"
    assert status_calls == 2
    assert "stop" not in [call[0] for call in calls]


def test_duplicate_start_is_rejected_without_another_new(tmp_path: Path):
    job_dir = tmp_path / "job-1"
    calls = []

    def run(arguments):
        calls.append(list(arguments))
        if arguments[0] in {"new", "upload"}:
            return completed(arguments)
        marker = {"status": {"state": "running", "child_identity": CHILD}}
        return completed(arguments, stdout="LAUNCH_JSON=" + json.dumps(marker) + "\n")

    campaign.start(job_dir, "session-1", CODE, CONTRACT, 2, 1, cli=run)
    with pytest.raises(FileExistsError):
        campaign.start(job_dir, "session-2", CODE, CONTRACT, 2, 1, cli=run)
    assert [call[0] for call in calls].count("new") == 1


def test_terminal_hash_mismatch_preserves_vm(tmp_path: Path):
    job_dir = tmp_path / "job-1"
    write_state(job_dir)
    files = payloads()
    observed = observation(files, wrong_hash=True)
    run, calls = terminal_cli(job_dir, files, observed)
    with pytest.raises(ValueError, match="hash"):
        campaign.collect(job_dir, cli=run)
    assert "stop" not in [call[0] for call in calls]
    assert not (job_dir / "released.json").exists()


def validate_files(tmp_path: Path, files: dict[str, bytes], sqlite_seeds=2, postgres_seeds=1):
    evidence = tmp_path / "evidence"
    write_evidence(evidence, files)
    state = {
        "vv_agent_revision": CODE,
        "contract_revision": CONTRACT,
        "sqlite_seeds": sqlite_seeds,
        "postgres_seeds": postgres_seeds,
        "sqlite_chunk_size": campaign.SQLITE_CHUNK,
        "postgres_chunk_size": campaign.POSTGRES_CHUNK,
    }
    return campaign.validate_evidence(evidence, state, {"exit_code": 0})


def test_partial_result_cannot_claim_full_or_omit_parts(tmp_path: Path):
    files = payloads()
    result = json.loads(files["reports/result.json"])
    result["full_acceptance"] = True
    files["reports/result.json"] = json.dumps(result).encode()
    with pytest.raises(ValueError, match="partial"):
        validate_files(tmp_path / "claim", files)

    files = payloads()
    del files["reports/sqlite-0000000000-0000000002.json"]
    with pytest.raises(ValueError):
        validate_files(tmp_path / "missing", files)


def test_sample_requires_postgres_cleanup_before_acceptance(tmp_path: Path):
    files = payloads()
    result = json.loads(files["reports/result.json"])
    result["runtime"] = {}
    files["reports/result.json"] = json.dumps(result).encode()
    with pytest.raises(ValueError, match="cleanup"):
        validate_files(tmp_path, files)


def test_result_scenarios_must_equal_actual_part_sum(tmp_path: Path):
    files = payloads()
    result = json.loads(files["reports/result.json"])
    result["campaigns"]["sqlite"]["scenarios"]["precommit"] = 1
    files["reports/result.json"] = json.dumps(result).encode()
    with pytest.raises(ValueError, match="scenarios differ"):
        validate_files(tmp_path, files)


@pytest.mark.parametrize("field,value", [("dirty", True), ("contract_dirty", None), ("code_revision", "c" * 40)])
def test_part_identity_mismatch_is_rejected(tmp_path: Path, field, value):
    files = payloads()
    name = "reports/sqlite-0000000000-0000000002.json"
    body = json.loads(files[name])
    body[field] = value
    files[name] = json.dumps(body).encode()
    with pytest.raises(ValueError):
        validate_files(tmp_path, files)


def full_payloads(sqlite_filesystem="ext4") -> dict[str, bytes]:
    files = {}
    result = {
        "status": "complete",
        "full_acceptance": True,
        "identity": {"vv_agent_revision": CODE, "contract_revision": CONTRACT},
        "campaigns": {},
        "runtime": {"postgres": {"temporary_role_removed": True, "temporary_databases_removed": True}},
    }
    coverage = sorted(campaign.REQUIRED_COVERED)
    scenario_names = [f"scenario-{index:02d}" for index in range(21)]
    for backend, total, chunk in (("sqlite", 1_000_000, 100_000), ("postgres", 10_000, 1_000)):
        names, aggregate = [], {}
        start = 0
        index = 0
        while start < total:
            count = min(chunk, total - start)
            covered = [coverage[index % len(coverage)]]
            body = part(backend, start, count, covered)
            primary = scenario_names[(index * 2) % len(scenario_names)]
            secondary = scenario_names[(index * 2 + 1) % len(scenario_names)]
            body["scenarios"] = {primary: count - 1, secondary: 1}
            if index == 9:
                body["scenarios"] = {primary: count - 2, secondary: 1, scenario_names[20]: 1}
            if backend == "sqlite":
                body["environment"]["sqlite_temporary_storage"]["observations"][0]["filesystem"] = sqlite_filesystem
            name = f"reports/{backend}-{start:010d}-{count:010d}.json"
            files[name] = json.dumps(body).encode()
            names.append(Path(name).name)
            for scenario, scenario_count in body["scenarios"].items():
                aggregate[scenario] = aggregate.get(scenario, 0) + scenario_count
            start += count
            index += 1
        result["campaigns"][backend] = {
            "seed": 0,
            "seeds": total,
            "scenarios": aggregate,
            "covered": coverage,
            "pending": {"I9": "future", "I11": "future"},
            "reports": names,
        }
    files["reports/result.json"] = json.dumps(result).encode()
    return files


def test_full_acceptance_uses_backend_coverage_and_ordinary_sqlite_storage(tmp_path: Path):
    files = full_payloads()
    result = validate_files(tmp_path / "ok", files, 1_000_000, 10_000)
    assert result["full_acceptance"] is True

    files = full_payloads(sqlite_filesystem="tmpfs")
    with pytest.raises(ValueError, match="ordinary filesystem"):
        validate_files(tmp_path / "tmpfs", files, 1_000_000, 10_000)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("code_digest", "f" * 64),
        ("contract_manifest_sha256", "f" * 64),
        ("in_memory_mutations", [{"target": "step"}]),
    ],
)
def test_full_rejects_part_identity_drift(tmp_path: Path, field, value):
    files = full_payloads()
    name = "reports/postgres-0000001000-0000001000.json"
    body = json.loads(files[name])
    body[field] = value
    files[name] = json.dumps(body).encode()
    with pytest.raises(ValueError):
        validate_files(tmp_path, files, 1_000_000, 10_000)


def test_full_campaign_coverage_must_equal_part_union(tmp_path: Path):
    files = full_payloads()
    result = json.loads(files["reports/result.json"])
    result["campaigns"]["sqlite"]["covered"].append("I99")
    files["reports/result.json"] = json.dumps(result).encode()
    with pytest.raises(ValueError, match="coverage differs"):
        validate_files(tmp_path, files, 1_000_000, 10_000)


def test_full_requires_21_aggregate_scenarios(tmp_path: Path):
    files = full_payloads()
    aggregate = {}
    for name in list(files):
        if not name.startswith("reports/"):
            continue
        body = json.loads(files[name])
        if name.endswith("result.json"):
            continue
        body["scenarios"].pop("scenario-20", None)
        files[name] = json.dumps(body).encode()
        backend = body["backend"]
        aggregate.setdefault(backend, {})
        for scenario, count in body["scenarios"].items():
            aggregate[backend][scenario] = aggregate[backend].get(scenario, 0) + count
    result = json.loads(files["reports/result.json"])
    result["campaigns"]["sqlite"]["scenarios"] = aggregate["sqlite"]
    result["campaigns"]["postgres"]["scenarios"] = aggregate["postgres"]
    files["reports/result.json"] = json.dumps(result).encode()
    with pytest.raises(ValueError, match="scenario count"):
        validate_files(tmp_path, files, 1_000_000, 10_000)


def test_five_seed_chunk_two_ranges_validate(tmp_path: Path):
    files = {}
    result = {
        "status": "complete",
        "full_acceptance": False,
        "identity": {"vv_agent_revision": CODE, "contract_revision": CONTRACT},
        "campaigns": {},
        "runtime": {
            "postgres": {
                "temporary_role_removed": True,
                "temporary_databases_removed": True,
            }
        },
    }
    for backend in ("sqlite", "postgres"):
        names, start = [], 0
        while start < 5:
            count = min(2, 5 - start)
            name = f"reports/{backend}-{start:010d}-{count:010d}.json"
            files[name] = json.dumps(part(backend, start, count)).encode()
            names.append(Path(name).name)
            start += count
        result["campaigns"][backend] = {
            "seed": 0,
            "seeds": 5,
            "scenarios": {"precommit": 5},
            "covered": [],
            "pending": {"I9": "future", "I11": "future"},
            "reports": names,
        }
    files["reports/result.json"] = json.dumps(result).encode()
    evidence = tmp_path / "evidence"
    write_evidence(evidence, files)
    state = {
        "vv_agent_revision": CODE,
        "contract_revision": CONTRACT,
        "sqlite_seeds": 5,
        "postgres_seeds": 5,
        "sqlite_chunk_size": 2,
        "postgres_chunk_size": 2,
    }
    accepted = campaign.validate_evidence(evidence, state, {"exit_code": 0})
    assert accepted["accepted"] is True
    assert accepted["backends"]["sqlite"]["reports"] == 3


def test_start_persists_explicit_chunk_sizes(tmp_path: Path):
    job_dir = tmp_path / "job-1"
    calls = []

    def run(arguments):
        calls.append(list(arguments))
        if arguments[0] in {"new", "upload", "status"}:
            return completed(arguments)
        marker = {"status": {"state": "running", "child_identity": CHILD}}
        return completed(arguments, stdout="LAUNCH_JSON=" + json.dumps(marker) + "\n")

    state = campaign.start(job_dir, "session-1", CODE, CONTRACT, 5, 5, 2, 2, cli=run)
    assert state["sqlite_chunk_size"] == 2 and state["postgres_chunk_size"] == 2
    assert campaign._json(job_dir / "state.json")["command"][12:16] == [
        "--sqlite-chunk-size",
        "2",
        "--postgres-chunk-size",
        "2",
    ]
    assert [call[0] for call in calls].count("new") == 1
    assert [call[0] for call in calls][-3:] == ["upload", "status", "exec"]


def test_start_refresh_failure_does_not_exec_or_stop(tmp_path: Path):
    job_dir = tmp_path / "job-1"
    calls = []

    def run(arguments):
        calls.append(list(arguments))
        if arguments[0] == "status":
            return completed(arguments, code=1, stderr="refresh failed")
        return completed(arguments)

    with pytest.raises(RuntimeError, match="refresh session"):
        campaign.start(job_dir, "session-1", CODE, CONTRACT, 2, 1, cli=run)
    assert [call[0] for call in calls] == ["new", "upload", "upload", "status"]
    assert not {"exec", "stop"} & {call[0] for call in calls}
    assert campaign._json(job_dir / "state.json")["launch_identity"] is None


def test_wait_max_observations_running_is_incomplete(tmp_path: Path):
    job_dir = tmp_path / "job-1"
    write_state(job_dir)
    running_files = {name: body for name, body in payloads().items() if name.startswith("reports/sqlite-")}
    running = observation(running_files, state="running")

    def run(arguments):
        if arguments[0] == "status":
            return completed(arguments, stdout="[session-1] Status: IDLE\n")
        if arguments[0] == "exec":
            return completed(arguments, stdout="OBSERVE_JSON=" + json.dumps(running) + "\n")
        if arguments[0] == "download":
            Path(arguments[-1]).write_bytes(running_files["reports/sqlite-0000000000-0000000002.json"])
            return completed(arguments)
        raise AssertionError(arguments)

    code = campaign.main(
        ["wait", "--job-dir", str(job_dir), "--max-observations", "1", "--interval", "1"],
        cli=run,
    )
    assert code == 2


def test_released_failure_propagates_accepted_status(tmp_path: Path):
    job_dir = tmp_path / "job-1"
    write_state(job_dir)
    campaign._write_json(
        job_dir / "released.json",
        {
            "terminal": True,
            "state": "terminal_failed",
            "acceptance": {"accepted": False, "reason": "evidence rejected"},
        },
    )
    code = campaign.main(["wait", "--job-dir", str(job_dir)], cli=lambda arguments: None)
    assert code == 1
    assert campaign.main(["collect", "--job-dir", str(job_dir)], cli=lambda arguments: None) == 1
    observed = campaign.observe(job_dir, cli=lambda arguments: None)
    assert observed["accepted"] is False


def test_partial_collect_failure_returns_unknown_reason(tmp_path: Path):
    job_dir = tmp_path / "job-1"
    write_state(job_dir)
    running_files = {name: body for name, body in payloads().items() if name.startswith("reports/sqlite-")}
    running = observation(running_files, state="running")
    calls = []

    def run(arguments):
        calls.append(list(arguments))
        if arguments[0] == "status":
            return completed(arguments, stdout="[session-1] Status: IDLE\n")
        if arguments[0] == "exec":
            return completed(arguments, stdout="OBSERVE_JSON=" + json.dumps(running) + "\n")
        if arguments[0] == "download":
            return completed(arguments, code=1, stderr="transfer interrupted")
        raise AssertionError(arguments)

    result = campaign.wait(job_dir, interval=0, max_observations=1, cli=run)
    assert result["state"] == "unknown" and result["terminal"] is False
    assert result["reason"].startswith("partial collection failed: download reports/sqlite-")
    assert result["reason"].endswith("failed with exit 1: transfer interrupted")
    assert "stop" not in [call[0] for call in calls]


def test_generated_launch_cell_detaches_local_child_without_argv_or_file(tmp_path: Path):
    job_dir = tmp_path / "remote-job"
    staged = tmp_path / "staged-simulation.py"
    marker = job_dir / "child-finished"
    staged.write_text(
        f"from pathlib import Path\nimport time\ntime.sleep(1)\nPath({str(marker)!r}).write_text('done')\nraise SystemExit(7)\n"
    )
    supervisor = tmp_path / "supervisor.py"
    config = {
        "job_dir": str(job_dir),
        "staged_script": str(staged),
        "script_sha256": sha(staged.read_bytes()),
        "command": [sys.executable, str(job_dir / "colab_simulation.py")],
    }
    supervisor.write_text(campaign._render(campaign.SUPERVISOR, config))
    launch_cell = tmp_path / "launch_cell.py"
    launch_cell.write_text(
        campaign._render(
            campaign.LAUNCH_CELL,
            {"job_dir": str(job_dir), "supervisor_path": str(supervisor)},
        )
    )
    launched = subprocess.run(
        [sys.executable, str(launch_cell), "ignored-argv"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert launched.returncode == 0
    value = campaign._marker(launched.stdout, "LAUNCH_JSON=")
    assert value is not None and value["status"]["state"] == "running"
    assert value["status"]["child_identity"]["live"] is True
    deadline = time.monotonic() + 8
    status = None
    while time.monotonic() < deadline:
        status = json.loads((job_dir / "process-status.json").read_text())
        if status.get("state") == "exited":
            break
        time.sleep(0.05)
    assert status is not None and status["exit_code"] == 7
    assert marker.read_text() == "done"
