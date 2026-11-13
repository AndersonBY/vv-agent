from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "colab_simulation.py"
SPEC = importlib.util.spec_from_file_location("colab_simulation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
colab_simulation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(colab_simulation)

CODE_REVISION = "a" * 40
CONTRACT_REVISION = "b" * 40


@pytest.fixture(scope="module")
def expectations():
    env = os.environ | {"PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}
    return colab_simulation.source_expectations(sys.executable, source=ROOT, env=env)


def real_report(backend, tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    (tmp_path / "tmp").mkdir()
    env["TMPDIR"] = str(tmp_path / "tmp")
    if backend == "postgres" and not env.get("VV_AGENT_TEST_POSTGRES_DSN"):
        pytest.skip("VV_AGENT_TEST_POSTGRES_DSN is required for the real PostgreSQL report check")
    report_path = tmp_path / f"{backend}.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/session_simulation.py",
            "--backend",
            backend,
            "--seed",
            "0",
            "--seeds",
            "1",
            "--workers",
            "1",
            "--scenario",
            "precommit",
            "--report",
            str(report_path),
        ],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(report_path.read_text()), tmp_path / "tmp"


@pytest.fixture(scope="module")
def sqlite_report(tmp_path_factory):
    return real_report("sqlite", tmp_path_factory.mktemp("real-sqlite"))


@pytest.fixture(scope="module")
def postgres_report(tmp_path_factory):
    return real_report("postgres", tmp_path_factory.mktemp("real-postgres"))


def normalize_validator_identity(report):
    value = json.loads(json.dumps(report))
    # The full structure comes from the real producer. These identity flags are
    # explicit validator inputs here and do not prove a checkout was clean.
    value["dirty"] = False
    value["contract_dirty"] = False
    return value


def test_parse_args_sets_full_defaults_and_requires_full_revisions():
    args = colab_simulation.parse_args(["--vv-agent-revision", CODE_REVISION, "--contract-revision", CONTRACT_REVISION])
    assert args.sqlite_seeds == 1_000_000
    assert args.postgres_seeds == 10_000
    assert args.workers == 8
    with pytest.raises(SystemExit):
        colab_simulation.parse_args(["--vv-agent-revision", "abc", "--contract-revision", CONTRACT_REVISION])


def test_campaign_command_uses_the_real_cli():
    assert colab_simulation.campaign_command(
        "/tmp/run/venv/bin/python",
        backend="postgres",
        seed=7,
        seeds=10_000,
        workers=8,
        report=Path("/tmp/run/report.json"),
    ) == [
        "/tmp/run/venv/bin/python",
        "scripts/session_simulation.py",
        "--backend",
        "postgres",
        "--seed",
        "7",
        "--seeds",
        "10000",
        "--workers",
        "8",
        "--report",
        "/tmp/run/report.json",
    ]


@pytest.mark.parametrize("fixture_name", ["sqlite_report", "postgres_report"])
def test_validator_accepts_real_producer_report(fixture_name, request, expectations):
    report, tempdir = request.getfixturevalue(fixture_name)
    scenarios, pending = expectations
    colab_simulation.validate_report(
        normalize_validator_identity(report),
        backend=report["backend"],
        seed=0,
        seeds=1,
        identity={
            "vv_agent_revision": report["code_revision"],
            "contract_revision": report["contract_revision"],
        },
        tempdir=tempdir,
        scenarios=scenarios,
        pending=pending,
    )
    storage = report["environment"]["sqlite_temporary_storage"]
    assert "requested_TMPDIR" not in storage
    if report["backend"] == "sqlite":
        assert storage["verified"] is True
        assert storage["observations"][0]["requested_TMPDIR"] == str(tempdir)
    else:
        assert storage == {"verified": False, "observations": []}


@pytest.mark.parametrize("field,value", [("pending", {"I9": "future"}), ("seeds", 2)])
def test_validator_rejects_bad_counts_from_real_report(sqlite_report, expectations, field, value):
    report = normalize_validator_identity(sqlite_report[0])
    report[field] = value
    scenarios, pending = expectations
    with pytest.raises(ValueError):
        colab_simulation.validate_report(
            report,
            backend="sqlite",
            seed=0,
            seeds=1,
            identity={
                "vv_agent_revision": report["code_revision"],
                "contract_revision": report["contract_revision"],
            },
            tempdir=sqlite_report[1],
            scenarios=scenarios,
            pending=pending,
        )


def test_sample_report_without_full_scenarios_or_coverage_is_not_full(sqlite_report, expectations):
    report, tempdir = sqlite_report
    scenarios, pending = expectations
    assert report["seeds"] == 1
    assert set(report["scenarios"]) != set(scenarios)
    assert not colab_simulation.REQUIRED_COVERED.issubset(report["covered"])
    colab_simulation.validate_report(
        normalize_validator_identity(report),
        backend="sqlite",
        seed=0,
        seeds=1,
        identity={"vv_agent_revision": report["code_revision"], "contract_revision": report["contract_revision"]},
        tempdir=tempdir,
        scenarios=scenarios,
        pending=pending,
    )


@pytest.mark.parametrize("field", ["dirty", "contract_dirty"])
def test_validator_rejects_unknown_dirty_identity(sqlite_report, expectations, field):
    report = normalize_validator_identity(sqlite_report[0])
    report[field] = None
    tempdir = sqlite_report[1]
    scenarios, pending = expectations
    with pytest.raises(ValueError, match="checkout"):
        colab_simulation.validate_report(
            report,
            backend="sqlite",
            seed=0,
            seeds=1,
            identity={"vv_agent_revision": report["code_revision"], "contract_revision": report["contract_revision"]},
            tempdir=tempdir,
            scenarios=scenarios,
            pending=pending,
        )


def test_source_expectations_rejects_unexpected_pending(monkeypatch):
    monkeypatch.setattr(
        colab_simulation,
        "run",
        lambda *args, **kwargs: json.dumps({"scenarios": ["tools"], "pending": ["I9", "I12"]}),
    )
    with pytest.raises(RuntimeError, match="unexpected pending"):
        colab_simulation.source_expectations("python", source=Path("/tmp/source"), env={})


def test_stream_campaign_redacts_only_exact_secrets(monkeypatch, capsys):
    password = "actual-random-password"
    dsn = f"postgresql://role:{password}@127.0.0.1:5432/db"
    digest = "c" * 64
    started = []

    class Process:
        stdout = io.StringIO(f"error={dsn} password={password} seed=17 digest={digest}\n")

        def wait(self):
            return 9

        def terminate(self):
            raise AssertionError("terminate should not run")

    def popen(command, **kwargs):
        started.append((command, kwargs))
        return Process()

    monkeypatch.setattr(colab_simulation.subprocess, "Popen", popen)
    returncode = colab_simulation.stream_campaign(
        ["python", "scripts/session_simulation.py"], cwd=Path("/tmp"), env={"A": "B"}, sensitive=(dsn, password)
    )
    output = capsys.readouterr().out
    assert returncode == 9
    assert len(started) == 1
    assert dsn not in output and password not in output
    assert "seed=17" in output and digest in output
    assert "[REDACTED]" in output
