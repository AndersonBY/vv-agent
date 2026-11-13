from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from support.contract import fixture_root

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "colab_simulation.py"
SPEC = importlib.util.spec_from_file_location("colab_simulation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
colab_simulation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(colab_simulation)

CODE_REVISION = "a" * 40
CONTRACT_REVISION = "b" * 40


@pytest.fixture(scope="module")
def canonical_contract(tmp_path_factory):
    lock = json.loads((ROOT / "contract.lock.json").read_text())
    selected = os.environ.get("VV_AGENT_CONTRACT_SOURCE")
    fixtures_source = fixture_root()
    version = lock["contract_version"]
    if selected:
        version = json.loads((Path(selected) / "contract.json").read_text())["version"]
    manifest_sha256 = hashlib.sha256((fixtures_source / "SHA256SUMS").read_bytes()).hexdigest()
    source = tmp_path_factory.mktemp("canonical")
    shutil.copytree(fixtures_source, source / "fixtures")
    metadata = {
        "schema_version": 1,
        "name": "vv-agent-contract",
        "version": version,
        "fixtures": {"path": "fixtures", "manifest_sha256": manifest_sha256},
    }
    (source / "contract.json").write_text(json.dumps(metadata))
    return source


def campaign_environment(tempdir, *, dsn=None, contract_source=None):
    tempdir.mkdir(parents=True, exist_ok=True)
    (tempdir.parent / "home").mkdir(exist_ok=True)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tempdir.parent / "home"),
        "TMPDIR": str(tempdir),
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
        "PYTHONPYCACHEPREFIX": str(tempdir.parent / "pycache"),
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    if dsn is not None:
        env["VV_AGENT_TEST_POSTGRES_DSN"] = dsn
    if contract_source is not None:
        env["VV_AGENT_CONTRACT_SOURCE"] = str(contract_source)
    return env


@pytest.fixture(scope="module")
def expectations(tmp_path_factory):
    env = campaign_environment(tmp_path_factory.mktemp("expectations") / "tmp")
    return colab_simulation.source_expectations(sys.executable, source=ROOT, env=env)


def real_report(backend, tmp_path):
    dsn = os.environ.get("VV_AGENT_TEST_POSTGRES_DSN")
    if backend == "postgres" and not dsn:
        pytest.skip("VV_AGENT_TEST_POSTGRES_DSN is required for the real PostgreSQL report check")
    env = campaign_environment(tmp_path / "tmp", dsn=dsn)
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


def normalize_report(report, *, identity=None, contract_source=None):
    value = copy.deepcopy(report)
    value["dirty"] = False
    value["contract_dirty"] = False
    value["in_memory_mutations"] = []
    if identity is not None:
        value["code_revision"] = identity["vv_agent_revision"]
        value["contract_revision"] = identity["contract_revision"]
    if contract_source is not None:
        value["contract_selection"] = "local"
        value["contract_source"] = str(contract_source)
    return value


@pytest.fixture
def private_checkout_identity(monkeypatch):
    original = colab_simulation.validate_report

    def normalized(report, **kwargs):
        return original(normalize_report(report, identity=kwargs["identity"]), **kwargs)

    monkeypatch.setattr(colab_simulation, "validate_report", normalized)


def execute_args(report_dir):
    return colab_simulation.parse_args(
        [
            "--vv-agent-revision",
            CODE_REVISION,
            "--contract-revision",
            CONTRACT_REVISION,
            "--report-dir",
            str(report_dir),
            "--sqlite-seeds",
            "5",
            "--postgres-seeds",
            "3",
            "--sqlite-chunk-size",
            "2",
            "--postgres-chunk-size",
            "2",
            "--workers",
            "2",
        ]
    )


def fake_part_report(source, command, *, contract_source, tempdir):
    backend = command[command.index("--backend") + 1]
    seed = int(command[command.index("--seed") + 1])
    seeds = int(command[command.index("--seeds") + 1])
    report = normalize_report(source, identity={"vv_agent_revision": CODE_REVISION, "contract_revision": CONTRACT_REVISION})
    report.update(
        backend=backend,
        seed=seed,
        seeds=seeds,
        scenarios={"precommit": seeds},
        workers=2,
        contract_selection="local",
        contract_source=str(contract_source),
    )
    report["environment"]["sqlite_temporary_storage"] = (
        {
            "verified": True,
            "observations": [{"directory": "/tmp/mock-sqlite", "filesystem": "ext2/ext3", "requested_TMPDIR": str(tempdir)}],
        }
        if backend == "sqlite"
        else {"verified": False, "observations": []}
    )
    return report


def run_execute(monkeypatch, tmp_path, canonical_contract, sqlite_report, stream_factory=None):
    def clone(repository, revision, destination, *, env):
        target = ROOT if repository == colab_simulation.VV_AGENT_REPOSITORY else canonical_contract
        destination.symlink_to(target, target_is_directory=True)
        return revision

    postgres = {"dsn": "unused", "runtime": {"password_authentication_verified": True}}

    def cleanup(state, *, env):
        state["runtime"]["temporary_role_removed"] = True
        state["runtime"]["temporary_databases_removed"] = True

    def write_report(command, *, cwd, env, sensitive=(), diagnostics=None):
        assert diagnostics is not None
        report_path = Path(command[command.index("--report") + 1])
        report = fake_part_report(
            sqlite_report[0], command, contract_source=canonical_contract, tempdir=report_path.parent / "tmp"
        )
        colab_simulation.atomic_write_json(report_path, report)
        diagnostics["tail"] = "fake part complete\n"
        return 0

    monkeypatch.setattr(colab_simulation, "clone_exact", clone)
    monkeypatch.setattr(colab_simulation, "prepare_python", lambda source, *, env: (sys.executable, {"python": "3.12"}))
    monkeypatch.setattr(colab_simulation, "prepare_postgres", lambda root, *, env, sensitive: postgres)
    monkeypatch.setattr(colab_simulation, "cleanup_postgres", cleanup)
    monkeypatch.setattr(colab_simulation, "stream_campaign", stream_factory(write_report) if stream_factory else write_report)
    report_dir = tmp_path / "evidence"
    return colab_simulation.execute(execute_args(report_dir)), report_dir


def run_real_chunks(backend, total, chunk_size, tmp_path, expectations, canonical_contract):
    dsn = os.environ.get("VV_AGENT_TEST_POSTGRES_DSN") if backend == "postgres" else None
    if backend == "postgres" and not dsn:
        pytest.skip("VV_AGENT_TEST_POSTGRES_DSN is required for real PostgreSQL")
    env = campaign_environment(tmp_path / "tmp", dsn=dsn, contract_source=canonical_contract)
    identity = {"vv_agent_revision": CODE_REVISION, "contract_revision": CONTRACT_REVISION}
    report_dir = tmp_path / "parts"
    report_dir.mkdir()
    names = []
    for seed in range(0, total, chunk_size):
        seeds = min(chunk_size, total - seed)
        name = f"{backend}-{seed:010d}-{seeds:010d}.json"
        path = tmp_path / name
        subprocess.run(
            [
                sys.executable,
                "scripts/session_simulation.py",
                "--backend",
                backend,
                "--seed",
                str(seed),
                "--seeds",
                str(seeds),
                "--workers",
                "2",
                "--report",
                str(path),
            ],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
        )
        report = normalize_report(json.loads(path.read_text()), identity=identity, contract_source=canonical_contract)
        colab_simulation.atomic_write_json(report_dir / name, report)
        names.append(name)
    campaign = colab_simulation.aggregate_campaign(
        report_dir,
        backend=backend,
        seed=0,
        total=total,
        chunk_size=chunk_size,
        scenarios=expectations,
        identity=identity,
        tempdir=tmp_path / "tmp",
        workers=2,
        contract_source=canonical_contract,
    )
    return campaign, names


def test_parse_args_defaults_and_required_report_dir(tmp_path):
    args = colab_simulation.parse_args(
        ["--vv-agent-revision", CODE_REVISION, "--contract-revision", CONTRACT_REVISION, "--report-dir", str(tmp_path)]
    )
    assert (args.sqlite_seeds, args.postgres_seeds) == (1_000_000, 10_000)
    assert (args.sqlite_chunk_size, args.postgres_chunk_size) == (100_000, 1_000)
    assert (args.seed, args.workers) == (0, 8)
    with pytest.raises(SystemExit):
        colab_simulation.parse_args(["--vv-agent-revision", CODE_REVISION, "--contract-revision", CONTRACT_REVISION])


def test_report_directory_must_be_empty(tmp_path):
    report_dir = tmp_path / "evidence"
    report_dir.mkdir()
    retained = report_dir / "existing.txt"
    retained.write_text("keep")
    with pytest.raises(ValueError, match="must be empty"):
        colab_simulation.prepare_report_dir(report_dir)
    assert retained.read_text() == "keep"


def test_atomic_report_does_not_overwrite_and_redacts_json(tmp_path):
    password = "actual-random-password"
    dsn = f"postgresql://role:{password}@127.0.0.1:5432/db"
    path = tmp_path / "part.json"
    payload = {"error": f"{dsn} {password}", "code_digest": "d" * 64}
    colab_simulation.atomic_write_json(path, colab_simulation.scrub_json(payload, (dsn, password)))
    text = path.read_text()
    assert password not in text and dsn not in text
    assert json.loads(text)["code_digest"] == "d" * 64
    with pytest.raises(FileExistsError):
        colab_simulation.atomic_write_json(path, {"replacement": True})
    assert path.read_text() == text


@pytest.mark.parametrize("fixture_name", ["sqlite_report", "postgres_report"])
def test_validate_real_report_and_reject_bad_counts(fixture_name, request, expectations, canonical_contract):
    report, tempdir = request.getfixturevalue(fixture_name)
    identity = {"vv_agent_revision": CODE_REVISION, "contract_revision": CONTRACT_REVISION}
    report = normalize_report(report, identity=identity, contract_source=canonical_contract)
    colab_simulation.validate_report(
        report,
        backend=report["backend"],
        seed=0,
        seeds=1,
        identity=identity,
        tempdir=tempdir,
        scenarios=expectations,
        workers=1,
        contract_source=canonical_contract,
    )
    report["seeds"] = 2
    with pytest.raises(ValueError, match="count"):
        colab_simulation.validate_report(
            report,
            backend=report["backend"],
            seed=0,
            seeds=1,
            identity=identity,
            tempdir=tempdir,
            scenarios=expectations,
            workers=1,
            contract_source=canonical_contract,
        )


def test_validate_requires_clean_and_unmutated_source(sqlite_report, expectations):
    report, tempdir = sqlite_report
    identity = {"vv_agent_revision": CODE_REVISION, "contract_revision": CONTRACT_REVISION}
    for field, value in (("dirty", True), ("in_memory_mutations", [{"target": "mutated"}])):
        bad = normalize_report(report, identity=identity)
        bad[field] = value
        with pytest.raises(ValueError):
            colab_simulation.validate_report(
                bad,
                backend="sqlite",
                seed=0,
                seeds=1,
                identity=identity,
                tempdir=tempdir,
                scenarios=expectations,
                workers=1,
                contract_source=Path(report["contract_source"]),
            )


def test_source_expectations_rejects_wrong_pending(monkeypatch):
    monkeypatch.setattr(
        colab_simulation,
        "run",
        lambda *args, **kwargs: json.dumps({"scenarios": ["tools"], "pending": ["I9", "I12"]}),
    )
    with pytest.raises(RuntimeError, match="unexpected pending"):
        colab_simulation.source_expectations("python", source=Path("/tmp/source"), env={})


def test_stream_redaction_and_bounded_tail(monkeypatch, capsys):
    password = "actual-random-password"
    dsn = f"postgresql://role:{password}@127.0.0.1:5432/db"
    digest = "c" * 64

    class Process:
        stdout = io.StringIO(f"{dsn} {password} seed=17 digest={digest}\n")

        def wait(self):
            return 9

        def terminate(self):
            raise AssertionError("terminate should not run")

    monkeypatch.setattr(colab_simulation.subprocess, "Popen", lambda *args, **kwargs: Process())
    diagnostics = {"tail": ""}
    code = colab_simulation.stream_campaign(
        ["python", "scripts/session_simulation.py"],
        cwd=Path("/tmp"),
        env={},
        sensitive=(dsn, password),
        diagnostics=diagnostics,
    )
    output = capsys.readouterr().out
    assert code == 9 and len(diagnostics["tail"]) <= 20_000
    assert dsn not in output and password not in diagnostics["tail"]
    assert "seed=17" in output and digest in output


def part(report, seed, seeds, *, identity, contract_source):
    value = normalize_report(report, identity=identity, contract_source=contract_source)
    value.update(seed=seed, seeds=seeds, scenarios={"precommit": seeds})
    return value


def aggregate_parts(tmp_path, entries, sqlite_report, expectations, canonical_contract):
    report, tempdir = sqlite_report
    identity = {"vv_agent_revision": CODE_REVISION, "contract_revision": CONTRACT_REVISION}
    report_dir = tmp_path / "parts"
    report_dir.mkdir(parents=True)
    for name, value in entries:
        (report_dir / name).write_text(json.dumps(value))
    return colab_simulation.aggregate_campaign(
        report_dir,
        backend="sqlite",
        seed=0,
        total=5,
        chunk_size=2,
        scenarios=expectations,
        identity=identity,
        tempdir=tempdir,
        workers=report["workers"],
        contract_source=canonical_contract,
    )


def test_cursor_rejects_missing_overlap_identity_and_failed_parts(sqlite_report, expectations, canonical_contract, tmp_path):
    source = sqlite_report[0]
    identity = {"vv_agent_revision": CODE_REVISION, "contract_revision": CONTRACT_REVISION}
    cases = [
        [
            ("sqlite-0000000000-0000000002.json", part(source, 0, 2, identity=identity, contract_source=canonical_contract)),
            ("sqlite-0000000004-0000000001.json", part(source, 4, 1, identity=identity, contract_source=canonical_contract)),
        ],
        [
            ("sqlite-0000000000-0000000002.json", part(source, 0, 2, identity=identity, contract_source=canonical_contract)),
            ("sqlite-0000000001-0000000002.json", part(source, 1, 2, identity=identity, contract_source=canonical_contract)),
        ],
    ]
    for index, entries in enumerate(cases):
        with pytest.raises(ValueError, match="missing, overlapping or non-contiguous"):
            aggregate_parts(tmp_path / str(index), entries, sqlite_report, expectations, canonical_contract)
    identity_entries = [
        ("sqlite-0000000000-0000000002.json", part(source, 0, 2, identity=identity, contract_source=canonical_contract)),
        ("sqlite-0000000002-0000000002.json", part(source, 2, 2, identity=identity, contract_source=canonical_contract)),
        ("sqlite-0000000004-0000000001.json", part(source, 4, 1, identity=identity, contract_source=canonical_contract)),
    ]
    identity_entries[1][1]["code_digest"] = "e" * 64
    with pytest.raises(ValueError, match="identity"):
        aggregate_parts(tmp_path / "identity", identity_entries, sqlite_report, expectations, canonical_contract)
    failed = part(source, 0, 2, identity=identity, contract_source=canonical_contract)
    failed["error"] = "invariant failed"
    with pytest.raises(ValueError, match="contains an error"):
        aggregate_parts(
            tmp_path / "failed",
            [("sqlite-0000000000-0000000002.json", failed)],
            sqlite_report,
            expectations,
            canonical_contract,
        )


def test_real_sqlite5_chunk2(tmp_path, expectations, canonical_contract):
    campaign, names = run_real_chunks("sqlite", 5, 2, tmp_path, expectations, canonical_contract)
    assert campaign["seeds"] == 5 and campaign["reports"] == names
    assert names == [
        "sqlite-0000000000-0000000002.json",
        "sqlite-0000000002-0000000002.json",
        "sqlite-0000000004-0000000001.json",
    ]


def test_real_postgres3_chunk2_optional(tmp_path, expectations, canonical_contract):
    campaign, names = run_real_chunks("postgres", 3, 2, tmp_path, expectations, canonical_contract)
    assert campaign["seeds"] == 3 and campaign["reports"] == names


def test_execute_sample_and_full_false_return_nonzero(
    monkeypatch, tmp_path, sqlite_report, canonical_contract, private_checkout_identity
):
    code, report_dir = run_execute(monkeypatch, tmp_path / "sample", canonical_contract, sqlite_report)
    result = json.loads((report_dir / "result.json").read_text())
    assert code == 0 and result["status"] == "complete" and result["full_acceptance"] is False
    assert result["runtime"]["temporary_directory_removed"] is True and "cleanup" not in result["runtime"]
    assert result["campaigns"]["sqlite"]["reports"] == [
        "sqlite-0000000000-0000000002.json",
        "sqlite-0000000002-0000000002.json",
        "sqlite-0000000004-0000000001.json",
    ]

    monkeypatch.setattr(colab_simulation, "SQLITE_FULL_SEEDS", 5)
    monkeypatch.setattr(colab_simulation, "POSTGRES_FULL_SEEDS", 3)
    code, report_dir = run_execute(monkeypatch, tmp_path / "full", canonical_contract, sqlite_report)
    result = json.loads((report_dir / "result.json").read_text())
    assert code != 0 and result["status"] == "failed" and result["full_acceptance"] is False


def test_second_part_missing_report_does_not_copy_first(
    monkeypatch, tmp_path, sqlite_report, canonical_contract, private_checkout_identity
):
    def missing_second(write_report):
        def stream(command, *, cwd, env, sensitive=(), diagnostics=None):
            seed = command[command.index("--seed") + 1]
            if command[command.index("--backend") + 1] == "sqlite" and seed == "2":
                assert diagnostics is not None
                diagnostics["tail"] = "second part emitted no report\n"
                return 0
            return write_report(command, cwd=cwd, env=env, sensitive=sensitive, diagnostics=diagnostics)

        return stream

    code, report_dir = run_execute(monkeypatch, tmp_path, canonical_contract, sqlite_report, stream_factory=missing_second)
    result = json.loads((report_dir / "result.json").read_text())
    failed = json.loads((report_dir / "sqlite-0000000002-0000000002.json").read_text())
    assert code != 0 and result["status"] == "failed"
    assert failed["seed"] == 2 and failed["seeds"] == 2 and "code_digest" not in failed
    assert "missing or invalid" in failed["error"] and "second part" in failed["trace"]
    assert set(result["reports"]) == {"sqlite-0000000000-0000000002.json", "sqlite-0000000002-0000000002.json"}


def test_stream_exception_retains_part_failure(
    monkeypatch, tmp_path, sqlite_report, canonical_contract, private_checkout_identity
):
    def failed(write_report):
        def stream(command, *, cwd, env, sensitive=(), diagnostics=None):
            assert diagnostics is not None
            diagnostics["tail"] = "stream failed before report\n"
            raise RuntimeError("stream transport lost")

        return stream

    code, report_dir = run_execute(monkeypatch, tmp_path, canonical_contract, sqlite_report, stream_factory=failed)
    result = json.loads((report_dir / "result.json").read_text())
    failed_report = json.loads((report_dir / "sqlite-0000000000-0000000002.json").read_text())
    assert code != 0 and result["status"] == "failed"
    assert "stream transport lost" in failed_report["error"]
    assert "stream failed before report" in failed_report["trace"]
