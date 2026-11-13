#!/usr/bin/env python3
"""Run the real P1 simulation CLI sequentially on a fresh Colab VM."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

VV_AGENT_REPOSITORY = "https://github.com/AndersonBY/vv-agent.git"
CONTRACT_REPOSITORY = "https://github.com/AndersonBY/vv-agent-contract.git"
SQLITE_FULL_SEEDS = 1_000_000
POSTGRES_FULL_SEEDS = 10_000
SQLITE_CHUNK_SIZE = 100_000
POSTGRES_CHUNK_SIZE = 1_000
MAX_WORKERS = 8
UV_VERSION = "0.10.4"
REQUIRED_COVERED = frozenset({"I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I10"})
EXPECTED_PENDING = frozenset({"I9", "I11"})
SCENARIO_COUNT = 21
FULL_REVISION = re.compile(r"[0-9a-f]{40}")
REPORT_NAME = re.compile(r"^(sqlite|postgres)-([0-9]{10})-([0-9]{10})\.json$")
RESULT_NAME = "result.json"
IDENTITY_FIELDS = (
    "vv_agent_revision",
    "contract_revision",
    "code_digest",
    "contract_manifest_sha256",
    "in_memory_mutations",
)


def parse_args(argv=None):
    def revision(value):
        value = value.lower()
        if FULL_REVISION.fullmatch(value) is None:
            raise argparse.ArgumentTypeError("revision must be a full 40-character Git hash")
        return value

    def positive(value):
        number = int(value)
        if number < 1:
            raise argparse.ArgumentTypeError("value must be positive")
        return number

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vv-agent-revision", required=True, type=revision)
    parser.add_argument("--contract-revision", required=True, type=revision)
    parser.add_argument("--sqlite-seeds", type=positive, default=SQLITE_FULL_SEEDS)
    parser.add_argument("--postgres-seeds", type=positive, default=POSTGRES_FULL_SEEDS)
    parser.add_argument("--sqlite-chunk-size", type=positive, default=SQLITE_CHUNK_SIZE)
    parser.add_argument("--postgres-chunk-size", type=positive, default=POSTGRES_CHUNK_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, choices=range(1, MAX_WORKERS + 1), default=MAX_WORKERS)
    parser.add_argument("--report-dir", required=True, type=Path)
    return parser.parse_args(argv)


def scrub(text, sensitive=()):
    for value in sorted((item for item in sensitive if item), key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    return text


def scrub_json(value, sensitive=()):
    if isinstance(value, str):
        return scrub(value, sensitive)
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            clean_key = scrub_json(key, sensitive)
            if clean_key in cleaned:
                raise ValueError("redaction produced duplicate JSON keys")
            cleaned[clean_key] = scrub_json(item, sensitive)
        return cleaned
    if isinstance(value, list):
        return [scrub_json(item, sensitive) for item in value]
    return value


def prepare_report_dir(path):
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise ValueError("report directory path is not a directory")
        if any(path.iterdir()):
            raise ValueError("report directory must be empty; resume and replay are not supported")
    else:
        path.mkdir(parents=True)
    return path.resolve()


def atomic_write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite retained evidence: {path.name}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def emit(*, identity, event, **values):
    print(json.dumps({"event": event, "identity": identity, **values}, sort_keys=True), flush=True)


def environment(root):
    (root / "tmp").mkdir()
    return {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": str(root),
        "TMPDIR": str(root / "tmp"),
        "PYTHONUNBUFFERED": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "PIP_CONFIG_FILE": os.devnull,
        "PGPASSFILE": os.devnull,
        "UV_CACHE_DIR": str(root / ".uv-cache"),
        "UV_PYTHON_INSTALL_DIR": str(root / ".python"),
        "UV_PROJECT_ENVIRONMENT": str(root / ".venv"),
        "UV_NO_CONFIG": "1",
        "UV_KEYRING_PROVIDER": "disabled",
        "UV_DEFAULT_INDEX": "https://pypi.org/simple",
        "DEBIAN_FRONTEND": "noninteractive",
    }


def run(argv, *, cwd, env, input_text=None, sensitive=()):
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        input=input_text,
        stdin=None if input_text is not None else subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    for output in (result.stdout, result.stderr):
        if output:
            sys.stdout.write(scrub(output, sensitive))
            sys.stdout.flush()
    if result.returncode:
        raise RuntimeError(f"command failed with exit {result.returncode}: {Path(argv[0]).name}")
    return result.stdout.strip()


def clone_exact(repository, revision, destination, *, env):
    destination.mkdir()
    run(["git", "init", "-q"], cwd=destination, env=env)
    run(["git", "remote", "add", "origin", repository], cwd=destination, env=env)
    run(["git", "fetch", "--depth=1", "origin", revision], cwd=destination, env=env)
    run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=destination, env=env)
    actual = run(["git", "rev-parse", "HEAD"], cwd=destination, env=env)
    clean = not run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=destination, env=env)
    if actual != revision or not clean:
        raise RuntimeError(f"checkout identity mismatch for {destination.name}")
    return actual


def prepare_python(source, *, env):
    uv = shutil.which("uv")
    if uv is None:
        run([sys.executable, "-m", "pip", "install", "--user", f"uv=={UV_VERSION}"], cwd=source, env=env)
        uv = str(Path(env["HOME"]) / ".local" / "bin" / "uv")
        if not Path(uv).is_file():
            raise RuntimeError("uv installation did not create the user executable")
    uv_version = run([uv, "--version"], cwd=source, env=env)
    run([uv, "python", "install", "3.12"], cwd=source, env=env)
    managed = run([uv, "python", "find", "--managed-python", "3.12"], cwd=source, env=env)
    run([uv, "sync", "--frozen", "--dev", "--python", managed], cwd=source, env=env)
    python = Path(env["UV_PROJECT_ENVIRONMENT"]) / "bin" / "python"
    version = run([str(python), "--version"], cwd=source, env=env)
    if "3.12" not in version:
        raise RuntimeError(f"managed Python is not 3.12: {version}")
    if run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=source, env=env):
        raise RuntimeError("vv-agent checkout became dirty during setup")
    return str(python), {"uv": uv_version, "python": version, "managed_python": managed}


def source_expectations(python, *, source, env):
    code = (
        "import json;"
        "from tests.sim.scenarios import SCENARIOS;"
        "from tests.sim.world import PENDING;"
        "print(json.dumps({'scenarios':SCENARIOS,'pending':sorted(PENDING)}))"
    )
    payload = json.loads(run([python, "-c", code], cwd=source, env=env).splitlines()[-1])
    pending = frozenset(payload["pending"])
    if pending != EXPECTED_PENDING:
        raise RuntimeError(f"selected source has unexpected pending invariants: {sorted(pending)}")
    scenarios = frozenset(payload["scenarios"])
    if len(scenarios) != SCENARIO_COUNT:
        raise RuntimeError(f"selected source has {len(scenarios)} scenarios; expected {SCENARIO_COUNT}")
    return scenarios


def quote_ident(value):
    if re.fullmatch(r"[a-z_][a-z0-9_]*", value) is None:
        raise ValueError("invalid generated PostgreSQL identifier")
    return f'"{value}"'


def pg_admin(psql, runuser, *, port, query, env):
    result = subprocess.run(
        [
            runuser,
            "-u",
            "postgres",
            "--",
            psql,
            "--no-psqlrc",
            "-X",
            "-qAt",
            "-v",
            "ON_ERROR_STOP=1",
            "-p",
            str(port),
            "-d",
            "postgres",
        ],
        cwd="/",
        env=env,
        input=query,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("PostgreSQL admin command failed")
    return result.stdout.strip()


def prepare_postgres(root, *, env, sensitive):
    run(["apt-get", "update"], cwd=root, env=env)
    run(["apt-get", "install", "-y", "postgresql"], cwd=root, env=env)
    psql = shutil.which("psql", path=env["PATH"])
    runuser = shutil.which("runuser", path=env["PATH"])
    if psql is None or runuser is None:
        raise RuntimeError("PostgreSQL client tools are unavailable")
    run(["service", "postgresql", "start"], cwd=root, env=env)
    port = 5432
    server_version = pg_admin(psql, runuser, port=port, query="SHOW server_version;", env=env)
    role, database, password, wrong_password = (
        f"vv_colab_role_{secrets.token_hex(6)}",
        f"vv_colab_db_{secrets.token_hex(6)}",
        secrets.token_urlsafe(32),
        secrets.token_urlsafe(32),
    )
    sensitive.extend((password, wrong_password))
    state = {
        "role": role,
        "port": port,
        "dsn": None,
        "psql": psql,
        "runuser": runuser,
        "runtime": {
            "package_major": server_version.split(".", 1)[0],
            "server_version": server_version,
            "host": "127.0.0.1",
            "port": port,
            "password_authentication_verified": True,
        },
    }
    try:
        pg_admin(
            psql,
            runuser,
            port=int(port),
            query=(
                f"CREATE ROLE {quote_ident(role)} LOGIN CREATEDB PASSWORD '{password}';"
                f"CREATE DATABASE {quote_ident(database)} OWNER {quote_ident(role)};"
            ),
            env=env,
        )
        auth_command = [
            psql,
            "--no-psqlrc",
            "-X",
            "-qAt",
            "-v",
            "ON_ERROR_STOP=1",
            "-h",
            "127.0.0.1",
            "-p",
            str(port),
            "-U",
            role,
            "-d",
            database,
            "-c",
            "SELECT 1;",
        ]
        auth = subprocess.run(
            auth_command,
            cwd=root,
            env=env | {"PGPASSWORD": password},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if auth.returncode:
            raise RuntimeError("PostgreSQL TCP/password authentication failed")
        wrong_auth = subprocess.run(
            auth_command,
            cwd=root,
            env=env | {"PGPASSWORD": wrong_password},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if wrong_auth.returncode == 0 or "password authentication failed" not in wrong_auth.stderr.lower():
            raise RuntimeError("PostgreSQL TCP did not reject the wrong password")
        state["dsn"] = (
            f"postgresql://{quote(role, safe='')}:{quote(password, safe='')}@127.0.0.1:{port}/"
            f"{quote(database, safe='')}?sslmode=disable&gssencmode=disable"
        )
        sensitive.append(state["dsn"])
        return state
    except Exception:
        cleanup_postgres(state, env=env)
        raise


def cleanup_postgres(state, *, env):
    try:
        role, port = state["role"], state["port"]
        psql, runuser = state["psql"], state["runuser"]
        owned = pg_admin(
            psql,
            runuser,
            port=port,
            query=(
                "SELECT datname FROM pg_database WHERE datdba=(SELECT oid FROM pg_roles WHERE rolname="
                f"'{role}') ORDER BY datname;"
            ),
            env=env,
        )
        pg_admin(
            psql,
            runuser,
            port=port,
            query=f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename='{role}' AND pid<>pg_backend_pid();",
            env=env,
        )
        for name in filter(None, owned.splitlines()):
            pg_admin(psql, runuser, port=port, query=f"DROP DATABASE IF EXISTS {quote_ident(name)};", env=env)
        pg_admin(
            psql,
            runuser,
            port=port,
            query=f"DROP OWNED BY {quote_ident(role)}; DROP ROLE IF EXISTS {quote_ident(role)};",
            env=env,
        )
        state["runtime"]["temporary_role_removed"] = True
        state["runtime"]["temporary_databases_removed"] = True
    finally:
        state["dsn"] = None


def campaign_command(python, *, backend, seed, seeds, workers, report):
    return [
        python,
        "scripts/session_simulation.py",
        "--backend",
        backend,
        "--seed",
        str(seed),
        "--seeds",
        str(seeds),
        "--workers",
        str(workers),
        "--report",
        str(report),
    ]


def stream_campaign(command, *, cwd, env, sensitive=(), diagnostics=None):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("campaign stdout is unavailable")
    try:
        for line in process.stdout:
            cleaned = scrub(line, sensitive)
            sys.stdout.write(cleaned)
            sys.stdout.flush()
            if diagnostics is not None:
                diagnostics["tail"] = (diagnostics.get("tail", "") + cleaned)[-20_000:]
        return process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise


def validate_report(
    report,
    *,
    backend,
    seed,
    seeds,
    identity,
    tempdir,
    scenarios,
    workers,
    contract_source,
):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(isinstance(report, dict), "report must be an object")
    require("error" not in report, "report contains an error")
    require(report.get("backend") == backend, "actual backend mismatch")
    require(report.get("seed") == seed and report.get("seeds") == seeds, "actual seed count mismatch")
    require(report.get("code_revision") == identity["vv_agent_revision"], "actual code revision mismatch")
    require(report.get("contract_revision") == identity["contract_revision"], "actual contract revision mismatch")
    require(report.get("dirty") is False, "actual code checkout is dirty or unknown")
    require(report.get("contract_dirty") is False, "actual contract checkout is dirty or unknown")
    require(report.get("in_memory_mutations") == [], "in-memory source mutations must be empty")
    require(report.get("workers") == workers, "actual worker count mismatch")
    require(report.get("contract_selection") == "local", "canonical contract was not selected")
    require(report.get("contract_source") == str(contract_source), "canonical contract source mismatch")
    actual_pending = report.get("pending")
    require(isinstance(actual_pending, dict) and frozenset(actual_pending) == EXPECTED_PENDING, "actual pending set mismatch")
    counts = report.get("scenarios")
    require(isinstance(counts, dict) and set(counts).issubset(scenarios), "scenario counts are missing or unknown")
    require(all(isinstance(count, int) and count > 0 for count in counts.values()), "invalid scenario count")
    require(sum(counts.values()) == seeds, "scenario seed total mismatch")
    covered = report.get("covered")
    require(isinstance(covered, list) and all(isinstance(item, str) for item in covered), "covered invariants are missing")
    storage = (report.get("environment") or {}).get("sqlite_temporary_storage")
    require(isinstance(storage, dict), "SQLite storage evidence is missing")
    assert isinstance(storage, dict)
    if backend == "sqlite":
        observations = storage.get("observations")
        require(isinstance(observations, list) and observations, "SQLite storage observations are missing")
        require(
            all(
                isinstance(item, dict)
                and item.get("requested_TMPDIR") == str(tempdir)
                and Path(item.get("directory", "/")).is_relative_to("/tmp")
                for item in observations
            ),
            "SQLite campaign TMPDIR does not match the selected /tmp directory",
        )
    else:
        require(storage == {"verified": False, "observations": []}, "PostgreSQL storage evidence is invalid")


def aggregate_campaign(report_dir, *, backend, seed, total, chunk_size, scenarios, identity, tempdir, workers, contract_source):
    paths = sorted(Path(report_dir).glob(f"{backend}-*.json"))
    cursor = seed
    actual_identity = None
    actual_pending = None
    scenario_counts = Counter()
    covered = set()
    reports = []
    storage_verified = backend == "sqlite"
    regular_filesystem = backend == "sqlite"
    for path in paths:
        match = REPORT_NAME.fullmatch(path.name)
        if match is None or match.group(1) != backend:
            raise ValueError(f"retained report has an invalid basename: {path.name}")
        report_seed, report_seeds = int(match.group(2)), int(match.group(3))
        planned_seeds = min(chunk_size, seed + total - cursor)
        if report_seed != cursor or report_seeds != planned_seeds:
            raise ValueError(f"campaign parts are missing, overlapping or non-contiguous at {cursor}")
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"retained report is not readable JSON: {path.name}") from exc
        validate_report(
            report,
            backend=backend,
            seed=cursor,
            seeds=planned_seeds,
            identity=identity,
            tempdir=tempdir,
            scenarios=scenarios,
            workers=workers,
            contract_source=contract_source,
        )
        report_identity = {
            "vv_agent_revision": report.get("code_revision"),
            **{field: report.get(field) for field in IDENTITY_FIELDS if field != "vv_agent_revision"},
        }
        if actual_identity is None:
            actual_identity = report_identity
            actual_pending = report["pending"]
        elif report_identity != actual_identity or report["pending"] != actual_pending:
            raise ValueError("campaign identity or pending metadata is inconsistent")
        scenario_counts.update(report["scenarios"])
        covered.update(report["covered"])
        reports.append(path.name)
        if backend == "sqlite":
            observations = report["environment"]["sqlite_temporary_storage"]["observations"]
            regular_filesystem &= bool(observations) and all(
                item.get("filesystem") not in {None, "tmpfs"} for item in observations
            )
            storage_verified &= report["environment"]["sqlite_temporary_storage"]["verified"] is True
        cursor += planned_seeds
    if actual_identity is None:
        raise ValueError(f"{backend} campaign has no completed part reports")
    if cursor != seed + total:
        raise ValueError(f"{backend} campaign is missing completed part reports")
    if sum(scenario_counts.values()) != total:
        raise ValueError("aggregate scenario counts do not equal completed seeds")
    return {
        "seed": seed,
        "seeds": cursor - seed,
        "scenarios": dict(sorted(scenario_counts.items())),
        "covered": sorted(covered),
        "pending": actual_pending,
        "reports": reports,
        "storage": {
            "verified": storage_verified,
            "regular_filesystem": regular_filesystem,
        },
        "identity": actual_identity,
    }


def execute(args):
    report_dir = None
    root = None
    postgres = {}
    runtime = {}
    sensitive = []
    identity = {
        "vv_agent_repository": VV_AGENT_REPOSITORY,
        "vv_agent_revision": args.vv_agent_revision,
        "contract_repository": CONTRACT_REPOSITORY,
        "contract_revision": args.contract_revision,
    }
    try:
        if args.seed < 0:
            raise ValueError("seed must be nonnegative")
        report_dir = prepare_report_dir(args.report_dir)
        with TemporaryDirectory(prefix="vv-agent-colab-", dir="/tmp") as directory:
            root = Path(directory)
            env = environment(root)
            source, contract = root / "vv-agent", root / "vv-agent-contract"
            identity = identity | {
                "vv_agent_revision": clone_exact(VV_AGENT_REPOSITORY, args.vv_agent_revision, source, env=env),
                "contract_revision": clone_exact(CONTRACT_REPOSITORY, args.contract_revision, contract, env=env),
            }
            contract_source = contract.resolve()
            python, python_runtime = prepare_python(source, env=env)
            runtime = {"python": python_runtime, "postgres": {}, "temporary_directory_removed": False}
            campaign_env = env | {
                "PYTHONPATH": f"{source / 'src'}:{source}",
                "VV_AGENT_CONTRACT_SOURCE": str(contract),
            }
            expected_scenarios = source_expectations(python, source=source, env=campaign_env)
            postgres = prepare_postgres(root, env=env, sensitive=sensitive)
            runtime["postgres"] = postgres["runtime"]
            try:
                for backend, total, chunk_size in (
                    ("sqlite", args.sqlite_seeds, args.sqlite_chunk_size),
                    ("postgres", args.postgres_seeds, args.postgres_chunk_size),
                ):
                    child_env = campaign_env.copy()
                    if backend == "postgres":
                        child_env["VV_AGENT_TEST_POSTGRES_DSN"] = postgres["dsn"]
                    for part_seed in range(args.seed, args.seed + total, chunk_size):
                        part_seeds = min(chunk_size, args.seed + total - part_seed)
                        basename = f"{backend}-{part_seed:010d}-{part_seeds:010d}.json"
                        retained = report_dir / basename
                        temporary_report = root / basename
                        diagnostics = {"tail": ""}
                        emit(
                            identity=identity,
                            event="part_started",
                            backend=backend,
                            seed=part_seed,
                            seeds=part_seeds,
                            report=basename,
                        )
                        report = None
                        returncode = 0
                        try:
                            returncode = stream_campaign(
                                campaign_command(
                                    python,
                                    backend=backend,
                                    seed=part_seed,
                                    seeds=part_seeds,
                                    workers=args.workers,
                                    report=temporary_report,
                                ),
                                cwd=source,
                                env=child_env,
                                sensitive=sensitive,
                                diagnostics=diagnostics,
                            )
                            try:
                                report = json.loads(temporary_report.read_text())
                            except (OSError, json.JSONDecodeError) as exc:
                                raise ValueError(f"simulation CLI report is missing or invalid: {exc}") from exc
                            if returncode:
                                raise ValueError(f"simulation CLI exited with status {returncode}")
                            if "error" in report:
                                raise ValueError("simulation CLI report contains an error")
                            validate_report(
                                report,
                                backend=backend,
                                seed=part_seed,
                                seeds=part_seeds,
                                identity=identity,
                                tempdir=root / "tmp",
                                scenarios=expected_scenarios,
                                workers=args.workers,
                                contract_source=contract_source,
                            )
                            atomic_write_json(retained, scrub_json(report, sensitive))
                        except Exception as exc:
                            failure = {"backend": backend, "seed": part_seed, "seeds": part_seeds} | (
                                report if isinstance(report, dict) else {}
                            )
                            failure.setdefault("error", f"{type(exc).__name__}: {exc}")
                            failure.setdefault("trace", diagnostics["tail"] or traceback.format_exc())
                            failure.setdefault("returncode", returncode)
                            if not retained.exists():
                                atomic_write_json(retained, scrub_json(failure, sensitive))
                            emit(
                                identity=identity,
                                event="part_failed",
                                backend=backend,
                                seed=part_seed,
                                seeds=part_seeds,
                                report=basename,
                                returncode=returncode,
                                message=scrub(str(exc), sensitive)[:1000],
                            )
                            raise ValueError(f"{basename}: {exc}") from exc
                        emit(
                            identity=identity,
                            event="part_completed",
                            backend=backend,
                            seed=part_seed,
                            seeds=part_seeds,
                            report=basename,
                        )
            finally:
                cleanup_postgres(postgres, env=env)
        runtime["temporary_directory_removed"] = not root.exists()
        campaigns = {
            backend: aggregate_campaign(
                report_dir,
                backend=backend,
                seed=args.seed,
                total=total,
                chunk_size=args.sqlite_chunk_size if backend == "sqlite" else args.postgres_chunk_size,
                scenarios=expected_scenarios,
                identity=identity,
                tempdir=root / "tmp",
                workers=args.workers,
                contract_source=contract_source,
            )
            for backend, total in (("sqlite", args.sqlite_seeds), ("postgres", args.postgres_seeds))
        }
        sqlite, postgres_campaign = campaigns["sqlite"], campaigns["postgres"]
        postgres_runtime = runtime["postgres"]
        accepted = (
            sqlite["seeds"] == args.sqlite_seeds >= SQLITE_FULL_SEEDS
            and postgres_campaign["seeds"] == args.postgres_seeds >= POSTGRES_FULL_SEEDS
            and all(
                set(campaign["scenarios"]) == set(expected_scenarios)
                and all(campaign["scenarios"].values())
                and REQUIRED_COVERED.issubset(campaign["covered"])
                and set(campaign["pending"]) == EXPECTED_PENDING
                for campaign in (sqlite, postgres_campaign)
            )
            and sqlite["storage"]["verified"]
            and sqlite["storage"]["regular_filesystem"]
            and postgres_runtime.get("password_authentication_verified") is True
            and postgres_runtime.get("temporary_role_removed") is True
            and postgres_runtime.get("temporary_databases_removed") is True
            and runtime.get("temporary_directory_removed") is True
            and sqlite["identity"] == {field: postgres_campaign["identity"].get(field) for field in IDENTITY_FIELDS}
        )
        full_requested = args.sqlite_seeds >= SQLITE_FULL_SEEDS and args.postgres_seeds >= POSTGRES_FULL_SEEDS
        status = "complete" if accepted or not full_requested else "failed"
        report_identity = next(iter(campaigns.values()))["identity"]
        result_identity = identity | report_identity
        result = {
            "status": status,
            "full_acceptance": accepted,
            "identity": result_identity,
            "campaigns": campaigns,
            "runtime": runtime,
        }
        if status == "failed":
            result["error"] = {"message": "configured full campaign did not satisfy acceptance evidence"}
        atomic_write_json(report_dir / RESULT_NAME, scrub_json(result, sensitive))
        emit(identity=result_identity, event=status, full_acceptance=accepted, runtime=runtime)
        return 0 if status == "complete" else 1
    except Exception as exc:
        if root is not None and runtime:
            runtime["temporary_directory_removed"] = not root.exists()
        error = {
            "message": scrub(str(exc), sensitive)[:1000],
            "trace": scrub(traceback.format_exc(), sensitive)[-20_000:],
        }
        if report_dir is not None and not (report_dir / RESULT_NAME).exists():
            saved_reports = sorted(path.name for path in report_dir.iterdir() if REPORT_NAME.fullmatch(path.name) is not None)
            result = {
                "status": "failed",
                "full_acceptance": False,
                "error": error,
                "runtime": runtime,
                "reports": saved_reports,
            }
            atomic_write_json(report_dir / RESULT_NAME, scrub_json(result, sensitive))
        emit(identity=identity, event="failed", returncode=1, message=error["message"])
        return 1


def main(argv=None):
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
