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
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

VV_AGENT_REPOSITORY = "https://github.com/AndersonBY/vv-agent.git"
CONTRACT_REPOSITORY = "https://github.com/AndersonBY/vv-agent-contract.git"
SQLITE_FULL_SEEDS = 1_000_000
POSTGRES_FULL_SEEDS = 10_000
MAX_WORKERS = 8
UV_VERSION = "0.10.4"
REQUIRED_COVERED = frozenset({"I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I10"})
EXPECTED_PENDING = frozenset({"I9", "I11"})
FULL_REVISION = re.compile(r"[0-9a-f]{40}")


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, choices=range(1, MAX_WORKERS + 1), default=MAX_WORKERS)
    return parser.parse_args(argv)


def scrub(text, sensitive=()):
    for value in sorted((item for item in sensitive if item), key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    return text


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
    return frozenset(payload["scenarios"]), EXPECTED_PENDING


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


def prepare_postgres(root, *, env):
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
    state = {
        "role": role,
        "port": port,
        "dsn": None,
        "psql": psql,
        "runuser": runuser,
        "sensitive": (password, wrong_password, ""),
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
        state["sensitive"] = (password, wrong_password, state["dsn"])
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


def stream_campaign(command, *, cwd, env, sensitive=()):
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
            sys.stdout.write(scrub(line, sensitive))
            sys.stdout.flush()
        return process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise


def validate_report(report, *, backend, seed, seeds, identity, tempdir, scenarios, pending, contract_source=None):
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
    if contract_source is not None:
        require(report.get("contract_selection") == "local", "canonical contract was not selected")
        require(report.get("contract_source") == str(contract_source), "canonical contract source mismatch")
    actual_pending = report.get("pending")
    if not isinstance(actual_pending, dict):
        raise ValueError("pending metadata is missing")
    require(frozenset(actual_pending) == pending, "actual pending set mismatch")
    counts = report.get("scenarios")
    require(isinstance(counts, dict), "scenario counts are missing")
    if isinstance(counts, dict):
        require(set(counts).issubset(scenarios), "unknown scenario in report")
        require(all(isinstance(count, int) and count > 0 for count in counts.values()), "invalid scenario count")
        require(sum(counts.values()) == seeds, "scenario seed total mismatch")
        if seeds >= (SQLITE_FULL_SEEDS if backend == "sqlite" else POSTGRES_FULL_SEEDS):
            require(set(counts) == scenarios, "full campaign did not run every source scenario")
    covered = report.get("covered")
    require(isinstance(covered, list), "covered invariants are missing")
    full = seeds >= (SQLITE_FULL_SEEDS if backend == "sqlite" else POSTGRES_FULL_SEEDS)
    if full:
        require(isinstance(covered, list) and REQUIRED_COVERED.issubset(covered), "required invariants are not covered")
    storage = (report.get("environment") or {}).get("sqlite_temporary_storage")
    require(isinstance(storage, dict), "SQLite storage evidence is missing")
    if backend == "sqlite" and isinstance(storage, dict):
        observations = storage.get("observations")
        require(isinstance(observations, list) and bool(observations), "SQLite storage observations are missing")
        if isinstance(observations, list):
            require(
                all(
                    isinstance(item, dict)
                    and item.get("requested_TMPDIR") == str(tempdir)
                    and Path(item.get("directory", "/")).is_relative_to("/tmp")
                    for item in observations
                ),
                "SQLite campaign TMPDIR does not match the selected /tmp directory",
            )
        if full:
            require(storage.get("verified") is True, "full SQLite storage is unverified")
            require(
                isinstance(observations, list)
                and all(isinstance(item, dict) and item.get("filesystem") not in {None, "tmpfs"} for item in observations),
                "full SQLite campaign did not verify regular filesystem storage",
            )
    elif isinstance(storage, dict):
        require(storage.get("verified") is False, "PostgreSQL storage should have no SQLite verification")
        require(storage.get("observations") == [], "PostgreSQL campaign unexpectedly has SQLite observations")


def execute(args):
    identity = {
        "vv_agent_repository": VV_AGENT_REPOSITORY,
        "vv_agent_revision": args.vv_agent_revision,
        "contract_repository": CONTRACT_REPOSITORY,
        "contract_revision": args.contract_revision,
    }
    campaign_returncode, sensitive = 0, []
    try:
        with TemporaryDirectory(prefix="vv-agent-colab-", dir="/tmp") as directory:
            root = Path(directory)
            env = environment(root)
            source, contract = root / "vv-agent", root / "vv-agent-contract"
            identity = identity | {
                "vv_agent_revision": clone_exact(VV_AGENT_REPOSITORY, args.vv_agent_revision, source, env=env),
                "contract_revision": clone_exact(CONTRACT_REPOSITORY, args.contract_revision, contract, env=env),
            }
            python, python_runtime = prepare_python(source, env=env)
            campaign_env = env | {
                "PYTHONPATH": f"{source / 'src'}:{source}",
                "VV_AGENT_CONTRACT_SOURCE": str(contract),
            }
            expected_scenarios, expected_pending = source_expectations(python, source=source, env=campaign_env)
            postgres = prepare_postgres(root, env=env)
            sensitive = list(postgres["sensitive"])
            runtime = python_runtime | {"postgres": postgres["runtime"]}
            try:
                for backend, seeds in (("sqlite", args.sqlite_seeds), ("postgres", args.postgres_seeds)):
                    report_path = root / f"{backend}-report.json"
                    child_env = campaign_env.copy()
                    if backend == "postgres":
                        child_env["VV_AGENT_TEST_POSTGRES_DSN"] = postgres["dsn"]
                    campaign_returncode = stream_campaign(
                        campaign_command(
                            python,
                            backend=backend,
                            seed=args.seed,
                            seeds=seeds,
                            workers=args.workers,
                            report=report_path,
                        ),
                        cwd=source,
                        env=child_env,
                        sensitive=sensitive,
                    )
                    if campaign_returncode:
                        emit(
                            identity=identity,
                            event="failed",
                            backend=backend,
                            returncode=campaign_returncode,
                            message="simulation CLI failed; inspect the redacted streamed report without automatic retry",
                        )
                        return campaign_returncode
                    report = json.loads(report_path.read_text())
                    validate_report(
                        report,
                        backend=backend,
                        seed=args.seed,
                        seeds=seeds,
                        identity=identity,
                        tempdir=root / "tmp",
                        scenarios=expected_scenarios,
                        pending=expected_pending,
                        contract_source=contract,
                    )
            finally:
                cleanup_postgres(postgres, env=env)
            full_acceptance = args.sqlite_seeds >= SQLITE_FULL_SEEDS and args.postgres_seeds >= POSTGRES_FULL_SEEDS
            emit(
                identity=identity,
                event="complete",
                full_acceptance=full_acceptance,
                runtime=runtime,
            )
            return 0
    except Exception as exc:
        code = campaign_returncode or 1
        emit(identity=identity, event="failed", returncode=code, message=scrub(str(exc), sensitive)[:1000])
        return code


def main(argv=None):
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
