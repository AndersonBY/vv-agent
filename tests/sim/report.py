"""Disposable storage, replay metadata and bounded failure minimization."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import marshal
import multiprocessing
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import cache
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
MAX_WORKERS = 8
AFFINITY = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
DEFAULT_WORKERS = min(5, len(AFFINITY) if AFFINITY is not None else (os.cpu_count() or 1))


@contextmanager
def database(backend):
    if backend == "sqlite":
        with TemporaryDirectory(prefix="vv-agent-sim-") as directory:
            yield Path(directory) / "sessions.sqlite"
        return
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    dsn = os.environ.get("VV_AGENT_TEST_POSTGRES_DSN", "dbname=postgres")
    name = f"vvsk_sim_{uuid4().hex}"
    with psycopg.connect(dsn, autocommit=True, connect_timeout=3) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            yield make_conninfo(dsn, dbname=name)
        finally:
            admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))


def _git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


@cache
def _snapshot_tools():
    spec = importlib.util.spec_from_file_location("_sim_contract_snapshot", ROOT / "scripts/contract_snapshot.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract_identity(lock):
    tools = _snapshot_tools()
    SnapshotError = tools.SnapshotError

    pinned = {key: lock[key] for key in ("contract_version", "contract_revision", "fixture_manifest_sha256")}
    selected = os.environ.get("VV_AGENT_CONTRACT_SOURCE")
    if selected is not None and not selected.strip():
        raise SnapshotError("VV_AGENT_CONTRACT_SOURCE must name a canonical source directory")
    # Same selection as tests/support/contract.py, plus validation for replay.
    source = Path(selected).resolve() if selected is not None else None
    fixtures = source / "fixtures" if source is not None else ROOT / "tests/fixtures/parity"
    manifest = tools.validate_snapshot(fixtures)["manifest_sha256"]
    revision, dirty = lock["contract_revision"], None
    version, expected_manifest = lock["contract_version"], lock["fixture_manifest_sha256"]
    if source is not None:
        canonical = tools.load_json(source / "contract.json")
        if (
            not isinstance(canonical, dict)
            or canonical.get("schema_version") != 1
            or canonical.get("name") != "vv-agent-contract"
            or not isinstance(canonical.get("version"), str)
            or tools.SEMVER_RE.fullmatch(canonical["version"]) is None
        ):
            raise SnapshotError("local contract.json must declare vv-agent-contract and a valid version")
        configuration = canonical.get("fixtures")
        if not isinstance(configuration, dict) or configuration.get("path") != "fixtures":
            raise SnapshotError("local contract fixtures.path must be fixtures")
        version, expected_manifest = canonical["version"], configuration.get("manifest_sha256")
        revision = None
        if _git(source, "rev-parse", "--show-toplevel") == str(source):
            revision = _git(source, "rev-parse", "HEAD")
            status = _git(source, "status", "--porcelain")
            dirty = bool(status) if status is not None else None
    if manifest != expected_manifest:
        raise SnapshotError("selected contract fixture manifest does not match its declared identity")
    return {
        "contract_selection": "local" if source is not None else "released",
        "contract_version": version,
        "contract_revision": revision,
        "contract_manifest_sha256": manifest,
        "contract_source": str(source if source is not None else fixtures),
        "contract_dirty": dirty,
        "pinned_contract": pinned,
    }


def versions():
    lock = json.loads((ROOT / "contract.lock.json").read_text())
    checksum = hashlib.sha256()
    files = [
        *ROOT.joinpath("src").rglob("*.py"),
        *ROOT.joinpath("tests/sim").glob("*.py"),
        ROOT / "scripts/session_simulation.py",
        ROOT / "scripts/contract_snapshot.py",
    ]
    for file in sorted(files):
        checksum.update(file.relative_to(ROOT).as_posix().encode() + b"\0" + file.read_bytes())
    status = _git(ROOT, "status", "--porcelain")
    return contract_identity(lock) | {
        "code_digest": checksum.hexdigest(),
        "code_revision": _git(ROOT, "rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "in_memory_mutations": mutation_identity(),
    }


def mutation_identity():
    """Identify the actual functions changed by the five implementation mutants."""
    from vv_agent.session import _driver_recovery, kernel
    from vv_agent.session.sql import SQLSessionTx

    functions = {
        "vv_agent.session.sql.SQLSessionTx.append": SQLSessionTx.append,
        "vv_agent.session.sql.SQLSessionTx.push": SQLSessionTx.push,
        "vv_agent.session._driver_recovery.step": _driver_recovery.step,
        "vv_agent.session._driver_recovery.unknown": _driver_recovery.unknown,
        "vv_agent.session.kernel._LeaseHeartbeat.poll": kernel._LeaseHeartbeat.poll,
    }
    return [
        {
            "target": target,
            "mutation": getattr(function, "_sim_mutation", function.__code__.co_filename),
            "bytecode_digest": hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
        }
        for target, function in functions.items()
        if hasattr(function, "_sim_mutation") or function.__code__.co_filename.startswith("<mutation:")
    ]


def replay_provenance(recorded, current):
    """Source/contract/mutant differences reject replay; Git availability is metadata."""
    required = (
        "code_digest",
        "contract_selection",
        "contract_version",
        "contract_manifest_sha256",
        "pinned_contract",
        "in_memory_mutations",
    )
    differences = {
        field: {"recorded": recorded.get(field), "current": current[field]}
        for field in (*required, "contract_revision", "code_revision", "dirty", "contract_source", "contract_dirty")
        if recorded.get(field) != current[field] or field not in recorded
    }
    known_revision_changed = (
        recorded.get("contract_revision") is not None
        and current["contract_revision"] is not None
        and recorded["contract_revision"] != current["contract_revision"]
    )
    return {
        "matched": not known_revision_changed and not any(field in differences for field in required),
        "differences": differences,
    }


def _run_seed(job):
    # Imported in the child, after spawn; no SQL connection crosses processes.
    from .scenarios import run_scenario
    from .world import InvariantViolation, SimWorld

    backend, seed, scenario, replay, budget = job
    with database(backend) as db, SimWorld(db, seed=seed, replay=replay) as world:
        storage = (
            {"sqlite_temporary_storage": temporary_storage(db.parent.parent)} if isinstance(db, Path) and db.is_file() else {}
        )
        try:
            summary = run_scenario(world, scenario)
            if replay is not None and world.choices.used != set(world.choices.overrides):
                raise ValueError("replay has unused decisions; code or scenario differs")
            return summary | storage
        except Exception as exc:
            failure = world.summary() | storage | {"error": f"{type(exc).__name__}: {exc}"}
            invariant = exc.invariant if isinstance(exc, InvariantViolation) else None
    failure["minimized_decisions"], failure["minimization_trials"] = failure["decisions"], 0
    if invariant and budget:

        def reproduces(candidate):
            try:
                with database(backend) as trial_db, SimWorld(trial_db, seed=seed, replay=candidate) as trial:
                    run_scenario(trial, failure["scenario"])
            except InvariantViolation as error:
                return error.invariant == invariant
            except Exception:
                return False
            return False

        failure["minimized_decisions"], failure["minimization_trials"] = minimize(failure["decisions"], reproduces, budget=budget)
    return failure


def seed_results(*, backend="sqlite", seed=0, seeds=1, scenario=None, workers=1, replay=None, minimize_budget=0):
    """Bound submission AND result memory independently of campaign length."""
    if seeds < 1 or not 1 <= workers <= MAX_WORKERS or minimize_budget < 0:
        raise ValueError(f"positive seeds, 1..{MAX_WORKERS} workers and a nonnegative minimization budget required")
    if replay is not None and seeds != 1:
        raise ValueError("a replay runs exactly one seed")
    jobs = ((backend, number, scenario, replay, minimize_budget) for number in range(seed, seed + seeds))
    if workers == 1:
        for job in jobs:
            yield _run_seed(job)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for batch in batched(jobs, workers * 2):
            yield from pool.map(_run_seed, batch)


@cache
def _filesystem(directory):
    # One probe per worker/location, not one subprocess per seed.
    try:
        result = subprocess.run(["stat", "-f", "-c", "%T", "--", directory], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        return None


def temporary_storage(directory=None):
    """Describe the supplied actual database location, or this process's default."""
    directory = str(Path(directory if directory is not None else gettempdir()).resolve())
    return {"directory": directory, "filesystem": _filesystem(directory), "requested_TMPDIR": os.environ.get("TMPDIR")}


def run_campaign(*, backend="sqlite", seed=0, seeds=1000, scenario=None, workers=DEFAULT_WORKERS, replay=None, minimize_budget=0):
    """Time identity, child imports/warmup, storage, checks and pool shutdown."""
    started = time.perf_counter()
    from .world import PENDING

    identity = versions() | {
        "workers": workers,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "cpu_affinity": AFFINITY,
            "sqlite_temporary_storage": {"verified": False, "observations": []},
            "visible_memory_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else None,
        },
    }
    provenance = replay_provenance(replay, identity) if replay is not None else None
    if provenance is not None and not provenance["matched"]:
        return identity | {
            "error": "Replay provenance mismatch; source, contract and in-memory mutations must match",
            "replay_provenance": provenance,
        }
    if identity["in_memory_mutations"] and workers > 1:
        raise ValueError("in-memory mutants require workers=1; spawned workers execute the source on disk")
    if replay is not None:
        seed, seeds, backend, scenario = replay["seed"], 1, replay["backend"], replay["scenario"]
    scenarios, decisions, covered = {}, 0, set()
    results = seed_results(
        backend=backend,
        seed=seed,
        seeds=seeds,
        scenario=scenario,
        workers=workers,
        replay=replay["decisions"] if replay is not None else None,
        minimize_budget=minimize_budget,
    )
    try:
        for result in results:
            storage = identity["environment"]["sqlite_temporary_storage"]
            observed = result.get("sqlite_temporary_storage")
            if observed is not None and observed not in storage["observations"]:
                storage["observations"].append(observed)
                storage["verified"] = all(item["filesystem"] is not None for item in storage["observations"])
            if "error" in result:
                return result | identity | {"replay_provenance": provenance}
            scenarios[result["scenario"]] = scenarios.get(result["scenario"], 0) + 1
            decisions += len(result["decisions"])
            covered.update(result["covered"])
    finally:
        results.close()
    elapsed = time.perf_counter() - started
    return identity | {
        "backend": backend,
        "seed": seed,
        "seeds": seeds,
        "workers": workers,
        "elapsed_seconds": round(elapsed, 4),
        "seeds_per_minute": round(seeds / elapsed * 60, 1),
        "scenarios": scenarios,
        "decisions": decisions,
        "covered": sorted(covered),
        "pending": PENDING,
        "replay_provenance": provenance,
    }


def minimize(decisions, reproduces, *, budget=12):
    """Bounded ddmin; only retain deletions reproducing the same invariant failure."""
    result, trials, width = list(decisions), 0, max(1, len(decisions) // 2)
    while width and trials < budget:
        changed = False
        for offset in range(0, len(result), width):
            candidate = result[:offset] + result[offset + width :]
            trials += 1
            if reproduces(candidate):
                result, changed = candidate, True
                break
            if trials >= budget:
                break
        if not changed:
            width //= 2
    return result, trials
