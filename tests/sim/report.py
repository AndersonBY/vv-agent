"""Disposable storage, replay metadata and bounded failure minimization."""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import json
import marshal
import multiprocessing
import os
import platform
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, suppress
from functools import cache, lru_cache
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Any, Self
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
# The formal 1,000-seed gate runs on the 20-CPU reference host. Spawn fallback
# plans are additionally capped to the process affinity below.
MAX_WORKERS = 20
POSTGRES_CONNECTIONS_PER_WORKER = 16
POSTGRES_ADMIN_CONNECTION_RESERVE = 4
POSTGRES_CLEANUP_RETRY_DELAYS = (0.025, 0.05, 0.1, 0.2)
POSTGRES_SEEDS_PER_BATCH = 16
POSTGRES_TEST_RESET_TABLE = "sim_effect"
AFFINITY = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
DEFAULT_WORKERS = min(MAX_WORKERS, len(AFFINITY) if AFFINITY is not None else (os.cpu_count() or 1))


class _PostgresDatabase(str):
    database_name: str
    cleanup_evidence: dict[str, Any] | None
    cleanup_error: dict[str, str] | None

    def __new__(cls, value: str) -> Self:
        instance = super().__new__(cls, value)
        instance.cleanup_evidence = None
        instance.cleanup_error = None
        return instance


def canonical_scenario_inventory():
    from .scenarios import SCENARIOS

    inventory = sorted(SCENARIOS)
    if not all(isinstance(name, str) and name for name in inventory) or len(inventory) != len(set(inventory)):
        raise ValueError("SCENARIOS must contain unique nonempty string names")
    return inventory


def scenario_inventory_sha256(inventory):
    return hashlib.sha256(json.dumps(inventory, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def postgres_connection_counts(dsn):
    import psycopg
    from psycopg.conninfo import make_conninfo

    admin_dsn = make_conninfo(dsn, options="-c statement_timeout=0 -c lock_timeout=0")
    with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=3) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT current_setting('max_connections')::int,"
            "current_setting('superuser_reserved_connections')::int,"
            "(SELECT count(*)::int FROM pg_stat_activity)"
        )
        row: tuple[int, int, int] | None = cursor.fetchone()
        if row is None:
            raise RuntimeError("PostgreSQL connection projection returned no row")
        return row


def postgres_worker_plan(workers):
    """Cap one campaign from a startup slot snapshot; no cross-campaign lock is created."""
    dsn = os.environ.get("VV_AGENT_TEST_POSTGRES_DSN", "dbname=postgres")
    maximum, reserved, active = postgres_connection_counts(dsn)
    available = maximum - reserved - active - POSTGRES_ADMIN_CONNECTION_RESERVE
    if available < POSTGRES_CONNECTIONS_PER_WORKER:
        raise RuntimeError(
            "PostgreSQL cannot admit one simulation worker safely: "
            f"{available} slots available after reserves, {POSTGRES_CONNECTIONS_PER_WORKER} required"
        )
    return {
        "requested_workers": workers,
        "effective_workers": min(workers, available // POSTGRES_CONNECTIONS_PER_WORKER),
        "connections_per_worker": POSTGRES_CONNECTIONS_PER_WORKER,
        "admin_connection_reserve": POSTGRES_ADMIN_CONNECTION_RESERVE,
        "max_connections": maximum,
        "superuser_reserved_connections": reserved,
        "active_connections": active,
        "available_connections": available,
        "reservation_scope": "single_campaign",
        "capacity_source": "startup_snapshot",
    }


def simulation_worker_plan(backend, workers):
    if backend == "postgres":
        return postgres_worker_plan(workers) | {
            "lifecycle": "bounded_seed_batch",
            "seeds_per_batch": POSTGRES_SEEDS_PER_BATCH,
            "reset_strategy": "transactional_truncate_s1_drop_test_tables",
            "cleanup_strategy": "terminate_current_role_then_drop",
        }
    return {"requested_workers": workers, "effective_workers": workers}


def _has_open_sqlite_connections():
    """Detect a live parent SQLite handle before forking a worker pool."""
    import sqlite3

    for candidate in gc.get_objects():
        if not isinstance(candidate, sqlite3.Connection):
            continue
        try:
            _ = candidate.in_transaction
        except sqlite3.ProgrammingError:
            continue
        return True
    return False


def _has_active_worker_threads():
    """A fork is safe only when no non-main worker thread can own shared locks."""
    main_thread = threading.main_thread()
    return any(thread.is_alive() for thread in threading.enumerate() if thread is not main_thread)


def _process_start_method(backend):
    """Use fork only for single-threaded SQLite parents without inheritable handles."""
    if (
        backend == "sqlite"
        and "fork" in multiprocessing.get_all_start_methods()
        and not _has_active_worker_threads()
        and not _has_open_sqlite_connections()
    ):
        return "fork"
    return "spawn"


@cache
def _warm_fork_worker_runtime():
    """Load the tokenizer and provider imports once for workers that inherit them."""
    from vv_agent._token_counting import count_messages_tokens

    count_messages_tokens([{"role": "user", "content": "fork worker warmup"}], model="scripted")


def _worker_process_context(backend, workers):
    """Resolve and warm one final process context for this worker pool."""
    if workers == 1:
        return "serial", None
    start_method = _process_start_method(backend)
    if start_method == "fork":
        _warm_fork_worker_runtime()
        # Warmup may initialize native threads or database handles.
        start_method = _process_start_method(backend)
    return start_method, multiprocessing.get_context(start_method)


def _spawn_worker_limit():
    """Bound spawned interpreters by the CPUs this process can actually run on."""
    return max(1, len(AFFINITY) if AFFINITY is not None else (os.cpu_count() or 1))


def _prepare_worker_execution(backend, plan):
    """Freeze the final start method, context and resource-limited worker count."""
    start_method, mp_context = _worker_process_context(backend, plan["effective_workers"])
    if start_method == "spawn":
        limit = _spawn_worker_limit()
        plan = plan | {
            "effective_workers": min(plan["effective_workers"], limit),
            "spawn_worker_limit": limit,
            "resource_source": "process_affinity",
        }
    if plan["effective_workers"] == 1:
        return plan, "serial", None
    return plan, start_method, mp_context


def _cleanup_postgres_database(admin, name):
    """Terminate this role's sessions in the temporary database, then drop it."""
    import psycopg
    from psycopg import sql

    delays = (*POSTGRES_CLEANUP_RETRY_DELAYS, None)
    for attempt, delay in enumerate(delays, start=1):
        try:
            with admin.cursor() as cursor:
                cursor.execute("SELECT current_user")
                role_row: tuple[str] | None = cursor.fetchone()
                if role_row is None:
                    raise RuntimeError("PostgreSQL cleanup could not resolve current_user")
                role = role_row[0]
                cursor.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = %s AND usename = %s AND pid <> pg_backend_pid()",
                    (name, role),
                )
                signal_rows: list[tuple[bool]] = cursor.fetchall()
                signals = [result[0] for result in signal_rows if result]
                cursor.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
            return {
                "database": name,
                "role": role,
                "same_role_sessions": len(signals),
                "terminated_sessions": sum(signal is True for signal in signals),
                "drop_attempts": attempt,
                "drop_completed": True,
            }
        except psycopg.errors.ObjectInUse:
            if delay is None:
                raise
            time.sleep(delay)
    raise AssertionError("PostgreSQL cleanup retry loop exhausted without an outcome")


def _reset_postgres_schema(database_handle, *, seed, batch_id, purpose="seed"):
    """Clear S1 and test-owned relations in one transaction after each world closes."""
    import psycopg
    from psycopg import sql

    from vv_agent.kernel._postgres import SESSION_TABLES

    tables = sorted(SESSION_TABLES)
    test_table = POSTGRES_TEST_RESET_TABLE
    test_table_exists = (
        "SELECT current_schema(),"
        "to_regclass(quote_ident(current_schema()) || '.' || quote_ident(%s)) IS NOT NULL,"
        "floor(extract(epoch FROM clock_timestamp()) * 1000)::bigint"
    )
    test_table_absent = (
        "SELECT to_regclass(quote_ident(current_schema()) || '.' || quote_ident(%s)) IS NULL,"
        "(SELECT count(*) FROM pg_constraint con "
        "JOIN pg_class rel ON rel.oid = con.conrelid "
        "JOIN pg_namespace n ON n.oid = rel.relnamespace "
        "WHERE n.nspname = current_schema() "
        "AND rel.relname = ANY(%s) AND con.contype = 'f')"
    )
    drop_test_table = sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(test_table))
    truncate = sql.SQL("TRUNCATE TABLE {}").format(sql.SQL(", ").join(map(sql.Identifier, tables)))
    with (
        psycopg.connect(str(database_handle), connect_timeout=3) as connection,
        connection.transaction(),
        connection.cursor() as cursor,
    ):
        cursor.execute(test_table_exists, (test_table,))
        before = cursor.fetchone()
        if before is None:
            raise RuntimeError("PostgreSQL reset could not inspect the test table")
        schema, existed_before, reset_at_ms = before
        cursor.execute(drop_test_table)
        cursor.execute(truncate)
        cursor.execute(test_table_absent, (test_table, tables))
        after = cursor.fetchone()
    if after is None:
        raise RuntimeError("PostgreSQL reset evidence returned no row")
    absent_after, foreign_key_constraints = after
    exists_after = absent_after is False
    return {
        "batch_id": batch_id,
        "worker_pid": os.getpid(),
        "seed": seed,
        "purpose": purpose,
        "strategy": "single_transaction_truncate_s1_drop_test_tables",
        "schema": schema,
        "tables": tables,
        "foreign_key_constraints": foreign_key_constraints,
        "test_tables": {
            "strategy": "drop_if_exists",
            "names": [POSTGRES_TEST_RESET_TABLE],
            "existed_before": existed_before,
            "dropped": [test_table] if existed_before else [],
            "exists_after": exists_after,
            "cleanup_completed": exists_after is False,
        },
        "reset_at_ms": reset_at_ms,
        "reset_completed": True,
    }


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
    admin_dsn = make_conninfo(dsn, options="-c statement_timeout=0 -c lock_timeout=0")
    with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=3) as admin:
        with admin.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        database_handle = _PostgresDatabase(make_conninfo(dsn, dbname=name))
        database_handle.database_name = name
        original_error = None
        try:
            yield database_handle
        except BaseException as exc:
            original_error = exc
            raise
        finally:
            try:
                database_handle.cleanup_evidence = _cleanup_postgres_database(admin, name)
            except Exception as cleanup_error:
                database_handle.cleanup_error = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                }
                if original_error is None:
                    raise
                original_error.add_note(f"PostgreSQL cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")


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
    inventory = canonical_scenario_inventory()
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
        "scenario_inventory": inventory,
        "scenario_inventory_sha256": scenario_inventory_sha256(inventory),
        "code_revision": _git(ROOT, "rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "in_memory_mutations": mutation_identity(),
    }


def mutation_identity():
    """Identify the actual functions changed by the five implementation mutants."""
    from vv_agent.kernel import _driver_outcomes, _driver_recovery
    from vv_agent.kernel._lease_scope import LeaseOwnership
    from vv_agent.kernel._sql import SQLSessionTx

    functions = {
        "vv_agent.kernel._sql.SQLSessionTx.append": SQLSessionTx.append,
        "vv_agent.kernel._sql.SQLSessionTx.push": SQLSessionTx.push,
        "vv_agent.kernel._driver_recovery.step": _driver_recovery.step,
        "vv_agent.kernel._driver_outcomes.unknown_steps": _driver_outcomes.unknown_steps,
        "vv_agent.kernel._lease_scope.LeaseOwnership.poll": LeaseOwnership.poll,
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
        "scenario_inventory",
        "scenario_inventory_sha256",
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


@contextmanager
def _seed_record_parse_cache():
    """Validate one seed's immutable bodies without sharing mutable values."""
    from vv_agent.kernel._records import InboxItem, Record, _MessageCache, copy_json

    record_descriptor = Record.__dict__["_parse"]
    inbox_descriptor = InboxItem.__dict__["parse"]

    @lru_cache(maxsize=4096)
    def parse(cls: type[Record], body: bytes) -> Record:
        return record_descriptor.__get__(None, cls)(body)

    def cached_parse(cls: type[Record], body: bytes, messages: _MessageCache | None = None) -> Record:
        if type(body) is not bytes:
            return record_descriptor.__get__(None, cls)(body, messages)
        template = parse(cls, body)
        record = copy.copy(template)
        record._retain(template.encode(), copy_json(template._value))
        if messages is not None:
            record._share_messages(messages)
        return record

    @lru_cache(maxsize=4096)
    def parse_inbox(cls: type[InboxItem], body: bytes) -> InboxItem:
        return inbox_descriptor.__get__(None, cls)(body)

    def cached_inbox_parse(cls: type[InboxItem], body: bytes) -> InboxItem:
        if type(body) is not bytes:
            return inbox_descriptor.__get__(None, cls)(body)
        template = parse_inbox(cls, body)
        return InboxItem(
            input_id=template.input_id,
            kind=template.kind,
            payload=copy_json(template.payload),
            target_turn_id=template.target_turn_id,
            generation=template.generation,
            available_ms=template.available_ms,
            actor=copy_json(template.actor),
            admission_proof=copy_json(template.admission_proof),
            schema_version=template.schema_version,
            feature_level=template.feature_level,
            requirements=copy_json(template.requirements),
        )

    type.__setattr__(Record, "_parse", classmethod(cached_parse))
    type.__setattr__(InboxItem, "parse", classmethod(cached_inbox_parse))
    try:
        yield parse
    finally:
        type.__setattr__(Record, "_parse", record_descriptor)
        type.__setattr__(InboxItem, "parse", inbox_descriptor)
        parse.cache_clear()
        parse_inbox.cache_clear()


def _run_seed_on_database(job, db, *, reset_before_retry=None):
    """Run one seed against a caller-owned local database handle."""
    import tempfile

    from .scenarios import run_scenario
    from .world import InvariantViolation, SimWorld

    if multiprocessing.parent_process() is not None:
        # Fork inherits the parent's tempfile cache; workers must honor their current TMPDIR.
        tempfile.tempdir = None

    backend, seed, scenario, replay, budget = job
    world = None
    failure = None
    invariant = None
    storage = {}
    try:
        with _seed_record_parse_cache(), SimWorld(db, seed=seed, replay=replay) as world:
            storage = (
                {"sqlite_temporary_storage": temporary_storage(db.parent.parent)} if isinstance(db, Path) and db.is_file() else {}
            )
            try:
                summary = run_scenario(world, scenario)
                if replay is not None and world.choices.used != set(world.choices.overrides):
                    raise ValueError("replay has unused decisions; code or scenario differs")
            except Exception as exc:
                failure = world.summary() | storage | {"error": f"{type(exc).__name__}: {exc}"}
                invariant = exc.invariant if isinstance(exc, InvariantViolation) else None
                raise
        result = summary | storage
        if isinstance(db, _PostgresDatabase):
            result["postgres_cleanup"] = db.cleanup_evidence
        return result
    except Exception as exc:
        if failure is None:
            summary = world.summary() if world is not None else {"seed": seed, "scenario": scenario}
            failure = summary | storage | {"error": f"{type(exc).__name__}: {exc}"}
        if isinstance(db, _PostgresDatabase):
            if db.cleanup_error is not None:
                failure["cleanup_error"] = db.cleanup_error
            if db.cleanup_evidence is not None:
                failure["postgres_cleanup"] = db.cleanup_evidence
    failure["minimized_decisions"], failure["minimization_trials"] = failure["decisions"], 0
    if invariant and budget:
        reset_errors = []

        def reproduces(candidate):
            if reset_before_retry is not None:
                try:
                    reset_before_retry()
                except _PostgresResetError as exc:
                    reset_errors.append(exc)
                    return False
            try:
                if reset_before_retry is None:
                    with (
                        _seed_record_parse_cache(),
                        database(backend) as trial_db,
                        SimWorld(trial_db, seed=seed, replay=candidate) as trial,
                    ):
                        run_scenario(trial, failure["scenario"])
                else:
                    with _seed_record_parse_cache(), SimWorld(db, seed=seed, replay=candidate) as trial:
                        run_scenario(trial, failure["scenario"])
            except InvariantViolation as error:
                return error.invariant == invariant
            except Exception:
                return False
            return False

        failure["minimized_decisions"], failure["minimization_trials"] = minimize(failure["decisions"], reproduces, budget=budget)
        if reset_errors:
            error = reset_errors[0]
            failure["postgres_minimization_reset"] = error.evidence
            failure["cleanup_error"] = {
                "stage": "minimization_reset",
                "seed": seed,
                "type": type(error.__cause__ or error).__name__,
                "message": str(error.__cause__ or error),
            }
    return failure


def _run_seed(job):
    """Run one direct SQLite/PostgreSQL world; jobs never carry SQL handles."""
    import tempfile

    if multiprocessing.parent_process() is not None:
        # Fork inherits the parent's tempfile cache; workers must honor their current TMPDIR.
        tempfile.tempdir = None
    with database(job[0]) as db:
        result = _run_seed_on_database(job, db)
    if isinstance(db, _PostgresDatabase):
        result["postgres_cleanup"] = db.cleanup_evidence
    return result


class _PostgresResetError(RuntimeError):
    def __init__(self, error: Exception, evidence: dict[str, Any]) -> None:
        super().__init__(str(error))
        self.evidence = evidence


def _reset_postgres_batch_database(database_handle, *, seed, batch_id, purpose="seed"):
    try:
        return _reset_postgres_schema(database_handle, seed=seed, batch_id=batch_id, purpose=purpose)
    except Exception as exc:
        evidence = {
            "batch_id": batch_id,
            "worker_pid": os.getpid(),
            "seed": seed,
            "purpose": purpose,
            "strategy": "single_transaction_truncate_s1_drop_test_tables",
            "test_tables": {
                "strategy": "drop_if_exists",
                "names": [POSTGRES_TEST_RESET_TABLE],
                "cleanup_completed": False,
            },
            "reset_completed": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        raise _PostgresResetError(exc, evidence) from exc


def _add_postgres_cleanup_error(result, *, stage, error, seed, batch_id):
    entry = {
        "stage": stage,
        "seed": seed,
        "batch_id": batch_id,
        "type": type(error).__name__,
        "message": str(error),
    }
    if "cleanup_error" not in result:
        result["cleanup_error"] = entry
    else:
        result.setdefault("additional_cleanup_errors", []).append(entry)
    if "error" not in result:
        result["error"] = f"PostgreSQL {stage.replace('_', ' ')} failed: {type(error).__name__}: {error}"


def _run_postgres_batch(jobs):
    """Run a bounded seed batch against one process-local temporary database."""
    batch_id = uuid4().hex
    jobs = tuple(jobs)
    if not jobs or any(job[0] != "postgres" for job in jobs):
        raise ValueError("PostgreSQL batches require one or more postgres jobs")

    results: list[dict[str, Any]] = []
    resets: list[dict[str, Any]] = []
    database_handle = None
    lifecycle_error = None
    try:
        with database("postgres") as database_handle:
            for job in jobs:
                seed = job[1]

                def reset_trial(batch_seed=seed):
                    _reset_postgres_batch_database(
                        database_handle,
                        seed=batch_seed,
                        batch_id=batch_id,
                        purpose="minimization",
                    )

                try:
                    result = _run_seed_on_database(job, database_handle, reset_before_retry=reset_trial)
                except _PostgresResetError as exc:
                    result = {
                        "seed": seed,
                        "scenario": job[2],
                        "decisions": [],
                        "covered": [],
                        "postgres_reset": exc.evidence,
                        "error": f"PostgreSQL reset failed: {exc}",
                    }
                    resets.append(exc.evidence)
                    _add_postgres_cleanup_error(
                        result,
                        stage="reset",
                        error=exc.__cause__ or exc,
                        seed=seed,
                        batch_id=batch_id,
                    )
                    results.append(result)
                    break
                except Exception as exc:
                    result = {
                        "seed": seed,
                        "scenario": job[2],
                        "decisions": [],
                        "covered": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                try:
                    reset = _reset_postgres_batch_database(database_handle, seed=seed, batch_id=batch_id)
                except _PostgresResetError as exc:
                    result["postgres_reset"] = exc.evidence
                    resets.append(exc.evidence)
                    _add_postgres_cleanup_error(
                        result,
                        stage="reset",
                        error=exc.__cause__ or exc,
                        seed=seed,
                        batch_id=batch_id,
                    )
                else:
                    result["postgres_reset"] = reset
                    resets.append(reset)
                results.append(result)
                if "error" in result:
                    break
    except Exception as exc:
        lifecycle_error = exc

    if lifecycle_error is not None and not results:
        first = jobs[0]
        results.append(
            {
                "seed": first[1],
                "scenario": first[2],
                "decisions": [],
                "covered": [],
            }
        )
    completed_resets = sum(reset["reset_completed"] for reset in resets)
    cleanup_evidence = database_handle.cleanup_evidence if database_handle is not None else None
    cleanup = {
        "batch_id": batch_id,
        "worker_pid": os.getpid(),
        "batch_seeds": len(results),
        "database_creates": int(database_handle is not None),
        "database_drops": int(cleanup_evidence is not None and cleanup_evidence["drop_completed"]),
        "seeds_reset": completed_resets,
        "reset_completed": len(resets) == len(results),
    }
    if cleanup_evidence is not None:
        cleanup = cleanup_evidence | cleanup
    if lifecycle_error is not None:
        cleanup["drop_completed"] = False
        cleanup.setdefault("database", getattr(database_handle, "database_name", None))
        cleanup["error"] = {"type": type(lifecycle_error).__name__, "message": str(lifecycle_error)}
        _add_postgres_cleanup_error(
            results[-1],
            stage="database_create" if database_handle is None else "final_drop",
            error=lifecycle_error,
            seed=results[-1]["seed"],
            batch_id=batch_id,
        )

    for result in results:
        result["postgres_cleanup"] = copy.deepcopy(cleanup)
    return results


def _validate_worker_options(workers, minimize_budget):
    if not 1 <= workers <= MAX_WORKERS or minimize_budget < 0:
        raise ValueError(f"1..{MAX_WORKERS} workers and a nonnegative minimization budget required")


def _seed_results(
    *,
    backend="sqlite",
    seed=0,
    seeds=1,
    scenario=None,
    workers=1,
    replay=None,
    minimize_budget=0,
    start_method=None,
    mp_context=None,
):
    if seeds < 1:
        raise ValueError("positive seeds required")
    _validate_worker_options(workers, minimize_budget)
    if replay is not None and seeds != 1:
        raise ValueError("a replay runs exactly one seed")
    jobs = ((backend, number, scenario, replay, minimize_budget) for number in range(seed, seed + seeds))

    def batches():
        while True:
            batch = tuple(islice(jobs, POSTGRES_SEEDS_PER_BATCH))
            if not batch:
                return
            yield batch

    if workers == 1:
        if start_method not in (None, "serial"):
            raise ValueError("serial worker count requires the serial start method")
        if backend == "postgres":
            for batch in batches():
                yield from _run_postgres_batch(batch)
            return
        for job in jobs:
            yield _run_seed(job)
        return
    if mp_context is None:
        selected_method, mp_context = _worker_process_context(backend, workers)
        if start_method is not None and start_method != selected_method:
            raise ValueError("declared and selected worker start methods differ")
    selected_method = mp_context.get_start_method()
    if start_method is not None and start_method != selected_method:
        raise ValueError("declared and actual worker start methods differ")
    work = batches() if backend == "postgres" else ((job,) for job in jobs)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as pool:
        in_flight = deque()
        for _ in range(workers * 2):
            try:
                item = next(work)
            except StopIteration:
                break
            future = pool.submit(_run_postgres_batch, item) if backend == "postgres" else pool.submit(_run_seed, item[0])
            in_flight.append(future)
        while in_flight:
            result = in_flight.popleft().result()
            with suppress(StopIteration):
                item = next(work)
                future = pool.submit(_run_postgres_batch, item) if backend == "postgres" else pool.submit(_run_seed, item[0])
                in_flight.append(future)
            if backend == "postgres":
                yield from result
            else:
                yield result


def seed_results(*, backend="sqlite", seed=0, seeds=1, scenario=None, workers=1, replay=None, minimize_budget=0):
    """Bound submission AND result memory, including PostgreSQL connection slots."""
    _validate_worker_options(workers, minimize_budget)
    plan = simulation_worker_plan(backend, workers)
    plan, start_method, mp_context = _prepare_worker_execution(backend, plan)
    yield from _seed_results(
        backend=backend,
        seed=seed,
        seeds=seeds,
        scenario=scenario,
        workers=plan["effective_workers"],
        replay=replay,
        minimize_budget=minimize_budget,
        start_method=start_method,
        mp_context=mp_context,
    )


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
    _validate_worker_options(workers, minimize_budget)
    from .world import PENDING

    identity = versions() | {
        "workers": workers,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "cpu_affinity": AFFINITY,
            "sqlite_temporary_storage": {"verified": False, "observations": []},
            "postgres_worker_plan": None,
            "postgres_cleanup": None,
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
        raise ValueError("in-memory mutants require workers=1; workers execute the source on disk")
    if replay is not None:
        seed, seeds, backend, scenario = replay["seed"], 1, replay["backend"], replay["scenario"]
    worker_plan = simulation_worker_plan(backend, workers)
    worker_plan, worker_start_method, mp_context = _prepare_worker_execution(backend, worker_plan)
    identity["environment"]["postgres_worker_plan"] = worker_plan
    identity["environment"]["worker_start_method"] = worker_start_method
    if backend == "postgres":
        identity["environment"]["postgres_cleanup"] = {
            "verified": False,
            "seeds_observed": 0,
            "seeds_reset": 0,
            "batch_size": POSTGRES_SEEDS_PER_BATCH,
            "reset_strategy": "transactional_truncate_s1_drop_test_tables",
            "drop_strategy": "terminate_current_role_then_drop",
            "database_creates": 0,
            "database_drops": 0,
            "databases": [],
            "terminated_sessions": 0,
            "max_drop_attempts": 0,
            "cleanup_errors": [],
        }
    scenarios, decisions, covered = {}, 0, set()
    database_indexes = {}
    results = _seed_results(
        backend=backend,
        seed=seed,
        seeds=seeds,
        scenario=scenario,
        workers=worker_plan["effective_workers"],
        replay=replay["decisions"] if replay is not None else None,
        minimize_budget=minimize_budget,
        start_method=worker_start_method,
        mp_context=mp_context,
    )
    try:
        for result in results:
            storage = identity["environment"]["sqlite_temporary_storage"]
            observed = result.get("sqlite_temporary_storage")
            if observed is not None and observed not in storage["observations"]:
                storage["observations"].append(observed)
                storage["verified"] = all(item["filesystem"] is not None for item in storage["observations"])
            cleanup = identity["environment"]["postgres_cleanup"]
            if cleanup is not None:
                cleanup["seeds_observed"] += 1
                reset = result.get("postgres_reset")
                if reset is not None and reset.get("reset_completed") is True:
                    cleanup["seeds_reset"] += 1
                evidence = result.get("postgres_cleanup")
                if evidence is not None:
                    database = evidence.get("database") or evidence.get("batch_id") or f"unknown-{cleanup['seeds_observed']}"
                    if database not in database_indexes:
                        database_indexes[database] = len(cleanup["databases"])
                        cleanup["databases"].append(
                            {
                                "database": database,
                                "batch_id": evidence.get("batch_id"),
                                "worker_pid": evidence.get("worker_pid"),
                                "batch_seeds": evidence.get("batch_seeds"),
                                "seeds_reset": evidence.get("seeds_reset"),
                                "drop_attempts": evidence.get("drop_attempts", 0),
                                "drop_completed": evidence.get("drop_completed") is True,
                            }
                        )
                        cleanup["database_creates"] += int(evidence.get("database_creates", 1))
                        cleanup["database_drops"] += int(evidence.get("drop_completed") is True)
                        cleanup["terminated_sessions"] += int(evidence.get("terminated_sessions", 0))
                        cleanup["max_drop_attempts"] = max(cleanup["max_drop_attempts"], int(evidence.get("drop_attempts", 0)))
                    elif evidence.get("drop_completed") is True:
                        recorded = cleanup["databases"][database_indexes[database]]
                        if not recorded["drop_completed"]:
                            recorded["drop_completed"] = True
                            cleanup["database_drops"] += 1
                            cleanup["max_drop_attempts"] = max(
                                cleanup["max_drop_attempts"], int(evidence.get("drop_attempts", 0))
                            )
                    if evidence.get("error") is not None:
                        cleanup["cleanup_errors"].append({"database": database, "stage": "final_drop", **evidence["error"]})
                cleanup_error = result.get("cleanup_error")
                if cleanup_error is not None:
                    cleanup["cleanup_errors"].append(cleanup_error)
                cleanup["cleanup_errors"].extend(result.get("additional_cleanup_errors", []))
            if "error" in result:
                return result | identity | {"replay_provenance": provenance}
            scenarios[result["scenario"]] = scenarios.get(result["scenario"], 0) + 1
            decisions += len(result["decisions"])
            covered.update(result["covered"])
    finally:
        results.close()
    cleanup = identity["environment"]["postgres_cleanup"]
    if cleanup is not None:
        cleanup["verified"] = (
            cleanup["seeds_observed"] == seeds
            and cleanup["seeds_reset"] == seeds
            and cleanup["database_creates"] > 0
            and cleanup["database_creates"] == cleanup["database_drops"]
            and not cleanup["cleanup_errors"]
        )
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
