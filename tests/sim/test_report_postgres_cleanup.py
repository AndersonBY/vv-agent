import os
from contextlib import contextmanager
from types import SimpleNamespace

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from . import report


def query_text(query):
    return query if isinstance(query, str) else query.as_string(None)


class FakeAdmin:
    def __init__(self, *, termination_results=(), object_in_use_count=0, always_object_in_use=False):
        self.termination_results = termination_results
        self.remaining_object_in_use = object_in_use_count
        self.always_object_in_use = always_object_in_use
        self.termination_error = None
        self.drop_error = None
        self.queries = []
        self.terminate_calls = []
        self.drop_calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def cursor(self):
        return FakeCursor(self)


class FakeCursor:
    def __init__(self, admin):
        self.admin = admin
        self.rows = ()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def execute(self, query, params=None):
        text = query_text(query)
        self.admin.queries.append((text, params))
        if text == "SELECT current_user":
            self.rows = (("vvsk_temporary_role",),)
        elif "pg_terminate_backend" in text:
            self.admin.terminate_calls.append(params)
            if self.admin.termination_error is not None:
                raise self.admin.termination_error
            self.rows = self.admin.termination_results
        elif text.startswith(("CREATE DATABASE", "DROP DATABASE")):
            if text.startswith("DROP DATABASE"):
                self.admin.drop_calls.append(text)
                if self.admin.drop_error is not None:
                    raise self.admin.drop_error
                if self.admin.always_object_in_use:
                    raise psycopg.errors.ObjectInUse("database is being accessed by other users")
                if self.admin.remaining_object_in_use:
                    self.admin.remaining_object_in_use -= 1
                    raise psycopg.errors.ObjectInUse("database is being accessed by other users")
            self.rows = ()
        else:
            raise AssertionError(f"unexpected query: {text}")

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class FakeResetTransaction:
    def __init__(self):
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1

    def __exit__(self, *args):
        self.exited += 1


class FakeResetConnection:
    def __init__(self, *, provider_table_exists=True):
        self.transaction_state = FakeResetTransaction()
        self.queries = []
        self.closed = False
        self.provider_table_exists = provider_table_exists
        self.drop_test_tables = []

    def __enter__(self):
        self.closed = False
        return self

    def __exit__(self, *args):
        self.closed = True

    def transaction(self):
        return self.transaction_state

    def cursor(self):
        return FakeResetCursor(self)


class FakeResetCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = ()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def execute(self, query, params=None):
        text = query_text(query)
        self.connection.queries.append((text, params))
        if text.startswith("DROP TABLE IF EXISTS"):
            self.connection.drop_test_tables.append(text)
            self.connection.provider_table_exists = False
            self.rows = ()
        elif text.startswith("TRUNCATE TABLE"):
            self.rows = ()
        elif text.startswith("SELECT current_schema()"):
            self.rows = (("public", self.connection.provider_table_exists, 123456),)
        elif text.startswith("SELECT to_regclass"):
            self.rows = ((not self.connection.provider_table_exists, 9),)
        else:
            raise AssertionError(f"unexpected query: {text}")

    def fetchone(self):
        return self.rows[0] if self.rows else None


def test_cleanup_terminates_only_current_roles_sessions_then_plain_drop():
    name = "vvsk_sim_success"
    admin = FakeAdmin(termination_results=((True,), (False,)))

    evidence = report._cleanup_postgres_database(admin, name)

    assert admin.terminate_calls == [(name, "vvsk_temporary_role")]
    assert evidence == {
        "database": name,
        "role": "vvsk_temporary_role",
        "same_role_sessions": 2,
        "terminated_sessions": 1,
        "drop_attempts": 1,
        "drop_completed": True,
    }
    terminate_query = next(text for text, _params in admin.queries if "pg_terminate_backend" in text)
    assert "datname = %s" in terminate_query
    assert "usename = %s" in terminate_query
    assert "pid <> pg_backend_pid()" in terminate_query
    assert admin.drop_calls == [f'DROP DATABASE "{name}"']
    assert "FORCE" not in admin.drop_calls[0]


def test_cleanup_does_not_swallow_permission_denied_for_another_role():
    admin = FakeAdmin()
    admin.termination_error = psycopg.errors.InsufficientPrivilege("permission denied to terminate process")

    with pytest.raises(psycopg.errors.InsufficientPrivilege, match="permission denied to terminate process"):
        report._cleanup_postgres_database(admin, "vvsk_sim_denied")

    assert admin.terminate_calls == [("vvsk_sim_denied", "vvsk_temporary_role")]
    assert admin.drop_calls == []


def test_cleanup_retries_object_in_use_with_bounded_backoff(monkeypatch):
    admin = FakeAdmin(object_in_use_count=2)
    sleeps = []
    monkeypatch.setattr(report.time, "sleep", sleeps.append)

    evidence = report._cleanup_postgres_database(admin, "vvsk_sim_retry")

    assert evidence["drop_attempts"] == 3
    assert len(admin.terminate_calls) == 3
    assert len(admin.drop_calls) == 3
    assert sleeps == list(report.POSTGRES_CLEANUP_RETRY_DELAYS[:2])


def test_cleanup_object_in_use_retry_budget_is_exhausted(monkeypatch):
    admin = FakeAdmin(always_object_in_use=True)
    sleeps = []
    monkeypatch.setattr(report.time, "sleep", sleeps.append)

    with pytest.raises(psycopg.errors.ObjectInUse, match="being accessed"):
        report._cleanup_postgres_database(admin, "vvsk_sim_stuck")

    assert len(admin.terminate_calls) == len(admin.drop_calls) == len(report.POSTGRES_CLEANUP_RETRY_DELAYS) + 1
    assert sleeps == list(report.POSTGRES_CLEANUP_RETRY_DELAYS)


def test_database_preserves_primary_and_cleanup_errors(monkeypatch):
    admin = FakeAdmin()
    admin.drop_error = psycopg.errors.InsufficientPrivilege("permission denied to drop database")
    monkeypatch.setenv("VV_AGENT_TEST_POSTGRES_DSN", "dbname=postgres user=vvsk_temporary_role")
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: admin)

    with pytest.raises(RuntimeError, match="simulation failed") as caught, report.database("postgres") as database_handle:
        raise RuntimeError("simulation failed")

    assert database_handle.cleanup_error == {
        "type": "InsufficientPrivilege",
        "message": "permission denied to drop database",
    }
    assert any(
        "PostgreSQL cleanup also failed: InsufficientPrivilege: permission denied to drop database" in note
        for note in caught.value.__notes__
    )


def test_database_exposes_successful_cleanup_evidence(monkeypatch):
    admin = FakeAdmin()
    monkeypatch.setenv("VV_AGENT_TEST_POSTGRES_DSN", "dbname=postgres user=vvsk_temporary_role")
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: admin)

    with report.database("postgres") as database_handle:
        assert database_handle.cleanup_evidence is None

    assert database_handle.cleanup_error is None
    assert database_handle.cleanup_evidence["role"] == "vvsk_temporary_role"
    assert database_handle.cleanup_evidence["drop_completed"] is True


def test_reset_truncates_all_s1_tables_in_one_transaction_without_cascade(monkeypatch):
    connection = FakeResetConnection()
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: connection)

    evidence = report._reset_postgres_schema(
        report._PostgresDatabase("dbname=temporary"),
        seed=17,
        batch_id="batch-id",
    )

    truncate = next(text for text, _params in connection.queries if text.startswith("TRUNCATE TABLE"))
    assert "sk_session" in truncate
    assert "sk_content_link" in truncate
    assert "CASCADE" not in truncate
    assert "DELETE FROM" not in truncate
    drop = next(text for text, _params in connection.queries if text.startswith("DROP TABLE IF EXISTS"))
    assert "sim_effect" in drop
    assert "IF EXISTS" in drop
    assert connection.transaction_state.entered == connection.transaction_state.exited == 1
    assert connection.closed is True
    assert evidence["seed"] == 17
    assert evidence["schema"] == "public"
    assert evidence["foreign_key_constraints"] == 9
    assert evidence["test_tables"] == {
        "strategy": "drop_if_exists",
        "names": ["sim_effect"],
        "existed_before": True,
        "dropped": ["sim_effect"],
        "exists_after": False,
        "cleanup_completed": True,
    }
    assert evidence["reset_completed"] is True


def test_reset_allows_provider_table_to_be_recreated_for_the_next_seed(monkeypatch):
    connection = FakeResetConnection(provider_table_exists=True)
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: connection)
    database_handle = report._PostgresDatabase("dbname=temporary")

    first = report._reset_postgres_schema(database_handle, seed=0, batch_id="batch-id")
    connection.provider_table_exists = True
    second = report._reset_postgres_schema(database_handle, seed=1, batch_id="batch-id")

    assert first["test_tables"]["existed_before"] is True
    assert second["test_tables"]["existed_before"] is True
    assert first["test_tables"]["dropped"] == second["test_tables"]["dropped"] == ["sim_effect"]
    assert first["test_tables"]["exists_after"] is second["test_tables"]["exists_after"] is False
    assert len(connection.drop_test_tables) == 2


def test_seed_success_result_contains_cleanup_evidence(monkeypatch):
    from . import scenarios, world

    handle: report._PostgresDatabase | None = None

    @contextmanager
    def fake_database(_backend):
        nonlocal handle
        handle = report._PostgresDatabase("dbname=temporary")
        yield handle
        handle.cleanup_evidence = {
            "database": "vvsk_sim_seed",
            "role": "vvsk_temporary_role",
            "same_role_sessions": 0,
            "terminated_sessions": 0,
            "drop_attempts": 1,
            "drop_completed": True,
        }

    class FakeWorld:
        def __init__(self, _database, *, seed, replay):
            self.choices = SimpleNamespace(used=set(), overrides=set())
            self.summary_value = {"seed": seed, "scenario": "cleanup", "decisions": [], "covered": []}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def summary(self):
            return self.summary_value

    monkeypatch.setattr(report, "database", fake_database)
    monkeypatch.setattr(world, "SimWorld", FakeWorld)
    monkeypatch.setattr(world, "InvariantViolation", RuntimeError)
    monkeypatch.setattr(scenarios, "run_scenario", lambda _world, _scenario: {"seed": 0, "scenario": "cleanup"})

    result = report._run_seed(("postgres", 0, "cleanup", None, 0))

    assert handle is not None
    assert result["postgres_cleanup"] == handle.cleanup_evidence
    assert result["postgres_cleanup"]["drop_completed"] is True


def test_postgres_seed_batch_creates_once_resets_each_seed_and_drops_once(monkeypatch):
    lifecycle = {"creates": 0, "drops": 0}

    @contextmanager
    def fake_database(_backend):
        handle = report._PostgresDatabase("dbname=temporary")
        handle.database_name = "vvsk_sim_batch"
        lifecycle["creates"] += 1
        yield handle
        lifecycle["drops"] += 1
        handle.cleanup_evidence = {
            "database": handle.database_name,
            "role": "vvsk_temporary_role",
            "same_role_sessions": 0,
            "terminated_sessions": 0,
            "drop_attempts": 1,
            "drop_completed": True,
        }

    reset_seeds = []

    def fake_reset(_database, *, seed, batch_id, purpose="seed"):
        reset_seeds.append((seed, purpose, batch_id))
        return {
            "batch_id": batch_id,
            "seed": seed,
            "purpose": purpose,
            "reset_completed": True,
        }

    def fake_run(job, _database, *, reset_before_retry=None):
        assert reset_before_retry is not None
        return {"seed": job[1], "scenario": "cleanup", "decisions": [], "covered": []}

    monkeypatch.setattr(report, "POSTGRES_SEEDS_PER_BATCH", 3)
    monkeypatch.setattr(report, "database", fake_database)
    monkeypatch.setattr(report, "_reset_postgres_schema", fake_reset)
    monkeypatch.setattr(report, "_run_seed_on_database", fake_run)

    results = list(report._seed_results(backend="postgres", seed=10, seeds=3, workers=1, start_method="serial"))

    assert lifecycle == {"creates": 1, "drops": 1}
    assert [(seed, purpose) for seed, purpose, _batch_id in reset_seeds] == [
        (10, "seed"),
        (11, "seed"),
        (12, "seed"),
    ]
    assert len({result["postgres_reset"]["batch_id"] for result in results}) == 1
    assert all(result["postgres_cleanup"]["database_creates"] == 1 for result in results)
    assert all(result["postgres_cleanup"]["database_drops"] == 1 for result in results)


def test_postgres_seed_batch_reset_failure_is_retained_and_database_is_dropped(monkeypatch):
    lifecycle = {"creates": 0, "drops": 0}

    @contextmanager
    def fake_database(_backend):
        handle = report._PostgresDatabase("dbname=temporary")
        handle.database_name = "vvsk_sim_reset_failure"
        lifecycle["creates"] += 1
        yield handle
        lifecycle["drops"] += 1
        handle.cleanup_evidence = {
            "database": handle.database_name,
            "role": "vvsk_temporary_role",
            "same_role_sessions": 0,
            "terminated_sessions": 0,
            "drop_attempts": 1,
            "drop_completed": True,
        }

    reset_seeds = []

    def fake_reset(_database, *, seed, batch_id, purpose="seed"):
        reset_seeds.append(seed)
        if purpose == "seed" and seed == 1:
            raise RuntimeError("reset failed")
        return {
            "batch_id": batch_id,
            "seed": seed,
            "purpose": purpose,
            "reset_completed": True,
        }

    monkeypatch.setattr(report, "POSTGRES_SEEDS_PER_BATCH", 3)
    monkeypatch.setattr(report, "database", fake_database)
    monkeypatch.setattr(report, "_reset_postgres_schema", fake_reset)
    monkeypatch.setattr(
        report,
        "_run_seed_on_database",
        lambda job, _database, *, reset_before_retry: {
            "seed": job[1],
            "scenario": "cleanup",
            "decisions": [],
            "covered": [],
        },
    )

    results = list(report._seed_results(backend="postgres", seed=0, seeds=3, workers=1, start_method="serial"))

    assert reset_seeds == [0, 1]
    assert lifecycle == {"creates": 1, "drops": 1}
    assert [result["seed"] for result in results] == [0, 1]
    assert results[1]["postgres_reset"]["reset_completed"] is False
    assert results[1]["cleanup_error"]["stage"] == "reset"
    assert results[1]["error"] == "PostgreSQL reset failed: RuntimeError: reset failed"
    assert all(result["postgres_cleanup"]["drop_completed"] is True for result in results)


def test_postgres_seed_batch_final_drop_failure_has_explicit_evidence(monkeypatch):
    lifecycle = {"creates": 0}

    @contextmanager
    def fake_database(_backend):
        handle = report._PostgresDatabase("dbname=temporary")
        handle.database_name = "vvsk_sim_drop_failure"
        lifecycle["creates"] += 1
        yield handle
        handle.cleanup_error = {
            "type": "InsufficientPrivilege",
            "message": "permission denied to drop database",
        }
        raise psycopg.errors.InsufficientPrivilege("permission denied to drop database")

    monkeypatch.setattr(report, "POSTGRES_SEEDS_PER_BATCH", 2)
    monkeypatch.setattr(report, "database", fake_database)
    monkeypatch.setattr(
        report,
        "_reset_postgres_schema",
        lambda _database, *, seed, batch_id, purpose="seed": {
            "batch_id": batch_id,
            "seed": seed,
            "purpose": purpose,
            "reset_completed": True,
        },
    )
    monkeypatch.setattr(
        report,
        "_run_seed_on_database",
        lambda job, _database, *, reset_before_retry: {
            "seed": job[1],
            "scenario": "cleanup",
            "decisions": [],
            "covered": [],
        },
    )

    result = next(report._seed_results(backend="postgres", seed=0, seeds=1, workers=1, start_method="serial"))

    assert lifecycle == {"creates": 1}
    assert result["cleanup_error"]["stage"] == "final_drop"
    assert result["postgres_cleanup"]["database"] == "vvsk_sim_drop_failure"
    assert result["postgres_cleanup"]["database_creates"] == 1
    assert result["postgres_cleanup"]["database_drops"] == 0
    assert result["postgres_cleanup"]["drop_completed"] is False
    assert result["postgres_cleanup"]["error"]["type"] == "InsufficientPrivilege"


def test_campaign_success_records_cleanup_evidence(monkeypatch):
    def fake_seed_results(**_kwargs):
        for seed in range(2):
            yield {
                "seed": seed,
                "scenario": "cleanup",
                "decisions": [],
                "covered": [],
                "postgres_reset": {
                    "seed": seed,
                    "purpose": "seed",
                    "reset_completed": True,
                },
                "postgres_cleanup": {
                    "database": f"vvsk_sim_{seed}",
                    "batch_id": f"batch-{seed}",
                    "worker_pid": 100 + seed,
                    "batch_seeds": 1,
                    "seeds_reset": 1,
                    "database_creates": 1,
                    "database_drops": 1,
                    "role": "vvsk_temporary_role",
                    "same_role_sessions": seed,
                    "terminated_sessions": seed,
                    "drop_attempts": seed + 1,
                    "drop_completed": True,
                },
            }

    monkeypatch.setattr(report, "versions", lambda: {"in_memory_mutations": []})
    monkeypatch.setattr(report, "simulation_worker_plan", lambda _backend, workers: {"effective_workers": workers})
    monkeypatch.setattr(report, "_seed_results", fake_seed_results)

    result = report.run_campaign(backend="postgres", seeds=2, workers=1)

    assert result["environment"]["postgres_cleanup"] == {
        "verified": True,
        "seeds_observed": 2,
        "seeds_reset": 2,
        "batch_size": report.POSTGRES_SEEDS_PER_BATCH,
        "reset_strategy": "transactional_truncate_s1_drop_test_tables",
        "drop_strategy": "terminate_current_role_then_drop",
        "database_creates": 2,
        "database_drops": 2,
        "databases": [
            {
                "database": "vvsk_sim_0",
                "batch_id": "batch-0",
                "worker_pid": 100,
                "batch_seeds": 1,
                "seeds_reset": 1,
                "drop_attempts": 1,
                "drop_completed": True,
            },
            {
                "database": "vvsk_sim_1",
                "batch_id": "batch-1",
                "worker_pid": 101,
                "batch_seeds": 1,
                "seeds_reset": 1,
                "drop_attempts": 2,
                "drop_completed": True,
            },
        ],
        "terminated_sessions": 1,
        "max_drop_attempts": 2,
        "cleanup_errors": [],
    }


@pytest.mark.postgres
def test_real_postgres_tools_batch_recreates_provider_table_for_each_seed():
    dsn = os.environ.get("VV_AGENT_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set a local VV_AGENT_TEST_POSTGRES_DSN for a real PostgreSQL batch")
    host = conninfo_to_dict(dsn).get("host")
    if host not in {"localhost", "127.0.0.1", "::1"}:
        pytest.skip("real PostgreSQL batch coverage only permits a loopback TCP DSN")

    results = list(
        report._seed_results(
            backend="postgres",
            seed=0,
            seeds=2,
            scenario="tools",
            workers=1,
            start_method="serial",
        )
    )

    assert [result["seed"] for result in results] == [0, 1]
    assert all("error" not in result for result in results), results
    for result in results:
        assert result["postgres_reset"]["test_tables"] == {
            "strategy": "drop_if_exists",
            "names": ["sim_effect"],
            "existed_before": True,
            "dropped": ["sim_effect"],
            "exists_after": False,
            "cleanup_completed": True,
        }
        assert result["postgres_cleanup"]["drop_completed"] is True


@pytest.mark.postgres
def test_local_postgres_cleanup_terminates_same_role_tcp_session():
    dsn = os.environ.get("VV_AGENT_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set a local VV_AGENT_TEST_POSTGRES_DSN for real PostgreSQL cleanup coverage")
    host = conninfo_to_dict(dsn).get("host")
    if host not in {"localhost", "127.0.0.1", "::1"}:
        pytest.skip("real cleanup coverage only permits a loopback TCP PostgreSQL DSN")

    concurrent = None
    try:
        with report.database("postgres") as database_handle:
            concurrent = psycopg.connect(str(database_handle), autocommit=True, connect_timeout=3)
            concurrent.execute("SELECT 1").fetchone()

        evidence = database_handle.cleanup_evidence
        assert evidence["same_role_sessions"] == 1
        assert evidence["terminated_sessions"] == 1
        assert evidence["drop_completed"] is True
    finally:
        if concurrent is not None:
            concurrent.close()
