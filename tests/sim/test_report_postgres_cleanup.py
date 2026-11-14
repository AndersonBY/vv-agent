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


def test_seed_success_result_contains_cleanup_evidence(monkeypatch):
    from . import scenarios, world

    handle = None

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

    assert result["postgres_cleanup"] == handle.cleanup_evidence
    assert result["postgres_cleanup"]["drop_completed"] is True


def test_campaign_success_records_cleanup_evidence(monkeypatch):
    def fake_seed_results(**_kwargs):
        for seed in range(2):
            yield {
                "seed": seed,
                "scenario": "cleanup",
                "decisions": [],
                "covered": [],
                "postgres_cleanup": {
                    "database": f"vvsk_sim_{seed}",
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
        "seeds_cleaned": 2,
        "terminated_sessions": 1,
        "max_drop_attempts": 2,
    }


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
