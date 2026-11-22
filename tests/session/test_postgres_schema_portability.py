"""PostgreSQL catalog fingerprint portability across supported server versions."""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

from vv_agent.kernel import _postgres

FIXTURE_DIR = Path(__file__).with_name("fixtures")
PG16_CATALOG_SHA256 = "a01be8fe955c70ed8d93cdab2bcb39a5dfa21b38eef1d9a4ba83e1baa05a322a"
PG16_RAW_FINGERPRINT = "7d1c35061ca2456ab4c689df21faadf0c8a46c764e585a8418503c6995f33066"
PG18_CATALOG_SHA256 = "90b693353a12cde0520688bb9d8021e4afbfbf683939503669c849ae727364b5"
PG18_RAW_FINGERPRINT = "2ddbf9c21971c767318b891334a75fda774bfdc6bc13d3e6c0e5c4c28b95e09f"


def _fold_factory():
    raise AssertionError("schema tests do not execute a fold")


def _load_catalog(filename: str) -> dict[str, list[list[Any]]]:
    fixture = json.loads((FIXTURE_DIR / filename).read_text(encoding="utf-8"))
    assert set(fixture) == {"catalog"}
    catalog = fixture["catalog"]
    assert isinstance(catalog, dict)
    return cast(dict[str, list[list[Any]]], catalog)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _raw_fingerprint(manifest: dict[str, list[list[object]]]) -> str:
    material = {"version": "S1-postgres-schema-v1", "catalog": manifest}
    return sha256(_canonical_json(material)).hexdigest()


def test_real_pg16_and_pg18_catalog_evidence_has_one_semantic_difference():
    pg16 = _load_catalog("pg16_catalog_manifest.json")
    pg18 = _load_catalog("pg18_catalog_manifest.json")

    assert sha256(_canonical_json(pg16)).hexdigest() == PG16_CATALOG_SHA256
    assert sha256(_canonical_json(pg18)).hexdigest() == PG18_CATALOG_SHA256
    assert _raw_fingerprint(pg16) == PG16_RAW_FINGERPRINT
    assert _raw_fingerprint(pg18) == PG18_RAW_FINGERPRINT

    for section in ("tables", "columns", "indexes", "triggers"):
        assert pg16[section] == pg18[section]

    pg16_constraints = set(map(tuple, pg16["constraints"]))
    pg18_constraints = set(map(tuple, pg18["constraints"]))
    added = pg18_constraints - pg16_constraints
    assert not pg16_constraints - pg18_constraints
    assert len(added) == 93
    assert all(row[2] == "n" and row[6] for row in added)
    attnotnull_columns = {(row[0], row[2]) for row in pg18["columns"] if row[4]}
    assert {(row[0], row[3].removeprefix("NOT NULL ")) for row in added} == attnotnull_columns


def test_postgres_store_keeps_runtime_binding_checks(monkeypatch):
    calls: list[tuple[str, object]] = []
    worker_connection = object()
    tenant_connection = object()

    def schema_status(connection):
        calls.append(("schema", connection))
        return "missing"

    def verified_role_binding(connection, binding):
        calls.append(("role", binding))
        return binding

    def verify_bound_policy(_connection, binding):
        calls.append(("policy", binding))

    def verify_tenant_connection(_cls, connection, tenant_id):
        calls.append(("tenant", (connection, tenant_id)))

    monkeypatch.setattr(_postgres, "_schema_status", schema_status)
    monkeypatch.setattr(_postgres, "_verified_role_binding", verified_role_binding)
    monkeypatch.setattr(_postgres.PostgresStore, "_verify_bound_policy", staticmethod(verify_bound_policy))
    monkeypatch.setattr(
        _postgres.PostgresStore,
        "verify_tenant_connection",
        classmethod(verify_tenant_connection),
    )

    worker_binding = _postgres.RuntimeBinding("worker")
    worker = _postgres.PostgresStore(cast(Any, worker_connection), fold_factory=_fold_factory, runtime_binding=worker_binding)
    tenant_binding = _postgres.RuntimeBinding("tenant", "tenant-a")
    tenant = _postgres.PostgresStore(cast(Any, tenant_connection), fold_factory=_fold_factory, runtime_binding=tenant_binding)

    assert calls == [
        ("role", worker_binding),
        ("policy", worker_binding),
        ("schema", worker_connection),
        ("tenant", (tenant_connection, "tenant-a")),
        ("schema", tenant_connection),
    ]
    assert worker.runtime_binding == worker_binding
    assert tenant.runtime_binding == tenant_binding
    assert tenant.credential_tenant_id == "tenant-a"


@pytest.mark.postgres
def test_postgres_fingerprint_accepts_both_captures_and_current_schema(monkeypatch, database):
    import psycopg

    pg16 = _load_catalog("pg16_catalog_manifest.json")
    pg18 = _load_catalog("pg18_catalog_manifest.json")
    with psycopg.connect(database, autocommit=True, connect_timeout=3) as connection:
        for manifest in (pg16, pg18):
            captured = deepcopy(manifest)
            monkeypatch.setattr(
                _postgres,
                "_catalog_manifest",
                lambda _connection, manifest=captured: manifest,
            )
            assert _postgres._schema_fingerprint(connection) == _postgres.SCHEMA_FINGERPRINT

        monkeypatch.undo()
        _postgres.PostgresStore(connection, fold_factory=_fold_factory).install_schema()
        assert _postgres._schema_fingerprint(connection) == _postgres.SCHEMA_FINGERPRINT


@pytest.mark.postgres
def test_postgres_store_reuses_relation_catalog_scan(monkeypatch, database):
    import psycopg

    with psycopg.connect(database, autocommit=True, connect_timeout=3) as connection:
        _postgres.PostgresStore(connection, fold_factory=_fold_factory).install_schema()

    with psycopg.connect(database, autocommit=True, connect_timeout=3) as connection:
        relation_scans: list[str] = []
        original_probe_rows = _postgres._probe_rows

        def counting_probe_rows(probe_connection, query, params=(), *, binary=False):
            if "SELECT c.relname,c.relkind,c.relpersistence" in query:
                relation_scans.append(query)
            return original_probe_rows(probe_connection, query, params, binary=binary)

        monkeypatch.setattr(_postgres, "_probe_rows", counting_probe_rows)
        store = _postgres.PostgresStore(connection, fold_factory=_fold_factory)

        assert store.runtime_mode == "unbound-internal"
        assert len(relation_scans) == 1


@pytest.mark.postgres
def test_postgres_schema_validation_is_scoped_to_each_database_and_connection(database, postgres_admin_dsn):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    database_name = f"vv_schema_scope_{uuid4().hex}"
    with psycopg.connect(postgres_admin_dsn, autocommit=True, connect_timeout=3) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
    other_dsn = make_conninfo(postgres_admin_dsn, dbname=database_name)

    try:
        with (
            psycopg.connect(database, autocommit=True, connect_timeout=3) as first_database,
            psycopg.connect(other_dsn, autocommit=True, connect_timeout=3) as second_database,
        ):
            _postgres.PostgresStore(first_database, fold_factory=_fold_factory).install_schema()
            _postgres.PostgresStore(second_database, fold_factory=_fold_factory).install_schema()

        with (
            psycopg.connect(database, autocommit=True, connect_timeout=3) as first_connection,
            psycopg.connect(other_dsn, autocommit=True, connect_timeout=3) as second_connection,
        ):
            _postgres.PostgresStore(first_connection, fold_factory=_fold_factory)
            _postgres.PostgresStore(second_connection, fold_factory=_fold_factory)

        with psycopg.connect(other_dsn, autocommit=True, connect_timeout=3) as drifted:
            drifted.execute("CREATE INDEX sk_extra_record_created ON sk_record(created_ms)")

        with (
            psycopg.connect(other_dsn, autocommit=True, connect_timeout=3) as drifted,
            pytest.raises(ValueError, match="unsupported PostgreSQL session schema version"),
        ):
            _postgres.PostgresStore(drifted, fold_factory=_fold_factory)

        with psycopg.connect(database, autocommit=True, connect_timeout=3) as still_current:
            _postgres.PostgresStore(still_current, fold_factory=_fold_factory)
    finally:
        with psycopg.connect(postgres_admin_dsn, autocommit=True, connect_timeout=3) as admin:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(database_name)))


@pytest.mark.postgres
@pytest.mark.parametrize(
    "drift",
    [
        ("ALTER TABLE sk_record ALTER COLUMN turn_id SET NOT NULL",),
        ("ALTER TABLE sk_record ALTER COLUMN kind DROP NOT NULL",),
        (
            "ALTER TABLE sk_settings DROP CONSTRAINT sk_settings_write_level_check",
            "ALTER TABLE sk_settings ADD CONSTRAINT sk_settings_write_level_check CHECK (write_level >= 0)",
        ),
        ("ALTER TABLE sk_session ADD CONSTRAINT sk_extra_positive CHECK (level > 0)",),
        ("CREATE INDEX sk_extra_record_created ON sk_record(created_ms)",),
    ],
)
def test_postgres_fingerprint_rejects_semantic_catalog_drift(database, drift):
    import psycopg

    with psycopg.connect(database, autocommit=True, connect_timeout=3) as connection:
        _postgres.PostgresStore(connection, fold_factory=_fold_factory).install_schema()
        for statement in drift:
            connection.execute(statement)

        assert _postgres._schema_fingerprint(connection) != _postgres.SCHEMA_FINGERPRINT
        with pytest.raises(ValueError, match="unsupported PostgreSQL session schema version"):
            _postgres.PostgresStore(connection, fold_factory=_fold_factory)


@pytest.mark.postgres
def test_postgres_18_not_valid_constraint_is_not_normalized(database):
    import psycopg

    with psycopg.connect(database, autocommit=True, connect_timeout=3) as connection:
        version = connection.execute("SHOW server_version_num").fetchone()
        assert version is not None
        version_num = version[0]
        if int(version_num) < 180000:
            pytest.skip("table-level NOT NULL constraints require PostgreSQL 18")
        _postgres.PostgresStore(connection, fold_factory=_fold_factory).install_schema()
        connection.execute("ALTER TABLE sk_record ADD CONSTRAINT sk_record_turn_id_not_null NOT NULL turn_id NOT VALID")

        raw = _postgres._catalog_manifest(connection)
        constraint = next(row for row in raw["constraints"] if row[1] == "sk_record_turn_id_not_null")
        assert constraint[2] == "n"
        assert constraint[6] is False
        assert _postgres._schema_fingerprint(connection) != _postgres.SCHEMA_FINGERPRINT
