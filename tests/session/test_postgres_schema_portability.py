"""PostgreSQL catalog fingerprint portability across supported server versions."""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent.kernel import _postgres

FIXTURE_DIR = Path(__file__).with_name("fixtures")
PG16_CATALOG_SHA256 = "29e391fb004563a15d3c26b91f5d1eef18c3aa587d1504edc09e52ee387b051a"
PG16_RAW_FINGERPRINT = "b6975c70ad23fabb6dab4f56480ca2eb3c63cd9209561c5e8c209112fa04a7a7"
PG18_CATALOG_SHA256 = "6c7b47899cfba255e2f107505d36224f8c47ae61d4d53359f52176273491eb23"
PG18_RAW_FINGERPRINT = "66e62f44aebe17b8c6c93391e795c0e324a2765947e49d5832bc23cb96580a7e"


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
    assert len(added) == 92
    assert all(row[2] == "n" and row[6] for row in added)
    attnotnull_columns = {(row[0], row[2]) for row in pg18["columns"] if row[4]}
    assert {(row[0], row[3].removeprefix("NOT NULL ")) for row in added} == attnotnull_columns


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
