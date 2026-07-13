from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from vv_agent import MemorySession, Message, SQLiteSessionStore
from vv_agent.sessions.base import _deserialize_message, _serialize_message
from vv_agent.types import Role

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parity"
CODEC_FIXTURE = FIXTURE_DIR / "session_codec_v1.json"
LEGACY_SQL_FIXTURE = FIXTURE_DIR / "session_sqlite_rust_legacy_v0.sql"
CANONICAL_SQL_FIXTURE = FIXTURE_DIR / "session_sqlite_canonical_v1.sql"
PYTHON_UNVERSIONED_SQL_FIXTURE = FIXTURE_DIR / "session_sqlite_python_unversioned_v0.sql"
INVALID_LEGACY_SQL_FIXTURE = FIXTURE_DIR / "session_sqlite_invalid_legacy_v0.sql"

FIXTURE_SHA256 = {
    "session_codec_v1.json": "ddb771fd89827145557297d8bfc6d734684fadf9ce019ed87a9b38b884782eb8",
    "session_sqlite_rust_legacy_v0.sql": "1cfa0fa6550cb7ddf6b6029cfb63fdb007287cb4289bd481e86d28942fcbbed5",
    "session_sqlite_canonical_v1.sql": "03e1dbf36e2299cf8f7d0b9d4e85ec685a7e10c46bcdc2cef4ee8b87d0d5d18d",
    "session_sqlite_python_unversioned_v0.sql": "215ae69f2fb44b18a0a7e2473d11ab0d86cb93552791b00350de0a056d7dc956",
    "session_sqlite_invalid_legacy_v0.sql": "37e8b0662fc346d3c424a83d213357ee54545c8b66a9d2df65ca7ec100845aa6",
}


def _seed_database(path: Path, fixture: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(fixture.read_text(encoding="utf-8"))
        connection.commit()
    finally:
        connection.close()


def _schema_state(path: Path) -> tuple[int, list[str], list[tuple[int, str, str]]]:
    connection = sqlite3.connect(path)
    try:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        columns = [str(row[1]) for row in connection.execute("PRAGMA table_info(session_items)")]
        if columns == ["session_id", "item_index", "payload"]:
            rows = [
                (int(row[0]), str(row[1]), str(row[2]))
                for row in connection.execute(
                    "SELECT item_index, session_id, payload FROM session_items ORDER BY item_index"
                )
            ]
        else:
            rows = [
                (int(row[0]), str(row[1]), str(row[2]))
                for row in connection.execute(
                    "SELECT id, session_id, item_json FROM session_items ORDER BY id"
                )
            ]
        return version, columns, rows
    finally:
        connection.close()


def test_shared_session_parity_fixtures_have_stable_hashes() -> None:
    for name, expected in FIXTURE_SHA256.items():
        assert hashlib.sha256((FIXTURE_DIR / name).read_bytes()).hexdigest() == expected


def test_session_codec_matches_canonical_and_legacy_contract() -> None:
    fixture = json.loads(CODEC_FIXTURE.read_text(encoding="utf-8"))

    for case in [*fixture["canonical_cases"], *fixture["legacy_cases"]]:
        raw = json.dumps(case["input"], ensure_ascii=False, separators=(",", ":"))
        message = _deserialize_message(raw)
        actual = json.loads(_serialize_message(message))
        assert actual == case["canonical"], case["name"]
        assert json.loads(_serialize_message(_deserialize_message(_serialize_message(message)))) == actual

    for case in fixture["invalid_cases"]:
        raw = json.dumps(case["input"], ensure_ascii=False, separators=(",", ":"))
        with pytest.raises(ValueError):
            _deserialize_message(raw)


def test_memory_session_uses_validated_snapshots_and_atomic_batches() -> None:
    session = MemorySession("memory-parity")
    original = Message(
        role="user",
        content="snapshot",
        metadata={"nested": {"value": 1}},
    )
    session.add_items([original])
    original.metadata["nested"]["value"] = 2
    snapshot = session.get_items()
    snapshot[0].metadata["nested"]["value"] = 3

    assert session.get_items()[0].metadata == {"nested": {"value": 1}}

    invalid = Message(role=cast(Role, "developer"), content="invalid")
    with pytest.raises(ValueError, match="unknown message role"):
        session.add_items([Message(role="user", content="must not persist"), invalid])
    assert [item.content for item in session.get_items()] == ["snapshot"]


def test_sqlite_migrates_rust_legacy_schema_and_payloads_transactionally(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    _seed_database(db_path, LEGACY_SQL_FIXTURE)

    store = SQLiteSessionStore(db_path)
    shared = store.session("shared")
    assert [json.loads(_serialize_message(item)) for item in shared.get_items()] == [
        {"role": "user", "content": "legacy user"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_legacy",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"a":{"x":1,"y":2},"z":1}',
                    },
                }
            ],
        },
    ]
    assert [item.content for item in store.session("other").get_items()] == ["other session"]
    shared.add_items([Message(role="tool", content="done", tool_call_id="call_legacy")])
    store.close()

    version, columns, rows = _schema_state(db_path)
    assert version == 1
    assert columns == ["session_id", "item_index", "payload"]
    assert [row[:2] for row in rows] == [(2, "shared"), (5, "other"), (8, "shared"), (9, "shared")]
    assert all(set(json.loads(payload)).isdisjoint({"type", "message"}) for _, _, payload in rows)


def test_sqlite_opens_canonical_schema_written_by_either_runtime(tmp_path: Path) -> None:
    db_path = tmp_path / "canonical.sqlite3"
    _seed_database(db_path, CANONICAL_SQL_FIXTURE)

    store = SQLiteSessionStore(db_path)
    shared = store.session("shared")
    assert [item.content for item in shared.get_items()] == ["canonical user", ""]
    assert shared.get_items()[1].tool_calls == [
        {
            "id": "call_canonical",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"a":1,"z":2}'},
        }
    ]
    shared.add_items([Message(role="tool", content="canonical result", tool_call_id="call_canonical")])
    store.close()

    version, columns, rows = _schema_state(db_path)
    assert version == 1
    assert columns == ["session_id", "item_index", "payload"]
    assert rows[-1][0] == 10
    assert json.loads(rows[-1][2]) == {
        "role": "tool",
        "content": "canonical result",
        "tool_call_id": "call_canonical",
    }


def test_sqlite_upgrades_unversioned_python_schema_in_place(tmp_path: Path) -> None:
    db_path = tmp_path / "python-unversioned.sqlite3"
    _seed_database(db_path, PYTHON_UNVERSIONED_SQL_FIXTURE)

    store = SQLiteSessionStore(db_path)
    assert [item.content for item in store.session("shared").get_items()] == ["python unversioned"]
    store.close()

    version, columns, rows = _schema_state(db_path)
    assert version == 1
    assert columns == ["session_id", "item_index", "payload"]
    assert rows == [(4, "shared", '{"role":"user","content":"python unversioned"}')]

    connection = sqlite3.connect(db_path)
    try:
        indexes = {str(row[1]) for row in connection.execute("PRAGMA index_list(session_items)")}
    finally:
        connection.close()
    assert "idx_session_items_session_id_item_index" in indexes


def test_sqlite_failed_legacy_migration_rolls_back_schema_and_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "invalid-legacy.sqlite3"
    _seed_database(db_path, INVALID_LEGACY_SQL_FIXTURE)

    with pytest.raises(ValueError, match="unknown session item type"):
        SQLiteSessionStore(db_path)

    version, columns, rows = _schema_state(db_path)
    assert version == 0
    assert columns == ["id", "session_id", "item_json"]
    assert [row[0] for row in rows] == [1, 2]


def test_sqlite_batch_validation_and_corrupt_pop_are_atomic(tmp_path: Path) -> None:
    db_path = tmp_path / "atomic.sqlite3"
    store = SQLiteSessionStore(db_path)
    session = store.session("shared")
    invalid = Message(role=cast(Role, "developer"), content="invalid")

    with pytest.raises(ValueError, match="unknown message role"):
        session.add_items([Message(role="user", content="must not persist"), invalid])
    assert session.get_items() == []
    store.close()

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "INSERT INTO session_items (session_id, item_index, payload) VALUES (?, ?, ?)",
            ("shared", 20, '{"role":"developer","content":"corrupt"}'),
        )
        connection.commit()
    finally:
        connection.close()

    store = SQLiteSessionStore(db_path)
    with pytest.raises(ValueError, match="unknown message role"):
        store.session("shared").pop_item()
    store.close()

    _, _, rows = _schema_state(db_path)
    assert [row[0] for row in rows] == [20]


def test_sqlite_rejects_newer_schema_without_mutating_it(tmp_path: Path) -> None:
    db_path = tmp_path / "newer.sqlite3"
    _seed_database(db_path, CANONICAL_SQL_FIXTURE)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA user_version = 2")
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="newer than supported"):
        SQLiteSessionStore(db_path)

    version, columns, rows = _schema_state(db_path)
    assert version == 2
    assert columns == ["session_id", "item_index", "payload"]
    assert len(rows) == 3
