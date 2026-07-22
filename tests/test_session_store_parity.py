from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from vv_agent import MemorySession, Message, SQLiteSessionStore
from vv_agent.sessions.base import _deserialize_message, _serialize_message
from vv_agent.types import Role

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parity"
CODEC_FIXTURE = FIXTURE_DIR / "session_codec.json"
CANONICAL_SQL_FIXTURE = FIXTURE_DIR / "session_sqlite_canonical.sql"


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
        rows = [
            (int(row[0]), str(row[1]), str(row[2]))
            for row in connection.execute("SELECT item_index, session_id, payload FROM session_items ORDER BY item_index")
        ]
        return version, columns, rows
    finally:
        connection.close()


def test_session_codec_matches_canonical_contract() -> None:
    fixture = json.loads(CODEC_FIXTURE.read_text(encoding="utf-8"))

    for case in fixture["canonical_cases"]:
        raw = json.dumps(case["input"], ensure_ascii=False, separators=(",", ":"))
        message = _deserialize_message(raw)
        actual = json.loads(_serialize_message(message))
        assert actual == case["canonical"], case["name"]
        assert json.loads(_serialize_message(_deserialize_message(_serialize_message(message)))) == actual

    for case in fixture["invalid_cases"]:
        raw = json.dumps(case["input"], ensure_ascii=False, separators=(",", ":"))
        with pytest.raises(ValueError):
            _deserialize_message(raw)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"role": "user"}, 'missing required string field "content"'),
        ({"role": "user", "content": None}, 'field "content" must be a string'),
        ({"role": "user", "content": "x", "unknown": True}, "Message contains unknown fields"),
        ({"role": "user", "content": "x", "name": None}, 'field "name" must be a string'),
        ({"role": "assistant", "content": "", "tool_calls": None}, '"tool_calls" must be an array'),
        (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "name": "lookup", "arguments": {}}],
            },
            "ToolCall contains unknown fields",
        ),
        (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {}},
                    }
                ],
            },
            'field "arguments" must be a string',
        ),
        (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "not-json"},
                    }
                ],
            },
            '"arguments" must contain a JSON object',
        ),
        (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "[]"},
                    }
                ],
            },
            '"arguments" must contain a JSON object',
        ),
    ],
)
def test_session_codec_rejects_noncanonical_wire(payload: object, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _deserialize_message(json.dumps(payload))


def test_session_codec_accepts_only_openai_function_tool_calls() -> None:
    payload = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": ' {"z":2,"a":1} '},
                "extra_content": {"provider": "test"},
            }
        ],
    }

    assert json.loads(_serialize_message(_deserialize_message(json.dumps(payload)))) == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"a":1,"z":2}'},
                "extra_content": {"provider": "test"},
            }
        ],
    }


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


def test_sqlite_rejects_existing_schema_without_current_version(tmp_path: Path) -> None:
    db_path = tmp_path / "missing-version.sqlite3"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(CANONICAL_SQL_FIXTURE.read_text(encoding="utf-8"))
        connection.execute("PRAGMA user_version = 0")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="does not match required version"):
        SQLiteSessionStore(db_path)

    connection = sqlite3.connect(db_path)
    try:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 0
    finally:
        connection.close()


def test_sqlite_creates_only_the_current_schema_on_an_empty_database(tmp_path: Path) -> None:
    db_path = tmp_path / "new.sqlite3"
    store = SQLiteSessionStore(db_path)
    store.close()

    connection = sqlite3.connect(db_path)
    try:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 1
        assert [
            (str(row[0]), str(row[1]), str(row[2]))
            for row in connection.execute(
                "SELECT type, name, tbl_name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            )
        ] == [
            ("index", "idx_session_items_session_id_item_index", "session_items"),
            ("table", "session_commits", "session_commits"),
            ("table", "session_items", "session_items"),
        ]
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("name", "mutation", "error"),
    [
        (
            "missing-table",
            "DROP TABLE session_commits",
            "unsupported session schema objects",
        ),
        (
            "unexpected-object",
            "CREATE VIEW session_view AS SELECT session_id FROM session_items",
            "unsupported session schema objects",
        ),
        (
            "wrong-index",
            "DROP INDEX idx_session_items_session_id_item_index; "
            "CREATE INDEX idx_session_items_session_id_item_index ON session_items (item_index, session_id)",
            "unsupported idx_session_items_session_id_item_index schema columns",
        ),
    ],
)
def test_sqlite_rejects_noncanonical_schema_objects(
    tmp_path: Path,
    name: str,
    mutation: str,
    error: str,
) -> None:
    db_path = tmp_path / f"{name}.sqlite3"
    _seed_database(db_path, CANONICAL_SQL_FIXTURE)
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(mutation)
        connection.commit()
    finally:
        connection.close()

    before = db_path.read_bytes()
    with pytest.raises(RuntimeError, match=error):
        SQLiteSessionStore(db_path)
    assert db_path.read_bytes() == before


def test_sqlite_rejects_canonical_column_names_with_wrong_constraints(tmp_path: Path) -> None:
    db_path = tmp_path / "wrong-constraints.sqlite3"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            PRAGMA user_version = 1;
            CREATE TABLE session_items (
                session_id TEXT,
                item_index INTEGER PRIMARY KEY AUTOINCREMENT,
                payload TEXT NOT NULL
            );
            CREATE INDEX idx_session_items_session_id_item_index
                ON session_items (session_id, item_index);
            CREATE TABLE session_commits (
                session_id TEXT NOT NULL,
                commit_id TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                PRIMARY KEY (session_id, commit_id)
            );
            """
        )
        connection.commit()
    finally:
        connection.close()

    before = db_path.read_bytes()
    with pytest.raises(RuntimeError, match="unsupported session_items schema columns"):
        SQLiteSessionStore(db_path)
    assert db_path.read_bytes() == before


def test_sqlite_rejects_superseded_session_columns_without_migrating(tmp_path: Path) -> None:
    db_path = tmp_path / "superseded-columns.sqlite3"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            PRAGMA user_version = 1;
            CREATE TABLE session_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                item_json TEXT NOT NULL
            );
            CREATE INDEX idx_session_items_session_id_item_index
                ON session_items (session_id, id);
            CREATE TABLE session_commits (
                session_id TEXT NOT NULL,
                commit_id TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                PRIMARY KEY (session_id, commit_id)
            );
            INSERT INTO session_items (id, session_id, item_json)
                VALUES (4, 'shared', '{"type":"user","content":"stored"}');
            """
        )
        connection.commit()
    finally:
        connection.close()

    before = db_path.read_bytes()
    with pytest.raises(RuntimeError, match="unsupported session_items schema columns"):
        SQLiteSessionStore(db_path)
    assert db_path.read_bytes() == before


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

    with pytest.raises(RuntimeError, match="does not match required version"):
        SQLiteSessionStore(db_path)

    version, columns, rows = _schema_state(db_path)
    assert version == 2
    assert columns == ["session_id", "item_index", "payload"]
    assert len(rows) == 3
