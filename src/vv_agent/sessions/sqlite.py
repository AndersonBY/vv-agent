from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import RLock

from vv_agent.sessions.base import (
    SessionCommitError,
    _deserialize_message,
    _serialize_message,
    validate_session_commit,
)
from vv_agent.types import Message

_SESSION_SCHEMA_VERSION = 1
_CANONICAL_SCHEMA_OBJECTS = [
    ("index", "idx_session_items_session_id_item_index", "session_items"),
    ("table", "session_commits", "session_commits"),
    ("table", "session_items", "session_items"),
]
_CANONICAL_TABLE_COLUMNS = {
    "session_items": [
        (0, "session_id", "TEXT", 1, None, 0, 0),
        (1, "item_index", "INTEGER", 0, None, 1, 0),
        (2, "payload", "TEXT", 1, None, 0, 0),
    ],
    "session_commits": [
        (0, "session_id", "TEXT", 1, None, 1, 0),
        (1, "commit_id", "TEXT", 1, None, 2, 0),
        (2, "payload_digest", "TEXT", 1, None, 0, 0),
    ],
}
_CANONICAL_INDEXES = {
    "session_items": [("idx_session_items_session_id_item_index", 0, "c", 0)],
    "session_commits": [("sqlite_autoindex_session_commits_1", 1, "pk", 0)],
}
_CANONICAL_INDEX_COLUMNS = {
    "idx_session_items_session_id_item_index": [
        (0, "session_id", 0, "BINARY", 1),
        (1, "item_index", 0, "BINARY", 1),
        (-1, None, 0, "BINARY", 0),
    ],
    "sqlite_autoindex_session_commits_1": [
        (0, "session_id", 0, "BINARY", 1),
        (1, "commit_id", 0, "BINARY", 1),
        (-1, None, 0, "BINARY", 0),
    ],
}
_CREATE_SESSION_ITEMS_TABLE = """
CREATE TABLE IF NOT EXISTS session_items (
    session_id TEXT NOT NULL,
    item_index INTEGER PRIMARY KEY AUTOINCREMENT,
    payload TEXT NOT NULL
)
"""
_CREATE_SESSION_ITEMS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_session_items_session_id_item_index
    ON session_items (session_id, item_index)
"""
_CREATE_SESSION_COMMITS_TABLE = """
CREATE TABLE IF NOT EXISTS session_commits (
    session_id TEXT NOT NULL,
    commit_id TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    PRIMARY KEY (session_id, commit_id)
)
"""


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA busy_timeout = 5000")


def _enable_wal(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode = WAL").fetchone()


def _initialize_session_schema(connection: sqlite3.Connection, lock: RLock) -> None:
    with lock:
        try:
            connection.execute("BEGIN IMMEDIATE")
            version = _read_schema_version(connection)
            schema_objects = _schema_objects(connection)
            if not schema_objects:
                if version != 0:
                    raise RuntimeError(f"empty session database has unexpected schema version {version}")
                connection.execute(_CREATE_SESSION_ITEMS_TABLE)
                connection.execute(_CREATE_SESSION_ITEMS_INDEX)
                connection.execute(_CREATE_SESSION_COMMITS_TABLE)
                connection.execute(f"PRAGMA user_version = {_SESSION_SCHEMA_VERSION}")
            else:
                if version != _SESSION_SCHEMA_VERSION:
                    raise RuntimeError(
                        f"session schema version {version} does not match required version {_SESSION_SCHEMA_VERSION}"
                    )
            _validate_session_schema(connection)
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise


def _read_schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    if row is None or len(row) != 1 or isinstance(row[0], bool) or not isinstance(row[0], int):
        raise RuntimeError("session schema marker is malformed")
    return row[0]


def _schema_objects(connection: sqlite3.Connection) -> list[tuple[str, str, str]]:
    return [
        (str(row[0]), str(row[1]), str(row[2]))
        for row in connection.execute(
            "SELECT type, name, tbl_name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        )
    ]


def _validate_session_schema(connection: sqlite3.Connection) -> None:
    version = _read_schema_version(connection)
    if version != _SESSION_SCHEMA_VERSION:
        raise RuntimeError(f"session schema version {version} does not match required version {_SESSION_SCHEMA_VERSION}")

    schema_objects = _schema_objects(connection)
    if schema_objects != _CANONICAL_SCHEMA_OBJECTS:
        raise RuntimeError(f"unsupported session schema objects: {schema_objects!r}")

    for table, expected_columns in _CANONICAL_TABLE_COLUMNS.items():
        columns = [tuple(row) for row in connection.execute(f"PRAGMA table_xinfo({table})")]
        if columns != expected_columns:
            raise RuntimeError(f"unsupported {table} schema columns: {columns!r}")

        table_list = connection.execute(f"PRAGMA table_list({table})").fetchall()
        table_properties = [(str(row[1]), str(row[2]), int(row[3]), int(row[4]), int(row[5])) for row in table_list]
        if table_properties != [(table, "table", len(expected_columns), 0, 0)]:
            raise RuntimeError(f"unsupported {table} schema properties: {table_properties!r}")

        if connection.execute(f"PRAGMA foreign_key_list({table})").fetchall():
            raise RuntimeError(f"unsupported {table} foreign keys")

        indexes = [
            (str(row[1]), int(row[2]), str(row[3]), int(row[4])) for row in connection.execute(f"PRAGMA index_list({table})")
        ]
        if indexes != _CANONICAL_INDEXES[table]:
            raise RuntimeError(f"unsupported {table} indexes: {indexes!r}")

    for index, expected_columns in _CANONICAL_INDEX_COLUMNS.items():
        columns = [
            (int(row[1]), None if row[2] is None else str(row[2]), int(row[3]), str(row[4]), int(row[5]))
            for row in connection.execute(f"PRAGMA index_xinfo({index})")
        ]
        if columns != expected_columns:
            raise RuntimeError(f"unsupported {index} schema columns: {columns!r}")

    session_items_sql_row = connection.execute(
        "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = 'session_items'"
    ).fetchone()
    session_items_sql = "" if session_items_sql_row is None else str(session_items_sql_row[0])
    if "AUTOINCREMENT" not in session_items_sql.upper():
        raise RuntimeError("session_items.item_index must use AUTOINCREMENT")


class SQLiteSession:
    def __init__(
        self,
        session_id: str,
        db_path: str | Path = "agent_sessions.sqlite3",
        *,
        _connection: sqlite3.Connection | None = None,
        _lock: RLock | None = None,
    ) -> None:
        self.session_id = session_id
        self._db_path = str(db_path)
        self._conn = _connection if _connection is not None else sqlite3.connect(self._db_path, check_same_thread=False)
        self._lock = _lock if _lock is not None else RLock()
        self._owns_connection = _connection is None
        if self._owns_connection:
            try:
                _configure_connection(self._conn)
                _initialize_session_schema(self._conn, self._lock)
                _enable_wal(self._conn)
            except BaseException:
                self._conn.close()
                raise

    def get_items(self, limit: int | None = None) -> list[Message]:
        with self._lock:
            if limit is None:
                cursor = self._conn.execute(
                    "SELECT payload FROM session_items WHERE session_id = ? ORDER BY item_index ASC",
                    (self.session_id,),
                )
            else:
                cursor = self._conn.execute(
                    """
                    SELECT payload FROM (
                        SELECT item_index, payload FROM session_items
                        WHERE session_id = ?
                        ORDER BY item_index DESC
                        LIMIT ?
                    )
                    ORDER BY item_index ASC
                    """,
                    (self.session_id, max(int(limit), 0)),
                )
            rows = cursor.fetchall()
        return [_deserialize_message(row[0]) for row in rows]

    def add_items(self, items: list[Message]) -> None:
        payloads = [(self.session_id, _serialize_message(item)) for item in items]
        if not payloads:
            return
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO session_items (session_id, payload) VALUES (?, ?)",
                payloads,
            )

    def add_items_once(
        self,
        commit_id: str,
        payload_digest: str,
        items: list[Message],
    ) -> str:
        normalized = validate_session_commit(commit_id, payload_digest, items)
        payloads = [(self.session_id, _serialize_message(item)) for item in normalized]
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                existing = self._conn.execute(
                    "SELECT payload_digest FROM session_commits WHERE session_id = ? AND commit_id = ?",
                    (self.session_id, commit_id),
                ).fetchone()
                if existing is not None:
                    if str(existing[0]) != payload_digest:
                        raise SessionCommitError(
                            "session commit id already has a different payload",
                            code="session_commit_identity_conflict",
                        )
                    self._conn.commit()
                    return "replayed"
                if payloads:
                    self._conn.executemany(
                        "INSERT INTO session_items (session_id, payload) VALUES (?, ?)",
                        payloads,
                    )
                self._conn.execute(
                    "INSERT INTO session_commits (session_id, commit_id, payload_digest) VALUES (?, ?, ?)",
                    (self.session_id, commit_id, payload_digest),
                )
                self._conn.commit()
                return "committed"
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise

    def pop_item(self) -> Message | None:
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT item_index, payload FROM session_items WHERE session_id = ? ORDER BY item_index DESC LIMIT 1",
                    (self.session_id,),
                ).fetchone()
                if row is None:
                    self._conn.commit()
                    return None
                message = _deserialize_message(row[1])
                self._conn.execute("DELETE FROM session_items WHERE item_index = ?", (row[0],))
                self._conn.commit()
                return message
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM session_items WHERE session_id = ?", (self.session_id,))
            self._conn.execute("DELETE FROM session_commits WHERE session_id = ?", (self.session_id,))

    def clear_session(self) -> None:
        self.clear()

    def close(self) -> None:
        if self._owns_connection:
            with self._lock:
                self._conn.close()


class SQLiteSessionStore:
    def __init__(self, db_path: str | Path = "agent_sessions.sqlite3") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._lock = RLock()
        try:
            _configure_connection(self._conn)
            _initialize_session_schema(self._conn, self._lock)
            _enable_wal(self._conn)
        except BaseException:
            self._conn.close()
            raise

    def session(self, session_id: str) -> SQLiteSession:
        return SQLiteSession(
            session_id,
            self._db_path,
            _connection=self._conn,
            _lock=self._lock,
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
