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
_CANONICAL_COLUMNS = ["session_id", "item_index", "payload"]
_RUST_LEGACY_COLUMNS = ["id", "session_id", "item_json"]
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
    connection.execute("PRAGMA journal_mode = WAL").fetchone()


def _initialize_session_schema(connection: sqlite3.Connection, lock: RLock) -> None:
    with lock:
        try:
            connection.execute("BEGIN IMMEDIATE")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > _SESSION_SCHEMA_VERSION:
                raise RuntimeError(
                    f"session schema version {version} is newer than supported version "
                    f"{_SESSION_SCHEMA_VERSION}"
                )

            table_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'session_items'"
            ).fetchone()
            if table_exists is None:
                connection.execute(_CREATE_SESSION_ITEMS_TABLE)
            else:
                columns = [
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(session_items)").fetchall()
                ]
                if columns == _RUST_LEGACY_COLUMNS:
                    _migrate_rust_legacy_schema(connection)
                elif columns != _CANONICAL_COLUMNS:
                    raise RuntimeError(f"unsupported session_items schema columns: {columns!r}")

            connection.execute(_CREATE_SESSION_ITEMS_INDEX)
            connection.execute(_CREATE_SESSION_COMMITS_TABLE)
            connection.execute(f"PRAGMA user_version = {_SESSION_SCHEMA_VERSION}")
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise


def _migrate_rust_legacy_schema(connection: sqlite3.Connection) -> None:
    legacy_table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'session_items_legacy_v0'"
    ).fetchone()
    if legacy_table_exists is not None:
        raise RuntimeError("cannot migrate session_items while session_items_legacy_v0 exists")

    legacy_rows = connection.execute(
        "SELECT id, session_id, item_json FROM session_items ORDER BY id ASC"
    ).fetchall()
    canonical_rows = [
        (int(item_index), str(session_id), _serialize_message(_deserialize_message(item_json)))
        for item_index, session_id, item_json in legacy_rows
    ]

    connection.execute("ALTER TABLE session_items RENAME TO session_items_legacy_v0")
    connection.execute(_CREATE_SESSION_ITEMS_TABLE)
    connection.executemany(
        "INSERT INTO session_items (item_index, session_id, payload) VALUES (?, ?, ?)",
        canonical_rows,
    )
    connection.execute("DROP TABLE session_items_legacy_v0")


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
                    "SELECT payload_digest FROM session_commits "
                    "WHERE session_id = ? AND commit_id = ?",
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
                    "INSERT INTO session_commits (session_id, commit_id, payload_digest) "
                    "VALUES (?, ?, ?)",
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
                    "SELECT item_index, payload FROM session_items "
                    "WHERE session_id = ? ORDER BY item_index DESC LIMIT 1",
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
