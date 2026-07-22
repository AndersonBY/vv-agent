from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vv_agent.app_server.protocol import ThreadItem

_THREAD_STORE_SCHEMA_VERSION = 1
_TABLE_COLUMNS = {
    "threads": (
        ("id", "INTEGER", 0, None, 1),
        ("thread_id", "TEXT", 1, None, 0),
        ("agent_key", "TEXT", 1, None, 0),
        ("cwd", "TEXT", 0, None, 0),
        ("created_at", "REAL", 1, None, 0),
        ("updated_at", "REAL", 1, None, 0),
        ("archived_at", "REAL", 0, None, 0),
        ("status", "TEXT", 1, "'idle'", 0),
        ("active_turn_id", "TEXT", 0, None, 0),
        ("metadata_json", "TEXT", 1, None, 0),
    ),
    "turns": (
        ("id", "INTEGER", 0, None, 1),
        ("turn_id", "TEXT", 1, None, 0),
        ("thread_id", "TEXT", 1, None, 0),
        ("run_id", "TEXT", 0, None, 0),
        ("status", "TEXT", 1, None, 0),
        ("started_at", "REAL", 1, None, 0),
        ("completed_at", "REAL", 0, None, 0),
        ("input_json", "TEXT", 1, None, 0),
        ("result_json", "TEXT", 1, None, 0),
    ),
    "items": (
        ("id", "INTEGER", 0, None, 1),
        ("item_id", "TEXT", 1, None, 0),
        ("thread_id", "TEXT", 1, None, 0),
        ("turn_id", "TEXT", 1, None, 0),
        ("run_event_id", "TEXT", 0, None, 0),
        ("type", "TEXT", 1, None, 0),
        ("status", "TEXT", 1, None, 0),
        ("payload_json", "TEXT", 1, None, 0),
        ("created_at", "REAL", 1, None, 0),
        ("updated_at", "REAL", 1, None, 0),
    ),
}


@dataclass(frozen=True, slots=True)
class ThreadRecord:
    thread_id: str
    agent_key: str
    cwd: str | None = None
    created_at: float = 0
    updated_at: float = 0
    archived_at: float | None = None
    status: str = "idle"
    active_turn_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TurnRecord:
    turn_id: str
    thread_id: str
    run_id: str | None = None
    status: str = "running"
    started_at: float = 0
    completed_at: float | None = None
    input: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ThreadSnapshot:
    thread: ThreadRecord
    turns: list[TurnRecord]
    items: list[ThreadItem]


class ThreadStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = ":memory:" if db_path is None else str(db_path)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        try:
            self._initialize()
        except BaseException:
            self._connection.close()
            raise

    def create_thread(self, *, agent_key: str, cwd: str | None = None, metadata: dict[str, Any] | None = None) -> ThreadRecord:
        with self._lock:
            thread_id = self._next_id("threads", "thread_id", "thread")
            now = time.time()
            record = ThreadRecord(
                thread_id=thread_id,
                agent_key=agent_key,
                cwd=cwd,
                created_at=now,
                updated_at=now,
                metadata=dict(metadata or {}),
            )
            self._connection.execute(
                """
                INSERT INTO threads (
                    thread_id, agent_key, cwd, created_at, updated_at, archived_at,
                    status, active_turn_id, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.thread_id,
                    record.agent_key,
                    record.cwd,
                    record.created_at,
                    record.updated_at,
                    record.archived_at,
                    record.status,
                    record.active_turn_id,
                    json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                ),
            )
            self._connection.commit()
            return record

    def create_turn(
        self,
        *,
        thread_id: str,
        input: list[dict[str, Any]],
        run_id: str | None = None,
        status: str = "running",
    ) -> TurnRecord:
        with self._lock:
            self._require_thread(thread_id)
            turn_id = self._next_id("turns", "turn_id", "turn")
            now = time.time()
            record = TurnRecord(
                turn_id=turn_id,
                thread_id=thread_id,
                run_id=run_id,
                status=status,
                started_at=now,
                input=[dict(item) for item in input],
            )
            self._connection.execute(
                """
                INSERT INTO turns (turn_id, thread_id, run_id, status, started_at, completed_at, input_json, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.turn_id,
                    record.thread_id,
                    record.run_id,
                    record.status,
                    record.started_at,
                    record.completed_at,
                    json.dumps(record.input, ensure_ascii=False, sort_keys=True),
                    json.dumps(record.result, ensure_ascii=False, sort_keys=True),
                ),
            )
            self._touch_thread(thread_id, now)
            self._connection.commit()
            return record

    def append_item(self, item: ThreadItem, *, run_event_id: str | None = None) -> bool:
        with self._lock:
            self._require_thread(item.thread_id)
            if run_event_id is not None and self._event_projection_matches(run_event_id, item):
                return False
            try:
                self._connection.execute(
                    """
                    INSERT INTO items (
                        item_id,
                        thread_id,
                        turn_id,
                        run_event_id,
                        type,
                        status,
                        payload_json,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.item_id,
                        item.thread_id,
                        item.turn_id,
                        run_event_id,
                        item.item_type,
                        item.status,
                        json.dumps(item.payload, ensure_ascii=False, sort_keys=True),
                        item.created_at,
                        item.updated_at,
                    ),
                )
            except sqlite3.IntegrityError:
                self._connection.rollback()
                if run_event_id is not None and self._event_projection_matches(run_event_id, item):
                    return False
                raise
            self._touch_thread(item.thread_id, time.time())
            self._connection.commit()
            return True

    def update_turn(
        self,
        turn_id: str,
        *,
        status: str,
        run_id: str | None = None,
        completed_at: float | None = None,
        result: dict[str, Any] | None = None,
    ) -> TurnRecord:
        with self._lock:
            existing = self._fetch_turn(turn_id)
            completed = time.time() if completed_at is None else completed_at
            merged_result = dict(result or {})
            self._connection.execute(
                """
                UPDATE turns
                SET run_id = ?, status = ?, completed_at = ?, result_json = ?
                WHERE turn_id = ?
                """,
                (
                    run_id if run_id is not None else existing.run_id,
                    status,
                    completed,
                    json.dumps(merged_result, ensure_ascii=False, sort_keys=True),
                    turn_id,
                ),
            )
            self._touch_thread(existing.thread_id, completed)
            self._connection.commit()
            return self._fetch_turn(turn_id)

    def resume_turn(self, turn_id: str, *, run_id: str) -> TurnRecord:
        with self._lock:
            existing = self._fetch_turn(turn_id)
            now = time.time()
            self._connection.execute(
                """
                UPDATE turns
                SET run_id = ?, status = 'running', completed_at = NULL, result_json = '{}'
                WHERE turn_id = ?
                """,
                (run_id, turn_id),
            )
            self._touch_thread(existing.thread_id, now)
            self._connection.commit()
            return self._fetch_turn(turn_id)

    def read_thread(self, thread_id: str) -> ThreadSnapshot:
        with self._lock:
            thread = self._fetch_thread(thread_id)
            turn_rows = self._connection.execute("SELECT * FROM turns WHERE thread_id = ? ORDER BY id", (thread_id,))
            item_rows = self._connection.execute("SELECT * FROM items WHERE thread_id = ? ORDER BY id", (thread_id,))
            turns = [self._turn_from_row(row) for row in turn_rows]
            items = [self._item_from_row(row) for row in item_rows]
            return ThreadSnapshot(thread=thread, turns=turns, items=items)

    def list_threads(self, *, include_archived: bool = False) -> list[ThreadRecord]:
        with self._lock:
            where = "" if include_archived else "WHERE archived_at IS NULL"
            rows = self._connection.execute(f"SELECT * FROM threads {where} ORDER BY id")
            return [self._thread_from_row(row) for row in rows]

    def archive_thread(self, thread_id: str) -> None:
        with self._lock:
            self._require_thread(thread_id)
            now = time.time()
            self._connection.execute(
                """
                UPDATE threads
                SET archived_at = ?, status = 'archived', active_turn_id = NULL, updated_at = ?
                WHERE thread_id = ?
                """,
                (now, now, thread_id),
            )
            self._connection.commit()

    def set_active_turn(self, thread_id: str, active_turn_id: str | None, status: str) -> None:
        with self._lock:
            self._require_thread(thread_id)
            now = time.time()
            self._connection.execute(
                "UPDATE threads SET active_turn_id = ?, status = ?, updated_at = ? WHERE thread_id = ?",
                (active_turn_id, status, now, thread_id),
            )
            self._connection.commit()

    def _initialize(self) -> None:
        with self._lock:
            version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            existing_tables = {
                str(row[0])
                for row in self._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if not existing_tables:
                if version != 0:
                    raise RuntimeError(
                        f"App Server thread-store schema version {version} does not match required version "
                        f"{_THREAD_STORE_SCHEMA_VERSION}"
                    )
                self._create_schema()
            elif version != _THREAD_STORE_SCHEMA_VERSION:
                raise RuntimeError(
                    f"App Server thread-store schema version {version} does not match required version "
                    f"{_THREAD_STORE_SCHEMA_VERSION}"
                )

            self._validate_schema()
            self._connection.execute("UPDATE threads SET status = 'idle', active_turn_id = NULL WHERE status = 'running'")
            self._connection.commit()

    def _create_schema(self) -> None:
        self._connection.executescript(
            f"""
                PRAGMA user_version = {_THREAD_STORE_SCHEMA_VERSION};

                CREATE TABLE IF NOT EXISTS threads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL UNIQUE,
                    agent_key TEXT NOT NULL,
                    cwd TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    archived_at REAL,
                    status TEXT NOT NULL DEFAULT 'idle',
                    active_turn_id TEXT,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_id TEXT NOT NULL UNIQUE,
                    thread_id TEXT NOT NULL,
                    run_id TEXT,
                    status TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    completed_at REAL,
                    input_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL UNIQUE,
                    thread_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    run_event_id TEXT,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS items_run_event_id_unique
                ON items(run_event_id)
                WHERE run_event_id IS NOT NULL;
            """
        )

    def _validate_schema(self) -> None:
        objects = {
            (str(row[0]), str(row[1]))
            for row in self._connection.execute(
                "SELECT type, name FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' AND type IN ('table', 'index', 'view', 'trigger')"
            )
        }
        expected_objects = {("table", table) for table in _TABLE_COLUMNS} | {("index", "items_run_event_id_unique")}
        if objects != expected_objects:
            raise RuntimeError(
                "App Server thread-store schema objects do not match the current schema: "
                f"expected={sorted(expected_objects)}, actual={sorted(objects)}"
            )

        for table, expected in _TABLE_COLUMNS.items():
            actual = tuple(
                (str(row["name"]), str(row["type"]), int(row["notnull"]), row["dflt_value"], int(row["pk"]))
                for row in self._connection.execute(f"PRAGMA table_info({table})")
            )
            if actual != expected:
                raise RuntimeError(
                    f"App Server thread-store table {table} does not match the current schema: "
                    f"expected={expected}, actual={actual}"
                )

        expected_unique_indexes = {
            "threads": {("thread_id", False)},
            "turns": {("turn_id", False)},
            "items": {("item_id", False), ("run_event_id", True)},
        }
        for table, expected in expected_unique_indexes.items():
            actual: set[tuple[str, bool]] = set()
            for row in self._connection.execute(f"PRAGMA index_list({table})"):
                if not bool(row["unique"]):
                    continue
                columns = tuple(str(column["name"]) for column in self._connection.execute(f"PRAGMA index_info({row['name']})"))
                if len(columns) == 1:
                    actual.add((columns[0], bool(row["partial"])))
            if actual != expected:
                raise RuntimeError(
                    f"App Server thread-store unique indexes for {table} do not match the current schema: "
                    f"expected={sorted(expected)}, actual={sorted(actual)}"
                )

    def _next_id(self, table: str, column: str, prefix: str) -> str:
        row = self._connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        count = int(row["count"]) if row is not None else 0
        return f"{prefix}_{count + 1}"

    def _touch_thread(self, thread_id: str, updated_at: float) -> None:
        self._connection.execute("UPDATE threads SET updated_at = ? WHERE thread_id = ?", (updated_at, thread_id))

    def _require_thread(self, thread_id: str) -> None:
        self._fetch_thread(thread_id)

    def _event_projection_matches(self, run_event_id: str, item: ThreadItem) -> bool:
        row = self._connection.execute(
            "SELECT * FROM items WHERE run_event_id = ?",
            (run_event_id,),
        ).fetchone()
        if row is None:
            return False
        if self._item_from_row(row) != item:
            raise sqlite3.IntegrityError(f"run event {run_event_id!r} has a conflicting App Server item projection")
        return True

    def _fetch_thread(self, thread_id: str) -> ThreadRecord:
        row = self._connection.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown App Server thread: {thread_id}")
        return self._thread_from_row(row)

    def _fetch_turn(self, turn_id: str) -> TurnRecord:
        row = self._connection.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown App Server turn: {turn_id}")
        return self._turn_from_row(row)

    def _thread_from_row(self, row: sqlite3.Row) -> ThreadRecord:
        return ThreadRecord(
            thread_id=str(row["thread_id"]),
            agent_key=str(row["agent_key"]),
            cwd=row["cwd"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            archived_at=None if row["archived_at"] is None else float(row["archived_at"]),
            status=str(row["status"]),
            active_turn_id=None if row["active_turn_id"] is None else str(row["active_turn_id"]),
            metadata=json.loads(str(row["metadata_json"])),
        )

    def _turn_from_row(self, row: sqlite3.Row) -> TurnRecord:
        return TurnRecord(
            turn_id=str(row["turn_id"]),
            thread_id=str(row["thread_id"]),
            run_id=row["run_id"],
            status=str(row["status"]),
            started_at=float(row["started_at"]),
            completed_at=None if row["completed_at"] is None else float(row["completed_at"]),
            input=json.loads(str(row["input_json"])),
            result=json.loads(str(row["result_json"])),
        )

    def _item_from_row(self, row: sqlite3.Row) -> ThreadItem:
        return ThreadItem(
            item_id=str(row["item_id"]),
            thread_id=str(row["thread_id"]),
            turn_id=str(row["turn_id"]),
            item_type=str(row["type"]),
            status=str(row["status"]),
            payload=json.loads(str(row["payload_json"])),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )
