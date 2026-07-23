from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any

from vv_agent.checkpoint import EventCursor
from vv_agent.runtime.checkpoint_codec import (
    _strict_json_loads,
    checkpoint_from_dict,
    checkpoint_from_json,
    checkpoint_to_dict,
    checkpoint_to_json,
)
from vv_agent.runtime.state import (
    Checkpoint,
    CheckpointConflictError,
    ClaimMode,
    _LeaseOperationClock,
    _validate_claim,
    _validate_renew,
    check_claim,
    prepare_claimed_terminal,
    prepare_event_delivery,
    validate_model_journal_accounting,
)
from vv_agent.types import AgentStatus


class SqliteCheckpointStore:
    """Current checkpoint store backed by SQLite."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        raw_path = str(db_path)
        self._db_path = raw_path if raw_path == ":memory:" else str(Path(raw_path).resolve())
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._lock = RLock()
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()

    def _create_table(self) -> None:
        existing_table = self._schema_sql("table", "checkpoints")
        if existing_table is None:
            self._conn.execute(_CREATE_TABLE_SQL)
            self._conn.execute(_CREATE_INDEX_SQL)
            self._conn.commit()
            return

        if _normalize_schema_sql(existing_table) != _normalize_schema_sql(_CREATE_TABLE_SQL):
            raise RuntimeError("existing checkpoints table does not match the current schema; create a new database")
        existing_index = self._schema_sql("index", "checkpoints_status_idx")
        if existing_index is None or _normalize_schema_sql(existing_index) != _normalize_schema_sql(_CREATE_INDEX_SQL):
            raise RuntimeError("existing checkpoints index does not match the current schema; create a new database")
        self._conn.commit()

    def _schema_sql(self, object_type: str, name: str) -> str | None:
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
            (object_type, name),
        ).fetchone()
        return str(row[0]) if row is not None and row[0] is not None else None

    def create_checkpoint(self, checkpoint: Checkpoint) -> bool:
        snapshot = checkpoint_from_json(checkpoint_to_json(checkpoint))
        if snapshot.revision != 0 or snapshot.resume_attempt != 1 or snapshot.claim_token is not None:
            raise ValueError("new checkpoint v3 records must be unclaimed at revision zero")
        row = _checkpoint_row(snapshot)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                f"INSERT OR IGNORE INTO checkpoints ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' for _ in _COLUMNS)})",
                row,
            )
            return cursor.rowcount == 1

    def load_checkpoint(self, checkpoint_key: str) -> Checkpoint | None:
        with self._lock:
            row = self._conn.execute(
                _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?",
                (checkpoint_key,),
            ).fetchone()
        return _checkpoint_from_row(row) if row is not None else None

    def claim_checkpoint(
        self,
        checkpoint_key: str,
        cycle_index: int,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
        claim_mode: ClaimMode,
    ) -> Checkpoint | None:
        _validate_claim(cycle_index, claim_token, lease_expires_at_ms, now_ms)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?",
                    (checkpoint_key,),
                ).fetchone()
                if row is None:
                    self._conn.commit()
                    return None
                checkpoint = _checkpoint_from_row(row)
                try:
                    check_claim(checkpoint, cycle_index, now_ms, claim_mode)
                except ValueError as exc:
                    raise CheckpointConflictError(str(exc)) from exc
                if checkpoint.claim_token is not None and claim_mode != "recovery":
                    raise CheckpointConflictError("expired checkpoint claims require recovery mode")
                if checkpoint.status is AgentStatus.RECONCILIATION_REQUIRED and claim_mode != "recovery":
                    raise CheckpointConflictError("reconciliation checkpoints require recovery mode")
                cursor = self._conn.execute(
                    """
                    UPDATE checkpoints
                    SET revision = revision + 1,
                        resume_attempt = resume_attempt + ?, status = ?,
                        claim_token = ?, claimed_cycle = ?, lease_expires_at_ms = ?
                    WHERE checkpoint_key = ? AND revision = ?
                      AND (claim_token IS NULL OR lease_expires_at_ms <= ?)
                      AND terminal_result IS NULL
                    """,
                    (
                        int(claim_mode == "recovery"),
                        AgentStatus.RUNNING.value,
                        claim_token,
                        cycle_index,
                        lease_expires_at_ms,
                        checkpoint_key,
                        checkpoint.revision,
                        now_ms,
                    ),
                )
                if cursor.rowcount != 1:
                    raise CheckpointConflictError(f"checkpoint cycle {cycle_index} for key {checkpoint_key} is already claimed")
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        checkpoint.revision += 1
        if claim_mode == "recovery":
            checkpoint.resume_attempt += 1
        checkpoint.status = AgentStatus.RUNNING
        checkpoint.claim_token = claim_token
        checkpoint.claimed_cycle = cycle_index
        checkpoint.lease_expires_at_ms = lease_expires_at_ms
        return checkpoint

    def progress_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        if checkpoint.revision != expected_revision:
            return False
        snapshot = checkpoint_from_json(checkpoint_to_json(checkpoint))
        if snapshot.status is not AgentStatus.RUNNING or snapshot.terminal_result is not None:
            return False
        snapshot.revision = expected_revision + 1
        row = dict(zip(_COLUMNS, _checkpoint_row(snapshot), strict=True))
        columns = _PROGRESS_COLUMNS
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE checkpoints SET "
                + ", ".join(f"{column} = ?" for column in columns)
                + " WHERE checkpoint_key = ? AND revision = ? AND claim_token = ?"
                + " AND claimed_cycle = ? AND terminal_result IS NULL"
                + _IDENTITY_WHERE,
                (
                    *(row[column] for column in columns),
                    snapshot.checkpoint_key,
                    expected_revision,
                    claim_token,
                    snapshot.claimed_cycle,
                    *_identity_values(row),
                ),
            )
            return cursor.rowcount == 1

    def suspend_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        if checkpoint.revision != expected_revision:
            return False
        claimed_cycle = checkpoint.claimed_cycle
        if claimed_cycle is None:
            raise ValueError("checkpoint v3 suspend requires an active claim")
        snapshot = replace(
            checkpoint,
            revision=expected_revision + 1,
            claim_token=None,
            claimed_cycle=None,
            lease_expires_at_ms=None,
        )
        snapshot = checkpoint_from_json(checkpoint_to_json(snapshot))
        if snapshot.status is not AgentStatus.RECONCILIATION_REQUIRED or snapshot.cycle_index != claimed_cycle - 1:
            return False
        row = dict(zip(_COLUMNS, _checkpoint_row(snapshot), strict=True))
        columns = _PROGRESS_COLUMNS
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE checkpoints SET "
                + ", ".join(f"{column} = ?" for column in columns)
                + ", claim_token = NULL, claimed_cycle = NULL, lease_expires_at_ms = NULL"
                + " WHERE checkpoint_key = ? AND revision = ? AND claim_token = ?"
                + " AND claimed_cycle = ? AND terminal_result IS NULL"
                + _IDENTITY_WHERE,
                (
                    *(row[column] for column in columns),
                    snapshot.checkpoint_key,
                    expected_revision,
                    claim_token,
                    claimed_cycle,
                    *_identity_values(row),
                ),
            )
            return cursor.rowcount == 1

    def commit_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        if checkpoint.revision != expected_revision:
            return False
        claimed_cycle = checkpoint.claimed_cycle
        if claimed_cycle is None:
            raise ValueError("checkpoint v3 commit requires an active claim")
        if (
            checkpoint.cycle_index != claimed_cycle
            or checkpoint.status is not AgentStatus.RUNNING
            or checkpoint.terminal_result is not None
        ):
            return False
        validate_model_journal_accounting(checkpoint)
        snapshot = replace(
            checkpoint,
            revision=expected_revision + 1,
            claim_token=None,
            claimed_cycle=None,
            lease_expires_at_ms=None,
            event_outbox=[entry for entry in checkpoint.event_outbox if entry.state == "pending"],
            model_call_journal=[],
            tool_journal=[],
        )
        snapshot = checkpoint_from_json(checkpoint_to_json(snapshot))
        row = dict(zip(_COLUMNS, _checkpoint_row(snapshot), strict=True))
        columns = _PROGRESS_COLUMNS
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE checkpoints SET "
                + ", ".join(f"{column} = ?" for column in columns)
                + ", claim_token = NULL, claimed_cycle = NULL, lease_expires_at_ms = NULL"
                + " WHERE checkpoint_key = ? AND revision = ? AND claim_token = ?"
                + " AND claimed_cycle = ? AND terminal_result IS NULL"
                + _IDENTITY_WHERE,
                (
                    *(row[column] for column in columns),
                    snapshot.checkpoint_key,
                    expected_revision,
                    claim_token,
                    claimed_cycle,
                    *_identity_values(row),
                ),
            )
            return cursor.rowcount == 1

    def finalize_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        expected_revision: int,
    ) -> bool:
        if checkpoint.revision != expected_revision:
            return False
        snapshot = checkpoint_from_json(checkpoint_to_json(checkpoint))
        if snapshot.terminal_result is None or snapshot.claim_token is not None:
            raise ValueError("finalized checkpoint v3 must be terminal and unclaimed")
        snapshot.revision = expected_revision + 1
        row = dict(zip(_COLUMNS, _checkpoint_row(snapshot), strict=True))
        columns = _FINALIZE_COLUMNS
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE checkpoints SET "
                + ", ".join(f"{column} = ?" for column in columns)
                + " WHERE checkpoint_key = ? AND revision = ?"
                + " AND claim_token IS NULL AND terminal_result IS NULL"
                + _IDENTITY_WHERE,
                (
                    *(row[column] for column in columns),
                    snapshot.checkpoint_key,
                    expected_revision,
                    *_identity_values(row),
                ),
            )
            return cursor.rowcount == 1

    def finalize_claimed_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        claim_token: str,
        expected_revision: int,
    ) -> bool:
        with self._lock, self._conn:
            current_row = self._conn.execute(
                _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?",
                (checkpoint.checkpoint_key,),
            ).fetchone()
            if current_row is None:
                return False
            terminal = prepare_claimed_terminal(
                _checkpoint_from_row(current_row),
                checkpoint,
                claim_token=claim_token,
                expected_revision=expected_revision,
            )
            if terminal is None:
                return False
            row = dict(zip(_COLUMNS, _checkpoint_row(terminal), strict=True))
            columns = _FINALIZE_COLUMNS
            cursor = self._conn.execute(
                "UPDATE checkpoints SET "
                + ", ".join(f"{column} = ?" for column in columns)
                + ", claim_token = NULL, claimed_cycle = NULL, lease_expires_at_ms = NULL"
                + " WHERE checkpoint_key = ? AND revision = ? AND claim_token = ?"
                + " AND terminal_result IS NULL"
                + _IDENTITY_WHERE,
                (
                    *(row[column] for column in columns),
                    terminal.checkpoint_key,
                    expected_revision,
                    claim_token,
                    *_identity_values(row),
                ),
            )
            return cursor.rowcount == 1

    def record_event_delivery(
        self,
        checkpoint_key: str,
        *,
        event_id: str,
        payload_digest: str,
        cursor: EventCursor,
        expected_revision: int,
        claim_token: str | None,
    ) -> bool:
        with self._lock, self._conn:
            current_row = self._conn.execute(
                _SELECT_CHECKPOINT + " WHERE checkpoint_key = ?",
                (checkpoint_key,),
            ).fetchone()
            if current_row is None:
                return False
            delivered = prepare_event_delivery(
                _checkpoint_from_row(current_row),
                event_id=event_id,
                payload_digest=payload_digest,
                cursor=cursor,
                expected_revision=expected_revision,
                claim_token=claim_token,
            )
            if delivered is None:
                return False
            row = dict(zip(_COLUMNS, _checkpoint_row(delivered), strict=True))
            claim_clause = "claim_token IS NULL" if claim_token is None else "claim_token = ?"
            parameters: tuple[object, ...] = (
                row["revision"],
                row["event_cursor"],
                row["event_outbox"],
                checkpoint_key,
                expected_revision,
            )
            if claim_token is not None:
                parameters = (*parameters, claim_token)
            result = self._conn.execute(
                "UPDATE checkpoints SET revision = ?, event_cursor = ?, event_outbox = ?"
                + f" WHERE checkpoint_key = ? AND revision = ? AND {claim_clause}",
                parameters,
            )
            return result.rowcount == 1

    def renew_checkpoint_claim(
        self,
        checkpoint_key: str,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> bool:
        _validate_renew(claim_token, lease_expires_at_ms, now_ms)
        clock = _LeaseOperationClock(now_ms)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                current_now_ms = clock.now_ms()
                if lease_expires_at_ms <= current_now_ms:
                    self._conn.rollback()
                    return False
                cursor = self._conn.execute(
                    """
                    UPDATE checkpoints
                    SET lease_expires_at_ms = ?
                    WHERE checkpoint_key = ? AND claim_token = ?
                      AND lease_expires_at_ms > ?
                    """,
                    (
                        lease_expires_at_ms,
                        checkpoint_key,
                        claim_token,
                        current_now_ms,
                    ),
                )
                self._conn.commit()
                return cursor.rowcount == 1
            except BaseException:
                self._conn.rollback()
                raise

    def acknowledge_terminal(self, checkpoint_key: str, *, expected_revision: int) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE checkpoints
                SET revision = revision + 1, terminal_acknowledged = 1
                WHERE checkpoint_key = ? AND revision = ?
                  AND terminal_result IS NOT NULL AND claim_token IS NULL
                  AND terminal_acknowledged = 0
                """,
                (checkpoint_key, expected_revision),
            )
            return cursor.rowcount == 1

    def delete_checkpoint(self, checkpoint_key: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM checkpoints WHERE checkpoint_key = ?",
                (checkpoint_key,),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_key TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL CHECK (schema_version = 'vv-agent.checkpoint.v3'),
    run_definition_schema TEXT NOT NULL CHECK (run_definition_schema = 'vv-agent.run-definition.v2'),
    run_definition TEXT NOT NULL,
    task_id TEXT NOT NULL,
    root_run_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    run_definition_digest TEXT NOT NULL,
    resume_attempt INTEGER NOT NULL CHECK (resume_attempt >= 1),
    cycle_index INTEGER NOT NULL CHECK (cycle_index >= 0),
    status TEXT NOT NULL,
    messages TEXT NOT NULL,
    cycles TEXT NOT NULL,
    model_calls TEXT NOT NULL,
    shared_state TEXT NOT NULL,
    budget_usage TEXT,
    event_cursor TEXT,
    event_outbox TEXT NOT NULL,
    extension_state TEXT NOT NULL,
    model_call_journal TEXT NOT NULL,
    tool_journal TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
    claim_token TEXT,
    claimed_cycle INTEGER,
    lease_expires_at_ms INTEGER,
    terminal_result TEXT,
    terminal_acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (terminal_acknowledged IN (0, 1)),
    CHECK (
        (claim_token IS NULL AND claimed_cycle IS NULL AND lease_expires_at_ms IS NULL)
        OR
        (claim_token IS NOT NULL AND claimed_cycle IS NOT NULL AND lease_expires_at_ms IS NOT NULL)
    ),
    CHECK (claim_token IS NULL OR claimed_cycle = cycle_index + 1),
    CHECK (terminal_result IS NULL OR claim_token IS NULL)
)
"""
_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS checkpoints_status_idx ON checkpoints(status)
"""


def _normalize_schema_sql(sql: str) -> str:
    return " ".join(sql.replace("IF NOT EXISTS", "").split())


_COLUMNS = (
    "checkpoint_key",
    "schema_version",
    "run_definition_schema",
    "run_definition",
    "task_id",
    "root_run_id",
    "trace_id",
    "run_definition_digest",
    "resume_attempt",
    "cycle_index",
    "status",
    "messages",
    "cycles",
    "model_calls",
    "shared_state",
    "budget_usage",
    "event_cursor",
    "event_outbox",
    "extension_state",
    "model_call_journal",
    "tool_journal",
    "revision",
    "claim_token",
    "claimed_cycle",
    "lease_expires_at_ms",
    "terminal_result",
    "terminal_acknowledged",
)
_PROGRESS_COLUMNS = tuple(
    column
    for column in _COLUMNS
    if column
    not in {
        "checkpoint_key",
        "schema_version",
        "run_definition_schema",
        "run_definition",
        "task_id",
        "root_run_id",
        "trace_id",
        "run_definition_digest",
        "resume_attempt",
        "claim_token",
        "claimed_cycle",
        "lease_expires_at_ms",
        "terminal_result",
        "terminal_acknowledged",
    }
)
_FINALIZE_COLUMNS = (*_PROGRESS_COLUMNS, "terminal_result")
_IDENTITY_WHERE = (
    " AND schema_version = ? AND run_definition_schema = ? AND run_definition = ?"
    " AND task_id = ? AND root_run_id = ? AND trace_id = ?"
    " AND run_definition_digest = ? AND resume_attempt = ?"
    " AND terminal_acknowledged = ?"
)
_SELECT_CHECKPOINT = f"SELECT {', '.join(_COLUMNS)} FROM checkpoints"


def _checkpoint_row(checkpoint: Checkpoint) -> tuple[object, ...]:
    full = checkpoint_to_dict(checkpoint)
    return (
        full["checkpoint_key"],
        full["schema_version"],
        full["run_definition_schema"],
        _json_dump(full["run_definition"]),
        full["task_id"],
        full["root_run_id"],
        full["trace_id"],
        full["run_definition_digest"],
        full["resume_attempt"],
        full["cycle_index"],
        full["status"],
        _json_dump(full["messages"]),
        _json_dump(full["cycles"]),
        _json_dump(full["model_calls"]),
        _json_dump(full["shared_state"]),
        _json_dump(full["budget_usage"]) if full["budget_usage"] is not None else None,
        _json_dump(full["event_cursor"]) if full["event_cursor"] is not None else None,
        _json_dump(full["event_outbox"]),
        _json_dump(full["extension_state"]),
        _json_dump(full["model_call_journal"]),
        _json_dump(full["tool_journal"]),
        full["revision"],
        full["claim_token"],
        full["claimed_cycle"],
        full["lease_expires_at_ms"],
        _json_dump(full["terminal_result"]) if full["terminal_result"] is not None else None,
        int(full["terminal_acknowledged"]),
    )


def _checkpoint_from_row(row: tuple[object, ...]) -> Checkpoint:
    values = dict(zip(_COLUMNS, row, strict=True))
    payload = {
        "checkpoint_key": values["checkpoint_key"],
        "schema_version": values["schema_version"],
        "run_definition_schema": values["run_definition_schema"],
        "run_definition": (
            _json_load(values["run_definition"], "run_definition") if values["run_definition"] is not None else None
        ),
        "task_id": values["task_id"],
        "root_run_id": values["root_run_id"],
        "trace_id": values["trace_id"],
        "run_definition_digest": values["run_definition_digest"],
        "resume_attempt": values["resume_attempt"],
        "cycle_index": values["cycle_index"],
        "status": values["status"],
        "messages": _json_load(values["messages"], "messages"),
        "cycles": _json_load(values["cycles"], "cycles"),
        "model_calls": _json_load(values["model_calls"], "model_calls"),
        "shared_state": _json_load(values["shared_state"], "shared_state"),
        "budget_usage": (_json_load(values["budget_usage"], "budget_usage") if values["budget_usage"] is not None else None),
        "event_cursor": (_json_load(values["event_cursor"], "event_cursor") if values["event_cursor"] is not None else None),
        "event_outbox": _json_load(values["event_outbox"], "event_outbox"),
        "extension_state": _json_load(values["extension_state"], "extension_state"),
        "model_call_journal": _json_load(
            values["model_call_journal"],
            "model_call_journal",
        ),
        "tool_journal": _json_load(values["tool_journal"], "tool_journal"),
        "revision": values["revision"],
        "claim_token": values["claim_token"],
        "claimed_cycle": values["claimed_cycle"],
        "lease_expires_at_ms": values["lease_expires_at_ms"],
        "terminal_result": (
            _json_load(values["terminal_result"], "terminal_result") if values["terminal_result"] is not None else None
        ),
        "terminal_acknowledged": bool(values["terminal_acknowledged"]),
    }
    return checkpoint_from_dict(payload)


def _identity_values(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["schema_version"],
        row["run_definition_schema"],
        row["run_definition"],
        row["task_id"],
        row["root_run_id"],
        row["trace_id"],
        row["run_definition_digest"],
        row["resume_attempt"],
        row["terminal_acknowledged"],
    )


def _json_dump(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_load(value: object, field_name: str) -> Any:
    try:
        return _strict_json_loads(str(value))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint v3 {field_name} JSON") from exc
