from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from threading import RLock

from vv_agent.runtime.checkpoint_codec import checkpoint_from_dict, checkpoint_from_json, checkpoint_to_dict, checkpoint_to_json
from vv_agent.runtime.state import (
    Checkpoint,
    CheckpointConflictError,
    StateStoreSpec,
    _check_claim,
    _LeaseOperationClock,
    _validate_claim,
    _validate_renew,
)


class SqliteStateStore:
    """Persistent state store backed by SQLite."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        raw_path = str(db_path)
        self._db_path = raw_path if raw_path == ":memory:" else str(Path(raw_path).resolve())
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._lock = RLock()
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                task_id TEXT PRIMARY KEY,
                cycle_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                messages TEXT NOT NULL,
                cycles TEXT NOT NULL,
                shared_state TEXT NOT NULL
                ,revision INTEGER NOT NULL DEFAULT 0
                ,claim_token TEXT
                ,claimed_cycle INTEGER
                ,lease_expires_at_ms INTEGER
                ,terminal_result TEXT
            )
            """
        )
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(checkpoints)")}
        for name, declaration in (
            ("revision", "INTEGER NOT NULL DEFAULT 0"),
            ("claim_token", "TEXT"),
            ("claimed_cycle", "INTEGER"),
            ("lease_expires_at_ms", "INTEGER"),
            ("terminal_result", "TEXT"),
        ):
            if name not in columns:
                self._conn.execute(f"ALTER TABLE checkpoints ADD COLUMN {name} {declaration}")
        self._conn.commit()

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        payload = checkpoint_from_json(checkpoint_to_json(checkpoint))
        values = _checkpoint_columns(payload)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                    (task_id, cycle_index, status, messages, cycles, shared_state,
                     revision, claim_token, claimed_cycle, lease_expires_at_ms, terminal_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.task_id,
                    payload.cycle_index,
                    payload.status.value,
                    *values,
                ),
            )

    def create_checkpoint(self, checkpoint: Checkpoint) -> bool:
        payload = checkpoint_from_json(checkpoint_to_json(checkpoint))
        values = _checkpoint_columns(payload)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO checkpoints
                    (task_id, cycle_index, status, messages, cycles, shared_state,
                     revision, claim_token, claimed_cycle, lease_expires_at_ms, terminal_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.task_id,
                    payload.cycle_index,
                    payload.status.value,
                    *values,
                ),
            )
            return cursor.rowcount == 1

    def commit_checkpoint(self, checkpoint: Checkpoint, *, claim_token: str, expected_revision: int) -> bool:
        payload = checkpoint_from_json(
            checkpoint_to_json(replace(checkpoint, claim_token=None, claimed_cycle=None, lease_expires_at_ms=None))
        )
        messages_json, cycles_json, shared_json, _, _, _, _, terminal_result = _checkpoint_columns(payload)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE checkpoints
                SET cycle_index = ?, status = ?, messages = ?, cycles = ?, shared_state = ?,
                    revision = ?, claim_token = NULL, claimed_cycle = NULL, lease_expires_at_ms = NULL,
                    terminal_result = ?
                WHERE task_id = ? AND revision = ? AND claim_token = ? AND claimed_cycle = ?
                """,
                (
                    payload.cycle_index,
                    payload.status.value,
                    messages_json,
                    cycles_json,
                    shared_json,
                    expected_revision + 1,
                    terminal_result,
                    payload.task_id,
                    expected_revision,
                    claim_token,
                    payload.cycle_index,
                ),
            )
            return cursor.rowcount == 1

    def finalize_checkpoint(self, checkpoint: Checkpoint, *, expected_revision: int) -> bool:
        if checkpoint.terminal_result is None:
            raise ValueError("finalized checkpoint must include terminal_result")
        payload = checkpoint_from_json(checkpoint_to_json(checkpoint))
        messages_json, cycles_json, shared_json, _, _, _, _, terminal_result = _checkpoint_columns(payload)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE checkpoints
                SET cycle_index = ?, status = ?, messages = ?, cycles = ?, shared_state = ?,
                    revision = ?, claim_token = NULL, claimed_cycle = NULL, lease_expires_at_ms = NULL,
                    terminal_result = ?
                WHERE task_id = ? AND revision = ? AND claim_token IS NULL AND terminal_result IS NULL
                """,
                (
                    payload.cycle_index,
                    payload.status.value,
                    messages_json,
                    cycles_json,
                    shared_json,
                    expected_revision + 1,
                    terminal_result,
                    payload.task_id,
                    expected_revision,
                ),
            )
            return cursor.rowcount == 1

    def renew_checkpoint_claim(
        self,
        task_id: str,
        *,
        claim_token: str,
        expected_revision: int,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> bool:
        _validate_renew(claim_token, expected_revision, lease_expires_at_ms, now_ms)
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
                    WHERE task_id = ? AND revision = ? AND claim_token = ?
                      AND lease_expires_at_ms > ?
                    """,
                    (lease_expires_at_ms, task_id, expected_revision, claim_token, current_now_ms),
                )
                self._conn.commit()
                return cursor.rowcount == 1
            except BaseException:
                self._conn.rollback()
                raise

    def load_checkpoint(self, task_id: str) -> Checkpoint | None:
        with self._lock:
            cursor = self._conn.execute(
                _SELECT_CHECKPOINT + " WHERE task_id = ?",
                (task_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return _checkpoint_from_row(row)

    def claim_checkpoint(
        self,
        task_id: str,
        cycle_index: int,
        *,
        claim_token: str,
        lease_expires_at_ms: int,
        now_ms: int,
    ) -> Checkpoint | None:
        _validate_claim(cycle_index, claim_token, lease_expires_at_ms, now_ms)
        with self._lock, self._conn:
            row = self._conn.execute(_SELECT_CHECKPOINT + " WHERE task_id = ?", (task_id,)).fetchone()
            if row is None:
                return None
            checkpoint = _checkpoint_from_row(row)
            _check_claim(checkpoint, cycle_index, now_ms)
            cursor = self._conn.execute(
                """
                UPDATE checkpoints
                SET revision = revision + 1, claim_token = ?, claimed_cycle = ?, lease_expires_at_ms = ?
                WHERE task_id = ? AND revision = ?
                  AND (claim_token IS NULL OR lease_expires_at_ms <= ?)
                """,
                (claim_token, cycle_index, lease_expires_at_ms, task_id, checkpoint.revision, now_ms),
            )
            if cursor.rowcount != 1:
                raise CheckpointConflictError(f"checkpoint cycle {cycle_index} for task {task_id} is already claimed")
            checkpoint.revision += 1
            checkpoint.claim_token = claim_token
            checkpoint.claimed_cycle = cycle_index
            checkpoint.lease_expires_at_ms = lease_expires_at_ms
            return checkpoint

    def delete_checkpoint(self, task_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM checkpoints WHERE task_id = ?", (task_id,))

    def acknowledge_terminal(self, task_id: str, *, expected_revision: int) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM checkpoints WHERE task_id = ? AND revision = ? AND terminal_result IS NOT NULL",
                (task_id, expected_revision),
            )
            return cursor.rowcount == 1

    def list_checkpoints(self) -> list[str]:
        with self._lock:
            cursor = self._conn.execute("SELECT task_id FROM checkpoints ORDER BY task_id")
            return [row[0] for row in cursor.fetchall()]

    def state_store_spec(self) -> StateStoreSpec | None:
        if self._db_path == ":memory:":
            return None
        return StateStoreSpec(kind="sqlite", location=self._db_path)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_SELECT_CHECKPOINT = (
    "SELECT task_id, cycle_index, status, messages, cycles, shared_state, "
    "revision, claim_token, claimed_cycle, lease_expires_at_ms, terminal_result FROM checkpoints"
)


def _checkpoint_columns(checkpoint: Checkpoint) -> tuple[object, ...]:
    full = checkpoint_to_dict(checkpoint)
    return (
        json.dumps(full["messages"], ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        json.dumps(full["cycles"], ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        json.dumps(full["shared_state"], ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        full.get("revision", 0),
        full.get("claim_token"),
        full.get("claimed_cycle"),
        full.get("lease_expires_at_ms"),
        json.dumps(full["terminal_result"], ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        if "terminal_result" in full
        else None,
    )


def _checkpoint_from_row(row: tuple[object, ...]) -> Checkpoint:
    return checkpoint_from_dict(
        {
            "task_id": row[0],
            "cycle_index": row[1],
            "status": row[2],
            "messages": json.loads(str(row[3])),
            "cycles": json.loads(str(row[4])),
            "shared_state": json.loads(str(row[5])),
            "revision": row[6],
            "claim_token": row[7],
            "claimed_cycle": row[8],
            "lease_expires_at_ms": row[9],
            "terminal_result": json.loads(str(row[10])) if row[10] is not None else None,
        }
    )
