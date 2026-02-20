from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from vv_agent.runtime.state import Checkpoint
from vv_agent.types import AgentStatus, CycleRecord, Message


class SqliteStateStore:
    """Persistent state store backed by SQLite."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
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
            )
            """
        )
        self._conn.commit()

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        messages_json = json.dumps(
            [m.to_dict() for m in checkpoint.messages],
            ensure_ascii=False,
        )
        cycles_json = json.dumps(
            [c.to_dict() for c in checkpoint.cycles],
            ensure_ascii=False,
        )
        shared_json = json.dumps(
            checkpoint.shared_state, ensure_ascii=False, default=str,
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO checkpoints
                (task_id, cycle_index, status, messages, cycles, shared_state)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint.task_id,
                checkpoint.cycle_index,
                checkpoint.status.value,
                messages_json,
                cycles_json,
                shared_json,
            ),
        )
        self._conn.commit()

    def load_checkpoint(self, task_id: str) -> Checkpoint | None:
        cursor = self._conn.execute(
            "SELECT task_id, cycle_index, status, messages, cycles, shared_state FROM checkpoints WHERE task_id = ?",
            (task_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return Checkpoint(
            task_id=row[0],
            cycle_index=row[1],
            status=AgentStatus(row[2]),
            messages=[Message.from_dict(m) for m in json.loads(row[3])],
            cycles=[CycleRecord.from_dict(c) for c in json.loads(row[4])],
            shared_state=json.loads(row[5]),
        )

    def delete_checkpoint(self, task_id: str) -> None:
        self._conn.execute("DELETE FROM checkpoints WHERE task_id = ?", (task_id,))
        self._conn.commit()

    def list_checkpoints(self) -> list[str]:
        cursor = self._conn.execute("SELECT task_id FROM checkpoints ORDER BY task_id")
        return [row[0] for row in cursor.fetchall()]

    def close(self) -> None:
        self._conn.close()
