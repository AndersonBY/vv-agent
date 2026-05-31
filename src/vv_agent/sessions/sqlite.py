from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from vv_agent.types import Message


class SQLiteSession:
    def __init__(self, session_id: str, db_path: str | Path = "agent_sessions.sqlite3") -> None:
        self.session_id = session_id
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_items (
                session_id TEXT NOT NULL,
                item_index INTEGER PRIMARY KEY AUTOINCREMENT,
                payload TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get_items(self, limit: int | None = None) -> list[Message]:
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
        return [Message.from_dict(json.loads(row[0])) for row in cursor.fetchall()]

    def add_items(self, items: list[Message]) -> None:
        self._conn.executemany(
            "INSERT INTO session_items (session_id, payload) VALUES (?, ?)",
            [(self.session_id, json.dumps(item.to_dict(), ensure_ascii=False)) for item in items],
        )
        self._conn.commit()

    def pop_item(self) -> Message | None:
        cursor = self._conn.execute(
            "SELECT item_index, payload FROM session_items WHERE session_id = ? ORDER BY item_index DESC LIMIT 1",
            (self.session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        self._conn.execute("DELETE FROM session_items WHERE item_index = ?", (row[0],))
        self._conn.commit()
        return Message.from_dict(json.loads(row[1]))

    def clear_session(self) -> None:
        self._conn.execute("DELETE FROM session_items WHERE session_id = ?", (self.session_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
