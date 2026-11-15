"""SDK defaults for the actual core SQLite adapter."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from typing import Any

from vv_agent.kernel._content import KeyProvider
from vv_agent.kernel._sql import SQLSessionTx
from vv_agent.kernel._sqlite import DDL as DDL
from vv_agent.kernel._sqlite import SQLiteStore as CoreSQLiteStore

from .content_store import SQLContentStore
from .reducer import Fold
from .sql import SQLStore
from .timing import Clock, SystemClock


class SQLiteStore(CoreSQLiteStore, SQLStore):
    def __init__(self, path: str, *, clock: Clock | None = None):
        super().__init__(path, fold_factory=Fold, clock=clock if clock is not None else SystemClock())

    @classmethod
    @contextmanager
    def standalone(cls, path: str, *, clock: Clock | None = None) -> Iterator[SQLiteStore]:
        with cls._owned(cls(path, clock=clock)) as store:
            yield store

    def _heartbeat_connection(self):
        return SQLiteStore.standalone(self.path, clock=self.clock)


class _SQLiteContentBackend:
    lock_clause = ""

    def __init__(self, connection: sqlite3.Connection, connection_lock: RLock | None) -> None:
        self.connection = connection
        self.connection_lock = connection_lock

    @contextmanager
    def _serialized(self) -> Iterator[None]:
        if self.connection_lock is None:
            yield
            return
        with self.connection_lock:
            yield

    @staticmethod
    def _query(query: str) -> str:
        return query.replace("{p}", "?")

    def fetchone(self, query: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
        with self._serialized():
            row = self.connection.execute(self._query(query), params).fetchone()
        return None if row is None else tuple(row)

    def fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with self._serialized():
            rows = self.connection.execute(self._query(query), params).fetchall()
        return [tuple(row) for row in rows]

    def run(self, query: str, params: tuple[Any, ...] = ()) -> int:
        with self._serialized():
            return self.connection.execute(self._query(query), params).rowcount

    def require_transaction(self) -> None:
        with self._serialized():
            if not self.connection.in_transaction:
                raise RuntimeError("durable content writes require the caller's outer session transaction")

    def now_ms(self) -> int:
        row = self.fetchone("SELECT session_now_ms()")
        if row is None:
            raise RuntimeError("SQLite content clock is unavailable")
        return row[0]


class SQLiteContentStore(SQLContentStore):
    """SQLite content adapter bound to a caller-owned connection or transaction."""

    def __init__(
        self,
        source: SQLiteStore | SQLSessionTx | sqlite3.Connection,
        key_provider: KeyProvider,
        *,
        provider_id: str | None = None,
    ) -> None:
        connection: Any
        connection_lock: RLock | None = None
        if isinstance(source, SQLSessionTx):
            if not isinstance(source.store, SQLiteStore):
                raise TypeError("SQLiteContentStore transaction must belong to SQLiteStore")
            connection = source.store.connection
            connection_lock = source.store._connection_lock
        else:
            connection = source.connection if isinstance(source, SQLiteStore) else source
            if isinstance(source, SQLiteStore):
                connection_lock = source._connection_lock
        if not isinstance(connection, sqlite3.Connection):
            raise TypeError("SQLiteContentStore requires a SQLite connection or SQLiteStore")
        super().__init__(
            _SQLiteContentBackend(connection, connection_lock),
            key_provider,
            provider_id=provider_id,
        )
