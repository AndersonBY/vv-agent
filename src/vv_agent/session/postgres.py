"""SDK defaults and host transaction assembly for the actual core PostgreSQL adapter."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, LiteralString, cast

from vv_agent.kernel._content import KeyProvider
from vv_agent.kernel._postgres import CLOCK as CLOCK
from vv_agent.kernel._postgres import DDL as DDL
from vv_agent.kernel._postgres import PostgresStore as CorePostgresStore
from vv_agent.kernel._postgres import _connection_conninfo as _connection_conninfo

from .content_store import SQLContentStore
from .reducer import Fold
from .sql import SQLSessionTx, SQLStore

if TYPE_CHECKING:
    from psycopg import Connection


class PostgresStore(CorePostgresStore, SQLStore):
    def __init__(self, connection: Connection[Any]):
        super().__init__(connection, fold_factory=Fold)

    @classmethod
    @contextmanager
    def standalone(cls, conninfo: str) -> Iterator[PostgresStore]:
        with cls._connect(conninfo) as connection:
            yield cls(connection)

    def _heartbeat_connection(self, conninfo: str):
        return PostgresStore.standalone(conninfo)


def join_transaction(connection: Connection[Any]) -> SQLSessionTx:
    """Join the host's open transaction; never open a connection or commit it."""
    return SQLSessionTx(PostgresStore(connection))


class _PostgresContentBackend:
    def __init__(self, connection: Connection[Any]) -> None:
        self.connection = connection

    @staticmethod
    def _query(query: str) -> LiteralString:
        return cast(LiteralString, query.replace("{p}", "%s"))

    def fetchone(self, query: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
        from psycopg.rows import tuple_row

        with self.connection.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(self._query(query), params)
            row = cursor.fetchone()
            return None if row is None else tuple(row)

    def fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        from psycopg.rows import tuple_row

        with self.connection.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(self._query(query), params)
            return [tuple(row) for row in cursor.fetchall()]

    def run(self, query: str, params: tuple[Any, ...] = ()) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute(self._query(query), params)
            return cursor.rowcount

    def require_transaction(self) -> None:
        from psycopg.pq import TransactionStatus

        if self.connection.info.transaction_status != TransactionStatus.INTRANS:
            raise RuntimeError("durable content writes require the caller's outer session transaction")

    def now_ms(self) -> int:
        row = self.fetchone(f"SELECT {CLOCK}")
        if row is None:
            raise RuntimeError("PostgreSQL content clock is unavailable")
        return row[0]


class PostgresContentStore(SQLContentStore):
    """PostgreSQL content adapter bound to a caller-owned connection or transaction."""

    def __init__(
        self,
        source: PostgresStore | SQLSessionTx | Connection[Any],
        key_provider: KeyProvider,
        *,
        provider_id: str | None = None,
    ) -> None:
        connection: Any
        if isinstance(source, SQLSessionTx):
            if not isinstance(source.store, PostgresStore):
                raise TypeError("PostgresContentStore transaction must belong to PostgresStore")
            connection = source.store.connection
        else:
            connection = source.connection if isinstance(source, PostgresStore) else source
        super().__init__(_PostgresContentBackend(connection), key_provider, provider_id=provider_id)
