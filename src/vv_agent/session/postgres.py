"""SDK defaults and host transaction assembly for the actual core PostgreSQL adapter."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from vv_agent.kernel._postgres import CLOCK as CLOCK
from vv_agent.kernel._postgres import DDL as DDL
from vv_agent.kernel._postgres import PostgresStore as CorePostgresStore
from vv_agent.kernel._postgres import _connection_conninfo as _connection_conninfo

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
