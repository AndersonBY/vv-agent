"""SDK defaults and host transaction assembly for the actual core PostgreSQL adapter."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, LiteralString, cast

from vv_agent.kernel._content import KeyProvider
from vv_agent.kernel._postgres import CLOCK as CLOCK
from vv_agent.kernel._postgres import DDL as DDL
from vv_agent.kernel._postgres import UNBOUND_INTERNAL_BINDING, RuntimeBinding, _read_runtime_binding
from vv_agent.kernel._postgres import PostgresStore as CorePostgresStore
from vv_agent.kernel._postgres import _connection_conninfo as _connection_conninfo

from .content_store import SQLContentStore
from .reducer import Fold
from .sql import SQLSessionTx, SQLStore

if TYPE_CHECKING:
    from psycopg import Connection


class PostgresStore(CorePostgresStore, SQLStore):
    def __init__(
        self,
        connection: Connection[Any],
        *,
        tenant_id: str | None = None,
        runtime_binding: RuntimeBinding | None = None,
    ):
        if tenant_id is not None:
            if runtime_binding is not None and runtime_binding != RuntimeBinding("tenant", tenant_id):
                raise ValueError("session tenant conflicts with the PostgreSQL runtime binding")
            runtime_binding = RuntimeBinding("tenant", tenant_id)
        elif runtime_binding is None:
            runtime_binding = _read_runtime_binding(connection, allow_unbound_internal=False)
        super().__init__(connection, fold_factory=Fold, runtime_binding=runtime_binding)
        binding = self.runtime_binding
        self._credential_tenant_id = tenant_id or (binding.tenant_id if binding.mode == "tenant" else None)
        self._conninfo = _connection_conninfo(connection)
        if self._credential_tenant_id is not None:
            self._tenant_id = self._credential_tenant_id

    @property
    def credential_isolation(self) -> bool:
        """Whether this exact store is backed by a verified tenant login."""

        credential_tenant = getattr(self, "_credential_tenant_id", None)
        binding = self.runtime_binding
        return (
            binding.mode == "tenant"
            and credential_tenant is not None
            and binding.tenant_id == credential_tenant
            and self.tenant_id == credential_tenant
        )

    @property
    def credential_tenant_id(self) -> str | None:
        return getattr(self, "_credential_tenant_id", None)

    def for_tenant(self, tenant_id: str):
        credential_tenant = self.credential_tenant_id
        if credential_tenant is not None and tenant_id != credential_tenant:
            raise ValueError("tenant-isolated PostgreSQL stores cannot be routed to another tenant")
        return super().for_tenant(tenant_id)

    @classmethod
    @contextmanager
    def standalone(cls, conninfo: str) -> Iterator[PostgresStore]:
        with cls._connect(conninfo) as connection:
            binding = _read_runtime_binding(connection, allow_unbound_internal=False)
            yield cls(connection, runtime_binding=binding)

    @classmethod
    @contextmanager
    def tenant_standalone(cls, conninfo: str, *, tenant_id: str) -> Iterator[PostgresStore]:
        with cls._connect(conninfo) as connection:
            yield cls(connection, tenant_id=tenant_id)

    @classmethod
    @contextmanager
    def worker_standalone(cls, conninfo: str) -> Iterator[PostgresStore]:
        with cls._connect(conninfo) as connection:
            yield cls(connection, runtime_binding=RuntimeBinding("worker"))

    @classmethod
    @contextmanager
    def shared_gateway_standalone(cls, conninfo: str) -> Iterator[PostgresStore]:
        with cls._connect(conninfo) as connection:
            yield cls(connection, runtime_binding=RuntimeBinding("shared-gateway"))

    @classmethod
    @contextmanager
    def schema_admin_standalone(cls, conninfo: str) -> Iterator[PostgresStore]:
        with cls._connect(conninfo) as connection:
            yield cls(connection, runtime_binding=UNBOUND_INTERNAL_BINDING)

    def _heartbeat_connection(self, conninfo: str):
        if self.credential_isolation:
            manager = PostgresStore.tenant_standalone(conninfo, tenant_id=self.tenant_id)
        elif self.runtime_mode == "worker":
            manager = PostgresStore.worker_standalone(conninfo)
        elif self.runtime_mode == "shared-gateway":
            manager = PostgresStore.shared_gateway_standalone(conninfo)
        else:
            manager = PostgresStore.standalone(conninfo)
        return self._inherit_runtime_configuration(manager)


def join_transaction(connection: Connection[Any]) -> SQLSessionTx:
    """Join the host's open transaction; never open a connection or commit it."""

    from psycopg.pq import TransactionStatus

    if connection.info.transaction_status != TransactionStatus.INTRANS:
        raise RuntimeError("join_transaction requires an open INTRANS PostgreSQL transaction")
    binding = _read_runtime_binding(connection, allow_unbound_internal=False)
    return SQLSessionTx(PostgresStore(connection, runtime_binding=binding))


class _PostgresContentBackend:
    lock_clause = " FOR UPDATE"

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
