"""SDK defaults and host transaction assembly for the actual core PostgreSQL adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from vv_agent.kernel._content import KeyProvider
from vv_agent.kernel._postgres import CLOCK as CLOCK
from vv_agent.kernel._postgres import DDL as DDL
from vv_agent.kernel._postgres import UNBOUND_INTERNAL_BINDING, CorePostgresContentStore, RuntimeBinding, _read_runtime_binding
from vv_agent.kernel._postgres import PostgresStore as CorePostgresStore
from vv_agent.kernel._postgres import _connection_conninfo as _connection_conninfo

from .content_store import AESGCMCipher
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
        fold_factory: Callable[[], Fold] = Fold,
    ):
        if tenant_id is not None:
            if runtime_binding is not None and runtime_binding != RuntimeBinding("tenant", tenant_id):
                raise ValueError("session tenant conflicts with the PostgreSQL runtime binding")
            runtime_binding = RuntimeBinding("tenant", tenant_id)
        elif runtime_binding is None:
            runtime_binding = _read_runtime_binding(connection, allow_unbound_internal=False)
        super().__init__(connection, fold_factory=fold_factory, runtime_binding=runtime_binding)
        self._conninfo = _connection_conninfo(connection)
        if self.credential_tenant_id is not None:
            self._tenant_id = self.credential_tenant_id

    def for_tenant(self, tenant_id: str):
        """Return an explicit tenant view that binds transaction-local RLS context."""

        credential_tenant = self.credential_tenant_id
        if credential_tenant is not None and tenant_id != credential_tenant:
            raise ValueError("tenant-isolated PostgreSQL stores cannot be routed to another tenant")
        return super().for_tenant(tenant_id)

    @classmethod
    @contextmanager
    def standalone(cls, conninfo: str, *, fold_factory: Callable[[], Fold] = Fold) -> Iterator[PostgresStore]:
        with cls._connect(conninfo) as connection:
            binding = _read_runtime_binding(connection, allow_unbound_internal=False)
            yield cls(connection, runtime_binding=binding, fold_factory=fold_factory)

    @classmethod
    @contextmanager
    def tenant_standalone(
        cls,
        conninfo: str,
        *,
        tenant_id: str,
        fold_factory: Callable[[], Fold] = Fold,
    ) -> Iterator[PostgresStore]:
        with cls._connect(conninfo) as connection:
            yield cls(connection, tenant_id=tenant_id, fold_factory=fold_factory)

    @classmethod
    @contextmanager
    def worker_standalone(cls, conninfo: str, *, fold_factory: Callable[[], Fold] = Fold) -> Iterator[PostgresStore]:
        with cls._connect(conninfo) as connection:
            yield cls(connection, runtime_binding=RuntimeBinding("worker"), fold_factory=fold_factory)

    @classmethod
    @contextmanager
    def shared_gateway_standalone(
        cls,
        conninfo: str,
        *,
        fold_factory: Callable[[], Fold] = Fold,
    ) -> Iterator[PostgresStore]:
        with cls._connect(conninfo) as connection:
            yield cls(connection, runtime_binding=RuntimeBinding("shared-gateway"), fold_factory=fold_factory)

    @classmethod
    @contextmanager
    def schema_admin_standalone(
        cls,
        conninfo: str,
        *,
        fold_factory: Callable[[], Fold] = Fold,
    ) -> Iterator[PostgresStore]:
        with cls._connect(conninfo) as connection:
            yield cls(connection, runtime_binding=UNBOUND_INTERNAL_BINDING, fold_factory=fold_factory)


def join_transaction(connection: Connection[Any]) -> SQLSessionTx:
    """Join the host's open transaction; never open a connection or commit it."""

    from psycopg.pq import TransactionStatus

    if connection.info.transaction_status != TransactionStatus.INTRANS:
        raise RuntimeError("join_transaction requires an open INTRANS PostgreSQL transaction")
    binding = _read_runtime_binding(connection, allow_unbound_internal=False)
    return SQLSessionTx(PostgresStore(connection, runtime_binding=binding))


class PostgresContentStore(CorePostgresContentStore):
    """SDK PostgreSQL adapter with the production cipher bound by default."""

    def __init__(
        self,
        source: PostgresStore | SQLSessionTx | Connection[Any],
        key_provider: KeyProvider,
        *,
        provider_id: str | None = None,
    ) -> None:
        super().__init__(
            source,
            key_provider,
            AESGCMCipher(),
            provider_id=provider_id,
        )
