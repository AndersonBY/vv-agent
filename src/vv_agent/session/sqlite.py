"""SDK defaults for the actual core SQLite adapter."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from vv_agent.kernel._content import KeyProvider
from vv_agent.kernel._sql import SQLSessionTx
from vv_agent.kernel._sqlite import DDL as DDL
from vv_agent.kernel._sqlite import DEFAULT_BUSY_TIMEOUT_MS, DEFAULT_HEARTBEAT_BUSY_TIMEOUT_MS, CoreSQLiteContentStore
from vv_agent.kernel._sqlite import SQLiteStore as CoreSQLiteStore

from .content_store import AESGCMCipher
from .reducer import Fold
from .sql import SQLStore
from .timing import Clock, SystemClock


class SQLiteStore(CoreSQLiteStore, SQLStore):
    def __init__(
        self,
        path: str,
        *,
        clock: Clock | None = None,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        heartbeat_busy_timeout_ms: int = DEFAULT_HEARTBEAT_BUSY_TIMEOUT_MS,
    ):
        super().__init__(
            path,
            fold_factory=Fold,
            clock=clock if clock is not None else SystemClock(),
            busy_timeout_ms=busy_timeout_ms,
            heartbeat_busy_timeout_ms=heartbeat_busy_timeout_ms,
        )

    @classmethod
    @contextmanager
    def standalone(
        cls,
        path: str,
        *,
        clock: Clock | None = None,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        heartbeat_busy_timeout_ms: int = DEFAULT_HEARTBEAT_BUSY_TIMEOUT_MS,
    ) -> Iterator[SQLiteStore]:
        with cls._owned(
            cls(
                path,
                clock=clock,
                busy_timeout_ms=busy_timeout_ms,
                heartbeat_busy_timeout_ms=heartbeat_busy_timeout_ms,
            )
        ) as store:
            yield store

    def _heartbeat_connection(self):
        return self._inherit_runtime_configuration(
            SQLiteStore.standalone(
                self.path,
                clock=self.clock,
                busy_timeout_ms=self.heartbeat_busy_timeout_ms,
                heartbeat_busy_timeout_ms=self.heartbeat_busy_timeout_ms,
            )
        )


class SQLiteContentStore(CoreSQLiteContentStore):
    """SDK SQLite adapter with the production cipher bound by default."""

    def __init__(
        self,
        source: SQLiteStore | SQLSessionTx | sqlite3.Connection,
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
