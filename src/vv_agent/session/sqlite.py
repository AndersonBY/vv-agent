"""SDK defaults for the actual core SQLite adapter."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from vv_agent.kernel._sqlite import DDL as DDL
from vv_agent.kernel._sqlite import SQLiteStore as CoreSQLiteStore

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
