"""SDK SQL assembly over the actual shared core transaction rules."""

from collections.abc import Callable

from vv_agent.kernel._reducer import Fold as CoreFold
from vv_agent.kernel._sql import SQLSessionTx as SQLSessionTx
from vv_agent.kernel._sql import SQLStore as CoreSQLStore

from .reducer import Fold


class SQLStore(CoreSQLStore):
    # Concrete driver constructors own connections and need not call super().
    _fold_factory: Callable[[], CoreFold] = Fold

    def __init__(self, *, fold_factory: Callable[[], CoreFold] = Fold):
        super().__init__(fold_factory=fold_factory)
