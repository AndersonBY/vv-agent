"""SDK fold assembly; all state transitions and validation live in core."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from vv_agent.kernel._message_values import Message
from vv_agent.kernel._records import InboxItem, Record
from vv_agent.kernel._reducer import CountMessages, NormalizeMemorySettings
from vv_agent.kernel._reducer import Fold as CoreFold
from vv_agent.kernel._reducer import TransitionError as TransitionError
from vv_agent.kernel._reducer import require as require
from vv_agent.kernel._sql import SQLStore as CoreSQLStore
from vv_agent.kernel._state import Attempt as Attempt
from vv_agent.kernel._state import ExecutionState as ExecutionState
from vv_agent.kernel._state import Operation as Operation
from vv_agent.kernel._state import Turn as Turn
from vv_agent.kernel._token_count import _StopIterationTransport
from vv_agent.memory.manager import MemoryManager
from vv_agent.memory.token_utils import count_messages_tokens


def _normalize_memory_settings(settings: dict[str, Any], protected_start: int | None) -> dict[str, Any]:
    return MemoryManager(**settings, protected_start=protected_start)._definition_settings()


class Fold(CoreFold):
    """SDK assembly resolves token counts outside suspended core transitions."""

    def __init__(
        self,
        *,
        normalize_memory_settings: NormalizeMemorySettings | None = None,
        count_messages: CountMessages | None = None,
    ) -> None:
        super().__init__(
            normalize_memory_settings=_normalize_memory_settings
            if normalize_memory_settings is None
            else normalize_memory_settings,
            count_messages=count_messages_tokens if count_messages is None else count_messages,
        )

    def extend(
        self, records: Iterable[Record], *, consumed_inputs: Iterable[InboxItem] = (), bodies: Iterable[bytes] | None = None
    ) -> ExecutionState:
        steps = self.extend_steps(records, consumed_inputs=consumed_inputs, bodies=bodies)
        count = None
        try:
            while True:
                try:
                    request = next(steps) if count is None else steps.send(count)
                except StopIteration as done:
                    return done.value
                except _StopIterationTransport as transport:
                    raise transport.original from transport.original.__cause__
                messages = [request.messages] if isinstance(request.messages, Message) else request.messages
                count = self._count_messages([m.to_openai_message() for m in messages], model=request.model)
        finally:
            steps.close()


def fold(records: Iterable[Record], *, consumed_inputs: Iterable[InboxItem] = ()) -> ExecutionState:
    return Fold().extend(records, consumed_inputs=consumed_inputs)


def _fold_factory_for(store: object) -> Callable[[], CoreFold]:
    # Core SQL stores own explicit bindings; other SessionStore implementations keep SDK defaults.
    return store.fold_factory if isinstance(store, CoreSQLStore) else Fold
