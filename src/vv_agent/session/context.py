"""Context projection with the outer historical-prefix rebuild boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from vv_agent.kernel._context import InputProtection as InputProtection
from vv_agent.kernel._context import message_ids as message_ids
from vv_agent.kernel._context import project_context as _project_context
from vv_agent.kernel._context import protected_start as protected_start
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._message_values import Message
from vv_agent.kernel._records import InboxItem, _MessageCache
from vv_agent.kernel._reducer import Fold as CoreFold
from vv_agent.kernel._state import ExecutionState
from vv_agent.kernel._store import StoredRecord


def project_context(
    records: Sequence[StoredRecord],
    state: ExecutionState,
    *,
    compacted: bool = True,
    _message_cache: _MessageCache | None = None,
    _initial: list[Message] | None = None,
    _task: AgentTask | None = None,
    _seen_turn: bool = False,
    _fold_factory: Callable[[], CoreFold] | None = None,
) -> list[Message]:
    if (
        compacted
        and _initial is None
        and state._protection is not None
        and records
        and records[-1].seq != state._protection.prefix_seq
    ):
        from .reducer import Fold

        logical = [row.record for row in records]
        inputs = [InboxItem(**r._payload["input"]) for r in logical if r.kind == "input_applied"]
        factory = Fold if _fold_factory is None else _fold_factory
        state = factory().extend(logical, consumed_inputs=inputs)
        return _project_context(records, state)
    return _project_context(
        records,
        state,
        compacted=compacted,
        _message_cache=_message_cache,
        _initial=_initial,
        _task=_task,
        _seen_turn=_seen_turn,
    )
