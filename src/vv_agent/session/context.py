"""Context projection with the outer historical-prefix rebuild boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from vv_agent.kernel._context import InputProtection as InputProtection
from vv_agent.kernel._context import message_ids as message_ids
from vv_agent.kernel._context import project_context as _project_context
from vv_agent.kernel._context import protected_start as protected_start
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._message_values import Message
from vv_agent.kernel._records import InboxItem, Record, _MessageCache
from vv_agent.kernel._reducer import Fold as CoreFold
from vv_agent.kernel._reducer import logical_inputs
from vv_agent.kernel._state import ExecutionState
from vv_agent.kernel._store import StoredRecord


def _rebuild_inputs(
    records: Sequence[StoredRecord], bodies: Sequence[bytes] | None
) -> tuple[list[Record], list[InboxItem], dict[str, InboxItem]]:
    if bodies is None:
        raise ValueError("context rebuild requires raw record bodies")
    if len(records) != len(bodies):
        raise ValueError(f"context rebuild record body count mismatch: {len(records)} records, {len(bodies)} bodies")

    logical_records = []
    inputs = []
    logical_items = {}
    for stored, body in zip(records, bodies, strict=True):
        raw = Record.parse(body)
        if raw.record_id != stored.record.record_id:
            raise ValueError(f"context rebuild record body mismatch: expected {stored.record.record_id}, got {raw.record_id}")
        if raw.kind != stored.record.kind:
            raise ValueError(f"context rebuild record kind mismatch: expected {stored.record.kind}, got {raw.kind}")
        logical = stored.record
        if raw.kind == "input_applied":
            logical = replace(logical, payload=logical._payload | {"input": raw._payload["input"]})
            item = InboxItem(**raw._payload["input"])
            item.encode()
            inputs.append(item)
            logical_items[raw.record_id] = InboxItem(**stored.record._payload["input"])
        logical_records.append(logical)
    return logical_records, inputs, logical_items


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
    _record_bodies: Sequence[bytes] | None = None,
) -> list[Message]:
    if (
        compacted
        and _initial is None
        and state._protection is not None
        and records
        and records[-1].seq != state._protection.prefix_seq
    ):
        from .reducer import Fold

        logical, inputs, logical_items = _rebuild_inputs(records, _record_bodies)
        factory = Fold if _fold_factory is None else _fold_factory
        with logical_inputs(logical_items):
            state = factory().extend(logical, consumed_inputs=inputs, bodies=_record_bodies)
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
