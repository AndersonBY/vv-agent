"""Selected primary cycles projected from the existing attempt ledger."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from ._execution_values import CycleRecord, TaskTokenUsage
from ._records import Record
from ._state import ExecutionState
from ._store import StoredRecord
from ._tool_values import ToolCall, ToolExecutionResult


def project_cycles(
    state: ExecutionState,
    records: Sequence[StoredRecord],
    turn_id: str,
    usage: TaskTokenUsage,
) -> list[CycleRecord]:
    """Use the caller's committed prefix and already-projected usage ledger."""
    turn = state.turns[turn_id]
    cycles = []
    plan_seqs = {r.record.record_id: r.seq for r in records}
    compaction_seqs = [r.seq for r in records if r.record.kind == "context_compacted" and r.record.turn_id == turn_id]
    previous_primary_seq = plan_seqs[turn.start.record_id]
    tools_by_model = {}
    for child in state.operations.values():
        dependencies = child.attempts[1].plan._payload["dependencies"]
        if child.kind != "model" and len(dependencies) == 1:
            tools_by_model.setdefault(dependencies[0], []).append(child)
    for call in usage.model_calls:
        oid, number = call.operation_id, call.attempt
        op = state.operations[oid]
        attempt = op.attempts[number]
        purpose = attempt.plan._payload["purpose"]
        cycle = call.cycle_index
        error_code = call.error_code
        if purpose != "primary" or op.selected_attempt != number or not attempt.result or error_code:
            continue
        response = attempt.result.payload["result"]
        tool_results = []
        for child in tools_by_model.get(oid, ()):
            child_attempt = child.attempts[child.selected_attempt or max(child.attempts)]
            if child_attempt.result:
                tool_results.append(ToolExecutionResult.from_dict(child_attempt.result.payload["result"]))
            elif child_attempt.wait and "interaction_result" in child_attempt.wait:
                tool_results.append(ToolExecutionResult.from_dict(deepcopy(child_attempt.wait["interaction_result"])))
        cycles.append(
            CycleRecord(
                cycle,
                response.get("content") or "",
                [ToolCall.from_dict(c) for c in response.get("tool_calls", [])],
                tool_results,
                memory_compacted=any(previous_primary_seq < seq < plan_seqs[attempt.plan.record_id] for seq in compaction_seqs),
            )
        )
        previous_primary_seq = plan_seqs[attempt.plan.record_id]
    return cycles


def turn_result_prefix(records: tuple[StoredRecord, ...], turn_id: str) -> tuple[Record | None, tuple[StoredRecord, ...]]:
    terminal = next((r.record for r in reversed(records) if r.record.kind == "turn_ended" and r.record.turn_id == turn_id), None)
    through = next((i + 1 for i, r in enumerate(records) if r.record is terminal), len(records))
    records = records[:through]
    return terminal, records
