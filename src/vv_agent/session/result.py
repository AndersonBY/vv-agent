"""Read-only projection of a turn into the existing SDK result types."""

from __future__ import annotations

from copy import deepcopy

from vv_agent.budget import BudgetUsageSnapshot
from vv_agent.result import RunResult
from vv_agent.runner import Runner
from vv_agent.runtime.token_usage import summarize_task_token_usage
from vv_agent.types import (
    AgentResult,
    AgentStatus,
    AgentTask,
    CompletionReason,
    CycleRecord,
    ModelCallOperation,
    ModelCallRecord,
    ModelCallStatus,
    ToolCall,
    ToolExecutionResult,
)

from .context import project_context
from .projection import project_records
from .runtime import Runtime, model_usage
from .store import SessionStore


def project_result(store: SessionStore, session_id: str, turn_id: str, *, runtime: Runtime) -> RunResult:
    state, records, _ = store.read_state(session_id)
    turn = state.turns[turn_id]
    task = AgentTask.from_dict(turn.start.payload["definition"]["task"])
    terminal = next((r.record for r in reversed(records) if r.record.kind == "turn_ended" and r.record.turn_id == turn_id), None)
    messages = project_context(records, state)
    cycles, calls = [], []
    shared = deepcopy(task.initial_shared_state)
    for stored in records:
        r = stored.record
        if r.turn_id != turn_id or r.kind != "op_completed":
            continue
        if r.payload["context"] == "normal" and "session_shared_state" in r.payload["usage"]:
            shared = deepcopy(r.payload["usage"]["session_shared_state"])
    for oid, op in state.operations.items():
        if op.turn_id != turn_id or op.kind != "model":
            continue
        for number, attempt in op.attempts.items():
            if not attempt.started:
                continue
            purpose = attempt.plan.payload["purpose"]
            if purpose == "output_repair":
                continue  # The current public v23 ledger has no repair operation discriminator.
            cycle = (
                int(oid.rsplit("/", 1)[1]) if purpose == "primary" else attempt.plan.payload["request"]["metadata"]["cycle_index"]
            )
            calls.append(
                ModelCallRecord(
                    call_id=f"{oid}/{number}",
                    operation_id=oid,
                    attempt=number,
                    operation=ModelCallOperation.AGENT_CYCLE if purpose == "primary" else ModelCallOperation.MEMORY_COMPACTION,
                    cycle_index=cycle,
                    backend=runtime.resolved.backend,
                    model=task.model,
                    status=ModelCallStatus.COMPLETED if attempt.result else ModelCallStatus.AMBIGUOUS,
                    usage=model_usage(attempt.result.payload["usage"] if attempt.result else None),
                    error_code=None if attempt.result else "model_outcome_unknown",
                )
            )
            if purpose != "primary" or op.selected_attempt != number or not attempt.result:
                continue
            response = attempt.result.payload["result"]
            tool_results = []
            for child in state.operations.values():
                if child.kind != "model" and child.attempts[1].plan.payload["dependencies"] == [oid] and child.selected_attempt:
                    result = child.attempts[child.selected_attempt].result
                    if result:
                        tool_results.append(ToolExecutionResult.from_dict(result.payload["result"]))
            cycles.append(
                CycleRecord(
                    cycle,
                    response.get("content", ""),
                    [ToolCall.from_dict(c) for c in response.get("tool_calls", [])],
                    tool_results,
                )
            )
    usage = summarize_task_token_usage(calls)
    status, reason, output, error, budget_usage = AgentStatus.RUNNING, None, None, None, None
    completion_tool_name = None
    if terminal:
        p = terminal.payload
        output = p["result"]
        status = AgentStatus.COMPLETED if p["status"] == "completed" else AgentStatus.FAILED
        reason = CompletionReason.NO_TOOL_FINISH if status == AgentStatus.COMPLETED else CompletionReason.FAILED
        if p["reason"] in {r.value for r in CompletionReason}:
            reason = CompletionReason(p["reason"])
        if p["status"] in {"cancelled", "aborted"}:
            reason = CompletionReason.CANCELLED
        if p["reason"] == "max_cycles":
            status = AgentStatus.MAX_CYCLES
        if status == AgentStatus.FAILED:
            error = {"code": p["reason"] or "agent_failed", "message": str(output or p["reason"] or "failed"), "retryable": False}
        if p["budget"]:
            budget_usage = BudgetUsageSnapshot.from_dict(p["budget"])
        if reason in {CompletionReason.TOOL_FINISH, CompletionReason.STOP_ON_FIRST_TOOL, CompletionReason.STOP_AT_TOOL_NAME}:
            completion_tool_name = next((c.name for cycle in reversed(cycles) for c in cycle.tool_calls), None)
    else:
        status = AgentStatus.SUSPENDED if turn.suspended else AgentStatus.WAIT_USER
    raw = AgentResult(
        status=status,
        messages=messages,
        cycles=cycles,
        final_answer=output if status == AgentStatus.COMPLETED else None,
        error=error,
        shared_state=shared,
        token_usage=usage,
        completion_reason=reason,
        completion_tool_name=completion_tool_name,
        budget_usage=budget_usage,
        partial_output=cycles[-1].assistant_message or None if cycles and status != AgentStatus.COMPLETED else None,
    )
    return RunResult(
        input=task.user_prompt,
        new_items=Runner._new_session_items(initial_messages=task.initial_messages, result=raw),
        final_output=output,
        status=status,
        raw_result=raw,
        events=[e for e in project_records(records) if e.run_id == turn_id],
        token_usage=usage,
        run_id=turn_id,
        trace_id=turn_id,
        agent_name=runtime.agent.name,
        resolved_model=runtime.resolved,
    )
