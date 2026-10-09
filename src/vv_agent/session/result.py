"""Read-only projection of a turn into the existing SDK result types."""

from __future__ import annotations

import json
from copy import deepcopy

from vv_agent.budget import BudgetExhaustion, BudgetUsageSnapshot
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
from .reducer import ExecutionState
from .runtime import Runtime, model_usage
from .store import SessionStore, StoredRecord


def project_result(
    store: SessionStore,
    session_id: str,
    turn_id: str,
    *,
    runtime: Runtime | None = None,
    snapshot: tuple[ExecutionState, tuple[StoredRecord, ...], int] | None = None,
) -> RunResult:
    state, records, _ = snapshot if snapshot is not None else store.read_state(session_id)
    turn = state.turns[turn_id]
    task = AgentTask.from_dict(turn.start.payload["definition"]["task"])
    terminal = next((r.record for r in reversed(records) if r.record.kind == "turn_ended" and r.record.turn_id == turn_id), None)
    through = next((i + 1 for i, r in enumerate(records) if r.record == terminal), len(records))
    records = records[:through]
    messages = project_context(records, state)
    cycles, calls, ledger = [], [], []
    plan_seqs = {r.record.record_id: r.seq for r in records}
    compaction_seqs = [r.seq for r in records if r.record.kind == "context_compacted" and r.record.turn_id == turn_id]
    previous_primary_seq = plan_seqs[turn.start.record_id]
    shared = deepcopy(task.initial_shared_state)
    for stored in records:
        r = stored.record
        if r.turn_id != turn_id:
            continue
        if r.kind == "boundary_recorded" and "shared_state" in r._payload["data"]:
            shared = r.payload["data"]["shared_state"]
        if r.kind != "op_completed":
            continue
        if r.payload["context"] == "normal" and r.payload["shared_state"] is not None:
            shared = deepcopy(r.payload["shared_state"])
    for oid, op in state.operations.items():
        if op.turn_id != turn_id or op.kind != "model":
            continue
        for number, attempt in op.attempts.items():
            if not attempt.started:
                continue
            purpose = attempt.plan.payload["purpose"]
            cycle = (
                attempt.plan._payload["request"]["metadata"]
                .get("vv_session", {})
                .get("cycle_index", int(oid.rsplit("/", 1)[1]) if purpose in {"primary", "output_repair"} else 1)
            )
            ledger.append(
                {
                    "purpose": purpose,
                    "operation_id": oid,
                    "attempt": number,
                    "usage": model_usage(attempt.result._payload["usage"] if attempt.result else None).to_dict(),
                    "status": "completed"
                    if attempt.result and not attempt.result._payload["result"].get("error_code")
                    else "failed"
                    if attempt.result
                    else "ambiguous",
                }
            )
            error_code = attempt.result._payload["result"].get("error_code") if attempt.result else "model_outcome_unknown"
            calls.append(
                ModelCallRecord(
                    call_id=f"{oid}/{number}",
                    operation_id=oid,
                    attempt=number,
                    operation={
                        "compaction": ModelCallOperation.MEMORY_COMPACTION,
                        "session_memory": ModelCallOperation.SESSION_MEMORY,
                        "output_repair": ModelCallOperation.OUTPUT_REPAIR,
                    }.get(purpose, ModelCallOperation.AGENT_CYCLE),
                    cycle_index=cycle,
                    backend=turn.start._payload["definition"]["model_binding"]["backend"],
                    model=attempt.plan._payload["request"]["model"],
                    status=ModelCallStatus.FAILED
                    if attempt.result and error_code
                    else ModelCallStatus.COMPLETED
                    if attempt.result
                    else ModelCallStatus.AMBIGUOUS,
                    usage=model_usage(attempt.result.payload["usage"] if attempt.result else None),
                    error_code=error_code,
                    _kernel=True,
                )
            )
            if purpose != "primary" or op.selected_attempt != number or not attempt.result or error_code:
                continue
            response = attempt.result.payload["result"]
            tool_results = []
            for child in state.operations.values():
                if child.kind != "model" and child.attempts[1].plan._payload["dependencies"] == [oid]:
                    child_attempt = child.attempts[child.selected_attempt or max(child.attempts)]
                    if child_attempt.result:
                        tool_results.append(ToolExecutionResult.from_dict(child_attempt.result.payload["result"]))
                    elif child_attempt.wait and "interaction_result" in child_attempt.wait:
                        tool_results.append(ToolExecutionResult.from_dict(deepcopy(child_attempt.wait["interaction_result"])))
            cycles.append(
                CycleRecord(
                    cycle,
                    response.get("content", ""),
                    [ToolCall.from_dict(c) for c in response.get("tool_calls", [])],
                    tool_results,
                    memory_compacted=any(
                        previous_primary_seq < seq < plan_seqs[attempt.plan.record_id] for seq in compaction_seqs
                    ),
                )
            )
            previous_primary_seq = plan_seqs[attempt.plan.record_id]
    usage = summarize_task_token_usage(calls)
    usage._kernel = True
    status, reason, output, error, budget_usage = AgentStatus.RUNNING, None, None, None, None
    completion_tool_name, wait_reason, exhaustion = None, None, None
    for (tid, stage, _), r in state.boundaries.items():
        if tid == turn_id and stage == "budget" and r._payload["data"]["exhaustion"]:
            exhaustion = BudgetExhaustion.from_dict(r._payload["data"]["exhaustion"])
            break
    if terminal:
        p = terminal.payload
        output = p["result"]
        if p["reason"] == "budget_exhausted":
            output = "Run budget exhausted."
        status = AgentStatus.COMPLETED if p["status"] == "completed" else AgentStatus.FAILED
        reason = CompletionReason.NO_TOOL_FINISH if status == AgentStatus.COMPLETED else CompletionReason.FAILED
        if p["reason"] in {r.value for r in CompletionReason}:
            reason = CompletionReason(p["reason"])
        if p["status"] in {"cancelled", "aborted"}:
            reason = CompletionReason.CANCELLED
        if p["reason"] == "max_cycles":
            status = AgentStatus.MAX_CYCLES
        if status == AgentStatus.FAILED:
            error = {
                "code": "run_budget_exhausted" if p["reason"] == "budget_exhausted" else p["reason"] or "agent_failed",
                "message": str(output or p["reason"] or "failed"),
                "retryable": False,
            }
        if p["budget"]:
            budget_usage = BudgetUsageSnapshot.from_dict(p["budget"])
        if reason in {CompletionReason.TOOL_FINISH, CompletionReason.STOP_ON_FIRST_TOOL, CompletionReason.STOP_AT_TOOL_NAME}:
            completion_tool_name = next(
                (
                    c.name
                    for cycle in reversed(cycles)
                    for c, result in zip(cycle.tool_calls, cycle.tool_results, strict=False)
                    if result.directive.value == "finish"
                ),
                None,
            )
    else:
        waits = [a.wait for op in state.operations.values() if op.turn_id == turn_id for a in op.attempts.values() if a.wait]
        status = AgentStatus.SUSPENDED if turn.suspended else AgentStatus.WAIT_USER
        if turn.wait:
            wait_reason = turn.wait._payload["question"] or "No tool call and runtime is waiting for user."
            output = cycles[-1].assistant_message if cycles else None
            reason = CompletionReason.WAIT_USER
        elif waits:
            handle = waits[0]["handle"]
            wait_reason = handle.get("question", "approval" if handle["kind"] == "approval" else "deferred_pending")
            output = wait_reason
            reason = CompletionReason.WAIT_USER
            completion_tool_name = next((c.name for c in cycles[-1].tool_calls if c.name == "ask_user"), None) if cycles else None
        if (
            (runtime.config.budget_limits and runtime.config.budget_limits.has_limits)
            if runtime
            else any(turn.start._payload["budget"].values())
        ):
            from .runtime import budget

            evaluator = budget(state, turn_id)
            budget_usage = evaluator.snapshot() if evaluator else None
    transferred = None
    if terminal is not None:
        for op in state.operations.values():
            if op.turn_id != turn_id or op.kind == "model" or op.selected_attempt is None:
                continue
            attempt = op.attempts[op.selected_attempt]
            if attempt.result and attempt.result._payload["result"].get("metadata", {}).get("mode") == "handoff":
                if attempt.child_handle is None:
                    continue
                child_id = attempt.child_handle["session_id"]
                transferred = project_result(
                    store,
                    child_id,
                    attempt.child_handle["turn_id"],
                    runtime=runtime.child_runtime(store, child_id) if runtime else None,
                )
                output, shared = transferred.final_output, transferred.raw_result.shared_state
                reason, completion_tool_name = transferred.completion_reason, transferred.completion_tool_name
                error = transferred.raw_result.error
                break
    if status == AgentStatus.COMPLETED and transferred is None and runtime:
        output = Runner._coerce_output_type(agent=runtime.agent, final_output=output)
    partial = cycles[-1].assistant_message or None if cycles and status != AgentStatus.COMPLETED else None
    checked = state.boundaries.get((turn_id, "output_checked", "final"))
    if checked and checked._payload["data"]["status"] == "failed" and "partial_output" in checked._payload["data"]:
        candidate = checked.payload["data"]["partial_output"]
        partial = (
            candidate
            if isinstance(candidate, str)
            else json.dumps(candidate, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        ) or partial
    raw = AgentResult(
        status=status,
        messages=messages,
        cycles=cycles,
        final_answer=(
            output
            if isinstance(output, str) or output is None
            else json.dumps(RunResult._serializable_output(output), ensure_ascii=False)
        )
        if status == AgentStatus.COMPLETED
        else None,
        wait_reason=wait_reason,
        error=error,
        shared_state=shared,
        token_usage=usage,
        completion_reason=reason,
        completion_tool_name=completion_tool_name,
        budget_usage=budget_usage,
        budget_exhaustion=exhaustion,
        error_code=str(error["code"]) if error else None,
        partial_output=partial,
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
        metadata={"session_model_calls": ledger},
        agent_name=transferred.agent_name
        if transferred
        else runtime.agent.name
        if runtime
        else turn.start._payload["definition"]["agent_name"],
        resolved_model=transferred.resolved_model if transferred else runtime.resolved if runtime else None,
    )
