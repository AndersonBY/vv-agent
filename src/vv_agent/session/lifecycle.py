"""Durable hook decisions reconstructed from completed cycle receipts."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from vv_agent.runtime.lifecycle import (
    AfterCycleAction,
    AfterCycleDecision,
    AfterCycleHookError,
    AfterCycleHookManager,
    AfterCycleSnapshot,
    NativeCycleOutcome,
    NativeCycleOutcomeKind,
    persist_after_cycle_disallowed_tools,
    read_after_cycle_disallowed_tools,
)
from vv_agent.runtime.tool_call_runner import ToolCallRunner
from vv_agent.types import CompletionReason, ToolCall, ToolDirective

from .providers import Definitive
from .result import project_result

if TYPE_CHECKING:
    from .kernel import _Driver


def after_cycle(driver: _Driver) -> bool:
    manager = AfterCycleHookManager(driver.runtime.config.after_cycle_hooks)
    if not manager.has_hooks():
        return False
    tid = driver.state.active_turn_id
    assert tid is not None
    models = [
        (oid, op)
        for oid, op in driver.state.operations.items()
        if op.turn_id == tid and op.kind == "model" and op.attempts[1].plan._payload["purpose"] == "primary"
    ]
    if not models:
        return False
    oid, op = models[-1]
    if op.selected_attempt is None:
        return False
    receipt = op.attempts[op.selected_attempt].result
    if receipt is None or receipt._payload["result"].get("error_code"):
        return False
    recorded = driver.boundary("after_cycle", oid)
    if recorded is not None:
        error = recorded._payload["data"]["error"]
        if error:
            driver.close("failed", "agent_failed", error)
            return True
        return False
    children = [
        op
        for op in driver.state.operations.values()
        if op.turn_id == tid and op.kind != "model" and op.attempts[1].plan._payload["dependencies"] == [oid]
    ]
    finish = next(
        (
            op
            for op in children
            if op.selected_attempt is not None
            and (receipt_value := op.attempts[op.selected_attempt].result) is not None
            and receipt_value._payload["result"].get("directive") == ToolDirective.FINISH.value
        ),
        None,
    )
    if finish:
        skipped = []
        for child in children:
            a = child.attempts[max(child.attempts)]
            if a.state == "planned":
                value = ToolCallRunner._build_skipped_result(
                    ToolCall.from_dict(a.execution_plan.payload["request"]),
                    error_code="skipped_due_to_finish",
                    message="Tool skipped because a previous tool finished the task.",
                )
                skipped.extend(driver.completed(a.execution_plan, Definitive(value.to_dict())))
        if skipped:
            driver.commit(skipped, guarded=True)
            return True
    waits = [
        a.wait
        for child in children
        for a in child.attempts.values()
        if a.wait and a.wait["handle"]["kind"] == "user" and "interaction_result" in a.wait
    ]
    if any(op.state != "completed" and not any(a.wait in waits for a in op.attempts.values()) for op in children):
        return False
    task = driver.task()
    index = (
        op.attempts[1].plan._payload["request"]["metadata"].get("vv_session", {}).get("cycle_index", int(oid.rsplit("/", 1)[1]))
    )
    if waits:
        native = NativeCycleOutcome(NativeCycleOutcomeKind.WAIT_USER, CompletionReason.WAIT_USER, "ask_user", steer_allowed=False)
    elif finish:
        native = NativeCycleOutcome(
            NativeCycleOutcomeKind.COMPLETED,
            CompletionReason.TOOL_FINISH,
            finish.attempts[1].plan._payload["request"]["name"],
            index < task.max_cycles,
        )
    elif not receipt._payload["result"].get("tool_calls") and task.no_tool_policy != "continue":
        waiting = task.no_tool_policy == "wait_user"
        native = NativeCycleOutcome(
            NativeCycleOutcomeKind.WAIT_USER if waiting else NativeCycleOutcomeKind.COMPLETED,
            CompletionReason.WAIT_USER if waiting else CompletionReason.NO_TOOL_FINISH,
            steer_allowed=not waiting and index < task.max_cycles,
        )
    else:
        native = NativeCycleOutcome(
            NativeCycleOutcomeKind.MAX_CYCLES if index >= task.max_cycles else NativeCycleOutcomeKind.CONTINUE,
            steer_allowed=index < task.max_cycles,
        )
    raw = project_result(driver.store, driver.sid, tid, runtime=driver.runtime).raw_result
    shared = driver.shared_state()
    error = None
    decision = AfterCycleDecision.continue_run()
    try:
        snapshot = AfterCycleSnapshot.capture(
            task_id=tid,
            cycle_index=index,
            max_cycles=task.max_cycles,
            cycle=raw.cycles[-1],
            messages=driver.transcript(),
            shared_state=shared,
            cumulative_token_usage=raw.token_usage,
            available_tool_names=[s["function"]["name"] for s in driver.state.turns[tid].start._payload["definition"]["tools"]],
            disallowed_tool_names=list(
                dict.fromkeys([*task.metadata.get("_vv_agent_disallowed_tools", []), *read_after_cycle_disallowed_tools(shared)])
            ),
            native_outcome=native,
        )
        decision = manager.apply(snapshot)
        persist_after_cycle_disallowed_tools(shared, decision.disallow_tools)
        if decision.action == AfterCycleAction.STOP_NON_SUCCESS:
            assert decision.stop is not None
            error = f"{decision.stop.code}: {decision.stop.message}"
        elif decision.action == AfterCycleAction.STEER and not native.steer_allowed:
            error = "after_cycle_steer_unavailable: after-cycle steering is unavailable at this boundary"
    except AfterCycleHookError as exc:
        error = f"{exc.code}: {exc}"
    driver.commit(
        [
            driver.boundary_record(
                "after_cycle",
                oid,
                {
                    **asdict(decision),
                    "steering_messages": list(decision.steering_messages),
                    "disallow_tools": list(decision.disallow_tools),
                    "shared_state": driver.runtime.durable_state(shared),
                    "error": error,
                },
                source_operation_id=oid,
            )
        ],
        guarded=True,
    )
    return True
