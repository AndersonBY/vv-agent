"""After-cycle decisions over receipts, with explicit host requests."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import asdict, dataclass
from typing import Any

from ._boundary_hydration import active_task, hydrate_boundary_data
from ._cycle_values import (
    AfterCycleAction,
    AfterCycleDecision,
    AfterCycleHookError,
    AfterCycleSnapshot,
    NativeCycleOutcome,
    NativeCycleOutcomeKind,
    persist_after_cycle_disallowed_tools,
    read_after_cycle_disallowed_tools,
)
from ._driver import CoreDriver
from ._driver_requests import CommitRecords
from ._execution_values import CompletionReason
from ._records import Record
from ._state import ExecutionState
from ._tool_behavior import build_skipped_result
from ._tool_values import ToolCall, ToolDirective


@dataclass(frozen=True, slots=True)
class CompleteSkippedTool:
    plan: Record
    result: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CloseAfterCycle:
    error: str


@dataclass(frozen=True, slots=True)
class ReadCycleProjection:
    turn_id: str


@dataclass(frozen=True, slots=True)
class ReadSharedState:
    pass


@dataclass(frozen=True, slots=True)
class ReadTranscript:
    pass


@dataclass(frozen=True, slots=True)
class ApplyAfterCycleHooks:
    snapshot: AfterCycleSnapshot


@dataclass(frozen=True, slots=True)
class RecordAfterCycle:
    operation_id: str
    data: dict[str, Any]


AfterCycleRequest = (
    CompleteSkippedTool
    | CommitRecords
    | CloseAfterCycle
    | ReadCycleProjection
    | ReadSharedState
    | ReadTranscript
    | ApplyAfterCycleHooks
    | RecordAfterCycle
)


def after_cycle(source: ExecutionState | CoreDriver, *, has_hooks: bool) -> Generator[AfterCycleRequest, Any, bool]:
    if isinstance(source, CoreDriver):
        driver = source
        state = source.state
    else:
        driver = None
        state = source
    if not has_hooks:
        return False
    tid = state.active_turn_id
    assert tid is not None
    models = [
        (oid, op)
        for oid, op in state.operations.items()
        if op.turn_id == tid and op.kind == "model" and op.attempts[1].plan._payload["purpose"] == "primary"
    ]
    if not models:
        return False
    oid, op = models[-1]
    if op.selected_attempt is None:
        return False
    receipt = op.attempts[op.selected_attempt].result
    receipt_result = _completed_result(driver, receipt) if receipt is not None else None
    if receipt_result is None or receipt_result.get("error_code"):
        return False
    recorded = state.boundaries.get((tid, "after_cycle", oid))
    if recorded is not None:
        error = _boundary_data(driver, recorded).get("error")
        if error:
            yield CloseAfterCycle(error)
            return True
        return False
    children = [
        op
        for op in state.operations.values()
        if op.turn_id == tid and op.kind != "model" and op.attempts[1].plan._payload["dependencies"] == [oid]
    ]
    finish = next(
        (
            op
            for op in children
            if op.selected_attempt is not None
            and (receipt_value := op.attempts[op.selected_attempt].result) is not None
            and _completed_result(driver, receipt_value).get("directive") == ToolDirective.FINISH.value
        ),
        None,
    )
    if finish:
        skipped = []
        for child in children:
            a = child.attempts[max(child.attempts)]
            if a.state == "planned":
                value = build_skipped_result(
                    ToolCall.from_dict(_plan(driver, a.execution_plan).payload["request"]),
                    error_code="skipped_due_to_finish",
                    message="Tool skipped because a previous tool finished the task.",
                )
                skipped.extend((yield CompleteSkippedTool(a.execution_plan, value.to_dict())))
        if skipped:
            yield CommitRecords(skipped)
            return True
    waits = [
        a.wait
        for child in children
        for a in child.attempts.values()
        if a.wait and a.wait["handle"]["kind"] == "user" and "interaction_result" in a.wait
    ]
    if any(op.state != "completed" and not any(a.wait in waits for a in op.attempts.values()) for op in children):
        return False
    task = active_task(driver) if driver is not None else state.turns[tid].start.task()
    primary_plan = _plan(driver, op.attempts[1].plan)
    index = primary_plan._payload["request"]["metadata"].get("vv_session", {}).get("cycle_index", int(oid.rsplit("/", 1)[1]))
    if waits:
        native = NativeCycleOutcome(NativeCycleOutcomeKind.WAIT_USER, CompletionReason.WAIT_USER, "ask_user", steer_allowed=False)
    elif finish:
        native = NativeCycleOutcome(
            NativeCycleOutcomeKind.COMPLETED,
            CompletionReason.TOOL_FINISH,
            _plan(driver, finish.attempts[1].plan)._payload["request"]["name"],
            index < task.max_cycles,
        )
    elif not receipt_result.get("tool_calls") and task.no_tool_policy != "continue":
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
    cycles, usage = yield ReadCycleProjection(tid)
    shared = yield ReadSharedState()
    error = None
    decision = AfterCycleDecision.continue_run()
    try:
        snapshot = AfterCycleSnapshot.capture(
            task_id=tid,
            cycle_index=index,
            max_cycles=task.max_cycles,
            cycle=cycles[-1],
            messages=(yield ReadTranscript()),
            shared_state=shared,
            cumulative_token_usage=usage,
            available_tool_names=[
                s["function"]["name"]
                for s in (
                    driver.hydrated_definition(state.turns[tid].start).tools
                    if driver is not None
                    else state.turns[tid].start._payload["definition"]["tools"]
                )
            ],
            disallowed_tool_names=list(
                dict.fromkeys([*task.metadata.get("_vv_agent_disallowed_tools", []), *read_after_cycle_disallowed_tools(shared)])
            ),
            native_outcome=native,
        )
        decision = yield ApplyAfterCycleHooks(snapshot)
        persist_after_cycle_disallowed_tools(shared, decision.disallow_tools)
        if decision.action == AfterCycleAction.STOP_NON_SUCCESS:
            assert decision.stop is not None
            error = f"{decision.stop.code}: {decision.stop.message}"
        elif decision.action == AfterCycleAction.STEER and not native.steer_allowed:
            error = "after_cycle_steer_unavailable: after-cycle steering is unavailable at this boundary"
    except AfterCycleHookError as exc:
        error = f"{exc.code}: {exc}"
    yield RecordAfterCycle(
        oid,
        {
            **asdict(decision),
            "steering_messages": list(decision.steering_messages),
            "disallow_tools": list(decision.disallow_tools),
            "shared_state": shared,
            "error": error,
        },
    )
    return True


def _plan(driver: CoreDriver | None, record: Record) -> Record:
    return driver.hydrate_plan(record) if driver is not None else record


def _completed_result(driver: CoreDriver | None, record: Record) -> dict[str, Any]:
    value = (
        driver.hydrate_record_field(record, "result", record._payload["result"])
        if driver is not None
        else record._payload["result"]
    )
    if type(value) is not dict:
        raise ValueError("completed result hydration must return an object")
    return value


def _boundary_data(driver: CoreDriver | None, record: Record) -> dict[str, Any]:
    if driver is not None:
        return hydrate_boundary_data(driver, record)
    data = record._payload.get("data")
    if type(data) is not dict:
        raise ValueError("boundary data must be an object")
    return data
