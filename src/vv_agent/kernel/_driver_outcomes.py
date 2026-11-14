"""Receipt, unknown and parking decisions over the actual shared driver ledger."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from vv_agent.kernel._budget import BudgetObservation, CoreBudgetEvaluator
from vv_agent.kernel._budget_values import BudgetExhaustion, BudgetUsageSnapshot, RunBudgetLimits
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_planning import PlanningRequest
from vv_agent.kernel._driver_planning import plan as plan_steps
from vv_agent.kernel._provider_outcomes import Accepted, Definitive, validate_acceptance, validate_provider_result
from vv_agent.kernel._records import InboxItem, Record, copy_json, digest, make_record
from vv_agent.kernel._retained_budget import retained_budget_inputs, retained_evaluator
from vv_agent.kernel._state import Attempt, child_handles
from vv_agent.kernel._tool_result_checks import validate_definitive_result, validate_output_body
from vv_agent.kernel._tool_values import ToolDirective, ToolExecutionResult
from vv_agent.kernel._usage import model_usage


class OutcomeRead(Enum):
    CANCELLATION_REASON = "cancellation_reason"
    SCOPE_CANCELLED = "scope_cancelled"
    HOST_CANCELLED = "host_cancelled"
    ACTIVE_INTERVAL_NS = "active_interval_ns"
    BIND_BUDGET_METER = "bind_budget_meter"
    APPROVAL_TIMEOUT = "approval_timeout"
    POLL_LEASE = "poll_lease"
    POLL_INTERVAL = "poll_interval"


@dataclass(frozen=True, slots=True)
class PrepareToolResources:
    plan: Record
    result: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LookupYieldCallback:
    plan: Record
    child: bool


@dataclass(frozen=True, slots=True)
class InvokeYieldCallback:
    callback: Callable[[Record, Any, str], Any]
    plan: Record
    handles: Any


@dataclass(frozen=True, slots=True)
class YieldCallbackError:
    error: Exception


class _OutcomeStopIteration(Exception):
    def __init__(self, original: StopIteration):
        self.original = original


OutcomeRequest = (
    OutcomeRead | PrepareToolResources | LookupYieldCallback | InvokeYieldCallback | BudgetObservation | PlanningRequest
)
MODEL_REJECTION_BACKOFF_SECONDS = 2


def compaction_intent(self: CoreDriver, plan: Record, result: dict[str, Any]) -> dict[str, Any]:
    if plan._payload["op_kind"] == "model" or result.get("metadata", {}).get("request_compaction") is not True:
        return result
    source = self.state.operations[plan._payload["dependencies"][0]]
    source_plan = source.attempts[source.selected_attempt or 1].plan
    return result | {
        "metadata": result["metadata"]
        | {
            "compaction_source": {
                "operation_id": plan.operation_id,
                "context_digest": digest(source_plan._payload["request"]["messages"]),
            }
        }
    }


def check_provider_result(self: CoreDriver, plan: Record, outcome: Definitive) -> None:
    assert plan.operation_id is not None and plan.attempt is not None
    attempt = self.state.operations[plan.operation_id].attempts[plan.attempt]
    handle = attempt.wait["handle"] if attempt.wait and attempt.wait["handle"]["kind"] == "provider" else None
    validate_provider_result(plan, outcome, handle)


def parked_steps(
    self: CoreDriver, plan: Record, handle: dict[str, Any], *, after: bool
) -> Generator[OutcomeRequest, Any, Record]:
    try:
        timeout = (yield OutcomeRead.APPROVAL_TIMEOUT) if handle["kind"] == "approval" else None
        now = (yield OutcomeRead.POLL_LEASE).db_now_ms if handle["kind"] == "provider" or timeout is not None else 0
        return self.record(
            "op_parked",
            {
                "phase": "after_dispatch" if after else "before_dispatch",
                "handle": handle,
                "poll_at_ms": now + (yield OutcomeRead.POLL_INTERVAL) if handle["kind"] == "provider" else None,
                "deadline_ms": now + max(0, int(timeout * 1000)) if timeout is not None else None,
            },
            plan,
        )
    except StopIteration as error:
        raise _OutcomeStopIteration(error) from error


def accepted_steps(self: CoreDriver, plan: Record, outcome: Accepted) -> Generator[OutcomeRequest, Any, Record]:
    try:
        validate_acceptance(plan, outcome)
        parked = yield from parked_steps(self, plan, outcome.handle, after=True)
        if outcome.yield_on_steer:
            parked = replace(parked, payload=parked._payload | {"yield_on_steer": True})
        return parked
    except StopIteration as error:
        raise _OutcomeStopIteration(error) from error


def unknown_steps(self: CoreDriver, plan: Record, reason: str) -> Generator[OutcomeRequest, Any, Record]:
    try:
        assert plan.operation_id is not None and plan.attempt is not None
        attempt = self.state.operations[plan.operation_id].attempts[plan.attempt]
        is_model = plan._payload["op_kind"] == "model"
        retry = (
            plan._payload["purpose"] != "output_repair"
            and plan.attempt
            < max(2, len(plan._payload["request"].get("metadata", {}).get("vv_session", {}).get("endpoint_order", [])))
            and (is_model or (plan._payload["tool"] or {}).get("idempotency") == "supported")
        )
        retry = (
            retry
            and reason != "provider_result_invalid"
            and (yield OutcomeRead.CANCELLATION_REASON) not in {"cancel", "abort", "close", "lease_lost"}
            and not self.state.cancel_requested
        )
        assert attempt.dispatch is not None
        return self.record(
            "op_unknown",
            {
                "reason": reason,
                "dispatch_evidence": [attempt.dispatch.record_id],
                "observation": {
                    "code": "duplicate_model_request_and_cost" if is_model else "tool_outcome_unknown",
                    "retryable": retry,
                    "usage": None,
                    "active_interval_missing": reason == "worker lost before durable result",
                },
                "retry": "retry" if retry else "stop",
                "retry_at_ms": 0 if retry else None,
                "duplicate_cost_risk": is_model,
                "measurement_missing": True,
            },
            plan,
        )
    except StopIteration as error:
        raise _OutcomeStopIteration(error) from error


def live_budget_inputs_steps(
    self: CoreDriver,
) -> Generator[OutcomeRequest, Any, tuple[RunBudgetLimits, BudgetUsageSnapshot] | None]:
    try:
        tid = self.state.active_turn_id
        assert tid is not None
        inputs = retained_budget_inputs(self.state, tid)
        if inputs is None:
            return None
        evaluator = retained_evaluator(inputs)
        current_meter = f"active/{tid}/{self.scope.lease.epoch}"
        saved = self.state.usage_values.get(current_meter)
        retained_ms = saved._payload["usage"].get("elapsed_ms", 0) if saved else 0
        initial = evaluator.snapshot()
        initial = replace(
            initial,
            elapsed_ms=initial.elapsed_ms
            + max(0, ((yield OutcomeRead.ACTIVE_INTERVAL_NS) - self.active_since) // 1000000 - retained_ms),
        )
        return evaluator.limits, initial
    except StopIteration as error:
        raise _OutcomeStopIteration(error) from error


def live_budget_steps(self: CoreDriver) -> Generator[OutcomeRequest, Any, CoreBudgetEvaluator | None]:
    try:
        inputs = yield from live_budget_inputs_steps(self)
        if inputs is None:
            return None
        limits, initial = inputs
        yield OutcomeRead.BIND_BUDGET_METER
        return (yield from CoreBudgetEvaluator.create_steps(limits, initial_usage=initial))
    except StopIteration as error:
        raise _OutcomeStopIteration(error) from error


def budget_record(
    self: CoreDriver,
    identity: str,
    evaluator: CoreBudgetEvaluator,
    exhaustion: BudgetExhaustion | None,
    names: list[str] | None = None,
    source: str | None = None,
) -> Record:
    return self.boundary_record(
        "budget",
        identity,
        {
            "usage": evaluator.snapshot().to_dict(),
            "exhaustion": exhaustion.to_dict() if exhaustion else None,
            "tool_names": names or [],
        },
        source_operation_id=source,
    )


def completed_steps(self: CoreDriver, plan: Record, outcome: Definitive) -> Generator[OutcomeRequest, Any, list[Record]]:
    try:
        result = compaction_intent(self, plan, outcome.result)
        if plan._payload["op_kind"] != "model":
            result = yield PrepareToolResources(plan, result)
        outcome = replace(outcome, result=result)
        assert plan.operation_id is not None and plan.attempt is not None and plan.turn_id is not None
        op = self.state.operations[plan.operation_id]
        attempt = op.attempts[plan.attempt]
        turn = self.state.turns[plan.turn_id]
        audit = turn.cancelled or turn.ended or any(a.started for n, a in op.attempts.items() if n > plan.attempt)
        context = "audit" if audit else "correction" if attempt.unknown_consumed else "normal"
        result = self.record(
            "op_completed",
            {
                "result": outcome.result,
                "result_digest": digest(outcome.result),
                "usage": outcome.usage,
                "shared_state": outcome.shared_state,
                "evidence": list(outcome.evidence),
                "execution_started": attempt.started,
                "context": context,
                "request_digest": plan._payload["request_digest"],
                "provider_binding": plan._payload["provider_binding"],
            },
            plan,
        )
        records = [result, *(yield from completion_effects_steps(self, plan, outcome, context))]
        rejection = outcome.result.get("rejection", {})
        order = plan._payload["request"].get("metadata", {}).get("vv_session", {}).get("endpoint_order", [])
        if (
            op.kind == "model"
            and context == "normal"
            and outcome.result.get("error_code") == "model_request_rejected"
            and rejection.get("retryable") is True
            and plan._payload["purpose"] != "output_repair"
            and plan.attempt < max(2, len(order))
            and not (yield OutcomeRead.SCOPE_CANCELLED)
            and not self.state.cancel_requested
            and not any(r.kind == "boundary_recorded" and r._payload["data"].get("exhaustion") for r in records[1:])
        ):
            delay = (
                0 if rejection.get("status_code") in {401, 403, 404} else MODEL_REJECTION_BACKOFF_SECONDS * 1000 * plan.attempt
            )
            due = (yield OutcomeRead.POLL_LEASE).db_now_ms + delay if delay else 0
            records.append(
                make_record(
                    "op_planned",
                    session_id=self.sid,
                    turn_id=plan.turn_id,
                    operation_id=plan.operation_id,
                    attempt=plan.attempt + 1,
                    payload=plan._payload | {"not_before_ms": due},
                )
            )
        return records
    except StopIteration as error:
        raise _OutcomeStopIteration(error) from error


def completion_effects_steps(
    self: CoreDriver, plan: Record, outcome: Definitive, context: str
) -> Generator[OutcomeRequest, Any, list[Record]]:
    try:
        assert plan.operation_id is not None and plan.attempt is not None and plan.turn_id is not None
        records = []
        op = self.state.operations[plan.operation_id]
        turn = self.state.turns[plan.turn_id]
        if op.kind == "model" and context == "normal":
            if not op.attempts[plan.attempt].started:
                return records
            if any(
                r._payload["data"]["exhaustion"]
                for (tid, stage, _), r in self.state.boundaries.items()
                if tid == plan.turn_id and stage == "budget"
            ):
                return records
            evaluator = yield from live_budget_steps(self)
            if evaluator and self.boundary("budget", f"{plan.operation_id}/complete/{plan.attempt}") is None:
                exhaustion = yield from evaluator.model_call_complete_steps(model_usage(outcome.usage))
                if (yield OutcomeRead.SCOPE_CANCELLED) or (yield OutcomeRead.HOST_CANCELLED):
                    exhaustion = None
                records.append(
                    budget_record(
                        self, f"{plan.operation_id}/complete/{plan.attempt}", evaluator, exhaustion, source=plan.operation_id
                    )
                )
                if exhaustion:
                    return records
                if plan._payload["purpose"] != "primary":
                    return records
                names = [call["name"] for call in outcome.result.get("tool_calls", [])]
                if names:
                    exhaustion = yield from evaluator.preflight_tools_steps(names)
                    records.append(
                        budget_record(
                            self,
                            f"{plan.operation_id}/tool_batch",
                            evaluator,
                            exhaustion,
                            names if exhaustion is None else [],
                            source=plan.operation_id,
                        )
                    )
                    if exhaustion:
                        return records
            if plan._payload["purpose"] != "primary":
                return records
            definition = turn.start._payload["definition"]
            for i, call in enumerate(outcome.result.get("tool_calls", [])):
                oid = f"{plan.operation_id}/attempt/{plan.attempt}/tool/{i}"
                if oid not in self.state.operations:
                    planned = yield from plan_steps(
                        self,
                        oid,
                        call,
                        "interaction" if call["name"] == "ask_user" else "tool",
                        dependencies=[plan.operation_id],
                        capability=definition["capabilities"].get(call["name"], {}),
                    )
                    records.append(planned)
        elif op.kind != "model" and context == "normal":
            evaluator = yield from live_budget_steps(self)
            if evaluator is not None:
                before = evaluator.snapshot()
                exhaustion = yield from evaluator.tool_batch_complete_steps(
                    operation_failed=outcome.result.get("status_code") == "ERROR"
                )
                if (yield OutcomeRead.SCOPE_CANCELLED) or (yield OutcomeRead.HOST_CANCELLED):
                    exhaustion = None
                if evaluator.snapshot() != before or exhaustion:
                    records.append(
                        budget_record(
                            self,
                            f"{plan.operation_id}/tool_complete/{plan.attempt}",
                            evaluator,
                            exhaustion,
                            source=plan.operation_id,
                        )
                    )
        return records
    except StopIteration as error:
        raise _OutcomeStopIteration(error) from error


def yield_results_steps(
    self: CoreDriver, attempts: list[Attempt], item: InboxItem
) -> Generator[OutcomeRequest, Any, list[Record]]:
    try:
        records = []
        for a in attempts:
            plan = a.execution_plan
            assert plan.operation_id is not None and plan.attempt is not None
            assert a.wait is not None
            handle = a.wait["handle"]
            handles = child_handles(handle) if a.child_handle else handle
            callback = yield LookupYieldCallback(plan, bool(a.child_handle))
            metadata: dict[str, Any] = {"yield_reason": "steer"}
            if a.child_handle:
                completed = self.state.child_completions or {}
                handles = [
                    h
                    | {
                        "status": (
                            completed[key]._payload["input"]["payload"]["status"]
                            if (key := (plan.operation_id, plan.attempt, h["session_id"])) in completed
                            else "running"
                        )
                    }
                    | (
                        {k: completed[key]._payload["input"]["payload"][k] for k in ("result", "terminal_seq", "terminal_digest")}
                        if key in completed
                        else {}
                    )
                    for h in handles
                ]
                metadata["children"] = [{"session_id": h["session_id"], "status": h["status"]} for h in handles]
                metadata["running_children"] = [
                    h["session_id"] for h in handles if (plan.operation_id, plan.attempt, h["session_id"]) not in completed
                ]
            result = ToolExecutionResult(
                plan._payload["request"]["id"],
                "Wait interrupted by new input; observed work has not been stopped.",
                metadata=metadata,
            )
            if a.child_handle:
                result.content += "\n" + canonical_json_bytes(
                    {"children": metadata["children"], "running_children": metadata["running_children"]}
                ).decode("utf-8")
            if callback is not None:
                try:
                    rendered = yield InvokeYieldCallback(callback, plan, copy_json(handles))
                    if isinstance(rendered, YieldCallbackError):
                        raise rendered.error
                    validate_definitive_result(rendered)
                    validate_output_body(rendered)
                    if rendered.tool_call_id != result.tool_call_id or rendered.directive != ToolDirective.CONTINUE:
                        raise ValueError("invalid yield result")
                    canonical_json_bytes(rendered.to_dict())
                    result = rendered
                except Exception:
                    result.metadata["yield_callback_failed"] = True
            evidence = (f"steer/{item.input_id}",)
            if not a.child_handle:
                evidence += (handle["evidence"],)
            completed_records = yield from completed_steps(self, plan, Definitive(result.to_dict(), evidence))
            records.extend(
                replace(r, payload=r._payload | {"yielded_by": item.input_id}) if r.kind == "op_completed" else r
                for r in completed_records
            )
        return records

    except StopIteration as error:
        raise _OutcomeStopIteration(error) from error
