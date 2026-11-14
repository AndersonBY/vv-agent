"""Dispatch and tool finalization decisions over the shared driver ledger."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any

from . import _child_delegation as delegation
from . import _driver_outcomes as outcomes
from . import _tool_admission as admission
from ._call_context import CompiledCallFacts
from ._child_store import ChildSession, InvalidChildBatch, create_child
from ._child_summary import child_summary
from ._driver import CoreDriver
from ._driver_planning import plan as plan_steps
from ._execution_values import AgentTask
from ._interactions import InteractionError, user_interaction
from ._provider_outcomes import Accepted, Definitive, ProviderResultInvalid, Unknown, validate_acceptance
from ._records import Record, copy_json, digest, inherited_seed_policies
from ._state import Attempt
from ._store import SequenceConflict, SessionTx
from ._token_count import _StopIterationTransport
from ._tool_behavior import apply_tool_use_behavior, build_skipped_result
from ._tool_values import ToolCall, ToolExecutionResult


class DispatchRead(Enum):
    HAS_HOOKS = auto()
    PREPARE_AFTER_TOOL = auto()
    CONTEXT_CYCLE = auto()
    CONTEXT_SHARED = auto()
    DURABLE_CONTEXT = auto()
    HANDLER_VERSION = auto()
    POLL_LEASE = auto()
    CANCELLED = auto()


@dataclass(frozen=True, slots=True)
class PrepareContext:
    task: AgentTask
    plan: Record
    approved: bool


@dataclass(frozen=True, slots=True)
class FinishContext:
    facts: CompiledCallFacts


@dataclass(frozen=True, slots=True)
class SetContextCycle:
    cycle: int


@dataclass(frozen=True, slots=True)
class BeforeTool:
    task: AgentTask
    call: ToolCall


@dataclass(frozen=True, slots=True)
class BindPreparedState:
    shared: dict[str, Any]
    task: AgentTask


@dataclass(frozen=True, slots=True)
class ApprovalDenied:
    plan: Record
    answer: dict[str, Any]
    action: str


@dataclass(frozen=True, slots=True)
class ApproveSessionCall:
    call: ToolCall


@dataclass(frozen=True, slots=True)
class PrepareModel:
    plan: Record


@dataclass(frozen=True, slots=True)
class SelectProvider:
    plan: Record


@dataclass(frozen=True, slots=True)
class SubmitInteraction:
    plan: Record


@dataclass(frozen=True, slots=True)
class InteractionFailure:
    message: str
    code: str


@dataclass(frozen=True, slots=True)
class ChildBinding:
    name: str | None


@dataclass(frozen=True, slots=True)
class HasChild:
    name: str | None


@dataclass(frozen=True, slots=True)
class HasDelegation:
    name: str | None


@dataclass(frozen=True, slots=True)
class AdmitChildren:
    records: list[Record]
    plan: Record
    summary: str | None
    child_specs: Any


@dataclass(frozen=True, slots=True)
class CallChildAdmission:
    plan: Record
    summary: str | None
    child_specs: Any


@dataclass(frozen=True, slots=True)
class BackgroundAdmissionResult:
    plan: Record
    handles: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class InvokeOperation:
    plan: Record
    started: Record
    task: AgentTask
    has_provider: bool


@dataclass(frozen=True, slots=True)
class DispatchHook:
    point: str
    plan: Record


@dataclass(frozen=True, slots=True)
class CommitDispatch:
    records: list[Record]
    guarded: bool = False


@dataclass(frozen=True, slots=True)
class PrepareAfterTool:
    task: AgentTask
    call: ToolCall


@dataclass(frozen=True, slots=True)
class AfterTool:
    result: ToolExecutionResult


@dataclass(frozen=True, slots=True)
class PreflightHandler:
    plan: Record


@dataclass(frozen=True, slots=True)
class BeginPreflight:
    plan: Record


@dataclass(frozen=True, slots=True)
class PrepareAdmission:
    call: ToolCall
    arguments: dict[str, Any]
    allowed_tool_names: Any


@dataclass(frozen=True, slots=True)
class AdmissionException:
    error: Exception


@dataclass(frozen=True, slots=True)
class FormatAdmissionException:
    call: ToolCall
    error: Exception


@dataclass(frozen=True, slots=True)
class InvokePreflight:
    call: ToolCall
    mark_started: bool


@dataclass(frozen=True, slots=True)
class FinishPreflight:
    call: ToolCall
    result: Any


class EndPreflight(Enum):
    CLEANUP = auto()


def _task(driver: CoreDriver, *, copy: bool = False) -> AgentTask:
    assert driver.state.active_turn_id is not None
    start = driver.state.turns[driver.state.active_turn_id].start
    return start.task() if copy else start._task()


def preflight(plan: Record) -> Generator[object, Any, ToolExecutionResult | None]:
    try:
        blocked = admission.session_preflight_result(plan, handler_managed=(yield PreflightHandler(plan)))
        if blocked is not None:
            return blocked
        names = yield BeginPreflight(plan)
        result = yield from _preflight_admission(plan, names)
        yield EndPreflight.CLEANUP
        return result
    except StopIteration as error:
        raise _StopIterationTransport(error) from error


def _preflight_admission(plan: Record, names: Any) -> Generator[object, Any, ToolExecutionResult | None]:
    try:
        call = ToolCall.from_dict(plan.payload["request"])
        arguments, parse_error = admission._parse_arguments(call.id, call.arguments)
        if parse_error is not None:
            return parse_error
        normalized_call = ToolCall(id=call.id, name=call.name, arguments=arguments, extra_content=call.extra_content)
        executor_present = yield PrepareAdmission(normalized_call, arguments, names)
        if not executor_present:
            result = yield from admission.admit_steps(
                normalized_call, original_call=call, executor_present=False, allowed_tool_names=names
            )
        else:
            try:
                decision = yield from admission.admit_steps(
                    normalized_call, original_call=call, executor_present=True, allowed_tool_names=names
                )
                if isinstance(decision, admission.Dispatch):
                    ready, result = yield InvokePreflight(normalized_call, decision.mark_started)
                    if ready:
                        return None
                else:
                    result = decision
            except Exception as error:
                if isinstance(error, (admission._AdmissionStopIteration, _StopIterationTransport)):
                    error = error.original
                if (yield AdmissionException(error)):
                    raise
                result = yield FormatAdmissionException(normalized_call, error)
        return (yield FinishPreflight(normalized_call, result))
    except StopIteration as error:
        raise _StopIterationTransport(error) from error


def finalize_tool(driver: CoreDriver, plan: Record, outcome: Definitive) -> Generator[object, Any, Definitive]:
    try:
        call = ToolCall.from_dict(plan.payload["request"])
        yield DispatchRead.PREPARE_AFTER_TOOL
        yield PrepareAfterTool(_task(driver, copy=bool((yield DispatchRead.HAS_HOOKS))), call)
        result = yield AfterTool(ToolExecutionResult.from_dict(outcome.result))
        if not result.tool_call_id.strip():
            result.tool_call_id = call.id
        stop = apply_tool_use_behavior(task=_task(driver), call=call, result=result)
        if stop is not None:
            result.metadata["completion_reason"] = stop.value
        return replace(outcome, result=result.to_dict(), shared_state=(yield DispatchRead.DURABLE_CONTEXT))
    except StopIteration as error:
        raise _StopIterationTransport(error) from error


def admit_children(driver: CoreDriver, tx: SessionTx, request: AdmitChildren) -> Generator[object, Any, list[Record]]:
    try:
        plan = request.plan
        specs = yield CallChildAdmission(plan, request.summary, request.child_specs)
        if isinstance(specs, ChildSession):
            specs = [specs]
        if not isinstance(specs, Sequence) or not specs or any(not isinstance(c, ChildSession) for c in specs):
            raise InvalidChildBatch("Expected one ChildSession or a nonempty sequence of ChildSession")
        if any(c.background != specs[0].background for c in specs):
            raise InvalidChildBatch("All children in a batch must have the same background flag")
        if any(type(c.yield_on_steer) is not bool or (c.background and c.yield_on_steer) for c in specs):
            raise InvalidChildBatch("Only foreground children may declare yield_on_steer=True")
        assert specs is not None
        generation = driver.state.turns[plan.turn_id or ""].start._payload["generation"]
        inherited = inherited_seed_policies(driver.records[0].record._payload["attributes"])
        inherited_specs = []
        for child in specs:
            attributes = copy_json(child.spec.attributes or {})
            attributes.setdefault("seed", {}).update(inherited)
            inherited_specs.append(replace(child, spec=replace(child.spec, attributes=attributes)))
        handles = [create_child(tx, child, plan, generation) for child in inherited_specs]
        handle = handles[0]
        if len(handles) > 1:
            handle = handle | {"siblings": [{k: h[k] for k in ("session_id", "turn_id", "generation")} for h in handles[1:]]}
        parked = yield from outcomes.parked_steps(driver, plan, handle, after=True)
        yieldable = [c.spec.session_id for c in specs if c.yield_on_steer]
        if yieldable:
            parked = replace(parked, payload=parked._payload | {"yield_on_steer": True, "yieldable_children": yieldable})
        child_admission = specs[0].spec.attributes.get("child_admission") if specs[0].spec.attributes else None
        if child_admission:
            marker = {key: child_admission[key] for key in ("mode", "handoff_count", "max_handoffs")}
            marker["agent_name"] = child_admission["definition"]["agent_name"]
            marker["metadata"] = child_admission["handoff_metadata"]
            parked = replace(parked, payload=parked._payload | {"delegation": marker})
        records = [parked]
        if specs[0].background:
            result = yield BackgroundAdmissionResult(plan, handles)
            finalized = yield from finalize_tool(
                driver, plan, Definitive(result.to_dict(), tuple(f"session/{h['session_id']}/created" for h in handles))
            )
            records.extend((yield from outcomes.completed_steps(driver, plan, finalized)))
            records[-1] = replace(records[-1], payload=records[-1]._payload | {"execution_started": True})
        return records
    except StopIteration as error:
        raise _StopIterationTransport(error) from error


def dispatch(driver: CoreDriver, attempt: Attempt) -> Generator[object, Any, None]:
    try:
        plan = attempt.execution_plan
        kind = plan._payload["op_kind"]
        assert driver.state.active_turn_id is not None
        task = _task(driver) if kind == "model" and not (yield DispatchRead.HAS_HOOKS) else _task(driver, copy=True)
        yield PrepareContext(task, plan, attempt.approval in {"approve", "allow_session"})
        metadata = yield FinishContext(CompiledCallFacts(task, plan))
        metadata["fence_epoch"] = driver.scope.lease.epoch
        if kind != "model":
            parent = driver.state.operations[plan._payload["dependencies"][0]]
            source = parent.attempts[parent.selected_attempt or max(parent.attempts)].plan
            metadata["session_tool_names"] = [s["function"]["name"] for s in source._payload["request"]["tools"]]
            yield SetContextCycle(
                source._payload["request"]
                .get("metadata", {})
                .get("vv_session", {})
                .get("cycle_index", (yield DispatchRead.CONTEXT_CYCLE))
            )
        if kind != "model" and (yield DispatchRead.HAS_HOOKS) and attempt.prepared is None:
            patched, short = yield BeforeTool(task, ToolCall.from_dict(plan.payload["request"]))
            capabilities = driver.state.turns[plan.turn_id or ""].start._payload["definition"]["capabilities"]
            prepared = yield from plan_steps(
                driver,
                plan.operation_id or "",
                patched.to_dict(),
                "interaction" if patched.name == "ask_user" else "tool",
                dependencies=plan._payload["dependencies"],
                capability=capabilities.get(patched.name, {}),
            )
            yield CommitDispatch(
                [
                    driver.record(
                        "op_prepared",
                        {
                            "request": patched.to_dict(),
                            "request_digest": digest(patched.to_dict()),
                            "tool": prepared._payload["tool"],
                            "op_kind": prepared._payload["op_kind"],
                            "provider_binding": prepared._payload["provider_binding"],
                            "idempotency_key": prepared._payload["idempotency_key"],
                            "hook_result": short.to_dict() if short else None,
                            "shared_state": (yield DispatchRead.DURABLE_CONTEXT),
                        },
                        plan,
                    )
                ],
                guarded=True,
            )
            return
        if attempt.prepared is not None:
            shared = yield DispatchRead.CONTEXT_SHARED
            shared.clear()
            bound = yield BindPreparedState(copy_json(plan._payload["budget_admission"]["shared_state"]), task)
            shared.update(bound)
        hook_result = plan._payload["budget_admission"].get("hook_result")
        if hook_result is not None:
            final = yield from finalize_tool(driver, plan, Definitive(hook_result))
            yield CommitDispatch((yield from outcomes.completed_steps(driver, plan, final)), guarded=True)
            return
        if attempt.approval in {"deny", "timeout"}:
            assert attempt.approval_answer is not None
            answer = attempt.approval_answer._payload["input"]["payload"]
            result = yield ApprovalDenied(plan, answer, attempt.approval)
            final = yield from finalize_tool(driver, plan, Definitive(result.to_dict(), ()))
            yield CommitDispatch((yield from outcomes.completed_steps(driver, plan, final)))
            return
        if kind != "model":
            name = plan._payload["request"]["name"]
            if any(
                r._payload["disposition"] == "applied"
                and r._payload["input"]["kind"] == "approval_answer"
                and r._payload["input"]["payload"]["decision"] == "allow_session"
                and name in r._payload["input"]["payload"]["scope"]
                for r in driver.state.applied_inputs.values()
            ):
                yield ApproveSessionCall(ToolCall.from_dict(plan.payload["request"]))
            result = yield from preflight(plan)
            if result is not None:
                if result.error_code == "tool_approval_required":
                    handle = {
                        "kind": "approval",
                        "request_id": f"approval/{plan.operation_id}/{plan.attempt}",
                        "request_digest": plan._payload["request_digest"],
                        "scope": [plan._payload["request"]["name"]],
                    }
                    yield CommitDispatch([(yield from outcomes.parked_steps(driver, plan, handle, after=False))], guarded=True)
                else:
                    final = yield from finalize_tool(driver, plan, Definitive(result.to_dict(), ()))
                    yield CommitDispatch((yield from outcomes.completed_steps(driver, plan, final)), guarded=True)
                return
        if kind == "model" and not (yield PrepareModel(plan)):
            yield CommitDispatch(
                (
                    yield from outcomes.completed_steps(
                        driver, plan, Definitive({"content": "", "tool_calls": [], "error_code": "endpoint_unavailable"}, ())
                    )
                ),
                guarded=True,
            )
            return
        evaluator = yield from outcomes.live_budget_steps(driver)
        exhaustion = None
        budget_records = []
        if evaluator:
            exhaustion = (
                (
                    yield from (
                        evaluator.cycle_start_steps()
                        if plan._payload["purpose"] == "primary"
                        else evaluator.model_call_start_steps()
                    )
                )
                if kind == "model"
                else (yield from evaluator.run_start_steps())
            )
            if driver.boundary("budget", f"{plan.operation_id}/start/{plan.attempt}") is None:
                budget_records = [
                    outcomes.budget_record(
                        driver, f"{plan.operation_id}/start/{plan.attempt}", evaluator, exhaustion, source=plan.operation_id
                    )
                ]
        if exhaustion:
            yield CommitDispatch(budget_records, guarded=True)
            from ._driver_recovery import close

            yield from close(driver, "failed", "budget_exhausted")
            return
        has_provider = (yield SelectProvider(plan)) if kind != "model" else False
        poll = yield DispatchRead.POLL_LEASE
        if poll.inbox_seq != driver.watermark or (yield DispatchRead.CANCELLED):
            raise SequenceConflict("input arrived before dispatch")
        if kind == "interaction" or plan._payload["request"].get("name") == "ask_user":
            outcome = yield SubmitInteraction(plan)
            if not isinstance(outcome, Definitive):
                raise ValueError("ask_user must be a synchronous interaction")
            if outcome.result.get("directive") != "wait_user":
                final = yield from finalize_tool(driver, plan, outcome)
                yield CommitDispatch((yield from outcomes.completed_steps(driver, plan, final)), guarded=True)
                return
            question = outcome.result.get("metadata", {}).get("question", outcome.result["content"])
            parked = yield from outcomes.parked_steps(
                driver,
                plan,
                {"kind": "user", "interaction_id": f"interaction/{plan.operation_id}", "question": question},
                after=False,
            )
            parked = replace(parked, payload=parked._payload | {"interaction_result": outcome.result})
            assert plan.turn_id is not None
            try:
                user_interaction(
                    driver.sid,
                    plan.turn_id,
                    driver.state.turns[plan.turn_id].start._payload["generation"],
                    parked._payload,
                    operation_id=plan.operation_id,
                    attempt=plan.attempt,
                )
            except InteractionError:
                invalid = yield InteractionFailure(
                    "Interaction form exceeds the UTF-8 size or option limit.", "interaction_form_too_large"
                )
                invalid.tool_call_id = plan._payload["request"]["id"]
                final = yield from finalize_tool(driver, plan, Definitive(invalid.to_dict(), ()))
                yield CommitDispatch((yield from outcomes.completed_steps(driver, plan, final)), guarded=True)
                return
            skipped = []
            for oid, op in driver.state.operations.items():
                pending = op.attempts[max(op.attempts)]
                if (
                    oid != plan.operation_id
                    and op.turn_id == plan.turn_id
                    and op.kind != "model"
                    and pending.state == "planned"
                    and pending.plan._payload["dependencies"] == plan._payload["dependencies"]
                ):
                    result = build_skipped_result(
                        ToolCall.from_dict(pending.execution_plan.payload["request"]),
                        error_code="skipped_due_to_wait_user",
                        message="Tool skipped because a previous tool requested user input.",
                    )
                    skipped.extend(
                        (yield from outcomes.completed_steps(driver, pending.execution_plan, Definitive(result.to_dict(), ())))
                    )
            yield CommitDispatch([parked, *skipped], guarded=True)
            return
        started = driver.record(
            "op_started",
            {
                "dispatch_id": f"{plan.operation_id}/{plan.attempt}",
                "authorization_version": (yield DispatchRead.HANDLER_VERSION),
                "epoch": driver.scope.lease.epoch,
                "mode": "sync" if kind == "model" else "provider",
            },
            plan,
        )
        if kind == "model" and plan._payload["request"]["metadata"].get("vv_session", {}).get("endpoint_order"):
            order = plan._payload["request"]["metadata"]["vv_session"]["endpoint_order"]
            started = replace(started, payload=started._payload | {"endpoint_id": order[((plan.attempt or 1) - 1) % len(order)]})
        name = plan._payload["request"].get("name")
        summary = None
        is_summary = yield ChildBinding(name)
        if kind != "model" and is_summary:
            ready, summary = yield from child_summary(driver, plan)
            if not ready:
                return
        if kind != "model" and ((yield HasChild(name)) or (yield HasDelegation(name))):
            child_specs = None if (yield HasChild(name)) else (yield from delegation.assemble(driver.store, plan, task))
            if isinstance(child_specs, ToolExecutionResult):
                child_specs.tool_call_id = plan._payload["request"]["id"]
                final = yield from finalize_tool(driver, plan, Definitive(child_specs.to_dict()))
                yield CommitDispatch((yield from outcomes.completed_steps(driver, plan, final)), guarded=True)
                return
            if child_specs is not None or (yield HasChild(name)):
                assert plan.turn_id is not None
                yield AdmitChildren([*budget_records, started], plan, summary, child_specs)
                return
        yield CommitDispatch([*budget_records, started], guarded=True)
        yield DispatchHook("before_external_call", plan)
        outcome = yield InvokeOperation(plan, started, task, has_provider)
        yield DispatchHook("after_external_call", plan)
        if has_provider and plan._payload["provider_binding"] != "function":
            try:
                if isinstance(outcome, Definitive):
                    outcomes.check_provider_result(driver, plan, outcome)
                elif isinstance(outcome, Accepted):
                    validate_acceptance(plan, outcome)
            except ProviderResultInvalid:
                outcome = Unknown("provider_result_invalid")
        if isinstance(outcome, Definitive):
            if kind != "model":
                outcome = yield from finalize_tool(driver, plan, outcome)
                if plan._payload["provider_binding"] != "function":
                    try:
                        outcomes.check_provider_result(driver, plan, outcome)
                    except ProviderResultInvalid:
                        yield CommitDispatch([(yield from outcomes.unknown_steps(driver, plan, "provider_result_invalid"))])
                        return
            outcome = replace(outcome, shared_state=(yield DispatchRead.DURABLE_CONTEXT))
            yield CommitDispatch((yield from outcomes.completed_steps(driver, plan, outcome)))
        elif isinstance(outcome, Accepted):
            yield CommitDispatch([(yield from outcomes.accepted_steps(driver, plan, outcome))])
        else:
            yield CommitDispatch([(yield from outcomes.unknown_steps(driver, plan, outcome.reason))])
    except StopIteration as error:
        raise _StopIterationTransport(error) from error
