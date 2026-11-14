"""Actual recovery/termination control flow over the shared core phases and ledger."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from . import _after_cycle as cycle
from . import _compaction_phase as compaction
from . import _driver_inputs as inputs
from . import _driver_outcomes as outcomes
from . import _driver_planning as planning
from . import _output_phase as output
from ._context import protected_start
from ._driver import CoreDriver
from ._driver_requests import CommitRecords, CreateBoundary, PlanModel
from ._execution_values import AgentTask
from ._message_values import Message
from ._provider_outcomes import Accepted, Definitive, ProviderResultInvalid, validate_acceptance
from ._records import InboxItem, Record, copy_json, digest, make_record
from ._retained_budget import retained_budget_inputs, retained_evaluator
from ._state import Attempt
from ._summary_requests import pending_compaction_requests, summary_pending
from ._token_count import _StopIterationTransport
from ._tool_behavior import build_skipped_result
from ._tool_values import ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus


class Read(Enum):
    REFRESH = auto()
    CANCELLED = auto()
    RESET_TOKEN = auto()
    HANDLER_VERSION = auto()
    AFTER_CYCLE_CONFIGURED = auto()
    START_AFTER_CYCLE = auto()
    START_OUTPUT = auto()
    MEMORY_HOOKS = auto()
    MAX_PTL_RETRIES = auto()
    NEXT_DRIVE_DELAY = auto()


@dataclass(frozen=True, slots=True)
class Definition:
    start: Record


@dataclass(frozen=True, slots=True)
class BindHostState:
    task: AgentTask


@dataclass(frozen=True, slots=True)
class BeforeMemory:
    task: AgentTask
    cycle_index: int
    messages: list[Message]
    shared_state: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Hook:
    point: str
    record: Record


@dataclass(frozen=True, slots=True)
class SelectProvider:
    plan: Record


@dataclass(frozen=True, slots=True)
class ProviderCall:
    method: str
    handle: dict[str, Any]
    timeout: float


@dataclass(frozen=True, slots=True)
class CommitStopping:
    records: list[Record]
    turn_id: str
    stopping: bool


class _RecoveryStopIteration(Exception):
    def __init__(self, original: StopIteration):
        self.original = original


class ApprovalRead(Enum):
    DRAIN = auto()
    DISCARD = auto()
    PENDING = auto()
    HAS_ANSWER = auto()
    SHOULD_REQUEST = auto()
    REGISTER = auto()
    DECIDE = auto()
    TIMEOUT = auto()
    ALLOW = auto()


@dataclass(frozen=True, slots=True)
class BeginApproval:
    plan: Record
    handle: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    plan: Record
    handle: dict[str, Any]
    deadline: int | None


@dataclass(frozen=True, slots=True)
class ResolveApproval:
    decision: Any


@dataclass(frozen=True, slots=True)
class PushApprovalAnswer:
    item: InboxItem


def _task(driver: CoreDriver, *, copy: bool = False):
    assert driver.state.active_turn_id is not None
    start = driver.state.turns[driver.state.active_turn_id].start
    return start.task() if copy else start._task()


def _retained_budget(state, tid):
    value = retained_budget_inputs(state, tid)
    return retained_evaluator(value) if value is not None else None


def _definition_digest(driver, start):
    facts = yield Definition(start)
    return facts.definitions.definition_digest(start, facts.bindings, facts.catalog)


def _compact(driver, *, requested_by=None):
    retries = yield Read.MAX_PTL_RETRIES
    return (yield from _compose(driver, compaction.compact_context(driver, max_ptl_retries=retries, requested_by=requested_by)))


def _after_cycle(driver):
    configured = yield Read.START_AFTER_CYCLE
    return (yield from _compose(driver, cycle.after_cycle(driver.state, has_hooks=configured)))


def _output(driver, value):
    yield Read.START_OUTPUT
    return (yield from _compose(driver, output.prepare_output(driver.state, value)))


def _dispatch(driver, attempt):
    from ._driver_dispatch import dispatch

    return (yield from _compose(driver, dispatch(driver, attempt)))


def _compose(driver: CoreDriver, steps):
    """Compose the actual child generator; only concrete host observations escape."""
    response = None
    error = None
    try:
        while True:
            try:
                request = steps.throw(error) if error is not None else steps.send(response)
            except StopIteration as done:
                return done.value
            response, error = None, None
            try:
                if isinstance(request, PlanModel):
                    response = yield from _compose(
                        driver, planning.plan(driver, request.oid, request.request, "model", purpose=request.purpose)
                    )
                elif isinstance(request, CreateBoundary):
                    response = driver.boundary_record(
                        request.stage, request.identity, request.data, source_operation_id=request.source_operation_id
                    )
                elif isinstance(request, compaction.CreateRecord):
                    response = driver.record(request.kind, request.payload)
                elif isinstance(request, (inputs.CompleteResult, cycle.CompleteSkippedTool)):
                    outcome = request.outcome if isinstance(request, inputs.CompleteResult) else Definitive(request.result)
                    response = yield from _compose(driver, outcomes.completed_steps(driver, request.plan, outcome))
                elif isinstance(request, inputs.YieldResults):
                    response = yield from _compose(driver, outcomes.yield_results_steps(driver, request.attempts, request.item))
                elif isinstance(request, inputs.FinalizeTool):
                    from ._driver_dispatch import finalize_tool

                    response = yield from _compose(driver, finalize_tool(driver, request.plan, request.outcome))
                elif isinstance(request, compaction.CloseCompaction):
                    response = yield from close(driver, "failed", request.reason)
                elif isinstance(request, planning.ClosePlanning):
                    response = yield from close(driver, request.status, request.reason)
                elif isinstance(request, cycle.CloseAfterCycle):
                    response = yield from close(driver, "failed", "agent_failed", request.error)
                elif isinstance(request, cycle.RecordAfterCycle):
                    shared = yield planning.DurableState(request.data["shared_state"])
                    record = driver.boundary_record(
                        "after_cycle",
                        request.operation_id,
                        request.data | {"shared_state": shared},
                        source_operation_id=request.operation_id,
                    )
                    yield CommitRecords([record])
                else:
                    response = yield request
            except Exception as exc:
                error = exc
    finally:
        steps.close()


def _approval(driver: CoreDriver, attempt: Attempt):
    assert attempt.wait is not None
    plan = attempt.execution_plan
    assert plan.turn_id is not None
    handle = attempt.wait["handle"]
    has_provider, has_broker = yield BeginApproval(plan, handle)
    deadline = attempt.wait["deadline_ms"]
    decision = None
    if has_broker:
        decision = yield ApprovalRead.DRAIN
    if decision is None and deadline is not None and (yield outcomes.OutcomeRead.POLL_LEASE).db_now_ms >= deadline:
        decision = yield ApprovalRead.TIMEOUT
        if has_broker:
            yield ApprovalRead.DISCARD
    if decision is None and has_provider and has_broker:
        if (yield ApprovalRead.PENDING):
            return False
        yield ApprovalRequest(plan, handle, deadline)
        if not (yield ApprovalRead.SHOULD_REQUEST):
            decision = yield ApprovalRead.ALLOW
        else:
            yield ApprovalRead.REGISTER
            try:
                decision = yield ApprovalRead.DECIDE
            except Exception as exc:
                if isinstance(exc, (_RecoveryStopIteration, _StopIterationTransport)):
                    exc = exc.original
                yield ApprovalRead.DISCARD
                yield from close(driver, "failed", "agent_failed", str(exc))
                return True
            if decision is not None:
                yield ResolveApproval(decision)
            decision = yield ApprovalRead.DRAIN
    if decision is None:
        return bool((yield ApprovalRead.HAS_ANSWER))
    payload = {
        "operation_id": plan.operation_id,
        "attempt": plan.attempt,
        "request_id": handle["request_id"],
        "request_digest": handle["request_digest"],
        "scope": handle["scope"],
        "decision": "approve" if decision.action == "allow" else decision.action,
        "reason": decision.reason,
        "metadata": decision.metadata,
    }
    answer = InboxItem(
        f"approval/{plan.operation_id}/{plan.attempt}/answer/{digest(payload)}",
        "approval_answer",
        payload,
        plan.turn_id,
        driver.state.turns[plan.turn_id].start._payload["generation"],
    )
    yield PushApprovalAnswer(answer)
    return True


def step(self: CoreDriver, *, one_turn: bool) -> Generator[object, Any, bool]:
    try:
        yield Read.REFRESH
        if (
            self.state.active_turn_id
            and (not self.state.cancel_requested)
            and (not (yield Read.CANCELLED))
            and any(
                op.turn_id == self.state.active_turn_id and op.attempts[1].plan._payload["purpose"] == "compaction"
                for op in self.state.operations.values()
            )
            and (self.state.turns[self.state.active_turn_id].start._payload["handler_version"] == (yield Read.HANDLER_VERSION))
            and (
                (yield from _definition_digest(self, self.state.turns[self.state.active_turn_id].start))
                == self.state.turns[self.state.active_turn_id].start._payload["definition_digest"]
            )
            and (yield from _compose(self, compaction.finish_summary(self)))
        ):
            return True
        if (yield from _compose(self, inputs.apply_input(self, one_turn=one_turn))):
            return True
        tid = self.state.active_turn_id
        if tid is None:
            yield compaction.CompactionRequest.SAVE_MEMORY_PROJECTION
            return not self.state.closed and (yield from _compose(self, inputs.start_turn(self)))
        turn = self.state.turns[tid]
        if turn.cancelled:
            actions = [
                r._payload["input"]["payload"]["action"]
                for r in self.state.applied_inputs.values()
                if r.turn_id == tid and r._payload["disposition"] == "applied" and (r._payload["input"]["kind"] == "control")
            ]
            yield from close(self, "aborted" if "abort" in actions else "cancelled", "cancel_requested")
            return True
        if turn.suspended or turn.wait is not None:
            return False
        if (yield Read.CANCELLED):
            yield Read.RESET_TOKEN
        if turn.start._payload["handler_version"] != (yield Read.HANDLER_VERSION):
            yield from close(self, "failed", "handler_version_mismatch")
            return True
        task = turn.start._task()
        if task.metadata.get("vv_session", {}).get("host_binding_names"):
            yield BindHostState(task)
        if task.metadata.get("vv_session", {}).get("input_blocked"):
            yield from close(self, "failed", "agent_failed", task.metadata.setdefault("vv_session", {})["input_blocked"])
            return True
        if (yield from _definition_digest(self, turn.start)) != turn.start._payload["definition_digest"]:
            yield from close(self, "failed", "handler_schema_or_capability_mismatch")
            return True
        evaluator = yield from _compose(self, outcomes.live_budget_steps(self))
        if evaluator and self.boundary("budget", "run_start") is None:
            exhaustion = yield from evaluator.run_start_steps()
            yield CommitRecords([outcomes.budget_record(self, "run_start", evaluator, exhaustion)])
            return True
        for (turn_id, stage, _), recorded in self.state.boundaries.items():
            if turn_id == tid and stage == "budget" and recorded._payload["data"]["exhaustion"]:
                yield from close(self, "failed", "budget_exhausted")
                return True
        if (yield from _after_cycle(self)):
            return True
        operations = [(oid, op) for oid, op in self.state.operations.items() if op.turn_id == tid]
        for _, op in operations:
            if op.kind == "model" or op.selected_attempt is None:
                continue
            result = op.attempts[op.selected_attempt].result
            dependencies = op.attempts[1].plan._payload["dependencies"]
            decision = self.boundary("after_cycle", dependencies[0]) if dependencies else None
            if decision and decision._payload["data"]["action"] == "steer":
                continue
            if result and result._payload["result"].get("directive") == ToolDirective.FINISH.value:
                if (yield Read.AFTER_CYCLE_CONFIGURED) and decision is None:
                    continue
                repair = self.state.operations.get(f"{tid}/model/output_repair/1")
                if repair and repair.state in {"planned", "started", "parked"}:
                    continue
                transfer_failed = (
                    result._payload["result"].get("metadata", {}).get("mode") == "handoff"
                    and result._payload["result"].get("status_code") == "ERROR"
                )
                yield from close(
                    self,
                    "failed" if transfer_failed else "completed",
                    result._payload["result"].get("error_code")
                    if transfer_failed
                    else result._payload["result"].get("metadata", {}).get("completion_reason", "tool_finish"),
                    result._payload["result"].get("metadata", {}).get("final_message", result._payload["result"]["content"]),
                    transferred=result._payload["result"].get("metadata", {}).get("mode") == "handoff",
                )
                return True
            if result and result._payload["result"].get("directive") == ToolDirective.WAIT_USER.value:
                interaction_id = f"interaction/{result.operation_id}/{result.attempt}"
                if any(
                    r._payload["disposition"] == "applied" and r._payload["target_wait_id"] == interaction_id
                    for r in self.state.applied_inputs.values()
                ):
                    continue
                skipped = []
                for other_id, other in operations:
                    pending = other.attempts[max(other.attempts)]
                    if other_id != result.operation_id and pending.state == "planned":
                        value = build_skipped_result(
                            ToolCall.from_dict(pending.execution_plan.payload["request"]),
                            error_code="skipped_due_to_wait_user",
                            message="Tool skipped because a previous tool requested user input.",
                        )
                        skipped.extend(
                            (
                                yield from _compose(
                                    self, outcomes.completed_steps(self, pending.execution_plan, Definitive(value.to_dict()))
                                )
                            )
                        )
                if skipped:
                    yield CommitRecords(skipped)
                    return True
                if any((other.state != "completed" for _, other in operations)):
                    return False
                yield CommitRecords(
                    [
                        self.record(
                            "turn_parked",
                            {
                                "interaction_id": interaction_id,
                                "question": result._payload["result"]
                                .get("metadata", {})
                                .get("question", result._payload["result"]["content"]),
                                "source_operation_id": result.operation_id,
                                "source_attempt": result.attempt,
                            },
                        )
                    ]
                )
                return True
        for oid, op in operations:
            if op.kind == "model" and op.attempts[1].plan._payload["purpose"] == "primary" and (op.selected_attempt is not None):
                known = op.attempts[op.selected_attempt]
                if known.result and known.context == "normal":
                    repairs = yield from _compose(
                        self,
                        outcomes.completion_effects_steps(
                            self,
                            known.plan,
                            Definitive(
                                known.result._payload["result"],
                                tuple(known.result._payload["evidence"]),
                                known.result._payload["usage"],
                            ),
                            known.context,
                        ),
                    )
                    if repairs:
                        yield CommitRecords(repairs)
                        return True
            a = op.attempts[max(op.attempts)]
            if (
                op.kind == "model"
                and a.result
                and (a.result._payload["result"].get("error_code") == "model_request_rejected")
                and (a.plan._payload["purpose"] != "compaction")
            ):
                yield from close(self, "failed", "model_request_rejected", a.result._payload["result"]["content"])
                return True
            if a.state == "started":
                yield CommitRecords(
                    [
                        (
                            yield from _compose(
                                self, outcomes.unknown_steps(self, a.execution_plan, "worker lost before durable result")
                            )
                        )
                    ]
                )
                return True
            if a.state == "unknown" and a.unknown:
                if a.unknown._payload["retry"] == "retry":
                    now = (yield outcomes.OutcomeRead.POLL_LEASE).db_now_ms
                    due = a.unknown._payload["retry_at_ms"] or 0
                    if due > now:
                        return False
                    next_plan = make_record(
                        "op_planned",
                        session_id=self.sid,
                        turn_id=tid,
                        operation_id=oid,
                        attempt=(a.plan.attempt or 1) + 1,
                        payload=a.execution_plan._payload | {"not_before_ms": due},
                    )
                    yield CommitRecords([next_plan])
                    return True
                if op.kind == "model" and a.plan._payload["purpose"] != "compaction":
                    yield from close(self, "failed", "model_outcome_unknown")
                    return True
            if a.state == "planned" or (
                a.state == "parked" and a.approval and a.wait and (a.wait["handle"]["kind"] == "approval")
            ):
                if (a.plan._payload["not_before_ms"] or 0) > (yield outcomes.OutcomeRead.POLL_LEASE).db_now_ms:
                    return False
                if summary_pending(self, oid):
                    continue
                yield from _dispatch(self, a)
                return True
            if a.state == "parked" and a.wait and (a.wait["handle"]["kind"] == "approval"):
                return (yield from _approval(self, a))
            if a.state == "parked" and a.wait and (a.wait["handle"]["kind"] == "user"):
                return False
            if a.state == "parked" and a.wait and (a.wait["handle"]["kind"] == "provider"):
                key = (oid, a.plan.attempt or 1)
                if key not in self.polled:
                    if (yield Read.NEXT_DRIVE_DELAY) != 0:
                        return False
                    self.polled.add(key)
                    handle = copy_json(a.wait["handle"])
                    yield SelectProvider(a.execution_plan)
                    outcome = yield ProviderCall("query", handle, 1)
                    if isinstance(outcome, Definitive):
                        outcomes.check_provider_result(self, a.execution_plan, outcome)
                        yield CommitRecords(
                            (yield from _compose(self, outcomes.completed_steps(self, a.execution_plan, outcome)))
                        )
                        return True
                    if isinstance(outcome, Accepted):
                        validate_acceptance(a.execution_plan, outcome)
                        if outcome.handle != handle or outcome.yield_on_steer != a.wait.get("yield_on_steer", False):
                            raise ProviderResultInvalid()
                return False
        if self.pending_batch():
            return False
        requested = [key for key in pending_compaction_requests(self) if key in self.state.applied_inputs]
        if requested and (yield from _compact(self, requested_by=requested)):
            return True
        if task.metadata.get("_vv_agent_app_server_compaction_only"):
            yield from close(self, "completed", "context_compaction")
            return True
        models = [
            (oid, op) for oid, op in operations if op.kind == "model" and op.attempts[1].plan._payload["purpose"] == "primary"
        ]
        if models:
            _, last = models[-1]
            a = last.attempts[last.selected_attempt or max(last.attempts)]
            if a.result:
                plan_seq = next(r.seq for r in self.records if r.record.record_id == last.attempts[1].plan.record_id)
                changed = any(
                    r.seq > plan_seq
                    and (
                        (
                            r.record.kind == "input_applied"
                            and r.record._payload["disposition"] == "applied"
                            and (r.record.turn_id == tid)
                            and (
                                r.record._payload["input"]["kind"] == "steer"
                                or (r.record._payload["input"]["kind"] == "user" and r.record._payload["target_wait_id"])
                                or r.record._payload["reason"] == "background child notification"
                            )
                        )
                        or (r.record.kind == "op_completed" and r.record._payload["context"] == "correction")
                    )
                    for r in self.records
                )
                cycle_decision = self.boundary("after_cycle", a.plan.operation_id or "")
                if (
                    not a.result._payload["result"].get("tool_calls")
                    and (not a.result._payload["result"].get("error_code"))
                    and (not changed)
                    and (not (cycle_decision and cycle_decision._payload["data"]["action"] == "steer"))
                ):
                    if task.no_tool_policy == "finish":
                        yield from close(self, "completed", None, a.result._payload["result"]["content"])
                        return True
                    if task.no_tool_policy == "wait_user":
                        yield CommitRecords(
                            [
                                self.record(
                                    "turn_parked",
                                    {
                                        "interaction_id": f"interaction/{a.plan.operation_id}/{a.plan.attempt}",
                                        "question": a.result._payload["result"]["content"],
                                        "source_operation_id": a.plan.operation_id,
                                        "source_attempt": a.plan.attempt,
                                    },
                                )
                            ]
                        )
                        return True
        cycles = sum(
            (
                not (a.result and a.result._payload["result"].get("error_code") == "prompt_too_long")
                for _, op in models
                for a in [op.attempts[op.selected_attempt or max(op.attempts)]]
            )
        )
        if cycles >= task.max_cycles:
            yield from close(self, "failed", "max_cycles", "Reached max cycles without finish signal.")
            return True
        hook_id = str(cycles + 1)
        if (yield Read.MEMORY_HOOKS) and self.boundary("before_memory", hook_id) is None:
            shared = yield cycle.ReadSharedState()
            source = yield cycle.ReadTranscript()
            from vv_agent.kernel._context_messages import retained_start

            protection = protected_start(self.records, self.state)
            source_digest = digest([m.to_dict() for m in source])
            protected_source = [Message.from_dict(copy_json(m.to_dict())) for m in source] if protection is not None else []
            replacement = yield BeforeMemory(
                task=_task(self, copy=True), cycle_index=cycles + 1, messages=source, shared_state=shared
            )
            retained_start(protected_source, replacement, protection)
            yield CommitRecords(
                [
                    self.boundary_record(
                        "before_memory",
                        hook_id,
                        {"messages": [m.to_dict() for m in replacement], "shared_state": (yield planning.DurableState(shared))},
                        source_digest=source_digest,
                    )
                ]
            )
            return True
        if (yield from _compact(self, requested_by=pending_compaction_requests(self))):
            return True
        plan = yield from _compose(self, planning.model_plan(self, (yield cycle.ReadTranscript())))
        if plan is not None:
            yield CommitRecords([plan])
        return True
    except StopIteration as error:
        raise _RecoveryStopIteration(error) from error


def close(
    self: CoreDriver, status: str, reason: str | None, result: Any = None, *, transferred: bool = False
) -> Generator[object, Any, None]:
    try:
        tid = self.state.active_turn_id
        assert tid is not None
        evaluator = (yield from _compose(self, outcomes.live_budget_steps(self))) if status == "completed" else None
        exhaustion = (yield from evaluator.terminal_steps()) if evaluator else None
        if exhaustion:
            status, reason = ("failed", "budget_exhausted")
        if status == "completed" and (not transferred) and (not _task(self).metadata.get("_vv_agent_app_server_compaction_only")):
            prepared = yield from _output(self, result)
            if prepared is None:
                return
            status, output_reason, result = prepared
            reason = output_reason or reason
        if evaluator and self.boundary("budget", "terminal") is None:
            yield CommitRecords([outcomes.budget_record(self, "terminal", evaluator, exhaustion)])
        records: list[Record] = []
        for op in self.state.operations.values():
            if op.turn_id != tid:
                continue
            for a in op.attempts.values():
                if a.state != "completed" and (not a.started):
                    records.extend(
                        (
                            yield from _compose(
                                self,
                                outcomes.completed_steps(
                                    self,
                                    a.execution_plan,
                                    Definitive(
                                        (
                                            build_skipped_result(
                                                ToolCall.from_dict(a.execution_plan.payload["request"]),
                                                error_code="skipped_due_to_finish",
                                                message="Tool skipped because a previous tool finished the task.",
                                            )
                                            if status == "completed" and op.kind != "model"
                                            else ToolExecutionResult(
                                                tool_call_id=a.plan._payload["request"].get("id", "model"),
                                                content=reason or "not executed",
                                                status_code=ToolResultStatus.ERROR,
                                                error_code="not_executed",
                                            )
                                        ).to_dict(),
                                        (),
                                    ),
                                ),
                            )
                        )
                    )
                elif a.state in {"started", "parked"}:
                    if a.child_handle and a.state == "parked" and (not self.state.cancel_requested):
                        continue
                    if a.wait and a.wait["handle"]["kind"] == "provider":
                        can_be_unknown = a.state == "started" or (
                            a.state == "parked" and a.started and self.state.turns[tid].cancelled
                        )
                        handle = copy_json(a.wait["handle"])
                        yield SelectProvider(a.execution_plan)
                        outcome = yield ProviderCall("cancel", handle, 1)
                        if isinstance(outcome, Definitive):
                            try:
                                outcomes.check_provider_result(self, a.execution_plan, outcome)
                            except ProviderResultInvalid:
                                if can_be_unknown:
                                    records.append(
                                        (
                                            yield from _compose(
                                                self, outcomes.unknown_steps(self, a.execution_plan, "provider_result_invalid")
                                            )
                                        )
                                    )
                            else:
                                records.extend(
                                    (yield from _compose(self, outcomes.completed_steps(self, a.execution_plan, outcome)))
                                )
                            continue
                        if not can_be_unknown:
                            continue
                    records.append(
                        (yield from _compose(self, outcomes.unknown_steps(self, a.execution_plan, reason or "turn stopped")))
                    )
        stopping = status in {"failed", "cancelled", "aborted"}
        if records or stopping:
            yield CommitStopping(records, tid, stopping)
        evaluator = _retained_budget(self.state, tid)
        terminal = self.record(
            "turn_ended",
            {
                "status": status,
                "reason": reason,
                "result": (yield output.SerializeOutput(result)),
                "adopted_results": [
                    {"operation_id": oid, "attempt": op.selected_attempt}
                    for oid, op in self.state.operations.items()
                    if op.turn_id == tid and op.selected_attempt is not None
                ],
                "budget": evaluator.snapshot().to_dict() if evaluator else {},
                "unconfirmed_operations": list(
                    dict.fromkeys(
                        (
                            oid
                            for oid, op in self.state.operations.items()
                            if op.turn_id == tid
                            for a in op.attempts.values()
                            if a.state != "completed"
                        )
                    )
                ),
            },
        )
        yield Hook("before_turn_end", terminal)
        yield CommitRecords([terminal])
    except StopIteration as error:
        raise _RecoveryStopIteration(error) from error
