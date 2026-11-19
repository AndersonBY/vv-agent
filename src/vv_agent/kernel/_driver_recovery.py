"""Actual recovery/termination control flow over the shared core phases and ledger."""

from __future__ import annotations

from collections.abc import Generator
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, cast

from . import _after_cycle as cycle
from . import _compaction_phase as compaction
from . import _driver_inputs as inputs
from . import _driver_outcomes as outcomes
from . import _driver_planning as planning
from . import _output_phase as output
from ._boundary_hydration import hydrate_record
from ._content_publication import make_retained_record_before_publication
from ._context import protected_start
from ._driver import CoreDriver
from ._driver_requests import CommitRecords, CreateBoundary, PlanModel
from ._environment_values import (
    InvalidLookupBinding,
    InvalidProviderLookupResult,
    LookupBinding,
    LookupStatus,
    ProviderLookupResult,
)
from ._execution_values import AgentTask
from ._hydration import KeyUnavailable
from ._message_values import Message
from ._provider_outcomes import (
    Accepted,
    Definitive,
    ProviderResultInvalid,
    validate_acceptance,
    validate_lookup_provider_handle,
    validate_provider_result,
    validate_reference_provider_handle,
)
from ._provider_outcomes import (
    Unknown as ProviderUnknown,
)
from ._record_codec import RecordError
from ._records import InboxItem, Record, copy_json, digest
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
    handle: dict[str, Any] | LookupBinding
    timeout: float


@dataclass(frozen=True, slots=True)
class CommitStopping:
    records: list[Record]
    turn_id: str
    stopping: bool


@dataclass(frozen=True, slots=True)
class PushProviderEvidence:
    item: InboxItem


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
    task = driver.hydrated_definition(start).task
    return task if not copy else deepcopy(task)


def stop_unknown(record: Record, *, observation: dict[str, Any] | None = None) -> Record:
    """Keep an unknown execution pinned to stop while retaining recovery evidence."""

    payload = copy_json(record._payload)
    retained = dict(payload["observation"])
    if observation:
        retained.update(observation)
    retained["retryable"] = False
    payload["observation"] = retained
    payload["retry"] = "stop"
    payload["retry_at_ms"] = None
    return replace(record, payload=payload)


def _committed_lookup_binding(driver: CoreDriver, plan: Record) -> LookupBinding | None:
    """Build the single authenticated lookup binding from committed execution data."""

    if plan.feature_level != 3 or plan._payload.get("op_kind") != "tool" or not planning.is_w3_command_plan(driver, plan):
        return None
    try:
        identity = plan._payload["execution_identity"]
        tenant_id = driver.tenant_id
        operation_id = plan.operation_id
        attempt = plan.attempt
        if (
            type(identity) is not dict
            or not isinstance(tenant_id, str)
            or not tenant_id
            or operation_id is None
            or attempt is None
            or identity.get("session_id") != plan.session_id
            or identity.get("operation_id") != operation_id
            or identity.get("tenant_id") != tenant_id
            or plan.turn_id is None
        ):
            return None
        environment = driver.state.turns[plan.turn_id].start._payload["environment"]
        if type(environment) is not dict or identity.get("environment_id") != environment.get("environment_id"):
            return None
        return LookupBinding(
            tenant_id=tenant_id,
            session_id=plan.session_id,
            operation_id=operation_id,
            attempt=attempt,
            request_digest=plan._payload["request_digest"],
            idempotency_key=plan._payload["idempotency_key"],
            execution_id=identity["execution_id"],
            command_digest=plan._payload["command_digest"],
            environment_id=environment["environment_id"],
            generation=environment["generation"],
            provider_id=plan._payload["provider_binding"],
        )
    except (InvalidLookupBinding, KeyError, TypeError, AssertionError):
        return None


def _lookup_evidence(driver: CoreDriver, plan: Record, lookup: ProviderLookupResult) -> InboxItem:
    if lookup.feature_level != 3 or plan.operation_id is None or plan.attempt is None or plan.turn_id is None:
        raise ValueError("provider lookup evidence is not persistable")
    binding = lookup.binding
    outcome = lookup.outcome
    if isinstance(outcome, Accepted):
        encoded_outcome = {
            "kind": "Accepted",
            "handle": outcome.handle,
            "yield_on_steer": outcome.yield_on_steer,
        }
    elif isinstance(outcome, Definitive):
        encoded_outcome = {
            "kind": "Definitive",
            "result": outcome.result,
            "usage": outcome.usage,
            "shared_state": outcome.shared_state,
            "evidence": list(outcome.evidence),
        }
    elif isinstance(outcome, ProviderUnknown):
        encoded_outcome = {"kind": "Unknown", "reason": outcome.reason}
    elif outcome is None:
        encoded_outcome = None
    else:
        raise ValueError("provider lookup outcome is not persistable")
    provider_lookup = {
        "schema_version": "vv-agent.provider-lookup.v1",
        "status": lookup.status.value,
        "binding": binding.to_dict(),
        "outcome": encoded_outcome,
        "evidence": list(lookup.evidence),
    }
    input_id = inputs.provider_recovery_lookup_input_id(
        plan.operation_id,
        plan.attempt,
        plan._payload["request_digest"],
        provider_lookup,
    )
    return InboxItem(
        input_id,
        "provider_evidence",
        {
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan._payload["request_digest"],
            "provider_lookup": provider_lookup,
        },
        target_turn_id=plan.turn_id,
        generation=driver.state.turns[plan.turn_id].start._payload["generation"],
        feature_level=3,
    )


def _lookup_recovery_unknown(
    driver: CoreDriver,
    attempt: Attempt,
    plan: Record,
    binding: LookupBinding,
):
    def push_evidence(lookup: ProviderLookupResult):
        receipt = yield PushProviderEvidence(_lookup_evidence(driver, plan, lookup))
        if not receipt.replayed:
            return None
        unknown = yield from outcomes.unknown_steps(
            driver, attempt.execution_plan, "provider lookup evidence was already consumed"
        )
        return stop_unknown(
            unknown,
            observation={
                "code": "provider_lookup_replayed",
                "lookup_status": lookup.status.value,
                "lookup_evidence": list(lookup.evidence),
                "evidence_input_id": receipt.input_id,
                "evidence_replayed": True,
            },
        )

    try:
        yield SelectProvider(plan)
        lookup = yield ProviderCall("lookup", binding, 1)
    except TimeoutError:
        status, evidence = "unknown", ("provider-lookup-timeout",)
    except InvalidProviderLookupResult:
        status, evidence = "invalid", ("provider-lookup-result-invalid",)
    except Exception:
        status, evidence = "unknown", ("provider-lookup-error",)
    else:
        if isinstance(lookup, ProviderLookupResult):
            if lookup.status is LookupStatus.NOT_STARTED:
                if attempt.plan.attempt == 1:
                    replayed = yield from push_evidence(lookup)
                    if replayed is not None:
                        return replayed
                    return None
            elif lookup.status is LookupStatus.EXITED:
                try:
                    validate_provider_result(plan, lookup.outcome)
                except ProviderResultInvalid:
                    status, evidence = "invalid", ("provider-lookup-result-invalid",)
                else:
                    replayed = yield from push_evidence(lookup)
                    if replayed is not None:
                        return replayed
                    return None
            elif lookup.status is LookupStatus.RUNNING:
                replayed = yield from push_evidence(lookup)
                if replayed is not None:
                    return replayed
                return None
            status, evidence = lookup.status.value, lookup.evidence
        elif isinstance(lookup, ProviderUnknown):
            status, evidence = "unknown", ("provider-lookup-dispatch-unknown",)
        else:
            status, evidence = "invalid", ("provider-lookup-result-invalid",)
    unknown = yield from outcomes.unknown_steps(driver, attempt.execution_plan, "worker lost before durable result")
    return stop_unknown(
        unknown,
        observation={
            "code": "provider_lookup_recovery_unavailable",
            "lookup_status": status,
            "lookup_evidence": list(evidence),
        },
    )


def _completed_values(driver: CoreDriver, record: Record) -> tuple[dict[str, Any], tuple[str, ...], dict[str, Any] | None]:
    result = driver.hydrate_record_field(record, "result", record._payload["result"])
    evidence = driver.hydrate_record_field(record, "evidence", record._payload["evidence"])
    usage = driver.hydrate_record_field(record, "usage", record._payload["usage"])
    if type(result) is not dict or type(evidence) is not list or not all(isinstance(item, str) for item in evidence):
        raise RecordError("completed result/evidence hydration has the wrong JSON shape")
    if usage is not None and type(usage) is not dict:
        raise RecordError("completed usage hydration has the wrong JSON shape")
    return result, tuple(evidence), usage


def _result(driver: CoreDriver, record: Record) -> dict[str, Any]:
    value = driver.hydrate_record_field(record, "result", record._payload["result"])
    if type(value) is not dict:
        raise RecordError("completed result hydration must return an object")
    return value


def _hydrated_wait(driver: CoreDriver, attempt: Attempt) -> dict[str, Any]:
    assert attempt.plan.operation_id is not None and attempt.plan.attempt is not None and attempt.wait is not None
    return driver.hydrate_parked_wait(attempt.plan.operation_id, attempt.plan.attempt, attempt.wait)


def _provider_operation_handle(
    driver: CoreDriver,
    plan: Record,
    handle: dict[str, Any],
) -> dict[str, Any]:
    """Restore the W3 identity needed to address an already parked execution."""

    retained_handle = next(
        (
            stored.record._payload["handle"]
            for stored in reversed(driver.records)
            if stored.record.kind == "op_parked"
            and stored.record.operation_id == plan.operation_id
            and stored.record.attempt == plan.attempt
            and stored.record._payload["handle"]["kind"] == "provider"
        ),
        None,
    )
    if retained_handle is None:
        raise ProviderResultInvalid()
    validate_reference_provider_handle(plan, retained_handle)
    if plan.feature_level != 3:
        return handle
    binding = _committed_lookup_binding(driver, plan)
    if binding is None:
        raise ProviderResultInvalid()
    rehydrated = handle | {
        "execution_id": binding.execution_id,
        "command_digest": binding.command_digest,
        "environment_id": binding.environment_id,
        "generation": binding.generation,
    }
    return rehydrated


def _retained_budget(driver: CoreDriver, tid: str):
    value = retained_budget_inputs(driver.fold.state, tid, hydrator=driver)
    return retained_evaluator(value) if value is not None else None


def _definition_matches(driver, start):
    definition = driver.hydrated_definition(start)
    facts = yield Definition(definition.task)
    frozen = {
        "agent_name": definition.agent_name,
        "memory_settings": definition.memory_settings,
        "child_tools": list(definition.child_tools),
        "model_binding": definition.model_binding,
        "capabilities": definition.capabilities,
        "task": definition.task.to_dict(),
        "tools": list(definition.tools),
    }
    current = facts.definitions.definition_digest(definition.task, facts.bindings, facts.catalog)
    return current == digest(frozen)


def _compact(driver, *, requested_by=None):
    retries = yield Read.MAX_PTL_RETRIES
    return (yield from _compose(driver, compaction.compact_context(driver, max_ptl_retries=retries, requested_by=requested_by)))


def _after_cycle(driver):
    configured = yield Read.START_AFTER_CYCLE
    return (yield from _compose(driver, cycle.after_cycle(driver, has_hooks=configured)))


def _output(driver, value):
    yield Read.START_OUTPUT
    return (yield from _compose(driver, output.prepare_output(driver, value)))


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

                    response = yield from _compose(
                        driver, finalize_tool(driver, driver.hydrate_plan(request.plan), request.outcome)
                    )
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
    plan = driver.hydrate_plan(attempt.execution_plan)
    assert plan.turn_id is not None
    wait = _hydrated_wait(driver, attempt)
    handle = wait["handle"]
    has_provider, has_broker = yield BeginApproval(plan, handle)
    deadline = wait["deadline_ms"]
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
            and (yield from _definition_matches(self, self.state.turns[self.state.active_turn_id].start))
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
        task = self.hydrated_definition(turn.start).task
        if task.metadata.get("vv_session", {}).get("host_binding_names"):
            yield BindHostState(task)
        if task.metadata.get("vv_session", {}).get("input_blocked"):
            yield from close(self, "failed", "agent_failed", task.metadata.setdefault("vv_session", {})["input_blocked"])
            return True
        if not (yield from _definition_matches(self, turn.start)):
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
            result_value = _result(self, result) if result is not None else None
            dependencies = op.attempts[1].plan._payload["dependencies"]
            decision = self.boundary("after_cycle", dependencies[0]) if dependencies else None
            if decision and decision._payload["data"]["action"] == "steer":
                continue
            if result is not None and result_value and result_value.get("directive") == ToolDirective.FINISH.value:
                if (yield Read.AFTER_CYCLE_CONFIGURED) and decision is None:
                    continue
                repair = self.state.operations.get(f"{tid}/model/output_repair/1")
                if repair and repair.state in {"planned", "started", "parked"}:
                    continue
                transfer_failed = (
                    result_value.get("metadata", {}).get("mode") == "handoff" and result_value.get("status_code") == "ERROR"
                )
                yield from close(
                    self,
                    "failed" if transfer_failed else "completed",
                    result_value.get("error_code")
                    if transfer_failed
                    else result_value.get("metadata", {}).get("completion_reason", "tool_finish"),
                    result_value.get("metadata", {}).get("final_message", result_value["content"]),
                    transferred=result_value.get("metadata", {}).get("mode") == "handoff",
                )
                return True
            if result is not None and result_value and result_value.get("directive") == ToolDirective.WAIT_USER.value:
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
                        pending_plan = self.hydrate_plan(pending.execution_plan)
                        value = build_skipped_result(
                            ToolCall.from_dict(pending_plan.payload["request"]),
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
                                "question": result_value.get("metadata", {}).get("question", result_value["content"]),
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
                    result, evidence, usage = _completed_values(self, known.result)
                    repairs = yield from _compose(
                        self,
                        outcomes.completion_effects_steps(
                            self,
                            known.plan,
                            Definitive(result, evidence, cast(dict[str, Any], usage)),
                            known.context,
                        ),
                    )
                    if repairs:
                        yield CommitRecords(repairs)
                        return True
            a = op.attempts[max(op.attempts)]
            rejected = None
            if op.kind == "model" and a.result is not None:
                rejected = _result(self, a.result)
            if (
                rejected
                and (rejected.get("error_code") == "model_request_rejected")
                and (a.plan._payload["purpose"] != "compaction")
            ):
                yield from close(self, "failed", "model_request_rejected", rejected["content"])
                return True
            if a.state == "started":
                plan = self.hydrate_plan(a.execution_plan)
                binding = _committed_lookup_binding(self, plan)
                if binding is not None:
                    unknown = yield from _compose(self, _lookup_recovery_unknown(self, a, plan, binding))
                else:
                    unknown = yield from _compose(
                        self, outcomes.unknown_steps(self, a.execution_plan, "worker lost before durable result")
                    )
                if unknown is not None:
                    yield CommitRecords([unknown])
                return True
            if a.state == "unknown" and a.unknown:
                if a.plan.feature_level != 3 and a.unknown._payload["retry"] == "retry":
                    now = (yield outcomes.OutcomeRead.POLL_LEASE).db_now_ms
                    due = a.unknown._payload["retry_at_ms"] or 0
                    if due > now:
                        return False
                    payload = a.plan._payload
                    if a.prepared is not None:
                        payload = hydrate_record(self, a.plan)._payload
                    next_plan = make_retained_record_before_publication(
                        "op_planned",
                        session_id=self.sid,
                        turn_id=tid,
                        operation_id=oid,
                        attempt=(a.plan.attempt or 1) + 1,
                        payload=payload | {"not_before_ms": due},
                        feature_level=a.plan.feature_level,
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
                    wait = _hydrated_wait(self, a)
                    handle = wait["handle"]
                    plan = self.hydrate_plan(a.execution_plan)
                    yield SelectProvider(plan)
                    try:
                        query_handle = _provider_operation_handle(self, plan, handle)
                        outcome = yield ProviderCall("query", query_handle, 1)
                        if isinstance(outcome, Definitive):
                            outcomes.check_provider_result(self, a.execution_plan, outcome)
                        elif isinstance(outcome, Accepted):
                            if plan.feature_level == 3:
                                binding = _committed_lookup_binding(self, plan)
                                if binding is None:
                                    raise ProviderResultInvalid()
                                validate_lookup_provider_handle(binding, outcome.handle, feature_level=3)
                            else:
                                validate_acceptance(plan, outcome)
                            if outcome.handle != query_handle or outcome.yield_on_steer != wait.get("yield_on_steer", False):
                                raise ProviderResultInvalid()
                        elif plan.feature_level == 3 and not isinstance(outcome, ProviderUnknown):
                            raise ProviderResultInvalid()
                    except ProviderResultInvalid as exc:
                        exc._query_operation_key = key
                        raise
                    if isinstance(outcome, Definitive):
                        yield CommitRecords(
                            (yield from _compose(self, outcomes.completed_steps(self, a.execution_plan, outcome)))
                        )
                        return True
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
                result_value = _result(self, a.result)
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
                                or self.hydrate_record_field(r.record, "input_reason", r.record._payload["reason"])
                                == "background child notification"
                            )
                        )
                        or (r.record.kind == "op_completed" and r.record._payload["context"] == "correction")
                    )
                    for r in self.records
                )
                cycle_decision = self.boundary("after_cycle", a.plan.operation_id or "")
                if (
                    not result_value.get("tool_calls")
                    and (not result_value.get("error_code"))
                    and (not changed)
                    and (not (cycle_decision and cycle_decision._payload["data"]["action"] == "steer"))
                ):
                    if task.no_tool_policy == "finish":
                        yield from close(self, "completed", None, result_value["content"])
                        return True
                    if task.no_tool_policy == "wait_user":
                        yield CommitRecords(
                            [
                                self.record(
                                    "turn_parked",
                                    {
                                        "interaction_id": f"interaction/{a.plan.operation_id}/{a.plan.attempt}",
                                        "question": result_value["content"],
                                        "source_operation_id": a.plan.operation_id,
                                        "source_attempt": a.plan.attempt,
                                    },
                                )
                            ]
                        )
                        return True
        cycles = sum(
            (
                not (a.result and _result(self, a.result).get("error_code") == "prompt_too_long")
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
    except KeyUnavailable:
        return False
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
                    completion_plan = self.hydrate_plan(a.execution_plan)
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
                                                ToolCall.from_dict(completion_plan.payload["request"]),
                                                error_code="skipped_due_to_finish",
                                                message="Tool skipped because a previous tool finished the task.",
                                            )
                                            if status == "completed" and op.kind != "model"
                                            else ToolExecutionResult(
                                                tool_call_id=completion_plan._payload["request"].get("id", "model"),
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
                        wait = _hydrated_wait(self, a)
                        handle = wait["handle"]
                        plan = self.hydrate_plan(a.execution_plan)
                        yield SelectProvider(plan)
                        try:
                            cancel_handle = _provider_operation_handle(self, plan, handle)
                            outcome = yield ProviderCall("cancel", cancel_handle, 1)
                        except ProviderResultInvalid:
                            if can_be_unknown:
                                records.append(
                                    (
                                        yield from _compose(
                                            self, outcomes.unknown_steps(self, a.execution_plan, "provider_result_invalid")
                                        )
                                    )
                                )
                            continue
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
        evaluator = _retained_budget(self, tid)
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
