"""Inbox disposition and turn admission over the actual core driver ledger."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, cast

from ._boundary_hydration import hydrate_input, hydrate_record
from ._budget_values import RunBudgetLimits
from ._canonical_json import canonical_json_bytes
from ._child_store import child_handles, verify_completion
from ._content_publication import declared_content_ref
from ._definition import DefinitionBindings, DefinitionCache
from ._driver import CoreDriver
from ._execution_values import AgentTask
from ._message_values import Message
from ._provider_outcomes import (
    Accepted,
    Definitive,
    ProviderResultInvalid,
    validate_acceptance,
    validate_provider_result,
    validate_reference_provider_handle,
)
from ._records import InboxItem, Record, _RetainedTools, copy_json, digest
from ._requirements import (
    definition_requirements,
    model_capability_tags,
    normalize_requirements,
)
from ._sql import stored_input_proof
from ._state import Attempt
from ._store import Conflict
from ._summary_requests import compaction_request
from ._token_count import _StopIterationTransport
from ._tool_planner import ToolCatalog
from ._tool_result_checks import provider_result_identity
from ._tool_values import ToolExecutionResult


class InputSetting(Enum):
    POLL_NOW = "poll_now"
    POLL_MS = "poll_ms"
    CANCELLED = "cancelled"
    BUDGET_LIMITS = "budget_limits"
    HANDLER_VERSION = "handler_version"
    RESET_TOKEN = "reset_token"


@dataclass(frozen=True, slots=True)
class CommitInput:
    records: list[Record]
    input_id: str
    additional_input_ids: tuple[str, ...] = ()

    @property
    def input_ids(self) -> tuple[str, ...]:
        return (self.input_id, *self.additional_input_ids)


class _BatchDriverView:
    """Expose an uncommitted logical child completion to existing outcome generation."""

    __slots__ = ("_driver", "content_reader", "fold", "state")

    def __init__(self, driver: CoreDriver, *, content_reader) -> None:
        self._driver = driver
        self.state = driver.state
        self.fold = driver.fold
        self.content_reader = content_reader

    def __getattr__(self, name: str) -> Any:
        return getattr(self._driver, name)


@dataclass(frozen=True, slots=True)
class PublishTurnStarted:
    input_id: str
    tid: str
    definition: dict[str, Any]
    budget: dict[str, Any]
    handler_version: str
    generation: int
    admission_definition_digest: str | None = None
    requirements: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class CompileTurn:
    content: str
    turn_id: str
    seed: dict[str, Any] | None
    control: bool


@dataclass(frozen=True, slots=True)
class CurrentDefinition:
    task: AgentTask


@dataclass(frozen=True, slots=True)
class DefinitionFacts:
    definitions: DefinitionCache
    bindings: DefinitionBindings
    catalog: ToolCatalog


@dataclass(frozen=True, slots=True)
class ResolveModelSelection:
    selection: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AuthenticateProvider:
    item: InboxItem
    retained_item: InboxItem
    plan: Record


@dataclass(frozen=True, slots=True)
class ProviderAuthentication:
    """Expose logical provider fields while retaining the authenticated raw item."""

    item: InboxItem
    retained_item: InboxItem

    def __getattr__(self, name: str) -> Any:
        return getattr(self.item, name)


@dataclass(frozen=True, slots=True)
class ProjectProviderResult:
    plan: Record
    result: dict[str, Any]
    selected_artifact: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class CompleteResult:
    plan: Record
    outcome: Definitive


@dataclass(frozen=True, slots=True)
class PrepareToolContext:
    plan: Record
    include_store: bool


@dataclass(frozen=True, slots=True)
class FinalizeTool:
    plan: Record
    outcome: Definitive


@dataclass(frozen=True, slots=True)
class ChildResult:
    plan: Record
    handles: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class YieldResults:
    attempts: list[Attempt]
    item: InboxItem


def _hydrated_provider_wait(self: CoreDriver, attempt: Attempt) -> dict[str, Any] | None:
    if attempt.wait is None:
        return None
    operation_id = attempt.plan.operation_id
    assert operation_id is not None
    if self.content_reader is not None and getattr(self, "records", None):
        return self.hydrate_parked_wait(operation_id, attempt.plan.attempt or 1, attempt.wait)
    return attempt.wait


def _retained_reply_state(self: CoreDriver):
    """Hydrate only the waits read by inbox reply validation."""

    logical_state = getattr(self, "_logical_state", None)
    if callable(logical_state):
        return logical_state()
    if self.content_reader is None or not getattr(self, "records", None):
        return self.state

    state = self.state
    tid = state.active_turn_id
    turns = state.turns
    if tid is not None:
        turn = state.turns[tid]
        wait = turn.wait
        if wait is not None:
            turns = turns | {tid: replace(turn, wait=hydrate_record(self, wait))}
    operations = state.operations
    if tid is not None:
        hydrated_operations = {}
        for operation_id, operation in state.operations.items():
            attempts = operation.attempts
            if operation.turn_id == tid and any(
                attempt.wait and attempt.wait.get("handle", {}).get("kind") == "user" for attempt in operation.attempts.values()
            ):
                attempts = {
                    number: replace(
                        attempt,
                        wait=self.hydrate_parked_wait(operation_id, number, attempt.wait),
                    )
                    if attempt.wait and attempt.wait.get("handle", {}).get("kind") == "user"
                    else attempt
                    for number, attempt in operation.attempts.items()
                }
            hydrated_operations[operation_id] = replace(operation, attempts=attempts)
        operations = hydrated_operations
    return replace(state, operations=operations, turns=turns)


def _check_provider_result(self: CoreDriver, plan: Record, outcome: Definitive) -> None:
    assert plan.operation_id is not None and plan.attempt is not None
    attempt = self.state.operations[plan.operation_id].attempts[plan.attempt]
    handle = attempt.wait["handle"] if attempt.wait and attempt.wait["handle"]["kind"] == "provider" else None
    if handle is not None:
        hydrated_wait = _hydrated_provider_wait(self, attempt)
        assert hydrated_wait is not None
        handle = hydrated_wait["handle"]
    logical_plan = self.hydrate_plan(plan) if self.content_reader is not None and getattr(self, "records", None) else plan
    validate_provider_result(logical_plan, outcome, handle)


def _hydratable_definition(value: Any) -> Any:
    if declared_content_ref(value) is not None or type(value) is not dict:
        return value
    return copy_json(value)


def _pending_batch_in_state(state) -> bool:
    return any(
        op.turn_id == state.active_turn_id
        and (
            attempt.state in {"planned", "started", "parked"}
            or (attempt.state == "unknown" and attempt.unknown is not None and attempt.unknown._payload["retry"] == "retry")
        )
        for op in state.operations.values()
        for attempt in [op.attempts[max(op.attempts)]]
    )


def _yieldable_batch_in_state(state, *, one_turn: bool, cancelled: bool) -> list[Attempt]:
    """Return the whole remaining batch only when every observation can yield."""

    tid = state.active_turn_id
    if one_turn or tid is None:
        return []
    turn = state.turns[tid]
    if turn.cancelled or turn.suspended or turn.wait or cancelled:
        return []
    pending = []
    for op in state.operations.values():
        a = op.attempts[max(op.attempts)]
        if op.turn_id != tid or a.state == "completed":
            continue
        if a.state == "unknown" and a.unknown and a.unknown._payload["retry"] != "retry":
            continue
        if a.state != "parked" or not a.wait or not a.wait.get("yield_on_steer"):
            return []
        if a.child_handle:
            completed = state.child_completions or {}
            if any(
                h["session_id"] not in a.wait["yieldable_children"]
                and (a.plan.operation_id, a.plan.attempt, h["session_id"]) not in completed
                for h in child_handles(a.child_handle)
            ):
                return []
        pending.append(a)
    return pending


def _declared_wait_in_state(state) -> bool:
    return any(
        op.turn_id == state.active_turn_id
        and (attempt := op.attempts[max(op.attempts)]).state == "parked"
        and attempt.wait
        and attempt.wait.get("yield_on_steer")
        for op in state.operations.values()
    )


def _steer_yield_plan_in_state(
    state,
    item: InboxItem,
    *,
    one_turn: bool,
    cancelled: bool = False,
) -> list[Attempt] | None:
    """Return the core yield plan: None retains, [] applies, non-empty yields."""

    if item.kind != "steer":
        return []
    tid = state.active_turn_id
    if tid is None:
        return None
    target_tid = item.target_turn_id
    declared_wait = not one_turn and _declared_wait_in_state(state)
    target_matches = target_tid == tid or (declared_wait and target_tid is None)
    if not _pending_batch_in_state(state) or not target_matches:
        return []
    current_generation = state.turns[tid].start._payload["generation"]
    if item.generation not in {None, current_generation}:
        return None
    yielding = _yieldable_batch_in_state(state, one_turn=one_turn, cancelled=cancelled)
    return yielding or None


def _steer_held_in_state(
    state,
    item: InboxItem,
    *,
    one_turn: bool,
    cancelled: bool = False,
) -> bool:
    return (
        _steer_yield_plan_in_state(
            state,
            item,
            one_turn=one_turn,
            cancelled=cancelled,
        )
        is None
    )


def _steer_after_child(
    self: CoreDriver,
    *,
    child_item: InboxItem,
    child_receipt: Record,
    child_extra: list[Record],
    steer_item: InboxItem,
    stored_steer,
    one_turn: bool,
) -> Generator[object, Any, tuple[Record, list[Record]] | None]:
    original_state, original_fold = self.state, self.fold
    provisional = self.fold.fork()
    state = provisional.state
    state.applied_inputs[child_item.input_id] = child_receipt
    if state.child_completions is None:
        state.child_completions = {}
    key = (child_item.payload["operation_id"], child_item.payload["attempt"], child_item.payload["session_id"])
    state.child_completions[key] = child_receipt
    provisional.consumed[child_item.input_id] = child_item
    for child_record in child_extra:
        if child_record.kind != "op_completed":
            continue
        assert child_record.operation_id is not None and child_record.attempt is not None
        provisional._own_operation(child_record.operation_id)
        operation = state.operations[child_record.operation_id]
        attempt = operation.attempts[child_record.attempt]
        attempt.state = "completed"
        attempt.result = child_record
        attempt.context = child_record._payload["context"]
        attempt.wait = None
        operation.selected_attempt = child_record.attempt
    try:
        self.state, self.fold = provisional.state, provisional
        cancelled = yield InputSetting.CANCELLED
        yielding = _steer_yield_plan_in_state(
            self.state,
            steer_item,
            one_turn=one_turn,
            cancelled=cancelled,
        )
        if not yielding:
            return None
        from . import _driver_outcomes as outcomes

        extra = []
        for attempt in yielding:
            view = _BatchDriverView(self, content_reader=None if attempt.child_handle else self.content_reader)
            extra.extend((yield from outcomes.yield_results_steps(cast(CoreDriver, view), [attempt], steer_item)))
        raw_steer = stored_steer.item
        applied = self.record(
            "input_applied",
            {
                "input": raw_steer.to_dict(),
                "input_digest": raw_steer.digest,
                "input_proof": stored_input_proof(stored_steer),
                "disposition": "applied",
                "reason": None,
                "target_operation_id": None,
                "target_wait_id": None,
                "position": "before_model",
            },
            tid=self.state.active_turn_id,
        )
        return applied, extra
    finally:
        self.state, self.fold = original_state, original_fold


def apply_input(self: CoreDriver, *, one_turn: bool) -> Generator[object, Any, bool]:
    inputs = self.store.peek_inbox(self.sid, through_input_seq=self.watermark)
    # At one captured watermark, authenticated child terminals win over steering.
    declared_wait = not one_turn and bool(inputs) and _declared_wait_in_state(self.state)
    if declared_wait:
        if len(inputs) == 256 and self.watermark > 256:
            inputs = self.store.peek_inbox(self.sid, through_input_seq=self.watermark, limit=self.watermark)
        inputs = sorted(inputs, key=lambda stored: stored.item.kind != "child_result")
    for index, stored in enumerate(inputs):
        item, extra, disposition, reason, target, wait = stored.item, [], "applied", None, None, None
        logical_item = hydrate_input(self, item) if self.content_reader is not None else item
        tid = self.state.active_turn_id
        target_tid = logical_item.target_turn_id
        evidence = logical_item.kind in {"provider_result", "provider_evidence", "child_result"}
        retained_reply = logical_item.kind in {"user", "approval_answer"} and any(
            r._payload["disposition"] == "applied"
            and r._payload["input"]["kind"] == logical_item.kind
            and r._payload["input"]["target_turn_id"] == target_tid
            and r._payload["input"]["generation"] == logical_item.generation
            and r._payload["target_wait_id"] is not None
            and (
                r._payload["input"]["payload"].get("request_id") == logical_item.payload.get("request_id")
                if logical_item.kind == "approval_answer"
                else isinstance(logical_item.payload["content"], dict)
                and r._payload["target_wait_id"] == logical_item.payload["content"].get("interaction_id")
            )
            for r in self.state.applied_inputs.values()
        )
        # Cancellation may arrive before the child's initial user input opens its turn.
        if (
            logical_item.kind == "control"
            and logical_item.payload["action"] in {"cancel", "abort"}
            and tid is None
            and any(
                r._payload["disposition"] == "queued"
                and key not in self.state.admitted_inputs
                and target_tid == (r._payload["input"]["target_turn_id"] or f"{self.sid}/turn/{key}")
                for key, r in self.state.applied_inputs.items()
            )
        ):
            continue
        if logical_item.kind == "model_selection" and self.pending_batch():
            continue
        yielding = []
        if logical_item.kind == "steer":
            cancelled = yield InputSetting.CANCELLED
            yield_plan = _steer_yield_plan_in_state(
                self.state,
                logical_item,
                one_turn=one_turn,
                cancelled=cancelled,
            )
            if yield_plan is None:
                continue
            yielding = yield_plan
        if compaction_request(logical_item) is not None:
            if self.state.closed:
                disposition, reason = "rejected", "session closed"
            else:
                # Compaction is work selected by the existing compaction phase, not a
                # reply to a parked user wait. Keep one queued disposition in both idle
                # and active states so the raw SQL fold can use the retained record
                # without hydrating its content-only reason.
                disposition = "queued"
                reason = "host compaction request"
        elif logical_item.kind == "child_result":
            # Completion identity and dedupe bind the retained inbox bytes; projections stay logical.
            target, extra, disposition, reason = yield from apply_child(self, item)
            if disposition == "pending":
                continue
        elif (target_tid is not None and target_tid != tid and not evidence and not retained_reply) or (
            target_tid in self.state.turns
            and logical_item.generation is not None
            and logical_item.generation != self.state.turns[target_tid].start._payload["generation"]
        ):
            disposition, reason = "rejected", "stale turn or generation"
        elif evidence:
            target = logical_item.payload["operation_id"]
            op = self.state.operations.get(target)
            number = logical_item.payload["attempt"]
            attempt = op.attempts.get(number) if op else None
            if (
                attempt is None
                or op is None
                or op.turn_id != target_tid
                or not attempt.started
                or logical_item.generation != self.state.turns[op.turn_id].start._payload["generation"]
                or logical_item.payload["request_digest"] != attempt.execution_plan._payload["request_digest"]
            ):
                disposition, reason = "rejected", "untrusted or mismatched provider evidence"
            else:
                try:
                    if logical_item.kind == "provider_evidence" and self.content_reader is not None:
                        validate_reference_provider_handle(attempt.execution_plan, item.payload["handle"])
                except ProviderResultInvalid:
                    disposition, reason = "rejected", "provider_result_invalid"
                    reference_invalid = True
                else:
                    reference_invalid = False
                if not reference_invalid:
                    if not (yield AuthenticateProvider(logical_item, item, attempt.execution_plan)):
                        disposition, reason = "rejected", "untrusted or mismatched provider evidence"
                    elif logical_item.kind == "provider_result":
                        retained_result = hydrate_record(self, attempt.result) if attempt.result is not None else None
                        if logical_item.payload["provider_binding"] != attempt.execution_plan._payload["provider_binding"]:
                            disposition, reason = "rejected", "provider binding mismatch"
                        elif attempt.yielded:
                            disposition, reason = "noop", "yielded provider handle: audit only"
                        elif retained_result is not None:
                            equal = True
                            for key in ("result", "usage"):
                                retained = canonical_json_bytes(
                                    provider_result_identity(retained_result._payload[key])
                                    if key == "result"
                                    else retained_result._payload[key]
                                )
                                incoming = logical_item.payload[key]
                                if key == "result":
                                    incoming = yield ProjectProviderResult(
                                        attempt.execution_plan,
                                        incoming,
                                        retained_result._payload["result"].get("artifact"),
                                    )
                                    incoming = provider_result_identity(incoming)
                                if retained != canonical_json_bytes(incoming):
                                    equal = False
                                    break
                            disposition = "noop" if equal else "rejected"
                            reason = "retained result" if disposition == "noop" else "result conflict"
                        else:
                            outcome = Definitive(
                                logical_item.payload["result"],
                                tuple(logical_item.payload["evidence"]),
                                usage=logical_item.payload["usage"],
                            )
                            try:
                                _check_provider_result(self, attempt.execution_plan, outcome)
                            except ProviderResultInvalid:
                                disposition, reason = "rejected", "provider_result_invalid"
                            else:
                                extra = yield CompleteResult(attempt.execution_plan, outcome)
                    elif attempt.yielded:
                        disposition, reason = "noop", "yielded provider handle: audit only"
                    elif attempt.state == "started":
                        try:
                            accepted = Accepted(
                                logical_item.payload["handle"],
                                yield_on_steer=logical_item.payload.get("yield_on_steer", False),
                            )
                            if self.content_reader is not None:
                                validate_reference_provider_handle(attempt.execution_plan, item.payload["handle"])
                            validate_acceptance(attempt.execution_plan, accepted)
                            now = yield InputSetting.POLL_NOW
                            parked = self.record(
                                "op_parked",
                                {
                                    "phase": "after_dispatch",
                                    "handle": accepted.handle,
                                    "poll_at_ms": now + (yield InputSetting.POLL_MS),
                                    "deadline_ms": None,
                                },
                                attempt.execution_plan,
                            )
                            if accepted.yield_on_steer:
                                parked = replace(parked, payload=parked._payload | {"yield_on_steer": True})
                        except ProviderResultInvalid:
                            disposition, reason = "rejected", "provider_result_invalid"
                        else:
                            extra = [parked]
                    elif attempt.wait:
                        hydrated_wait = _hydrated_provider_wait(self, attempt)
                        assert hydrated_wait is not None
                        if hydrated_wait["handle"] != logical_item.payload["handle"]:
                            disposition, reason = "rejected", "acceptance is not attachable"
                        elif hydrated_wait.get("yield_on_steer", False) == logical_item.payload.get("yield_on_steer", False):
                            disposition = "noop"
                        else:
                            disposition, reason = "rejected", "acceptance declaration conflict"
                    else:
                        disposition, reason = "rejected", "acceptance is not attachable"
        elif item.kind == "approval_answer":
            target = logical_item.payload["operation_id"]
            op = self.state.operations.get(target)
            attempt = op.attempts.get(logical_item.payload["attempt"]) if op else None
            hydrated_wait = _hydrated_provider_wait(self, attempt) if attempt and attempt.wait else None
            handle = hydrated_wait["handle"] if hydrated_wait is not None else {}
            wait = logical_item.payload["request_id"]
            previous = attempt.approval_answer if attempt else None
            previous_item = (
                hydrate_input(self, InboxItem(**previous._payload["input"]))
                if previous is not None and self.content_reader is not None
                else InboxItem(**previous._payload["input"])
                if previous is not None
                else None
            )
            if (
                previous is not None
                and previous_item is not None
                and op is not None
                and target_tid == op.turn_id
                and logical_item.payload == previous_item.payload
            ):
                disposition, reason = "noop", "approval already resolved"
            elif (
                target_tid != tid
                or not handle
                or handle.get("kind") != "approval"
                or any(handle[k] != logical_item.payload[k] for k in ("request_id", "request_digest", "scope"))
            ):
                disposition, reason = "rejected", "approval identity or scope mismatch"
            elif (
                attempt
                and attempt.wait
                and attempt.wait["deadline_ms"] is not None
                and logical_item.payload["decision"] != "timeout"
                and (yield InputSetting.POLL_NOW) >= attempt.wait["deadline_ms"]
            ):
                disposition, reason = "rejected", "approval deadline expired"
            elif attempt and attempt.approval:
                assert previous_item is not None
                equal = previous_item.payload == logical_item.payload
                disposition, reason = ("noop", "approval already resolved") if equal else ("rejected", "approval reply conflict")
        elif item.kind in {"user", "follow_up"}:
            content = logical_item.payload["content"]
            previous = next(
                (
                    r
                    for r in self.state.applied_inputs.values()
                    if logical_item.kind == "user"
                    and isinstance(content, dict)
                    and r._payload["disposition"] == "applied"
                    and r._payload["input"]["kind"] == "user"
                    and r._payload["input"]["target_turn_id"] == target_tid
                    and r._payload["target_wait_id"] == content.get("interaction_id")
                ),
                None,
            )
            typed_content = (
                isinstance(content, dict)
                and isinstance(content.get("metadata"), dict)
                and "interaction_response" in content["metadata"]
            )
            if item.kind == "follow_up" and typed_content:
                disposition, reason = "rejected", "invalid_interaction_answer"
            elif previous is not None:
                previous_item = (
                    hydrate_input(self, InboxItem(**previous._payload["input"]))
                    if self.content_reader is not None
                    else InboxItem(**previous._payload["input"])
                )
                equal = (
                    previous_item.payload == logical_item.payload
                    and previous._payload["input"]["generation"] == logical_item.generation
                )
                disposition, reason = ("noop", "retained user reply") if equal else ("rejected", "user reply conflict")
            elif logical_item.kind == "user" and (
                target_tid is not None or typed_content or (isinstance(content, dict) and "interaction_id" in content)
            ):
                from ._interactions import InteractionError, validate_inbox_reply

                try:
                    validation_state = _retained_reply_state(self)
                    reply_target, answer = validate_inbox_reply(validation_state, self.sid, logical_item)
                except InteractionError as exc:
                    disposition, reason = "rejected", exc.reason
                else:
                    target, wait = reply_target.operation_id, reply_target.interaction_id
                    if target is not None:
                        retained_plan = self.state.operations[target].attempts[reply_target.attempt].execution_plan
                        logical_plan = (
                            self.hydrate_plan(retained_plan)
                            if self.content_reader is not None and getattr(self, "records", None)
                            else retained_plan
                        )
                        yield PrepareToolContext(logical_plan, include_store=True)
                        outcome = yield FinalizeTool(
                            retained_plan,
                            Definitive(
                                ToolExecutionResult(
                                    tool_call_id=logical_plan._payload["request"]["id"], content=answer.model_text()
                                ).to_dict(),
                                (),
                            ),
                        )
                        extra = yield CompleteResult(retained_plan, outcome)
            elif tid and logical_item.kind == "user":
                disposition, reason = "rejected", "reply must identify the parked interaction"
            else:
                disposition = "queued"
        elif logical_item.kind == "model_selection":
            if self.state.closed:
                disposition, reason = "rejected", "session closed"
            elif logical_item.payload["selection"] is not None:
                try:
                    selection_binding = yield ResolveModelSelection(logical_item.payload["selection"])
                except Exception:
                    disposition, reason = "rejected", "model_selection_unresolvable"
            if logical_item.feature_level == 2 and disposition == "applied":
                assert logical_item.requirements is not None
                requirements = normalize_requirements(logical_item.requirements)
                current_handler = (
                    self.state.turns[self.state.active_turn_id].start._payload["handler_version"]
                    if self.state.active_turn_id is not None
                    else (yield InputSetting.HANDLER_VERSION)
                )
                if requirements.handler_version != current_handler:
                    disposition, reason = "rejected", "model_selection_handler_mismatch"
                elif logical_item.payload["selection"] is not None and set(requirements.capabilities) != set(
                    model_capability_tags(selection_binding)
                ):
                    disposition, reason = "rejected", "model_selection_requirements_mismatch"
        elif logical_item.kind == "control":
            if logical_item.payload["action"] not in {"archive", "close"} and (tid is None or target_tid != tid):
                disposition, reason = "rejected", "control requires active target"
        elif logical_item.kind == "steer":
            if tid is None and target_tid is None:
                continue
            if tid is None or target_tid not in {None, tid}:
                disposition, reason = "rejected", "steer requires active target"
            elif yielding:
                extra = yield YieldResults(yielding, logical_item)
        else:
            disposition, reason = "rejected", "unsupported input"
        applied = self.record(
            "input_applied",
            {
                "input": item.to_dict(),
                "input_digest": item.digest,
                "input_proof": stored_input_proof(stored),
                "disposition": disposition,
                "reason": reason,
                "target_operation_id": target,
                "target_wait_id": wait,
                "position": "before_model" if item.kind == "steer" else "between_actions",
            },
            tid=target_tid if evidence else tid,
        )
        if logical_item.kind == "model_selection" and disposition == "applied" and logical_item.payload["selection"] is not None:
            applied = replace(applied, payload=applied._payload | {"model_binding": selection_binding})
        if (
            declared_wait
            and logical_item.kind == "child_result"
            and disposition == "applied"
            and (reason == "batch member completed" or any(record.kind == "op_completed" for record in extra))
            and index + 1 < len(inputs)
            and inputs[index + 1].item.kind == "steer"
        ):
            steer_stored = inputs[index + 1]
            steer_item = hydrate_input(self, steer_stored.item) if self.content_reader is not None else steer_stored.item
            batch = yield from _steer_after_child(
                self,
                child_item=item,
                child_receipt=replace(applied, payload=applied._payload | {"input": logical_item.to_dict()}),
                child_extra=extra,
                steer_item=steer_item,
                stored_steer=steer_stored,
                one_turn=one_turn,
            )
            if batch is not None:
                steer_applied, steer_extra = batch
                yield CommitInput(
                    [applied, *extra, steer_applied, *steer_extra],
                    item.input_id,
                    (steer_item.input_id,),
                )
                return True
        yield CommitInput([applied, *extra], item.input_id)
        return True
    return False


def apply_child(self: CoreDriver, item: InboxItem) -> Generator[object, Any, tuple[str, list[Record], str, str | None]]:
    p = item.payload
    target = p["operation_id"]
    op = self.state.operations.get(target)
    a = op.attempts.get(p["attempt"]) if op else None
    handle = a.child_handle if a else None
    terminal = verify_completion(self.store, self.sid, handle, item) if handle else None
    if a is None or op is None or handle is None or terminal is None:
        return target, [], "rejected", "untrusted or mismatched child completion"
    turn = self.state.turns[op.turn_id]
    if item.generation != turn.start._payload["generation"]:
        return target, [], "rejected", "authenticated child completion: stale generation audit only"
    cache = self.fold.state.child_completions
    prior = cache.get((target, p["attempt"], p["session_id"])) if cache else None
    previous = prior._payload["input"]["payload"] if prior else None
    if previous is not None:
        return target, [], "noop" if digest(previous) == digest(p) else "rejected", "retained child completion"
    if handle["background"] or a.yielded:
        if turn.cancelled or turn.ended or op.turn_id != self.state.active_turn_id:
            return target, [], "noop", "late background completion: audit only"
        if self.pending_batch():
            return target, [], "pending", None
        return target, [], "applied", "background child notification"
    handles = child_handles(handle)
    delivered = {p["session_id"]: terminal}
    for h in handles:
        prior = cache.get((target, p["attempt"], h["session_id"])) if cache else None
        if prior:
            old = InboxItem(**prior._payload["input"])
            verified = verify_completion(self.store, self.sid, handle, old)
            if verified:
                delivered[old.payload["session_id"]] = verified
    if any(h["session_id"] not in delivered for h in handles):
        return target, [], "applied", "batch member completed"
    sdk = a.wait and a.wait.get("delegation")
    result = yield ChildResult(a.execution_plan, handles)
    if a.result:
        return (
            target,
            [],
            "noop" if digest(a.result._payload["result"]) == digest(result) else "rejected",
            "retained child result",
        )
    outcome = Definitive(result, tuple(f"child/{r.seq}/{r.record.digest}" for r in delivered.values()))
    if sdk and not turn.cancelled and not turn.ended and self.state.active_turn_id == op.turn_id:
        yield PrepareToolContext(a.execution_plan, include_store=False)
        outcome = yield FinalizeTool(a.execution_plan, outcome)
    return target, (yield CompleteResult(a.execution_plan, outcome)), "applied", None


def _validate_actual_requirements(bound, actual) -> None:
    if actual.level < bound.level:
        raise ValueError("unsatisfiable pre-admission requirements: actual level is lower than the bound")
    if bound.handler_version is not None and actual.handler_version != bound.handler_version:
        raise ValueError("unsatisfiable pre-admission requirements: handler version changed")
    exceeds = set(actual.capabilities) - set(bound.capabilities)
    if exceeds:
        raise ValueError(
            "unsatisfiable pre-admission requirements: compiled definition exceeds the trusted bound: "
            + ", ".join(sorted(exceeds))
        )


def start_turn(self: CoreDriver) -> Generator[object, Any, bool]:
    inputs = sorted(self.state.applied_inputs.items(), key=lambda pair: pair[1]._payload["input"]["kind"] == "follow_up")
    attributes = self.session_attributes()
    for input_id, applied in inputs:
        if applied._payload["disposition"] != "queued" or input_id in self.state.admitted_inputs:
            continue
        item = applied._payload["input"]
        tid = item["target_turn_id"] or f"{self.sid}/turn/{input_id}"
        retained_content = item["payload"]["content"]
        content = (
            self.input_content(input_id, retained_content)
            if declared_content_ref(retained_content) is not None
            else retained_content
        )
        logical_item = replace(InboxItem(**item), payload=item["payload"] | {"content": content})
        bound = normalize_requirements(item["requirements"]) if item["feature_level"] == 2 else None
        if bound is None:
            created = self.records[0].record
            if created.feature_level == 2:
                bound = normalize_requirements(created._payload["bootstrap"])
        control = compaction_request(logical_item) is not None
        seed_attributes = attributes.get("seed")
        task = yield CompileTurn(
            content["text"] if isinstance(content, dict) and "messages" in content else str(content),
            tid,
            seed=(
                seed_attributes
                if not self.state.turns and isinstance(seed_attributes, dict) and "messages" in seed_attributes
                else None
            ),
            control=control,
        )
        task.metadata.pop("_vv_agent_app_server_compaction_only", None)
        if control:
            task.metadata["_vv_agent_app_server_compaction_only"] = True
            if self.state.turns:
                task.initial_messages = []
        if isinstance(content, dict) and "messages" in content:
            task.metadata.setdefault("vv_session", {})["input_messages"] = content["messages"]
        task.task_id = tid
        if not self.state.turns:
            seed = seed_attributes
            if seed is not None and "messages" in seed:
                task.initial_messages = [Message.from_dict(copy_json(m)) for m in seed["messages"]]
                task.initial_shared_state = copy_json(seed["shared_state"])
        facts = yield CurrentDefinition(task)
        definitions = facts.definitions
        try:
            definition = definitions._definition(task, facts.bindings, facts.catalog)
        except StopIteration as exc:
            raise _StopIterationTransport(exc) from exc
        prior = definitions._retained_task_key
        stored_definition = _hydratable_definition(definition)
        if prior is not None:
            retained = _RetainedTools(prior[0])
            if retained.encode() == definitions._definition_suffix[len(b',"tools":') : -1]:
                definition = definition | {"tools": retained}
                stored_definition = _hydratable_definition(definition | {"tools": retained.value})
        limits = (yield InputSetting.BUDGET_LIMITS) or RunBudgetLimits()
        handler_version = yield InputSetting.HANDLER_VERSION
        admission = attributes.get("child_admission") if input_id == "start" else None
        admission_digest: str | None = None
        budget: dict[str, Any] = limits.to_dict()
        if admission is not None:
            if admission["handler_version"] != handler_version:
                raise Conflict("child admission handler version mismatch")
            admitted_definition = admission["definition"]
            budget = admission["budget"]
            if (
                declared_content_ref(admitted_definition) is None
                and digest(admitted_definition) != admission["definition_digest"]
            ):
                raise Conflict("child admission definition digest mismatch")
            stored_definition = _hydratable_definition(admitted_definition)
        actual_requirements = None
        if bound is not None:
            actual_requirements = definition_requirements(stored_definition, handler_version=handler_version)
            _validate_actual_requirements(bound, actual_requirements)
        yield PublishTurnStarted(
            input_id=input_id,
            tid=tid,
            definition=stored_definition,
            budget=budget,
            handler_version=handler_version,
            generation=item["generation"] or 0,
            admission_definition_digest=admission_digest,
            requirements=actual_requirements.to_dict() if actual_requirements is not None else None,
        )
        yield InputSetting.RESET_TOKEN
        return True
    return False
