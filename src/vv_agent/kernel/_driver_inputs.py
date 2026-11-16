"""Inbox disposition and turn admission over the actual core driver ledger."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

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


@dataclass(frozen=True, slots=True)
class PublishTurnStarted:
    input_id: str
    tid: str
    definition: dict[str, Any]
    budget: dict[str, Any]
    handler_version: str
    generation: int
    admission_definition_digest: str | None = None


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
    definition = copy_json(value)
    binding = definition.get("model_binding")
    if type(binding) is dict:
        binding.setdefault("endpoints", [])
    return definition


def _yieldable_batch(self: CoreDriver, *, one_turn: bool) -> Generator[object, Any, list[Attempt]]:
    """Return the whole remaining batch only when every observation can yield."""
    tid = self.state.active_turn_id
    # shortcut: local drivers keep normal waits; enable yielding with local scheduling.
    if one_turn or tid is None:
        return []
    turn = self.state.turns[tid]
    if turn.cancelled or turn.suspended or turn.wait or (yield InputSetting.CANCELLED):
        return []
    pending = []
    for op in self.state.operations.values():
        a = op.attempts[max(op.attempts)]
        if op.turn_id != tid or a.state == "completed":
            continue
        if a.state == "unknown" and a.unknown and a.unknown._payload["retry"] != "retry":
            continue
        if a.state != "parked" or not a.wait or not a.wait.get("yield_on_steer"):
            return []
        if a.child_handle:
            completed = self.state.child_completions or {}
            if any(
                h["session_id"] not in a.wait["yieldable_children"]
                and (a.plan.operation_id, a.plan.attempt, h["session_id"]) not in completed
                for h in child_handles(a.child_handle)
            ):
                return []
        pending.append(a)
    return pending


def apply_input(self: CoreDriver, *, one_turn: bool) -> Generator[object, Any, bool]:
    inputs = self.store.peek_inbox(self.sid, through_input_seq=self.watermark)
    # At one captured watermark, authenticated child terminals win over steering.
    declared_wait = (
        not one_turn
        and bool(inputs)
        and any(
            op.turn_id == self.state.active_turn_id
            and (a := op.attempts[max(op.attempts)]).state == "parked"
            and a.wait
            and a.wait.get("yield_on_steer")
            for op in self.state.operations.values()
        )
    )
    if declared_wait:
        if len(inputs) == 256 and self.watermark > 256:
            inputs = self.store.peek_inbox(self.sid, through_input_seq=self.watermark, limit=self.watermark)
        inputs = sorted(inputs, key=lambda stored: stored.item.kind != "child_result")
    for stored in inputs:
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
        if (
            logical_item.kind == "steer"
            and tid is not None
            and (target_tid == tid or (declared_wait and target_tid is None))
            and logical_item.generation in {None, self.state.turns[tid].start._payload["generation"]}
        ):
            yielding = yield from _yieldable_batch(self, one_turn=one_turn)
        if logical_item.kind == "steer" and (
            tid is None
            or (self.pending_batch() and (target_tid == tid or (declared_wait and target_tid is None)) and not yielding)
        ):
            continue
        if compaction_request(logical_item) is not None:
            if self.state.closed:
                disposition, reason = "rejected", "session closed"
            else:
                disposition = "applied" if tid else "queued"
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
                    validation_state = getattr(self, "_logical_state", lambda: self.state)()
                    reply_target, answer = validate_inbox_reply(validation_state, self.sid, logical_item)
                except InteractionError as exc:
                    disposition, reason = "rejected", exc.reason
                else:
                    target, wait = reply_target.operation_id, reply_target.interaction_id
                    if target is not None:
                        plan = self.state.operations[target].attempts[reply_target.attempt].execution_plan
                        yield PrepareToolContext(plan, include_store=True)
                        outcome = yield FinalizeTool(
                            plan,
                            Definitive(
                                ToolExecutionResult(
                                    tool_call_id=plan._payload["request"]["id"], content=answer.model_text()
                                ).to_dict(),
                                (),
                            ),
                        )
                        extra = yield CompleteResult(plan, outcome)
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
        if logical_item.kind == "approval_answer" and disposition == "applied":
            from ._reducer import logical_inputs

            # The raw inbox proof still binds the submitted refs. The reducer must
            # compare this exact validated answer with the hydrated parked wait.
            with logical_inputs({applied.record_id: logical_item}):
                yield CommitInput([applied, *extra], item.input_id)
        else:
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
        yield PublishTurnStarted(
            input_id=input_id,
            tid=tid,
            definition=stored_definition,
            budget=budget,
            handler_version=handler_version,
            generation=item["generation"] or 0,
            admission_definition_digest=admission_digest,
        )
        yield InputSetting.RESET_TOKEN
        return True
    return False
