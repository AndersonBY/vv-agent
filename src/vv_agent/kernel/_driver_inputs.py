"""Inbox disposition and turn admission over the actual core driver ledger."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from ._budget_values import RunBudgetLimits
from ._canonical_json import canonical_json_bytes
from ._child_store import child_handles, verify_completion
from ._definition import DefinitionBindings, DefinitionCache
from ._driver import CoreDriver
from ._driver_requests import CommitRecords
from ._execution_values import AgentTask
from ._message_values import Message
from ._provider_outcomes import Accepted, Definitive, ProviderResultInvalid, validate_acceptance, validate_provider_result
from ._records import InboxItem, Record, _RetainedTools, copy_json, digest
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
    plan: Record


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


def _check_provider_result(self: CoreDriver, plan: Record, outcome: Definitive) -> None:
    assert plan.operation_id is not None and plan.attempt is not None
    attempt = self.state.operations[plan.operation_id].attempts[plan.attempt]
    handle = attempt.wait["handle"] if attempt.wait and attempt.wait["handle"]["kind"] == "provider" else None
    validate_provider_result(plan, outcome, handle)


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
        tid = self.state.active_turn_id
        target_tid = item.target_turn_id
        evidence = item.kind in {"provider_result", "provider_evidence", "child_result"}
        retained_reply = item.kind in {"user", "approval_answer"} and any(
            r._payload["disposition"] == "applied"
            and r._payload["input"]["kind"] == item.kind
            and r._payload["input"]["target_turn_id"] == target_tid
            and r._payload["input"]["generation"] == item.generation
            and r._payload["target_wait_id"] is not None
            and (
                r._payload["input"]["payload"].get("request_id") == item.payload.get("request_id")
                if item.kind == "approval_answer"
                else isinstance(item.payload["content"], dict)
                and r._payload["target_wait_id"] == item.payload["content"].get("interaction_id")
            )
            for r in self.state.applied_inputs.values()
        )
        # Cancellation may arrive before the child's initial user input opens its turn.
        if (
            item.kind == "control"
            and item.payload["action"] in {"cancel", "abort"}
            and tid is None
            and any(
                r._payload["disposition"] == "queued"
                and key not in self.state.admitted_inputs
                and target_tid == (r._payload["input"]["target_turn_id"] or f"{self.sid}/turn/{key}")
                for key, r in self.state.applied_inputs.items()
            )
        ):
            continue
        if item.kind == "model_selection" and self.pending_batch():
            continue
        yielding = []
        if (
            item.kind == "steer"
            and tid is not None
            and (target_tid == tid or (declared_wait and target_tid is None))
            and item.generation in {None, self.state.turns[tid].start._payload["generation"]}
        ):
            yielding = yield from _yieldable_batch(self, one_turn=one_turn)
        if item.kind == "steer" and (
            tid is None
            or (self.pending_batch() and (target_tid == tid or (declared_wait and target_tid is None)) and not yielding)
        ):
            continue
        if compaction_request(item) is not None:
            if self.state.closed:
                disposition, reason = "rejected", "session closed"
            else:
                disposition = "applied" if tid else "queued"
                reason = "host compaction request"
        elif item.kind == "child_result":
            target, extra, disposition, reason = yield from apply_child(self, item)
            if disposition == "pending":
                continue
        elif (target_tid is not None and target_tid != tid and not evidence and not retained_reply) or (
            target_tid in self.state.turns
            and item.generation is not None
            and item.generation != self.state.turns[target_tid].start._payload["generation"]
        ):
            disposition, reason = "rejected", "stale turn or generation"
        elif evidence:
            target = item.payload["operation_id"]
            op = self.state.operations.get(target)
            number = item.payload["attempt"]
            attempt = op.attempts.get(number) if op else None
            if (
                attempt is None
                or op is None
                or op.turn_id != target_tid
                or not attempt.started
                or item.generation != self.state.turns[op.turn_id].start._payload["generation"]
                or item.payload["request_digest"] != attempt.execution_plan._payload["request_digest"]
                or not (yield AuthenticateProvider(item, attempt.execution_plan))
            ):
                disposition, reason = "rejected", "untrusted or mismatched provider evidence"
            elif item.kind == "provider_result":
                if item.payload["provider_binding"] != attempt.execution_plan._payload["provider_binding"]:
                    disposition, reason = "rejected", "provider binding mismatch"
                elif attempt.yielded:
                    disposition, reason = "noop", "yielded provider handle: audit only"
                elif attempt.result is not None:
                    equal = True
                    for key in ("result", "usage"):
                        retained = canonical_json_bytes(
                            provider_result_identity(attempt.result._payload[key])
                            if key == "result"
                            else attempt.result._payload[key]
                        )
                        incoming = item.payload[key]
                        if key == "result":
                            incoming = yield ProjectProviderResult(
                                attempt.execution_plan, incoming, attempt.result._payload["result"].get("artifact")
                            )
                            incoming = provider_result_identity(incoming)
                        if retained != canonical_json_bytes(incoming):
                            equal = False
                            break
                    disposition = "noop" if equal else "rejected"
                    reason = "retained result" if disposition == "noop" else "result conflict"
                else:
                    outcome = Definitive(item.payload["result"], tuple(item.payload["evidence"]), usage=item.payload["usage"])
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
                    accepted = Accepted(item.payload["handle"], yield_on_steer=item.payload.get("yield_on_steer", False))
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
            elif attempt.wait and attempt.wait["handle"] == item.payload["handle"]:
                if attempt.wait.get("yield_on_steer", False) == item.payload.get("yield_on_steer", False):
                    disposition = "noop"
                else:
                    disposition, reason = "rejected", "acceptance declaration conflict"
            else:
                disposition, reason = "rejected", "acceptance is not attachable"
        elif item.kind == "approval_answer":
            target = item.payload["operation_id"]
            op = self.state.operations.get(target)
            attempt = op.attempts.get(item.payload["attempt"]) if op else None
            handle = attempt.wait["handle"] if attempt and attempt.wait else {}
            wait = item.payload["request_id"]
            previous = attempt.approval_answer if attempt else None
            if (
                previous is not None
                and op is not None
                and target_tid == op.turn_id
                and item.payload == previous._payload["input"]["payload"]
            ):
                disposition, reason = "noop", "approval already resolved"
            elif (
                target_tid != tid
                or not handle
                or handle.get("kind") != "approval"
                or any(handle[k] != item.payload[k] for k in ("request_id", "request_digest", "scope"))
            ):
                disposition, reason = "rejected", "approval identity or scope mismatch"
            elif (
                attempt
                and attempt.wait
                and attempt.wait["deadline_ms"] is not None
                and item.payload["decision"] != "timeout"
                and (yield InputSetting.POLL_NOW) >= attempt.wait["deadline_ms"]
            ):
                disposition, reason = "rejected", "approval deadline expired"
            elif attempt and attempt.approval:
                original = attempt.approval_answer
                assert original is not None
                equal = original._payload["input"]["payload"] == item.payload
                disposition, reason = ("noop", "approval already resolved") if equal else ("rejected", "approval reply conflict")
        elif item.kind in {"user", "follow_up"}:
            content = item.payload["content"]
            previous = next(
                (
                    r
                    for r in self.state.applied_inputs.values()
                    if item.kind == "user"
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
                equal = (
                    previous._payload["input"]["payload"] == item.payload
                    and previous._payload["input"]["generation"] == item.generation
                )
                disposition, reason = ("noop", "retained user reply") if equal else ("rejected", "user reply conflict")
            elif item.kind == "user" and (
                target_tid is not None or typed_content or (isinstance(content, dict) and "interaction_id" in content)
            ):
                from ._interactions import InteractionError, validate_inbox_reply

                try:
                    reply_target, answer = validate_inbox_reply(self.state, self.sid, item)
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
            elif tid and item.kind == "user":
                disposition, reason = "rejected", "reply must identify the parked interaction"
            else:
                disposition = "queued"
        elif item.kind == "model_selection":
            if self.state.closed:
                disposition, reason = "rejected", "session closed"
            elif item.payload["selection"] is not None:
                try:
                    selection_binding = yield ResolveModelSelection(item.payload["selection"])
                except Exception:
                    disposition, reason = "rejected", "model_selection_unresolvable"
        elif item.kind == "control":
            if item.payload["action"] not in {"archive", "close"} and (tid is None or target_tid != tid):
                disposition, reason = "rejected", "control requires active target"
        elif item.kind == "steer":
            if tid is None and target_tid is None:
                continue
            if tid is None or target_tid not in {None, tid}:
                disposition, reason = "rejected", "steer requires active target"
            elif yielding:
                extra = yield YieldResults(yielding, item)
        else:
            disposition, reason = "rejected", "unsupported input"
        applied = self.record(
            "input_applied",
            {
                "input": item.to_dict(),
                "input_digest": item.digest,
                "disposition": disposition,
                "reason": reason,
                "target_operation_id": target,
                "target_wait_id": wait,
                "position": "before_model" if item.kind == "steer" else "between_actions",
            },
            tid=target_tid if evidence else tid,
        )
        if item.kind == "model_selection" and disposition == "applied" and item.payload["selection"] is not None:
            applied = replace(applied, payload=applied._payload | {"model_binding": selection_binding})
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
    cache = self.state.child_completions
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
    for input_id, applied in inputs:
        if applied._payload["disposition"] != "queued" or input_id in self.state.admitted_inputs:
            continue
        item = applied._payload["input"]
        tid = item["target_turn_id"] or f"{self.sid}/turn/{input_id}"
        content = item["payload"]["content"]
        task = yield CompileTurn(
            content["text"] if isinstance(content, dict) and "messages" in content else str(content),
            tid,
            seed=(
                self.records[0].record._payload["attributes"].get("seed")
                if not self.state.turns and "messages" in self.records[0].record._payload["attributes"].get("seed", {})
                else None
            ),
            control=compaction_request(InboxItem(**item)) is not None,
        )
        task.metadata.pop("_vv_agent_app_server_compaction_only", None)
        if compaction_request(InboxItem(**item)) is not None:
            task.metadata["_vv_agent_app_server_compaction_only"] = True
            if self.state.turns:
                task.initial_messages = []
        if isinstance(content, dict) and "messages" in content:
            task.metadata.setdefault("vv_session", {})["input_messages"] = content["messages"]
        task.task_id = tid
        if not self.state.turns:
            seed = self.records[0].record._payload["attributes"].get("seed")
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
        if prior is not None:
            retained = _RetainedTools(prior[0])
            if retained.encode() == definitions._definition_suffix[len(b',"tools":') : -1]:
                definition = definition | {"tools": retained}
        limits = (yield InputSetting.BUDGET_LIMITS) or RunBudgetLimits()
        admission = self.records[0].record._payload["attributes"].get("child_admission") if input_id == "start" else None
        if admission is not None:
            if admission["handler_version"] != (yield InputSetting.HANDLER_VERSION):
                raise Conflict("child admission handler version mismatch")
            definition = admission["definition"]
            limits = RunBudgetLimits.from_dict(admission["budget"])
            if digest(definition) != admission["definition_digest"]:
                raise Conflict("child admission definition digest mismatch")
        yield CommitRecords(
            [
                self.record(
                    "turn_started",
                    {
                        "input_ids": [input_id],
                        "definition": definition,
                        "definition_digest": admission["definition_digest"] if admission else definitions._definition_digest,
                        "handler_version": (yield InputSetting.HANDLER_VERSION),
                        "budget": limits.to_dict(),
                        "binding": None,
                        "generation": item["generation"] or 0,
                    },
                    tid=tid,
                )
            ]
        )
        yield InputSetting.RESET_TOKEN
        return True
    return False
