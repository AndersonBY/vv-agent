"""One deterministic fold; this module neither reads clocks nor performs effects."""

from __future__ import annotations

from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any

from vv_agent.memory.manager import MemoryManager
from vv_agent.memory.microcompact import (
    EXCERPT_METADATA_KEY,
    build_compacted_tool_content,
    replace_with_compacted_marker,
)
from vv_agent.memory.token_utils import count_messages_tokens
from vv_agent.types import Message

from .context import message_ids, project_context
from .records import InboxItem, Record, copy_json, digest
from .store import StoredRecord


class TransitionError(ValueError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise TransitionError(reason)


@dataclass
class Attempt:
    plan: Record
    state: str = "planned"
    dispatch: Record | None = None
    wait: dict[str, Any] | None = None
    child_handle: dict[str, Any] | None = None
    unknown: Record | None = None
    result: Record | None = None
    context: str | None = None
    unknown_consumed: bool = False
    approval: str | None = None
    superseded: bool = False
    prepared: Record | None = None
    approval_answer: Record | None = None

    @property
    def execution_plan(self) -> Record:
        if self.prepared is None:
            return self.plan
        p = self.prepared._payload
        return replace(
            self.plan,
            payload=self.plan._payload
            | {
                **{
                    key: p[key] for key in ("request", "request_digest", "tool", "op_kind", "provider_binding", "idempotency_key")
                },
                "budget_admission": {"hook_result": p["hook_result"], "shared_state": p["shared_state"]},
            },
        )

    @property
    def started(self) -> bool:
        return self.dispatch is not None


@dataclass
class Operation:
    turn_id: str
    kind: str
    attempts: dict[int, Attempt] = field(default_factory=dict)
    selected_attempt: int | None = None

    @property
    def state(self) -> str:
        return self.attempts[max(self.attempts)].state


@dataclass
class Turn:
    start: Record
    ended: bool = False
    cancelled: bool = False
    suspended: bool = False
    wait: Record | None = None


@dataclass
class ExecutionState:
    session_id: str | None = None
    operations: dict[str, Operation] = field(default_factory=dict)
    turns: dict[str, Turn] = field(default_factory=dict)
    active_turn_id: str | None = None
    closed: bool = False
    archived: bool = False
    cancel_requested: bool = False
    suspend_requested: bool = False
    phase: str = "idle"
    next_drive_ms: int | None = None
    terminal_seq: int = 0
    applied_inputs: dict[str, Record] = field(default_factory=dict)
    admitted_inputs: set[str] = field(default_factory=set)
    usage_observations: dict[str, int] = field(default_factory=dict)
    usage_values: dict[str, Record] = field(default_factory=dict)
    compactions: list[Record] = field(default_factory=list)

    @property
    def waits(self) -> dict[tuple[str, int], dict[str, Any]]:
        return {
            (oid, number): attempt.wait
            for oid, op in self.operations.items()
            for number, attempt in op.attempts.items()
            if attempt.state == "parked" and attempt.wait is not None
        }


def _input(state: ExecutionState, record: Record, consumed: dict[str, InboxItem]) -> None:
    p = record._payload
    item = InboxItem(**p["input"])
    require(
        item.input_id in consumed and consumed[item.input_id].encode() == item.encode(), "input not consumed with identical bytes"
    )
    require(item.input_id not in state.applied_inputs, "input applied twice")
    state.applied_inputs[item.input_id] = record
    if p["disposition"] != "applied":
        return
    tid = item.target_turn_id
    if tid is not None:
        require(tid == record.turn_id, "input targets another turn")
    session_control = item.kind == "control" and item.payload["action"] in {"close", "archive"}
    evidence_input = item.kind in {"deferred_result", "provider_evidence", "child_result"}
    if evidence_input:
        require(tid in state.turns, "evidence targets an unknown turn")
        op = state.operations.get(item.payload["operation_id"])
        require(op is not None and op.turn_id == tid, "evidence operation mismatch")
        assert op is not None
        if item.kind == "child_result":
            attempt = op.attempts.get(item.payload["attempt"])
            handle = attempt.child_handle if attempt else None
            require(handle is not None, "child result has no child handle")
            assert handle is not None
            target = handle["delivery_target"]
            require(
                handle["session_id"] == item.payload["session_id"]
                and handle["turn_id"] == item.payload["turn_id"]
                and target["session_id"] == state.session_id
                and target["turn_id"] == tid
                and target["generation"] == item.generation
                and target["operation_id"] == item.payload["operation_id"]
                and target["attempt"] == item.payload["attempt"],
                "child result target mismatch",
            )
        else:
            number = item.payload["attempt"]
            require(number in op.attempts, "evidence attempt mismatch")
            plan = op.attempts[number].execution_plan._payload
            require(item.payload["request_digest"] == plan["request_digest"], "evidence request mismatch")
            if item.kind == "deferred_result":
                require(item.payload["provider_binding"] == plan["provider_binding"], "evidence provider mismatch")
    if session_control and tid is not None:
        require(tid == state.active_turn_id, "session control targets an inactive turn")
    if item.kind not in {"user", "follow_up"} and not session_control and not evidence_input:
        require(tid == state.active_turn_id and tid in state.turns, "applied input needs active target turn")
    if tid in state.turns and item.generation is not None:
        require(item.generation == state.turns[tid].start._payload["generation"], "input generation mismatch")
    if item.kind == "control":
        action = item.payload["action"]
        control_tid = tid if tid is not None else state.active_turn_id
        turn = state.turns.get(control_tid) if control_tid is not None else None
        if action == "archive":
            state.archived = True
        elif action == "close":
            state.closed = True
            if turn is not None:
                turn.cancelled = True
        else:
            require(turn is not None, "turn control needs a turn")
            assert turn is not None
            if action in {"cancel", "abort"}:
                turn.cancelled = True
            else:
                require(not turn.cancelled, "cannot resume or suspend a cancelled turn")
                turn.suspended = action == "suspend"
    if item.kind == "user" and tid in state.turns and state.turns[tid].wait is not None:
        waiting = state.turns[tid].wait
        assert waiting is not None
        content = item.payload["content"]
        require(
            isinstance(content, dict)
            and content.get("interaction_id") == waiting._payload["interaction_id"]
            and p["target_wait_id"] == waiting._payload["interaction_id"]
            and p["target_operation_id"] is None,
            "no-tool reply identity mismatch",
        )
        state.turns[tid].wait = None
    if item.kind == "approval_answer":
        answer = item.payload
        op = state.operations.get(answer["operation_id"])
        require(op is not None and op.turn_id == tid and answer["attempt"] in op.attempts, "approval operation mismatch")
        assert op is not None
        attempt = op.attempts[answer["attempt"]]
        require(attempt.state == "parked" and attempt.wait is not None, "approval has no wait")
        assert attempt.wait is not None
        handle = attempt.wait["handle"]
        require(
            handle["kind"] == "approval"
            and handle["request_id"] == answer["request_id"]
            and handle["request_digest"] == answer["request_digest"]
            and handle["scope"] == answer["scope"],
            "approval identity or scope mismatch",
        )
        require(
            p["target_operation_id"] == answer["operation_id"] and p["target_wait_id"] == answer["request_id"],
            "approval application target mismatch",
        )
        require(attempt.approval is None, "approval already answered")
        attempt.approval = answer["decision"]
        attempt.approval_answer = record


def _plan(state: ExecutionState, record: Record) -> None:
    p, oid, tid, number = record._payload, record.operation_id, record.turn_id, record.attempt
    assert oid is not None and tid is not None and number is not None
    require(tid == state.active_turn_id, "plan outside active turn")
    turn = state.turns[tid]
    require(
        not turn.cancelled and not turn.suspended and not state.closed and turn.wait is None,
        "plan after cancel/suspend/close/wait",
    )
    op = state.operations.get(oid)
    if op is None:
        require(number == 1, "first attempt must be 1")
        op = Operation(tid, p["op_kind"])
        state.operations[oid] = op
    else:
        require(op.turn_id == tid and op.kind == p["op_kind"], "operation identity changed")
        previous = op.attempts[max(op.attempts)]
        require(number == max(op.attempts) + 1 and previous.state == "unknown", "retry requires previous unknown")
        require(previous.unknown is not None and previous.unknown._payload["retry"] == "retry", "retry not authorized")
        require(op.selected_attempt is None, "retry after adopted result")
        require(
            p["request_digest"] == previous.execution_plan._payload["request_digest"]
            and p["provider_binding"] == previous.execution_plan._payload["provider_binding"]
            and p["context_version"] == previous.plan._payload["context_version"]
            and p["purpose"] == previous.plan._payload["purpose"],
            "retry changed request, context, purpose or provider",
        )
        if op.kind == "model":
            require(number <= 2, "model retry limit exceeded")
        due = previous.unknown._payload["retry_at_ms"] if previous.unknown else None
        require(due is None or (p["not_before_ms"] or 0) >= due, "retry before not-before")
    for dependency in p["dependencies"]:
        require(dependency in state.operations and state.operations[dependency].turn_id == tid, "unknown dependency")
        require(dependency != oid, "operation depends on itself")
    for ref in p["consumed_unknowns"]:
        source = state.operations.get(ref["operation_id"])
        require(op.kind == "model" and source is not None and ref["attempt"] in source.attempts, "unknown consumption reference")
        assert source is not None
        source_attempt = source.attempts[ref["attempt"]]
        require(source.turn_id == tid and source_attempt.state == "unknown", "request did not consume an unknown")
        source_attempt.unknown_consumed = True
    op.attempts[number] = Attempt(record, prepared=previous.prepared if number > 1 else None)


def _operation(state: ExecutionState, record: Record) -> None:
    p, oid, number = record._payload, record.operation_id, record.attempt
    assert oid is not None and number is not None
    op = state.operations.get(oid)
    require(op is not None and op.turn_id == record.turn_id and number in op.attempts, "operation/attempt/turn mismatch")
    assert op is not None
    attempt, turn = op.attempts[number], state.turns[op.turn_id]
    if record.kind == "op_prepared":
        require(op.kind != "model" and attempt.state == "planned" and attempt.prepared is None, "invalid hook preparation")
        require(
            op.turn_id == state.active_turn_id and not turn.cancelled and not turn.suspended, "prepare outside executable turn"
        )
        require(p["request"]["id"] == attempt.plan._payload["request"]["id"], "hook changed tool call identity")
        require(
            p["tool"] == turn.start._payload["definition"]["capabilities"].get(p["request"]["name"], {}),
            "hook changed frozen capability",
        )
        require(p["op_kind"] == ("interaction" if p["request"]["name"] == "ask_user" else "tool"), "hook kind mismatch")
        attempt.prepared = record
    elif record.kind == "op_started":
        require(
            op.turn_id == state.active_turn_id
            and not turn.cancelled
            and not turn.suspended
            and not state.closed
            and turn.wait is None,
            "dispatch outside executable turn",
        )
        require(number == max(op.attempts) and op.selected_attempt is None, "superseded dispatch")
        ready = attempt.state == "planned" or (attempt.state == "parked" and attempt.approval in {"approve", "allow_session"})
        require(ready, "start requires plan or approved before-dispatch wait")
        require(
            all(state.operations[dep].state == "completed" for dep in attempt.plan._payload["dependencies"]),
            "unfinished dependency",
        )
        attempt.state, attempt.dispatch, attempt.wait = "started", record, None
    elif record.kind == "op_parked":
        require(op.turn_id == state.active_turn_id and not turn.cancelled, "park outside active turn")
        before = p["phase"] == "before_dispatch"
        require(attempt.state == ("planned" if before else "started"), "invalid park transition")
        handle = p["handle"]
        require(not before or handle["kind"] in {"approval", "user", "child"}, "provider park needs dispatch")
        if handle["kind"] == "user":
            require(
                attempt.execution_plan._payload["op_kind"] == "interaction" and before,
                "user wait requires undispatched interaction",
            )
        if handle["kind"] == "approval":
            require(
                before and handle["request_digest"] == attempt.execution_plan._payload["request_digest"],
                "approval request mismatch",
            )
        if handle["kind"] == "provider":
            require(
                handle["operation_id"] == oid
                and handle["attempt"] == number
                and handle["request_digest"] == attempt.execution_plan._payload["request_digest"]
                and handle["provider"] == attempt.execution_plan._payload["provider_binding"],
                "provider handle mismatch",
            )
        if handle["kind"] == "child":
            target = handle["delivery_target"]
            require(
                not before
                and target
                == {
                    "session_id": state.session_id,
                    "turn_id": op.turn_id,
                    "generation": turn.start._payload["generation"],
                    "operation_id": oid,
                    "attempt": number,
                },
                "child delivery target mismatch",
            )
            attempt.child_handle = copy_json(handle)
        attempt.state, attempt.wait = "parked", copy_json(p)
    elif record.kind == "op_unknown":
        require(
            attempt.state == "started" or (attempt.state == "parked" and attempt.started and turn.cancelled),
            "only dispatched work or cancelled provider waits can become unknown",
        )
        require(bool(p["dispatch_evidence"]), "unknown needs dispatch evidence")
        attempt.state, attempt.unknown = "unknown", record
    elif record.kind == "op_completed":
        require(attempt.state in {"planned", "started", "parked", "unknown"}, "result after completion")
        require(
            p["request_digest"] == attempt.execution_plan._payload["request_digest"]
            and p["provider_binding"] == attempt.execution_plan._payload["provider_binding"],
            "result request/provider mismatch",
        )
        require(p["execution_started"] == attempt.started, "execution_started disagrees with dispatch")
        if attempt.started:
            require(bool(p["evidence"]), "dispatched result requires authenticated evidence reference")
        if attempt.child_handle and not attempt.child_handle["background"]:
            completions = [
                r._payload["input"]["payload"]
                for r in state.applied_inputs.values()
                if r._payload["disposition"] == "applied"
                and r._payload["input"]["kind"] == "child_result"
                and r._payload["input"]["payload"]["operation_id"] == oid
                and r._payload["input"]["payload"]["attempt"] == number
            ]
            require(
                any(f"child/{c['terminal_seq']}/{c['terminal_digest']}" in p["evidence"] for c in completions),
                "child completion needs terminal input",
            )
        if attempt.wait and attempt.wait["handle"]["kind"] == "provider":
            require(attempt.wait["handle"]["evidence"] in p["evidence"], "result does not bind parked evidence")
        later_started = any(a.started for n, a in op.attempts.items() if n > number)
        audit = turn.cancelled or turn.ended or op.turn_id != state.active_turn_id or later_started
        context = "audit" if audit else "correction" if attempt.unknown_consumed else "normal"
        require(p["context"] == context, f"result context must be {context}")
        attempt.state, attempt.result, attempt.context, attempt.wait = "completed", record, context, None
        if not audit:
            op.selected_attempt = number
            for n, successor in op.attempts.items():
                if n > number:
                    require(not successor.started, "cannot supersede dispatched attempt")
                    successor.state, successor.superseded, successor.wait = "completed", True, None


def _schedule(state: ExecutionState) -> None:
    state.cancel_requested = state.suspend_requested = False
    state.next_drive_ms = None
    if state.active_turn_id is None:
        state.phase = "closed" if state.closed else "idle"
        if not state.closed and any(
            input_id not in state.admitted_inputs
            and applied._payload["disposition"] == "queued"
            and applied._payload["input"]["kind"] in {"user", "follow_up"}
            for input_id, applied in state.applied_inputs.items()
        ):
            state.next_drive_ms = 0
        return
    turn = state.turns[state.active_turn_id]
    state.cancel_requested, state.suspend_requested = turn.cancelled, turn.suspended
    if turn.cancelled:
        state.phase, state.next_drive_ms = "active", 0
        return
    if turn.suspended:
        state.phase = "suspended"
        return
    if turn.wait is not None:
        state.phase = "parked"
        return
    due: list[int] = []
    waiting = False
    unresolved = False
    for op in state.operations.values():
        if op.turn_id != state.active_turn_id:
            continue
        attempt = op.attempts[max(op.attempts)]
        if attempt.state == "completed":
            continue
        unresolved = True
        if attempt.state == "planned":
            if all(state.operations[dep].state == "completed" for dep in attempt.plan._payload["dependencies"]):
                due.append(attempt.plan._payload["not_before_ms"] or 0)
            else:
                waiting = True
        elif attempt.state == "started":
            due.append(0)
        elif attempt.state == "unknown":
            assert attempt.unknown is not None
            if attempt.unknown._payload["retry"] == "retry":
                due.append(attempt.unknown._payload["retry_at_ms"] or 0)
            elif attempt.unknown._payload["retry"] == "stop":
                due.append(0)  # Finalization still has to be committed.
            else:
                waiting = True
        else:
            assert attempt.wait is not None
            waiting = True
            due.extend(t for t in (attempt.wait["poll_at_ms"], attempt.wait["deadline_ms"]) if t is not None)
            if attempt.approval is not None:
                due.append(0)
    if not unresolved:
        due.append(0)  # Next model step or unfinished turn finalization.
    state.phase = "parked" if waiting and not due else "active"
    state.next_drive_ms = min(due) if due else None


def _compacted(state: ExecutionState, record: Record, history: list[StoredRecord]) -> None:
    p, tid = record._payload, record.turn_id
    require(tid == state.active_turn_id and tid in state.turns, "compaction outside active turn")
    assert tid is not None
    require(not state.turns[tid].cancelled and not state.turns[tid].suspended, "compaction after cancel/suspend")
    source = project_context(tuple(history), state)
    require(digest([m.to_dict() for m in source]) == p["source_digest"], "compaction source mismatch")
    replacement = [Message.from_dict(m) for m in p["replacement"]]
    ids = message_ids(source)
    definition = state.turns[tid].start._payload["definition"]
    manager = MemoryManager(**definition["memory_settings"])
    manager.recovery_tool_available = any(t["function"]["name"] == "read_file" for t in definition["tools"])
    if p["mode"] == "micro":
        require(p["summary_operation_id"] is None and len(source) == len(replacement), "invalid micro replacement")
        removed = {i for i, (a, b) in enumerate(zip(source, replacement, strict=True)) if a != b}
        require(bool(removed), "empty micro replacement")
        prune = manager.plan_microcompaction(
            source,
            cycle_index=1,
            current_tokens=count_messages_tokens([m.to_openai_message() for m in source], model=manager.model),
        )
        candidates = {c.message_index: c for c in prune.candidates} if prune else {}
        require(removed <= candidates.keys(), "micro changed ineligible messages")
        for i in removed:
            before, after, candidate = source[i], replacement[i], candidates[i]
            artifact = after.artifact_ref
            require(artifact is not None, "micro requires archived tool body")
            assert artifact is not None
            if candidate.existing_artifact is not None:
                require(artifact == candidate.existing_artifact, "micro changed existing artifact")
            else:
                data = before.content.encode("utf-8")
                require(
                    artifact.sha256 == sha256(data).hexdigest() and artifact.size_bytes == len(data),
                    "micro artifact does not bind original body",
                )
            excerpt = before.metadata.get(EXCERPT_METADATA_KEY)
            marker = build_compacted_tool_content(
                excerpt if isinstance(excerpt, str) else before.content,
                artifact_path=artifact.path,
                tool_name=candidate.tool_name,
                excerpt_head_chars=manager.tool_result_excerpt_head,
                excerpt_tail_chars=manager.tool_result_excerpt_tail,
            )
            require(
                after == replace_with_compacted_marker(before, candidate, artifact=artifact, marker=marker),
                "micro changed message skeleton or recovery marker",
            )
    else:
        op = state.operations.get(p["summary_operation_id"])
        require(
            op is not None and op.turn_id == tid and op.kind == "model" and op.selected_attempt is not None,
            "summary has no adopted result",
        )
        assert op is not None and op.selected_attempt is not None
        a = op.attempts[op.selected_attempt]
        require(
            a.plan._payload["purpose"] == "compaction" and a.result is not None and a.context == "normal", "not a summary receipt"
        )
        assert a.result is not None
        meta = a.plan._payload["request"]["metadata"]
        require(meta["source_digest"] == p["source_digest"] and meta["mode"] == p["mode"], "summary source or mode mismatch")
        plan = manager.plan_summary(source, keep_recent=meta["keep_recent"])
        require(plan is not None, "invalid summary prefix")
        assert plan is not None
        require(a.plan._payload["request"]["messages"] == [Message("user", plan.prompt).to_dict()], "summary prompt mismatch")
        expected, accepted = manager.accept_summary(plan, a.result._payload["result"]["content"], notify=False)
        require(accepted and expected == replacement, "replacement differs from accepted receipt")
        prefix = {id(m) for m in [*plan.previous, *plan.prefix]}
        removed = {i for i, m in enumerate(source) if id(m) in prefix}
    require(
        p["prefix_ids"] == [v for i, v in enumerate(ids) if i in removed]
        and p["tail_ids"] == [v for i, v in enumerate(ids) if i not in removed],
        "compaction identities mismatch",
    )
    require(p["evidence_manifest"] == manager.compaction_evidence(replacement), "compaction evidence mismatch")
    state.compactions.append(record)


class Fold:
    """Disposable validated prefix; fork before extending a provisional transaction."""

    def __init__(self) -> None:
        self.state = ExecutionState()
        self.seen: dict[str, bytes] = {}
        self.consumed: dict[str, InboxItem] = {}
        self.history: list[StoredRecord] = []
        self._shared_operations: set[str] = set()
        self._shared_turns: set[str] = set()

    def fork(self) -> Fold:
        result = Fold()
        result.state = copy(self.state)
        result.state.operations = self.state.operations.copy()
        result.state.turns = self.state.turns.copy()
        result._shared_operations = set(self.state.operations)
        result._shared_turns = set(self.state.turns)
        result.state.applied_inputs = self.state.applied_inputs.copy()
        result.state.admitted_inputs = self.state.admitted_inputs.copy()
        result.state.usage_observations = self.state.usage_observations.copy()
        result.state.usage_values = self.state.usage_values.copy()
        result.state.compactions = self.state.compactions.copy()
        result.seen, result.consumed = self.seen.copy(), self.consumed.copy()
        result.history = self.history.copy()
        return result

    def _own_operation(self, oid: str | None) -> None:
        if oid in self._shared_operations:
            assert oid is not None
            op = self.state.operations[oid]
            self.state.operations[oid] = replace(op, attempts={n: Attempt(**vars(a)) for n, a in op.attempts.items()})
            self._shared_operations.remove(oid)

    def _own_turn(self, tid: str | None) -> None:
        if tid in self._shared_turns:
            assert tid is not None
            self.state.turns[tid] = copy(self.state.turns[tid])
            self._shared_turns.remove(tid)

    def snapshot(self) -> ExecutionState:
        result = self.fork()
        for oid in result.state.operations:
            result._own_operation(oid)
            for attempt in result.state.operations[oid].attempts.values():
                attempt.wait = copy_json(attempt.wait)
                attempt.child_handle = copy_json(attempt.child_handle)
        for tid in result.state.turns:
            result._own_turn(tid)
        return result.state

    def extend(
        self, records: Iterable[Record], *, consumed_inputs: Iterable[InboxItem] = (), bodies: Iterable[bytes] | None = None
    ) -> ExecutionState:
        state, seen, consumed = self.state, self.seen, self.consumed
        for item in consumed_inputs:
            item.encode()
            require(item.input_id not in consumed, "duplicate consumed input identity")
            consumed[item.input_id] = item
        seq, history = len(self.history), self.history
        encoded = iter(bodies) if bodies is not None else None
        for record in records:
            body = next(encoded) if encoded is not None else record.encode()
            if record.record_id in seen:
                require(seen[record.record_id] == body, "record identity conflict")
                continue
            seen[record.record_id] = body
            seq += 1
            history.append(StoredRecord(record, seq, "", 0, 0))
            p, kind, tid = record._payload, record.kind, record.turn_id
            if kind == "session_created":
                require(seq == 1, "session already created")
                state.session_id = record.session_id
                continue
            require(state.session_id == record.session_id, "missing creation or wrong session")
            if kind == "turn_started":
                require(
                    tid is not None and tid not in state.turns and state.active_turn_id is None and not state.closed,
                    "turn start requires idle open session and a new turn identity",
                )
                assert tid is not None
                require(len(p["input_ids"]) == len(set(p["input_ids"])), "duplicate turn input")
                for input_id in p["input_ids"]:
                    require(input_id not in state.admitted_inputs, "input already admitted to a turn")
                    require(
                        input_id in state.applied_inputs
                        and state.applied_inputs[input_id]._payload["disposition"] in {"applied", "queued"},
                        "turn input not applied",
                    )
                    incoming = state.applied_inputs[input_id]._payload["input"]
                    require(
                        incoming["target_turn_id"] in {None, tid} and incoming["generation"] in {None, p["generation"]},
                        "turn input target mismatch",
                    )
                state.admitted_inputs.update(p["input_ids"])
                state.turns[tid], state.active_turn_id = Turn(record), tid
            elif kind == "input_applied":
                incoming = p["input"]
                self._own_turn(incoming["target_turn_id"])
                self._own_turn(state.active_turn_id)
                self._own_operation(incoming["payload"].get("operation_id"))
                _input(state, record, consumed)
            elif kind == "op_planned":
                self._own_operation(record.operation_id)
                for ref in p["consumed_unknowns"]:
                    self._own_operation(ref["operation_id"])
                _plan(state, record)
            elif kind.startswith("op_"):
                self._own_operation(record.operation_id)
                _operation(state, record)
            elif kind == "turn_parked":
                require(tid == state.active_turn_id and tid in state.turns, "wait outside active turn")
                assert tid is not None
                turn = state.turns[tid]
                require(turn.wait is None and not turn.cancelled and not turn.suspended, "invalid turn wait")
                source = state.operations.get(p["source_operation_id"])
                require(
                    source is not None
                    and source.turn_id == tid
                    and source.kind == "model"
                    and source.selected_attempt == p["source_attempt"],
                    "wait source mismatch",
                )
                assert source is not None
                result = source.attempts[p["source_attempt"]].result
                require(
                    result is not None
                    and not result._payload["result"].get("tool_calls")
                    and not result._payload["result"].get("error_code")
                    and source.attempts[p["source_attempt"]].plan._payload["purpose"] == "primary"
                    and p["question"] == result._payload["result"]["content"]
                    and turn.start._payload["definition"]["task"]["no_tool_policy"] == "wait_user",
                    "invalid no-tool wait source",
                )
                require(
                    all(op.state == "completed" for op in state.operations.values() if op.turn_id == tid),
                    "wait has unfinished operations",
                )
                self._own_turn(tid)
                state.turns[tid].wait = record
            elif kind == "turn_ended":
                require(tid == state.active_turn_id and tid in state.turns, "end outside active turn")
                assert tid is not None
                require(
                    not state.turns[tid].cancelled or p["status"] in {"cancelled", "aborted"},
                    "cancelled turn cannot end successfully",
                )
                attempts = [
                    (oid, attempt)
                    for oid, op in state.operations.items()
                    if op.turn_id == tid
                    for attempt in op.attempts.values()
                    if attempt.state != "completed"
                ]
                require(all(attempt.started for _, attempt in attempts), "undispatched work requires explicit closure")
                require(
                    p["status"] != "completed"
                    or not any(a.child_handle and not a.child_handle["background"] for _, a in attempts),
                    "HasLiveDescendants: waiting child has no terminal evidence",
                )
                unresolved = {oid for oid, _ in attempts}
                require(set(p["unconfirmed_operations"]) == unresolved, "terminal unresolved operations mismatch")
                for ref in p["adopted_results"]:
                    op = state.operations.get(ref["operation_id"])
                    require(
                        op is not None and op.turn_id == tid and op.selected_attempt == ref["attempt"],
                        "terminal result not adopted",
                    )
                self._own_turn(tid)
                state.turns[tid].ended = True
                state.active_turn_id, state.terminal_seq = None, seq
            elif kind == "usage_observed":
                require(tid is None or tid in state.turns, "usage for unknown turn")
                previous = state.usage_observations.get(p["meter_id"], 0)
                require(p["observation"] > previous, "usage observation must increase")
                state.usage_observations[p["meter_id"]] = p["observation"]
                state.usage_values[p["meter_id"]] = record
            elif kind == "context_compacted":
                _compacted(state, record, history[:-1])
        require(set(consumed) == set(state.applied_inputs), "consumed input without input_applied")
        _schedule(state)
        return state


def fold(records: Iterable[Record], *, consumed_inputs: Iterable[InboxItem] = ()) -> ExecutionState:
    return Fold().extend(records, consumed_inputs=consumed_inputs)
