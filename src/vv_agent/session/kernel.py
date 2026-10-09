"""Synchronous, lease-fenced driver. Durable facts live only in the session log."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from vv_agent.budget import BudgetEvaluator, BudgetExhaustion, RunBudgetLimits
from vv_agent.events import _project_provider_stream_payload
from vv_agent.llm.errors import is_prompt_too_long_error
from vv_agent.llm.vv_llm_client import VvLlmClient
from vv_agent.memory.microcompact import is_microcompacted_tool_content
from vv_agent.output_validation import OutputRepairRequest
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.lifecycle import read_after_cycle_disallowed_tools
from vv_agent.runtime.tool_results import apply_tool_use_behavior, build_skipped_result
from vv_agent.tools.orchestrator import ToolOrchestrator
from vv_agent.types import AgentTask, LLMResponse, Message, ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus

from .approval import resolve_approval
from .children import cancel_children, child_handles, child_outcome, create_child, verify_completion
from .compaction import compact_context, finish_summary
from .context import project_context
from .delegation import admitted_result, assemble, result_for_children
from .lifecycle import after_cycle
from .memory import save_projection
from .output import prepare_output, serializable_output
from .providers import Accepted, Definitive, Outcome, Unknown
from .records import InboxItem, Record, _RetainedTools, copy_json, digest, make_record
from .reducer import Attempt, ExecutionState, Fold
from .runtime import Runtime, budget, model_usage, request_from_dict
from .store import Conflict, Lease, LeaseLost, SequenceConflict, SessionStore, SessionTx, StoredRecord


def read_state(store: SessionStore, sid: str) -> tuple[ExecutionState, tuple[StoredRecord, ...], int]:
    return store.read_state(sid)


class _Scope:
    def __init__(self, lease: Lease, runtime: Runtime):
        self.lease, self.runtime = lease, runtime
        self.token = CancellationToken()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.lost = False
        self.turn_id: str | None = None
        self.generation: int | None = None
        self.thread = threading.Thread(target=self._heartbeat, daemon=True, name="session-heartbeat")

    def poll(self, store: SessionStore):
        with self.lock:
            if self.lost:
                raise LeaseLost("heartbeat lost ownership")
            poll = store.renew(self.lease, ttl_ms=self.runtime.ttl_ms)
            self.lease = poll.lease
        for item in poll.controls:
            if (
                item.target_turn_id in {None, self.turn_id}
                and item.generation in {None, self.generation}
                and item.payload["action"] in {"cancel", "abort", "close", "suspend"}
            ):
                self.token.cancel(item.payload["action"])
        return poll

    def _heartbeat(self) -> None:
        try:
            with self.runtime.heartbeat_store() as store:
                while not self.stop.wait(self.runtime.heartbeat_seconds):
                    self.poll(store)
        except Exception:
            self.lost = True
            self.token.cancel("lease_lost")


def _invoke(scope: _Scope, callback: Callable[[], Outcome], timeout: float) -> Outcome:
    done = threading.Event()
    result: list[Outcome] = []

    def run() -> None:
        try:
            result.append(callback())
        except Exception as exc:
            result.append(Unknown(f"{type(exc).__name__}: external outcome unavailable"))
        finally:
            done.set()

    worker = threading.Thread(target=run, daemon=True, name="session-external-call")
    worker.start()
    deadline = time.monotonic() + timeout
    while not done.wait(0.02):
        if scope.token.cancelled:
            if done.wait(scope.runtime.cancellation_grace):
                break
            return Unknown("cancellation without confirmed external stop")
        if time.monotonic() >= deadline:
            return Unknown("external call timeout")
    worker.join()
    return result[0] if result else Unknown("external worker exited without a receipt")


class _Driver:
    def __init__(self, store: SessionStore, sid: str, runtime: Runtime, scope: _Scope):
        self.store, self.sid, self.runtime, self.scope = store, sid, runtime, scope
        self.load_state()
        self.polled: set[tuple[str, int]] = set()
        self.active_since = time.monotonic_ns()
        self.active_tid: str | None = None

    def load_state(self) -> None:
        self.state, self.records, self.watermark = read_state(self.store, self.sid)
        self.fold = Fold()
        self.fold.state = self.state
        self.fold.history = list(self.records)
        self.fold.seen = {r.record.record_id: r.record.encode() for r in self.records}
        self.fold.consumed = {key: InboxItem(**r._payload["input"]) for key, r in self.state.applied_inputs.items()}

    def refresh(self) -> None:
        page = self.store.read(self.sid, after_seq=self.records[-1].seq)
        head = page.head_seq
        if head < self.records[-1].seq:
            self.load_state()
        else:
            self.watermark = page.inbox_seq
            self.extend(page.records)
            while self.records[-1].seq < head:
                if not page.records:
                    raise Conflict("log sequence gap")
                page = self.store.read(self.sid, after_seq=self.records[-1].seq, through_seq=head)
                self.extend(page.records)
        self.bind_turn()

    def extend(self, records: tuple[StoredRecord, ...]) -> None:
        if not records:
            return
        if any(r.seq != self.records[-1].seq + index for index, r in enumerate(records, 1)):
            raise Conflict("log sequence gap")
        self.state = self.fold.extend(
            (r.record for r in records),
            consumed_inputs=(InboxItem(**r.record._payload["input"]) for r in records if r.record.kind == "input_applied"),
        )
        self.records += records

    def bind_turn(self) -> None:
        tid = self.state.active_turn_id
        if tid != self.active_tid:
            self.active_since = time.monotonic_ns()
            self.active_tid = tid
        self.scope.turn_id = tid
        self.scope.generation = self.state.turns[tid].start._payload["generation"] if tid else None

    def record(self, kind: str, payload: dict[str, Any], plan: Record | None = None, *, tid: str | None = None) -> Record:
        return make_record(
            kind,
            session_id=self.sid,
            turn_id=plan.turn_id if plan else tid or self.state.active_turn_id,
            operation_id=plan.operation_id if plan else None,
            attempt=plan.attempt if plan else None,
            payload=payload,
        )

    def commit(
        self,
        records: list[Record],
        consumed: tuple[str, ...] = (),
        *,
        guarded: bool = False,
        prepare: Callable[[SessionTx], list[Record]] | None = None,
    ) -> None:
        tid = self.state.active_turn_id
        if tid is not None:
            meter = f"active/{tid}/{self.scope.lease.epoch}"
            cost = None
            if self.runtime.config.host_cost_meter is not None:
                with suppress(Exception):
                    observed = self.runtime.config.host_cost_meter.read()
                    cost = observed.to_dict() if observed else None
            records = [
                self.record(
                    "usage_observed",
                    {
                        "meter_id": meter,
                        "observation": self.state.usage_observations.get(meter, 0) + 1,
                        "mode": "cumulative",
                        "source": "kernel-active-scope",
                        "usage": {"elapsed_ms": (time.monotonic_ns() - self.active_since) // 1000000, "host_cost": cost},
                    },
                ),
                *records,
            ]
        with self.scope.lock:
            if self.scope.lost:
                raise LeaseLost("heartbeat lost ownership")
            with self.store.atomic() as tx:
                if prepare is not None:
                    records.extend(prepare(tx))
                self.runtime.hook("before_commit", records[-1] if records else None)
                receipt = tx.append(
                    self.sid,
                    lease=self.scope.lease,
                    expected_seq=self.records[-1].seq,
                    commit_id=digest({"record_ids": [r.record_id for r in records], "consumed": list(consumed)}),
                    records=tuple(records),
                    consume_input_ids=consumed,
                    expected_inbox_seq=self.watermark if guarded else None,
                )
        self.runtime.hook("after_commit", records[-1] if records else None)
        if receipt.replayed:
            self.refresh()
        else:
            self.extend(receipt.records)
            self.watermark = receipt.inbox_seq
            self.bind_turn()

    def boundary(self, stage: str, identity: str) -> Record | None:
        return self.state.boundaries.get((self.state.active_turn_id or "", stage, identity))

    def boundary_record(
        self,
        stage: str,
        identity: str,
        data: dict[str, Any],
        *,
        source_operation_id: str | None = None,
        source_digest: str | None = None,
    ) -> Record:
        return self.record(
            "boundary_recorded",
            {
                "stage": stage,
                "boundary_id": identity,
                "source_operation_id": source_operation_id,
                "source_digest": source_digest,
                "data": data,
            },
        )

    def live_budget(self) -> BudgetEvaluator | None:
        tid = self.state.active_turn_id
        assert tid is not None
        evaluator = budget(self.state, tid)
        if evaluator is None:
            return None
        current_meter = f"active/{tid}/{self.scope.lease.epoch}"
        saved = self.state.usage_values.get(current_meter)
        retained_ms = saved._payload["usage"].get("elapsed_ms", 0) if saved else 0
        initial = evaluator.snapshot()
        initial = replace(
            initial, elapsed_ms=initial.elapsed_ms + max(0, (time.monotonic_ns() - self.active_since) // 1000000 - retained_ms)
        )
        return BudgetEvaluator(evaluator.limits, initial_usage=initial, host_cost_meter=self.runtime.config.host_cost_meter)

    def budget_record(
        self,
        identity: str,
        evaluator: BudgetEvaluator,
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

    def task(self) -> AgentTask:
        assert self.state.active_turn_id is not None
        return self.state.turns[self.state.active_turn_id].start.task()

    def plan(
        self,
        oid: str,
        request: dict[str, Any],
        kind: str,
        *,
        number: int = 1,
        dependencies: list[str] | None = None,
        capability: dict[str, Any] | None = None,
        purpose: str = "primary",
    ) -> Record:
        client = self.runtime.model_route(purpose)[0] if kind == "model" else None
        if (
            kind == "model"
            and isinstance(client, VvLlmClient)
            and "endpoint_order" not in request["metadata"].get("vv_session", {})
        ):
            request = request | {
                "metadata": request["metadata"]
                | {
                    "vv_session": request["metadata"].get("vv_session", {})
                    | {"endpoint_order": [t.endpoint_id for t in client.ordered_targets(self.state.preferred_endpoint_id)]}
                }
            }
        if kind == "model" and self.state.active_turn_id is not None:
            source = self.state.turns[self.state.active_turn_id].start
            tools = request.get("tools")
            frozen = source._payload["definition"].get("tools")
            if (
                isinstance(tools, list)
                and isinstance(frozen, list)
                and len(tools) == len(frozen)
                and all(a is b for a, b in zip(tools, frozen, strict=True))
            ):
                request = request | {"tools": _RetainedTools(source)}
        binding = "model" if kind == "model" else request["name"] if request["name"] in self.runtime.providers else "function"
        return make_record(
            "op_planned",
            session_id=self.sid,
            turn_id=self.state.active_turn_id,
            operation_id=oid,
            attempt=number,
            payload={
                "op_kind": kind,
                "purpose": purpose if kind == "model" else None,
                "request": request,
                "request_digest": digest(request),
                "context_version": str(self.records[-1].seq),
                "dependencies": dependencies or [],
                "tool": capability,
                "idempotency_key": None if capability and capability.get("idempotency") == "unsupported" else f"{self.sid}/{oid}",
                "budget_admission": {},
                "not_before_ms": None,
                "provider_binding": binding,
                "consumed_unknowns": [
                    {"operation_id": key, "attempt": n}
                    for key, op in self.state.operations.items()
                    for n, a in op.attempts.items()
                    if kind == "model" and op.turn_id == self.state.active_turn_id and a.state == "unknown" and op.kind != "model"
                ],
            },
        )

    def completed(self, plan: Record, outcome: Definitive) -> list[Record]:
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
        return [result, *self.completion_effects(plan, outcome, context)]

    def completion_effects(self, plan: Record, outcome: Definitive, context: str) -> list[Record]:
        assert plan.operation_id is not None and plan.attempt is not None and plan.turn_id is not None
        records = []
        op = self.state.operations[plan.operation_id]
        turn = self.state.turns[plan.turn_id]
        if op.kind == "model" and context == "normal":
            if any(
                r._payload["data"]["exhaustion"]
                for (tid, stage, _), r in self.state.boundaries.items()
                if tid == plan.turn_id and stage == "budget"
            ):
                return records
            evaluator = self.live_budget()
            if evaluator and self.boundary("budget", f"{plan.operation_id}/complete/{plan.attempt}") is None:
                exhaustion = evaluator.model_call_complete(model_usage(outcome.usage))
                if self.scope.token.cancelled or (
                    self.runtime.config.cancellation_token is not None and self.runtime.config.cancellation_token.cancelled
                ):
                    exhaustion = None
                records.append(
                    self.budget_record(
                        f"{plan.operation_id}/complete/{plan.attempt}", evaluator, exhaustion, source=plan.operation_id
                    )
                )
                if exhaustion:
                    return records
                if plan._payload["purpose"] != "primary":
                    return records
                names = [call["name"] for call in outcome.result.get("tool_calls", [])]
                if names:
                    exhaustion = evaluator.preflight_tools(names)
                    records.append(
                        self.budget_record(
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
                    planned = self.plan(
                        oid,
                        call,
                        "interaction" if call["name"] == "ask_user" else "tool",
                        dependencies=[plan.operation_id],
                        capability=definition["capabilities"].get(call["name"], {}),
                    )
                    records.append(planned)
        elif op.kind != "model" and context == "normal":
            evaluator = self.live_budget()
            if evaluator is not None:
                before = evaluator.snapshot()
                exhaustion = evaluator.tool_batch_complete(operation_failed=outcome.result.get("status_code") == "ERROR")
                if self.scope.token.cancelled or (
                    self.runtime.config.cancellation_token is not None and self.runtime.config.cancellation_token.cancelled
                ):
                    exhaustion = None
                if evaluator.snapshot() != before or exhaustion:
                    records.append(
                        self.budget_record(
                            f"{plan.operation_id}/tool_complete/{plan.attempt}", evaluator, exhaustion, source=plan.operation_id
                        )
                    )
        return records

    def unknown(self, plan: Record, reason: str) -> Record:
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
            and self.scope.token.reason not in {"cancel", "abort", "close", "lease_lost"}
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

    def provider(self, plan: Record):
        return self.runtime.providers.get(plan._payload["provider_binding"], self.runtime.functions)

    def parked(self, plan: Record, handle: dict[str, Any], *, after: bool) -> Record:
        timeout = self.runtime.config.approval_timeout_seconds if handle["kind"] == "approval" else None
        now = self.scope.poll(self.store).db_now_ms if handle["kind"] == "provider" or timeout is not None else 0
        return self.record(
            "op_parked",
            {
                "phase": "after_dispatch" if after else "before_dispatch",
                "handle": handle,
                "poll_at_ms": now + self.runtime.poll_ms if handle["kind"] == "provider" else None,
                "deadline_ms": now + max(0, int(timeout * 1000)) if timeout is not None else None,
            },
            plan,
        )

    def pending_batch(self) -> bool:
        return any(
            op.turn_id == self.state.active_turn_id
            and (
                a.state in {"planned", "started", "parked"}
                or (a.state == "unknown" and a.unknown is not None and a.unknown._payload["retry"] == "retry")
            )
            for op in self.state.operations.values()
            for a in [op.attempts[max(op.attempts)]]
        )

    def apply_input(self) -> bool:
        for stored in self.store.peek_inbox(self.sid, through_input_seq=self.watermark):
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
            if item.kind == "steer" and (tid is None or (self.pending_batch() and target_tid == tid)):
                continue
            if item.kind == "child_result":
                target, extra, disposition, reason = self.apply_child(item)
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
                    or item.payload["request_digest"] != attempt.execution_plan._payload["request_digest"]
                    or not self.provider(attempt.execution_plan).authenticate(item, attempt.execution_plan)
                ):
                    disposition, reason = "rejected", "untrusted or mismatched provider evidence"
                elif item.kind == "provider_result":
                    if item.payload["provider_binding"] != attempt.execution_plan._payload["provider_binding"]:
                        disposition, reason = "rejected", "provider binding mismatch"
                    elif attempt.result is not None:
                        disposition = "noop" if attempt.result._payload["result"] == item.payload["result"] else "rejected"
                        reason = "retained result" if disposition == "noop" else "result conflict"
                    else:
                        extra = self.completed(
                            attempt.execution_plan, Definitive(item.payload["result"], tuple(item.payload["evidence"]))
                        )
                elif attempt.state == "started":
                    extra = [self.parked(attempt.execution_plan, item.payload["handle"], after=True)]
                elif attempt.wait and attempt.wait["handle"] == item.payload["handle"]:
                    disposition = "noop"
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
                    and self.scope.poll(self.store).db_now_ms >= attempt.wait["deadline_ms"]
                ):
                    disposition, reason = "rejected", "approval deadline expired"
                elif attempt and attempt.approval:
                    original = attempt.approval_answer
                    assert original is not None
                    equal = original._payload["input"]["payload"] == item.payload
                    disposition, reason = (
                        ("noop", "approval already resolved") if equal else ("rejected", "approval reply conflict")
                    )
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
                turn_wait = self.state.turns[tid].wait if tid else None
                if previous is not None:
                    equal = previous._payload["input"]["payload"] == item.payload
                    disposition, reason = ("noop", "retained user reply") if equal else ("rejected", "user reply conflict")
                elif turn_wait is not None and item.kind == "user":
                    wait = turn_wait._payload["interaction_id"]
                    if (
                        target_tid != tid
                        or not isinstance(content, dict)
                        or content.get("interaction_id") != wait
                        or "text" not in content
                    ):
                        disposition, reason = "rejected", "reply must identify the parked turn"
                else:
                    wait = None
                waits = [(oid, n, h) for (oid, n), h in self.state.waits.items() if h["handle"]["kind"] == "user"]
                if previous is not None or (turn_wait is not None and item.kind == "user"):
                    pass
                elif item.kind == "user" and tid and waits:
                    content = item.payload["content"]
                    match = next(
                        (
                            (oid, n, h)
                            for oid, n, h in waits
                            if isinstance(content, dict)
                            and content.get("interaction_id") == h["handle"]["interaction_id"]
                            and content.get("operation_id") == oid
                            and "text" in content
                            and target_tid == tid
                        ),
                        None,
                    )
                    if match is None:
                        disposition, reason = "rejected", "reply must identify the parked interaction"
                    else:
                        target, number, h = match
                        wait = h["handle"]["interaction_id"]
                        plan = self.state.operations[target].attempts[number].execution_plan
                        context = self.runtime.context(
                            self.task(),
                            plan,
                            self.scope.token,
                            shared_state=self.shared_state(),
                            store=self.store,
                            session_id=self.sid,
                        )
                        outcome = self.finalize_tool(
                            plan,
                            context,
                            Definitive(
                                ToolExecutionResult(
                                    tool_call_id=plan._payload["request"]["id"], content=str(content["text"])
                                ).to_dict(),
                                (),
                            ),
                        )
                        extra = self.completed(plan, outcome)
                elif tid and item.kind == "user":
                    disposition, reason = "rejected", "turn already active"
                else:
                    disposition = "queued"
            elif item.kind == "control":
                if item.payload["action"] not in {"archive", "close"} and (tid is None or target_tid != tid):
                    disposition, reason = "rejected", "control requires active target"
            elif item.kind == "steer":
                if tid is None and target_tid is None:
                    continue
                if tid is None or target_tid not in {None, tid}:
                    disposition, reason = "rejected", "steer requires active target"
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
            self.commit([applied, *extra], (item.input_id,), guarded=True)
            return True
        return False

    def apply_child(self, item: InboxItem) -> tuple[str, list[Record], str, str | None]:
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
        if handle["background"]:
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
        result = (
            result_for_children(self.store, a.execution_plan, handles, self.runtime) if sdk else child_outcome(a.plan, terminal)
        ).to_dict()
        if a.result:
            return (
                target,
                [],
                "noop" if digest(a.result._payload["result"]) == digest(result) else "rejected",
                "retained child result",
            )
        outcome = Definitive(result, tuple(f"child/{r.seq}/{r.record.digest}" for r in delivered.values()))
        if sdk and not turn.cancelled and not turn.ended and self.state.active_turn_id == op.turn_id:
            context = self.runtime.context(self.task(), a.execution_plan, self.scope.token, shared_state=self.shared_state())
            outcome = self.finalize_tool(a.execution_plan, context, outcome)
        return target, self.completed(a.execution_plan, outcome), "applied", None

    def start_turn(self) -> bool:
        inputs = sorted(self.state.applied_inputs.items(), key=lambda pair: pair[1]._payload["input"]["kind"] == "follow_up")
        for input_id, applied in inputs:
            if applied._payload["disposition"] != "queued" or input_id in self.state.admitted_inputs:
                continue
            item = applied._payload["input"]
            tid = item["target_turn_id"] or f"{self.sid}/turn/{input_id}"
            content = item["payload"]["content"]
            task = self.runtime.compile(
                content["text"] if isinstance(content, dict) and "messages" in content else str(content),
                tid,
                seed=self.records[0].record._payload["attributes"].get("seed") if not self.state.turns else None,
            )
            if isinstance(content, dict) and "messages" in content:
                task.metadata.setdefault("vv_session", {})["input_messages"] = content["messages"]
            task.task_id = tid
            if not self.state.turns:
                seed = self.records[0].record._payload["attributes"].get("seed")
                if seed is not None:
                    task.initial_messages = [Message.from_dict(copy_json(m)) for m in seed["messages"]]
                    task.initial_shared_state = copy_json(seed["shared_state"])
            definition = self.runtime._definition(task)
            prior = self.runtime._retained_task_key
            if prior is not None:
                retained = _RetainedTools(prior[0])
                if retained.encode() == self.runtime._definition_suffix[len(b',"tools":') : -1]:
                    definition = definition | {"tools": retained}
            limits = self.runtime.config.budget_limits or RunBudgetLimits()
            admission = self.records[0].record._payload["attributes"].get("child_admission") if input_id == "start" else None
            if admission is not None:
                if admission["handler_version"] != self.runtime.handler_version:
                    raise Conflict("child admission handler version mismatch")
                definition = admission["definition"]
                limits = RunBudgetLimits.from_dict(admission["budget"])
                if digest(definition) != admission["definition_digest"]:
                    raise Conflict("child admission definition digest mismatch")
            self.commit(
                [
                    self.record(
                        "turn_started",
                        {
                            "input_ids": [input_id],
                            "definition": definition,
                            "definition_digest": admission["definition_digest"] if admission else self.runtime._definition_digest,
                            "handler_version": self.runtime.handler_version,
                            "budget": limits.to_dict(),
                            "binding": None,
                            "generation": item["generation"] or 0,
                        },
                        tid=tid,
                    )
                ],
                guarded=True,
            )
            self.scope.token = CancellationToken()
            return True
        return False

    def transcript(self) -> list[Message]:
        return project_context(self.records, self.state)

    def durable_shared_state(self) -> dict[str, Any]:
        for stored in reversed(self.records):
            r = stored.record
            if r.turn_id != self.state.active_turn_id:
                continue
            if r.kind == "boundary_recorded" and "shared_state" in r._payload["data"]:
                return copy_json(r._payload["data"]["shared_state"])
            if r.kind == "op_prepared":
                return copy_json(r._payload["shared_state"])
            if r.kind == "op_completed" and r._payload["context"] == "normal":
                snapshot = r._payload["shared_state"]
                if snapshot is not None:
                    return copy_json(snapshot)
        assert self.state.active_turn_id is not None
        return copy_json(self.state.turns[self.state.active_turn_id].start._task().initial_shared_state)

    def shared_state(self) -> dict[str, Any]:
        assert self.state.active_turn_id is not None
        return self.runtime.bind_state(self.durable_shared_state(), self.state.turns[self.state.active_turn_id].start._task())

    def model_plan(self) -> Record | None:
        messages = self.transcript()
        tid = self.state.active_turn_id
        assert tid is not None
        start = self.state.turns[tid].start
        task = start.task() if self.runtime.hooks.has_hooks() else start._task()
        definition = start._payload["definition"]
        cycle = 1 + sum(
            op.turn_id == tid
            and op.kind == "model"
            and op.attempts[1].plan._payload["purpose"] == "primary"
            and not (
                op.selected_attempt
                and (receipt := op.attempts[op.selected_attempt].result)
                and receipt._payload["result"].get("error_code") == "prompt_too_long"
            )
            for op in self.state.operations.values()
        )
        shared = self.shared_state()
        if self.runtime.config.interruption_messages:
            messages.extend(self.runtime.config.interruption_messages())
        if self.runtime.config.before_cycle_messages:
            messages.extend(self.runtime.config.before_cycle_messages(cycle, list(messages), shared))
        messages, schemas = self.runtime.hooks.apply_before_llm(
            task=task,
            cycle_index=cycle,
            messages=messages,
            tool_schemas=copy_json(definition["tools"]) if self.runtime.hooks.has_hooks() else definition["tools"],
            shared_state=shared,
        )
        denied = read_after_cycle_disallowed_tools(shared)
        if denied:
            schemas = [schema for schema in schemas if schema["function"]["name"] not in denied]
        if not any(schema["function"]["name"] == "read_file" for schema in schemas) and any(
            (message.role == "tool" and is_microcompacted_tool_content(message.content))
            or any(message.metadata.get("_vv_agent_compaction", {}).get(key) for key in ("artifacts", "cursors"))
            for message in messages
        ):
            self.close("failed", "microcompaction_recovery_unavailable")
            return None
        request = {
            "model": task.model,
            "messages": [m.to_dict() for m in messages],
            "tools": schemas,
            "metadata": dict(task.metadata)
            | {"vv_session": {"shared_state": self.runtime.durable_state(shared), "cycle_index": cycle}},
            "prompt_bundle": task.prompt_bundle.to_dict(),
            "model_settings": task.model_settings.to_dict() if task.model_settings else None,
        }
        number = (
            sum(
                op.kind == "model" and op.turn_id == tid and op.attempts[1].plan._payload["purpose"] == "primary"
                for op in self.state.operations.values()
            )
            + 1
        )
        return self.plan(f"{tid}/model/primary/{number}", request, "model")

    def close(self, status: str, reason: str | None, result: Any = None, *, transferred: bool = False) -> None:
        tid = self.state.active_turn_id
        assert tid is not None
        evaluator = self.live_budget() if status == "completed" else None
        exhaustion = evaluator.terminal() if evaluator else None
        if exhaustion:
            status, reason = "failed", "budget_exhausted"
        if status == "completed" and not transferred:
            prepared = prepare_output(self, result)
            if prepared is None:
                return
            status, output_reason, result = prepared
            reason = output_reason or reason
        if evaluator and self.boundary("budget", "terminal") is None:
            self.commit([self.budget_record("terminal", evaluator, exhaustion)], guarded=True)
        records: list[Record] = []
        for op in self.state.operations.values():
            if op.turn_id != tid:
                continue
            for a in op.attempts.values():
                if a.state != "completed" and not a.started:
                    records.extend(
                        self.completed(
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
                        )
                    )
                elif a.state in {"started", "parked"}:
                    if a.child_handle and a.state == "parked" and not self.state.cancel_requested:
                        # Failure keeps the authentic child wait as unconfirmed, until terminal delivery.
                        continue
                    if a.wait and a.wait["handle"]["kind"] == "provider":
                        handle = copy_json(a.wait["handle"])
                        outcome = _invoke(
                            self.scope, lambda a=a, handle=handle: self.provider(a.execution_plan).cancel(handle), 1
                        )
                        if isinstance(outcome, Definitive):
                            records.extend(self.completed(a.execution_plan, outcome))
                            continue
                    records.append(self.unknown(a.execution_plan, reason or "turn stopped"))

        stopping = status in {"failed", "cancelled", "aborted"}
        if records or stopping:
            self.commit(
                records,
                guarded=True,
                prepare=(lambda tx: cancel_children(self.store, tx, self.state, self.sid, tid)) if stopping else None,
            )
        evaluator = budget(self.state, tid)
        terminal = self.record(
            "turn_ended",
            {
                "status": status,
                "reason": reason,
                "result": serializable_output(result),
                "adopted_results": [
                    {"operation_id": oid, "attempt": op.selected_attempt}
                    for oid, op in self.state.operations.items()
                    if op.turn_id == tid and op.selected_attempt is not None
                ],
                "budget": evaluator.snapshot().to_dict() if evaluator else {},
                "unconfirmed_operations": list(
                    dict.fromkeys(
                        oid
                        for oid, op in self.state.operations.items()
                        if op.turn_id == tid
                        for a in op.attempts.values()
                        if a.state != "completed"
                    )
                ),
            },
        )
        self.runtime.hook("before_turn_end", terminal)
        self.commit([terminal], guarded=True)

    def dispatch(self, attempt: Attempt) -> None:
        plan = attempt.execution_plan
        kind = plan._payload["op_kind"]
        assert self.state.active_turn_id is not None
        task = (
            self.state.turns[self.state.active_turn_id].start._task()
            if kind == "model" and not self.runtime.hooks.has_hooks()
            else self.task()
        )
        context = self.runtime.context(
            task,
            plan,
            self.scope.token,
            approved=attempt.approval in {"approve", "allow_session"},
            shared_state=self.shared_state(),
            store=self.store,
            session_id=self.sid,
        )
        context.metadata["fence_epoch"] = self.scope.lease.epoch
        if kind != "model":
            parent = self.state.operations[plan._payload["dependencies"][0]]
            source = parent.attempts[parent.selected_attempt or max(parent.attempts)].plan
            context.metadata["session_tool_names"] = [s["function"]["name"] for s in source._payload["request"]["tools"]]
            context.cycle_index = (
                source._payload["request"].get("metadata", {}).get("vv_session", {}).get("cycle_index", context.cycle_index)
            )
        if kind != "model" and self.runtime.hooks.has_hooks() and attempt.prepared is None:
            patched, short = self.runtime.hooks.apply_before_tool_call(
                task=task, cycle_index=context.cycle_index, call=ToolCall.from_dict(plan.payload["request"]), context=context
            )
            capabilities = self.state.turns[plan.turn_id or ""].start._payload["definition"]["capabilities"]
            prepared = self.plan(
                plan.operation_id or "",
                patched.to_dict(),
                "interaction" if patched.name == "ask_user" else "tool",
                dependencies=plan._payload["dependencies"],
                capability=capabilities.get(patched.name, {}),
            )
            self.commit(
                [
                    self.record(
                        "op_prepared",
                        {
                            "request": patched.to_dict(),
                            "request_digest": digest(patched.to_dict()),
                            "tool": prepared._payload["tool"],
                            "op_kind": prepared._payload["op_kind"],
                            "provider_binding": prepared._payload["provider_binding"],
                            "idempotency_key": prepared._payload["idempotency_key"],
                            "hook_result": short.to_dict() if short else None,
                            "shared_state": self.runtime.durable_state(context.shared_state),
                        },
                        plan,
                    )
                ],
                guarded=True,
            )
            return
        if attempt.prepared is not None:
            context.shared_state.clear()
            context.shared_state.update(
                self.runtime.bind_state(copy_json(plan._payload["budget_admission"]["shared_state"]), task)
            )
        hook_result = plan._payload["budget_admission"].get("hook_result")
        if hook_result is not None:
            self.commit(self.completed(plan, self.finalize_tool(plan, context, Definitive(hook_result))), guarded=True)
            return
        if attempt.approval in {"deny", "timeout"}:
            assert attempt.approval_answer is not None
            answer = attempt.approval_answer._payload["input"]["payload"]
            executor = self.runtime.functions.orchestrator._resolve_executor(plan._payload["request"]["name"])
            assert executor is not None
            result = ToolOrchestrator._approval_error_result(
                executor=executor,
                call=ToolCall.from_dict(plan.payload["request"]),
                request_id=answer["request_id"],
                error_code="tool_approval_timeout" if attempt.approval == "timeout" else "tool_approval_denied",
                action=attempt.approval,
                message=answer.get("reason")
                or (
                    "Approval request timed out."
                    if attempt.approval == "timeout"
                    else f"Approval denied for tool {executor.name}."
                ),
            )
            self.commit(self.completed(plan, self.finalize_tool(plan, context, Definitive(result.to_dict(), ()))))
            return
        if kind != "model":
            name = plan._payload["request"]["name"]
            if any(
                r._payload["disposition"] == "applied"
                and r._payload["input"]["kind"] == "approval_answer"
                and r._payload["input"]["payload"]["decision"] == "allow_session"
                and name in r._payload["input"]["payload"]["scope"]
                for r in self.state.applied_inputs.values()
            ):
                assert context.ctx is not None
                context.ctx._approved_tool_approval = SimpleNamespace(call=ToolCall.from_dict(plan.payload["request"]))
            preflight = self.runtime.functions.preflight(plan, context)
            if preflight is not None:
                if preflight.error_code == "tool_approval_required":
                    handle = {
                        "kind": "approval",
                        "request_id": f"approval/{plan.operation_id}/{plan.attempt}",
                        "request_digest": plan._payload["request_digest"],
                        "scope": [plan._payload["request"]["name"]],
                    }
                    self.commit([self.parked(plan, handle, after=False)], guarded=True)
                else:
                    self.commit(
                        self.completed(plan, self.finalize_tool(plan, context, Definitive(preflight.to_dict(), ()))), guarded=True
                    )
                return
        evaluator = self.live_budget()
        exhaustion = None
        budget_records = []
        if evaluator:
            exhaustion = (
                (evaluator.cycle_start() if plan._payload["purpose"] == "primary" else evaluator.model_call_start())
                if kind == "model"
                else evaluator.run_start()
            )
            if self.boundary("budget", f"{plan.operation_id}/start/{plan.attempt}") is None:
                budget_records = [
                    self.budget_record(
                        f"{plan.operation_id}/start/{plan.attempt}", evaluator, exhaustion, source=plan.operation_id
                    )
                ]
        if exhaustion:
            self.commit(budget_records, guarded=True)
            self.close("failed", "budget_exhausted")
            return
        poll = self.scope.poll(self.store)
        if poll.inbox_seq != self.watermark or self.scope.token.cancelled:
            raise SequenceConflict("input arrived before dispatch")
        if kind == "interaction" or plan._payload["request"].get("name") == "ask_user":
            outcome = self.runtime.functions.submit(plan, context=context)
            if not isinstance(outcome, Definitive):
                raise ValueError("ask_user must be a synchronous interaction")
            question = outcome.result.get("metadata", {}).get("question", outcome.result["content"])
            parked = self.parked(
                plan, {"kind": "user", "interaction_id": f"interaction/{plan.operation_id}", "question": question}, after=False
            )
            parked = replace(parked, payload=parked._payload | {"interaction_result": outcome.result})
            skipped = []
            for oid, op in self.state.operations.items():
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
                    skipped.extend(self.completed(pending.execution_plan, Definitive(result.to_dict(), ())))
            self.commit([parked, *skipped], guarded=True)
            return
        started = self.record(
            "op_started",
            {
                "dispatch_id": f"{plan.operation_id}/{plan.attempt}",
                "authorization_version": self.runtime.handler_version,
                "epoch": self.scope.lease.epoch,
                "mode": "sync" if kind == "model" else "provider",
            },
            plan,
        )
        if kind == "model" and plan._payload["request"]["metadata"].get("vv_session", {}).get("endpoint_order"):
            order = plan._payload["request"]["metadata"]["vv_session"]["endpoint_order"]
            started = replace(started, payload=started._payload | {"endpoint_id": order[((plan.attempt or 1) - 1) % len(order)]})
        name = plan._payload["request"].get("name")
        if kind != "model" and (name in self.runtime.children or name in self.runtime.delegated_tools):
            child_specs = None if name in self.runtime.children else assemble(self.runtime, plan, context, task, self.store)
            if isinstance(child_specs, ToolExecutionResult):
                child_specs.tool_call_id = plan._payload["request"]["id"]
                self.commit(
                    self.completed(plan, self.finalize_tool(plan, context, Definitive(child_specs.to_dict()))), guarded=True
                )
                return
            if child_specs is not None or name in self.runtime.children:
                assert plan.turn_id is not None

                def admit_child(tx: SessionTx) -> list[Record]:
                    specs = [self.runtime.children[name](plan)] if name in self.runtime.children else child_specs
                    assert specs is not None
                    generation = self.state.turns[plan.turn_id or ""].start._payload["generation"]
                    handles = [create_child(tx, child, plan, generation) for child in specs]
                    handle = handles[0]
                    if len(handles) > 1:
                        handle = handle | {
                            "siblings": [{k: h[k] for k in ("session_id", "turn_id", "generation")} for h in handles[1:]]
                        }
                    parked = self.parked(plan, handle, after=True)
                    admission = specs[0].spec.attributes.get("child_admission") if specs[0].spec.attributes else None
                    if admission:
                        marker = {key: admission[key] for key in ("mode", "handoff_count", "max_handoffs")}
                        marker["agent_name"] = admission["definition"]["agent_name"]
                        marker["metadata"] = admission["handoff_metadata"]
                        parked = replace(parked, payload=parked._payload | {"delegation": marker})
                    records = [parked]
                    if specs[0].background:
                        result = admitted_result(self.store, plan, handles)
                        records.extend(
                            self.completed(
                                plan,
                                self.finalize_tool(
                                    plan,
                                    context,
                                    Definitive(result.to_dict(), tuple(f"session/{h['session_id']}/created" for h in handles)),
                                ),
                            )
                        )
                        # completed() sees the pre-transaction state; admission starts in this commit.
                        records[-1] = replace(records[-1], payload=records[-1]._payload | {"execution_started": True})
                    return records

                self.commit([*budget_records, started], guarded=True, prepare=admit_child)
                return
        self.commit([*budget_records, started], guarded=True)
        self.runtime.hook("before_external_call", plan)

        def model() -> Outcome:
            request = request_from_dict(copy_json(plan._payload["request"]))
            request.metadata.update(
                {"operation_id": plan.operation_id, "attempt": plan.attempt, "call_id": f"{plan.operation_id}/{plan.attempt}"}
            )
            request.metadata["purpose"] = plan._payload["purpose"]
            if started._payload.get("endpoint_id"):
                request.metadata.setdefault("vv_session", {})["endpoint_id"] = started._payload["endpoint_id"]
            if plan._payload["purpose"] == "output_repair":
                repair = self.runtime.agent.output_repair
                assert repair is not None
                try:
                    repaired = repair(
                        OutputRepairRequest(
                            invalid_output=request.metadata["invalid_output"],
                            validation_code=request.metadata["validation_code"],
                            validation_message=request.metadata["validation_message"],
                            model=self.runtime.agent.output_repair_model,
                            model_settings=self.runtime.agent.output_repair_model_settings,
                        )
                    )
                except Exception as exc:
                    return Definitive({"content": str(exc), "tool_calls": [], "error_code": "repair_provider_error"})
                if isinstance(repaired, LLMResponse):
                    return Definitive({"content": repaired.content, "tool_calls": []}, usage=repaired.raw.get("usage") or {})
                return Definitive({"content": serializable_output(repaired), "tool_calls": []})
            if "shared_state" in request.metadata.get("vv_session", {}):
                context.shared_state.clear()
                context.shared_state.update(self.runtime.bind_state(request.metadata["vv_session"]["shared_state"], task))
            try:
                callback = None
                stream = self.runtime.config.stream
                if stream is not None and plan._payload["purpose"] == "primary":

                    def callback(payload):
                        event = _project_provider_stream_payload(
                            payload | {"cycle": context.cycle_index},
                            run_id=plan.turn_id or "",
                            trace_id=plan.turn_id or "",
                            agent_name=self.runtime.agent.name,
                            session_id=self.sid,
                            parent_run_id=None,
                        )
                        if event is not None and not self.scope.token.cancelled:
                            object.__setattr__(event, "version", "v6")
                            with suppress(Exception):
                                stream(event)

                response = self.runtime.complete(request, plan.attempt or 1, callback)
            except Exception as exc:
                if is_prompt_too_long_error(exc):
                    return Definitive({"error_code": "prompt_too_long", "content": "", "tool_calls": []})
                raise
            observed_usage = copy_json(response.raw.get("usage") or {})
            if plan._payload["purpose"] == "primary":
                response = self.runtime.hooks.apply_after_llm(
                    task=task,
                    cycle_index=context.cycle_index,
                    messages=request.messages,
                    tool_schemas=request.tools,
                    response=response,
                    shared_state=context.shared_state,
                )
            return Definitive(
                {
                    "content": response.content,
                    "tool_calls": [c.to_dict() for c in response.tool_calls],
                    "raw": response.raw,
                    "reasoning_content": response.raw.get("reasoning_content"),
                },
                usage=observed_usage,
            )

        outcome = _invoke(
            self.scope,
            model if kind == "model" else lambda: self.provider(plan).submit(plan, context=context),
            self.runtime.model_timeout if kind == "model" else self.runtime.tool_timeout,
        )
        self.runtime.hook("after_external_call", plan)
        if isinstance(outcome, Definitive):
            if kind != "model":
                outcome = self.finalize_tool(plan, context, outcome)
            outcome = replace(outcome, shared_state=self.runtime.durable_state(context.shared_state))
            self.commit(self.completed(plan, outcome))
        elif isinstance(outcome, Accepted):
            self.commit([self.parked(plan, outcome.handle, after=True)])
        else:
            self.commit([self.unknown(plan, outcome.reason)])

    def finalize_tool(self, plan: Record, context, outcome: Definitive) -> Definitive:
        call = ToolCall.from_dict(plan.payload["request"])
        result = self.runtime.hooks.apply_after_tool_call(
            task=self.task(),
            cycle_index=context.cycle_index,
            call=call,
            context=replace(context, tool_call_id=call.id, tool_name=call.name, arguments=dict(call.arguments)),
            result=ToolExecutionResult.from_dict(outcome.result),
        )
        if not result.tool_call_id.strip():
            result.tool_call_id = call.id
        stop = apply_tool_use_behavior(task=self.task(), call=call, result=result)
        if stop is not None:
            result.metadata["completion_reason"] = stop.value
        return replace(outcome, result=result.to_dict(), shared_state=self.runtime.durable_state(context.shared_state))

    def step(self) -> bool:
        self.refresh()
        if (
            self.state.active_turn_id
            and not self.state.cancel_requested
            and not self.scope.token.cancelled
            and any(
                op.turn_id == self.state.active_turn_id and op.attempts[1].plan._payload["purpose"] == "compaction"
                for op in self.state.operations.values()
            )
            and self.state.turns[self.state.active_turn_id].start._payload["handler_version"] == self.runtime.handler_version
            and self.runtime.definition_digest(self.state.turns[self.state.active_turn_id].start)
            == self.state.turns[self.state.active_turn_id].start._payload["definition_digest"]
            and finish_summary(self)
        ):
            return True
        if self.apply_input():
            return True
        tid = self.state.active_turn_id
        if tid is None:
            save_projection(self)
            return not self.state.closed and self.start_turn()
        turn = self.state.turns[tid]
        if turn.cancelled:
            actions = [
                r._payload["input"]["payload"]["action"]
                for r in self.state.applied_inputs.values()
                if r.turn_id == tid and r._payload["disposition"] == "applied" and r._payload["input"]["kind"] == "control"
            ]
            self.close("aborted" if "abort" in actions else "cancelled", "cancel_requested")
            return True
        if turn.suspended or turn.wait is not None:
            return False
        if self.scope.token.cancelled:
            # A suspend/resume pair may already have been folded in this drive.
            self.scope.token = CancellationToken()
        if turn.start._payload["handler_version"] != self.runtime.handler_version:
            self.close("failed", "handler_version_mismatch")
            return True
        task = turn.start._task()
        if task.metadata.get("vv_session", {}).get("host_binding_names"):
            self.runtime.bind_state({}, task)
        if task.metadata.get("vv_session", {}).get("input_blocked"):
            self.close("failed", "agent_failed", task.metadata.setdefault("vv_session", {})["input_blocked"])
            return True
        if self.runtime.definition_digest(turn.start) != turn.start._payload["definition_digest"]:
            self.close("failed", "handler_schema_or_capability_mismatch")
            return True
        evaluator = self.live_budget()
        if evaluator and self.boundary("budget", "run_start") is None:
            exhaustion = evaluator.run_start()
            self.commit([self.budget_record("run_start", evaluator, exhaustion)], guarded=True)
            return True
        for (turn_id, stage, _), recorded in self.state.boundaries.items():
            if turn_id == tid and stage == "budget" and recorded._payload["data"]["exhaustion"]:
                self.close("failed", "budget_exhausted")
                return True
        if after_cycle(self):
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
                if self.runtime.config.after_cycle_hooks and decision is None:
                    continue
                repair = self.state.operations.get(f"{tid}/model/output_repair/1")
                if repair and repair.state in {"planned", "started", "parked"}:
                    continue
                transfer_failed = (
                    result._payload["result"].get("metadata", {}).get("mode") == "handoff"
                    and result._payload["result"].get("status_code") == "ERROR"
                )
                self.close(
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
                        skipped.extend(self.completed(pending.execution_plan, Definitive(value.to_dict())))
                if skipped:
                    self.commit(skipped, guarded=True)
                    return True
                if any(other.state != "completed" for _, other in operations):
                    return False
                self.commit(
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
                    ],
                    guarded=True,
                )
                return True
        for oid, op in operations:
            if op.kind == "model" and op.attempts[1].plan._payload["purpose"] == "primary" and op.selected_attempt is not None:
                known = op.attempts[op.selected_attempt]
                if known.result and known.context == "normal":
                    repairs = self.completion_effects(
                        known.plan,
                        Definitive(
                            known.result._payload["result"],
                            tuple(known.result._payload["evidence"]),
                            known.result._payload["usage"],
                        ),
                        known.context,
                    )
                    if repairs:
                        self.commit(repairs, guarded=True)
                        return True
            a = op.attempts[max(op.attempts)]
            if a.state == "started":
                self.commit([self.unknown(a.execution_plan, "worker lost before durable result")], guarded=True)
                return True
            if a.state == "unknown" and a.unknown:
                if a.unknown._payload["retry"] == "retry":
                    now = self.scope.poll(self.store).db_now_ms
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
                    self.commit([next_plan], guarded=True)
                    return True
                if op.kind == "model" and a.plan._payload["purpose"] != "compaction":
                    self.close("failed", "model_outcome_unknown")
                    return True
            if a.state == "planned" or (a.state == "parked" and a.approval and a.wait and a.wait["handle"]["kind"] == "approval"):
                if (a.plan._payload["not_before_ms"] or 0) > self.scope.poll(self.store).db_now_ms:
                    return False
                self.dispatch(a)
                return True
            if a.state == "parked" and a.wait and a.wait["handle"]["kind"] == "approval":
                return resolve_approval(self, a)
            if a.state == "parked" and a.wait and a.wait["handle"]["kind"] == "user":
                return False
            if a.state == "parked" and a.wait and a.wait["handle"]["kind"] == "provider":
                key = (oid, a.plan.attempt or 1)
                if key not in self.polled:
                    self.polled.add(key)
                    handle = copy_json(a.wait["handle"])
                    outcome = _invoke(self.scope, lambda a=a, handle=handle: self.provider(a.execution_plan).query(handle), 1)
                    if isinstance(outcome, Definitive):
                        self.commit(self.completed(a.execution_plan, outcome), guarded=True)
                        return True
                return False
        if self.pending_batch():
            return False
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
                            and r.record.turn_id == tid
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
                    and not a.result._payload["result"].get("error_code")
                    and not changed
                    and not (cycle_decision and cycle_decision._payload["data"]["action"] == "steer")
                ):
                    if task.no_tool_policy == "finish":
                        self.close("completed", None, a.result._payload["result"]["content"])
                        return True
                    if task.no_tool_policy == "wait_user":
                        self.commit(
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
                            ],
                            guarded=True,
                        )
                        return True
        cycles = sum(
            not (a.result and a.result._payload["result"].get("error_code") == "prompt_too_long")
            for _, op in models
            for a in [op.attempts[op.selected_attempt or max(op.attempts)]]
        )
        if cycles >= task.max_cycles:
            self.close("failed", "max_cycles", "Reached max cycles without finish signal.")
            return True
        hook_id = str(cycles + 1)
        if self.runtime.hooks.has_hooks() and self.boundary("before_memory", hook_id) is None:
            shared = self.shared_state()
            source = self.transcript()
            replacement = self.runtime.hooks.apply_before_memory_compact(
                task=self.task(), cycle_index=cycles + 1, messages=source, shared_state=shared
            )
            self.commit(
                [
                    self.boundary_record(
                        "before_memory",
                        hook_id,
                        {"messages": [m.to_dict() for m in replacement], "shared_state": self.runtime.durable_state(shared)},
                        source_digest=digest([m.to_dict() for m in source]),
                    )
                ],
                guarded=True,
            )
            return True
        if compact_context(self):
            return True
        plan = self.model_plan()
        if plan is not None:
            self.commit([plan], guarded=True)
        return True


def drive(
    store: SessionStore, session_id: str, *, runtime: Runtime, _one_turn: bool = False, _wait_for_lease: bool = False
) -> None:
    lease = store.acquire(session_id, owner=uuid4().hex, ttl_ms=runtime.ttl_ms)
    while lease is None and _wait_for_lease:
        time.sleep(0.01)
        lease = store.acquire(session_id, owner=uuid4().hex, ttl_ms=runtime.ttl_ms)
    if lease is None:
        return
    scope = _Scope(lease, runtime)
    scope.thread.start()
    try:
        driver = _Driver(store, session_id, runtime, scope)
        while not scope.lost:
            try:
                if not driver.step():
                    with scope.lock:
                        if scope.lost:
                            raise LeaseLost("heartbeat lost ownership")
                        store.defer_idle_drive(scope.lease, poll_ms=runtime.poll_ms)
                    break
                if _one_turn and driver.records[-1].record.kind == "turn_ended":
                    break
            except SequenceConflict:
                continue
        if scope.lost:
            raise LeaseLost("heartbeat lost ownership")
    finally:
        scope.stop.set()
        scope.thread.join(timeout=2)
        with suppress(Exception):
            store.release(scope.lease)
        with suppress(Exception):
            runtime.hook("after_drive", None)
        with suppress(Exception):
            if store.is_runnable(session_id):
                runtime.wake(session_id)
