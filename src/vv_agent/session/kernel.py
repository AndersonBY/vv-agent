"""Synchronous, lease-fenced driver. Durable facts live only in the session log."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import replace
from typing import Any
from uuid import uuid4

from vv_agent.budget import BudgetEvaluator, BudgetExhaustion
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.memory.microcompact import is_microcompacted_tool_content
from vv_agent.model_settings import ModelSettings, RetrySettings
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.lifecycle import read_after_cycle_disallowed_tools
from vv_agent.runtime.tool_results import build_skipped_result
from vv_agent.tools.outcomes import validate_definitive_result
from vv_agent.types import AgentTask, Message, ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus

from . import _driver_dispatch, _driver_inputs
from ._driver_scope import HeartbeatStopped as HeartbeatStopped
from ._driver_scope import _invoke, _LeaseHeartbeat, _Scope
from .approval import resolve_approval
from .bindings import MissingHostBinding
from .children import (
    cancel_children,
    child_handles,
)
from .compaction import compact_context, finish_summary
from .context import project_context
from .lifecycle import after_cycle
from .memory import save_projection
from .output import prepare_output, serializable_output
from .providers import (
    Accepted,
    Definitive,
    ProviderResultInvalid,
    validate_acceptance,
    validate_provider_result,
)
from .records import InboxItem, Record, _RetainedTools, copy_json, digest, make_record
from .reducer import Attempt, ExecutionState, _fold_factory_for
from .runtime import Runtime, RuntimeNotReady, budget, model_usage
from .sql import SQLSessionTx, SQLStore
from .store import Conflict, Lease, LeaseLost, SequenceConflict, SessionStore, SessionTx, StoredRecord
from .summary_requests import pending_compaction_requests, summary_pending
from .timing import Clock, LeaseKeeper, SystemClock, ThreadLeaseKeeper

MODEL_REJECTION_BACKOFF_SECONDS = 2


def read_state(store: SessionStore, sid: str) -> tuple[ExecutionState, tuple[StoredRecord, ...], int]:
    return store.read_state(sid)


FACTORY_LEASE_TTL_MS = 15000
FACTORY_HEARTBEAT_SECONDS = 0.25
logger = logging.getLogger(__name__)


class _Driver:
    def __init__(self, store: SessionStore, sid: str, runtime: Runtime, scope: _Scope, *, one_turn: bool = False):
        self.store, self.sid, self.runtime, self.scope = store, sid, runtime, scope
        self.one_turn = one_turn
        self.load_state()
        self.polled: set[tuple[str, int]] = set()
        self.active_since = self.runtime.clock.monotonic_ns()
        self.active_tid: str | None = None

    def load_state(self) -> None:
        self.state, self.records, self.watermark = read_state(self.store, self.sid)
        self.fold = _fold_factory_for(self.store)()
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
            self.active_since = self.runtime.clock.monotonic_ns()
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
                        "usage": {
                            "elapsed_ms": (self.runtime.clock.monotonic_ns() - self.active_since) // 1000000,
                            "host_cost": cost,
                        },
                    },
                ),
                *records,
            ]
        timings: list[tuple[str, float, str, str | None]] = []
        atomic_start = self.runtime.clock.monotonic_ns()
        atomic_error = None

        @contextmanager
        def phase(name: str):
            begin = self.runtime.clock.monotonic_ns()
            error = None
            try:
                yield
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                timings.append((name, (self.runtime.clock.monotonic_ns() - begin) / 1_000_000, "error" if error else "ok", error))

        try:
            with ExitStack() as latch, self.store.atomic() as tx:
                if prepare is not None:
                    with phase("prepare"):
                        records.extend(prepare(tx))
                with phase("before_commit"):
                    self.runtime.hook("before_commit", records[-1] if records else None)
                # SQL append calls this only after encoding and immutable history
                # preparation. Keep the latch through the outer atomic exit.
                if isinstance(tx, SQLSessionTx):
                    tx._lease_guard = lambda: latch.enter_context(self.scope.committing())
                with phase("append"):
                    receipt = tx.append(
                        self.sid,
                        lease=self.scope.lease if isinstance(tx, SQLSessionTx) else latch.enter_context(self.scope.committing()),
                        expected_seq=self.records[-1].seq,
                        commit_id=digest({"record_ids": [r.record_id for r in records], "consumed": list(consumed)}),
                        records=tuple(records),
                        consume_input_ids=consumed,
                        expected_inbox_seq=self.watermark if guarded else None,
                    )
        except BaseException as exc:
            atomic_error = type(exc).__name__
            raise
        finally:
            atomic_ms = (self.runtime.clock.monotonic_ns() - atomic_start) / 1_000_000
            source = next((r for r in reversed(records) if r.operation_id), None)
            request_digest = source._payload.get("request_digest") if source else None
            if source and source.operation_id in self.state.operations:
                operation = self.state.operations[source.operation_id]
                attempt = operation.attempts.get(source.attempt or 1)
                if attempt is not None:
                    request_digest = attempt.execution_plan._payload["request_digest"]
            for name, duration, outcome, error in timings:
                logger.log(
                    logging.WARNING if duration >= self.runtime.ttl_ms / 2 else logging.DEBUG,
                    "session transaction phase",
                    extra={
                        "session_id": self.sid,
                        "turn_id": tid,
                        "operation_id": source.operation_id if source else None,
                        "request_digest": request_digest,
                        "epoch": self.scope.lease.epoch,
                        "phase": name,
                        "callback_ms": duration if name != "append" else 0,
                        "append_ms": duration if name == "append" else 0,
                        "kernel_atomic_ms": atomic_ms,
                        "outcome": "error" if atomic_error else outcome,
                        "exception_type": error or atomic_error,
                    },
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
            initial,
            elapsed_ms=initial.elapsed_ms
            + max(0, (self.runtime.clock.monotonic_ns() - self.active_since) // 1000000 - retained_ms),
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

    def _task(self) -> AgentTask:
        assert self.state.active_turn_id is not None
        return self.state.turns[self.state.active_turn_id].start._task()

    def effective_model(self) -> dict[str, Any]:
        selected = self.state.model_selection
        if selected is not None:
            selection = selected._payload["input"]["payload"]["selection"]
            return selection | {"binding": selected._payload["model_binding"]}
        fork = self.records[0].record._payload["attributes"].get("app_server_fork")
        if fork is not None:
            return copy_json(fork["effective_model"])
        assert self.state.active_turn_id is not None
        task = self.state.turns[self.state.active_turn_id].start._task()
        return {
            "model": task.model,
            "model_settings": task.model_settings.to_dict() if task.model_settings else {},
            "route": {},
            "binding": self.runtime.model_binding(self.runtime.llm, self.runtime.resolved),
        }

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
        if kind == "model":
            from vv_agent.llm.vv_llm_client import VvLlmClient

            client, resolved = self.runtime.resolve_model(request) if purpose == "primary" else self.runtime.model_route(purpose)
            session = request["metadata"].get("vv_session", {})
            if purpose != "output_repair":
                session = session | {"model_binding": self.runtime.model_binding(client, resolved)}
            if isinstance(client, VvLlmClient):
                preferred = self.state.preferred_endpoint_id
                if preferred not in {t.endpoint_id for t in client.endpoint_targets}:
                    preferred = None
                order = [t.endpoint_id for t in client.ordered_targets(preferred)]
                if order:
                    session = session | {"endpoint_order": order}
            request = request | {"metadata": request["metadata"] | {"vv_session": session}}
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
        if kind == "model" and isinstance(request.get("messages"), list):
            request = request | {"messages": self.fold._messages.retain_array(request["messages"])}
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
                    if kind == "model"
                    and request["metadata"].get("mode") != "child_summary"
                    and op.turn_id == self.state.active_turn_id
                    and a.state == "unknown"
                    and op.kind != "model"
                ],
            },
        )

    def compaction_intent(self, plan: Record, result: dict[str, Any]) -> dict[str, Any]:
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

    def completed(self, plan: Record, outcome: Definitive) -> list[Record]:
        from .tool_results import prepare_tool_resources

        result = self.compaction_intent(plan, outcome.result)
        if plan._payload["op_kind"] != "model":
            result = prepare_tool_resources(plan, result, self.runtime.workspace_backend)
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
        records = [result, *self.completion_effects(plan, outcome, context)]
        rejection = outcome.result.get("rejection", {})
        order = plan._payload["request"].get("metadata", {}).get("vv_session", {}).get("endpoint_order", [])
        if (
            op.kind == "model"
            and context == "normal"
            and outcome.result.get("error_code") == "model_request_rejected"
            and rejection.get("retryable") is True
            and plan._payload["purpose"] != "output_repair"
            and plan.attempt < max(2, len(order))
            and not self.scope.token.cancelled
            and not self.state.cancel_requested
            and not any(r.kind == "boundary_recorded" and r._payload["data"].get("exhaustion") for r in records[1:])
        ):
            delay = (
                0 if rejection.get("status_code") in {401, 403, 404} else MODEL_REJECTION_BACKOFF_SECONDS * 1000 * plan.attempt
            )
            due = self.scope.poll(self.store).db_now_ms + delay if delay else 0
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

    def completion_effects(self, plan: Record, outcome: Definitive, context: str) -> list[Record]:
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
            and reason != "provider_result_invalid"
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
        binding = plan._payload["provider_binding"]
        if binding == "function":
            return self.runtime.functions
        if binding not in self.runtime.providers:
            raise MissingHostBinding(f"Missing provider host binding: {binding}")
        return self.runtime.providers[binding]

    def check_provider_result(self, plan: Record, outcome: Definitive) -> None:
        assert plan.operation_id is not None and plan.attempt is not None
        attempt = self.state.operations[plan.operation_id].attempts[plan.attempt]
        handle = attempt.wait["handle"] if attempt.wait and attempt.wait["handle"]["kind"] == "provider" else None
        validate_provider_result(plan, outcome, handle)

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

    def accepted(self, plan: Record, outcome: Accepted) -> Record:
        validate_acceptance(plan, outcome)
        parked = self.parked(plan, outcome.handle, after=True)
        if outcome.yield_on_steer:
            parked = replace(parked, payload=parked._payload | {"yield_on_steer": True})
        return parked

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

    def yieldable_batch(self) -> list[Attempt]:
        """Return the whole remaining batch only when every observation can yield."""
        tid = self.state.active_turn_id
        # shortcut: local drivers keep normal waits; enable yielding with local scheduling.
        if self.one_turn or tid is None:
            return []
        turn = self.state.turns[tid]
        if turn.cancelled or turn.suspended or turn.wait or self.scope.token.cancelled:
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

    def yield_results(self, attempts: list[Attempt], item: InboxItem) -> list[Record]:
        records = []
        for a in attempts:
            plan = a.execution_plan
            assert plan.operation_id is not None and plan.attempt is not None
            assert a.wait is not None
            handle = a.wait["handle"]
            handles = child_handles(handle) if a.child_handle else handle
            binding = self.runtime.children.get(plan._payload["request"]["name"]) if a.child_handle else self.provider(plan)
            callback = getattr(binding, "on_yield", None)
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
                    rendered = callback(plan, copy_json(handles), "steer")
                    validate_definitive_result(rendered)
                    from .tool_results import validate_output_body

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
            completed_records = self.completed(plan, Definitive(result.to_dict(), evidence))
            records.extend(
                replace(r, payload=r._payload | {"yielded_by": item.input_id}) if r.kind == "op_completed" else r
                for r in completed_records
            )
        return records

    def apply_input(self) -> bool:
        return _driver_inputs.apply_input(self)

    def apply_child(self, item: InboxItem) -> tuple[str, list[Record], str, str | None]:
        return _driver_inputs.apply_child(self, item)

    def start_turn(self) -> bool:
        return _driver_inputs.start_turn(self)

    def transcript(self) -> list[Message]:
        return project_context(self.records, self.state, _fold_factory=_fold_factory_for(self.store))

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
        from vv_agent.kernel._context_messages import retained_start

        from .context import protected_start

        protection = protected_start(self.records, self.state)
        protected_source = [Message.from_dict(copy_json(m.to_dict())) for m in messages] if protection is not None else []
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
        retained_start(protected_source, messages, protection)
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
        effective = self.effective_model()
        request = {
            "model": effective["model"],
            "route": effective["route"],
            "messages": [m.to_dict() for m in messages],
            "tools": schemas,
            "metadata": dict(task.metadata)
            | {
                "vv_session": {
                    "shared_state": self.runtime.durable_state(shared),
                    "cycle_index": cycle,
                    "model_binding": effective["binding"],
                }
            },
            "prompt_bundle": task.prompt_bundle.to_dict(),
            "model_settings": replace(
                ModelSettings.from_dict(effective.get("model_settings", {})),
                retry=RetrySettings(max_attempts=1, backoff_seconds=0),
            ).to_dict(),
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
        if status == "completed" and not transferred and not self._task().metadata.get("_vv_agent_app_server_compaction_only"):
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
                        can_be_unknown = a.state == "started" or (
                            a.state == "parked" and a.started and self.state.turns[tid].cancelled
                        )
                        handle = copy_json(a.wait["handle"])
                        provider = self.provider(a.execution_plan)
                        outcome = _invoke(self.scope, lambda handle=handle, provider=provider: provider.cancel(handle), 1)
                        if isinstance(outcome, Definitive):
                            try:
                                self.check_provider_result(a.execution_plan, outcome)
                            except ProviderResultInvalid:
                                if can_be_unknown:
                                    records.append(self.unknown(a.execution_plan, "provider_result_invalid"))
                            else:
                                records.extend(self.completed(a.execution_plan, outcome))
                            continue
                        if not can_be_unknown:
                            # Guard closure keeps the authentic wait for terminal audit.
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
        return _driver_dispatch.dispatch(self, attempt)

    def finalize_tool(self, plan: Record, context, outcome: Definitive) -> Definitive:
        return _driver_dispatch.finalize_tool(self, plan, context, outcome)

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
            if (
                op.kind == "model"
                and a.result
                and a.result._payload["result"].get("error_code") == "model_request_rejected"
                and a.plan._payload["purpose"] != "compaction"
            ):
                self.close("failed", "model_request_rejected", a.result._payload["result"]["content"])
                return True
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
                if summary_pending(self, oid):
                    continue
                self.dispatch(a)
                return True
            if a.state == "parked" and a.wait and a.wait["handle"]["kind"] == "approval":
                return resolve_approval(self, a)
            if a.state == "parked" and a.wait and a.wait["handle"]["kind"] == "user":
                return False
            if a.state == "parked" and a.wait and a.wait["handle"]["kind"] == "provider":
                key = (oid, a.plan.attempt or 1)
                if key not in self.polled:
                    with self.scope.committing() as lease:
                        if self.store.next_drive_delay_ms(self.sid, lease=lease) != 0:
                            return False
                    self.polled.add(key)
                    handle = copy_json(a.wait["handle"])
                    provider = self.provider(a.execution_plan)
                    outcome = _invoke(self.scope, lambda handle=handle, provider=provider: provider.query(handle), 1)
                    if isinstance(outcome, Definitive):
                        self.check_provider_result(a.execution_plan, outcome)
                        self.commit(self.completed(a.execution_plan, outcome), guarded=True)
                        return True
                    if isinstance(outcome, Accepted):
                        validate_acceptance(a.execution_plan, outcome)
                        if outcome.handle != handle or outcome.yield_on_steer != a.wait.get("yield_on_steer", False):
                            raise ProviderResultInvalid()
                return False
        if self.pending_batch():
            return False
        requested = [key for key in pending_compaction_requests(self) if key in self.state.applied_inputs]
        if requested and compact_context(self, requested_by=requested):
            return True
        if task.metadata.get("_vv_agent_app_server_compaction_only"):
            self.close("completed", "context_compaction")
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
            from vv_agent.kernel._context_messages import retained_start

            from .context import protected_start

            protection = protected_start(self.records, self.state)
            source_digest = digest([m.to_dict() for m in source])
            protected_source = [Message.from_dict(copy_json(m.to_dict())) for m in source] if protection is not None else []
            replacement = self.runtime.hooks.apply_before_memory_compact(
                task=self.task(), cycle_index=cycles + 1, messages=source, shared_state=shared
            )
            retained_start(protected_source, replacement, protection)
            self.commit(
                [
                    self.boundary_record(
                        "before_memory",
                        hook_id,
                        {"messages": [m.to_dict() for m in replacement], "shared_state": self.runtime.durable_state(shared)},
                        source_digest=source_digest,
                    )
                ],
                guarded=True,
            )
            return True
        if compact_context(self, requested_by=pending_compaction_requests(self)):
            return True
        plan = self.model_plan()
        if plan is not None:
            self.commit([plan], guarded=True)
        return True


def _defer_after_error(store: SessionStore, lease: Lease, retry_after_ms: int, error: Exception) -> None:
    try:
        store.defer_drive(lease, retry_after_ms=retry_after_ms)
    except LeaseLost:
        pass
    except Exception as backoff_error:
        raise error from backoff_error


def drive(
    store: SessionStore,
    session_id: str,
    *,
    runtime: Runtime | Callable[[str], Runtime],
    failure_backoff_ms: int = 1000,
    _one_turn: bool = False,
    _wait_for_lease: bool = False,
) -> bool:
    """Run the session through the shared lease-fenced driver."""
    return _drive(
        store,
        session_id,
        runtime=runtime,
        failure_backoff_ms=failure_backoff_ms,
        _one_turn=_one_turn,
        _wait_for_lease=_wait_for_lease,
    )


def _drive(
    store: SessionStore,
    session_id: str,
    *,
    runtime: Runtime | Callable[[str], Runtime],
    failure_backoff_ms: int = 1000,
    _one_turn: bool = False,
    _wait_for_lease: bool = False,
    clock: Clock | None = None,
    lease_keeper: LeaseKeeper | None = None,
) -> bool:
    """Drive with runtime bindings and optional factory-admission seams."""
    if type(failure_backoff_ms) is not int or failure_backoff_ms <= 0:
        raise ValueError("failure_backoff_ms must be a positive integer")
    scheduled = not isinstance(runtime, Runtime)
    clock = clock if clock is not None else runtime.clock if isinstance(runtime, Runtime) else SystemClock()
    lease_keeper = (
        lease_keeper
        if lease_keeper is not None
        else runtime.lease_keeper
        if isinstance(runtime, Runtime)
        else ThreadLeaseKeeper()
    )
    ttl_ms = runtime.ttl_ms if isinstance(runtime, Runtime) else FACTORY_LEASE_TTL_MS
    lease = store.acquire(session_id, owner=uuid4().hex, ttl_ms=ttl_ms)
    delay = 0.05
    while lease is None and _wait_for_lease:
        clock.sleep(delay)
        delay = min(1.0, delay * 2)
        lease = store.acquire(session_id, owner=uuid4().hex, ttl_ms=ttl_ms)
    if lease is None:
        return False
    if not isinstance(runtime, Runtime):
        build = None
        # Resolve store setup on the driving thread (host connections can be thread-bound).
        try:
            heartbeat = store.heartbeat_store()
            build = _LeaseHeartbeat(
                lease,
                lambda: heartbeat,
                ttl_ms,
                FACTORY_HEARTBEAT_SECONDS,
                clock=clock,
                lease_keeper=lease_keeper,
                now_ms=store._now() if isinstance(store, SQLStore) else None,
            )
            build.start()
            try:
                if not build.ready.wait(ttl_ms / 1000):
                    raise LeaseLost("factory heartbeat did not become ready") from build.heartbeat_error
                if build.lost or not build.successful_ready:
                    raise LeaseLost("factory heartbeat lost ownership") from build.heartbeat_error
                runtime = runtime(session_id)
            finally:
                build.close()
                lease = build.lease
            if build.lost:
                raise LeaseLost("factory heartbeat lost ownership") from build.heartbeat_error
            lease = store.renew(lease, ttl_ms=runtime.ttl_ms).lease
        except RuntimeNotReady as exc:
            try:
                _defer_after_error(store, lease, exc.retry_after_ms, exc)
            finally:
                store.release(lease)
            return True
        except Exception as exc:
            if build is not None and build.lease_keeper.is_alive(build):
                raise
            try:
                _defer_after_error(store, lease, failure_backoff_ms, exc)
            finally:
                store.release(lease)
            raise
        except BaseException:
            if build is None or not build.lease_keeper.is_alive(build):
                with suppress(Exception):
                    store.release(lease)
            raise
    scope = _Scope(lease, runtime, now_ms=store._now() if isinstance(store, SQLStore) else None)
    scope.start()
    try:
        driver = _Driver(store, session_id, runtime, scope, one_turn=_one_turn)
        while not scope.lost:
            try:
                if not driver.step():
                    with ExitStack() as latch:
                        if isinstance(store, SQLStore):
                            store._defer_idle_drive(
                                scope.lease,
                                poll_ms=runtime.poll_ms,
                                lease_guard=lambda: latch.enter_context(scope.committing()),
                            )
                        else:
                            store.defer_idle_drive(latch.enter_context(scope.committing()), poll_ms=runtime.poll_ms)
                    break
                if _one_turn and driver.records[-1].record.kind == "turn_ended":
                    break
            except SequenceConflict:
                continue
        if scope.lost:
            raise LeaseLost("heartbeat lost ownership") from scope.heartbeat_error
    except Exception as exc:
        scope.close()
        if scheduled:
            _defer_after_error(store, scope.lease, failure_backoff_ms, exc)
        raise
    finally:
        scope.close()
        with suppress(Exception):
            store.release(scope.lease)
        with suppress(Exception):
            runtime.hook("after_drive", None)
        with suppress(Exception):
            if store.is_runnable(session_id):
                runtime.wake(session_id)
    return True
