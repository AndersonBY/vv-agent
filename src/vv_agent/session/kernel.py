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
from vv_agent.kernel._sql import SQLStore
from vv_agent.tools.outcomes import validate_definitive_result
from vv_agent.types import AgentTask, Message, ToolDirective, ToolExecutionResult

from . import _driver_dispatch, _driver_inputs, _driver_planning, _driver_recovery
from ._driver_scope import HeartbeatStopped as HeartbeatStopped
from ._driver_scope import _LeaseHeartbeat, _Scope
from .bindings import MissingHostBinding
from .children import child_handles
from .context import project_context
from .providers import (
    Accepted,
    Definitive,
    validate_acceptance,
    validate_provider_result,
)
from .records import InboxItem, Record, copy_json, digest, make_record
from .reducer import Attempt, ExecutionState, _fold_factory_for
from .runtime import Runtime, RuntimeNotReady, budget, model_usage
from .sql import SQLSessionTx
from .store import Conflict, Lease, LeaseLost, SequenceConflict, SessionStore, SessionTx, StoredRecord
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
        return _driver_planning.plan(
            self, oid, request, kind, number=number, dependencies=dependencies, capability=capability, purpose=purpose
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
        return _driver_recovery.unknown(self, plan, reason)

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
        return _driver_planning.model_plan(self)

    def close(self, status: str, reason: str | None, result: Any = None, *, transferred: bool = False) -> None:
        return _driver_recovery.close(self, status, reason, result, transferred=transferred)

    def dispatch(self, attempt: Attempt) -> None:
        return _driver_dispatch.dispatch(self, attempt)

    def finalize_tool(self, plan: Record, context, outcome: Definitive) -> Definitive:
        return _driver_dispatch.finalize_tool(self, plan, context, outcome)

    def step(self) -> bool:
        return _driver_recovery.step(self)


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
