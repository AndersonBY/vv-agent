"""Advance, recover and close turns using the driver's retained state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.tool_results import build_skipped_result
from vv_agent.types import Message, ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus

from ._driver_scope import _invoke
from .approval import resolve_approval
from .children import cancel_children
from .compaction import compact_context, finish_summary
from .lifecycle import after_cycle
from .memory import save_projection
from .output import prepare_output, serializable_output
from .providers import Accepted, Definitive, ProviderResultInvalid, validate_acceptance
from .records import Record, copy_json, digest, make_record
from .runtime import budget
from .summary_requests import pending_compaction_requests, summary_pending

if TYPE_CHECKING:
    from .kernel import _Driver


def unknown(self: _Driver, plan: Record, reason: str) -> Record:
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


def step(self: _Driver) -> bool:
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
    models = [(oid, op) for oid, op in operations if op.kind == "model" and op.attempts[1].plan._payload["purpose"] == "primary"]
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


def close(self: _Driver, status: str, reason: str | None, result: Any = None, *, transferred: bool = False) -> None:
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
