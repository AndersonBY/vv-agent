"""Pure durable-record -> existing RunEvent projection. Never executes or acknowledges."""

from collections.abc import Iterable
from typing import Any

from vv_agent.events import (
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    HostInteractionRequestedEvent,
    HostInteractionResponseConsumedEvent,
    ModelCallCompletedEvent,
    ModelCallFailedEvent,
    ModelCallStartedEvent,
    ModelRetryDuplicateRiskEvent,
    OperationAmbiguousEvent,
    RunCancelledEvent,
    RunCompletedEvent,
    RunEvent,
    RunFailedEvent,
    RunStartedEvent,
    RunStateChangedEvent,
    ToolCallCompletedEvent,
    ToolCallPlannedEvent,
    ToolCallStartedEvent,
)
from vv_agent.interaction import HostInteractionRequest
from vv_agent.types import ModelCallOperation, ToolExecutionResult

from .records import digest
from .runtime import model_usage
from .store import StoredRecord


def project_records(records: Iterable[StoredRecord]) -> list[RunEvent]:
    """Pass a prefix including plans; callers filter emitted metadata.session_seq by cursor."""
    plans, events = {}, []
    interactions: dict[tuple[str, int], HostInteractionRequest] = {}
    for stored in records:
        r, p = stored.record, stored.record.payload
        identity_digest = digest([r.session_id, r.record_id])
        common: dict[str, Any] = {
            "run_id": r.turn_id or r.session_id,
            "trace_id": r.turn_id or r.session_id,
            "session_id": r.session_id,
            "event_id": f"sk/{identity_digest}/event",
            "created_at": stored.created_ms / 1000,
            "metadata": {"session_seq": stored.seq},
        }
        event: RunEvent | None = None
        if r.kind == "turn_started":
            if p["definition"]["task"].get("metadata", {}).get("session_input_blocked"):
                continue
            event = RunStartedEvent(**common, input=p["definition"]["task"]["user_prompt"])
        elif r.kind == "turn_ended":
            if p["status"] == "completed":
                event = RunCompletedEvent(**common, final_output=p["result"], status="completed")
            elif p["status"] in {"cancelled", "aborted"}:
                event = RunCancelledEvent(**common, reason=p["reason"] or p["status"])
            else:
                event = RunFailedEvent(**common, error=p["reason"] or "failed", status="failed")
        elif r.kind == "input_applied" and p["disposition"] == "applied":
            item = p["input"]
            if item["kind"] == "control":
                action = item["payload"]["action"]
                event = RunStateChangedEvent(
                    **common,
                    state=action,
                    cancel_requested={"from": False, "to": True} if action in {"cancel", "abort", "close"} else None,
                )
            elif item["kind"] == "approval_answer":
                answer = item["payload"]
                plan = plans[(answer["operation_id"], answer["attempt"])]
                event = ApprovalResolvedEvent(
                    **common,
                    request_id=answer["request_id"],
                    tool_name=plan["request"]["name"],
                    tool_call_id=plan["request"]["id"],
                    action="allow" if answer["decision"] == "approve" else "deny",
                )
            elif item["kind"] == "user" and p["target_operation_id"]:
                plan = plans[(p["target_operation_id"], 1)]
                cycle = int(plan["dependencies"][0].rsplit("/", 1)[1])
                event = HostInteractionResponseConsumedEvent(
                    **common,
                    checkpoint_key=r.session_id,
                    resume_attempt=1,
                    interaction_id=p["target_wait_id"],
                    logical_cycle=cycle,
                    operation_id=p["target_operation_id"],
                    tool_call_id=plan["request"]["id"],
                    request_digest=str(interactions[(p["target_operation_id"], 1)].request_digest),
                    command_id=item["input_id"],
                    response_digest=p["input_digest"],
                    consumed_revision=stored.seq,
                    cycle_index=cycle,
                )
        elif r.kind.startswith("op_"):
            assert r.operation_id is not None and r.attempt is not None
            key = (r.operation_id, r.attempt)
            if r.kind == "op_planned":
                plans[key] = p
            plan = plans[key]
            kind, request = plan["op_kind"], plan["request"]
            source = r.operation_id if kind == "model" else plan["dependencies"][0]
            cycle = int(source.rsplit("/", 1)[1]) if plan["purpose"] != "compaction" else request["metadata"]["cycle_index"]
            if kind == "model":
                identity: dict[str, Any] = {
                    **common,
                    "call_id": f"{r.operation_id}/{r.attempt}",
                    "operation_id": r.operation_id,
                    "attempt": r.attempt,
                    "operation": ModelCallOperation.MEMORY_COMPACTION
                    if plan["purpose"] == "compaction"
                    else ModelCallOperation.AGENT_CYCLE,
                    "cycle_index": cycle,
                    "backend": plan["provider_binding"],
                    "model": request["model"],
                }
                if r.kind == "op_started":
                    event = ModelCallStartedEvent(**identity)
                elif r.kind == "op_completed":
                    if not p["execution_started"]:
                        event = ModelCallFailedEvent(
                            **identity,
                            outcome="definitive",
                            usage=model_usage(p["usage"]),
                            error_code=p["result"].get("error_code", "not_executed"),
                        )
                    elif p["context"] != "audit":
                        event = ModelCallCompletedEvent(**identity, usage=model_usage(p["usage"]))
                elif r.kind == "op_unknown":
                    events.append(
                        ModelCallFailedEvent(
                            **identity, outcome="ambiguous", usage=model_usage(None), error_code="model_outcome_unknown"
                        )
                    )
                    common["event_id"] = f"sk/{identity_digest}/duplicate-risk"
                    event = ModelRetryDuplicateRiskEvent(
                        **common,
                        checkpoint_key=r.session_id,
                        operation_id=r.operation_id,
                        operation_kind="model",
                        risk="duplicate_model_request_and_cost",
                    )
            else:
                tool: dict[str, Any] = {"tool_name": request["name"], "tool_call_id": request["id"], "cycle_index": cycle}
                if r.kind in {"op_planned", "op_started"}:
                    cls = ToolCallPlannedEvent if r.kind == "op_planned" else ToolCallStartedEvent
                    event = cls(**common, **tool, arguments=request["arguments"], tool_metadata=plan["tool"] or None)
                elif r.kind == "op_completed" and p["context"] == "normal":
                    result = ToolExecutionResult.from_dict(p["result"])
                    event = ToolCallCompletedEvent(
                        **common,
                        **tool,
                        status=result.status_code.value.lower(),
                        directive=result.directive.value,
                        error_code=result.error_code,
                        execution_started=p["execution_started"],
                        duration_ms=None,
                        operation_id=r.operation_id,
                        attempt=r.attempt,
                        checkpoint_key=r.session_id,
                    )
                elif r.kind == "op_unknown":
                    event = OperationAmbiguousEvent(
                        **common,
                        checkpoint_key=r.session_id,
                        operation_id=r.operation_id,
                        operation_kind="tool",
                        risk="tool_outcome_unknown",
                        idempotency_support=(plan["tool"] or {}).get("idempotency", "unknown"),
                    )
                elif r.kind == "op_parked":
                    h = p["handle"]
                    if h["kind"] == "approval":
                        event = ApprovalRequestedEvent(**common, **tool, request_id=h["request_id"], message="Approval required")
                    elif h["kind"] == "user":
                        interaction = HostInteractionRequest(
                            interaction_id=h["interaction_id"],
                            logical_cycle=cycle,
                            operation_id=r.operation_id,
                            tool_call_id=request["id"],
                            prompt=h["question"],
                        )
                        interactions[key] = interaction
                        event = HostInteractionRequestedEvent(
                            **common,
                            checkpoint_key=r.session_id,
                            resume_attempt=r.attempt,
                            interaction_id=h["interaction_id"],
                            logical_cycle=cycle,
                            operation_id=r.operation_id,
                            tool_call_id=request["id"],
                            request_digest=str(interaction.request_digest),
                            prompt=h["question"],
                            cycle_index=cycle,
                        )
                    else:
                        event = RunStateChangedEvent(**common, state="parked")
        if event is not None:
            events.append(event)
    return events
