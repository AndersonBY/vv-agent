"""Pure durable-record -> existing RunEvent projection. Never executes or acknowledges."""

from collections.abc import Iterable
from typing import Any

from vv_agent.budget import BudgetEnforcementBoundary, BudgetExhaustion, BudgetUsageSnapshot
from vv_agent.events import (
    AgentStartedEvent,
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    BudgetExhaustedEvent,
    BudgetSnapshotEvent,
    CycleStartedEvent,
    DiagnosticEvent,
    HandoffCompletedEvent,
    HandoffStartedEvent,
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
    SubRunCompletedEvent,
    SubRunStartedEvent,
    ToolCallCompletedEvent,
    ToolCallPlannedEvent,
    ToolCallStartedEvent,
    event_from_dict,
)
from vv_agent.interaction import HostInteractionRequest
from vv_agent.types import ModelCallOperation, ToolExecutionResult

from .children import child_handles
from .records import copy_json, digest
from .runtime import model_usage
from .store import StoredRecord


def project_records(records: Iterable[StoredRecord]) -> list[RunEvent]:
    """Pass a prefix including plans; callers filter emitted metadata.session_seq by cursor."""
    plans, events, turns = {}, [], {}
    delegations: dict[tuple[str, int], dict] = {}
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
        if r.turn_id in turns:
            common["agent_name"] = turns[r.turn_id]["definition"].get("agent_name")
        event: RunEvent | None = None
        if r.kind == "turn_started":
            turns[r.turn_id] = p
            common["agent_name"] = p["definition"]["task"].get("metadata", {}).get("agent_name")
            if p["definition"]["task"].get("metadata", {}).get("vv_session", {}).get("input_blocked"):
                continue
            event = RunStartedEvent(**common, input=p["definition"]["task"]["user_prompt"])
            events.append(event)
            agent_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/agent"}
            event = AgentStartedEvent(**agent_common)
        elif r.kind == "turn_ended":
            if p["status"] == "completed":
                completed_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/diagnostic"}
                events.append(
                    DiagnosticEvent(
                        **completed_common,
                        level="info",
                        code="run_completed",
                        details={"final_answer": p["result"], "completion_reason": p["reason"] or "no_tool_finish"},
                    )
                )
                event = RunCompletedEvent(**common, final_output=p["result"], status="completed")
            elif p["status"] in {"cancelled", "aborted"}:
                event = RunCancelledEvent(**common, reason=p["reason"] or p["status"])
            else:
                event = RunFailedEvent(**common, error=p["reason"] or "failed", status="failed")
        elif r.kind == "boundary_recorded":
            data, stage = p["data"], p["stage"]
            if stage in {"memory_started", "memory_completed"}:
                event = event_from_dict(data["event"] | {"metadata": data["event"].get("metadata", {}) | common["metadata"]})
            elif stage == "after_cycle":
                code = (
                    "after_cycle_failed"
                    if data["error"]
                    else "after_cycle_steered"
                    if data["action"] == "steer"
                    else "after_cycle_decision"
                )
                event = DiagnosticEvent(
                    **common, level="info", code=code, details={k: data[k] for k in ("action", "disallow_tools", "error")}
                )
            elif stage == "budget":
                exhaustion = BudgetExhaustion.from_dict(data["exhaustion"]) if data["exhaustion"] else None
                boundary = (
                    exhaustion.enforcement_boundary
                    if exhaustion
                    else (
                        BudgetEnforcementBoundary.TOOL_BATCH_COMPLETE
                        if "/tool_complete/" in p["boundary_id"]
                        else BudgetEnforcementBoundary.TOOL_BATCH_PREFLIGHT
                        if p["boundary_id"].endswith("/tool_batch")
                        else BudgetEnforcementBoundary.TERMINAL
                        if p["boundary_id"] == "terminal"
                        else BudgetEnforcementBoundary.MODEL_CALL_COMPLETE
                        if "/complete/" in p["boundary_id"]
                        else BudgetEnforcementBoundary.RUN_START
                        if p["boundary_id"] == "run_start"
                        else BudgetEnforcementBoundary.CYCLE_START
                    )
                )
                kwargs: dict[str, Any] = common | {
                    "enforcement_boundary": boundary,
                    "budget_usage": BudgetUsageSnapshot.from_dict(data["usage"]),
                }
                event = (
                    BudgetExhaustedEvent(**kwargs, budget_exhaustion=exhaustion) if exhaustion else BudgetSnapshotEvent(**kwargs)
                )
            elif stage == "session_memory_saved":
                event = DiagnosticEvent(
                    **common, level="info", code="session_memory_saved", details={"entry_count": len(data["state"]["entries"])}
                )
        elif r.kind == "turn_parked":
            event = RunStateChangedEvent(**common, state="wait_user")
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
                approval_common: dict[str, Any] = common | {
                    "metadata": common["metadata"]
                    | {"reason": answer.get("reason", ""), "decision_metadata": answer.get("metadata", {})}
                }
                event = ApprovalResolvedEvent(
                    **approval_common,
                    request_id=answer["request_id"],
                    tool_name=plan["request"]["name"],
                    tool_call_id=plan["request"]["id"],
                    action="allow" if answer["decision"] == "approve" else answer["decision"],
                )
            elif item["kind"] == "child_result":
                answer = item["payload"]
                parent_plan = plans[(answer["operation_id"], answer["attempt"])]
                marker = delegations.get((answer["operation_id"], answer["attempt"]), {})
                if marker.get("mode") == "handoff":
                    handoff_common = {k: v for k, v in common.items() if k != "agent_name"}
                    handoff_common["metadata"] = common["metadata"] | marker["metadata"]
                    event = HandoffCompletedEvent(
                        **handoff_common,
                        source_agent=common.get("agent_name") or "agent",
                        target_agent=marker["agent_name"],
                        tool_call_id=parent_plan["request"]["id"],
                        child_session_id=answer["session_id"],
                        child_run_id=answer["turn_id"],
                        status=answer["status"],
                    )
                else:
                    event = SubRunCompletedEvent(
                        **common,
                        parent_tool_call_id=parent_plan["request"]["id"],
                        child_session_id=answer["session_id"],
                        task_id=answer["turn_id"],
                        status=answer["status"],
                        final_output=str(answer["result"]),
                    )
            elif item["kind"] == "user" and p["target_wait_id"] and p["target_operation_id"] is None:
                event = RunStateChangedEvent(**common, state="running")
            elif item["kind"] == "user" and p["target_operation_id"]:
                plan = plans[(p["target_operation_id"], 1)]
                cycle = int(plan["dependencies"][0].rsplit("/", 1)[1])
                event = HostInteractionResponseConsumedEvent(
                    **common,
                    interaction_id=p["target_wait_id"],
                    logical_cycle=cycle,
                    operation_id=p["target_operation_id"],
                    tool_call_id=plan["request"]["id"],
                    request_digest=str(interactions[(p["target_operation_id"], 1)].request_digest),
                    command_id=item["input_id"],
                    response_digest=p["input_digest"],
                    cycle_index=cycle,
                )
        elif r.kind.startswith("op_"):
            assert r.operation_id is not None and r.attempt is not None
            key = (r.operation_id, r.attempt)
            if r.kind == "op_planned":
                plans[key] = p
            elif r.kind == "op_prepared":
                plans[key] = plans[key] | {"request": p["request"], "tool": p["tool"], "request_digest": p["request_digest"]}
            plan = plans[key]
            kind, request = plan["op_kind"], plan["request"]
            source = r.operation_id if kind == "model" else plan["dependencies"][0]
            cycle = (
                request.get("metadata", {})
                .get("vv_session", {})
                .get(
                    "cycle_index", int(source.rsplit("/", 1)[1]) if plan["purpose"] not in {"compaction", "session_memory"} else 1
                )
            )
            if kind != "model":
                cycle = plans[(source, 1)]["request"]["metadata"].get("vv_session", {}).get("cycle_index", cycle)
            if r.kind == "op_planned" and kind == "model" and plan["purpose"] == "primary" and r.attempt == 1:
                cycle_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/cycle"}
                events.append(CycleStartedEvent(**cycle_common, cycle_index=cycle))
            if r.kind == "op_completed" and kind == "model" and plan["purpose"] == "primary" and p["context"] == "normal":
                response_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/response"}
                events.append(
                    DiagnosticEvent(
                        **response_common,
                        cycle_index=cycle,
                        level="info",
                        code="cycle_llm_response",
                        details={
                            "assistant_message": p["result"].get("content", ""),
                            "tool_calls": p["result"].get("tool_calls", []),
                        },
                    )
                )
            if kind == "model":
                identity: dict[str, Any] = {
                    **common,
                    "call_id": f"{r.operation_id}/{r.attempt}",
                    "operation_id": r.operation_id,
                    "attempt": r.attempt,
                    "operation": {
                        "compaction": ModelCallOperation.MEMORY_COMPACTION,
                        "session_memory": ModelCallOperation.SESSION_MEMORY,
                        "output_repair": ModelCallOperation.OUTPUT_REPAIR,
                    }.get(plan["purpose"], ModelCallOperation.AGENT_CYCLE),
                    "metadata": common["metadata"] | {"purpose": plan["purpose"]},
                    "cycle_index": cycle,
                    "backend": turns[r.turn_id]["definition"].get("model_binding", {}).get("backend", plan["provider_binding"]),
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
                    elif p["result"].get("error_code"):
                        event = ModelCallFailedEvent(
                            **identity, outcome="definitive", usage=model_usage(p["usage"]), error_code=p["result"]["error_code"]
                        )
                    else:
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
                        operation_id=r.operation_id,
                        operation_kind="model",
                        risk="duplicate_model_request_and_cost",
                        cycle_index=cycle,
                    )
            else:
                tool: dict[str, Any] = {"tool_name": request["name"], "tool_call_id": request["id"], "cycle_index": cycle}
                if r.kind in {"op_planned", "op_started"}:
                    if r.kind == "op_started":
                        diagnostic_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/diagnostic"}
                        events.append(
                            DiagnosticEvent(
                                **diagnostic_common,
                                cycle_index=cycle,
                                level="info",
                                code="tool_started",
                                details={
                                    "tool_name": request["name"],
                                    "tool_call_id": request["id"],
                                    "tool_arguments": request["arguments"],
                                },
                            )
                        )
                    cls = ToolCallPlannedEvent if r.kind == "op_planned" else ToolCallStartedEvent
                    event = cls(
                        **common, **tool, arguments=copy_json(request["arguments"]), tool_metadata=copy_json(plan["tool"]) or None
                    )
                elif r.kind == "op_completed" and p["context"] == "normal":
                    result = ToolExecutionResult.from_dict(p["result"])
                    tool_result_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/diagnostic"}
                    events.append(
                        DiagnosticEvent(
                            **tool_result_common,
                            cycle_index=cycle,
                            level="info",
                            code="tool_result",
                            details={
                                "tool_name": request["name"],
                                "tool_call_id": request["id"],
                                "tool_arguments": request["arguments"],
                                "content": result.content,
                                "status": result.status_code.value,
                                "directive": result.directive.value,
                                "error_code": result.error_code,
                                "metadata": result.metadata,
                            },
                        )
                    )
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
                    )
                elif r.kind == "op_unknown":
                    event = OperationAmbiguousEvent(
                        **common,
                        operation_id=r.operation_id,
                        operation_kind="tool",
                        risk="tool_outcome_unknown",
                        cycle_index=cycle,
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
                            interaction_id=h["interaction_id"],
                            logical_cycle=cycle,
                            operation_id=r.operation_id,
                            tool_call_id=request["id"],
                            request_digest=str(interaction.request_digest),
                            prompt=h["question"],
                            cycle_index=cycle,
                        )
                    elif h["kind"] == "child":
                        marker = p.get("delegation", {})
                        delegations[key] = marker
                        if marker.get("mode") == "handoff":
                            handoff_common = {k: v for k, v in common.items() if k != "agent_name"}
                            handoff_common["metadata"] = common["metadata"] | marker["metadata"]
                            event = HandoffStartedEvent(
                                **handoff_common,
                                source_agent=common.get("agent_name") or "agent",
                                target_agent=marker["agent_name"],
                                tool_call_id=request["id"],
                                child_session_id=h["session_id"],
                            )
                        else:
                            for index, member in enumerate(child_handles(h)):
                                child_common: dict[str, Any] = common | {"event_id": f"sk/{identity_digest}/child/{index}"}
                                events.append(
                                    SubRunStartedEvent(
                                        **child_common,
                                        parent_tool_call_id=request["id"],
                                        child_session_id=member["session_id"],
                                        task_id=member["turn_id"],
                                    )
                                )
                    else:
                        event = RunStateChangedEvent(**common, state="parked")
        if event is not None:
            events.append(event)
    for projected in events:
        object.__setattr__(projected, "version", "v6")
    return events
