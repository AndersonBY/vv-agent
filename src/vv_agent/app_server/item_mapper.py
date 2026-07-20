from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vv_agent.app_server.protocol import ApprovalDecision, ApprovalRequestParams, ApprovalResolveParams, ThreadItem, WarningParams
from vv_agent.events import (
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    AssistantDeltaEvent,
    ModelToolCallProgressEvent,
    RunCompletedEvent,
    RunEvent,
    RunFailedEvent,
    RunStartedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)


@dataclass(frozen=True, slots=True)
class ItemProjection:
    item: ThreadItem | None = None
    notification_method: str | None = None
    notification_params: dict[str, Any] = field(default_factory=dict)
    additional_notifications: tuple[tuple[str, dict[str, Any]], ...] = ()
    terminal_turn: dict[str, Any] | None = None


def item_id_for_event(event: RunEvent) -> str:
    return f"item_{event.event_id}"


def map_run_event(event: RunEvent, *, thread_id: str, turn_id: str) -> ItemProjection:
    if isinstance(event, RunStartedEvent):
        item = _item(
            event,
            thread_id=thread_id,
            turn_id=turn_id,
            item_type="userMessage",
            status="completed",
            payload={"text": event.input},
        )
        return _projection(item, "item/completed")
    if isinstance(event, AssistantDeltaEvent):
        item = _item(
            event,
            thread_id=thread_id,
            turn_id=turn_id,
            item_type="agentMessage",
            status="inProgress",
            payload={"delta": event.delta},
        )
        return _projection(item, "item/agentMessage/delta", {"delta": event.delta})
    if isinstance(event, ModelToolCallProgressEvent):
        delta = {
            "toolCallId": event.tool_call_id,
            "toolCallIndex": event.tool_call_index,
            "toolName": event.tool_name,
            "argumentsChars": event.arguments_chars,
        }
        item = _item(
            event,
            thread_id=thread_id,
            turn_id=turn_id,
            item_type="toolCall",
            status="inProgress",
            payload={"toolCallId": delta["toolCallId"], "toolName": delta["toolName"]},
        )
        return _projection(item, "item/toolCall/delta", {"delta": delta})
    if event.type == "cycle_llm_response":
        assistant_message = event.metadata.get("assistant_message")
        if assistant_message:
            item = _item(
                event,
                thread_id=thread_id,
                turn_id=turn_id,
                item_type="agentMessage",
                status="completed",
                payload={"text": str(assistant_message)},
            )
            return _projection(item, "item/completed")
    if isinstance(event, ToolCallStartedEvent):
        item = _item(
            event,
            thread_id=thread_id,
            turn_id=turn_id,
            item_type="toolCall",
            status="started",
            payload={"toolName": event.tool_name, "toolCallId": event.tool_call_id},
        )
        arguments = event.metadata.get("tool_arguments")
        if not isinstance(arguments, dict):
            arguments = event.metadata.get("arguments")
        arguments = dict(arguments) if isinstance(arguments, dict) else {}
        return _projection(
            item,
            "item/started",
            additional_notifications=(("item/toolCall/delta", {**item.to_dict(), "delta": arguments}),),
        )
    if isinstance(event, ToolCallCompletedEvent):
        item = _item(
            event,
            thread_id=thread_id,
            turn_id=turn_id,
            item_type="toolCall",
            status=_tool_item_status(event.status),
            payload={"toolName": event.tool_name, "toolCallId": event.tool_call_id, "status": event.status},
        )
        return _projection(item, "item/completed")
    if isinstance(event, ApprovalRequestedEvent):
        item = _item(
            event,
            thread_id=thread_id,
            turn_id=turn_id,
            item_type="approval",
            status="started",
            payload={
                "requestId": event.request_id,
                "toolName": event.tool_name,
                "toolCallId": event.tool_call_id,
                "message": event.message,
                "arguments": dict(event.metadata.get("arguments") or {}),
            },
        )
        return _projection(
            item,
            "item/started",
            additional_notifications=(
                (
                    "approval/requested",
                    ApprovalRequestParams(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        request_id=event.request_id,
                        tool_call_id=event.tool_call_id,
                        tool_name=event.tool_name,
                        preview=event.message,
                        arguments=dict(event.metadata.get("arguments") or {}),
                    ).to_dict(),
                ),
            ),
        )
    if isinstance(event, ApprovalResolvedEvent):
        action = event.action
        if action is None:
            action = ApprovalDecision.ALLOW.value if event.approved else ApprovalDecision.DENY.value
        try:
            decision = ApprovalDecision.from_wire(action)
        except ValueError:
            decision = ApprovalDecision.ALLOW if event.approved else ApprovalDecision.DENY
        reason = str(event.metadata.get("reason") or "")
        decision_metadata = dict(event.metadata.get("decision_metadata") or {})
        item = _item(
            event,
            thread_id=thread_id,
            turn_id=turn_id,
            item_type="approval",
            status="completed",
            payload={
                "requestId": event.request_id,
                "toolName": event.tool_name,
                "toolCallId": event.tool_call_id,
                "action": decision.value,
                "approved": event.approved,
                "reason": reason,
                "decisionMetadata": decision_metadata,
            },
        )
        return _projection(
            item,
            "item/completed",
            additional_notifications=(
                (
                    "approval/resolved",
                    ApprovalResolveParams(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        request_id=event.request_id,
                        decision=decision,
                        reason=reason,
                        metadata=decision_metadata,
                    ).to_dict(),
                ),
            ),
        )
    if isinstance(event, RunCompletedEvent):
        terminal_turn = {"status": event.status, "finalOutput": event.final_output}
        if event.final_output is None:
            return ItemProjection(terminal_turn=terminal_turn)
        item = _item(
            event,
            thread_id=thread_id,
            turn_id=turn_id,
            item_type="agentMessage",
            status="completed",
            payload={"text": event.final_output},
        )
        return ItemProjection(
            item=item,
            notification_method="item/completed",
            notification_params=item.to_dict(),
            terminal_turn=terminal_turn,
        )
    if isinstance(event, RunFailedEvent):
        item = _item(
            event,
            thread_id=thread_id,
            turn_id=turn_id,
            item_type="error",
            status="completed",
            payload={"message": event.error},
        )
        return _projection(
            item,
            "item/completed",
            additional_notifications=(
                (
                    "error/warning",
                    WarningParams(
                        message=event.error,
                        code="run_failed",
                    ).to_dict(),
                ),
            ),
        )
    return ItemProjection()


def _tool_item_status(status: str) -> str:
    return {
        "started": "started",
        "success": "completed",
        "error": "failed",
        "wait_response": "inProgress",
    }.get(status, "completed")


def _item(
    event: RunEvent,
    *,
    thread_id: str,
    turn_id: str,
    item_type: str,
    status: str,
    payload: dict[str, Any],
) -> ThreadItem:
    return ThreadItem(
        item_id=item_id_for_event(event),
        thread_id=thread_id,
        turn_id=turn_id,
        item_type=item_type,
        status=status,
        payload=payload,
        created_at=event.created_at,
        updated_at=event.created_at,
    )


def _projection(
    item: ThreadItem,
    method: str,
    extra_params: dict[str, Any] | None = None,
    *,
    additional_notifications: tuple[tuple[str, dict[str, Any]], ...] = (),
) -> ItemProjection:
    params = item.to_dict()
    if extra_params:
        params.update(extra_params)
    return ItemProjection(
        item=item,
        notification_method=method,
        notification_params=params,
        additional_notifications=additional_notifications,
    )
