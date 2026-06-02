from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vv_agent.app_server.protocol import ThreadItem
from vv_agent.events import (
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    AssistantDeltaEvent,
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
    terminal_turn: dict[str, Any] | None = None


def item_id_for_event(event: RunEvent) -> str:
    return f"item_{event.event_id.removeprefix('evt_')}"


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
        return _projection(item, "item/started")
    if isinstance(event, ToolCallCompletedEvent):
        item = _item(
            event,
            thread_id=thread_id,
            turn_id=turn_id,
            item_type="toolCall",
            status="completed",
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
            },
        )
        return _projection(item, "item/started")
    if isinstance(event, ApprovalResolvedEvent):
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
                "approved": event.approved,
            },
        )
        return _projection(item, "item/completed")
    if isinstance(event, RunCompletedEvent):
        return ItemProjection(terminal_turn={"status": event.status, "finalOutput": event.final_output})
    if isinstance(event, RunFailedEvent):
        item = _item(
            event,
            thread_id=thread_id,
            turn_id=turn_id,
            item_type="error",
            status="completed",
            payload={"message": event.error},
        )
        return _projection(item, "item/completed")
    return ItemProjection()


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


def _projection(item: ThreadItem, method: str, extra_params: dict[str, Any] | None = None) -> ItemProjection:
    params = item.to_dict()
    if extra_params:
        params.update(extra_params)
    return ItemProjection(item=item, notification_method=method, notification_params=params)
