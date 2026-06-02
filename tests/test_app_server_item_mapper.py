from __future__ import annotations

from vv_agent.app_server.item_mapper import item_id_for_event, map_run_event
from vv_agent.events import (
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    AssistantDeltaEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)


def test_item_id_for_event_is_stable_for_replay() -> None:
    event = RunStartedEvent(run_id="run_1", trace_id="trace_1", input="hello", event_id="evt_abc")

    assert item_id_for_event(event) == "item_abc"


def test_run_started_maps_to_completed_user_message() -> None:
    event = RunStartedEvent(run_id="run_1", trace_id="trace_1", input="hello", event_id="evt_1", created_at=1)

    projection = map_run_event(event, thread_id="thread_1", turn_id="turn_1")

    assert projection.item is not None
    assert projection.item.item_type == "userMessage"
    assert projection.item.status == "completed"
    assert projection.item.payload == {"text": "hello"}
    assert projection.notification_method == "item/completed"


def test_assistant_delta_maps_to_delta_notification() -> None:
    event = AssistantDeltaEvent(run_id="run_1", trace_id="trace_1", delta="hel", event_id="evt_2", created_at=2)

    projection = map_run_event(event, thread_id="thread_1", turn_id="turn_1")

    assert projection.item is not None
    assert projection.item.item_type == "agentMessage"
    assert projection.item.status == "inProgress"
    assert projection.notification_method == "item/agentMessage/delta"
    assert projection.notification_params["delta"] == "hel"


def test_tool_call_events_map_to_started_and_completed_items() -> None:
    started = ToolCallStartedEvent(
        run_id="run_1",
        trace_id="trace_1",
        tool_name="bash",
        tool_call_id="call_1",
        event_id="evt_3",
        created_at=3,
    )
    completed = ToolCallCompletedEvent(
        run_id="run_1",
        trace_id="trace_1",
        tool_name="bash",
        tool_call_id="call_1",
        status="success",
        event_id="evt_4",
        created_at=4,
    )

    started_projection = map_run_event(started, thread_id="thread_1", turn_id="turn_1")
    completed_projection = map_run_event(completed, thread_id="thread_1", turn_id="turn_1")

    assert started_projection.item is not None
    assert started_projection.item.item_type == "toolCall"
    assert started_projection.item.status == "started"
    assert started_projection.notification_method == "item/started"
    assert completed_projection.item is not None
    assert completed_projection.item.status == "completed"
    assert completed_projection.notification_method == "item/completed"


def test_approval_events_map_to_approval_items() -> None:
    requested = ApprovalRequestedEvent(
        run_id="run_1",
        trace_id="trace_1",
        tool_name="bash",
        tool_call_id="call_1",
        message="Run command",
        request_id="approval_1",
        event_id="evt_5",
        created_at=5,
    )
    resolved = ApprovalResolvedEvent(
        run_id="run_1",
        trace_id="trace_1",
        tool_name="bash",
        tool_call_id="call_1",
        approved=True,
        request_id="approval_1",
        event_id="evt_6",
        created_at=6,
    )

    requested_projection = map_run_event(requested, thread_id="thread_1", turn_id="turn_1")
    resolved_projection = map_run_event(resolved, thread_id="thread_1", turn_id="turn_1")

    assert requested_projection.item is not None
    assert requested_projection.item.item_type == "approval"
    assert requested_projection.item.status == "started"
    assert requested_projection.item.payload["requestId"] == "approval_1"
    assert resolved_projection.item is not None
    assert resolved_projection.item.status == "completed"
    assert resolved_projection.item.payload["approved"] is True


def test_terminal_events_map_to_turn_metadata_or_error_item() -> None:
    completed = RunCompletedEvent(run_id="run_1", trace_id="trace_1", final_output="done", status="completed", event_id="evt_7")
    failed = RunFailedEvent(run_id="run_1", trace_id="trace_1", error="boom", event_id="evt_8", created_at=8)

    completed_projection = map_run_event(completed, thread_id="thread_1", turn_id="turn_1")
    failed_projection = map_run_event(failed, thread_id="thread_1", turn_id="turn_1")

    assert completed_projection.item is None
    assert completed_projection.terminal_turn == {"status": "completed", "finalOutput": "done"}
    assert failed_projection.item is not None
    assert failed_projection.item.item_type == "error"
    assert failed_projection.item.payload == {"message": "boom"}
