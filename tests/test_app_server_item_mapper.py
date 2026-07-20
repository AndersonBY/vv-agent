from __future__ import annotations

import json
from pathlib import Path

from vv_agent.app_server.item_mapper import item_id_for_event, map_run_event
from vv_agent.events import (
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    AssistantDeltaEvent,
    ModelToolCallProgressEvent,
    ReasoningDeltaEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    ToolCallCompletedEvent,
    ToolCallPlannedEvent,
    ToolCallStartedEvent,
)
from vv_agent.tools.metadata import ToolMetadata

_TOOL_METADATA_CONTRACT = json.loads(
    (Path(__file__).parent / "fixtures" / "parity" / "tool_metadata_v1.json").read_text(encoding="utf-8")
)


def test_item_id_for_event_is_stable_for_replay() -> None:
    event = RunStartedEvent(run_id="run_1", trace_id="trace_1", input="hello", event_id="evt_abc")

    assert item_id_for_event(event) == "item_evt_abc"


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
    assert projection.item.payload == {"delta": "hel"}
    assert projection.notification_params["delta"] == "hel"


def test_tool_call_events_map_to_started_and_completed_items() -> None:
    started = ToolCallStartedEvent(
        run_id="run_1",
        trace_id="trace_1",
        tool_name="bash",
        tool_call_id="call_1",
        event_id="evt_3",
        created_at=3,
        metadata={"tool_arguments": {"cmd": "pytest"}},
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
    assert started_projection.additional_notifications == (
        (
            "item/toolCall/delta",
            {**started_projection.item.to_dict(), "delta": {"cmd": "pytest"}},
        ),
    )
    assert completed_projection.item is not None
    assert completed_projection.item.status == "completed"
    assert completed_projection.notification_method == "item/completed"

    failed = ToolCallCompletedEvent(
        run_id="run_1",
        trace_id="trace_1",
        tool_name="bash",
        tool_call_id="call_2",
        status="error",
    )
    failed_projection = map_run_event(failed, thread_id="thread_1", turn_id="turn_1")
    assert failed_projection.item is not None
    assert failed_projection.item.status == "failed"


def test_planned_tool_call_is_not_projected_as_an_app_server_item() -> None:
    projection_contract = _TOOL_METADATA_CONTRACT["app_server_projection"]
    assert projection_contract["tool_call_planned"] == "no_notification"
    assert projection_contract["planned_is_never_presented_as_execution_started"] is True
    event = ToolCallPlannedEvent(
        run_id="run_1",
        trace_id="trace_1",
        tool_name="lookup",
        tool_call_id="call_planned",
        arguments={"query": "parity"},
        event_id="evt_planned",
        created_at=3,
    )

    projection = map_run_event(event, thread_id="thread_1", turn_id="turn_1")

    assert projection == projection.__class__()


def test_typed_tool_metadata_and_completed_observations_are_projected_without_legacy_fabrication() -> None:
    projection_contract = _TOOL_METADATA_CONTRACT["app_server_projection"]
    assert "toolMetadata" in projection_contract["tool_call_started"]
    assert all(
        field_name in projection_contract["tool_call_completed"]
        for field_name in ("directive", "errorCode", "executionStarted", "durationMs", "toolMetadata")
    )
    producer_case = next(
        case for case in _TOOL_METADATA_CONTRACT["producer_cases"] if case["name"] == "executed_tool"
    )
    metadata = ToolMetadata.from_dict(producer_case["tool_metadata"])
    started = ToolCallStartedEvent(
        run_id="run_1",
        trace_id="trace_1",
        tool_name="inspect",
        tool_call_id="call_typed",
        arguments={"path": "README.md"},
        tool_metadata=metadata,
        event_id="evt_typed_started",
        created_at=3,
    )
    completed = ToolCallCompletedEvent(
        run_id="run_1",
        trace_id="trace_1",
        tool_name="inspect",
        tool_call_id="call_typed",
        status="success",
        directive="continue",
        error_code=None,
        execution_started=True,
        duration_ms=7,
        tool_metadata=metadata,
        event_id="evt_typed_completed",
        created_at=4,
    )
    legacy = ToolCallCompletedEvent(
        run_id="run_1",
        trace_id="trace_1",
        tool_name="lookup",
        tool_call_id="call_legacy",
        status="success",
        event_id="evt_legacy_completed",
        created_at=5,
    )

    started_projection = map_run_event(started, thread_id="thread_1", turn_id="turn_1")
    completed_projection = map_run_event(completed, thread_id="thread_1", turn_id="turn_1")
    legacy_projection = map_run_event(legacy, thread_id="thread_1", turn_id="turn_1")

    assert started_projection.item is not None
    assert started_projection.item.payload["toolMetadata"] == {
        "sideEffect": "read",
        "idempotency": "supported",
        "terminal": False,
        "capabilityTags": ["source.inspect"],
        "costDimensions": ["workspace.bytes_read"],
    }
    assert started_projection.additional_notifications[0][1]["payload"]["toolMetadata"] == started_projection.item.payload[
        "toolMetadata"
    ]
    assert started_projection.additional_notifications[0][1]["delta"] == {"path": "README.md"}
    assert completed_projection.item is not None
    assert completed_projection.item.payload == {
        "toolName": "inspect",
        "toolCallId": "call_typed",
        "status": "success",
        "directive": "continue",
        "errorCode": None,
        "executionStarted": True,
        "durationMs": 7,
        "toolMetadata": {
            "sideEffect": "read",
            "idempotency": "supported",
            "terminal": False,
            "capabilityTags": ["source.inspect"],
            "costDimensions": ["workspace.bytes_read"],
        },
    }
    assert legacy_projection.item is not None
    assert legacy_projection.item.payload == {
        "toolName": "lookup",
        "toolCallId": "call_legacy",
        "status": "success",
    }


def test_tool_call_progress_maps_to_tool_call_delta_notification() -> None:
    event = ModelToolCallProgressEvent(
        run_id="run_1",
        trace_id="trace_1",
        tool_call_id="call_1",
        tool_call_index=0,
        tool_name="bash",
        arguments_chars=12,
        event_id="evt_delta",
        created_at=3,
    )

    projection = map_run_event(event, thread_id="thread_1", turn_id="turn_1")

    assert projection.notification_method == "item/toolCall/delta"
    assert projection.notification_params == {
        "itemId": "item_evt_delta",
        "threadId": "thread_1",
        "turnId": "turn_1",
        "type": "toolCall",
        "status": "inProgress",
        "payload": {"toolCallId": "call_1", "toolName": "bash"},
        "createdAt": 3,
        "updatedAt": 3,
        "delta": {
            "toolCallId": "call_1",
            "toolCallIndex": 0,
            "toolName": "bash",
            "argumentsChars": 12,
        },
    }


def test_reasoning_delta_is_not_projected_as_visible_assistant_output() -> None:
    event = ReasoningDeltaEvent(
        run_id="run_1",
        trace_id="trace_1",
        delta="private plan",
        event_id="evt_reasoning",
        created_at=3,
    )

    projection = map_run_event(event, thread_id="thread_1", turn_id="turn_1")

    assert projection.item is None
    assert projection.notification_method is None
    assert projection.notification_params == {}


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
        metadata={"arguments": {"cmd": "pytest"}, "tool_name": "bash"},
    )
    resolved = ApprovalResolvedEvent(
        run_id="run_1",
        trace_id="trace_1",
        tool_name="bash",
        tool_call_id="call_1",
        action="allow_session",
        request_id="approval_1",
        event_id="evt_6",
        created_at=6,
        metadata={
            "action": "allow_session",
            "reason": "approved by owner",
            "decision_metadata": {"ticket": 7},
        },
    )

    requested_projection = map_run_event(requested, thread_id="thread_1", turn_id="turn_1")
    resolved_projection = map_run_event(resolved, thread_id="thread_1", turn_id="turn_1")

    assert requested_projection.item is not None
    assert requested_projection.item.item_type == "approval"
    assert requested_projection.item.status == "started"
    assert requested_projection.item.payload["requestId"] == "approval_1"
    assert requested_projection.item.payload["arguments"] == {"cmd": "pytest"}
    assert requested_projection.additional_notifications[0][0] == "approval/requested"
    assert requested_projection.additional_notifications[0][1]["arguments"] == {"cmd": "pytest"}
    assert resolved_projection.item is not None
    assert resolved_projection.item.status == "completed"
    assert resolved_projection.item.payload["approved"] is True
    assert resolved_projection.item.payload["action"] == "allow_session"
    assert resolved_projection.item.payload["reason"] == "approved by owner"
    assert resolved_projection.item.payload["decisionMetadata"] == {"ticket": 7}
    assert resolved_projection.additional_notifications[0][0] == "approval/resolved"
    assert resolved_projection.additional_notifications[0][1] == {
        "threadId": "thread_1",
        "turnId": "turn_1",
        "requestId": "approval_1",
        "decision": "allow_session",
        "reason": "approved by owner",
        "metadata": {"ticket": 7},
    }


def test_terminal_events_map_to_turn_metadata_or_error_item() -> None:
    completed = RunCompletedEvent(run_id="run_1", trace_id="trace_1", final_output="done", status="completed", event_id="evt_7")
    failed = RunFailedEvent(run_id="run_1", trace_id="trace_1", error="boom", event_id="evt_8", created_at=8)

    completed_projection = map_run_event(completed, thread_id="thread_1", turn_id="turn_1")
    failed_projection = map_run_event(failed, thread_id="thread_1", turn_id="turn_1")

    assert completed_projection.item is not None
    assert completed_projection.item.item_type == "agentMessage"
    assert completed_projection.item.status == "completed"
    assert completed_projection.item.payload == {"text": "done"}
    assert completed_projection.notification_method == "item/completed"
    assert completed_projection.terminal_turn == {"status": "completed", "finalOutput": "done"}
    assert failed_projection.item is not None
    assert failed_projection.item.item_type == "error"
    assert failed_projection.item.payload == {"message": "boom"}
    assert failed_projection.additional_notifications == (
        (
            "error/warning",
            {"message": "boom", "code": "run_failed"},
        ),
    )
