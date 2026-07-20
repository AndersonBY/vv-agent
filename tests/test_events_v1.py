from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict

import pytest

from vv_agent import (
    AgentStartedEvent,
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    AssistantDeltaEvent,
    CycleStartedEvent,
    HandoffCompletedEvent,
    HandoffEvent,
    HandoffStartedEvent,
    LLMStartedEvent,
    MemoryCompactedEvent,
    ModelToolCallProgressEvent,
    ModelToolCallStartedEvent,
    ReasoningDeltaEvent,
    RunCancelledEvent,
    RunCompletedEvent,
    RunEvent,
    RunFailedEvent,
    RunStartedEvent,
    RunStateChangedEvent,
    SessionPersistedEvent,
    SubRunCompletedEvent,
    SubRunStartedEvent,
    ToolApprovalRequestedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
    event_from_dict,
)
from vv_agent.events import ToolCallPlannedEvent, event_from_runtime_log
from vv_agent.tools.metadata import ToolMetadata

PARITY_FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "run_events_v1.jsonl"
PARITY_FIXTURE_SHA256 = "3c45e094fdc8b2555e0d91664778866c4bc1dfe8c516a19d5cbe99209f6e3e41"
BUDGET_FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "budget_events_v1.jsonl"
BUDGET_FIXTURE_SHA256 = "3267292737ac6bf63ec4ee691fe0ef07f3e2cadd5a69098e3e267f4f6b692d2e"
PARITY_EVENT_TYPES = [
    "run_started",
    "agent_started",
    "cycle_started",
    "llm_started",
    "assistant_delta",
    "reasoning_delta",
    "model_tool_call_started",
    "model_tool_call_progress",
    "tool_call_planned",
    "tool_call_started",
    "tool_call_completed",
    "approval_requested",
    "approval_resolved",
    "memory_compact_started",
    "memory_compact_completed",
    "sub_run_started",
    "sub_run_completed",
    "handoff",
    "handoff_started",
    "handoff_completed",
    "session_persisted",
    "run_state_changed",
    "run_completed",
    "run_failed",
    "run_cancelled",
    "budget_snapshot",
    "budget_exhausted",
    "checkpoint_created",
    "checkpoint_resumed",
    "operation_replayed",
    "operation_ambiguous",
    "model_retry_duplicate_risk",
    "reconciliation_resolved",
    "reconciliation_required",
]


def test_run_event_v1_has_stable_identity_and_timing() -> None:
    event = RunStartedEvent(
        run_id="run_1",
        trace_id="trace_1",
        input="hello",
        session_id="session_1",
        agent_name="assistant",
    )

    payload = event.to_dict()

    assert payload["version"] == "v1"
    assert payload["type"] == "run_started"
    assert payload["event_id"].startswith("evt_")
    assert payload["run_id"] == "run_1"
    assert payload["trace_id"] == "trace_1"
    assert payload["session_id"] == "session_1"
    assert payload["agent_name"] == "assistant"
    assert payload["created_at"] > 0
    assert payload["input"] == "hello"


def test_event_from_dict_reads_legacy_created_at_milliseconds_as_seconds() -> None:
    event = event_from_dict(
        {
            "version": "v1",
            "type": "run_started",
            "event_id": "evt_legacy",
            "run_id": "run_legacy",
            "trace_id": "trace_legacy",
            "created_at_ms": 123456.789,
            "input": "hello",
        }
    )

    assert event.created_at == 123.456789
    assert event.to_dict()["created_at"] == 123.456789
    assert "created_at_ms" not in event.to_dict()


def test_tool_completion_status_is_canonical_lowercase() -> None:
    event = ToolCallCompletedEvent(
        run_id="run_status",
        trace_id="trace_status",
        tool_name="lookup",
        tool_call_id="call_status",
        status="WAIT_RESPONSE",
    )

    assert event.status == "wait_response"
    assert event.to_dict()["status"] == "wait_response"


def test_typed_tool_lifecycle_fields_are_normalized_and_round_trip() -> None:
    tool_metadata = ToolMetadata.from_dict(
        {
            "side_effect": "network",
            "idempotency": "supported",
            "terminal": False,
            "capability_tags": ["source.inspect"],
            "cost_dimensions": ["workflow.credit"],
        }
    )
    planned = ToolCallPlannedEvent(
        run_id="run_tool",
        trace_id="trace_tool",
        tool_name="search",
        tool_call_id="call_tool",
        arguments={"query": "wire parity"},
        tool_metadata=tool_metadata,
    )
    started = ToolCallStartedEvent(
        run_id="run_tool",
        trace_id="trace_tool",
        tool_name="search",
        tool_call_id="call_tool",
        arguments={"query": "wire parity"},
        tool_metadata=tool_metadata,
    )
    completed = ToolCallCompletedEvent(
        run_id="run_tool",
        trace_id="trace_tool",
        tool_name="search",
        tool_call_id="call_tool",
        status="success",
        directive="continue",
        error_code=None,
        execution_started=True,
        duration_ms=7,
        tool_metadata=tool_metadata,
    )

    for event in (planned, started, completed):
        restored = event_from_dict(event.to_dict())
        assert restored.to_dict() == event.to_dict()
        assert restored.to_dict()["tool_metadata"] == tool_metadata.to_dict()

    assert completed.to_dict()["error_code"] is None
    assert completed.to_dict()["execution_started"] is True
    assert completed.to_dict()["duration_ms"] == 7


def test_legacy_tool_completion_does_not_fabricate_additive_fields() -> None:
    legacy_payload = {
        "version": "v1",
        "type": "tool_call_completed",
        "event_id": "evt_legacy_tool",
        "run_id": "run_legacy_tool",
        "trace_id": "trace_legacy_tool",
        "created_at": 99.0,
        "tool_name": "lookup",
        "tool_call_id": "call_legacy",
        "status": "success",
    }

    event = event_from_dict(legacy_payload)

    assert isinstance(event, ToolCallCompletedEvent)
    assert event.directive is None
    assert event.error_code is None
    assert event.execution_started is None
    assert event.duration_ms is None
    assert all(
        not event.has_additive_field(field_name)
        for field_name in ("directive", "error_code", "execution_started", "duration_ms")
    )
    assert event.to_dict() == legacy_payload


def test_generic_event_metadata_does_not_fabricate_typed_tool_metadata() -> None:
    event = ToolCallStartedEvent(
        run_id="run_generic_metadata",
        trace_id="trace_generic_metadata",
        tool_name="lookup",
        tool_call_id="call_generic_metadata",
        arguments={},
        metadata={"tool_metadata": {"side_effect": "write"}},
    )

    assert event.tool_metadata is None
    assert "tool_metadata" not in event.to_dict()
    assert event.metadata["tool_metadata"] == {"side_effect": "write"}


def test_runtime_log_projection_preserves_typed_tool_lifecycle_fields() -> None:
    typed_metadata = {
        "side_effect": "read",
        "idempotency": "supported",
        "terminal": False,
        "capability_tags": ["source.inspect"],
        "cost_dimensions": ["workspace.bytes_read"],
    }
    common = {
        "run_id": "run_runtime_tool",
        "trace_id": "trace_runtime_tool",
        "agent_name": "assistant",
        "user_input": "inspect",
    }
    planned = event_from_runtime_log(
        "tool_call_planned",
        {
            "cycle": 1,
            "tool_name": "inspect",
            "tool_call_id": "call_runtime_tool",
            "arguments": {"path": "README.md"},
            "tool_metadata": typed_metadata,
        },
        **common,
    )
    completed = event_from_runtime_log(
        "tool_result",
        {
            "cycle": 1,
            "tool_name": "inspect",
            "tool_call_id": "call_runtime_tool",
            "status": "success",
            "directive": "continue",
            "error_code": None,
            "execution_started": True,
            "duration_ms": 7,
            "tool_metadata": typed_metadata,
        },
        **common,
    )

    assert isinstance(planned, ToolCallPlannedEvent)
    assert planned.to_dict()["tool_metadata"] == typed_metadata
    assert isinstance(completed, ToolCallCompletedEvent)
    assert completed.to_dict()["directive"] == "continue"
    assert completed.to_dict()["error_code"] is None
    assert completed.to_dict()["execution_started"] is True
    assert completed.to_dict()["duration_ms"] == 7
    assert completed.to_dict()["tool_metadata"] == typed_metadata


def test_approval_resolved_action_is_canonical_and_preserves_approved_compatibility() -> None:
    expected_approved = {
        "allow": True,
        "allow_session": True,
        "deny": False,
        "timeout": False,
    }

    for action, approved in expected_approved.items():
        event = ApprovalResolvedEvent(
            run_id="run_approval",
            trace_id="trace_approval",
            request_id="request_approval",
            tool_name="shell",
            tool_call_id="call_approval",
            action=action,
        )
        payload = event.to_dict()
        restored = event_from_dict(payload)

        assert event.action == action
        assert event.approved is approved
        assert payload["action"] == action
        assert payload["approved"] is approved
        assert isinstance(restored, ApprovalResolvedEvent)
        assert restored.action == action
        assert restored.approved is approved


def test_tool_event_can_point_to_parent_event_and_run() -> None:
    event = ToolCallStartedEvent(
        run_id="run_child",
        trace_id="trace_1",
        tool_name="browser",
        tool_call_id="call_1",
        session_id="session_1",
        parent_run_id="run_parent",
        parent_event_id="evt_parent",
        cycle_index=2,
    )

    payload = event.to_dict()

    assert payload["version"] == "v1"
    assert payload["type"] == "tool_call_started"
    assert payload["event_id"].startswith("evt_")
    assert payload["run_id"] == "run_child"
    assert payload["trace_id"] == "trace_1"
    assert payload["session_id"] == "session_1"
    assert payload["parent_run_id"] == "run_parent"
    assert payload["parent_event_id"] == "evt_parent"
    assert payload["cycle_index"] == 2
    assert payload["tool_name"] == "browser"
    assert payload["tool_call_id"] == "call_1"


def test_concrete_event_constructors_can_preserve_replayed_identity_and_timing() -> None:
    events = [
        RunStartedEvent(run_id="run_replay", trace_id="trace_replay", input="hello", **_replay_fields()),
        AgentStartedEvent(run_id="run_replay", trace_id="trace_replay", **_replay_fields()),
        LLMStartedEvent(run_id="run_replay", trace_id="trace_replay", model="model", **_replay_fields()),
        MemoryCompactedEvent(
            run_id="run_replay",
            trace_id="trace_replay",
            before_count=2,
            after_count=1,
            **_replay_fields(),
        ),
        AssistantDeltaEvent(run_id="run_replay", trace_id="trace_replay", delta="hi", **_replay_fields()),
        ToolCallStartedEvent(
            run_id="run_replay",
            trace_id="trace_replay",
            tool_name="browser",
            tool_call_id="call_replay",
            **_replay_fields(),
        ),
        ToolCallPlannedEvent(
            run_id="run_replay",
            trace_id="trace_replay",
            tool_name="browser",
            tool_call_id="call_replay",
            **_replay_fields(),
        ),
        ToolCallCompletedEvent(
            run_id="run_replay",
            trace_id="trace_replay",
            tool_name="browser",
            tool_call_id="call_replay",
            status="completed",
            **_replay_fields(),
        ),
        ApprovalRequestedEvent(
            run_id="run_replay",
            trace_id="trace_replay",
            tool_name="browser",
            tool_call_id="call_replay",
            message="approve",
            **_replay_fields(),
        ),
        ApprovalResolvedEvent(
            run_id="run_replay",
            trace_id="trace_replay",
            tool_name="browser",
            tool_call_id="call_replay",
            approved=True,
            **_replay_fields(),
        ),
        HandoffEvent(
            run_id="run_replay",
            trace_id="trace_replay",
            source_agent="assistant",
            target_agent="researcher",
            tool_call_id="call_replay",
            **_replay_fields(),
        ),
        RunCompletedEvent(
            run_id="run_replay",
            trace_id="trace_replay",
            final_output="done",
            status="completed",
            **_replay_fields(),
        ),
        RunFailedEvent(run_id="run_replay", trace_id="trace_replay", error="boom", **_replay_fields()),
    ]

    for event in events:
        payload = event.to_dict()
        assert payload["version"] == "v1"
        assert payload["event_id"] == "evt_replayed"
        assert payload["created_at"] == 123.45
        assert payload["session_id"] == "session_replay"
        assert payload["parent_event_id"] == "evt_parent"
        assert payload["parent_run_id"] == "run_parent"


class ReplayFields(TypedDict):
    event_id: str
    created_at: float
    session_id: str
    parent_event_id: str
    parent_run_id: str


def _replay_fields() -> ReplayFields:
    return {
        "event_id": "evt_replayed",
        "created_at": 123.45,
        "session_id": "session_replay",
        "parent_event_id": "evt_parent",
        "parent_run_id": "run_parent",
    }


def test_legacy_event_aliases_point_to_canonical_classes() -> None:
    assert ToolStartedEvent is ToolCallStartedEvent
    assert ToolFinishedEvent is ToolCallCompletedEvent
    assert ToolApprovalRequestedEvent is ApprovalRequestedEvent


def test_base_run_event_is_public() -> None:
    assert RunEvent.__name__ == "RunEvent"
    assert ReasoningDeltaEvent.__name__ == "ReasoningDeltaEvent"
    assert ModelToolCallStartedEvent.__name__ == "ModelToolCallStartedEvent"
    assert ModelToolCallProgressEvent.__name__ == "ModelToolCallProgressEvent"
    assert ToolCallPlannedEvent.__name__ == "ToolCallPlannedEvent"
    assert ToolCallStartedEvent.__name__ == "ToolCallStartedEvent"
    assert ToolCallCompletedEvent.__name__ == "ToolCallCompletedEvent"
    assert ApprovalRequestedEvent.__name__ == "ApprovalRequestedEvent"
    assert SubRunStartedEvent.__name__ == "SubRunStartedEvent"
    assert SubRunCompletedEvent.__name__ == "SubRunCompletedEvent"
    assert HandoffStartedEvent.__name__ == "HandoffStartedEvent"
    assert HandoffCompletedEvent.__name__ == "HandoffCompletedEvent"
    assert CycleStartedEvent.__name__ == "CycleStartedEvent"
    assert SessionPersistedEvent.__name__ == "SessionPersistedEvent"
    assert RunStateChangedEvent.__name__ == "RunStateChangedEvent"
    assert RunCancelledEvent.__name__ == "RunCancelledEvent"


def test_typed_stream_wire_requires_positive_cycle_index() -> None:
    payload = json.loads(PARITY_FIXTURE.read_text(encoding="ascii").splitlines()[4])
    payload.pop("cycle_index")

    with pytest.raises(ValueError, match="positive cycle_index"):
        event_from_dict(payload)


def test_run_events_v1_parity_fixture_has_stable_bytes_and_round_trips_compact_wire() -> None:
    fixture_bytes = PARITY_FIXTURE.read_bytes()
    assert hashlib.sha256(fixture_bytes).hexdigest() == PARITY_FIXTURE_SHA256

    lines = fixture_bytes.decode("ascii").splitlines()
    events = [event_from_dict(json.loads(line)) for line in lines]

    assert [event.type for event in events] == PARITY_EVENT_TYPES
    for index, (line, event) in enumerate(zip(lines, events, strict=True)):
        if event.run_id == "run_parity":
            assert event.event_id == "evt_parity"
            assert event.run_id == "run_parity"
            assert event.trace_id == "trace_parity"
            assert event.created_at == 123.456789
        if index < len(PARITY_EVENT_TYPES) - 7:
            assert json.dumps(event.to_dict(), separators=(",", ":")) == line
        else:
            assert event.to_dict() == json.loads(line)


def test_budget_events_v1_fixture_has_stable_bytes_and_round_trips() -> None:
    fixture_bytes = BUDGET_FIXTURE.read_bytes()
    assert hashlib.sha256(fixture_bytes).hexdigest() == BUDGET_FIXTURE_SHA256

    lines = fixture_bytes.decode("ascii").splitlines()
    events = [event_from_dict(json.loads(line)) for line in lines]

    assert [event.type for event in events] == [
        "budget_snapshot",
        "budget_exhausted",
        "run_failed",
        "run_completed",
    ]
    for line, event in zip(lines, events, strict=True):
        assert event.to_dict() == json.loads(line)


def test_run_event_v1_omits_none_and_empty_metadata() -> None:
    without_metadata = RunStartedEvent(
        run_id="run_compact",
        trace_id="trace_compact",
        input="hello",
        event_id="evt_compact",
        created_at=123.456789,
    ).to_dict()
    with_empty_metadata = RunStartedEvent(
        run_id="run_compact",
        trace_id="trace_compact",
        input="hello",
        event_id="evt_compact",
        created_at=123.456789,
        metadata={},
    ).to_dict()

    assert without_metadata == with_empty_metadata
    assert "metadata" not in without_metadata
    assert "session_id" not in without_metadata
    assert "cycle_index" not in without_metadata


def test_approval_preview_is_accepted_but_message_is_canonical() -> None:
    payload = next(
        json.loads(line)
        for line in PARITY_FIXTURE.read_text(encoding="ascii").splitlines()
        if json.loads(line)["type"] == "approval_requested"
    )
    payload["preview"] = payload.pop("message")

    event = event_from_dict(payload)
    encoded = event.to_dict()

    assert isinstance(event, ApprovalRequestedEvent)
    assert event.message == "Allow command?"
    assert encoded["message"] == "Allow command?"
    assert "preview" not in encoded
