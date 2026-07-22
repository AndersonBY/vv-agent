from __future__ import annotations

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
    HandoffStartedEvent,
    LLMStartedEvent,
    MemoryCompactStarted,
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
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    event_from_dict,
)
from vv_agent.events import ToolCallPlannedEvent
from vv_agent.tools.metadata import ToolMetadata

PARITY_FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "run_events.jsonl"
BUDGET_FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "budget_events.jsonl"
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
    "handoff_started",
    "handoff_completed",
    "session_persisted",
    "run_state_changed",
    "diagnostic",
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


def test_run_event_has_stable_identity_and_timing() -> None:
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


def test_event_from_dict_rejects_superseded_created_at_milliseconds() -> None:
    with pytest.raises(ValueError, match="unknown fields: created_at_ms"):
        event_from_dict(
            {
                "version": "v1",
                "type": "run_started",
                "event_id": "evt_old_time",
                "run_id": "run_old_time",
                "trace_id": "trace_old_time",
                "created_at_ms": 123456.789,
                "input": "hello",
            }
        )


def test_tool_completion_status_requires_current_exact_value() -> None:
    event = ToolCallCompletedEvent(
        run_id="run_status",
        trace_id="trace_status",
        tool_name="lookup",
        tool_call_id="call_status",
        status="wait_response",
        directive="continue",
        error_code=None,
        execution_started=True,
        duration_ms=1,
    )

    assert event.status == "wait_response"
    assert event.to_dict()["status"] == "wait_response"
    with pytest.raises(ValueError, match="Unsupported tool event status"):
        ToolCallCompletedEvent(
            run_id="run_status",
            trace_id="trace_status",
            tool_name="lookup",
            tool_call_id="call_status",
            status="WAIT_RESPONSE",
            directive="continue",
            error_code=None,
            execution_started=True,
            duration_ms=1,
        )


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


def test_tool_completion_rejects_missing_current_fields() -> None:
    incomplete_payload = {
        "version": "v1",
        "type": "tool_call_completed",
        "event_id": "evt_incomplete_tool",
        "run_id": "run_incomplete_tool",
        "trace_id": "trace_incomplete_tool",
        "created_at": 99.0,
        "tool_name": "lookup",
        "tool_call_id": "call_incomplete",
        "status": "success",
    }

    with pytest.raises(ValueError, match="missing required fields"):
        event_from_dict(incomplete_payload)


def test_memory_compaction_rejects_missing_current_fields() -> None:
    incomplete_started = {
        "version": "v1",
        "type": "memory_compact_started",
        "event_id": "evt_incomplete_memory_started",
        "run_id": "run_incomplete_memory",
        "trace_id": "trace_incomplete_memory",
        "created_at": 99.0,
        "message_count": 4,
        "estimated_tokens": 120,
    }
    incomplete_completed = {
        "version": "v1",
        "type": "memory_compact_completed",
        "event_id": "evt_incomplete_memory_completed",
        "run_id": "run_incomplete_memory",
        "trace_id": "trace_incomplete_memory",
        "created_at": 100.0,
        "before_count": 4,
        "after_count": 2,
        "summary_tokens": 20,
    }

    with pytest.raises(ValueError, match="missing required fields"):
        event_from_dict(incomplete_started)
    with pytest.raises(ValueError, match="missing required fields"):
        event_from_dict(incomplete_completed)


def test_memory_compact_started_accepts_explicit_null_model_output_capability() -> None:
    payload = {
        "version": "v1",
        "type": "memory_compact_started",
        "event_id": "evt_nullable_memory_capability",
        "run_id": "run_nullable_memory_capability",
        "trace_id": "trace_nullable_memory_capability",
        "created_at": 101.0,
        "message_count": 4,
        "trigger": "full_threshold",
        "configured_threshold": 250_000,
        "effective_threshold": 250_000,
        "microcompact_threshold": 187_500,
        "model_context_window": 1_000_000,
        "model_max_output_tokens": None,
        "reserved_output_tokens": 16_000,
        "reserved_output_source": "framework_fallback",
        "autocompact_buffer_tokens": 13_000,
    }

    event = event_from_dict(payload)

    assert isinstance(event, MemoryCompactStarted)
    assert event.model_max_output_tokens is None
    assert event.to_dict() == payload


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


def test_approval_resolved_action_is_the_only_wire_decision() -> None:
    for action in ("allow", "allow_session", "deny", "timeout"):
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
        assert payload["action"] == action
        assert "approved" not in payload
        assert isinstance(restored, ApprovalResolvedEvent)
        assert restored.action == action


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
            status="success",
            directive="continue",
            error_code=None,
            execution_started=True,
            duration_ms=1,
            **_replay_fields(),
        ),
        ApprovalRequestedEvent(
            run_id="run_replay",
            trace_id="trace_replay",
            tool_name="browser",
            tool_call_id="call_replay",
            message="approve",
            request_id="request_replay",
            **_replay_fields(),
        ),
        ApprovalResolvedEvent(
            run_id="run_replay",
            trace_id="trace_replay",
            tool_name="browser",
            tool_call_id="call_replay",
            action="allow",
            request_id="request_replay",
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

    with pytest.raises(ValueError, match="missing required fields: cycle_index"):
        event_from_dict(payload)


def test_run_events_parity_fixture_round_trips_current_wire() -> None:
    fixture_bytes = PARITY_FIXTURE.read_bytes()

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


def test_budget_events_fixture_round_trips() -> None:
    fixture_bytes = BUDGET_FIXTURE.read_bytes()

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


def test_run_event_omits_none_and_empty_metadata() -> None:
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


def test_approval_preview_field_is_rejected() -> None:
    payload = next(
        json.loads(line)
        for line in PARITY_FIXTURE.read_text(encoding="ascii").splitlines()
        if json.loads(line)["type"] == "approval_requested"
    )
    payload["preview"] = payload.pop("message")

    with pytest.raises(ValueError, match="unknown fields: preview"):
        event_from_dict(payload)
