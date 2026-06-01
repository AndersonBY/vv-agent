from __future__ import annotations

from typing import TypedDict

from vv_agent import (
    AgentStartedEvent,
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    AssistantDeltaEvent,
    HandoffCompletedEvent,
    HandoffEvent,
    HandoffStartedEvent,
    LLMStartedEvent,
    MemoryCompactedEvent,
    RunCompletedEvent,
    RunEvent,
    RunFailedEvent,
    RunStartedEvent,
    SubRunCompletedEvent,
    SubRunStartedEvent,
    ToolApprovalRequestedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
)


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
    assert ToolCallStartedEvent.__name__ == "ToolCallStartedEvent"
    assert ToolCallCompletedEvent.__name__ == "ToolCallCompletedEvent"
    assert ApprovalRequestedEvent.__name__ == "ApprovalRequestedEvent"
    assert SubRunStartedEvent.__name__ == "SubRunStartedEvent"
    assert SubRunCompletedEvent.__name__ == "SubRunCompletedEvent"
    assert HandoffStartedEvent.__name__ == "HandoffStartedEvent"
    assert HandoffCompletedEvent.__name__ == "HandoffCompletedEvent"
