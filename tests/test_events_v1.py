from __future__ import annotations

from vv_agent import RunEvent
from vv_agent.events import RunStartedEvent, ToolCallStartedEvent


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
    assert payload["session_id"] == "session_1"
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
    assert payload["parent_run_id"] == "run_parent"
    assert payload["parent_event_id"] == "evt_parent"
    assert payload["cycle_index"] == 2
    assert payload["tool_name"] == "browser"


def test_base_run_event_is_public() -> None:
    assert RunEvent.__name__ == "RunEvent"
