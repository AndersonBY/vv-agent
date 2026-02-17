from __future__ import annotations

import json

from v_agent.types import CycleStatus, ToolExecutionResult, ToolResultStatus


def test_tool_result_keeps_tool_message_shape() -> None:
    result = ToolExecutionResult(tool_call_id="call_1", status="success", content="ok")
    msg = result.to_tool_message()
    assert msg.role == "tool"
    assert msg.tool_call_id == "call_1"
    assert msg.content == "ok"


def test_tool_result_status_compatibility_mapping() -> None:
    legacy = ToolExecutionResult(tool_call_id="c1", status="error", content="bad")
    assert legacy.status == "error"
    assert legacy.status_code == ToolResultStatus.ERROR

    protocol = ToolExecutionResult(tool_call_id="c2", status_code=ToolResultStatus.RUNNING, content="running")
    assert protocol.status_code == ToolResultStatus.RUNNING
    assert protocol.status == "success"


def test_protocol_enums_are_json_serializable() -> None:
    payload = {
        "tool_status": ToolResultStatus.PENDING_COMPRESS,
        "cycle_status": CycleStatus.WAIT_RESPONSE,
    }
    encoded = json.dumps(payload)
    assert "\"PENDING_COMPRESS\"" in encoded
    assert "\"wait_response\"" in encoded
