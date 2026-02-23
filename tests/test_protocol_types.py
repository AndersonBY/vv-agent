from __future__ import annotations

import json

from vv_agent.types import AgentTask, CycleStatus, Message, SubAgentConfig, ToolExecutionResult, ToolResultStatus


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


def test_assistant_message_keeps_tool_calls_in_openai_payload() -> None:
    message = Message(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "todo_read",
                    "arguments": "{}",
                },
            }
        ],
        reasoning_content="analysis",
    )
    payload = message.to_openai_message()
    assert "tool_calls" in payload
    assert payload["content"] is None
    assert payload["reasoning_content"] == "analysis"
    assert payload["tool_calls"][0]["function"]["name"] == "todo_read"


def test_assistant_message_can_skip_reasoning_content() -> None:
    message = Message(role="assistant", content="answer", reasoning_content="analysis")
    payload = message.to_openai_message(include_reasoning_content=False)
    assert "reasoning_content" not in payload


def test_user_message_with_image_url_uses_multimodal_content() -> None:
    message = Message(
        role="user",
        content="Please inspect this image",
        image_url="https://example.com/demo.png",
    )
    payload = message.to_openai_message()
    assert isinstance(payload["content"], list)
    assert payload["content"][0]["type"] == "text"
    assert payload["content"][1]["type"] == "image_url"
    assert payload["content"][1]["image_url"]["url"] == "https://example.com/demo.png"


def test_agent_task_sub_agent_config_support() -> None:
    task = AgentTask(
        task_id="task_sub",
        model="m",
        system_prompt="sys",
        user_prompt="u",
        sub_agents={
            "research": SubAgentConfig(model="kimi-k2.5", description="collect data"),
        },
    )
    assert task.sub_agents_enabled is True
    assert task.sub_agents["research"].model == "kimi-k2.5"
