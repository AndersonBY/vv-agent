from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from vv_agent.prompt import build_raw_system_prompt_bundle
from vv_agent.runtime.cycle_runner import CycleRunner
from vv_agent.types import AgentTask, CycleStatus, Message, SubAgentConfig, ToolCall, ToolExecutionResult, ToolResultStatus

BOUNDED_RESULT_FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "bounded_tool_result.json"


def _set_dotted(payload: dict[str, Any], dotted: str, value: object) -> None:
    target = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        nested = target[part]
        assert isinstance(nested, dict)
        target = nested
    target[parts[-1]] = value


def _delete_dotted(payload: dict[str, Any], dotted: str) -> None:
    target = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        nested = target[part]
        assert isinstance(nested, dict)
        target = nested
    del target[parts[-1]]


def test_tool_result_keeps_tool_message_shape() -> None:
    result = ToolExecutionResult(
        tool_call_id="call_1",
        status_code=ToolResultStatus.SUCCESS,
        content="ok",
    )
    msg = result.to_tool_message()
    assert msg.role == "tool"
    assert msg.tool_call_id == "call_1"
    assert msg.content == "ok"


def test_tool_result_has_one_typed_status() -> None:
    result = ToolExecutionResult(
        tool_call_id="c2",
        status_code=ToolResultStatus.RUNNING,
        content="running",
    )

    assert result.status_code == ToolResultStatus.RUNNING
    assert "status" not in result.to_dict()


def test_tool_result_rejects_superseded_status_wire() -> None:
    with pytest.raises(ValueError, match="unknown=\\['status'\\]"):
        ToolExecutionResult.from_dict(
            {
                "tool_call_id": "c1",
                "content": "bad",
                "status": "error",
                "status_code": "ERROR",
                "directive": "continue",
            }
        )


def test_bounded_tool_result_canonical_codec_and_model_projection() -> None:
    fixture = json.loads(BOUNDED_RESULT_FIXTURE.read_text(encoding="utf-8"))

    for name, payload in fixture["canonical_results"].items():
        result = ToolExecutionResult.from_dict(payload)
        assert result.to_dict() == payload, name

    for case in fixture["tool_message_projection"]["cases"]:
        result = ToolExecutionResult.from_dict(fixture["canonical_results"][case["result_ref"]])
        assert result.to_tool_message().content == case["expected_message"], case["name"]


def test_bounded_tool_result_rejects_invalid_sparse_fixture_cases() -> None:
    fixture = json.loads(BOUNDED_RESULT_FIXTURE.read_text(encoding="utf-8"))
    runtime_only = {
        "artifact_symlink_segment",
        "artifact_collision",
        "artifact_persist_failure",
        "cursor_path_mismatch",
        "cursor_source_changed",
        "cursor_offset_past_end",
    }

    for case in fixture["invalid_cases"]:
        if case["name"] in runtime_only:
            continue
        payload = deepcopy(fixture["canonical_results"][case["base"]])
        mutation = case["mutation"]
        if "remove" in mutation:
            _delete_dotted(payload, mutation["remove"])
        for dotted, value in mutation.get("add", {}).items():
            _set_dotted(payload, dotted, deepcopy(value))
        for dotted, value in mutation.get("replace", {}).items():
            _set_dotted(payload, dotted, deepcopy(value))

        with pytest.raises(ValueError, match=case["expected_error_code"]):
            ToolExecutionResult.from_dict(payload)


def test_bounded_tool_result_optional_fields_reject_explicit_null() -> None:
    fixture = json.loads(BOUNDED_RESULT_FIXTURE.read_text(encoding="utf-8"))
    ordinary = fixture["canonical_results"]["ordinary"]

    for field_name in fixture["result_contract"]["optional_fields"]:
        with pytest.raises(ValueError, match="tool_result_invalid"):
            ToolExecutionResult.from_dict({**ordinary, field_name: None})


def test_bounded_tool_result_cursor_path_must_already_be_normalized() -> None:
    fixture = json.loads(BOUNDED_RESULT_FIXTURE.read_text(encoding="utf-8"))
    payload = deepcopy(fixture["canonical_results"]["truncated_read"])
    payload["cursor"]["path"] = "logs/./output.txt"

    with pytest.raises(ValueError, match="tool_result_invalid"):
        ToolExecutionResult.from_dict(payload)


def test_protocol_enums_are_json_serializable() -> None:
    payload = {
        "tool_status": ToolResultStatus.PENDING_COMPRESS,
        "cycle_status": CycleStatus.WAIT_RESPONSE,
    }
    encoded = json.dumps(payload)
    assert '"PENDING_COMPRESS"' in encoded
    assert '"wait_response"' in encoded


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


def test_assistant_message_preserves_tool_call_extra_content() -> None:
    message = Message(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "default_api:find_files",
                    "arguments": '{"path":"."}',
                },
                "extra_content": {
                    "google": {
                        "thought_signature": "sig_123",
                    }
                },
            }
        ],
    )
    payload = message.to_openai_message()
    assert payload["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "sig_123"


def test_cycle_runner_serializes_tool_call_extra_content() -> None:
    serialized = CycleRunner._serialize_tool_calls(
        [
            ToolCall(
                id="call_1",
                name="default_api:find_files",
                arguments={"path": "."},
                extra_content={"google": {"thought_signature": "sig_123"}},
            )
        ]
    )
    assert serialized[0]["extra_content"]["google"]["thought_signature"] == "sig_123"


def test_cycle_runner_serializes_tool_call_arguments_as_canonical_json() -> None:
    serialized = CycleRunner._serialize_tool_calls(
        [ToolCall(id="call_1", name="task_finish", arguments={"message": "done", "count": 2})]
    )

    assert serialized[0]["function"]["arguments"] == '{"message":"done","count":2}'


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
        prompt_bundle=build_raw_system_prompt_bundle("sys"),
        user_prompt="u",
        sub_agents={
            "research": SubAgentConfig(model="kimi-k2.5", description="collect data"),
        },
    )
    assert task.sub_agents_enabled is True
    assert task.sub_agents["research"].model == "kimi-k2.5"
