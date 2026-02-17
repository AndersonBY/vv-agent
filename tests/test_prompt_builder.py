from __future__ import annotations

from datetime import UTC, datetime

from v_agent.constants import ASK_USER_TOOL_NAME, READ_FILE_TOOL_NAME, TASK_FINISH_TOOL_NAME, WRITE_FILE_TOOL_NAME
from v_agent.prompt import build_system_prompt


def test_build_system_prompt_includes_required_sections() -> None:
    prompt = build_system_prompt(
        "You are a test agent.",
        language="en-US",
        current_time_utc=datetime(2026, 2, 17, 15, 0, 0, tzinfo=UTC),
    )

    assert "<Agent Definition>" in prompt
    assert "<Tools>" in prompt
    assert "<Current Time>" in prompt
    assert "2026-02-17T15:00:00Z" in prompt


def test_prompt_includes_tool_governance_rules() -> None:
    prompt = build_system_prompt("Agent", language="en-US")

    assert ASK_USER_TOOL_NAME in prompt
    assert TASK_FINISH_TOOL_NAME in prompt
    assert READ_FILE_TOOL_NAME in prompt
    assert WRITE_FILE_TOOL_NAME in prompt
    assert "Tool priority" in prompt


def test_prompt_can_include_computer_environment() -> None:
    prompt = build_system_prompt("Agent", language="zh-CN", agent_type="computer")

    assert "<Environment>" in prompt
    assert "Linux" in prompt
