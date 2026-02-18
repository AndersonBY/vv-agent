from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from v_agent.constants import (
    ASK_USER_TOOL_NAME,
    BATCH_SUB_TASKS_TOOL_NAME,
    CREATE_SUB_TASK_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
)
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


def test_prompt_can_include_sub_agent_guidance() -> None:
    prompt = build_system_prompt(
        "Agent",
        language="zh-CN",
        available_sub_agents={"research-sub": "负责资料检索", "writer-sub": "负责初稿撰写"},
    )

    assert CREATE_SUB_TASK_TOOL_NAME in prompt
    assert BATCH_SUB_TASKS_TOOL_NAME in prompt
    assert "research-sub" in prompt
    assert "writer-sub" in prompt


def test_prompt_can_include_available_skills_xml(tmp_path: Path) -> None:
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: demo
description: Demo skill
---
Body
""",
        encoding="utf-8",
    )

    prompt = build_system_prompt(
        "Agent",
        language="en-US",
        available_skills=[{"location": "demo"}],
        workspace=tmp_path,
    )

    assert "<available_skills>" in prompt
    assert "<name>\ndemo\n</name>" in prompt
    assert "<description>\nDemo skill\n</description>" in prompt


def test_prompt_can_expand_skill_root_directory(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    (root / "alpha").mkdir(parents=True)
    (root / "alpha" / "SKILL.md").write_text(
        """---
name: alpha
description: Alpha skill
---
Body
""",
        encoding="utf-8",
    )

    prompt = build_system_prompt(
        "Agent",
        language="en-US",
        available_skills=["skills"],
        workspace=tmp_path,
    )

    assert "<available_skills>" in prompt
    assert "<name>\nalpha\n</name>" in prompt
