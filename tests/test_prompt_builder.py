from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from vv_agent.constants import (
    ASK_USER_TOOL_NAME,
    CREATE_SUB_TASK_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    SUB_TASK_STATUS_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
)
from vv_agent.prompt import PromptSection, SystemPromptBuilder, build_system_prompt, build_system_prompt_bundle


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


def test_build_system_prompt_can_include_session_memory_context() -> None:
    prompt = build_system_prompt(
        "You are a test agent.",
        language="en-US",
        session_memory_context="<Session Memory>\n## key_fact\n- prior decision\n</Session Memory>",
    )

    assert "<Session Memory>" in prompt
    assert prompt.index("</Agent Definition>") < prompt.index("<Session Memory>")
    assert prompt.index("<Session Memory>") < prompt.index("<Tools>")


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
    assert "工作区环境中" in prompt


def test_prompt_can_include_sub_agent_guidance() -> None:
    prompt = build_system_prompt(
        "Agent",
        language="zh-CN",
        available_sub_agents={"research-sub": "负责资料检索", "writer-sub": "负责初稿撰写"},
    )

    assert CREATE_SUB_TASK_TOOL_NAME in prompt
    assert SUB_TASK_STATUS_TOOL_NAME in prompt
    assert "agent_id=`research-sub`" in prompt
    assert "agent_id=`writer-sub`" in prompt


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


def test_build_system_prompt_bundle_includes_cacheable_sections() -> None:
    bundle = build_system_prompt_bundle(
        "Agent",
        language="en-US",
        current_time_utc=datetime(2026, 2, 17, 15, 0, 0, tzinfo=UTC),
        session_memory_context="<Session Memory>\n- note\n</Session Memory>",
    )

    assert bundle.prompt.startswith("<Agent Definition>")
    assert [section["id"] for section in bundle.sections] == [
        "agent_definition",
        "session_memory",
        "tools",
        "current_time",
    ]
    assert bundle.sections[0]["stable"] is True
    assert bundle.sections[1]["stable"] is False
    assert bundle.sections[-1]["stable"] is False
    assert bundle.stable_hash


def test_system_prompt_builder_caches_stable_sections_only() -> None:
    counters = {"stable": 0, "volatile": 0}

    def build_stable() -> str:
        counters["stable"] += 1
        return "stable"

    def build_volatile() -> str:
        counters["volatile"] += 1
        return f"volatile-{counters['volatile']}"

    builder = SystemPromptBuilder()
    builder.add_section(PromptSection(id="stable", compute=build_stable, stable=True))
    builder.add_section(PromptSection(id="volatile", compute=build_volatile, stable=False))

    assert builder.build() == "stable\n\nvolatile-1"
    assert builder.build() == "stable\n\nvolatile-2"
    assert counters == {"stable": 1, "volatile": 2}


def test_system_prompt_builder_stable_hash_matches_build_result_hash() -> None:
    builder = SystemPromptBuilder()
    builder.add_section(PromptSection(id="stable", compute=lambda: "  stable section  ", stable=True))
    builder.add_section(PromptSection(id="volatile", compute=lambda: " volatile ", stable=False))

    assert builder.stable_hash() == builder.build_result().stable_hash
