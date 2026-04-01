from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vv_agent.prompt.templates import (
    ASK_USER_PROMPT,
    COMPUTER_AGENT_ENV_PROMPT,
    CURRENT_TIME_PROMPT,
    TASK_FINISH_PROMPT,
    TODO_PROMPT,
    TOOL_PRIORITY_PROMPT,
    render_available_skills,
    render_sub_agents,
    render_workspace_tools,
)


@dataclass(slots=True)
class PromptSection:
    """A lazily rendered system prompt section with optional local memoization."""

    id: str
    compute: Callable[[], str]
    stable: bool = True
    _cached_value: str | None = field(default=None, init=False, repr=False)
    _cache_valid: bool = field(default=False, init=False, repr=False)

    def get_value(self) -> str:
        if self.stable and self._cache_valid:
            return self._cached_value or ""
        value = str(self.compute() or "")
        if self.stable:
            self._cached_value = value
            self._cache_valid = True
        return value

    def invalidate(self) -> None:
        self._cached_value = None
        self._cache_valid = False

    def to_metadata(self) -> dict[str, Any] | None:
        text = self.get_value().strip()
        if not text:
            return None
        return {"id": self.id, "text": text, "stable": self.stable}


@dataclass(slots=True)
class BuiltSystemPrompt:
    prompt: str
    sections: list[dict[str, Any]]
    stable_hash: str


@dataclass(slots=True)
class SystemPromptBuilder:
    """Builds a system prompt and companion section metadata for Anthropic prompt cache."""

    _sections: list[PromptSection] = field(default_factory=list)

    def add_section(self, section: PromptSection) -> None:
        self._sections.append(section)

    def build(self) -> str:
        parts = [section.get_value().strip() for section in self._sections]
        return "\n\n".join(part for part in parts if part)

    def metadata_sections(self) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        for section in self._sections:
            payload = section.to_metadata()
            if payload is not None:
                sections.append(payload)
        return sections

    def invalidate_all(self) -> None:
        for section in self._sections:
            section.invalidate()

    def invalidate_volatile(self) -> None:
        for section in self._sections:
            if not section.stable:
                section.invalidate()

    def stable_hash(self) -> str:
        stable_text = "".join(section.get_value().strip() for section in self._sections if section.stable).encode(
            "utf-8"
        )
        return hashlib.sha256(stable_text).hexdigest()

    def build_result(self) -> BuiltSystemPrompt:
        prompt_parts: list[str] = []
        sections: list[dict[str, Any]] = []
        stable_parts: list[str] = []
        for section in self._sections:
            value = section.get_value().strip()
            if not value:
                continue
            prompt_parts.append(value)
            sections.append({"id": section.id, "text": value, "stable": section.stable})
            if section.stable:
                stable_parts.append(value)
        return BuiltSystemPrompt(
            prompt="\n\n".join(prompt_parts),
            sections=sections,
            stable_hash=hashlib.sha256("".join(stable_parts).encode("utf-8")).hexdigest(),
        )


def build_system_prompt(
    original_system_prompt: str,
    *,
    language: str = "en-US",
    allow_interruption: bool = True,
    use_workspace: bool = True,
    enable_todo_management: bool = True,
    agent_type: str | None = None,
    available_sub_agents: dict[str, str] | None = None,
    available_skills: list[dict[str, Any] | str] | None = None,
    workspace: str | Path | None = None,
    current_time_utc: datetime | None = None,
    session_memory_context: str = "",
) -> str:
    return build_system_prompt_bundle(
        original_system_prompt,
        language=language,
        allow_interruption=allow_interruption,
        use_workspace=use_workspace,
        enable_todo_management=enable_todo_management,
        agent_type=agent_type,
        available_sub_agents=available_sub_agents,
        available_skills=available_skills,
        workspace=workspace,
        current_time_utc=current_time_utc,
        session_memory_context=session_memory_context,
    ).prompt


def build_system_prompt_sections(
    original_system_prompt: str,
    *,
    language: str = "en-US",
    allow_interruption: bool = True,
    use_workspace: bool = True,
    enable_todo_management: bool = True,
    agent_type: str | None = None,
    available_sub_agents: dict[str, str] | None = None,
    available_skills: list[dict[str, Any] | str] | None = None,
    workspace: str | Path | None = None,
    current_time_utc: datetime | None = None,
    session_memory_context: str = "",
) -> list[dict[str, Any]]:
    return build_system_prompt_bundle(
        original_system_prompt,
        language=language,
        allow_interruption=allow_interruption,
        use_workspace=use_workspace,
        enable_todo_management=enable_todo_management,
        agent_type=agent_type,
        available_sub_agents=available_sub_agents,
        available_skills=available_skills,
        workspace=workspace,
        current_time_utc=current_time_utc,
        session_memory_context=session_memory_context,
    ).sections


def build_system_prompt_bundle(
    original_system_prompt: str,
    *,
    language: str = "en-US",
    allow_interruption: bool = True,
    use_workspace: bool = True,
    enable_todo_management: bool = True,
    agent_type: str | None = None,
    available_sub_agents: dict[str, str] | None = None,
    available_skills: list[dict[str, Any] | str] | None = None,
    workspace: str | Path | None = None,
    current_time_utc: datetime | None = None,
    session_memory_context: str = "",
) -> BuiltSystemPrompt:
    return create_system_prompt_builder(
        original_system_prompt,
        language=language,
        allow_interruption=allow_interruption,
        use_workspace=use_workspace,
        enable_todo_management=enable_todo_management,
        agent_type=agent_type,
        available_sub_agents=available_sub_agents,
        available_skills=available_skills,
        workspace=workspace,
        current_time_utc=current_time_utc,
        session_memory_context=session_memory_context,
    ).build_result()


def create_system_prompt_builder(
    original_system_prompt: str,
    *,
    language: str = "en-US",
    allow_interruption: bool = True,
    use_workspace: bool = True,
    enable_todo_management: bool = True,
    agent_type: str | None = None,
    available_sub_agents: dict[str, str] | None = None,
    available_skills: list[dict[str, Any] | str] | None = None,
    workspace: str | Path | None = None,
    current_time_utc: datetime | None = None,
    session_memory_context: str = "",
) -> SystemPromptBuilder:
    builder = SystemPromptBuilder()
    builder.add_section(
        PromptSection(
            id="agent_definition",
            compute=lambda prompt=original_system_prompt: f"<Agent Definition>\n{prompt}\n</Agent Definition>",
            stable=True,
        )
    )

    if session_memory_context:
        builder.add_section(
            PromptSection(
                id="session_memory",
                compute=lambda context=session_memory_context: context,
                stable=False,
            )
        )

    if agent_type == "computer":
        environment_text = COMPUTER_AGENT_ENV_PROMPT.get(language, COMPUTER_AGENT_ENV_PROMPT["en-US"])
        builder.add_section(
            PromptSection(
                id="environment",
                compute=lambda text=environment_text: f"<Environment>\n{text}\n</Environment>",
                stable=True,
            )
        )

    tools_lines: list[str] = []
    if allow_interruption:
        tools_lines.append(ASK_USER_PROMPT.get(language, ASK_USER_PROMPT["en-US"]))
    if use_workspace:
        tools_lines.append(render_workspace_tools(language))
        tools_lines.append(TOOL_PRIORITY_PROMPT.get(language, TOOL_PRIORITY_PROMPT["en-US"]))
    if enable_todo_management:
        tools_lines.append(TODO_PROMPT.get(language, TODO_PROMPT["en-US"]))
    if available_sub_agents:
        tools_lines.append(render_sub_agents(language, available_sub_agents))
    if available_skills:
        workspace_path = Path(workspace).resolve() if workspace is not None else None
        tools_lines.append(render_available_skills(language, available_skills, workspace=workspace_path))
    tools_lines.append(TASK_FINISH_PROMPT.get(language, TASK_FINISH_PROMPT["en-US"]))
    joined_tools = "\n\n".join(tools_lines)
    builder.add_section(
        PromptSection(
            id="tools",
            compute=lambda text=joined_tools: f"<Tools>\n{text}\n</Tools>",
            stable=True,
        )
    )

    def _render_time_section() -> str:
        now = current_time_utc or datetime.now(tz=UTC)
        now_text = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        time_header = CURRENT_TIME_PROMPT.get(language, CURRENT_TIME_PROMPT["en-US"])
        return f"<Current Time>\n{time_header}\n{now_text}\n</Current Time>"

    builder.add_section(
        PromptSection(
            id="current_time",
            compute=_render_time_section,
            stable=False,
        )
    )

    return builder


def build_raw_system_prompt_sections(system_prompt: str) -> list[dict[str, Any]]:
    text = str(system_prompt or "").strip()
    if not text:
        return []
    return [{"id": "raw_system_prompt", "text": text, "stable": True}]
