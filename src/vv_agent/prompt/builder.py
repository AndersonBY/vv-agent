from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vv_agent.kernel._prompt_values import PromptBundle as PromptBundle
from vv_agent.kernel._prompt_values import PromptSection as PromptSection
from vv_agent.kernel._prompt_values import _trim_text
from vv_agent.prompt.templates import (
    ASK_USER_PROMPT,
    COMPUTER_AGENT_ENV_PROMPT,
    CURRENT_TIME_PROMPT,
    TODO_PROMPT,
    render_available_skills,
    render_sub_agents,
    render_workspace_tools,
)


def inject_session_memory_section(bundle: PromptBundle, session_memory_context: str) -> PromptBundle:
    """Freeze loaded Session Memory into a new run's resolved prompt bundle."""

    context = _trim_text(session_memory_context)
    if not context or any(section.id == "session_memory" for section in bundle.sections):
        return bundle

    section = PromptSection(
        id="session_memory",
        text=context,
        stable=False,
        source="session.memory",
    )
    sections = list(bundle.sections)
    insertion_index = next((index for index, item in enumerate(sections) if item.id == "current_time"), len(sections))
    sections.insert(insertion_index, section)
    return PromptBundle(sections=tuple(sections))


@dataclass(slots=True)
class SystemPromptBuilder:
    """Build a resolved canonical prompt bundle."""

    _sections: list[PromptSection] = field(default_factory=list)

    def add_section(self, section: PromptSection) -> None:
        if not isinstance(section, PromptSection):
            raise TypeError("SystemPromptBuilder accepts PromptSection values")
        self._sections.append(section)

    def build(self) -> str:
        return self.build_result().flatten()

    def stable_hash(self) -> str:
        return self.build_result().stable_hash

    def build_result(self) -> PromptBundle:
        return PromptBundle(sections=tuple(self._sections))


def build_system_prompt(
    original_system_prompt: str,
    **kwargs: Any,
) -> str:
    return build_system_prompt_bundle(original_system_prompt, **kwargs).flatten()


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
    session_memory_enabled: bool = False,
    session_memory_context: str = "",
) -> PromptBundle:
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
        session_memory_enabled=session_memory_enabled,
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
    session_memory_enabled: bool = False,
    session_memory_context: str = "",
) -> SystemPromptBuilder:
    builder = SystemPromptBuilder()
    builder.add_section(
        PromptSection(
            id="agent_definition",
            text=f"<Agent Definition>\n{original_system_prompt}\n</Agent Definition>",
            stable=True,
            source="agent.instructions",
        )
    )

    if agent_type == "computer":
        environment_text = COMPUTER_AGENT_ENV_PROMPT.get(language, COMPUTER_AGENT_ENV_PROMPT["en-US"])
        builder.add_section(
            PromptSection(
                id="environment",
                text=f"<Environment>\n{environment_text}\n</Environment>",
                stable=True,
                source="runtime.environment",
            )
        )

    tool_lines: list[str] = []
    if allow_interruption:
        tool_lines.append(ASK_USER_PROMPT.get(language, ASK_USER_PROMPT["en-US"]))
    if use_workspace:
        tool_lines.append(render_workspace_tools(language))
    if enable_todo_management:
        tool_lines.append(TODO_PROMPT.get(language, TODO_PROMPT["en-US"]))
    if available_sub_agents:
        tool_lines.append(render_sub_agents(language, available_sub_agents))
    if available_skills:
        workspace_path = Path(workspace).resolve() if workspace is not None else None
        skills_prompt = render_available_skills(language, available_skills, workspace=workspace_path)
        if skills_prompt:
            tool_lines.append(skills_prompt)
    if tool_lines:
        joined_tool_lines = "\n\n".join(tool_lines)
        builder.add_section(
            PromptSection(
                id="tools",
                text=f"<Tools>\n{joined_tool_lines}\n</Tools>",
                stable=True,
                source="runtime.tools",
            )
        )

    if session_memory_enabled and session_memory_context:
        builder.add_section(
            PromptSection(
                id="session_memory",
                text=session_memory_context,
                stable=False,
                source="session.memory",
            )
        )

    task_start_time = current_time_utc or datetime.now(tz=UTC)
    task_start_time_text = task_start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    time_header = CURRENT_TIME_PROMPT.get(language, CURRENT_TIME_PROMPT["en-US"])
    builder.add_section(
        PromptSection(
            id="current_time",
            text=f"<Current Time>\n{time_header}\n{task_start_time_text}\n</Current Time>",
            stable=False,
            source="run.clock",
        )
    )
    return builder


def build_raw_system_prompt_bundle(system_prompt: str) -> PromptBundle:
    return PromptBundle(
        sections=(
            PromptSection(
                id="agent_instructions",
                text=system_prompt,
                stable=True,
                source="agent.instructions",
            ),
        )
    )
