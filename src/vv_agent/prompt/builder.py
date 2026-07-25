from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vv_agent.checkpoint import canonical_json_sha256
from vv_agent.prompt.templates import (
    ASK_USER_PROMPT,
    COMPUTER_AGENT_ENV_PROMPT,
    CURRENT_TIME_PROMPT,
    TASK_FINISH_PROMPT,
    TODO_PROMPT,
    render_available_skills,
    render_sub_agents,
    render_workspace_tools,
)

_ASCII_WHITESPACE = "\t\n\v\f\r "


def _trim_text(value: str) -> str:
    return value.strip(_ASCII_WHITESPACE)


@dataclass(frozen=True, slots=True)
class PromptSection:
    """One resolved, ordered system-prompt section."""

    id: str
    text: str
    stable: bool = True
    source: str | None = None
    cache_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("prompt section id must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("prompt section text must be a string")
        text = _trim_text(self.text)
        if not text:
            raise ValueError("prompt section text cannot be empty")
        if not isinstance(self.stable, bool):
            raise TypeError("prompt section stable must be a boolean")
        for name in ("source", "cache_hint"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"prompt section {name} must be omitted or a non-empty string")
        if not isinstance(self.metadata, dict) or not all(isinstance(key, str) for key in self.metadata):
            raise TypeError("prompt section metadata must be an object with string keys")
        canonical_json_sha256(self.metadata, "prompt section metadata")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.id, "text": self.text, "stable": self.stable}
        if self.source is not None:
            payload["source"] = self.source
        if self.cache_hint is not None:
            payload["cache_hint"] = self.cache_hint
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptSection:
        if not isinstance(payload, dict):
            raise TypeError("PromptSection payload must be an object")
        required = {"id", "text", "stable"}
        allowed = {*required, "source", "cache_hint", "metadata"}
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - allowed)
        if missing or unknown:
            raise ValueError(f"PromptSection fields are invalid: missing={missing}, unknown={unknown}")
        return cls(
            id=payload["id"],
            text=payload["text"],
            stable=payload["stable"],
            source=payload.get("source"),
            cache_hint=payload.get("cache_hint"),
            metadata=payload.get("metadata", {}),
        )


def _stable_hash(sections: tuple[PromptSection, ...]) -> str:
    return canonical_json_sha256(
        [section.to_dict() for section in sections if section.stable],
        "stable prompt sections",
    )


@dataclass(frozen=True, slots=True)
class PromptBundle:
    """The canonical resolved system prompt carried through one run."""

    sections: tuple[PromptSection, ...]
    stable_hash: str = ""

    def __post_init__(self) -> None:
        sections = tuple(self.sections)
        if not sections or not all(isinstance(section, PromptSection) for section in sections):
            raise ValueError("prompt bundle sections must be a non-empty sequence of PromptSection values")
        ids = [section.id for section in sections]
        if len(ids) != len(set(ids)):
            raise ValueError("prompt bundle section ids must be unique")
        expected_hash = _stable_hash(sections)
        if self.stable_hash and self.stable_hash != expected_hash:
            raise ValueError("prompt bundle stable_hash does not match stable sections")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "stable_hash", expected_hash)

    def flatten(self) -> str:
        return "\n\n".join(section.text for section in self.sections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sections": [section.to_dict() for section in self.sections],
            "stable_hash": self.stable_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptBundle:
        if not isinstance(payload, dict):
            raise TypeError("PromptBundle payload must be an object")
        if set(payload) != {"sections", "stable_hash"}:
            raise ValueError("PromptBundle must contain exactly sections and stable_hash")
        raw_sections = payload["sections"]
        if not isinstance(raw_sections, list):
            raise TypeError("PromptBundle sections must be an array")
        stable_hash = payload["stable_hash"]
        if not isinstance(stable_hash, str):
            raise TypeError("PromptBundle stable_hash must be a string")
        return cls(
            sections=tuple(PromptSection.from_dict(section) for section in raw_sections),
            stable_hash=stable_hash,
        )


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
    tool_lines.append(TASK_FINISH_PROMPT.get(language, TASK_FINISH_PROMPT["en-US"]))
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
