"""SDK-free memory settings defaults and definition projection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from vv_agent.kernel._microcompaction_policy import MicrocompactionPolicy
from vv_agent.kernel._tool_metadata import ToolResultRetention

SummaryCallback = Callable[[str, str | None, str | None], str | None]
ReservedOutputSource = Literal[
    "model_settings",
    "task_metadata",
    "framework_fallback",
    "framework_fallback_capped_by_model_capability",
]
WORKSPACE_BACKEND_DEFAULT = None
SESSION_MEMORY_DEFAULT = None


@dataclass(slots=True)
class MemoryManagerSettings:
    compact_threshold: int = 250_000
    keep_recent_messages: int = 10
    model: str = ""
    model_context_window: int = 279_000
    model_max_output_tokens: int | None = None
    reserved_output_tokens: int = 16_000
    reserved_output_source: ReservedOutputSource = "framework_fallback"
    autocompact_buffer_tokens: int = 13_000
    language: str = "zh-CN"
    warning_threshold_percentage: int = 90
    include_memory_warning: bool = False
    tool_result_excerpt_head: int = 200
    tool_result_excerpt_tail: int = 200
    microcompaction_policy: MicrocompactionPolicy = field(default_factory=MicrocompactionPolicy)
    tool_result_retentions: dict[str, ToolResultRetention] = field(default_factory=dict)
    # The public MemoryManager subclass gives these two fields their SDK types.
    workspace_backend: Any = WORKSPACE_BACKEND_DEFAULT
    recovery_tool_available: bool = False
    artifact_scope: str | tuple[str, str] = "run"
    summary_event_limit: int = 40
    summary_backend: str | None = None
    summary_model: str | None = None
    summary_callback: SummaryCallback | None = None
    base_system_prompt: str = ""
    session_memory: Any = SESSION_MEMORY_DEFAULT
    protected_start: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.microcompaction_policy, MicrocompactionPolicy):
            self.microcompaction_policy = MicrocompactionPolicy.from_dict(self.microcompaction_policy)

    def _definition_settings(self) -> dict[str, Any]:
        memory_settings = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name
            not in {
                "protected_start",
                "workspace_backend",
                "summary_callback",
                "session_memory",
                "microcompaction_policy",
                "recovery_tool_available",
            }
        }
        memory_settings["microcompaction_policy"] = self.microcompaction_policy.to_dict()
        return memory_settings


def normalize_memory_settings(settings: dict[str, Any], protected_start: int | None) -> dict[str, Any]:
    return MemoryManagerSettings(**settings, protected_start=protected_start)._definition_settings()
