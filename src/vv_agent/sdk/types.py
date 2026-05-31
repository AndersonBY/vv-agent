from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vv_agent.config import ResolvedModelConfig
from vv_agent.types import AgentResult, NoToolPolicy, SubAgentConfig


@dataclass(slots=True)
class LegacyAgentDefinition:
    description: str
    model: str
    backend: str | None = None
    language: str = "zh-CN"
    max_cycles: int = 10
    memory_compact_threshold: int = 128_000
    memory_threshold_percentage: int = 90
    no_tool_policy: NoToolPolicy = "continue"
    allow_interruption: bool = True
    use_workspace: bool = True
    enable_todo_management: bool = True
    agent_type: str | None = None
    native_multimodal: bool = False
    enable_sub_agents: bool = False
    sub_agents: dict[str, SubAgentConfig] = field(default_factory=dict)
    skill_directories: list[str] = field(default_factory=list)
    extra_tool_names: list[str] = field(default_factory=list)
    exclude_tools: list[str] = field(default_factory=list)
    bash_shell: str | None = None
    windows_shell_priority: list[str] = field(default_factory=list)
    bash_env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    system_prompt: str | None = None
    system_prompt_template: str | None = None


@dataclass(slots=True)
class AgentRun:
    agent_name: str
    result: AgentResult
    resolved: ResolvedModelConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "status": self.result.status.value,
            "final_answer": self.result.final_answer,
            "wait_reason": self.result.wait_reason,
            "error": self.result.error,
            "cycles": len(self.result.cycles),
            "todo_list": self.result.todo_list,
            "token_usage": self.result.token_usage.to_dict(),
            "resolved": {
                "backend": self.resolved.backend,
                "selected_model": self.resolved.selected_model,
                "model_id": self.resolved.model_id,
                "endpoint": self.resolved.endpoint.endpoint_id,
            },
        }
