from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from v_agent.config import ResolvedModelConfig
from v_agent.llm.base import LLMClient
from v_agent.tools.registry import ToolRegistry
from v_agent.types import AgentResult, NoToolPolicy, SubAgentConfig

RuntimeLogHandler = Callable[[str, dict[str, Any]], None]
ToolRegistryFactory = Callable[[], ToolRegistry]


class LLMBuilder(Protocol):
    def __call__(
        self,
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[LLMClient, ResolvedModelConfig]:
        ...


@dataclass(slots=True)
class AgentDefinition:
    description: str
    model: str
    backend: str | None = None
    language: str = "zh-CN"
    max_cycles: int = 10
    no_tool_policy: NoToolPolicy = "continue"
    allow_interruption: bool = True
    use_workspace: bool = True
    enable_todo_management: bool = True
    agent_type: str | None = None
    native_multimodal: bool = False
    enable_sub_agents: bool = False
    sub_agents: dict[str, SubAgentConfig] = field(default_factory=dict)
    extra_tool_names: list[str] = field(default_factory=list)
    exclude_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    system_prompt: str | None = None


@dataclass(slots=True)
class AgentSDKOptions:
    settings_file: Path
    default_backend: str
    workspace: Path = field(default_factory=lambda: Path("./workspace"))
    timeout_seconds: float = 90.0
    llm_builder: LLMBuilder | None = None
    tool_registry_factory: ToolRegistryFactory | None = None
    log_handler: RuntimeLogHandler | None = None


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
            "resolved": {
                "backend": self.resolved.backend,
                "selected_model": self.resolved.selected_model,
                "model_id": self.resolved.model_id,
                "endpoint": self.resolved.endpoint.endpoint_id,
            },
        }
