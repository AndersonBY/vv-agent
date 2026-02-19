from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from v_agent.config import ResolvedModelConfig
from v_agent.llm.base import LLMClient
from v_agent.runtime.backends.base import ExecutionBackend
from v_agent.runtime.context import StreamCallback
from v_agent.runtime.hooks import RuntimeHook
from v_agent.tools.registry import ToolRegistry
from v_agent.types import AgentResult, NoToolPolicy, SubAgentConfig

if TYPE_CHECKING:
    from v_agent.sdk.resources import AgentResourceLoader

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
    skill_directories: list[str] = field(default_factory=list)
    extra_tool_names: list[str] = field(default_factory=list)
    exclude_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    system_prompt: str | None = None
    system_prompt_template: str | None = None


@dataclass(slots=True)
class AgentSDKOptions:
    settings_file: Path
    default_backend: str
    workspace: Path = field(default_factory=lambda: Path("./workspace"))
    timeout_seconds: float = 90.0
    llm_builder: LLMBuilder | None = None
    tool_registry_factory: ToolRegistryFactory | None = None
    log_handler: RuntimeLogHandler | None = None
    runtime_hooks: list[RuntimeHook] = field(default_factory=list)
    resource_loader: AgentResourceLoader | None = None
    auto_discover_resources: bool = True
    execution_backend: ExecutionBackend | None = None
    stream_callback: StreamCallback | None = None


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
