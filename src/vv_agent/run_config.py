from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from vv_agent.approval import ApprovalBroker, ApprovalProvider
from vv_agent.config import ResolvedModelConfig
from vv_agent.context_providers import ContextProvider
from vv_agent.event_store import RunEventStore
from vv_agent.llm.base import LLMClient
from vv_agent.model_settings import ModelSettings
from vv_agent.runtime.backends.base import ExecutionBackend
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.hooks import RuntimeHook
from vv_agent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from vv_agent.memory.provider import MemoryProvider

StreamHandler = Callable[[Any], None]
ToolRegistryFactory = Callable[[], ToolRegistry]
ApprovalPolicy = Literal["default", "always", "never"]
CanUseTool = Callable[[str, dict[str, Any]], bool]


class ModelProvider(Protocol):
    def __call__(self, agent: Any, run_config: RunConfig) -> tuple[LLMClient, ResolvedModelConfig]:
        ...


@dataclass(slots=True)
class ToolPolicy:
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    approval: ApprovalPolicy = "default"
    can_use_tool: CanUseTool | None = None


@dataclass(slots=True)
class RunConfig:
    model: str | Any | None = None
    model_provider: ModelProvider | None = None
    model_settings: ModelSettings | None = None
    workspace: str | Path | Any | None = None
    session: Any | None = None
    max_cycles: int = 10
    memory_policy: Any | None = None
    tool_policy: ToolPolicy | None = None
    execution_backend: ExecutionBackend | None = None
    cancellation_token: CancellationToken | None = None
    approval_provider: ApprovalProvider | None = None
    approval_timeout_seconds: float | None = None
    approval_broker: ApprovalBroker | None = None
    event_store: RunEventStore | None = None
    event_store_fail_closed: bool = False
    stream: StreamHandler | None = None
    hooks: Any | None = None
    tracing: dict[str, Any] | None = None
    context: Any | None = None
    context_providers: list[ContextProvider] = field(default_factory=list)
    memory_providers: list[MemoryProvider] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    settings_file: str | Path = "local_settings.py"
    default_backend: str | None = None
    timeout_seconds: float = 90.0
    tool_registry_factory: ToolRegistryFactory | None = None
    runtime_hooks: list[RuntimeHook] = field(default_factory=list)
    log_preview_chars: int | None = None
    debug_dump_dir: str | None = None

    def with_cancellation_token(self, cancellation_token: CancellationToken) -> RunConfig:
        return replace(self, cancellation_token=cancellation_token)
