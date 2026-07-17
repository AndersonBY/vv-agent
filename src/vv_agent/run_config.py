from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from vv_agent.approval import ApprovalBroker, ApprovalProvider
from vv_agent.budget import HostCostMeter, RunBudgetLimits
from vv_agent.checkpoint import CheckpointConfig, CheckpointExtension, ReconciliationProvider
from vv_agent.config import ResolvedModelConfig
from vv_agent.context_providers import ContextProvider
from vv_agent.event_store import RunEventStore
from vv_agent.llm.base import LLMClient
from vv_agent.model_settings import ModelSettings
from vv_agent.runtime.backends.base import ExecutionBackend
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.hooks import RuntimeHook
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import Message, NoToolPolicy, _validate_no_tool_policy

if TYPE_CHECKING:
    from vv_agent.memory.provider import MemoryProvider

StreamHandler = Callable[[Any], None]
ToolRegistryFactory = Callable[[], ToolRegistry]
ApprovalPolicy = Literal["default", "always", "never", "on_request"]
_APPROVAL_POLICIES = frozenset({"default", "always", "never", "on_request"})
CanUseTool = Callable[[str, dict[str, Any]], bool]
RuntimeLogHandler = Callable[[str, dict[str, Any]], None]
BeforeCycleMessageProvider = Callable[[int, list[Message], dict[str, Any]], list[Message]]
InterruptionMessageProvider = Callable[[], list[Message]]
DEFAULT_SETTINGS_FILE = "local_settings.py"
DEFAULT_TIMEOUT_SECONDS = 90.0
_MAX_U32 = (1 << 32) - 1


def _validate_bounded_int(value: object, field_name: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= _MAX_U32:
        raise ValueError(f"{field_name} must be between {minimum} and {_MAX_U32}")
    return value


class LegacyModelProvider(Protocol):
    def __call__(self, agent: Any, run_config: RunConfig) -> tuple[LLMClient, ResolvedModelConfig]: ...


class RuntimeLLMBuilder(Protocol):
    def __call__(
        self,
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[LLMClient, ResolvedModelConfig]: ...


@dataclass(slots=True)
class ToolPolicy:
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    approval: ApprovalPolicy = "default"
    can_use_tool: CanUseTool | None = None

    def __post_init__(self) -> None:
        if self.approval not in _APPROVAL_POLICIES:
            supported = ", ".join(sorted(_APPROVAL_POLICIES))
            raise ValueError(f"approval must be one of: {supported}")


def merge_tool_policies(
    agent_policy: ToolPolicy | None,
    run_policy: ToolPolicy | None,
) -> ToolPolicy | None:
    return merge_tool_policy_layers(agent_policy, None, run_policy)


def merge_tool_policy_layers(
    agent_policy: ToolPolicy | None,
    runner_policy: ToolPolicy | None,
    run_policy: ToolPolicy | None,
) -> ToolPolicy | None:
    if agent_policy is None and runner_policy is None and run_policy is None:
        return None

    agent = agent_policy or ToolPolicy()
    runner = runner_policy or ToolPolicy()
    run = run_policy or ToolPolicy()
    allowed_tools = next(
        (policy.allowed_tools for policy in (run, runner, agent) if policy.allowed_tools is not None),
        None,
    )
    disallowed_tools = list(dict.fromkeys([*agent.disallowed_tools, *runner.disallowed_tools, *run.disallowed_tools]))

    approval = next(
        (policy.approval for policy in (run, agent, runner) if policy.approval != "default"),
        "default",
    )
    can_use_tool = _merge_can_use_tool(
        _merge_can_use_tool(agent.can_use_tool, runner.can_use_tool),
        run.can_use_tool,
    )
    return ToolPolicy(
        allowed_tools=list(allowed_tools) if allowed_tools is not None else None,
        disallowed_tools=disallowed_tools,
        approval=approval,
        can_use_tool=can_use_tool,
    )


def _merge_can_use_tool(
    agent_predicate: CanUseTool | None,
    run_predicate: CanUseTool | None,
) -> CanUseTool | None:
    if agent_predicate is None:
        return run_predicate
    if run_predicate is None:
        return agent_predicate

    def can_use_tool(tool_name: str, arguments: dict[str, Any]) -> bool:
        return bool(agent_predicate(tool_name, dict(arguments))) and bool(run_predicate(tool_name, dict(arguments)))

    return can_use_tool


@dataclass(slots=True)
class RunConfig:
    model: str | Any | None = None
    model_provider: Any | None = None
    model_settings: ModelSettings | None = None
    workspace: str | Path | Any | None = None
    workspace_backend: Any | None = None
    session: Any | None = None
    max_cycles: int | None = None
    max_handoffs: int | None = None
    tool_policy: ToolPolicy | None = None
    execution_backend: ExecutionBackend | None = None
    cancellation_token: CancellationToken | None = None
    approval_provider: ApprovalProvider | None = None
    approval_timeout_seconds: float | None = None
    approval_broker: ApprovalBroker | None = None
    event_store: RunEventStore | None = None
    event_store_fail_closed: bool = False
    stream: StreamHandler | None = None
    hooks: list[RuntimeHook] = field(default_factory=list)
    tracing: dict[str, Any] | None = None
    context: Any | None = None
    context_providers: list[ContextProvider] = field(default_factory=list)
    max_context_chars: int | None = None
    memory_providers: list[MemoryProvider] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    settings_file: str | Path | None = None
    default_backend: str | None = None
    llm_builder: RuntimeLLMBuilder | None = None
    timeout_seconds: float | None = None
    tool_registry_factory: ToolRegistryFactory | None = None
    log_preview_chars: int | None = None
    debug_dump_dir: str | None = None
    shared_state: dict[str, Any] | None = None
    initial_messages: list[Message] | None = None
    before_cycle_messages: BeforeCycleMessageProvider | None = None
    interruption_messages: InterruptionMessageProvider | None = None
    sub_task_manager: Any | None = None
    runtime_log_handler: RuntimeLogHandler | None = None
    runtime_stream_callback: StreamHandler | None = None
    no_tool_policy: NoToolPolicy | None = None
    budget_limits: RunBudgetLimits | None = None
    host_cost_meter: HostCostMeter | None = None
    checkpoint_config: CheckpointConfig | None = None
    checkpoint_extensions: list[CheckpointExtension] = field(default_factory=list)
    reconciliation_provider: ReconciliationProvider | None = None

    def __post_init__(self) -> None:
        _validate_bounded_int(self.max_cycles, "max_cycles", minimum=1)
        _validate_bounded_int(self.max_handoffs, "max_handoffs", minimum=0)
        _validate_no_tool_policy(self.no_tool_policy, "RunConfig.no_tool_policy")
        if self.budget_limits is not None and not isinstance(self.budget_limits, RunBudgetLimits):
            if not isinstance(self.budget_limits, dict):
                raise TypeError("RunConfig.budget_limits must be RunBudgetLimits, an object, or None")
            self.budget_limits = RunBudgetLimits.from_dict(self.budget_limits)
        if self.host_cost_meter is not None and not callable(getattr(self.host_cost_meter, "read", None)):
            raise TypeError("RunConfig.host_cost_meter must provide read() or be None")
        if self.checkpoint_config is not None and not isinstance(self.checkpoint_config, CheckpointConfig):
            if not isinstance(self.checkpoint_config, dict):
                raise TypeError("RunConfig.checkpoint_config must be CheckpointConfig, an object, or None")
            self.checkpoint_config = CheckpointConfig(**self.checkpoint_config)
        for extension in self.checkpoint_extensions:
            if not isinstance(extension, CheckpointExtension):
                raise TypeError("RunConfig.checkpoint_extensions must contain CheckpointExtension values")
        if self.reconciliation_provider is not None and not isinstance(
            self.reconciliation_provider, ReconciliationProvider
        ):
            raise TypeError("RunConfig.reconciliation_provider must provide reconcile() or be None")

    def with_cancellation_token(self, cancellation_token: CancellationToken) -> RunConfig:
        return replace(self, cancellation_token=cancellation_token)
