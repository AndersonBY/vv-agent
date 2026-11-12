from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vv_agent.approval import ApprovalBroker, ApprovalProvider
from vv_agent.budget import HostCostMeter, RunBudgetLimits
from vv_agent.context_providers import ContextProvider
from vv_agent.event_store import _RunEventSink
from vv_agent.events import RunEvent
from vv_agent.kernel._tool_policy import ApprovalPolicy as ApprovalPolicy
from vv_agent.kernel._tool_policy import CanUseTool as CanUseTool
from vv_agent.kernel._tool_policy import ToolPolicy as ToolPolicy
from vv_agent.kernel._tool_policy import merge_tool_policies as merge_tool_policies
from vv_agent.kernel._tool_policy import merge_tool_policy_layers as merge_tool_policy_layers
from vv_agent.microcompaction import MicrocompactionPolicy, normalize_microcompaction_policy
from vv_agent.model_settings import ModelSettings
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.hooks import RuntimeHook
from vv_agent.runtime.lifecycle import AfterCycleHook
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import Message, NoToolPolicy, _validate_no_tool_policy

if TYPE_CHECKING:
    from vv_agent.agent import Agent
    from vv_agent.config import ResolvedModelConfig
    from vv_agent.memory.provider import MemoryProvider
    from vv_agent.model import ModelProvider, ModelRef
    from vv_agent.session.providers import Provider

RunEventObserver = Callable[[RunEvent], None]
ToolRegistryFactory = Callable[[], ToolRegistry]
BeforeCycleMessageProvider = Callable[[int, list[Message], dict[str, Any]], list[Message]]
InterruptionMessageProvider = Callable[[], list[Message]]
_MAX_U32 = (1 << 32) - 1


def _validate_bounded_int(value: object, field_name: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= _MAX_U32:
        raise ValueError(f"{field_name} must be between {minimum} and {_MAX_U32}")
    return value


@dataclass(slots=True)
class RunConfig:
    model: str | ModelRef | ResolvedModelConfig | None = None
    model_provider: ModelProvider | None = None
    model_settings: ModelSettings | None = None
    workspace: str | Path | None = None
    workspace_backend: Any | None = None
    session_memory_enabled: bool = False
    microcompaction_policy: MicrocompactionPolicy = field(default_factory=MicrocompactionPolicy)
    max_cycles: int | None = None
    max_handoffs: int | None = None
    tool_policy: ToolPolicy | None = None
    cancellation_token: CancellationToken | None = None
    approval_provider: ApprovalProvider | None = None
    approval_timeout_seconds: float | None = None
    approval_broker: ApprovalBroker | None = None
    event_store: _RunEventSink | None = None
    event_store_fail_closed: bool = False
    stream: RunEventObserver | None = None
    hooks: list[RuntimeHook] = field(default_factory=list)
    after_cycle_hooks: list[AfterCycleHook] = field(default_factory=list)
    tracing: dict[str, Any] | None = None
    context: Any | None = None
    context_providers: list[ContextProvider] = field(default_factory=list)
    max_context_chars: int | None = None
    memory_providers: list[MemoryProvider] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_registry_factory: ToolRegistryFactory | None = None
    log_preview_chars: int | None = None
    debug_dump_dir: str | None = None
    shared_state: dict[str, Any] | None = None
    initial_messages: list[Message] | None = None
    before_cycle_messages: BeforeCycleMessageProvider | None = None
    interruption_messages: InterruptionMessageProvider | None = None
    no_tool_policy: NoToolPolicy | None = None
    budget_limits: RunBudgetLimits | None = None
    host_cost_meter: HostCostMeter | None = None
    tool_providers: dict[str, Provider] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tool_providers, dict):
            raise TypeError("RunConfig.tool_providers must be a dict of exact tool names to providers")
        for name, provider in self.tool_providers.items():
            if not isinstance(name, str) or not name or any(c.isspace() for c in name):
                raise ValueError("RunConfig.tool_providers requires exact nonempty tool names without whitespace")
            if name in {"model", "function", "ask_user", "create_sub_task"}:
                raise ValueError(f"RunConfig.tool_providers conflicts with reserved binding: {name}")
            missing = [
                method
                for method in ("submit", "query", "cancel", "authenticate")
                if not callable(getattr(provider, method, None))
            ]
            if missing:
                raise TypeError(f"RunConfig.tool_providers[{name!r}] is missing required methods: {', '.join(missing)}")
        self.tool_providers = dict(self.tool_providers)
        _validate_bounded_int(self.max_cycles, "max_cycles", minimum=1)
        _validate_bounded_int(self.max_handoffs, "max_handoffs", minimum=0)
        _validate_no_tool_policy(self.no_tool_policy, "RunConfig.no_tool_policy")
        if not isinstance(self.session_memory_enabled, bool):
            raise TypeError("RunConfig.session_memory_enabled must be a boolean")
        self.microcompaction_policy = normalize_microcompaction_policy(self.microcompaction_policy)
        if self.workspace is not None and not isinstance(self.workspace, (str, Path)):
            raise TypeError("RunConfig.workspace must be a string or Path; use workspace_backend for custom storage")
        if self.model_provider is not None:
            missing = [
                name
                for name in ("resolve", "client", "default_settings", "default_model_ref")
                if not callable(getattr(self.model_provider, name, None))
            ]
            if missing:
                raise TypeError(f"RunConfig.model_provider is missing required methods: {', '.join(missing)}")
        if self.budget_limits is not None and not isinstance(self.budget_limits, RunBudgetLimits):
            if not isinstance(self.budget_limits, dict):
                raise TypeError("RunConfig.budget_limits must be RunBudgetLimits, an object, or None")
            self.budget_limits = RunBudgetLimits.from_dict(self.budget_limits)
        if self.host_cost_meter is not None and not callable(getattr(self.host_cost_meter, "read", None)):
            raise TypeError("RunConfig.host_cost_meter must provide read() or be None")
        for hook in self.after_cycle_hooks:
            if not isinstance(hook, AfterCycleHook):
                raise TypeError("RunConfig.after_cycle_hooks must contain AfterCycleHook values")

    def with_cancellation_token(self, cancellation_token: CancellationToken) -> RunConfig:
        return replace(self, cancellation_token=cancellation_token)


def effective_run_config(
    agent: Agent,
    run_config: RunConfig | None,
    *,
    runner_defaults: RunConfig | None = None,
) -> RunConfig:
    defaults = runner_defaults or RunConfig()
    config = run_config or RunConfig()

    provider_overridden = config.model_provider is not None
    model = config.model
    if model is None:
        model = agent.model
    if model is None and not provider_overridden:
        model = defaults.model

    configured_max_cycles = next(
        (value for value in (config.max_cycles, defaults.max_cycles, agent.max_cycles) if value is not None),
        10,
    )
    configured_max_handoffs = next(
        (value for value in (config.max_handoffs, defaults.max_handoffs) if value is not None),
        10,
    )
    configured_no_tool_policy = next(
        (value for value in (config.no_tool_policy, defaults.no_tool_policy, agent.no_tool_policy) if value is not None),
        "finish",
    )
    effective_max_cycles = _validate_bounded_int(configured_max_cycles, "max_cycles", minimum=1)
    effective_max_handoffs = _validate_bounded_int(configured_max_handoffs, "max_handoffs", minimum=0)
    assert effective_max_cycles is not None
    assert effective_max_handoffs is not None
    model_settings = ModelSettings().resolve(defaults.model_settings).resolve(agent.model_settings).resolve(config.model_settings)
    shared_state = None
    if defaults.shared_state is not None or config.shared_state is not None:
        shared_state = {**(defaults.shared_state or {}), **(config.shared_state or {})}

    def prefer_run(name: str) -> Any:
        value = getattr(config, name)
        return value if value is not None else getattr(defaults, name)

    return replace(
        config,
        model=model,
        model_provider=config.model_provider or defaults.model_provider,
        tool_providers={**defaults.tool_providers, **config.tool_providers},
        model_settings=model_settings,
        workspace=prefer_run("workspace"),
        workspace_backend=prefer_run("workspace_backend"),
        max_cycles=effective_max_cycles,
        max_handoffs=effective_max_handoffs,
        no_tool_policy=configured_no_tool_policy,
        tool_policy=merge_tool_policy_layers(agent.tool_policy, defaults.tool_policy, config.tool_policy),
        cancellation_token=prefer_run("cancellation_token"),
        approval_provider=prefer_run("approval_provider"),
        approval_timeout_seconds=prefer_run("approval_timeout_seconds"),
        approval_broker=prefer_run("approval_broker"),
        event_store=prefer_run("event_store"),
        event_store_fail_closed=defaults.event_store_fail_closed or config.event_store_fail_closed,
        stream=prefer_run("stream"),
        hooks=[*defaults.hooks, *config.hooks],
        after_cycle_hooks=[
            *defaults.after_cycle_hooks,
            *config.after_cycle_hooks,
        ],
        tracing=prefer_run("tracing"),
        context=prefer_run("context"),
        context_providers=[*defaults.context_providers, *config.context_providers],
        max_context_chars=prefer_run("max_context_chars"),
        memory_providers=[*defaults.memory_providers, *config.memory_providers],
        metadata={**defaults.metadata, **config.metadata},
        tool_registry_factory=prefer_run("tool_registry_factory"),
        log_preview_chars=prefer_run("log_preview_chars"),
        debug_dump_dir=prefer_run("debug_dump_dir"),
        shared_state=shared_state,
        initial_messages=prefer_run("initial_messages"),
        before_cycle_messages=prefer_run("before_cycle_messages"),
        interruption_messages=prefer_run("interruption_messages"),
        budget_limits=prefer_run("budget_limits"),
        host_cost_meter=prefer_run("host_cost_meter"),
    )
