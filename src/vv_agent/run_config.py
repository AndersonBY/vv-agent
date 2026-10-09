from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from vv_agent.approval import ApprovalBroker, ApprovalProvider
from vv_agent.budget import HostCostMeter, RunBudgetLimits
from vv_agent.context_providers import ContextProvider
from vv_agent.event_store import _RunEventSink
from vv_agent.events import RunEvent
from vv_agent.microcompaction import MicrocompactionPolicy, normalize_microcompaction_policy
from vv_agent.model_settings import ModelSettings
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.hooks import RuntimeHook
from vv_agent.runtime.lifecycle import AfterCycleHook
from vv_agent.tools.metadata import (
    ToolSideEffect,
    normalize_denied_side_effects,
    normalize_metadata_labels,
)
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import Message, NoToolPolicy, _validate_no_tool_policy

if TYPE_CHECKING:
    from vv_agent.agent import Agent
    from vv_agent.config import ResolvedModelConfig
    from vv_agent.memory.provider import MemoryProvider
    from vv_agent.model import ModelProvider, ModelRef

RunEventObserver = Callable[[RunEvent], None]
ToolRegistryFactory = Callable[[], ToolRegistry]
ApprovalPolicy = Literal["default", "always", "never", "on_request"]
_APPROVAL_POLICIES = frozenset({"default", "always", "never", "on_request"})
CanUseTool = Callable[[str, dict[str, Any]], bool]
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
class ToolPolicy:
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    approval: ApprovalPolicy = "default"
    can_use_tool: CanUseTool | None = None
    denied_side_effects: Sequence[ToolSideEffect | str] = field(default_factory=list)
    denied_capability_tags: list[str] = field(default_factory=list)
    deny_terminal_tools: bool = False
    denied_cost_dimensions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.approval not in _APPROVAL_POLICIES:
            supported = ", ".join(sorted(_APPROVAL_POLICIES))
            raise ValueError(f"approval must be one of: {supported}")
        self.denied_side_effects = normalize_denied_side_effects(self.denied_side_effects)
        self.denied_capability_tags = normalize_metadata_labels(
            self.denied_capability_tags,
            field_name="denied_capability_tags",
        )
        if not isinstance(self.deny_terminal_tools, bool):
            raise TypeError("deny_terminal_tools must be a boolean")
        self.denied_cost_dimensions = normalize_metadata_labels(
            self.denied_cost_dimensions,
            field_name="denied_cost_dimensions",
        )


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
    denied_side_effects = normalize_denied_side_effects(
        [*agent.denied_side_effects, *runner.denied_side_effects, *run.denied_side_effects]
    )
    denied_capability_tags = normalize_metadata_labels(
        [
            *agent.denied_capability_tags,
            *runner.denied_capability_tags,
            *run.denied_capability_tags,
        ],
        field_name="denied_capability_tags",
    )
    denied_cost_dimensions = normalize_metadata_labels(
        [
            *agent.denied_cost_dimensions,
            *runner.denied_cost_dimensions,
            *run.denied_cost_dimensions,
        ],
        field_name="denied_cost_dimensions",
    )

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
        denied_side_effects=denied_side_effects,
        denied_capability_tags=denied_capability_tags,
        deny_terminal_tools=(agent.deny_terminal_tools or runner.deny_terminal_tools or run.deny_terminal_tools),
        denied_cost_dimensions=denied_cost_dimensions,
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

    def __post_init__(self) -> None:
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
