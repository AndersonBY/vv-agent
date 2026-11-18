"""Current model, tool, budget and memory bindings for the session kernel."""

from __future__ import annotations

import random
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

from vv_agent.agent import Agent, RunContext
from vv_agent.budget import (
    BudgetEvaluator,
    HostCost,
    HostCostMeter,
)
from vv_agent.config import ResolvedModelConfig
from vv_agent.guardrails import GuardrailResult, apply_input_guardrails
from vv_agent.kernel._canonical_json import validate_sha256
from vv_agent.kernel._definition import DefinitionBindings, DefinitionCache
from vv_agent.kernel._definition import model_identity as _model_identity
from vv_agent.kernel._hydration import ContentReader
from vv_agent.kernel._requirements import (
    SessionRequirements,
    capability_supports,
    model_capability_tags,
    static_capability_requirements,
)
from vv_agent.kernel._retained_budget import retained_budget_inputs
from vv_agent.kernel._tool_policy import _apply_tool_policy_metadata
from vv_agent.kernel._usage import model_usage as model_usage
from vv_agent.llm.base import LLMClient, LlmRequest
from vv_agent.memory import MemoryManager
from vv_agent.model import ModelRef, resolve_run_model
from vv_agent.model_settings import ModelSettings, RetrySettings
from vv_agent.prompt import PromptBundle, PromptSection
from vv_agent.run_config import RunConfig, effective_run_config, merge_tool_policies
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.compiler import AgentCompiler, tool_is_enabled
from vv_agent.runtime.hooks import RuntimeHookManager
from vv_agent.runtime.tool_planner import plan_tool_schemas
from vv_agent.tools.base import ToolContext
from vv_agent.tools.builtins import build_default_registry
from vv_agent.tools.executor import ToolExposure, is_tool_executor
from vv_agent.tools.function import FunctionTool, adapt_tool
from vv_agent.tools.orchestrator import ToolOrchestrator
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import AgentTask, LLMResponse, Message
from vv_agent.workspace.local import LocalWorkspaceBackend

from .bindings import bind_shared_state, durable_shared_state
from .children import ChildSession, ChildTool, SummaryChildTool
from .environments import EnvironmentCapabilities
from .providers import FunctionProvider, Provider
from .records import MODEL_BINDING, Record, copy_json, validate
from .reducer import ExecutionState
from .store import SessionStore
from .timing import Clock, Invoker, LeaseKeeper, SystemClock, ThreadInvoker, ThreadLeaseKeeper

DEFAULT_SQLITE_HEARTBEAT_RETRY_BUDGET_SECONDS = 5.0
DEFAULT_SQLITE_HEARTBEAT_RETRY_DELAY_SECONDS = 0.05


class RuntimeNotReady(Exception):
    """Host input preparation needs another scheduled attempt, without failing a turn."""

    def __init__(self, retry_after_ms: int):
        if type(retry_after_ms) is not int or retry_after_ms <= 0:
            raise ValueError("retry_after_ms must be a positive integer")
        self.retry_after_ms = retry_after_ms
        super().__init__(f"Runtime not ready; retry after {retry_after_ms} ms")


class InvalidTrustedEnvironmentBinding(ValueError):
    """A host-provided turn environment binding is incomplete or malformed."""

    code = "invalid_trusted_environment_binding"


@dataclass(frozen=True, slots=True)
class TrustedEnvironmentBinding:
    """Host-owned environment facts frozen before a turn is published."""

    capabilities: EnvironmentCapabilities
    descriptor: dict[str, Any]
    operational_policy: dict[str, Any]


class EndpointUnavailable(Exception):
    """The frozen endpoint disappeared before dispatch."""


class _EndpointClient(Protocol):
    endpoint_targets: list[Any]
    max_retries_per_endpoint: int
    randomize_endpoints: bool

    def ordered_targets(self, preferred: str | None = None) -> list[Any]: ...


def _is_vv_endpoint_client(client: object) -> bool:
    """Use the real class only when its defining module is already loaded."""

    module = sys.modules.get("vv_agent.llm.vv_llm_client")
    client_type = getattr(module, "VvLlmClient", None)
    return isinstance(client_type, type) and isinstance(client, client_type)


def _resolved_model_binding(resolved: ResolvedModelConfig) -> dict[str, Any]:
    return {
        "backend": resolved.backend,
        "model": resolved.model_id,
        "context_length": resolved.context_length,
        "max_output_tokens": resolved.max_output_tokens,
    }


def _metadata_text(metadata: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_static_model_config(agent: Agent, config: RunConfig) -> ResolvedModelConfig:
    """Resolve trusted model metadata without constructing an execution client."""

    provider = config.model_provider
    if provider is None:
        raise ValueError("RunConfig.model_provider is required")
    model = config.model or agent.model or provider.default_model_ref()
    if model is None:
        raise ValueError("Agent.model, RunConfig.model, or ModelProvider.default_model_ref() is required")
    return provider.resolve(ModelRef.coerce(model))


def _static_model_bindings(
    agent: Agent,
    config: RunConfig,
    resolved: ResolvedModelConfig,
) -> list[dict[str, Any]]:
    metadata = {**agent.metadata, **config.metadata}
    summary_backend = _metadata_text(metadata, "memory_summary_backend")
    summary_model = _metadata_text(metadata, "memory_summary_model") or resolved.model_id
    extraction_backend = _metadata_text(metadata, "session_memory_extraction_backend") or summary_backend
    extraction_model = _metadata_text(metadata, "session_memory_extraction_model") or summary_model
    routes = {
        "compaction": (summary_backend, summary_model),
        "session_memory": (extraction_backend, extraction_model),
    }
    provider = config.model_provider
    bindings = [_resolved_model_binding(resolved)]
    for backend, model in routes.values():
        if model == resolved.model_id and (backend is None or backend == resolved.backend):
            continue
        if provider is None:
            raise ValueError("RunConfig.model_provider is required for internal model requirements")
        route = provider.resolve(ModelRef.backend(backend, model) if backend else ModelRef.named(model))
        bindings.append(_resolved_model_binding(route))
    return bindings


def _build_configured_registry(
    agent: Agent,
    config: RunConfig,
    *,
    runtime: Runtime | None,
) -> tuple[ToolRegistry, dict[str, FunctionTool]]:
    registry = config.tool_registry_factory() if config.tool_registry_factory else build_default_registry()
    dynamic: dict[str, FunctionTool] = {}
    for candidate in agent.tools:
        if is_tool_executor(candidate):
            executor = candidate
        else:
            tool = candidate if isinstance(candidate, FunctionTool) else adapt_tool(candidate)
            if callable(tool.is_enabled):
                dynamic[tool.name] = tool
            elif not tool_is_enabled(tool=tool, agent=agent, run_config=config):
                continue
            executor = tool.to_executor()
        visible = executor.exposure == ToolExposure.DIRECT
        registry.register_executor(executor, expose_to_model=visible, planner_extra=visible)
    if runtime is None:
        from .delegation import register_handoffs

        register_handoffs(cast(Any, SimpleNamespace(agent=agent)), registry)
    else:
        from .delegation import register_adapters

        register_adapters(runtime, registry)
    return registry, dynamic


def _trusted_planning_task(
    agent: Agent,
    config: RunConfig,
    registry: ToolRegistry,
    resolved: ResolvedModelConfig,
) -> AgentTask:
    metadata = {**agent.metadata, **config.metadata}
    metadata["session_memory_enabled"] = config.session_memory_enabled
    policy = merge_tool_policies(agent.tool_policy, config.tool_policy)
    _apply_tool_policy_metadata(metadata, policy)
    extra_tool_names = [
        *[tool.name for tool in agent.tools if isinstance(tool, FunctionTool) and tool.exposure is ToolExposure.DIRECT],
        *(transfer.tool_name for transfer in agent.handoffs if transfer.tool_name is not None),
        *registry.list_planner_extra_tool_names(),
    ]
    return AgentTask(
        task_id="requirements-bound",
        model=resolved.model_id,
        prompt_bundle=PromptBundle((PromptSection(id="requirements_bound", text="requirements bound", stable=True),)),
        user_prompt="",
        microcompaction_policy=config.microcompaction_policy,
        no_tool_policy=config.no_tool_policy or agent.no_tool_policy or "finish",
        sub_agents=deepcopy(agent.sub_agents),
        native_multimodal=resolved.native_multimodal,
        extra_tool_names=list(dict.fromkeys(extra_tool_names)),
        metadata=metadata,
    )


def _static_capability_tags(
    agent: Agent,
    config: RunConfig,
    resolved: ResolvedModelConfig | None,
    *,
    extra_capabilities: Iterable[str],
    complete_catalog: bool = False,
) -> tuple[str, ...]:
    effective = effective_run_config(agent, config)
    if resolved is None:
        resolved = _resolve_static_model_config(agent, effective)
    registry, _ = _build_configured_registry(agent, effective, runtime=None)
    schemas = (
        registry.list_openai_schemas()
        if complete_catalog
        else plan_tool_schemas(
            registry=registry,
            task=_trusted_planning_task(agent, effective, registry, resolved),
            include_dynamic_hints=False,
        )
    )
    tools = [(schema, registry.tool_metadata(schema["function"]["name"])) for schema in schemas]
    return capability_supports(
        tools=tools,
        models=_static_model_bindings(agent, effective, resolved),
        extra_capabilities=extra_capabilities,
    )


def static_session_requirements(
    agent: Agent,
    config: RunConfig,
    resolved: ResolvedModelConfig | None = None,
    *,
    level: int = 2,
    handler_version: str | None,
    extra_capabilities: Iterable[str] = (),
) -> SessionRequirements:
    """Build an L2 pre-admission bound from trusted static configuration only."""

    extra = tuple(extra_capabilities)
    if handler_version is None and not extra:
        raise ValueError("null handler requirements require an explicit trusted family capability")
    return static_capability_requirements(
        level=level,
        handler_version=handler_version,
        extra_capabilities=_static_capability_tags(
            agent,
            config,
            resolved,
            extra_capabilities=extra,
        ),
    )


def static_worker_supports(
    agent: Agent,
    config: RunConfig,
    resolved: ResolvedModelConfig | None = None,
    *,
    extra_capabilities: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return the complete trusted tool and model capability tag set."""

    return _static_capability_tags(
        agent,
        config,
        resolved,
        extra_capabilities=extra_capabilities,
        complete_catalog=True,
    )


def static_model_selection_requirements(
    selection: dict[str, Any],
    *,
    resolve_model: Callable[[dict[str, Any]], tuple[LLMClient, ResolvedModelConfig]],
    handler_version: str,
    level: int = 2,
) -> SessionRequirements:
    """Tag one already validated trusted model-resolution result."""

    _, resolved = resolve_model(copy_json(selection))
    return SessionRequirements(
        level=level,
        handler_version=handler_version,
        capabilities=model_capability_tags(_resolved_model_binding(resolved)),
    )


@dataclass
class Runtime:
    agent: Agent
    config: RunConfig
    resolved: ResolvedModelConfig
    llm: LLMClient
    heartbeat_store: Callable[[], AbstractContextManager[SessionStore]]
    providers: dict[str, Provider] = field(default_factory=dict)  # tool name -> trusted adapter
    children: dict[str, Callable[[Record], ChildSession | Sequence[ChildSession]] | ChildTool | SummaryChildTool] = field(
        default_factory=dict
    )
    host_bindings: dict[str, Any] = field(default_factory=dict)
    trusted_environment_binding: TrustedEnvironmentBinding | None = field(default=None, kw_only=True)
    model_resolver: Callable[[dict[str, Any]], tuple[LLMClient, ResolvedModelConfig]] | None = None
    frozen_task: AgentTask | None = None
    handler_version: str = "1"
    ttl_ms: int = 60000
    heartbeat_seconds: float = 0.25
    poll_ms: int = 1000
    cancellation_grace: float = 0.25
    lease_retry_attempts: int = 5
    lease_retry_base_seconds: float = 0.01
    lease_retry_cap_seconds: float = 0.5
    lease_retry_jitter: Callable[[float, float], float] = random.uniform
    lease_retry_sleep: Callable[[float], None] = time.sleep
    sqlite_heartbeat_retry_budget_seconds: float = DEFAULT_SQLITE_HEARTBEAT_RETRY_BUDGET_SECONDS
    sqlite_heartbeat_retry_delay_seconds: float = DEFAULT_SQLITE_HEARTBEAT_RETRY_DELAY_SECONDS
    model_timeout: float = 300
    tool_timeout: float = 300
    hook: Callable[[str, Record | None], None] = lambda _point, _record: None
    wake: Callable[[str], None] = lambda _sid: None
    content_reader: ContentReader | None = None
    memory_manager: MemoryManager = field(default_factory=MemoryManager)
    clock: Clock = field(default_factory=SystemClock)
    lease_keeper: LeaseKeeper = field(default_factory=ThreadLeaseKeeper)
    invoker: Invoker = field(default_factory=ThreadInvoker)
    _dynamic_enablement_turn_id: str | None = field(default=None, init=False, repr=False)
    _requirements_bound: SessionRequirements | None = field(default=None, init=False, repr=False)
    _key_unavailable: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.lease_retry_attempts, bool)
            or not isinstance(self.lease_retry_attempts, int)
            or self.lease_retry_attempts < 1
        ):
            raise ValueError("lease_retry_attempts must be a positive integer")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (self.lease_retry_base_seconds, self.lease_retry_cap_seconds)
        ) or not 0 < self.lease_retry_base_seconds <= self.lease_retry_cap_seconds < float("inf"):
            raise ValueError("lease retry delays must be finite, positive, and base <= cap")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (self.sqlite_heartbeat_retry_budget_seconds, self.sqlite_heartbeat_retry_delay_seconds)
        ) or not all(
            0 < value < float("inf")
            for value in (self.sqlite_heartbeat_retry_budget_seconds, self.sqlite_heartbeat_retry_delay_seconds)
        ):
            raise ValueError("SQLite heartbeat retry settings must be finite and positive")
        if not 0 < self.heartbeat_seconds <= 1 or self.ttl_ms <= self.heartbeat_seconds * 2000:
            raise ValueError("heartbeat must be <=1s and less than half the lease TTL")
        if self.content_reader is not None and not isinstance(self.content_reader, ContentReader):
            raise TypeError("content_reader must implement ContentReader")

        self.config = effective_run_config(self.agent, self.config)
        self.providers = {**self.config.tool_providers, **self.providers}
        self._dynamic_tools: dict[str, FunctionTool] = {}
        self._enabled_dynamic_tools: set[str] = set()
        self.hooks = RuntimeHookManager([*self.agent.hooks, *self.config.hooks])
        self.approval_broker = self.config.approval_broker
        if self.config.approval_provider is not None and self.approval_broker is None:
            from vv_agent.approval import ApprovalBroker

            self.approval_broker = ApprovalBroker()
        self._definitions = DefinitionCache()
        self._memory_clients: dict[tuple[str | None, str], tuple[LLMClient, ResolvedModelConfig]] = {}
        self._memory_routes: dict[str, tuple[LLMClient, ResolvedModelConfig]] = {}
        self._memory_binding_key: tuple | None = None
        self._memory_binding: dict[str, Any] = {}
        if self.providers:
            conflicts = self.providers.keys() & (self.children.keys() | self.delegated_tools | {"ask_user", "function"})
            if conflicts:
                raise ValueError(f"Tool providers conflict with runtime bindings: {', '.join(sorted(conflicts))}")

    @cached_property
    def workspace_backend(self):
        return self.config.workspace_backend or LocalWorkspaceBackend(Path(self.config.workspace or "."))

    @cached_property
    def registry(self) -> ToolRegistry:
        # A read-only child result projection does not dispatch or plan tools.
        return self._configured_registry()

    def _configured_registry(self) -> ToolRegistry:
        registry, self._dynamic_tools = _build_configured_registry(
            self.agent,
            self.config,
            runtime=self,
        )
        return registry

    def _sync_dynamic_tools(self, turn_id: str) -> None:
        """Evaluate callable enablement once per turn before definition planning."""

        if self._dynamic_enablement_turn_id == turn_id:
            return
        registry = self.registry
        for name, tool in self._dynamic_tools.items():
            enabled = tool_is_enabled(tool=tool, agent=self.agent, run_config=self.config)
            registered = registry.has_executor(name)
            if enabled and not registered:
                visible = tool.exposure == ToolExposure.DIRECT
                registry.register_executor(tool.to_executor(), expose_to_model=visible, planner_extra=visible)
                self._enabled_dynamic_tools.add(name)
            elif not enabled and registered:
                registry.unregister(name)
                self._enabled_dynamic_tools.discard(name)
            elif enabled:
                self._enabled_dynamic_tools.add(name)
        self._dynamic_enablement_turn_id = turn_id

    def requirements_bound(self) -> SessionRequirements:
        """Return the trusted static pre-admission bound without compiling input."""

        if self._requirements_bound is None:
            self._requirements_bound = static_session_requirements(
                self.agent,
                self.config,
                self.resolved,
                handler_version=self.handler_version,
            )
        return self._requirements_bound

    def turn_environment_binding(self) -> TrustedEnvironmentBinding | None:
        """Snapshot the host-owned level-3 binding before turn publication."""

        binding = self.trusted_environment_binding
        if binding is None:
            return None
        if type(binding) is not TrustedEnvironmentBinding:
            raise InvalidTrustedEnvironmentBinding("trusted environment binding must be TrustedEnvironmentBinding")
        capabilities = binding.capabilities
        if type(capabilities) is not EnvironmentCapabilities:
            raise InvalidTrustedEnvironmentBinding("trusted environment capabilities must be EnvironmentCapabilities")
        if capabilities.capabilities_digest is None:
            raise InvalidTrustedEnvironmentBinding("trusted environment capabilities_digest is required")
        try:
            descriptor = copy_json(binding.descriptor)
            operational_policy = copy_json(binding.operational_policy)
        except (TypeError, ValueError) as exc:
            raise InvalidTrustedEnvironmentBinding("trusted environment content must be canonical JSON") from exc
        if type(descriptor) is not dict:
            raise InvalidTrustedEnvironmentBinding("trusted environment descriptor must be an object")
        if type(operational_policy) is not dict or set(operational_policy) != {
            "network_policy_digest",
            "isolation_policy_digest",
        }:
            raise InvalidTrustedEnvironmentBinding("trusted environment operational_policy must be a closed object")
        for field_name, digest_value in operational_policy.items():
            try:
                validate_sha256(digest_value, f"operational_policy.{field_name}")
            except ValueError as exc:
                raise InvalidTrustedEnvironmentBinding(str(exc)) from exc
        return TrustedEnvironmentBinding(
            capabilities=capabilities,
            descriptor=descriptor,
            operational_policy=operational_policy,
        )

    def model_selection_requirements(self, selection: dict[str, Any]) -> SessionRequirements:
        """Resolve one logged selection into its exact model-only requirements."""

        return static_model_selection_requirements(
            selection,
            resolve_model=self.resolve_model,
            handler_version=self.handler_version,
        )

    @cached_property
    def delegated_tools(self) -> set[str]:
        names = {
            name
            for name in self.registry.list_tool_names()
            if self.registry.get_executor(name).metadata.get("mode") in {"agent_as_tool", "background_task"}
        }
        names.update(t.tool_name for t in self.agent.handoffs if t.tool_name)
        if self.agent.sub_agents:
            names.add("create_sub_task")
        return names

    @cached_property
    def functions(self) -> FunctionProvider:
        return FunctionProvider(ToolOrchestrator.from_registry(self.registry))

    def durable_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return durable_shared_state(state, self.host_bindings) if self.host_bindings else state

    def bind_state(self, state: dict[str, Any], task: AgentTask) -> dict[str, Any]:
        required = task.metadata.get("vv_session", {}).get("host_binding_names", [])
        return bind_shared_state(state, self.host_bindings, required) if required or self.host_bindings else state

    def for_agent(self, agent: Agent, config: RunConfig) -> Runtime:

        client, resolved = (
            resolve_run_model(agent=agent, run_config=config) if config.model_provider else (self.llm, self.resolved)
        )
        return Runtime(
            agent,
            config,
            resolved,
            client,
            self.heartbeat_store,
            providers=dict(self.providers),
            model_resolver=self.model_resolver,
            children=self.children,
            host_bindings=self.host_bindings,
            clock=self.clock,
            lease_keeper=self.lease_keeper,
            invoker=self.invoker,
            handler_version=self.handler_version,
            ttl_ms=self.ttl_ms,
            heartbeat_seconds=self.heartbeat_seconds,
            poll_ms=self.poll_ms,
            cancellation_grace=self.cancellation_grace,
            lease_retry_attempts=self.lease_retry_attempts,
            lease_retry_base_seconds=self.lease_retry_base_seconds,
            lease_retry_cap_seconds=self.lease_retry_cap_seconds,
            lease_retry_jitter=self.lease_retry_jitter,
            lease_retry_sleep=self.lease_retry_sleep,
            sqlite_heartbeat_retry_budget_seconds=self.sqlite_heartbeat_retry_budget_seconds,
            sqlite_heartbeat_retry_delay_seconds=self.sqlite_heartbeat_retry_delay_seconds,
            model_timeout=self.model_timeout,
            tool_timeout=self.tool_timeout,
            hook=self.hook,
            wake=self.wake,
            memory_manager=replace(self.memory_manager),
        )

    def child_runtime(self, store: SessionStore, session_id: str) -> Runtime:
        from .delegation import reconstruct

        return reconstruct(self, store, session_id)

    def child_tasks(self, store: SessionStore, session_id: str):
        from .delegation import ChildTasks

        return ChildTasks(store, session_id, self)

    def run_context(self, tid: str) -> RunContext:
        return RunContext(
            context=self.config.context,
            run_id=tid,
            agent_name=self.agent.name,
            model=self.resolved.model_id,
            workspace=self.config.workspace,
            metadata={**self.agent.metadata, **self.config.metadata},
        )

    def model_route(self, purpose: str) -> tuple[LLMClient, ResolvedModelConfig]:
        route = self._memory_routes.get(purpose)
        if route is not None and self.config.model_provider is not None:
            provider = self.config.model_provider
            resolved = provider.resolve(ModelRef.backend(route[1].backend, route[1].requested_model))
            return provider.client(resolved), resolved
        return route or self.resolve_model({"model": self.resolved.model_id, "route": {}})

    def _memory_bindings(self, task: AgentTask) -> dict[str, Any]:
        meta = task.metadata
        summary_backend = meta.get("memory_summary_backend") or self.memory_manager.summary_backend
        summary_model = meta.get("memory_summary_model") or self.memory_manager.summary_model or task.model
        extraction_backend = meta.get("session_memory_extraction_backend") or summary_backend
        extraction_model = meta.get("session_memory_extraction_model") or summary_model
        key = (task.model, summary_backend, summary_model, extraction_backend, extraction_model)
        if key == self._memory_binding_key and not self._memory_routes:
            return self._memory_binding
        routes = {
            "compaction": (summary_backend, summary_model),
            "session_memory": (extraction_backend, extraction_model),
        }
        self._memory_routes = {}
        bindings = {}
        for purpose, (backend, model) in routes.items():
            if model == task.model and (backend is None or backend == self.resolved.backend):
                continue
            client_key = (backend, model)
            if client_key not in self._memory_clients:
                provider = self.config.model_provider
                if provider is None:
                    raise ValueError("RunConfig.model_provider is required.")
                resolved = provider.resolve(ModelRef.backend(backend, model) if backend else ModelRef.named(model))
                self._memory_clients[client_key] = (provider.client(resolved), resolved)
            _client, resolved = self._memory_routes[purpose] = self._memory_clients[client_key]
            bindings[purpose] = {
                "backend": resolved.backend,
                "model": resolved.model_id,
                "context_length": resolved.context_length,
                "max_output_tokens": resolved.max_output_tokens,
            }
        self._memory_binding_key, self._memory_binding = key, bindings
        return bindings

    def resolve_model(self, request: dict[str, Any]) -> tuple[LLMClient, ResolvedModelConfig]:
        """Resolve only the retained request; this binding must not submit provider work."""
        if self.model_resolver is not None:
            client, resolved = self.model_resolver(copy_json(request))
        elif request["model"] == self.resolved.model_id and not request.get("route", {}):
            client, resolved = self.llm, self.resolved
        else:
            raise ValueError("Runtime requires a model_resolver for this selection")
        if resolved.model_id != request["model"]:
            raise ValueError("Resolver changed the requested model")
        validate(self.model_binding(client, resolved), MODEL_BINDING)
        retained = request.get("metadata", {}).get("vv_session", {}).get("model_binding")
        if retained is not None and _model_identity(self.model_binding(client, resolved)) != _model_identity(retained):
            raise ValueError("Resolved model binding differs from the retained plan")
        return client, resolved

    @staticmethod
    def model_binding(client: LLMClient, resolved: ResolvedModelConfig) -> dict[str, Any]:
        is_vv_client = _is_vv_endpoint_client(client)
        endpoint_client = cast(_EndpointClient, client)
        return {
            "backend": resolved.backend,
            "model": resolved.model_id,
            "endpoints": [target.endpoint_id for target in endpoint_client.endpoint_targets] if is_vv_client else [],
            "context_length": resolved.context_length,
            "max_output_tokens": resolved.max_output_tokens,
        }

    def prepare_model(self, planned_request: dict[str, Any], purpose: str, attempt: int) -> LLMClient:
        """Bind one current endpoint before admitting a dispatch."""

        client, resolved = self.resolve_model(planned_request) if purpose == "primary" else self.model_route(purpose)
        retained = planned_request.get("metadata", {}).get("vv_session", {}).get("model_binding")
        if retained is not None and _model_identity(self.model_binding(client, resolved)) != _model_identity(retained):
            raise ValueError("Resolved model binding differs from the retained plan")
        if _is_vv_endpoint_client(client):
            # One endpoint per logged attempt; never stack client fallback and kernel retries.
            endpoint_client = cast(_EndpointClient, copy(client))
            order = planned_request.get("metadata", {}).get("vv_session", {}).get("endpoint_order", [])
            endpoint_id = order[(attempt - 1) % len(order)] if order else None
            target = next(
                (item for item in endpoint_client.endpoint_targets if item.endpoint_id == endpoint_id),
                None,
            )
            if target is None:
                raise EndpointUnavailable("The planned endpoint is no longer available")
            endpoint_client.endpoint_targets = [target]
            endpoint_client.max_retries_per_endpoint = 1
            endpoint_client.randomize_endpoints = False
            client = cast(LLMClient, endpoint_client)
        return client

    def complete(
        self, request: LlmRequest, attempt: int, stream_callback=None, *, planned_request=None, prepared_client=None
    ) -> LLMResponse:
        if planned_request is None:
            planned_request = {"model": request.model, "metadata": request.metadata, "route": {}}
        client = prepared_client or self.prepare_model(planned_request, request.metadata.get("purpose", "primary"), attempt)
        if stream_callback is not None:
            return client.complete_with_stream(request, stream_callback)
        return client.complete(request)

    def compile(self, content: str, tid: str, *, seed: dict[str, Any] | None = None, control: bool = False) -> AgentTask:

        if "vv_session" in self.agent.metadata or "vv_session" in self.config.metadata:
            raise ValueError("vv_session is reserved for kernel metadata")
        if self.frozen_task is not None:
            task = deepcopy(self.frozen_task)
            task.task_id = tid
            if tid != self.frozen_task.task_id:
                task.user_prompt = content
            return task
        self._sync_dynamic_tools(tid)
        guardrail = (
            GuardrailResult.allow()
            if control
            else apply_input_guardrails(agent=self.agent, run_context=self.run_context(tid), user_input=content)
        )
        if guardrail.outcome == "rewrite":
            content = str(guardrail.value)
        if guardrail.outcome in {"block", "require_approval"}:
            return AgentTask(
                task_id=tid,
                model=self.resolved.model_id,
                prompt_bundle=PromptBundle((PromptSection(id="blocked", text="Input blocked", stable=True),)),
                user_prompt=content,
                use_workspace=False,
                metadata={"vv_session": {"input_blocked": guardrail.message or "Input blocked by guardrail."}},
            )
        policy = merge_tool_policies(self.agent.tool_policy, self.config.tool_policy)
        task = AgentCompiler().compile(
            agent=self.agent,
            input=content,
            run_config=replace(
                self.config,
                tool_policy=policy,
                context_providers=[] if control else self.config.context_providers,
                **(
                    {
                        "initial_messages": [Message.from_dict(copy_json(m)) for m in seed["messages"]],
                        "shared_state": copy_json(seed["shared_state"]),
                    }
                    if seed is not None
                    else {}
                ),
            ),
            resolved=self.resolved,
            trace_id=tid,
            run_id=tid,
        )
        task.task_id = tid
        task.metadata["_vv_agent_tool_policy_approval"] = policy.approval if policy else "default"

        defaults = self.config.model_provider.default_settings(self.resolved) if self.config.model_provider else ModelSettings()
        task.model_settings = replace(
            defaults.resolve(task.model_settings), retry=RetrySettings(max_attempts=1, backoff_seconds=0)
        )
        if task.metadata.get("session_memory_enabled"):
            from .memory import session_memory

            memory = session_memory(task, Path(self.config.workspace or ".") if task.use_workspace else None)
            memory.load()
            task.metadata.setdefault("vv_session", {})["memory_initial_state"] = memory.state.to_dict()
        if self.agent.handoffs:
            task.metadata.setdefault("vv_session", {})["max_handoffs"] = self.config.max_handoffs
            task.metadata.setdefault("vv_session", {})["handoff_targets"] = {
                t.tool_name: t.agent.name for t in self.agent.handoffs
            }
        if self.host_bindings:
            if any(not isinstance(name, str) or not name for name in self.host_bindings):
                raise ValueError("host binding names must be non-empty strings")
            task.metadata.setdefault("vv_session", {})["host_binding_names"] = sorted(self.host_bindings)
            self.bind_state(task.initial_shared_state, task)
        task.initial_shared_state.setdefault("todo_list", [])
        for key in ("available_skills", "active_skills"):
            if key in task.metadata:
                task.initial_shared_state.setdefault(key, task.metadata[key])
        task.extra_tool_names = list(dict.fromkeys([*task.extra_tool_names, *self.registry.list_planner_extra_tool_names()]))
        return task

    def context(
        self,
        task: AgentTask,
        plan: Record,
        token: CancellationToken,
        *,
        approved: bool = False,
        shared_state: dict[str, Any] | None = None,
        store: SessionStore | None = None,
        session_id: str | None = None,
    ) -> ToolContext:
        from ._dispatch_context import context

        return context(self, task, plan, token, approved=approved, shared_state=shared_state, store=store, session_id=session_id)

    # Admission borrows these cached fragments; DefinitionCache remains their sole owner.
    @property
    def _retained_task_key(self) -> tuple[Record, str | bytes] | None:
        return self._definitions._retained_task_key

    @property
    def _definition_suffix(self) -> bytes:
        return self._definitions._definition_suffix

    @property
    def _definition_digest(self) -> str:
        return self._definitions._definition_digest

    def _definition_bindings(self, task: AgentTask | Record) -> DefinitionBindings:
        current_task = task._task() if isinstance(task, Record) else task
        memory_settings = self.memory_manager._definition_settings()
        model_binding = self.model_binding(self.llm, self.resolved)
        memory_bindings = self._memory_bindings(current_task)
        if memory_bindings:
            model_binding["internal"] = memory_bindings
        return DefinitionBindings(
            agent_name=self.agent.name,
            memory_settings=memory_settings,
            child_tools=list(self.children),
            model_binding=model_binding,
        )

    def definition(self, task: AgentTask) -> dict[str, Any]:
        self._sync_dynamic_tools(task.task_id)
        return self._definitions.definition(task, self._definition_bindings(task), self.registry)

    def definition_digest(self, task: AgentTask | Record) -> str:
        self._sync_dynamic_tools(task.task_id if isinstance(task, AgentTask) else str(task.turn_id or task.record_id))
        return self._definitions.definition_digest(task, self._definition_bindings(task), self.registry)

    def _definition(self, task: AgentTask | Record) -> dict[str, Any]:
        self._sync_dynamic_tools(task.task_id if isinstance(task, AgentTask) else str(task.turn_id or task.record_id))
        return self._definitions._definition(task, self._definition_bindings(task), self.registry)


def request_from_dict(value: dict[str, Any]) -> LlmRequest:
    return LlmRequest(
        model=value["model"],
        messages=[Message.from_dict(m) for m in value["messages"]],
        tools=value["tools"],
        metadata=value["metadata"],
        prompt_bundle=PromptBundle.from_dict(value["prompt_bundle"]) if value["prompt_bundle"] else None,
        model_settings=ModelSettings.from_dict(value["model_settings"]) if value["model_settings"] else None,
    )


def budget(state: ExecutionState, tid: str) -> BudgetEvaluator | None:
    inputs = retained_budget_inputs(state, tid)
    if inputs is None:
        return None
    evaluator = BudgetEvaluator(
        inputs.limits,
        initial_usage=inputs.usage,
        host_cost_meter=cast(
            HostCostMeter,
            SimpleNamespace(read=lambda: HostCost.from_dict(inputs.host_cost) if inputs.host_cost else None),
        ),
        clock_ns=lambda: 0,
    )
    inputs.account_models(evaluator)
    return evaluator
