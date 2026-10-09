"""Bindings to existing runtime components; no checkpoint controller or second executor."""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from copy import copy, deepcopy
from dataclasses import dataclass, field, fields, replace
from functools import cached_property
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from vv_agent.agent import Agent, RunContext
from vv_agent.budget import (
    BudgetDimension,
    BudgetEvaluator,
    BudgetUnavailableDimension,
    BudgetUnavailableReason,
    BudgetUsageSnapshot,
    HostCost,
    HostCostMeter,
    RunBudgetLimits,
)
from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.config import ResolvedModelConfig
from vv_agent.llm.base import LLMClient, LlmRequest
from vv_agent.llm.vv_llm_client import VvLlmClient
from vv_agent.memory import MemoryManager
from vv_agent.model import ModelRef
from vv_agent.model_settings import ModelSettings, RetrySettings
from vv_agent.prompt import PromptBundle, PromptSection
from vv_agent.run_config import RunConfig, ToolPolicy, merge_tool_policies
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.compiler import AgentCompiler, _apply_tool_policy_metadata
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.hooks import RuntimeHookManager
from vv_agent.runtime.token_usage import normalize_token_usage
from vv_agent.runtime.tool_planner import plan_tool_names, plan_tool_schemas
from vv_agent.tools.base import ToolContext
from vv_agent.tools.builtins import build_default_registry
from vv_agent.tools.executor import ToolExposure, is_tool_executor
from vv_agent.tools.function import FunctionTool, adapt_tool
from vv_agent.tools.metadata import ToolResultRetention
from vv_agent.tools.orchestrator import ToolOrchestrator
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import AgentTask, LLMResponse, Message, ToolCall
from vv_agent.workspace.local import LocalWorkspaceBackend

from .bindings import bind_shared_state, durable_shared_state
from .children import ChildSession
from .providers import FunctionProvider, Provider
from .records import Record, copy_json
from .reducer import ExecutionState
from .store import SessionStore


@dataclass
class Runtime:
    agent: Agent
    config: RunConfig
    resolved: ResolvedModelConfig
    llm: LLMClient
    heartbeat_store: Callable[[], AbstractContextManager[SessionStore]]
    providers: dict[str, Provider] = field(default_factory=dict)  # tool name -> trusted adapter
    children: dict[str, Callable[[Record], ChildSession]] = field(default_factory=dict)
    host_bindings: dict[str, Any] = field(default_factory=dict)
    frozen_task: AgentTask | None = None
    handler_version: str = "1"
    ttl_ms: int = 15000
    heartbeat_seconds: float = 0.25
    poll_ms: int = 1000
    cancellation_grace: float = 0.25
    lease_retry_attempts: int = 5
    lease_retry_base_seconds: float = 0.01
    lease_retry_cap_seconds: float = 0.5
    lease_retry_jitter: Callable[[float, float], float] = random.uniform
    lease_retry_sleep: Callable[[float], None] = time.sleep
    model_timeout: float = 300
    tool_timeout: float = 300
    hook: Callable[[str, Record | None], None] = lambda _point, _record: None
    wake: Callable[[str], None] = lambda _sid: None
    memory_manager: MemoryManager = field(default_factory=MemoryManager)

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
        if not 0 < self.heartbeat_seconds <= 1 or self.ttl_ms <= self.heartbeat_seconds * 2000:
            raise ValueError("heartbeat must be <=1s and less than half the lease TTL")
        from vv_agent.runner import Runner

        self.config = Runner._effective_run_config(self.agent, self.config)
        self._dynamic_tools: dict[str, FunctionTool] = {}
        self._enabled_dynamic_tools: set[str] = set()
        self.hooks = RuntimeHookManager([*self.agent.hooks, *self.config.hooks])
        self.approval_broker = self.config.approval_broker
        if self.config.approval_provider is not None and self.approval_broker is None:
            from vv_agent.approval import ApprovalBroker

            self.approval_broker = ApprovalBroker()
        self._definition_key: tuple[str | bytes, str | bytes] | None = None
        self._definition_value: dict[str, Any] = {}
        self._definition_digest = ""
        self._retained_task_key: tuple[Record, str | bytes] | None = None
        self._definition_binding_key: tuple | None = None
        self._definition_binding: dict[str, Any] = {}
        self._definition_prefix = b""
        self._definition_suffix = b""
        self._schema_key: tuple | None = None
        self._schemas: list[dict[str, Any]] = []
        self._memory_clients: dict[tuple[str | None, str], tuple[LLMClient, ResolvedModelConfig]] = {}
        self._memory_routes: dict[str, tuple[LLMClient, ResolvedModelConfig]] = {}
        self._memory_binding_key: tuple | None = None
        self._memory_binding: dict[str, Any] = {}

    @cached_property
    def workspace_backend(self):
        return self.config.workspace_backend or LocalWorkspaceBackend(Path(self.config.workspace or "."))

    @cached_property
    def registry(self) -> ToolRegistry:
        # A read-only child result projection does not dispatch or plan tools.
        from vv_agent.runner import Runner

        registry = self.config.tool_registry_factory() if self.config.tool_registry_factory else build_default_registry()
        for candidate in self.agent.tools:
            if is_tool_executor(candidate):
                executor = candidate
            else:
                tool = candidate if isinstance(candidate, FunctionTool) else adapt_tool(candidate)
                if callable(tool.is_enabled):
                    self._dynamic_tools[tool.name] = tool
                if not Runner._tool_is_enabled(tool=tool, agent=self.agent, run_config=self.config):
                    continue
                executor = tool.to_executor()
                if tool.name in self._dynamic_tools:
                    self._enabled_dynamic_tools.add(tool.name)
            visible = executor.exposure == ToolExposure.DIRECT
            registry.register_executor(executor, expose_to_model=visible, planner_extra=visible)
        from .delegation import register_adapters

        register_adapters(self, registry)
        return registry

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
        from vv_agent.runner import Runner

        client, resolved = (
            Runner._resolve_model(agent=agent, run_config=config) if config.model_provider else (self.llm, self.resolved)
        )
        return Runtime(
            agent,
            config,
            resolved,
            client,
            self.heartbeat_store,
            providers=self.providers,
            children=self.children,
            host_bindings=self.host_bindings,
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
            model_timeout=self.model_timeout,
            tool_timeout=self.tool_timeout,
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
        return self._memory_routes.get(purpose, (self.llm, self.resolved))

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
            client, resolved = self._memory_routes[purpose] = self._memory_clients[client_key]
            bindings[purpose] = {
                "backend": resolved.backend,
                "model": resolved.model_id,
                "endpoints": [t.endpoint_id for t in client.endpoint_targets] if isinstance(client, VvLlmClient) else [],
            }
        self._memory_binding_key, self._memory_binding = key, bindings
        return bindings

    def complete(self, request: LlmRequest, attempt: int, stream_callback=None) -> LLMResponse:
        client, _ = self.model_route(request.metadata.get("purpose", "primary"))
        if isinstance(client, VvLlmClient):
            # One endpoint per logged attempt; never stack client fallback and kernel retries.
            targets = client.endpoint_targets
            if not targets:
                raise ValueError("No endpoint targets configured")
            client = copy(client)
            endpoint_id = request.metadata.get("vv_session", {}).get("endpoint_id")
            client.endpoint_targets = (
                [next(t for t in targets if t.endpoint_id == endpoint_id)]
                if endpoint_id
                else [targets[(attempt - 1) % len(targets)]]
            )
            client.max_retries_per_endpoint = 1
            client.randomize_endpoints = False
        if stream_callback is not None:
            return client.complete_with_stream(request, stream_callback)
        return client.complete(request)

    def compile(self, content: str, tid: str, *, seed: dict[str, Any] | None = None) -> AgentTask:
        from vv_agent.runner import Runner

        if "vv_session" in self.agent.metadata or "vv_session" in self.config.metadata:
            raise ValueError("vv_session is reserved for kernel metadata")
        if self.frozen_task is not None:
            task = deepcopy(self.frozen_task)
            task.task_id = tid
            if tid != self.frozen_task.task_id:
                task.user_prompt = content
            return task
        _ = self.registry
        for name, tool in self._dynamic_tools.items():
            enabled = Runner._tool_is_enabled(tool=tool, agent=self.agent, run_config=self.config)
            if enabled and name not in self._enabled_dynamic_tools:
                visible = tool.exposure == ToolExposure.DIRECT
                self.registry.register_executor(tool.to_executor(), expose_to_model=visible, planner_extra=visible)
                self._enabled_dynamic_tools.add(name)
            elif not enabled and name in self._enabled_dynamic_tools:
                self.registry.unregister(name)
                self._enabled_dynamic_tools.remove(name)
        guardrail = Runner._apply_input_guardrails(agent=self.agent, run_context=self.run_context(tid), user_input=content)
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
        metadata = dict(task.metadata)
        if plan._payload["op_kind"] != "model":
            frozen = ToolPolicy(
                allowed_tools=metadata.get("_vv_agent_allowed_tools"),
                disallowed_tools=metadata.get("_vv_agent_disallowed_tools", []),
                denied_side_effects=metadata.get("_vv_agent_denied_side_effects", []),
                denied_capability_tags=metadata.get("_vv_agent_denied_capability_tags", []),
                denied_cost_dimensions=metadata.get("_vv_agent_denied_cost_dimensions", []),
                deny_terminal_tools=metadata.get("_vv_agent_deny_terminal_tools", False),
                approval=metadata.get("_vv_agent_tool_policy_approval", "default"),
            )
            current = merge_tool_policies(self.agent.tool_policy, self.config.tool_policy)
            policy = merge_tool_policies(frozen, current)
            assert policy is not None
            if frozen.allowed_tools is not None and policy.allowed_tools is not None:
                policy.allowed_tools = [n for n in frozen.allowed_tools if n in policy.allowed_tools]
            if frozen.approval == "always":
                policy.approval = "always"
            from vv_agent.runtime.lifecycle import read_after_cycle_disallowed_tools

            policy.disallowed_tools = list(
                dict.fromkeys([*policy.disallowed_tools, *read_after_cycle_disallowed_tools(shared_state or {})])
            )
            _apply_tool_policy_metadata(metadata, policy)
            metadata["_vv_agent_tool_policy_approval"] = policy.approval
            metadata["_vv_agent_tool_policy_can_use_tool"] = policy.can_use_tool
        ctx = ExecutionContext(cancellation_token=token, metadata=metadata)
        if approved:
            ctx._approved_tool_approval = SimpleNamespace(call=ToolCall.from_dict(copy_json(plan._payload["request"])))
        workspace = Path(self.config.workspace or ".")
        source = plan.operation_id if plan._payload["op_kind"] == "model" else plan._payload["dependencies"][0]
        assert source is not None
        return ToolContext(
            workspace=workspace,
            shared_state=shared_state if shared_state is not None else dict(task.initial_shared_state),
            run_context=self.run_context(task.task_id),
            cycle_index=(
                plan._payload["request"]["metadata"]["vv_session"]["cycle_index"]
                if plan._payload["op_kind"] == "model"
                and "cycle_index" in plan._payload["request"].get("metadata", {}).get("vv_session", {})
                else int(source.rsplit("/", 1)[1])
            ),
            workspace_backend=self.workspace_backend,
            task_id=plan.turn_id or task.task_id,
            ctx=ctx,
            task_metadata=metadata,
            sub_task_manager=self.child_tasks(store, session_id).tool_manager()
            if store and session_id and plan._payload["request"].get("name") == "sub_task_status"
            else None,
            idempotency_key=plan._payload["idempotency_key"],
            metadata={
                "operation_id": plan.operation_id,
                "attempt": plan.attempt,
                "session_tool_names": [s["function"]["name"] for s in plan._payload["request"].get("tools", [])],
            },
        )

    def definition(self, task: AgentTask) -> dict[str, Any]:
        return copy_json(self._definition(task))

    def definition_digest(self, task: AgentTask | Record) -> str:
        self._definition(task)
        return self._definition_digest

    def _definition(self, task: AgentTask | Record) -> dict[str, Any]:
        # Retained tasks are immutable; mutable runtime bindings are checked on every step.
        retained = task if isinstance(task, Record) else None
        if retained is not None:
            task_value = retained._payload["definition"]["task"]
            task = retained._task()
        else:
            assert isinstance(task, AgentTask)
            task_value = task.to_dict()
        if retained is not None and self._retained_task_key is not None and self._retained_task_key[0] is retained:
            task_key = self._retained_task_key[1]
        else:
            try:
                task_key = json.dumps(task_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            except TypeError:
                task_key = canonical_json_bytes(task_value)
            if retained is not None:
                self._retained_task_key = (retained, task_key)
        memory_settings = {
            f.name: getattr(self.memory_manager, f.name)
            for f in fields(self.memory_manager)
            if f.name
            not in {
                "workspace_backend",
                "summary_callback",
                "session_memory",
                "microcompaction_policy",
                "recovery_tool_available",
            }
        }
        memory_settings["microcompaction_policy"] = self.memory_manager.microcompaction_policy.to_dict()
        model_binding: dict[str, Any] = {
            "backend": self.resolved.backend,
            "model": self.resolved.model_id,
            "endpoints": [t.endpoint_id for t in self.llm.endpoint_targets] if isinstance(self.llm, VvLlmClient) else [],
        }
        memory_bindings = self._memory_bindings(task)
        if memory_bindings:
            model_binding["internal"] = memory_bindings
        value: dict[str, Any] = {
            "agent_name": self.agent.name,
            "memory_settings": memory_settings,
            "child_tools": sorted(self.children),
            "model_binding": model_binding,
        }
        signature = self.registry.planning_signature()
        # JSON fingerprints distinguish True from 1 and detach mutable cache inputs.
        try:
            binding_key = json.dumps([value, signature], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except TypeError:
            binding_key = canonical_json_bytes([value, signature])
        key = (task_key, binding_key)
        if key == self._definition_key:
            return self._definition_value
        schema_key = (
            tuple(plan_tool_names(task)),
            signature,
            copy_json(
                [
                    task.metadata.get(control, [])
                    for control in (
                        "_vv_agent_denied_side_effects",
                        "_vv_agent_denied_capability_tags",
                        "_vv_agent_denied_cost_dimensions",
                    )
                ]
            ),
            task.metadata.get("_vv_agent_deny_terminal_tools", False),
        )
        if schema_key != self._schema_key:
            self._schemas = plan_tool_schemas(registry=self.registry, task=task, include_dynamic_hints=False)
            self._schema_key = schema_key
        if (binding_key, schema_key) != self._definition_binding_key:
            schemas = self._schemas
            names = [s["function"]["name"] for s in schemas]
            retentions = dict(self.memory_manager.tool_result_retentions)
            for name in names:
                metadata = self.registry.tool_metadata(name)
                if metadata is not None:
                    retentions[name] = (
                        ToolResultRetention.PRESERVE
                        if ToolResultRetention.PRESERVE in (retentions.get(name), metadata.result_retention)
                        else metadata.result_retention
                    )
            memory_settings["tool_result_retentions"] = retentions
            value["capabilities"] = {
                n: metadata.to_dict()
                for n in names
                if self.registry.has_executor(n) and (metadata := self.registry.tool_metadata(n)) is not None
            }
            # These closed keys precede task, and tools follows it in JCS order.
            prefix = canonical_json_bytes(value)[:-1] + b',"task":'
            suffix = b',"tools":' + canonical_json_bytes(schemas) + b"}"
            self._definition_binding = value | {"tools": schemas}
            self._definition_prefix, self._definition_suffix = prefix, suffix
            self._definition_binding_key = (binding_key, schema_key)
        task_bytes = canonical_json_bytes(task_value)
        value = self._definition_binding | {"task": task_value}
        self._definition_digest = sha256(self._definition_prefix + task_bytes + self._definition_suffix).hexdigest()
        self._definition_value = value
        self._definition_key = key
        return value


def request_from_dict(value: dict[str, Any]) -> LlmRequest:
    return LlmRequest(
        model=value["model"],
        messages=[Message.from_dict(m) for m in value["messages"]],
        tools=value["tools"],
        metadata=value["metadata"],
        prompt_bundle=PromptBundle.from_dict(value["prompt_bundle"]) if value["prompt_bundle"] else None,
        model_settings=ModelSettings.from_dict(value["model_settings"]) if value["model_settings"] else None,
    )


def model_usage(value: dict[str, Any] | None):
    # Framework receipt annotations are not provider token-usage fields.
    return normalize_token_usage({k: v for k, v in (value or {}).items() if not k.startswith("session_")})


def budget(state: ExecutionState, tid: str) -> BudgetEvaluator | None:
    parsed = RunBudgetLimits.from_dict(state.turns[tid].start._payload["budget"])
    if not parsed.has_limits:
        return None
    observations = [r._payload["usage"] for r in state.usage_values.values() if r.turn_id == tid]
    elapsed = sum(v.get("elapsed_ms", 0) for v in observations)
    boundaries = [r._payload["data"] for (turn, stage, _), r in state.boundaries.items() if turn == tid and stage == "budget"]
    previous = BudgetUsageSnapshot.from_dict(boundaries[-1]["usage"]) if boundaries else BudgetUsageSnapshot()
    missing_interval = any(
        a.unknown and a.unknown._payload["observation"].get("active_interval_missing")
        for op in state.operations.values()
        if op.turn_id == tid
        for a in op.attempts.values()
    )
    unavailable = list(previous.unavailable_dimensions)
    if missing_interval and not any(v.dimension == BudgetDimension.WALL_TIME for v in unavailable):
        unavailable.append(BudgetUnavailableDimension(BudgetDimension.WALL_TIME, BudgetUnavailableReason.ACCOUNTING_MISSING))
    started = [(op.kind, a) for op in state.operations.values() if op.turn_id == tid for a in op.attempts.values() if a.started]
    tool_counts = Counter(name for b in boundaries for name in b["tool_names"])
    # Host readings and unavailable classifications are retained at every active-scope commit.
    host = observations[-1].get("host_cost") if observations else None
    evaluator = BudgetEvaluator(
        parsed,
        initial_usage=BudgetUsageSnapshot(
            cycles=sum(kind == "model" and a.plan._payload["purpose"] == "primary" for kind, a in started),
            tool_calls=sum(tool_counts.values()),
            tool_calls_by_name=tool_counts,
            elapsed_ms=elapsed,
            host_cost=previous.host_cost,
            unavailable_dimensions=tuple(unavailable),
        ),
        host_cost_meter=cast(HostCostMeter, SimpleNamespace(read=lambda: HostCost.from_dict(host) if host else None)),
        clock_ns=lambda: 0,
    )
    for kind, attempt in started:
        if kind == "model" and (attempt.result is not None or attempt.unknown is not None):
            evaluator._observe_token_usage(model_usage(attempt.result._payload["usage"] if attempt.result else {}))
    return evaluator
