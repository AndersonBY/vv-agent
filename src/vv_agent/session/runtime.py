"""Bindings to existing runtime components; no checkpoint controller or second executor."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from copy import copy
from dataclasses import dataclass, field, fields, replace
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
from vv_agent.types import AgentTask, LLMResponse, Message, ToolCall
from vv_agent.workspace.local import LocalWorkspaceBackend

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
    handler_version: str = "1"
    ttl_ms: int = 15000
    heartbeat_seconds: float = 0.25
    poll_ms: int = 1000
    cancellation_grace: float = 0.25
    model_timeout: float = 300
    tool_timeout: float = 300
    hook: Callable[[str, Record | None], None] = lambda _point, _record: None
    wake: Callable[[str], None] = lambda _sid: None
    memory_manager: MemoryManager = field(default_factory=MemoryManager)

    def __post_init__(self) -> None:
        if not 0 < self.heartbeat_seconds <= 1 or self.ttl_ms <= self.heartbeat_seconds * 2000:
            raise ValueError("heartbeat must be <=1s and less than half the lease TTL")
        from vv_agent.runner import Runner

        self.config = Runner._effective_run_config(self.agent, self.config)
        self.registry = self.config.tool_registry_factory() if self.config.tool_registry_factory else build_default_registry()
        self._dynamic_tools: dict[str, FunctionTool] = {}
        self._enabled_dynamic_tools: set[str] = set()
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
            self.registry.register_executor(executor, expose_to_model=visible, planner_extra=visible)
        self.functions = FunctionProvider(ToolOrchestrator.from_registry(self.registry))
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

    def run_context(self, tid: str) -> RunContext:
        return RunContext(
            context=self.config.context,
            run_id=tid,
            agent_name=self.agent.name,
            model=self.resolved.model_id,
            workspace=self.config.workspace,
            metadata={**self.agent.metadata, **self.config.metadata},
        )

    def complete(self, request: LlmRequest, attempt: int, stream_callback=None) -> LLMResponse:
        client = self.llm
        if isinstance(client, VvLlmClient):
            # One endpoint per logged attempt; never stack client fallback and kernel retries.
            targets = client.endpoint_targets
            if not targets:
                raise ValueError("No endpoint targets configured")
            client = copy(client)
            endpoint_id = request.metadata.get("session_endpoint_id")
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

    def compile(self, content: str, tid: str) -> AgentTask:
        from vv_agent.runner import Runner

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
                metadata={"session_input_blocked": guardrail.message or "Input blocked by guardrail."},
            )
        policy = merge_tool_policies(self.agent.tool_policy, self.config.tool_policy)
        task = AgentCompiler().compile(
            agent=self.agent,
            input=content,
            run_config=replace(self.config, tool_policy=policy),
            resolved=self.resolved,
            trace_id=tid,
            run_id=tid,
        )
        task.metadata["_vv_agent_tool_policy_approval"] = policy.approval if policy else "default"

        defaults = self.config.model_provider.default_settings(self.resolved) if self.config.model_provider else ModelSettings()
        task.model_settings = replace(
            defaults.resolve(task.model_settings), retry=RetrySettings(max_attempts=1, backoff_seconds=0)
        )
        if task.metadata.get("session_memory_enabled"):
            from .memory import session_memory

            memory = session_memory(task, Path(self.config.workspace or ".") if task.use_workspace else None)
            memory.load()
            task.metadata["_vv_agent_session_memory_initial_state"] = memory.state.to_dict()
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
    ) -> ToolContext:
        metadata = dict(task.metadata)
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
        if policy is not None:
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
                plan._payload["request"]["metadata"]["cycle_index"]
                if plan._payload["op_kind"] == "model" and "cycle_index" in plan._payload["request"].get("metadata", {})
                else int(source.rsplit("/", 1)[1])
            ),
            workspace_backend=self.config.workspace_backend or LocalWorkspaceBackend(workspace),
            task_id=plan.turn_id or task.task_id,
            ctx=ctx,
            task_metadata=metadata,
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
        value: dict[str, Any] = {
            "agent_name": self.agent.name,
            "memory_settings": memory_settings,
            "child_tools": sorted(self.children),
            "model_binding": {
                "backend": self.resolved.backend,
                "model": self.resolved.model_id,
                "endpoints": [t.endpoint_id for t in self.llm.endpoint_targets] if isinstance(self.llm, VvLlmClient) else [],
            },
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
