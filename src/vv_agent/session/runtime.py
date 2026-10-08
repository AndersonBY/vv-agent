"""Bindings to existing runtime components; no checkpoint controller or second executor."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from vv_agent.agent import Agent
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
from vv_agent.config import ResolvedModelConfig
from vv_agent.llm.base import LLMClient, LlmRequest
from vv_agent.llm.vv_llm_client import VvLlmClient
from vv_agent.memory import MemoryManager
from vv_agent.model_settings import ModelSettings, RetrySettings
from vv_agent.prompt import PromptBundle
from vv_agent.run_config import RunConfig, ToolPolicy, merge_tool_policies
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.runtime.compiler import AgentCompiler, _apply_tool_policy_metadata
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.token_usage import normalize_token_usage
from vv_agent.runtime.tool_planner import plan_tool_schemas
from vv_agent.tools.base import ToolContext
from vv_agent.tools.builtins import build_default_registry
from vv_agent.tools.function import adapt_tool
from vv_agent.tools.metadata import ToolResultRetention
from vv_agent.tools.orchestrator import ToolOrchestrator
from vv_agent.types import AgentTask, Message, ToolCall
from vv_agent.workspace.local import LocalWorkspaceBackend

from .children import ChildSession
from .providers import FunctionProvider, Provider
from .records import Record
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
        if isinstance(self.llm, VvLlmClient) and len(self.llm.endpoint_targets) != 1:
            raise ValueError("session kernel requires one bound model endpoint; retries belong to the log")
        self.registry = build_default_registry()
        for tool in self.agent.tools:
            self.registry.register_executor(adapt_tool(tool).to_executor())
        self.functions = FunctionProvider(ToolOrchestrator.from_registry(self.registry))

    def compile(self, content: str, tid: str) -> AgentTask:
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

        task.model_settings = replace(
            task.model_settings or ModelSettings(), retry=RetrySettings(max_attempts=1, backoff_seconds=0)
        )
        return task

    def context(self, task: AgentTask, plan: Record, token: CancellationToken, *, approved: bool = False) -> ToolContext:
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
        _apply_tool_policy_metadata(metadata, policy)
        if policy is not None:
            metadata["_vv_agent_tool_policy_approval"] = policy.approval
            metadata["_vv_agent_tool_policy_can_use_tool"] = policy.can_use_tool
        ctx = ExecutionContext(cancellation_token=token, metadata=metadata)
        if approved:
            ctx._approved_tool_approval = SimpleNamespace(call=ToolCall.from_dict(plan.payload["request"]))
        workspace = Path(self.config.workspace or ".")
        source = plan.operation_id if plan.payload["op_kind"] == "model" else plan.payload["dependencies"][0]
        assert source is not None
        return ToolContext(
            workspace=workspace,
            shared_state=dict(task.initial_shared_state),
            cycle_index=(
                plan.payload["request"]["metadata"]["cycle_index"]
                if plan.payload["purpose"] == "compaction"
                else int(source.rsplit("/", 1)[1])
            ),
            workspace_backend=self.config.workspace_backend or LocalWorkspaceBackend(workspace),
            task_id=plan.turn_id or task.task_id,
            ctx=ctx,
            task_metadata=metadata,
            idempotency_key=plan.payload["idempotency_key"],
            metadata={
                "operation_id": plan.operation_id,
                "attempt": plan.attempt,
                "session_tool_names": [s["function"]["name"] for s in self.definition(task)["tools"]],
            },
        )

    def definition(self, task: AgentTask) -> dict[str, Any]:
        # Freeze schema + capabilities with the task; current bindings are checked on resume.
        names = [*task.extra_tool_names, "ask_user", *(["read_file"] if task.use_workspace else [])]
        schemas = [
            s
            for s in plan_tool_schemas(registry=self.registry, task=task, include_dynamic_hints=False)
            if s["function"]["name"] in names
        ]
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
        return {
            "memory_settings": memory_settings,
            "task": task.to_dict(),
            "child_tools": sorted(self.children),
            "tools": schemas,
            "model_binding": {
                "backend": self.resolved.backend,
                "model": self.resolved.model_id,
                "endpoints": [t.endpoint_id for t in self.llm.endpoint_targets] if isinstance(self.llm, VvLlmClient) else [],
            },
            "capabilities": {
                n: metadata.to_dict()
                for n in names
                if self.registry.has_executor(n) and (metadata := self.registry.tool_metadata(n)) is not None
            },
        }


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
    limits = state.turns[tid].start.payload["budget"]
    parsed = RunBudgetLimits.from_dict(limits)
    if not parsed.has_limits:
        return None
    observations = [r.payload["usage"] for r in state.usage_values.values() if r.turn_id == tid]
    elapsed = sum(v.get("elapsed_ms", 0) for v in observations)
    missing_interval = any(
        a.unknown and a.unknown.payload["observation"].get("active_interval_missing")
        for op in state.operations.values()
        if op.turn_id == tid
        for a in op.attempts.values()
    )
    unavailable = (
        (BudgetUnavailableDimension(BudgetDimension.WALL_TIME, BudgetUnavailableReason.ACCOUNTING_MISSING),)
        if missing_interval
        else ()
    )
    host = observations[-1].get("host_cost") if observations else None
    started = [(op.kind, a) for op in state.operations.values() if op.turn_id == tid for a in op.attempts.values() if a.started]
    tool_counts = Counter(a.plan.payload["request"]["name"] for kind, a in started if kind != "model")
    evaluator = BudgetEvaluator(
        parsed,
        initial_usage=BudgetUsageSnapshot(
            cycles=sum(kind == "model" and a.plan.payload["purpose"] == "primary" for kind, a in started),
            tool_calls=sum(tool_counts.values()),
            tool_calls_by_name=tool_counts,
            elapsed_ms=elapsed,
            host_cost=HostCost.from_dict(host) if host else None,
            unavailable_dimensions=unavailable,
        ),
        host_cost_meter=cast(HostCostMeter, SimpleNamespace(read=lambda: HostCost.from_dict(host) if host else None)),
        clock_ns=lambda: 0,
    )
    for kind, attempt in started:
        if kind == "model":
            usage = attempt.result.payload["usage"] if attempt.result else {}
            evaluator.model_call_complete(normalize_token_usage(usage))
    return evaluator
