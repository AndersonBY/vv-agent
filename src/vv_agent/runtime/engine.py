from __future__ import annotations

import ast
import json
import logging
import uuid
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from vv_agent.approval import ApprovalError
from vv_agent.budget import (
    BudgetEnforcementBoundary,
    BudgetEvaluator,
    BudgetExhaustion,
    BudgetUsageSnapshot,
    HostCostMeter,
    RunBudgetLimits,
)
from vv_agent.checkpoint import utf16_sort_key
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig, build_openai_llm_from_local_settings
from vv_agent.constants import CREATE_SUB_TASK_TOOL_NAME, SUB_TASK_STATUS_TOOL_NAME, TASK_FINISH_TOOL_NAME
from vv_agent.events import (
    BudgetExhaustedEvent,
    BudgetSnapshotEvent,
    RunEvent,
    SubRunCompletedEvent,
    SubRunStartedEvent,
)
from vv_agent.llm.base import LLMClient, LlmRequest, complete_llm_request
from vv_agent.memory import MemoryManager, SessionMemory, SessionMemoryConfig
from vv_agent.memory.token_utils import resolve_model_token_limits
from vv_agent.model_settings import ModelSettings
from vv_agent.prompt import build_raw_system_prompt_sections, build_system_prompt_bundle
from vv_agent.runtime.backends.base import ExecutionBackend
from vv_agent.runtime.backends.inline import InlineBackend
from vv_agent.runtime.cancellation import CancellationToken, CancelledError
from vv_agent.runtime.checkpoint_resume import (
    CheckpointReconciliationRequired,
    CheckpointResumeController,
)
from vv_agent.runtime.context import ExecutionContext, StreamCallback
from vv_agent.runtime.cycle_runner import CycleRunner
from vv_agent.runtime.hooks import RuntimeHook, RuntimeHookManager
from vv_agent.runtime.lifecycle import (
    AfterCycleAction,
    AfterCycleDecision,
    AfterCycleHook,
    AfterCycleHookError,
    AfterCycleHookManager,
    AfterCycleSnapshot,
    NativeCycleOutcome,
    NativeCycleOutcomeKind,
    persist_after_cycle_disallowed_tools,
    read_after_cycle_disallowed_tools,
)
from vv_agent.runtime.sub_task_identity import normalize_identity_string, take_sub_task_identity
from vv_agent.runtime.sub_task_manager import (
    _TURN_LOG_HANDLER_METADATA_KEY,
    SubTaskManager,
    _SubTaskTurnSnapshot,
)
from vv_agent.runtime.token_usage import summarize_task_token_usage
from vv_agent.runtime.tool_call_runner import ToolCallRunner, _ConfiguredSubTaskCancelledError
from vv_agent.runtime.tool_planner import freeze_dynamic_tool_schema_hints, plan_tool_names
from vv_agent.tools import ToolContext, ToolRegistry
from vv_agent.tools.metadata import normalize_denied_side_effects, normalize_metadata_labels
from vv_agent.types import (
    AgentResult,
    AgentStatus,
    AgentTask,
    CompletionReason,
    CycleRecord,
    Message,
    SubAgentConfig,
    SubTaskOutcome,
    SubTaskRequest,
    ToolCall,
    ToolDirective,
    ToolExecutionResult,
    _last_assistant_output,
    _trim_portable_whitespace,
)
from vv_agent.workspace import (
    INVALID_EXCLUDE_FILES_PATTERN_CODE,
    INVALID_EXCLUDE_FILES_PATTERN_MESSAGE,
    DiscoveryFilteredWorkspaceBackend,
    InvalidPortableRegexError,
    LocalWorkspaceBackend,
    WorkspaceBackend,
)

RuntimeLogHandler = Callable[[str, dict[str, Any]], None]
RunEventHandler = Callable[[RunEvent], None]
BeforeCycleMessageProvider = Callable[[int, list[Message], dict[str, Any]], list[Message]]
InterruptionMessageProvider = Callable[[], list[Message]]

_ACTIVE_SUB_AGENT_SESSIONS_LOCK = RLock()
_ACTIVE_SUB_AGENT_SESSIONS: dict[str, Any] = {}

_INVALID_SUB_AGENT_MODEL_CODE = "invalid_sub_agent_model"
_INVALID_SUB_AGENT_MODEL_MESSAGE = "sub-agent model cannot be empty"
_INVALID_SUB_AGENT_SYSTEM_PROMPT_CODE = "invalid_sub_agent_system_prompt"
_INVALID_SUB_AGENT_SYSTEM_PROMPT_MESSAGE = "sub-agent system_prompt cannot be empty when provided"
_TOOL_POLICY_METADATA_KEYS = (
    "_vv_agent_allowed_tools",
    "_vv_agent_disallowed_tools",
    "_vv_agent_tool_policy_approval",
    "_vv_agent_tool_policy_can_use_tool",
    "_vv_agent_denied_side_effects",
    "_vv_agent_denied_capability_tags",
    "_vv_agent_deny_terminal_tools",
    "_vv_agent_denied_cost_dimensions",
)
_RESERVED_SUB_AGENT_METADATA_KEYS = (
    "browser_scope_key",
    "is_sub_task",
    "parent_task_id",
    "session_id",
    "session_memory_enabled",
    "sub_agent_name",
    "task_id",
    "workspace",
    "run_id",
    "trace_id",
    "parent_run_id",
    "parent_tool_call_id",
    "_vv_agent_run_id",
    "_vv_agent_trace_id",
    "_vv_agent_agent_name",
    "_vv_agent_session_id",
    "_vv_agent_parent_run_id",
    "_vv_agent_parent_tool_call_id",
    *_TOOL_POLICY_METADATA_KEYS,
)
_SUB_AGENT_STREAM_PRODUCER_FIELDS = {
    "assistant_delta": frozenset({"content_chars", "content_delta", "delta", "estimated_tokens", "event"}),
    "reasoning_delta": frozenset({"estimated_tokens", "event", "reasoning_chars", "reasoning_delta"}),
    "tool_call_started": frozenset(
        {"arguments_chars", "estimated_tokens", "event", "function_name", "tool_call_id", "tool_call_index"}
    ),
    "tool_call_progress": frozenset(
        {"arguments_chars", "estimated_tokens", "event", "function_name", "tool_call_id", "tool_call_index"}
    ),
}


class _CanonicalSubAgentStreamPayload(dict[str, Any]):
    """Marks a payload whose child identity was written by the runtime."""


def _enrich_sub_agent_payload(
    payload: dict[str, Any],
    *,
    task_id: str,
    session_id: str,
    sub_agent_name: str,
) -> dict[str, Any]:
    enriched = dict(payload)
    enriched["task_id"] = task_id
    enriched["session_id"] = session_id
    enriched["sub_agent_name"] = sub_agent_name
    return enriched


def _canonicalize_sub_agent_stream_event(
    payload: dict[str, Any],
    *,
    task_id: str,
    session_id: str,
    sub_agent_name: str,
    child_run_id: str,
    trace_id: str,
    parent_run_id: str,
    parent_tool_call_id: str,
) -> _CanonicalSubAgentStreamPayload | None:
    event_name = payload.get("event")
    if not isinstance(event_name, str) or event_name not in _SUB_AGENT_STREAM_PRODUCER_FIELDS:
        return None
    canonical = _CanonicalSubAgentStreamPayload(
        {
            key: value
            for key, value in payload.items()
            if key in _SUB_AGENT_STREAM_PRODUCER_FIELDS[event_name]
        }
    )
    canonical.update(
        {
            "event": event_name,
            "agent_name": sub_agent_name,
            "child_run_id": child_run_id,
            "child_session_id": session_id,
            "parent_run_id": parent_run_id,
            "parent_tool_call_id": parent_tool_call_id,
            "run_id": child_run_id,
            "session_id": session_id,
            "sub_agent_name": sub_agent_name,
            "task_id": task_id,
            "trace_id": trace_id,
        }
    )
    cycle_index = payload.get("cycle")
    if isinstance(cycle_index, int) and not isinstance(cycle_index, bool) and cycle_index > 0:
        canonical["cycle_index"] = cycle_index
    return canonical


class _SubTaskContractError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _RunBudgetController:
    def __init__(
        self,
        *,
        evaluator: BudgetEvaluator,
        task: AgentTask,
        ctx: ExecutionContext | None,
        emit_log: Callable[..., None],
    ) -> None:
        self.evaluator = evaluator
        self.task = task
        self.ctx = ctx
        self.emit_log = emit_log
        self.exhaustion: BudgetExhaustion | None = None
        self._last_emitted_snapshot: BudgetUsageSnapshot | None = None

    @property
    def snapshot(self) -> BudgetUsageSnapshot:
        return self.evaluator.snapshot()

    def run_start(self) -> BudgetExhaustion | None:
        return self._observe(
            BudgetEnforcementBoundary.RUN_START,
            self.evaluator.run_start,
            force_snapshot=True,
        )

    def cycle_start(self, cycle_index: int) -> BudgetExhaustion | None:
        return self._observe(
            BudgetEnforcementBoundary.CYCLE_START,
            self.evaluator.cycle_start,
            cycle_index=cycle_index,
        )

    def llm_complete(
        self,
        cycle_index: int,
        token_usage: Any,
        *,
        suppress_exhaustion: bool = False,
    ) -> BudgetExhaustion | None:
        return self._observe(
            BudgetEnforcementBoundary.LLM_COMPLETE,
            lambda: self.evaluator.llm_complete(token_usage),
            cycle_index=cycle_index,
            suppress_exhaustion=suppress_exhaustion,
        )

    def preflight_tools(self, cycle_index: int, tool_names: list[str]) -> BudgetExhaustion | None:
        return self._observe(
            BudgetEnforcementBoundary.TOOL_BATCH_PREFLIGHT,
            lambda: self.evaluator.preflight_tools(tool_names),
            cycle_index=cycle_index,
        )

    def tool_batch_complete(
        self,
        cycle_index: int,
        *,
        operation_failed: bool = False,
        suppress_exhaustion: bool = False,
    ) -> BudgetExhaustion | None:
        return self._observe(
            BudgetEnforcementBoundary.TOOL_BATCH_COMPLETE,
            lambda: self.evaluator.tool_batch_complete(operation_failed=operation_failed),
            cycle_index=cycle_index,
            suppress_exhaustion=operation_failed or suppress_exhaustion,
        )

    def terminal(self, *, suppress_exhaustion: bool = False) -> BudgetExhaustion | None:
        return self._observe(
            BudgetEnforcementBoundary.TERMINAL,
            self.evaluator.terminal,
            suppress_exhaustion=suppress_exhaustion,
        )

    def _observe(
        self,
        boundary: BudgetEnforcementBoundary,
        operation: Callable[[], BudgetExhaustion | None],
        *,
        cycle_index: int | None = None,
        force_snapshot: bool = False,
        suppress_exhaustion: bool = False,
    ) -> BudgetExhaustion | None:
        if self.exhaustion is not None:
            return self.exhaustion
        exhaustion = operation()
        snapshot = self.evaluator.snapshot()
        if exhaustion is not None and not suppress_exhaustion:
            self.exhaustion = exhaustion
            event = BudgetExhaustedEvent(
                run_id=self._identity("_vv_agent_run_id", "run_id") or self.task.task_id,
                trace_id=self._identity("_vv_agent_trace_id", "trace_id") or self.task.task_id,
                agent_name=self._identity("_vv_agent_agent_name", "agent_name"),
                session_id=self._identity("_vv_agent_session_id", "session_id"),
                parent_run_id=self._identity("_vv_agent_parent_run_id", "parent_run_id"),
                cycle_index=cycle_index,
                enforcement_boundary=boundary,
                budget_usage=snapshot,
                budget_exhaustion=exhaustion,
            )
            self._emit_event(event)
            self._last_emitted_snapshot = snapshot
            return exhaustion
        if force_snapshot or snapshot != self._last_emitted_snapshot:
            event = BudgetSnapshotEvent(
                run_id=self._identity("_vv_agent_run_id", "run_id") or self.task.task_id,
                trace_id=self._identity("_vv_agent_trace_id", "trace_id") or self.task.task_id,
                agent_name=self._identity("_vv_agent_agent_name", "agent_name"),
                session_id=self._identity("_vv_agent_session_id", "session_id"),
                parent_run_id=self._identity("_vv_agent_parent_run_id", "parent_run_id"),
                cycle_index=cycle_index,
                enforcement_boundary=boundary,
                budget_usage=snapshot,
            )
            self._emit_event(event)
            self._last_emitted_snapshot = snapshot
        return None

    def _identity(self, *keys: str) -> str | None:
        if self.ctx is None:
            return None
        return AgentRuntime._metadata_str(self.ctx.metadata, *keys)

    def _emit_event(self, event: RunEvent) -> None:
        if self.ctx is not None:
            emitter = self.ctx.metadata.get("_vv_agent_emit_event")
            if callable(emitter):
                emitter(event)
        payload = event.to_dict()
        payload.pop("type", None)
        self.emit_log(event.type, **payload)


def _parse_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def steer_sub_agent_session(*, session_id: str, prompt: str) -> bool:
    """Steer a running sub-agent session by its sub-session id."""
    normalized_session_id = str(session_id or "").strip()
    normalized_prompt = str(prompt or "").strip()
    if not normalized_session_id or not normalized_prompt:
        return False
    with _ACTIVE_SUB_AGENT_SESSIONS_LOCK:
        session = _ACTIVE_SUB_AGENT_SESSIONS.get(normalized_session_id)
    if session is None:
        return False
    session.steer(normalized_prompt)
    return True


def get_sub_agent_session(*, session_id: str) -> Any | None:
    normalized_session_id = str(session_id or "").strip()
    if not normalized_session_id:
        return None
    with _ACTIVE_SUB_AGENT_SESSIONS_LOCK:
        return _ACTIVE_SUB_AGENT_SESSIONS.get(normalized_session_id)


def subscribe_sub_agent_session(
    *,
    session_id: str,
    listener: Callable[[str, dict[str, Any]], None],
) -> Callable[[], None] | None:
    session = get_sub_agent_session(session_id=session_id)
    if session is None:
        return None
    return session.subscribe(listener)


def register_sub_agent_session(session_id: str, session: Any) -> None:
    normalized_session_id = str(session_id or "").strip()
    if not normalized_session_id:
        return
    with _ACTIVE_SUB_AGENT_SESSIONS_LOCK:
        _ACTIVE_SUB_AGENT_SESSIONS[normalized_session_id] = session


def unregister_sub_agent_session(session_id: str, session: Any | None = None) -> None:
    normalized_session_id = str(session_id or "").strip()
    if not normalized_session_id:
        return
    with _ACTIVE_SUB_AGENT_SESSIONS_LOCK:
        registered = _ACTIVE_SUB_AGENT_SESSIONS.get(normalized_session_id)
        if registered is None:
            return
        if session is not None and registered is not session:
            return
        _ACTIVE_SUB_AGENT_SESSIONS.pop(normalized_session_id, None)


def _register_sub_agent_session(session_id: str, session: Any) -> None:
    register_sub_agent_session(session_id, session)


def _unregister_sub_agent_session(session_id: str, session: Any | None = None) -> None:
    unregister_sub_agent_session(session_id, session)


class LLMBuilder(Protocol):
    def __call__(
        self,
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[LLMClient, ResolvedModelConfig]: ...


class AgentRuntime:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        default_workspace: str | Path | None = None,
        log_handler: RuntimeLogHandler | None = None,
        log_preview_chars: int | None = None,
        settings_file: str | Path | None = None,
        default_backend: str | None = None,
        llm_builder: LLMBuilder | None = None,
        tool_registry_factory: Callable[[], ToolRegistry] | None = None,
        sub_agent_timeout_seconds: float = 90.0,
        hooks: list[RuntimeHook] | None = None,
        after_cycle_hooks: list[AfterCycleHook] | None = None,
        execution_backend: ExecutionBackend | None = None,
        workspace_backend: WorkspaceBackend | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.default_workspace = Path(default_workspace).resolve() if default_workspace else None
        self.log_handler = log_handler
        if log_preview_chars is None:
            self.log_preview_chars: int | None = None
        else:
            self.log_preview_chars = max(int(log_preview_chars), 40)
        self.settings_file = Path(settings_file).resolve() if settings_file else None
        self.default_backend = default_backend
        self.llm_builder = llm_builder or build_openai_llm_from_local_settings
        self.tool_registry_factory = tool_registry_factory
        self.sub_agent_timeout_seconds = max(sub_agent_timeout_seconds, 1.0)
        self.hook_manager = RuntimeHookManager(hooks=list(hooks or []))
        self.after_cycle_hook_manager = AfterCycleHookManager(
            hooks=list(after_cycle_hooks or [])
        )
        self.execution_backend: ExecutionBackend = execution_backend or InlineBackend()
        self._workspace_backend = workspace_backend
        self._memory_summary_clients: dict[tuple[str, str], LLMClient] = {}
        self._memory_summary_defaults: tuple[str | None, str | None] | None = None
        self.cycle_runner = CycleRunner(
            llm_client=llm_client,
            tool_registry=tool_registry,
            hook_manager=self.hook_manager,
        )
        self.tool_call_runner = ToolCallRunner(
            tool_registry=tool_registry,
            hook_manager=self.hook_manager,
        )

    def run(
        self,
        task: AgentTask,
        *,
        workspace: str | Path | None = None,
        shared_state: dict[str, Any] | None = None,
        initial_messages: list[Message] | None = None,
        prepared_initial_messages: list[Message] | None = None,
        user_message: str | None = None,
        before_cycle_messages: BeforeCycleMessageProvider | None = None,
        interruption_messages: InterruptionMessageProvider | None = None,
        ctx: ExecutionContext | None = None,
        sub_task_manager: SubTaskManager | None = None,
        budget_limits: RunBudgetLimits | None = None,
        host_cost_meter: HostCostMeter | None = None,
        initial_budget_usage: BudgetUsageSnapshot | None = None,
        checkpoint_controller: CheckpointResumeController | None = None,
    ) -> AgentResult:
        workspace_path = self._prepare_workspace(workspace)
        had_configured_disallowed_tools = "_vv_agent_disallowed_tools" in task.metadata
        configured_disallowed_tools = deepcopy(
            task.metadata.get("_vv_agent_disallowed_tools")
        )
        effective_shared_state = shared_state if shared_state is not None else task.initial_shared_state
        shared = dict(effective_shared_state)
        shared.setdefault("todo_list", [])
        if isinstance(task.metadata, dict):
            if "available_skills" not in shared and task.metadata.get("available_skills") is not None:
                shared["available_skills"] = task.metadata.get("available_skills")
            if "active_skills" not in shared and task.metadata.get("active_skills") is not None:
                shared["active_skills"] = list(task.metadata.get("active_skills") or [])

        messages = (
            [self._copy_message(message) for message in prepared_initial_messages]
            if prepared_initial_messages is not None
            else self._build_initial_messages(
                task=task,
                initial_messages=(
                    initial_messages
                    if initial_messages is not None
                    else (list(task.initial_messages) or None)
                ),
                user_message=user_message,
            )
        )
        freeze_dynamic_tool_schema_hints(task)
        runtime_ctx = ctx if ctx is not None else ExecutionContext()
        runtime_ctx.metadata.setdefault("execution_backend", self.execution_backend)
        effective_budget_limits = budget_limits
        if effective_budget_limits is None:
            metadata_limits = runtime_ctx.metadata.get("_vv_agent_budget_limits")
            if isinstance(metadata_limits, RunBudgetLimits):
                effective_budget_limits = metadata_limits
            elif isinstance(metadata_limits, dict):
                effective_budget_limits = RunBudgetLimits.from_dict(metadata_limits)
        if initial_budget_usage is None:
            metadata_usage = runtime_ctx.metadata.get("_vv_agent_initial_budget_usage")
            if isinstance(metadata_usage, BudgetUsageSnapshot):
                initial_budget_usage = metadata_usage
            elif isinstance(metadata_usage, dict):
                initial_budget_usage = BudgetUsageSnapshot.from_dict(metadata_usage)
        if host_cost_meter is None:
            metadata_meter = runtime_ctx.metadata.get("_vv_agent_host_cost_meter")
            if callable(getattr(metadata_meter, "read", None)):
                host_cost_meter = metadata_meter

        self._emit_log(
            "run_started",
            task_id=task.task_id,
            model=task.model,
            workspace=str(workspace_path),
            max_cycles=task.max_cycles,
        )

        budget_controller: _RunBudgetController | None = None
        backend_manages_budget = bool(getattr(self.execution_backend, "manages_run_budget", False))
        if effective_budget_limits is not None and effective_budget_limits.has_limits:
            runtime_ctx.metadata["_vv_agent_budget_limits"] = effective_budget_limits
            runtime_ctx.metadata["_vv_agent_initial_budget_usage"] = initial_budget_usage
            if host_cost_meter is not None:
                runtime_ctx.metadata["_vv_agent_host_cost_meter"] = host_cost_meter
            if not backend_manages_budget:
                budget_controller = _RunBudgetController(
                    evaluator=BudgetEvaluator(
                        effective_budget_limits,
                        host_cost_meter=host_cost_meter,
                        initial_usage=initial_budget_usage,
                    ),
                    task=task,
                    ctx=runtime_ctx,
                    emit_log=self._emit_log,
                )

        result: AgentResult | None = None
        if budget_controller is not None:
            cancelled = bool(
                runtime_ctx.cancellation_token is not None and runtime_ctx.cancellation_token.cancelled
            )
            if cancelled:
                result = AgentResult(
                    status=AgentStatus.FAILED,
                    completion_reason=CompletionReason.CANCELLED,
                    messages=messages,
                    cycles=[],
                    error=runtime_ctx.cancellation_token.reason or "Operation was cancelled",
                    shared_state=shared,
                    token_usage=summarize_task_token_usage([]),
                    budget_usage=budget_controller.snapshot,
                )
            else:
                exhaustion = budget_controller.run_start()
                if exhaustion is not None:
                    result = self._budget_failure_result(
                        messages=messages,
                        cycles=[],
                        shared_state=shared,
                        controller=budget_controller,
                        exhaustion=exhaustion,
                    )

        if result is None:
            self._emit_log("agent_started", model=task.model)

        if checkpoint_controller is not None:
            runtime_ctx.metadata["_vv_agent_checkpoint_controller"] = checkpoint_controller
            runtime_ctx.metadata["_vv_agent_checkpoint_budget_snapshot"] = (
                (lambda: budget_controller.snapshot) if budget_controller is not None else (lambda: None)
            )

        memory_manager = self._build_memory_manager(
            task=task,
            workspace_path=workspace_path,
            ctx=runtime_ctx,
        )
        allow_outside_workspace_paths = self._allow_outside_workspace_paths(task)
        effective_sub_task_manager = sub_task_manager
        if effective_sub_task_manager is None:
            effective_sub_task_manager = SubTaskManager(
                register_session=register_sub_agent_session,
                unregister_session=unregister_sub_agent_session,
            )

        cycle_executor = self._build_cycle_executor(
            task=task,
            workspace_path=workspace_path,
            workspace_backend=self._workspace_backend
            or LocalWorkspaceBackend(
                workspace_path,
                allow_outside_root=allow_outside_workspace_paths,
            ),
            memory_manager=memory_manager,
            before_cycle_messages=before_cycle_messages,
            interruption_messages=interruption_messages,
            sub_task_manager=effective_sub_task_manager,
            budget_controller=budget_controller,
        )
        try:
            if result is None:
                try:
                    result = self.execution_backend.execute(
                        task=task,
                        initial_messages=messages,
                        shared_state=shared,
                        cycle_executor=cycle_executor,
                        ctx=runtime_ctx,
                        max_cycles=task.max_cycles,
                    )
                except CheckpointReconciliationRequired as interruption:
                    result = interruption.result
        finally:
            if had_configured_disallowed_tools:
                task.metadata["_vv_agent_disallowed_tools"] = configured_disallowed_tools
            else:
                task.metadata.pop("_vv_agent_disallowed_tools", None)

        if budget_controller is not None:
            cancelled = bool(
                result.status == AgentStatus.FAILED
                and runtime_ctx.cancellation_token is not None
                and runtime_ctx.cancellation_token.cancelled
            )
            operation_failed = bool(
                result.status == AgentStatus.FAILED
                and result.completion_reason not in {CompletionReason.BUDGET_EXHAUSTED, CompletionReason.CANCELLED}
            )
            if (
                budget_controller.exhaustion is None
                and result.status is not AgentStatus.RECONCILIATION_REQUIRED
            ):
                exhaustion = budget_controller.terminal(
                    suppress_exhaustion=cancelled or operation_failed,
                )
                if exhaustion is not None and not cancelled and not operation_failed:
                    result = self._budget_failure_result(
                        messages=result.messages,
                        cycles=result.cycles,
                        shared_state=result.shared_state,
                        controller=budget_controller,
                        exhaustion=exhaustion,
                    )
            result.budget_usage = budget_controller.snapshot
            result.budget_exhaustion = budget_controller.exhaustion

        cancelled = bool(runtime_ctx.cancellation_token is not None and runtime_ctx.cancellation_token.cancelled)
        if result.status == AgentStatus.FAILED and cancelled:
            result.completion_reason = CompletionReason.CANCELLED
            result.completion_tool_name = None
            result.partial_output = result.partial_output or _last_assistant_output(result.cycles)
            self._emit_log(
                "run_cancelled",
                cycle=len(result.cycles) or None,
                reason=self._preview_text(result.error or "Operation was cancelled"),
                completion_reason=result.completion_reason.value,
                partial_output=self._preview_text(result.partial_output or ""),
            )
        elif result.status == AgentStatus.MAX_CYCLES:
            result.completion_reason = CompletionReason.MAX_CYCLES
            result.completion_tool_name = None
            result.partial_output = result.partial_output or _last_assistant_output(result.cycles)
            self._emit_log(
                "run_max_cycles",
                cycle=len(result.cycles),
                final_answer=self._preview_text(result.final_answer or ""),
                error=self._preview_text(result.error or ""),
                completion_reason=result.completion_reason.value,
                partial_output=self._preview_text(result.partial_output or ""),
            )
        elif result.status == AgentStatus.FAILED and result.completion_reason is None:
            result.completion_reason = CompletionReason.FAILED
            result.partial_output = result.partial_output or _last_assistant_output(result.cycles)
        return result

    def _build_cycle_executor(
        self,
        *,
        task: AgentTask,
        workspace_path: Path,
        workspace_backend: WorkspaceBackend,
        memory_manager: MemoryManager,
        before_cycle_messages: BeforeCycleMessageProvider | None,
        interruption_messages: InterruptionMessageProvider | None,
        sub_task_manager: SubTaskManager,
        budget_controller: _RunBudgetController | None = None,
    ) -> Callable[[int, list[Message], list[CycleRecord], dict[str, Any], ExecutionContext | None], AgentResult | None]:
        def executor(
            cycle_index: int,
            messages: list[Message],
            cycles: list[CycleRecord],
            shared: dict[str, Any],
            ctx: ExecutionContext | None,
        ) -> AgentResult | None:
            def is_cancelled() -> bool:
                return bool(ctx is not None and ctx.cancellation_token is not None and ctx.cancellation_token.cancelled)

            if ctx is not None:
                ctx.metadata["_vv_agent_active_cycle_index"] = cycle_index

            def cancellation_result(error: str | None = None) -> AgentResult:
                reason = error
                if reason is None and ctx is not None and ctx.cancellation_token is not None:
                    reason = ctx.cancellation_token.reason
                return AgentResult(
                    status=AgentStatus.FAILED,
                    completion_reason=CompletionReason.CANCELLED,
                    partial_output=_last_assistant_output(cycles),
                    messages=messages,
                    cycles=cycles,
                    error=reason or "Operation was cancelled",
                    shared_state=shared,
                    token_usage=summarize_task_token_usage(cycles),
                    budget_usage=(budget_controller.snapshot if budget_controller is not None else None),
                )

            try:
                self._apply_persisted_after_cycle_denials(
                    task=task,
                    shared_state=shared,
                )
            except AfterCycleHookError as exc:
                self._emit_log(
                    "after_cycle_failed",
                    cycle=cycle_index,
                    error_code=exc.code,
                    error=str(exc),
                )
                return self._after_cycle_failure_result(
                    messages=messages,
                    cycles=cycles,
                    shared_state=shared,
                    error=f"{exc.code}: {exc}",
                    budget_controller=budget_controller,
                )

            if budget_controller is not None:
                if ctx is not None:
                    try:
                        ctx.check_cancelled()
                    except Exception as exc:
                        return cancellation_result(str(exc).strip())
                exhaustion = budget_controller.cycle_start(cycle_index)
                if exhaustion is not None:
                    return self._budget_failure_result(
                        messages=messages,
                        cycles=cycles,
                        shared_state=shared,
                        controller=budget_controller,
                        exhaustion=exhaustion,
                    )
            if before_cycle_messages is not None:
                injected = before_cycle_messages(cycle_index, messages, shared)
                if injected:
                    messages.extend(injected)
                    self._emit_log(
                        "cycle_injected_messages",
                        cycle=cycle_index,
                        count=len(injected),
                    )
            self._emit_log(
                "cycle_started",
                cycle=cycle_index,
                max_cycles=task.max_cycles,
                message_count=len(messages),
            )
            cycle_start_message_count = len(messages)
            previous_prompt_tokens: int | None = None
            recent_tool_call_ids: set[str] | None = None
            if cycles:
                last_cycle = cycles[-1]
                usage = last_cycle.token_usage
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                if prompt_tokens <= 0:
                    prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                if prompt_tokens > 0:
                    previous_prompt_tokens = prompt_tokens

                candidate_ids = {call.id for call in last_cycle.tool_calls if getattr(call, "id", "")}
                if candidate_ids:
                    recent_tool_call_ids = candidate_ids
            try:
                self._emit_log(
                    "llm_started",
                    cycle=cycle_index,
                    model=task.model,
                    message_count=len(messages),
                )
                updated_messages, cycle_record = self.cycle_runner.run_cycle(
                    task=task,
                    messages=messages,
                    cycle_index=cycle_index,
                    memory_manager=memory_manager,
                    previous_prompt_tokens=previous_prompt_tokens,
                    recent_tool_call_ids=recent_tool_call_ids,
                    shared_state=shared,
                    ctx=ctx,
                )
            except CheckpointReconciliationRequired:
                raise
            except Exception as exc:
                cancelled = isinstance(exc, CancelledError) or bool(
                    ctx is not None and ctx.cancellation_token is not None and ctx.cancellation_token.cancelled
                )
                if not cancelled:
                    self._emit_log("cycle_failed", cycle=cycle_index, error=str(exc))
                return AgentResult(
                    status=AgentStatus.FAILED,
                    completion_reason=CompletionReason.FAILED,
                    partial_output=_last_assistant_output(cycles),
                    messages=messages,
                    cycles=cycles,
                    error=f"LLM call failed in cycle {cycle_index}: {exc}",
                    shared_state=shared,
                    token_usage=summarize_task_token_usage(cycles),
                )
            # Replace messages list contents in-place so caller sees updates
            messages.clear()
            messages.extend(updated_messages)

            self._emit_log(
                "cycle_llm_response",
                cycle=cycle_index,
                assistant_message=cycle_record.assistant_message,
                assistant_preview=self._preview_text(cycle_record.assistant_message),
                tool_calls=[call.to_dict() for call in cycle_record.tool_calls],
                tool_call_names=[call.name for call in cycle_record.tool_calls],
                tool_call_count=len(cycle_record.tool_calls),
                memory_compacted=cycle_record.memory_compacted,
                token_usage=cycle_record.token_usage.to_dict(),
            )
            cancelled = is_cancelled()
            if budget_controller is not None:
                exhaustion = budget_controller.llm_complete(
                    cycle_index,
                    cycle_record.token_usage,
                    suppress_exhaustion=cancelled,
                )
                if exhaustion is not None:
                    cycles.append(cycle_record)
                    return self._budget_failure_result(
                        messages=messages,
                        cycles=cycles,
                        shared_state=shared,
                        controller=budget_controller,
                        exhaustion=exhaustion,
                    )
            if cancelled:
                cycles.append(cycle_record)
                return cancellation_result()
            if cycle_record.memory_compacted:
                self._emit_log(
                    "memory_compacted",
                    cycle=cycle_index,
                    before_count=cycle_start_message_count,
                    after_count=len(updated_messages),
                )

            if cycle_record.tool_calls:
                if budget_controller is not None:
                    exhaustion = budget_controller.preflight_tools(
                        cycle_index,
                        [call.name for call in cycle_record.tool_calls],
                    )
                    if exhaustion is not None:
                        cycles.append(cycle_record)
                        return self._budget_failure_result(
                            messages=messages,
                            cycles=cycles,
                            shared_state=shared,
                            controller=budget_controller,
                            exhaustion=exhaustion,
                        )
                tool_context_metadata = dict(task.metadata)
                if self.log_handler is not None:
                    tool_context_metadata[_TURN_LOG_HANDLER_METADATA_KEY] = self.log_handler
                context = ToolContext(
                    workspace=workspace_path,
                    shared_state=shared,
                    cycle_index=cycle_index,
                    workspace_backend=workspace_backend,
                    task_id=task.task_id,
                    sub_task_runner=self._build_sub_task_runner(
                        parent_task=task,
                        workspace_path=workspace_path,
                        workspace_backend=workspace_backend,
                        parent_shared_state=shared,
                        sub_task_manager=sub_task_manager,
                        ctx=ctx,
                    ),
                    sub_task_manager=sub_task_manager,
                    ctx=ctx,
                    task_metadata=dict(task.metadata),
                    run_context=(ctx.metadata.get("_vv_agent_run_context") if ctx is not None else None),
                    session=(ctx.metadata.get("_vv_agent_session") if ctx is not None else None),
                    metadata=tool_context_metadata,
                )

                def _on_tool_result(call: ToolCall, result: ToolExecutionResult, *, _cycle: int = cycle_index) -> None:
                    self._emit_log(
                        "tool_result",
                        cycle=_cycle,
                        tool_name=call.name,
                        tool_arguments=call.arguments,
                        tool_call_id=result.tool_call_id,
                        status=result.status_code.value if result.status_code else result.status,
                        directive=result.directive.value,
                        error_code=result.error_code,
                        content=result.content,
                        metadata=dict(result.metadata),
                        content_preview=self._preview_text(result.content),
                    )

                def _on_tool_start(call: ToolCall, *, _cycle: int = cycle_index) -> None:
                    self._emit_log(
                        "tool_started",
                        cycle=_cycle,
                        tool_name=call.name,
                        tool_arguments=call.arguments,
                        tool_call_id=call.id,
                    )

                try:
                    tool_outcome = self.tool_call_runner.run(
                        task=task,
                        tool_calls=cycle_record.tool_calls,
                        context=context,
                        messages=messages,
                        cycle_record=cycle_record,
                        interruption_provider=interruption_messages,
                        on_tool_start=_on_tool_start,
                        on_tool_result=_on_tool_result,
                        ctx=ctx,
                    )
                except ApprovalError as exc:
                    cycles.append(cycle_record)
                    if budget_controller is not None:
                        budget_controller.tool_batch_complete(cycle_index, operation_failed=True)
                    self._emit_log("cycle_failed", cycle=cycle_index, error=str(exc))
                    return AgentResult(
                        status=AgentStatus.FAILED,
                        completion_reason=CompletionReason.FAILED,
                        partial_output=_last_assistant_output(cycles),
                        messages=messages,
                        cycles=cycles,
                        error=str(exc),
                        shared_state=shared,
                        token_usage=summarize_task_token_usage(cycles),
                        budget_usage=(budget_controller.snapshot if budget_controller is not None else None),
                    )
                except _ConfiguredSubTaskCancelledError as exc:
                    cycles.append(cycle_record)
                    if budget_controller is not None:
                        budget_controller.tool_batch_complete(cycle_index, operation_failed=True)
                    return cancellation_result(str(exc).strip())
                tool_result = tool_outcome.directive_result
                cycles.append(cycle_record)
                cancelled = is_cancelled()
                if budget_controller is not None:
                    exhaustion = budget_controller.tool_batch_complete(
                        cycle_index,
                        suppress_exhaustion=cancelled,
                    )
                    if exhaustion is not None:
                        return self._budget_failure_result(
                            messages=messages,
                            cycles=cycles,
                            shared_state=shared,
                            controller=budget_controller,
                            exhaustion=exhaustion,
                        )
                if cancelled:
                    return cancellation_result()
                if tool_outcome.interruption_messages:
                    messages.extend(tool_outcome.interruption_messages)
                    self._emit_log(
                        "run_steered",
                        cycle=cycle_index,
                        steering_count=len(tool_outcome.interruption_messages),
                    )

                if tool_result and tool_result.directive == ToolDirective.WAIT_USER:
                    native_outcome = NativeCycleOutcome(
                        kind=NativeCycleOutcomeKind.WAIT_USER,
                        completion_reason=tool_outcome.completion_reason or CompletionReason.WAIT_USER,
                        completion_tool_name=tool_outcome.completion_tool_name,
                        steer_allowed=False,
                    )
                elif tool_result and tool_result.directive == ToolDirective.FINISH:
                    native_outcome = NativeCycleOutcome(
                        kind=NativeCycleOutcomeKind.COMPLETED,
                        completion_reason=tool_outcome.completion_reason or CompletionReason.TOOL_FINISH,
                        completion_tool_name=tool_outcome.completion_tool_name,
                        steer_allowed=cycle_index < task.max_cycles,
                    )
                elif cycle_index >= task.max_cycles:
                    native_outcome = NativeCycleOutcome(
                        kind=NativeCycleOutcomeKind.MAX_CYCLES,
                        completion_reason=CompletionReason.MAX_CYCLES,
                        steer_allowed=False,
                    )
                else:
                    native_outcome = NativeCycleOutcome(
                        kind=NativeCycleOutcomeKind.CONTINUE,
                        steer_allowed=True,
                    )
                after_cycle_decision, after_cycle_result = self._apply_after_cycle_hooks(
                    task=task,
                    cycle_record=cycle_record,
                    messages=messages,
                    cycles=cycles,
                    shared_state=shared,
                    native_outcome=native_outcome,
                    budget_controller=budget_controller,
                )
                if after_cycle_result is not None:
                    return after_cycle_result
                if (
                    after_cycle_decision is not None
                    and after_cycle_decision.action is AfterCycleAction.STEER
                ):
                    return None

                if tool_result and tool_result.directive == ToolDirective.WAIT_USER:
                    wait_reason = tool_result.metadata.get("question") if isinstance(tool_result.metadata, dict) else None
                    if not wait_reason:
                        wait_reason = tool_result.content
                    self._emit_log(
                        "run_wait_user",
                        cycle=cycle_index,
                        wait_reason=self._preview_text(str(wait_reason)),
                        completion_reason=CompletionReason.WAIT_USER.value,
                        completion_tool_name=tool_outcome.completion_tool_name,
                        partial_output=self._preview_text(_last_assistant_output(cycles) or ""),
                    )
                    return AgentResult(
                        status=AgentStatus.WAIT_USER,
                        completion_reason=tool_outcome.completion_reason or CompletionReason.WAIT_USER,
                        completion_tool_name=tool_outcome.completion_tool_name,
                        partial_output=_last_assistant_output(cycles),
                        messages=messages,
                        cycles=cycles,
                        wait_reason=str(wait_reason),
                        shared_state=shared,
                        token_usage=summarize_task_token_usage(cycles),
                    )

                if tool_result and tool_result.directive == ToolDirective.FINISH:
                    final_answer = self._extract_final_message(tool_result)
                    self._emit_log(
                        "run_completed",
                        cycle=cycle_index,
                        final_answer=self._preview_text(final_answer),
                        completion_reason=(tool_outcome.completion_reason or CompletionReason.TOOL_FINISH).value,
                        completion_tool_name=tool_outcome.completion_tool_name,
                    )
                    return AgentResult(
                        status=AgentStatus.COMPLETED,
                        completion_reason=tool_outcome.completion_reason or CompletionReason.TOOL_FINISH,
                        completion_tool_name=tool_outcome.completion_tool_name,
                        messages=messages,
                        cycles=cycles,
                        final_answer=final_answer,
                        shared_state=shared,
                        token_usage=summarize_task_token_usage(cycles),
                    )

                return None  # continue to next cycle

            cycles.append(cycle_record)
            if task.no_tool_policy == "finish":
                native_outcome = NativeCycleOutcome(
                    kind=NativeCycleOutcomeKind.COMPLETED,
                    completion_reason=CompletionReason.NO_TOOL_FINISH,
                    steer_allowed=cycle_index < task.max_cycles,
                )
            elif task.no_tool_policy == "wait_user":
                native_outcome = NativeCycleOutcome(
                    kind=NativeCycleOutcomeKind.WAIT_USER,
                    completion_reason=CompletionReason.WAIT_USER,
                    steer_allowed=False,
                )
            elif cycle_index >= task.max_cycles:
                native_outcome = NativeCycleOutcome(
                    kind=NativeCycleOutcomeKind.MAX_CYCLES,
                    completion_reason=CompletionReason.MAX_CYCLES,
                    steer_allowed=False,
                )
            else:
                native_outcome = NativeCycleOutcome(
                    kind=NativeCycleOutcomeKind.CONTINUE,
                    steer_allowed=True,
                )
            after_cycle_decision, after_cycle_result = self._apply_after_cycle_hooks(
                task=task,
                cycle_record=cycle_record,
                messages=messages,
                cycles=cycles,
                shared_state=shared,
                native_outcome=native_outcome,
                budget_controller=budget_controller,
            )
            if after_cycle_result is not None:
                return after_cycle_result
            if (
                after_cycle_decision is not None
                and after_cycle_decision.action is AfterCycleAction.STEER
            ):
                return None

            if task.no_tool_policy == "finish":
                self._emit_log(
                    "run_completed",
                    cycle=cycle_index,
                    final_answer=self._preview_text(cycle_record.assistant_message),
                    completion_reason=CompletionReason.NO_TOOL_FINISH.value,
                )
                return AgentResult(
                    status=AgentStatus.COMPLETED,
                    completion_reason=CompletionReason.NO_TOOL_FINISH,
                    messages=messages,
                    cycles=cycles,
                    final_answer=cycle_record.assistant_message,
                    shared_state=shared,
                    token_usage=summarize_task_token_usage(cycles),
                )

            if task.no_tool_policy == "wait_user":
                self._emit_log(
                    "run_wait_user",
                    cycle=cycle_index,
                    wait_reason=self._preview_text(cycle_record.assistant_message or "No tool call"),
                    completion_reason=CompletionReason.WAIT_USER.value,
                    partial_output=self._preview_text(cycle_record.assistant_message),
                )
                return AgentResult(
                    status=AgentStatus.WAIT_USER,
                    completion_reason=CompletionReason.WAIT_USER,
                    partial_output=cycle_record.assistant_message or None,
                    messages=messages,
                    cycles=cycles,
                    wait_reason=cycle_record.assistant_message or "No tool call and runtime is waiting for user.",
                    shared_state=shared,
                    token_usage=summarize_task_token_usage(cycles),
                )

            if cycle_index < task.max_cycles:
                messages.append(Message(role="user", content=self._build_continue_hint()))

            return None  # continue to next cycle

        return executor

    def _apply_after_cycle_hooks(
        self,
        *,
        task: AgentTask,
        cycle_record: CycleRecord,
        messages: list[Message],
        cycles: list[CycleRecord],
        shared_state: dict[str, Any],
        native_outcome: NativeCycleOutcome,
        budget_controller: _RunBudgetController | None,
    ) -> tuple[AfterCycleDecision | None, AgentResult | None]:
        if not self.after_cycle_hook_manager.has_hooks():
            return None, None

        try:
            disallowed = self._effective_after_cycle_denials(
                task=task,
                shared_state=shared_state,
            )
            available_tool_names = [
                name
                for name in plan_tool_names(task)
                if self.tool_registry.has_tool(name)
                and self.tool_registry.has_schema(name)
            ]
            snapshot = AfterCycleSnapshot.capture(
                task_id=task.task_id,
                cycle_index=cycle_record.index,
                max_cycles=task.max_cycles,
                cycle=cycle_record,
                messages=messages,
                shared_state=shared_state,
                cumulative_token_usage=summarize_task_token_usage(cycles),
                available_tool_names=available_tool_names,
                disallowed_tool_names=disallowed,
                native_outcome=native_outcome,
            )
            decision = self.after_cycle_hook_manager.apply(snapshot)
            if decision.disallow_tools:
                persist_after_cycle_disallowed_tools(
                    shared_state,
                    decision.disallow_tools,
                )
                self._apply_persisted_after_cycle_denials(
                    task=task,
                    shared_state=shared_state,
                )
        except AfterCycleHookError as exc:
            self._emit_log(
                "after_cycle_failed",
                cycle=cycle_record.index,
                error_code=exc.code,
                error=str(exc),
            )
            return None, self._after_cycle_failure_result(
                messages=messages,
                cycles=cycles,
                shared_state=shared_state,
                error=f"{exc.code}: {exc}",
                budget_controller=budget_controller,
            )
        except Exception as exc:
            code = "after_cycle_hook_failed"
            self._emit_log(
                "after_cycle_failed",
                cycle=cycle_record.index,
                error_code=code,
                error=str(exc),
            )
            return None, self._after_cycle_failure_result(
                messages=messages,
                cycles=cycles,
                shared_state=shared_state,
                error=f"{code}: failed to prepare after-cycle snapshot: {exc}",
                budget_controller=budget_controller,
            )

        if decision.action is AfterCycleAction.STOP_NON_SUCCESS:
            assert decision.stop is not None
            self._emit_log(
                "after_cycle_stopped",
                cycle=cycle_record.index,
                error_code=decision.stop.code,
                error=decision.stop.message,
            )
            return decision, self._after_cycle_failure_result(
                messages=messages,
                cycles=cycles,
                shared_state=shared_state,
                error=f"{decision.stop.code}: {decision.stop.message}",
                budget_controller=budget_controller,
            )

        if decision.action is AfterCycleAction.STEER:
            if not native_outcome.steer_allowed:
                code = "after_cycle_steer_unavailable"
                self._emit_log(
                    "after_cycle_failed",
                    cycle=cycle_record.index,
                    error_code=code,
                    error="after-cycle steering is unavailable at this boundary",
                )
                return decision, self._after_cycle_failure_result(
                    messages=messages,
                    cycles=cycles,
                    shared_state=shared_state,
                    error=f"{code}: after-cycle steering is unavailable at this boundary",
                    budget_controller=budget_controller,
                )
            messages.extend(
                Message(role="user", content=content)
                for content in decision.steering_messages
            )
            self._emit_log(
                "after_cycle_steered",
                cycle=cycle_record.index,
                steering_count=len(decision.steering_messages),
                disallowed_tools=list(decision.disallow_tools),
            )
        else:
            self._emit_log(
                "after_cycle_decision",
                cycle=cycle_record.index,
                action=decision.action.value,
                disallowed_tools=list(decision.disallow_tools),
            )
        return decision, None

    @staticmethod
    def _after_cycle_failure_result(
        *,
        messages: list[Message],
        cycles: list[CycleRecord],
        shared_state: dict[str, Any],
        error: str,
        budget_controller: _RunBudgetController | None,
    ) -> AgentResult:
        return AgentResult(
            status=AgentStatus.FAILED,
            completion_reason=CompletionReason.FAILED,
            partial_output=_last_assistant_output(cycles),
            messages=messages,
            cycles=cycles,
            error=error,
            shared_state=shared_state,
            token_usage=summarize_task_token_usage(cycles),
            budget_usage=(
                budget_controller.snapshot
                if budget_controller is not None
                else None
            ),
        )

    @staticmethod
    def _effective_after_cycle_denials(
        *,
        task: AgentTask,
        shared_state: dict[str, Any],
    ) -> list[str]:
        configured = task.metadata.get("_vv_agent_disallowed_tools")
        configured_values = (
            [value for value in configured if isinstance(value, str)]
            if isinstance(configured, list)
            else []
        )
        persisted = read_after_cycle_disallowed_tools(shared_state)
        return sorted(
            set([*configured_values, *persisted]),
            key=utf16_sort_key,
        )

    @classmethod
    def _apply_persisted_after_cycle_denials(
        cls,
        *,
        task: AgentTask,
        shared_state: dict[str, Any],
    ) -> None:
        effective = cls._effective_after_cycle_denials(
            task=task,
            shared_state=shared_state,
        )
        if effective:
            task.metadata["_vv_agent_disallowed_tools"] = effective

    def _emit_cycle_tool_results(self, *, cycle_record: CycleRecord) -> None:
        for idx, result in enumerate(cycle_record.tool_results):
            tool_name = None
            tool_arguments = None
            if idx < len(cycle_record.tool_calls):
                tool_name = cycle_record.tool_calls[idx].name
                tool_arguments = cycle_record.tool_calls[idx].arguments
            self._emit_log(
                "tool_result",
                cycle=cycle_record.index,
                tool_name=tool_name or "unknown",
                tool_arguments=tool_arguments,
                tool_call_id=result.tool_call_id,
                status=result.status_code.value if result.status_code else result.status,
                directive=result.directive.value,
                error_code=result.error_code,
                content=result.content,
                metadata=dict(result.metadata),
                content_preview=self._preview_text(result.content),
            )

    @staticmethod
    def _budget_failure_result(
        *,
        messages: list[Message],
        cycles: list[CycleRecord],
        shared_state: dict[str, Any],
        controller: _RunBudgetController,
        exhaustion: BudgetExhaustion,
    ) -> AgentResult:
        return AgentResult(
            status=AgentStatus.FAILED,
            completion_reason=CompletionReason.BUDGET_EXHAUSTED,
            completion_tool_name=None,
            partial_output=_last_assistant_output(cycles),
            messages=messages,
            cycles=cycles,
            final_answer=None,
            wait_reason=None,
            error="Run budget exhausted.",
            shared_state=shared_state,
            token_usage=summarize_task_token_usage(cycles),
            budget_usage=controller.snapshot,
            budget_exhaustion=exhaustion,
        )

    def _emit_log(self, event: str, **payload: Any) -> None:
        if self.log_handler is None:
            return
        self.log_handler(event, payload)

    @staticmethod
    def _event_emitter_from_context(ctx: ExecutionContext | None) -> RunEventHandler | None:
        if ctx is None:
            return None
        emitter = ctx.metadata.get("_vv_agent_emit_event")
        return emitter if callable(emitter) else None

    @staticmethod
    def _metadata_str(metadata: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            if normalized := normalize_identity_string(metadata.get(key)):
                return normalized
        return None

    @staticmethod
    def _copy_message(message: Message) -> Message:
        return Message(
            role=message.role,
            content=message.content,
            name=message.name,
            tool_call_id=message.tool_call_id,
            tool_calls=list(message.tool_calls) if message.tool_calls else None,
            reasoning_content=message.reasoning_content,
            image_url=message.image_url,
            metadata=dict(message.metadata),
        )

    def _build_initial_messages(
        self,
        *,
        task: AgentTask,
        initial_messages: list[Message] | None,
        user_message: str | None,
    ) -> list[Message]:
        if initial_messages:
            prepared = [self._copy_message(message) for message in initial_messages]
            if not prepared or prepared[0].role != "system":
                prepared.insert(0, Message(role="system", content=task.system_prompt, metadata=dict(task.metadata)))
            elif task.metadata:
                merged_metadata = dict(task.metadata)
                merged_metadata.update(prepared[0].metadata)
                if task.metadata.get("is_sub_task") is True:
                    for key in _RESERVED_SUB_AGENT_METADATA_KEYS:
                        if key in task.metadata:
                            merged_metadata[key] = task.metadata[key]
                        else:
                            merged_metadata.pop(key, None)
                prepared[0].metadata = merged_metadata
            message_to_append = task.user_prompt if user_message is None else user_message
            if message_to_append:
                prepared.append(Message(role="user", content=message_to_append))
            return prepared

        first_user_message = task.user_prompt if user_message is None else user_message
        return [
            Message(role="system", content=task.system_prompt, metadata=dict(task.metadata)),
            Message(role="user", content=first_user_message),
        ]

    def _build_memory_manager(
        self,
        *,
        task: AgentTask,
        workspace_path: Path,
        ctx: ExecutionContext | None = None,
    ) -> MemoryManager:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}

        def summarize(prompt: str, backend: str | None, model: str | None) -> str | None:
            return self._summarize_memory_prompt(
                prompt,
                backend,
                model,
                ctx=ctx,
            )

        def read_optional_int(key: str, *, minimum: int = 0) -> int | None:
            if key not in metadata:
                return None
            raw = metadata.get(key)
            if raw is None:
                return None
            try:
                value = int(raw)
            except (TypeError, ValueError):
                return None
            return max(value, minimum)

        def read_int(key: str, default: int, *, minimum: int = 0) -> int:
            value = read_optional_int(key, minimum=minimum)
            return max(default, minimum) if value is None else value

        def read_float(key: str, default: float, *, minimum: float = 0.0, maximum: float | None = None) -> float:
            raw = metadata.get(key, default)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = default
            value = max(value, minimum)
            if maximum is not None:
                value = min(value, maximum)
            return value

        def read_str_set(key: str) -> set[str] | None:
            raw = metadata.get(key)
            if not isinstance(raw, list):
                return None
            values = {str(item).strip() for item in raw if str(item).strip()}
            return values or None

        warning_threshold = max(1, min(task.memory_threshold_percentage, 100))
        local_summary_backend, local_summary_model = self._load_local_memory_summary_defaults()
        metadata_summary_backend = self._read_optional_str(
            metadata,
            "memory_summary_backend",
            "compress_memory_summary_backend",
            "memory_compress_backend",
        )
        metadata_summary_model = self._read_optional_str(
            metadata,
            "memory_summary_model",
            "compress_memory_summary_model",
            "memory_compress_model",
        )
        summary_backend = metadata_summary_backend or local_summary_backend or self.default_backend
        summary_model = metadata_summary_model or local_summary_model or task.model
        session_memory_extraction_backend = (
            self._read_optional_str(metadata, "session_memory_extraction_backend") or summary_backend
        )
        session_memory_extraction_model = (
            self._read_optional_str(metadata, "session_memory_extraction_model") or summary_model
        )
        session_memory_enabled = self._read_session_memory_enabled(metadata)
        model_context_window = self._metadata_token_limit(
            metadata,
            "model_context_window",
            minimum=1,
        )
        model_max_output_tokens = self._metadata_token_limit(
            metadata,
            "model_max_output_tokens",
            minimum=0,
        )
        if model_context_window is None or model_max_output_tokens is None:
            fallback_context_window, fallback_max_output_tokens = resolve_model_token_limits(task.model)
            if model_context_window is None:
                model_context_window = fallback_context_window
            if model_max_output_tokens is None:
                model_max_output_tokens = fallback_max_output_tokens
        model_context_window = model_context_window or 200_000

        effective_model_settings = task.model_settings
        if ctx is not None:
            runtime_model_settings = ctx.metadata.get("_vv_agent_model_settings")
            if isinstance(runtime_model_settings, ModelSettings):
                effective_model_settings = runtime_model_settings
        request_max_tokens = (
            effective_model_settings.max_tokens
            if effective_model_settings is not None
            else None
        )
        task_reserved_output_tokens = self._metadata_token_limit(
            metadata,
            "reserved_output_tokens",
            minimum=0,
        )
        if request_max_tokens is not None:
            reserved_output_tokens = request_max_tokens
            reserved_output_source = "model_settings"
        elif task_reserved_output_tokens is not None:
            reserved_output_tokens = task_reserved_output_tokens
            reserved_output_source = "task_metadata"
        else:
            reserved_output_tokens = 16_000
            reserved_output_source = "framework_fallback"
            if (
                model_max_output_tokens is not None
                and model_max_output_tokens < reserved_output_tokens
            ):
                reserved_output_tokens = model_max_output_tokens
                reserved_output_source = "framework_fallback_capped_by_model_capability"
        session_memory: SessionMemory | None = None
        if session_memory_enabled:
            session_memory_scope = self._read_optional_str(metadata, "session_id", "task_id") or str(task.task_id or "").strip()
            session_memory = SessionMemory(
                SessionMemoryConfig(
                    min_tokens_before_extraction=read_int("session_memory_min_tokens", 10_000, minimum=1),
                    max_tokens=read_int("session_memory_max_tokens", 40_000, minimum=1),
                    min_text_messages=read_int("session_memory_min_text_messages", 5, minimum=1),
                    storage_dir=str(metadata.get("session_memory_storage_dir", ".memory/session")),
                    extraction_callback=summarize,
                    extraction_backend=session_memory_extraction_backend,
                    extraction_model=session_memory_extraction_model,
                    token_model=task.model or "",
                ),
                workspace=workspace_path if task.use_workspace else None,
                storage_scope=session_memory_scope,
            )
            session_memory.load()
        return MemoryManager(
            compact_threshold=max(task.memory_compact_threshold, 0),
            keep_recent_messages=read_int("memory_keep_recent_messages", 10, minimum=1),
            model=task.model or "",
            model_context_window=model_context_window,
            model_max_output_tokens=model_max_output_tokens,
            reserved_output_tokens=reserved_output_tokens,
            reserved_output_source=reserved_output_source,
            autocompact_buffer_tokens=read_int("autocompact_buffer_tokens", 13_000, minimum=0),
            language=str(metadata.get("language", "zh-CN")),
            warning_threshold_percentage=warning_threshold,
            include_memory_warning=bool(metadata.get("include_memory_warning", False)),
            tool_result_compact_threshold=read_int("tool_result_compact_threshold", 2000),
            tool_result_keep_last=read_int("tool_result_keep_last", 3),
            tool_result_excerpt_head=read_int("tool_result_excerpt_head", 200),
            tool_result_excerpt_tail=read_int("tool_result_excerpt_tail", 200),
            tool_calls_keep_last=read_int("tool_calls_keep_last", 3),
            assistant_no_tool_keep_last=read_int("assistant_no_tool_keep_last", 1),
            microcompact_trigger_ratio=read_float("microcompact_trigger_ratio", 0.75, minimum=0.0, maximum=1.0),
            microcompact_keep_recent_cycles=read_int("microcompact_keep_recent_cycles", 3, minimum=0),
            microcompact_min_result_length=read_int("microcompact_min_result_length", 500, minimum=1),
            microcompact_compactable_tools=read_str_set("microcompact_compactable_tools"),
            tool_result_artifact_dir=str(metadata.get("tool_result_artifact_dir", ".memory/tool_results")),
            workspace=workspace_path if task.use_workspace else None,
            summary_event_limit=read_int("summary_event_limit", 40, minimum=1),
            summary_backend=summary_backend,
            summary_model=summary_model,
            summary_callback=summarize,
            base_system_prompt=task.system_prompt,
            session_memory=session_memory,
        )

    @staticmethod
    def _read_optional_str(metadata: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            raw = metadata.get(key)
            if isinstance(raw, str):
                value = raw.strip()
                if value:
                    return value
        return None

    @staticmethod
    def _read_session_memory_enabled(metadata: dict[str, Any]) -> bool:
        explicit = _parse_optional_bool(metadata.get("session_memory_enabled", metadata.get("enable_session_memory")))
        if explicit is not None:
            return explicit
        return not bool(metadata.get("is_sub_task"))

    def _load_local_memory_summary_defaults(self) -> tuple[str | None, str | None]:
        if self._memory_summary_defaults is not None:
            return self._memory_summary_defaults

        backend: str | None = None
        model: str | None = None
        settings_file = self.settings_file
        if settings_file is None or not settings_file.exists():
            self._memory_summary_defaults = (None, None)
            return self._memory_summary_defaults

        try:
            module = ast.parse(settings_file.read_text(encoding="utf-8"), filename=str(settings_file))
            backend = self._read_literal_setting(
                module,
                "DEFAULT_USER_MEMORY_SUMMARIZE_BACKEND",
                "DEFAULT_MEMORY_SUMMARIZE_BACKEND",
                "VV_AGENT_MEMORY_SUMMARY_BACKEND",
            )
            model = self._read_literal_setting(
                module,
                "DEFAULT_USER_MEMORY_SUMMARIZE_MODEL",
                "DEFAULT_MEMORY_SUMMARIZE_MODEL",
                "VV_AGENT_MEMORY_SUMMARY_MODEL",
            )
        except Exception:
            logging.getLogger(__name__).debug(
                "Failed to load memory summary defaults from settings file",
                exc_info=True,
            )

        self._memory_summary_defaults = (backend, model)
        return self._memory_summary_defaults

    @staticmethod
    def _read_literal_setting(module: ast.Module, *names: str) -> str | None:
        if not names:
            return None
        name_set = set(names)

        for node in module.body:
            target_name: str | None = None
            value_node: ast.expr | None = None

            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_name = node.target.id
                value_node = node.value
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        target_name = target.id
                        value_node = node.value
                        break

            if target_name not in name_set or value_node is None:
                continue

            try:
                literal = ast.literal_eval(value_node)
            except Exception:
                continue
            if isinstance(literal, str):
                value = literal.strip()
                if value:
                    return value
        return None

    def _summarize_memory_prompt(
        self,
        prompt: str,
        backend: str | None,
        model: str | None,
        *,
        ctx: ExecutionContext | None = None,
    ) -> str | None:
        backend_name = (backend or self.default_backend or "").strip()
        model_name = (model or "").strip()
        if not backend_name or not model_name:
            return None
        if self.settings_file is None:
            return None

        cache_key = (backend_name, model_name)
        client = self._memory_summary_clients.get(cache_key)
        if client is None:
            client, _ = self.llm_builder(
                self.settings_file,
                backend=backend_name,
                model=model_name,
                timeout_seconds=self.sub_agent_timeout_seconds,
            )
            self._memory_summary_clients[cache_key] = client

        request = LlmRequest(
            model=model_name,
            messages=[Message(role="user", content=prompt)],
            tools=[],
            metadata={
                "_vv_agent_checkpoint_model": {
                    "backend": backend_name,
                    "model_id": model_name,
                }
            },
        )
        checkpoint_controller = (
            ctx.metadata.get("_vv_agent_checkpoint_controller") if ctx is not None else None
        )
        cycle_index = (
            ctx.metadata.get("_vv_agent_active_cycle_index") if ctx is not None else None
        )

        def invoke() -> Any:
            return complete_llm_request(client, request)

        if isinstance(checkpoint_controller, CheckpointResumeController) and isinstance(
            cycle_index,
            int,
        ):
            assert ctx is not None
            summary_index = int(
                ctx.metadata.get("_vv_agent_checkpoint_memory_summary_index", 0)
            ) + 1
            ctx.metadata["_vv_agent_checkpoint_memory_summary_index"] = summary_index
            response = checkpoint_controller.complete_model(
                cycle_index=cycle_index,
                operation_slot=f"memory_summary:{summary_index}",
                request=request,
                invoke=invoke,
            )
        else:
            response = invoke()
        content = (response.content or "").strip()
        return content or None

    def _preview_text(self, text: str) -> str:
        cleaned = text.replace("\n", " ").strip()
        if self.log_preview_chars is None or self.log_preview_chars <= 0:
            return cleaned
        if len(cleaned) <= self.log_preview_chars:
            return cleaned
        return f"{cleaned[: self.log_preview_chars - 3]}..."

    def _prepare_workspace(self, workspace: str | Path | None) -> Path:
        target = Path(workspace) if workspace else self.default_workspace
        if target is None:
            target = Path.cwd() / ".vv-agent-workspace"
        target = target.resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _allow_outside_workspace_paths(task: AgentTask) -> bool:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        for key in (
            "allow_outside_workspace_paths",
            "allow_outside_workspace",
            "workspace_allow_outside_main",
            "workspace_allow_outside",
        ):
            parsed = _parse_optional_bool(metadata.get(key))
            if parsed is not None:
                return parsed
        return False

    @staticmethod
    def _build_continue_hint() -> str:
        return f"No tool call was produced. Continue the task and call `{TASK_FINISH_TOOL_NAME}` when all todo items are done."

    @staticmethod
    def _extract_final_message(result: ToolExecutionResult) -> str:
        if isinstance(result.metadata, dict):
            final = result.metadata.get("final_message")
            if isinstance(final, str) and final:
                return final

        try:
            payload = json.loads(result.content)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            message = payload.get("message")
            if isinstance(message, str) and message:
                return message

        return result.content

    def _build_sub_task_runner(
        self,
        *,
        parent_task: AgentTask,
        workspace_path: Path,
        workspace_backend: WorkspaceBackend,
        parent_shared_state: dict[str, Any],
        sub_task_manager: SubTaskManager,
        ctx: ExecutionContext | None = None,
    ) -> Callable[[SubTaskRequest], SubTaskOutcome] | None:
        if not parent_task.sub_agents:
            return None

        def runner(request: SubTaskRequest) -> SubTaskOutcome:
            return self._run_sub_task(
                parent_task=parent_task,
                workspace_path=workspace_path,
                workspace_backend=workspace_backend,
                parent_shared_state=parent_shared_state,
                request=request,
                sub_task_manager=sub_task_manager,
                ctx=ctx,
            )

        return runner

    def _run_sub_task(
        self,
        *,
        parent_task: AgentTask,
        workspace_path: Path,
        workspace_backend: WorkspaceBackend,
        parent_shared_state: dict[str, Any],
        request: SubTaskRequest,
        sub_task_manager: SubTaskManager | None = None,
        ctx: ExecutionContext | None = None,
    ) -> SubTaskOutcome:
        from vv_agent.interactive import AgentSessionRun, InteractiveAgentDefinition, create_agent_session

        request_metadata = request.metadata if isinstance(request.metadata, dict) else {}
        assigned_identity = take_sub_task_identity()
        if assigned_identity is None:
            sub_task_id = f"{parent_task.task_id}_sub_{request.agent_name}_{uuid.uuid4().hex[:8]}"
            sub_session_id = sub_task_id
        else:
            sub_task_id = assigned_identity.task_id
            sub_session_id = assigned_identity.session_id
        parent_metadata = ctx.metadata if ctx is not None else {}
        public_run_context = parent_metadata.get("_vv_agent_run_context")
        parent_run_id = (
            normalize_identity_string(getattr(public_run_context, "run_id", None))
            or self._metadata_str(parent_metadata, "_vv_agent_run_id")
            or self._metadata_str(request_metadata, "parent_run_id")
            or ""
        )
        parent_tool_call_id = self._metadata_str(request_metadata, "parent_tool_call_id") or ""
        child_run_id = f"run_{uuid.uuid4().hex}"
        public_run_metadata = getattr(public_run_context, "metadata", None)
        trace_id = (
            self._metadata_str(parent_metadata, "_vv_agent_trace_id", "trace_id")
            or (
                self._metadata_str(public_run_metadata, "_vv_agent_trace_id", "trace_id")
                if isinstance(public_run_metadata, dict)
                else None
            )
            or self._metadata_str(parent_task.metadata, "_vv_agent_trace_id", "trace_id")
            or child_run_id
        )
        event_emitter = self._event_emitter_from_context(ctx)
        sub_agent = parent_task.sub_agents.get(request.agent_name)
        if sub_agent is None:
            available = ", ".join(sorted(parent_task.sub_agents))
            outcome = SubTaskOutcome(
                task_id=sub_task_id,
                session_id=sub_session_id,
                agent_name=request.agent_name,
                status=AgentStatus.FAILED,
                completion_reason=CompletionReason.FAILED,
                error=f"Unknown sub-agent {request.agent_name!r}. Available: {available}",
                error_code="sub_task_failed",
            )
            if sub_task_manager is not None:
                sub_task_manager.record_outcome(
                    sub_task_id,
                    outcome,
                    workspace_backend=workspace_backend,
                    parent_run_id=parent_run_id or None,
                    parent_tool_call_id=parent_tool_call_id or None,
                )
            return outcome

        try:
            sub_workspace_backend = self._sub_agent_workspace_backend(
                workspace_backend,
                request.exclude_files_pattern,
            )
        except _SubTaskContractError as exc:
            outcome = SubTaskOutcome(
                task_id=sub_task_id,
                session_id=sub_session_id,
                agent_name=request.agent_name,
                status=AgentStatus.FAILED,
                completion_reason=CompletionReason.FAILED,
                error=str(exc),
                error_code=exc.code,
            )
            if sub_task_manager is not None:
                sub_task_manager.record_outcome(
                    sub_task_id,
                    outcome,
                    workspace_backend=workspace_backend,
                    parent_run_id=parent_run_id or None,
                    parent_tool_call_id=parent_tool_call_id or None,
                )
            return outcome

        manager_execution_token = (
            sub_task_manager._begin_execution(
                task_id=sub_task_id,
                session_id=sub_session_id,
                agent_name=request.agent_name,
                task_title=request.task_description,
                workspace_backend=sub_workspace_backend,
                parent_run_id=parent_run_id or None,
                parent_tool_call_id=parent_tool_call_id or None,
            )
            if sub_task_manager is not None
            else None
        )

        completed_sub_runs: set[str] = set()
        lifecycle_lock = RLock()

        def _emit_sub_run_started(
            run_id: str,
            *,
            emitter: RunEventHandler | None = event_emitter,
            current_trace_id: str = trace_id,
            current_parent_run_id: str = parent_run_id,
            current_parent_tool_call_id: str = parent_tool_call_id,
        ) -> None:
            if emitter is None:
                return
            emitter(
                SubRunStartedEvent(
                    run_id=run_id,
                    trace_id=current_trace_id,
                    session_id=sub_session_id,
                    child_session_id=sub_session_id,
                    parent_run_id=current_parent_run_id or None,
                    parent_tool_call_id=current_parent_tool_call_id,
                    agent_name=request.agent_name,
                    task_id=sub_task_id,
                    metadata={
                        "parent_task_id": parent_task.task_id,
                        "model": _trim_portable_whitespace(str(sub_agent.model or "")),
                    },
                )
            )

        def _emit_sub_run_completed(
            run_id: str,
            outcome: SubTaskOutcome,
            *,
            token_usage: dict[str, Any] | None = None,
            budget_usage: BudgetUsageSnapshot | None = None,
            budget_exhaustion: BudgetExhaustion | None = None,
            emitter: RunEventHandler | None = event_emitter,
            current_trace_id: str = trace_id,
            current_parent_run_id: str = parent_run_id,
            current_parent_tool_call_id: str = parent_tool_call_id,
        ) -> None:
            if emitter is None:
                return
            metadata: dict[str, Any] = {"cycles": outcome.cycles}
            error_code = outcome.error_code
            if outcome.status == AgentStatus.FAILED and error_code is None:
                error_code = "sub_task_failed"
            if error_code is not None:
                metadata["error_code"] = error_code
            emitter(
                SubRunCompletedEvent(
                    run_id=run_id,
                    trace_id=current_trace_id,
                    session_id=sub_session_id,
                    child_session_id=sub_session_id,
                    parent_run_id=current_parent_run_id or None,
                    parent_tool_call_id=current_parent_tool_call_id,
                    agent_name=request.agent_name,
                    task_id=sub_task_id,
                    status=outcome.status.value,
                    final_output=outcome.final_answer,
                    wait_reason=outcome.wait_reason,
                    error=outcome.error,
                    completion_reason=outcome.completion_reason,
                    completion_tool_name=outcome.completion_tool_name,
                    partial_output=outcome.partial_output,
                    token_usage=token_usage,
                    budget_usage=budget_usage,
                    budget_exhaustion=budget_exhaustion,
                    metadata=metadata,
                )
            )

        def _complete_sub_run_once(
            run_id: str,
            outcome: SubTaskOutcome,
            *,
            token_usage: dict[str, Any] | None = None,
            budget_usage: BudgetUsageSnapshot | None = None,
            budget_exhaustion: BudgetExhaustion | None = None,
            emitter: RunEventHandler | None = event_emitter,
            current_trace_id: str = trace_id,
            current_parent_run_id: str = parent_run_id,
            current_parent_tool_call_id: str = parent_tool_call_id,
        ) -> None:
            with lifecycle_lock:
                if run_id in completed_sub_runs:
                    return
                completed_sub_runs.add(run_id)
            try:
                _emit_sub_run_completed(
                    run_id,
                    outcome,
                    token_usage=token_usage,
                    budget_usage=budget_usage,
                    budget_exhaustion=budget_exhaustion,
                    emitter=emitter,
                    current_trace_id=current_trace_id,
                    current_parent_run_id=current_parent_run_id,
                    current_parent_tool_call_id=current_parent_tool_call_id,
                )
            except BaseException:
                logging.getLogger(__name__).exception("Configured sub-agent completion sink failed")

        def _emit_sub_run_started_or_fail(
            run_id: str,
            *,
            emitter: RunEventHandler | None = event_emitter,
            current_trace_id: str = trace_id,
            current_parent_run_id: str = parent_run_id,
            current_parent_tool_call_id: str = parent_tool_call_id,
        ) -> None:
            try:
                _emit_sub_run_started(
                    run_id,
                    emitter=emitter,
                    current_trace_id=current_trace_id,
                    current_parent_run_id=current_parent_run_id,
                    current_parent_tool_call_id=current_parent_tool_call_id,
                )
            except BaseException as exc:
                failed_outcome = SubTaskOutcome(
                    task_id=sub_task_id,
                    session_id=sub_session_id,
                    agent_name=request.agent_name,
                    status=AgentStatus.FAILED,
                    completion_reason=CompletionReason.FAILED,
                    error=str(exc),
                    error_code="sub_task_failed",
                )
                try:
                    _complete_sub_run_once(
                        run_id,
                        failed_outcome,
                        emitter=emitter,
                        current_trace_id=current_trace_id,
                        current_parent_run_id=current_parent_run_id,
                        current_parent_tool_call_id=current_parent_tool_call_id,
                    )
                except BaseException:
                    logging.getLogger(__name__).exception(
                        "Configured sub-agent completion sink failed after started sink failure"
                    )
                raise

        def _emit_sub_session_event(event: str, payload: dict[str, Any]) -> None:
            if self.log_handler is None:
                return
            enriched = _enrich_sub_agent_payload(
                payload,
                task_id=sub_task_id,
                session_id=sub_session_id,
                sub_agent_name=request.agent_name,
            )
            if parent_tool_call_id:
                enriched.setdefault("parent_tool_call_id", parent_tool_call_id)
            if parent_run_id:
                enriched.setdefault("parent_run_id", parent_run_id)
            self.log_handler(f"sub_agent_{event}", enriched)

        outcome_workspace_backend = sub_workspace_backend
        try:
            _emit_sub_run_started_or_fail(child_run_id)
            self._validate_sub_agent_config(sub_agent)
            effective_model_settings = self._effective_parent_model_settings(parent_task=parent_task, ctx=ctx)
            llm_client, model_id, resolved_payload, resolved_config = self._resolve_sub_agent_client(
                parent_task=parent_task,
                sub_agent=sub_agent,
                ctx=ctx,
            )
            sub_task = self._build_sub_agent_task(
                parent_task=parent_task,
                sub_task_id=sub_task_id,
                sub_session_id=sub_session_id,
                sub_agent_name=request.agent_name,
                sub_agent=sub_agent,
                resolved_model_id=model_id,
                resolved_native_multimodal=resolved_config.native_multimodal,
                resolved_context_length=resolved_config.context_length,
                resolved_max_output_tokens=resolved_config.max_output_tokens,
                child_run_id=child_run_id,
                trace_id=trace_id,
                parent_run_id=parent_run_id,
                parent_tool_call_id=parent_tool_call_id,
                request=request,
                parent_shared_state=parent_shared_state,
                workspace_path=workspace_path,
                effective_model_settings=effective_model_settings,
                parent_tool_policy_metadata=self._trusted_parent_tool_policy_metadata(
                    parent_task=parent_task,
                    ctx=ctx,
                ),
            )
            sub_runtime = AgentRuntime(
                llm_client=llm_client,
                tool_registry=self._build_sub_agent_registry(),
                default_workspace=workspace_path,
                log_preview_chars=self.log_preview_chars,
                settings_file=self.settings_file,
                default_backend=self.default_backend,
                llm_builder=self.llm_builder,
                tool_registry_factory=self.tool_registry_factory,
                sub_agent_timeout_seconds=self.sub_agent_timeout_seconds,
                workspace_backend=sub_workspace_backend,
            )

            sub_agent_definition = InteractiveAgentDefinition(
                description=sub_agent.description,
                model=sub_task.model,
                backend=(
                    _trim_portable_whitespace(sub_agent.backend) or self.default_backend
                    if isinstance(sub_agent.backend, str)
                    else self.default_backend
                ),
                language=str(parent_task.metadata.get("language", "zh-CN")),
                max_cycles=sub_task.max_cycles,
                no_tool_policy=sub_task.no_tool_policy,
                allow_interruption=sub_task.allow_interruption,
                use_workspace=sub_task.use_workspace,
                enable_todo_management=True,
                agent_type=sub_task.agent_type,
                native_multimodal=sub_task.native_multimodal,
                enable_sub_agents=False,
                sub_agents={},
                extra_tool_names=list(sub_task.extra_tool_names),
                exclude_tools=list(sub_task.exclude_tools),
                metadata=dict(sub_task.metadata),
                system_prompt=sub_task.system_prompt,
            )

            sub_run_invocation = 0

            def _execute_sub_run(
                *,
                prompt: str,
                task_name: str,
                workspace: str | Path,
                shared_state: dict[str, Any] | None = None,
                initial_messages: list[Message] | None = None,
                before_cycle_messages: BeforeCycleMessageProvider | None = None,
                interruption_messages: InterruptionMessageProvider | None = None,
                log_handler: RuntimeLogHandler | None = None,
                cancellation_token: CancellationToken | None = None,
                session: Any | None = None,
                _sub_task_turn_snapshot: _SubTaskTurnSnapshot | None = None,
                **_: Any,
            ) -> AgentSessionRun:
                nonlocal sub_run_invocation
                is_initial_sub_run = sub_run_invocation == 0
                current_child_run_id = child_run_id if is_initial_sub_run else f"run_{uuid.uuid4().hex}"
                sub_run_invocation += 1
                if _sub_task_turn_snapshot is None:
                    current_parent_ctx = ctx
                    current_event_emitter = event_emitter
                    current_trace_id = trace_id
                    current_parent_run_id = parent_run_id
                    current_parent_tool_call_id = parent_tool_call_id
                    current_policy_metadata = self._trusted_parent_tool_policy_metadata(
                        parent_task=parent_task,
                        ctx=ctx,
                    )
                else:
                    current_parent_ctx = self._context_from_turn_snapshot(
                        base_ctx=ctx,
                        snapshot=_sub_task_turn_snapshot,
                    )
                    current_event_emitter = _sub_task_turn_snapshot.event_sink
                    current_trace_id = _sub_task_turn_snapshot.trace_id or current_child_run_id
                    current_parent_run_id = _sub_task_turn_snapshot.parent_run_id or ""
                    current_parent_tool_call_id = _sub_task_turn_snapshot.parent_tool_call_id or ""
                    current_policy_metadata = _sub_task_turn_snapshot.tool_policy_metadata()
                if not is_initial_sub_run:
                    _emit_sub_run_started_or_fail(
                        current_child_run_id,
                        emitter=current_event_emitter,
                        current_trace_id=current_trace_id,
                        current_parent_run_id=current_parent_run_id,
                        current_parent_tool_call_id=current_parent_tool_call_id,
                    )

                try:
                    run_metadata = dict(sub_task.metadata)
                    if _sub_task_turn_snapshot is not None:
                        for key in _TOOL_POLICY_METADATA_KEYS:
                            run_metadata.pop(key, None)
                    run_metadata.update(current_policy_metadata)
                    run_task = replace(
                        sub_task,
                        task_id=sub_task_id,
                        user_prompt=prompt,
                        metadata=self._canonical_sub_run_metadata(
                            run_metadata,
                            sub_task_id=sub_task_id,
                            sub_session_id=sub_session_id,
                            sub_agent_name=request.agent_name,
                            child_run_id=current_child_run_id,
                            trace_id=current_trace_id,
                            parent_run_id=current_parent_run_id,
                            parent_tool_call_id=current_parent_tool_call_id,
                        ),
                    )
                    run_shared_state = dict(shared_state or {})
                    run_shared_state.setdefault("todo_list", [])

                    child_ctx = self._build_child_ctx(
                        current_parent_ctx,
                        cancellation_token=cancellation_token,
                        child_run_id=current_child_run_id,
                        child_session_id=sub_session_id,
                        child_agent_name=request.agent_name,
                        child_model=sub_task.model,
                        child_workspace=workspace_path,
                        child_metadata=run_task.metadata,
                        trace_id=current_trace_id,
                        parent_run_id=current_parent_run_id,
                        parent_tool_call_id=current_parent_tool_call_id,
                    )
                    parent_stream_callback = child_ctx.stream_callback if child_ctx is not None else None
                    if log_handler is not None or parent_stream_callback is not None:

                        def _sub_stream_callback(event: dict[str, Any]) -> None:
                            canonical = _canonicalize_sub_agent_stream_event(
                                event,
                                task_id=sub_task_id,
                                session_id=sub_session_id,
                                sub_agent_name=request.agent_name,
                                child_run_id=current_child_run_id,
                                trace_id=current_trace_id,
                                parent_run_id=current_parent_run_id,
                                parent_tool_call_id=current_parent_tool_call_id,
                            )
                            if canonical is None:
                                return
                            event_name = canonical["event"]
                            log_payload = dict(canonical)
                            log_payload.pop("event", None)
                            if log_handler is not None:
                                log_handler(event_name, log_payload)
                            if parent_stream_callback is not None:
                                try:
                                    parent_stream_callback(canonical)
                                except BaseException:
                                    logging.getLogger(__name__).exception(
                                        "Configured sub-agent stream observer failed"
                                    )

                        if child_ctx is None:
                            child_ctx = ExecutionContext(stream_callback=_sub_stream_callback)
                        else:
                            child_ctx.stream_callback = _sub_stream_callback

                    previous_log_handler = sub_runtime.log_handler
                    sub_runtime.log_handler = log_handler
                    try:
                        if initial_messages is None and session is not None:
                            persisted_messages = list(session.get_items())
                            initial_messages = persisted_messages or None
                        sub_result = sub_runtime.run(
                            run_task,
                            workspace=workspace,
                            shared_state=run_shared_state,
                            initial_messages=initial_messages,
                            user_message=prompt,
                            before_cycle_messages=before_cycle_messages,
                            interruption_messages=interruption_messages,
                            ctx=child_ctx,
                        )
                    finally:
                        sub_runtime.log_handler = previous_log_handler
                    session_run = AgentSessionRun(
                        agent_name=task_name,
                        result=sub_result,
                        resolved=resolved_config,
                    )
                except BaseException as exc:
                    if not is_initial_sub_run:
                        _complete_sub_run_once(
                            current_child_run_id,
                            SubTaskOutcome(
                                task_id=sub_task_id,
                                session_id=sub_session_id,
                                agent_name=request.agent_name,
                                status=AgentStatus.FAILED,
                                completion_reason=CompletionReason.FAILED,
                                error=str(exc),
                                error_code="sub_task_failed",
                                cycles=0,
                                resolved=resolved_payload,
                            ),
                            emitter=current_event_emitter,
                            current_trace_id=current_trace_id,
                            current_parent_run_id=current_parent_run_id,
                            current_parent_tool_call_id=current_parent_tool_call_id,
                        )
                    raise

                if not is_initial_sub_run:
                    continuation_outcome = SubTaskOutcome(
                        task_id=sub_task_id,
                        session_id=sub_session_id,
                        agent_name=request.agent_name,
                        status=sub_result.status,
                        final_answer=sub_result.final_answer,
                        wait_reason=sub_result.wait_reason,
                        error=sub_result.error,
                        error_code=("sub_task_failed" if sub_result.status == AgentStatus.FAILED else None),
                        completion_reason=sub_result.completion_reason,
                        completion_tool_name=sub_result.completion_tool_name,
                        partial_output=sub_result.partial_output,
                        cycles=len(sub_result.cycles),
                        todo_list=sub_result.todo_list,
                        resolved=resolved_payload,
                    )

                    def _complete_persisted_continuation() -> None:
                        _complete_sub_run_once(
                            current_child_run_id,
                            continuation_outcome,
                            token_usage=self._sub_run_token_usage(sub_result),
                            budget_usage=sub_result.budget_usage,
                            budget_exhaustion=sub_result.budget_exhaustion,
                            emitter=current_event_emitter,
                            current_trace_id=current_trace_id,
                            current_parent_run_id=current_parent_run_id,
                            current_parent_tool_call_id=current_parent_tool_call_id,
                        )

                    def _fail_unpersisted_continuation(error: BaseException) -> None:
                        _complete_sub_run_once(
                            current_child_run_id,
                            SubTaskOutcome(
                                task_id=sub_task_id,
                                session_id=sub_session_id,
                                agent_name=request.agent_name,
                                status=AgentStatus.FAILED,
                                completion_reason=CompletionReason.FAILED,
                                error=str(error),
                                error_code="sub_task_failed",
                                cycles=len(sub_result.cycles),
                                resolved=resolved_payload,
                            ),
                            token_usage=self._sub_run_token_usage(sub_result),
                            budget_usage=sub_result.budget_usage,
                            budget_exhaustion=sub_result.budget_exhaustion,
                            emitter=current_event_emitter,
                            current_trace_id=current_trace_id,
                            current_parent_run_id=current_parent_run_id,
                            current_parent_tool_call_id=current_parent_tool_call_id,
                        )

                    session_run._set_persistence_callbacks(
                        after_persist=_complete_persisted_continuation,
                        on_persist_failure=_fail_unpersisted_continuation,
                    )
                return session_run

            sub_session = create_agent_session(
                execute_run=_execute_sub_run,
                session_id=sub_session_id,
                agent_name=request.agent_name,
                definition=sub_agent_definition,
                workspace=workspace_path,
                shared_state={"todo_list": []},
                approval_broker=parent_metadata.get("_vv_agent_approval_broker"),
            )

            if sub_task_manager is not None:
                sub_task_manager.attach_session(
                    task_id=sub_task_id,
                    session_id=sub_session_id,
                    agent_name=request.agent_name,
                    task_title=request.task_description,
                    workspace_backend=sub_workspace_backend,
                    session=sub_session,
                    resolved=resolved_payload,
                    parent_run_id=parent_run_id or None,
                    parent_tool_call_id=parent_tool_call_id or None,
                    event_forwarder=_emit_sub_session_event,
                )
            else:
                sub_session.subscribe(_emit_sub_session_event)

            try:
                register_sub_agent_session(sub_session_id, sub_session)
                _emit_sub_session_event(
                    "session_created",
                    {
                        "agent_name": request.agent_name,
                        "model": sub_task.model,
                        "workspace": str(workspace_path),
                        "max_cycles": sub_task.max_cycles,
                    },
                )
                sub_run = sub_session.prompt(sub_task.user_prompt, auto_follow_up=False)
            finally:
                unregister_sub_agent_session(sub_session_id, sub_session)
        except BaseException as exc:
            error_code = "sub_task_failed"
            if isinstance(exc, _SubTaskContractError):
                error_code = exc.code
            outcome = SubTaskOutcome(
                task_id=sub_task_id,
                session_id=sub_session_id,
                agent_name=request.agent_name,
                status=AgentStatus.FAILED,
                completion_reason=CompletionReason.FAILED,
                error=str(exc),
                error_code=error_code,
            )
            _complete_sub_run_once(child_run_id, outcome)
            if sub_task_manager is not None:
                sub_task_manager.record_outcome(
                    sub_task_id,
                    outcome,
                    workspace_backend=outcome_workspace_backend,
                    parent_run_id=parent_run_id or None,
                    parent_tool_call_id=parent_tool_call_id or None,
                    execution_token=manager_execution_token,
                )
            return outcome

        outcome = SubTaskOutcome(
            task_id=sub_task_id,
            session_id=sub_session_id,
            agent_name=request.agent_name,
            status=sub_run.result.status,
            final_answer=sub_run.result.final_answer,
            wait_reason=sub_run.result.wait_reason,
            error=sub_run.result.error,
            error_code=("sub_task_failed" if sub_run.result.status == AgentStatus.FAILED else None),
            completion_reason=sub_run.result.completion_reason,
            completion_tool_name=sub_run.result.completion_tool_name,
            partial_output=sub_run.result.partial_output,
            cycles=len(sub_run.result.cycles),
            todo_list=sub_run.result.todo_list,
            resolved=resolved_payload,
        )
        _complete_sub_run_once(
            child_run_id,
            outcome,
            token_usage=self._sub_run_token_usage(sub_run.result),
            budget_usage=sub_run.result.budget_usage,
            budget_exhaustion=sub_run.result.budget_exhaustion,
        )
        if sub_task_manager is not None:
            sub_task_manager.record_outcome(
                sub_task_id,
                outcome,
                workspace_backend=outcome_workspace_backend,
                parent_run_id=parent_run_id or None,
                parent_tool_call_id=parent_tool_call_id or None,
                execution_token=manager_execution_token,
            )
        return outcome

    def _resolve_sub_agent_client(
        self,
        *,
        parent_task: AgentTask,
        sub_agent: SubAgentConfig,
        ctx: ExecutionContext | None,
    ) -> tuple[LLMClient, str, dict[str, str], ResolvedModelConfig]:
        requested_model = _trim_portable_whitespace(str(sub_agent.model))
        requested_backend = _trim_portable_whitespace(str(sub_agent.backend or "")) or None
        backend = requested_backend or _trim_portable_whitespace(str(self.default_backend or "")) or "inline"
        model_provider = ctx.metadata.get("_vv_agent_model_provider") if ctx is not None else None
        if model_provider is not None and hasattr(model_provider, "resolve") and hasattr(model_provider, "client"):
            from vv_agent.model import ModelRef

            model_ref = (
                ModelRef.backend(requested_backend, requested_model)
                if requested_backend is not None
                else ModelRef.named(requested_model)
            )
            resolved = model_provider.resolve(model_ref)
            return self._resolved_sub_agent_client(model_provider.client(resolved), resolved)
        if self.settings_file is None:
            if requested_backend is not None:
                raise ValueError(
                    "Sub-agent model resolution requires a model provider or settings_file when backend is explicit."
                )
            if requested_model != parent_task.model:
                raise ValueError(
                    "Sub-agent model resolution requires runtime settings_file when sub-agent model differs from parent model."
                )
            fallback_endpoint = EndpointConfig(
                endpoint_id="inline-session",
                api_key="",
                api_base="",
            )
            fallback_resolved = ResolvedModelConfig(
                backend=backend,
                requested_model=parent_task.model,
                selected_model=parent_task.model,
                model_id=parent_task.model,
                endpoint_options=[EndpointOption(endpoint=fallback_endpoint, model_id=parent_task.model)],
                context_length=self._metadata_token_limit(
                    parent_task.metadata,
                    "model_context_window",
                    minimum=1,
                ),
                max_output_tokens=self._metadata_token_limit(
                    parent_task.metadata,
                    "model_max_output_tokens",
                    minimum=0,
                ),
                native_multimodal=parent_task.native_multimodal,
            )
            return self.llm_client, parent_task.model, {}, fallback_resolved

        if not backend:
            raise ValueError("Sub-agent backend is required when settings_file is configured.")

        llm_client, resolved = self.llm_builder(
            self.settings_file,
            backend=backend,
            model=requested_model,
            timeout_seconds=self.sub_agent_timeout_seconds,
        )
        return self._resolved_sub_agent_client(llm_client, resolved)

    @staticmethod
    def _resolved_sub_agent_client(
        llm_client: LLMClient,
        resolved: ResolvedModelConfig,
    ) -> tuple[LLMClient, str, dict[str, str], ResolvedModelConfig]:
        payload = {
            "backend": resolved.backend,
            "selected_model": resolved.selected_model,
            "model_id": resolved.model_id,
        }
        if resolved.endpoint_options:
            payload["endpoint"] = resolved.endpoint.endpoint_id
        return llm_client, resolved.model_id, payload, resolved

    @staticmethod
    def _metadata_token_limit(metadata: dict[str, Any], key: str, *, minimum: int) -> int | None:
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            return None
        return value

    @staticmethod
    def _validate_sub_agent_config(sub_agent: SubAgentConfig) -> None:
        if not isinstance(sub_agent.model, str) or not _trim_portable_whitespace(sub_agent.model):
            raise _SubTaskContractError(_INVALID_SUB_AGENT_MODEL_CODE, _INVALID_SUB_AGENT_MODEL_MESSAGE)
        if sub_agent.system_prompt is not None and (
            not isinstance(sub_agent.system_prompt, str)
            or not _trim_portable_whitespace(sub_agent.system_prompt)
        ):
            raise _SubTaskContractError(
                _INVALID_SUB_AGENT_SYSTEM_PROMPT_CODE,
                _INVALID_SUB_AGENT_SYSTEM_PROMPT_MESSAGE,
            )

    @staticmethod
    def _sub_agent_workspace_backend(
        workspace_backend: WorkspaceBackend,
        exclude_files_pattern: str | None,
    ) -> WorkspaceBackend:
        if exclude_files_pattern is None or not exclude_files_pattern.strip():
            return workspace_backend
        try:
            return DiscoveryFilteredWorkspaceBackend(workspace_backend, exclude_files_pattern)
        except InvalidPortableRegexError as exc:
            raise _SubTaskContractError(
                INVALID_EXCLUDE_FILES_PATTERN_CODE,
                INVALID_EXCLUDE_FILES_PATTERN_MESSAGE,
            ) from exc

    @staticmethod
    def _effective_parent_model_settings(
        *,
        parent_task: AgentTask,
        ctx: ExecutionContext | None,
    ) -> ModelSettings | None:
        if ctx is not None:
            value = ctx.metadata.get("_vv_agent_model_settings")
            if isinstance(value, ModelSettings):
                return value
        return parent_task.model_settings

    @staticmethod
    def _trusted_parent_tool_policy_metadata(
        *,
        parent_task: AgentTask,
        ctx: ExecutionContext | None,
    ) -> dict[str, Any]:
        projected: dict[str, Any] = {}
        task_metadata = parent_task.metadata if isinstance(parent_task.metadata, dict) else {}
        runtime_metadata = ctx.metadata if ctx is not None and isinstance(ctx.metadata, dict) else {}
        for key in ("_vv_agent_allowed_tools", "_vv_agent_disallowed_tools"):
            value = runtime_metadata.get(key, task_metadata.get(key))
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                projected[key] = list(value)
        denied_side_effects = runtime_metadata.get(
            "_vv_agent_denied_side_effects",
            task_metadata.get("_vv_agent_denied_side_effects"),
        )
        if isinstance(denied_side_effects, list):
            projected["_vv_agent_denied_side_effects"] = [
                item.value for item in normalize_denied_side_effects(denied_side_effects)
            ]
        denied_capability_tags = runtime_metadata.get(
            "_vv_agent_denied_capability_tags",
            task_metadata.get("_vv_agent_denied_capability_tags"),
        )
        if isinstance(denied_capability_tags, list):
            projected["_vv_agent_denied_capability_tags"] = normalize_metadata_labels(
                denied_capability_tags,
                field_name="denied_capability_tags",
            )
        deny_terminal_tools = runtime_metadata.get(
            "_vv_agent_deny_terminal_tools",
            task_metadata.get("_vv_agent_deny_terminal_tools"),
        )
        if isinstance(deny_terminal_tools, bool):
            projected["_vv_agent_deny_terminal_tools"] = deny_terminal_tools
        denied_cost_dimensions = runtime_metadata.get(
            "_vv_agent_denied_cost_dimensions",
            task_metadata.get("_vv_agent_denied_cost_dimensions"),
        )
        if isinstance(denied_cost_dimensions, list):
            projected["_vv_agent_denied_cost_dimensions"] = normalize_metadata_labels(
                denied_cost_dimensions,
                field_name="denied_cost_dimensions",
            )
        can_use_tool = runtime_metadata.get("_vv_agent_tool_policy_can_use_tool")
        if callable(can_use_tool):
            projected["_vv_agent_tool_policy_can_use_tool"] = can_use_tool
        approval = runtime_metadata.get("_vv_agent_tool_policy_approval")
        if isinstance(approval, str) and approval in {"always", "never", "on_request"}:
            projected["_vv_agent_tool_policy_approval"] = approval
        return projected

    @staticmethod
    def _configured_child_metadata_denials(
        parent_policy_metadata: dict[str, Any],
        sub_agent: SubAgentConfig,
    ) -> dict[str, Any]:
        denied_side_effects = normalize_denied_side_effects(
            [
                *parent_policy_metadata.get("_vv_agent_denied_side_effects", []),
                *sub_agent.denied_side_effects,
            ]
        )
        denied_capability_tags = normalize_metadata_labels(
            [
                *parent_policy_metadata.get("_vv_agent_denied_capability_tags", []),
                *sub_agent.denied_capability_tags,
            ],
            field_name="denied_capability_tags",
        )
        denied_cost_dimensions = normalize_metadata_labels(
            [
                *parent_policy_metadata.get("_vv_agent_denied_cost_dimensions", []),
                *sub_agent.denied_cost_dimensions,
            ],
            field_name="denied_cost_dimensions",
        )
        projected: dict[str, Any] = {}
        if denied_side_effects:
            projected["_vv_agent_denied_side_effects"] = [
                item.value for item in denied_side_effects
            ]
        if denied_capability_tags:
            projected["_vv_agent_denied_capability_tags"] = denied_capability_tags
        if (
            parent_policy_metadata.get("_vv_agent_deny_terminal_tools") is True
            or sub_agent.deny_terminal_tools
        ):
            projected["_vv_agent_deny_terminal_tools"] = True
        if denied_cost_dimensions:
            projected["_vv_agent_denied_cost_dimensions"] = denied_cost_dimensions
        return projected

    @staticmethod
    def _context_from_turn_snapshot(
        *,
        base_ctx: ExecutionContext | None,
        snapshot: _SubTaskTurnSnapshot,
    ) -> ExecutionContext:
        del base_ctx
        metadata = dict(snapshot.execution_metadata)
        if snapshot.event_sink is not None:
            metadata["_vv_agent_emit_event"] = snapshot.event_sink
        if snapshot.run_context is not None:
            metadata["_vv_agent_run_context"] = snapshot.run_context
        if snapshot.parent_run_id:
            metadata["_vv_agent_run_id"] = snapshot.parent_run_id
        if snapshot.trace_id:
            metadata["_vv_agent_trace_id"] = snapshot.trace_id
            metadata["trace_id"] = snapshot.trace_id
        metadata.update(snapshot.tool_policy_metadata())
        return ExecutionContext(
            cancellation_token=snapshot.cancellation_token,
            stream_callback=snapshot.stream_callback,
            state_store=snapshot.state_store,
            metadata=metadata,
        )

    def _build_sub_agent_task(
        self,
        *,
        parent_task: AgentTask,
        sub_task_id: str,
        sub_session_id: str,
        sub_agent_name: str,
        sub_agent: SubAgentConfig,
        resolved_model_id: str,
        child_run_id: str,
        trace_id: str,
        parent_run_id: str,
        parent_tool_call_id: str,
        request: SubTaskRequest,
        parent_shared_state: dict[str, Any],
        workspace_path: Path,
        resolved_native_multimodal: bool | None = None,
        resolved_context_length: int | None = None,
        resolved_max_output_tokens: int | None = None,
        effective_model_settings: ModelSettings | None = None,
        parent_tool_policy_metadata: dict[str, Any] | None = None,
    ) -> AgentTask:
        language = str(parent_task.metadata.get("language", "zh-CN"))
        available_skills = parent_task.metadata.get("available_skills")
        if not isinstance(available_skills, list):
            available_skills = None

        generated_sections: list[dict[str, Any]] = []
        if sub_agent.system_prompt:
            system_prompt = sub_agent.system_prompt
            generated_sections = build_raw_system_prompt_sections(system_prompt)
        else:
            prompt_bundle = build_system_prompt_bundle(
                sub_agent.description,
                language=language,
                allow_interruption=False,
                use_workspace=parent_task.use_workspace,
                enable_todo_management=True,
                agent_type=parent_task.agent_type,
                available_skills=available_skills,
                workspace=workspace_path,
            )
            system_prompt = prompt_bundle.prompt
            generated_sections = prompt_bundle.sections

        user_prompt = request.task_description
        if request.output_requirements:
            user_prompt = f"{user_prompt}\n\n<Output Requirements>\n{request.output_requirements}\n</Output Requirements>"
        if request.include_main_summary:
            parent_summary = self._build_parent_summary(parent_task=parent_task, parent_shared_state=parent_shared_state)
            if parent_summary:
                user_prompt = f"{user_prompt}\n\n<Main Task Summary>\n{parent_summary}\n</Main Task Summary>"

        excluded_tools = set(parent_task.exclude_tools)
        excluded_tools.update(sub_agent.exclude_tools)
        excluded_tools.update({CREATE_SUB_TASK_TOOL_NAME, SUB_TASK_STATUS_TOOL_NAME})
        metadata: dict[str, Any] = {
            "is_sub_task": True,
            "parent_task_id": parent_task.task_id,
            "sub_agent_name": sub_agent_name,
            "session_memory_enabled": False,
            "workspace": str(workspace_path),
        }
        for key in (
            "bash_shell",
            "windows_shell_priority",
            "bash_env",
            "allow_outside_workspace_paths",
            "allow_outside_workspace",
            "workspace_allow_outside_main",
            "workspace_allow_outside",
            "language",
            "available_skills",
            "active_skills",
        ):
            value = parent_task.metadata.get(key)
            if value is not None:
                metadata[key] = value
        if sub_agent.metadata:
            metadata.update(sub_agent.metadata)
        if request.metadata:
            metadata.update(request.metadata)
        for key in _RESERVED_SUB_AGENT_METADATA_KEYS:
            metadata.pop(key, None)
        metadata.update(parent_tool_policy_metadata or {})
        metadata.update(
            self._configured_child_metadata_denials(
                parent_tool_policy_metadata or {},
                sub_agent,
            )
        )
        if resolved_context_length is not None:
            metadata.setdefault("model_context_window", resolved_context_length)
        if resolved_max_output_tokens is not None:
            metadata.setdefault("model_max_output_tokens", resolved_max_output_tokens)
        if generated_sections:
            metadata.setdefault("system_prompt_sections", generated_sections)
        metadata.update(
            {
                "is_sub_task": True,
                "parent_task_id": parent_task.task_id,
                "sub_agent_name": sub_agent_name,
                "session_memory_enabled": False,
                "workspace": str(workspace_path),
            }
        )
        metadata = self._canonical_sub_run_metadata(
            metadata,
            sub_task_id=sub_task_id,
            sub_session_id=sub_session_id,
            sub_agent_name=sub_agent_name,
            child_run_id=child_run_id,
            trace_id=trace_id,
            parent_run_id=parent_run_id,
            parent_tool_call_id=parent_tool_call_id,
        )

        return AgentTask(
            task_id=sub_task_id,
            model=resolved_model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_cycles=max(sub_agent.max_cycles, 1),
            memory_compact_threshold=parent_task.memory_compact_threshold,
            memory_threshold_percentage=parent_task.memory_threshold_percentage,
            no_tool_policy="continue",
            allow_interruption=False,
            use_workspace=parent_task.use_workspace,
            has_sub_agents=False,
            sub_agents={},
            agent_type=parent_task.agent_type,
            native_multimodal=(
                parent_task.native_multimodal if resolved_native_multimodal is None else resolved_native_multimodal
            ),
            extra_tool_names=list(parent_task.extra_tool_names),
            exclude_tools=sorted(excluded_tools),
            model_settings=(
                parent_task.model_settings if effective_model_settings is None else effective_model_settings
            ),
            metadata=metadata,
        )

    @staticmethod
    def _canonical_sub_run_metadata(
        metadata: dict[str, Any],
        *,
        sub_task_id: str,
        sub_session_id: str,
        sub_agent_name: str,
        child_run_id: str,
        trace_id: str,
        parent_run_id: str,
        parent_tool_call_id: str,
    ) -> dict[str, Any]:
        canonical = dict(metadata)
        for key in (
            "browser_scope_key",
            "session_id",
            "sub_agent_name",
            "task_id",
            "run_id",
            "trace_id",
            "parent_run_id",
            "parent_tool_call_id",
            "_vv_agent_run_id",
            "_vv_agent_trace_id",
            "_vv_agent_agent_name",
            "_vv_agent_session_id",
            "_vv_agent_parent_run_id",
            "_vv_agent_parent_tool_call_id",
        ):
            canonical.pop(key, None)
        canonical.update(
            {
                "task_id": sub_task_id,
                "session_id": sub_session_id,
                "browser_scope_key": sub_session_id,
                "sub_agent_name": sub_agent_name,
                "run_id": child_run_id,
                "trace_id": trace_id,
                "_vv_agent_run_id": child_run_id,
                "_vv_agent_trace_id": trace_id,
                "_vv_agent_agent_name": sub_agent_name,
                "_vv_agent_session_id": sub_session_id,
            }
        )
        if parent_run_id:
            canonical["parent_run_id"] = parent_run_id
            canonical["_vv_agent_parent_run_id"] = parent_run_id
        if parent_tool_call_id:
            canonical["parent_tool_call_id"] = parent_tool_call_id
            canonical["_vv_agent_parent_tool_call_id"] = parent_tool_call_id
        return canonical

    @staticmethod
    def _sub_run_token_usage(result: AgentResult) -> dict[str, Any] | None:
        if result.status == AgentStatus.FAILED and not result.cycles:
            return None
        return result.token_usage.to_dict()

    def _build_sub_agent_registry(self) -> ToolRegistry:
        if self.tool_registry_factory is not None:
            return self.tool_registry_factory()
        return self.tool_registry

    @staticmethod
    def _build_child_ctx(
        ctx: ExecutionContext | None,
        *,
        stream_callback: StreamCallback | None = None,
        cancellation_token: CancellationToken | None = None,
        child_run_id: str = "",
        child_session_id: str = "",
        child_agent_name: str = "",
        child_model: str = "",
        child_workspace: str | Path | WorkspaceBackend | None = None,
        child_metadata: dict[str, Any] | None = None,
        trace_id: str = "",
        parent_run_id: str = "",
        parent_tool_call_id: str = "",
    ) -> ExecutionContext | None:
        from vv_agent.agent import RunContext

        child_stream_callback = stream_callback if stream_callback is not None else (ctx.stream_callback if ctx else None)
        parent_child_token = ctx.cancellation_token.child() if ctx is not None and ctx.cancellation_token else None
        child_token = cancellation_token or parent_child_token
        if cancellation_token is not None and parent_child_token is not None:
            parent_child_token.on_cancel(
                lambda: cancellation_token.cancel(parent_child_token.reason or "Operation was cancelled")
            )
        metadata = AgentRuntime._inherited_child_context_metadata(ctx.metadata) if ctx is not None else {}
        if child_run_id:
            metadata["_vv_agent_run_id"] = child_run_id
        if child_agent_name:
            metadata["_vv_agent_agent_name"] = child_agent_name
        if child_session_id:
            metadata["_vv_agent_session_id"] = child_session_id
        if trace_id:
            metadata["_vv_agent_trace_id"] = trace_id
            metadata["trace_id"] = trace_id
        else:
            metadata.pop("_vv_agent_trace_id", None)
            metadata.pop("trace_id", None)
        if parent_run_id:
            metadata["_vv_agent_parent_run_id"] = parent_run_id
        if parent_tool_call_id:
            metadata["_vv_agent_parent_tool_call_id"] = parent_tool_call_id

        parent_run_context = ctx.metadata.get("_vv_agent_run_context") if ctx is not None else None
        app_state = parent_run_context.context if isinstance(parent_run_context, RunContext) else None
        if child_run_id or child_agent_name or child_model or child_workspace is not None:
            run_context_metadata = dict(child_metadata or {})
            if trace_id:
                run_context_metadata["trace_id"] = trace_id
            if parent_run_id:
                run_context_metadata["parent_run_id"] = parent_run_id
            if parent_tool_call_id:
                run_context_metadata["parent_tool_call_id"] = parent_tool_call_id
            metadata["_vv_agent_run_context"] = RunContext(
                context=app_state,
                run_id=child_run_id,
                agent_name=child_agent_name,
                model=child_model or None,
                workspace=child_workspace,
                metadata=run_context_metadata,
            )

        if ctx is None and child_token is None and child_stream_callback is None and not metadata:
            return None
        return ExecutionContext(
            cancellation_token=child_token,
            stream_callback=child_stream_callback,
            state_store=ctx.state_store if ctx is not None else None,
            metadata=metadata,
        )

    @staticmethod
    def _inherited_child_context_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        inherited_keys = {
            "_vv_agent_approval_provider",
            "_vv_agent_approval_broker",
            "_vv_agent_approval_timeout_seconds",
            "_vv_agent_budget_limits",
            "_vv_agent_emit_event",
            "_vv_agent_memory_providers",
            "_vv_agent_model_provider",
            "_vv_agent_model_settings",
            "_vv_agent_allowed_tools",
            "_vv_agent_disallowed_tools",
            "_vv_agent_tool_policy_approval",
            "_vv_agent_tool_policy_can_use_tool",
            "_vv_agent_denied_side_effects",
            "_vv_agent_denied_capability_tags",
            "_vv_agent_deny_terminal_tools",
            "_vv_agent_denied_cost_dimensions",
            "_vv_agent_trace_context",
            "_vv_agent_trace_id",
            "trace_context",
            "trace_id",
        }
        return {key: value for key, value in metadata.items() if key in inherited_keys}

    @staticmethod
    def _build_parent_summary(*, parent_task: AgentTask, parent_shared_state: dict[str, Any]) -> str:
        lines = [f"Parent task goal: {parent_task.user_prompt}"]
        todo_list = parent_shared_state.get("todo_list")
        if isinstance(todo_list, list) and todo_list:
            lines.append("Parent TODO status:")
            for item in todo_list:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "Untitled"))
                status = str(item.get("status", "pending"))
                lines.append(f"- [{status}] {title}")
        return "\n".join(lines)
