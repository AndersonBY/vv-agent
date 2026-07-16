from __future__ import annotations

import json
import uuid
import warnings
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, cast

from vv_agent.agent import Agent, RunContext
from vv_agent.approval import ApprovalBroker, ApprovalError, ApprovalRequest, bind_request_cancellation
from vv_agent.background_task import BackgroundAgentTask
from vv_agent.budget import BudgetEnforcementBoundary, BudgetEvaluator, BudgetUsageSnapshot
from vv_agent.config import (
    EndpointConfig,
    EndpointOption,
    ResolvedModelConfig,
    build_openai_llm_from_local_settings,
    load_llm_settings_from_file,
)
from vv_agent.events import (
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    BudgetExhaustedEvent,
    BudgetSnapshotEvent,
    HandoffCompletedEvent,
    HandoffEvent,
    HandoffStartedEvent,
    RunCancelledEvent,
    RunCompletedEvent,
    RunEvent,
    RunFailedEvent,
    SessionPersistedEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
    event_from_runtime_log,
    event_from_stream_payload,
    new_trace_id,
)
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm.base import LLMClient
from vv_agent.model import ModelRef
from vv_agent.model_settings import ModelSettings
from vv_agent.result import ApprovalSnapshot, RunResult, RunState, _PendingToolApproval, _RunResumeContext
from vv_agent.run_config import (
    DEFAULT_SETTINGS_FILE,
    DEFAULT_TIMEOUT_SECONDS,
    RunConfig,
    ToolPolicy,
    _validate_bounded_int,
    merge_tool_policy_layers,
)
from vv_agent.run_handle import RunHandle, RunHandleRunner
from vv_agent.runtime import AgentRuntime, ToolCallRunner
from vv_agent.runtime.compiler import AgentCompiler
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.engine import _CanonicalSubAgentStreamPayload
from vv_agent.tools import ToolContext, ToolExposure, ToolSpec, build_default_registry
from vv_agent.tools.executor import ToolExecutor, is_tool_executor
from vv_agent.tools.function import FunctionTool, Tool, adapt_tool
from vv_agent.tools.registry import ToolRegistry
from vv_agent.tracing import Span, TraceProcessor
from vv_agent.types import (
    AgentResult,
    AgentStatus,
    AgentTask,
    CompletionReason,
    Message,
    ToolDirective,
    ToolExecutionResult,
    ToolResultStatus,
    _last_assistant_output,
)

_TOOL_POLICY_METADATA_KEYS = (
    "_vv_agent_allowed_tools",
    "_vv_agent_disallowed_tools",
    "_vv_agent_tool_policy_approval",
    "_vv_agent_tool_policy_can_use_tool",
)
_PLANNED_TOOL_NAMES_METADATA_KEY = "_vv_agent_planned_tool_names"
_INITIAL_BUDGET_USAGE_METADATA_KEY = "_vv_agent_initial_budget_usage"
_RUNTIME_TERMINAL_LOG_EVENTS = frozenset(
    {
        "cycle_failed",
        "budget_exhausted",
        "budget_snapshot",
        "run_cancelled",
        "run_completed",
        "run_failed",
        "run_max_cycles",
        "run_wait_user",
    }
)


def _tool_policy_metadata(policy: ToolPolicy | None) -> dict[str, Any]:
    if policy is None:
        return {}
    metadata: dict[str, Any] = {}
    if policy.allowed_tools is not None:
        metadata["_vv_agent_allowed_tools"] = list(policy.allowed_tools)
    if policy.disallowed_tools:
        metadata["_vv_agent_disallowed_tools"] = list(policy.disallowed_tools)
    if policy.can_use_tool is not None:
        metadata["_vv_agent_tool_policy_can_use_tool"] = policy.can_use_tool
    if policy.approval != "default":
        metadata["_vv_agent_tool_policy_approval"] = policy.approval
    return metadata


@dataclass(frozen=True, slots=True)
class _HandoffRequest:
    source_agent: str
    target_agent: str
    tool_name: str
    tool_call_id: str
    input: str
    cycle_index: int | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PendingHandoff:
    request: _HandoffRequest
    source_run_id: str
    source_trace_id: str
    session_id: str | None
    event_config: RunConfig


@dataclass(frozen=True, slots=True)
class _CompiledTaskInvocation:
    task: AgentTask


class _RunTrace:
    def __init__(
        self,
        *,
        processors: list[TraceProcessor],
        trace_id: str,
        agent_name: str,
        workflow_name: str | None,
    ) -> None:
        self.processors = processors
        self.trace_id = trace_id
        self.agent_name = agent_name
        self.run_span = self._start(
            Span(
                name="run",
                trace_id=trace_id,
                metadata={"agent_name": agent_name, "workflow_name": workflow_name},
            )
        )
        self.agent_span = self._start(
            Span(
                name="agent",
                trace_id=trace_id,
                parent_id=self.run_span.span_id,
                metadata={"agent_name": agent_name},
            )
        )
        self.tool_spans: dict[str, Span] = {}
        self.ended_run_span: Span | None = None

    def on_event(self, event: RunEvent) -> None:
        if isinstance(event, ToolStartedEvent):
            previous = self.tool_spans.pop(event.tool_call_id, None)
            if previous is not None:
                self._end(previous, {"status": "replaced"})
            self.tool_spans[event.tool_call_id] = self._start(
                Span(
                    name="tool",
                    trace_id=self.trace_id,
                    parent_id=self.agent_span.span_id,
                    metadata={"tool_name": event.tool_name, "agent_name": self.agent_name},
                )
            )
        elif isinstance(event, ToolFinishedEvent | HandoffEvent):
            span = self.tool_spans.pop(event.tool_call_id, None)
            if span is not None:
                self._end(span, event.to_dict())

    def finish(self, metadata: dict[str, Any] | None = None) -> Span:
        if self.ended_run_span is not None:
            return self.ended_run_span
        for span in reversed(list(self.tool_spans.values())):
            self._end(span, {"status": "abandoned"})
        self.tool_spans.clear()
        self._end(self.agent_span, metadata)
        self.ended_run_span = self._end(self.run_span, metadata)
        for processor in self.processors:
            flush = getattr(processor, "flush", None)
            if callable(flush):
                self._notify(processor, "flush", flush)
        return self.ended_run_span

    def _start(self, span: Span) -> Span:
        for processor in self.processors:
            self._notify(processor, "on_span_start", processor.on_span_start, span)
        return span

    def _end(self, span: Span, metadata: dict[str, Any] | None = None) -> Span:
        ended = span.finish(metadata=metadata)
        for processor in self.processors:
            self._notify(processor, "on_span_end", processor.on_span_end, ended)
        return ended

    @staticmethod
    def _notify(processor: TraceProcessor, operation: str, callback: Callable[..., Any], *args: Any) -> None:
        try:
            callback(*args)
        except BaseException as exc:
            warnings.warn(
                f"Trace processor {type(processor).__name__}.{operation} failed: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )


class Runner:
    @classmethod
    def configured(cls, default_run_config: RunConfig | None = None) -> ConfiguredRunner:
        return ConfiguredRunner(default_run_config=default_run_config or RunConfig())

    @classmethod
    def run_sync(cls, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunResult:
        return cls._run(agent, input, run_config=run_config or RunConfig())

    @classmethod
    def stream_sync(cls, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> Iterator[RunEvent]:
        handle = cls.start(agent, input, run_config=run_config)
        yield from handle.events()
        handle.result()

    @classmethod
    def start(cls, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunHandle:
        return RunHandle._start_worker(
            agent=agent,
            input=input,
            run_config=run_config or RunConfig(),
            runner=cast(RunHandleRunner, cls),
        )

    @classmethod
    def _run_compiled_sync(
        cls,
        agent: Agent,
        input: str,
        *,
        task: AgentTask,
        run_config: RunConfig | None = None,
    ) -> RunResult:
        return cls._run(
            agent,
            input,
            run_config=run_config or RunConfig(),
            _compiled_invocation=_CompiledTaskInvocation(task=task),
        )

    @classmethod
    def _start_compiled(
        cls,
        agent: Agent,
        input: str,
        *,
        task: AgentTask,
        run_config: RunConfig | None = None,
    ) -> RunHandle:
        return RunHandle._start_worker(
            agent=agent,
            input=input,
            run_config=run_config or RunConfig(),
            runner=cast(RunHandleRunner, cls),
            _compiled_invocation=_CompiledTaskInvocation(task=task),
        )

    @classmethod
    def resume(cls, state: RunState, *, input: str | None = None) -> RunResult:
        source, approved_ids = state._into_inner()
        resume_context = source._resume_context
        if resume_context is None:
            raise ValueError("run state does not include resume context")

        approved = next(
            (snapshot for snapshot in source.approvals if snapshot.interruption_id in approved_ids),
            None,
        )
        if approved is not None:
            return cls._resume_approved_tool_call(source, resume_context, approved, resume_input=input)

        config = replace(
            resume_context.run_config,
            initial_messages=deepcopy(source.raw_result.messages),
            shared_state=deepcopy(source.raw_result.shared_state),
            metadata=cls._metadata_with_initial_budget_usage(
                resume_context.run_config.metadata,
                source.budget_usage,
            ),
        )
        return resume_context.runner._run(
            resume_context.agent,
            resume_context.input if input is None else input,
            run_config=config,
        )

    @classmethod
    def _resume_approved_tool_call(
        cls,
        source: RunResult,
        resume_context: _RunResumeContext,
        approval: ApprovalSnapshot,
        *,
        resume_input: str | None,
    ) -> RunResult:
        effective_config = resume_context.effective_run_config or resume_context.run_config
        pending = resume_context.pending_tool_approval
        if pending is None or pending.interruption_id != approval.interruption_id:
            raise ValueError("approved tool call is missing its captured interruption context")
        if not cls._approval_snapshot_matches(source, approval, pending):
            raise ValueError("approved tool call does not match the captured interruption")
        if resume_input is not None:
            raise ValueError("input cannot be provided when resuming an approved tool call")
        cancellation_token = effective_config.cancellation_token
        if cancellation_token is not None and cancellation_token.cancelled:
            return cls._cancelled_approval_resume_result(
                source,
                resume_context,
                approval,
                effective_config=effective_config,
            )
        if not resume_context.claim_approval(approval.interruption_id):
            raise RuntimeError("approval_already_consumed")
        resumed_run_id = f"run_{uuid.uuid4().hex}"
        resume_budget_evaluator: BudgetEvaluator | None = None
        if effective_config.budget_limits is not None and effective_config.budget_limits.has_limits:
            resume_budget_evaluator = BudgetEvaluator(
                effective_config.budget_limits,
                host_cost_meter=effective_config.host_cost_meter,
                initial_usage=source.budget_usage,
            )
        call = pending.call
        context = replace(
            pending.context,
            shared_state=deepcopy(source.raw_result.shared_state),
        )
        if context.ctx is None:
            raise ValueError("captured approval context is missing its execution context")
        context.ctx._approved_tool_approval = pending
        tool_result = pending.orchestrator.run_one(
            call,
            context=context,
            allowed_tool_names=pending.allowed_tool_names,
        )
        tool_result = pending.hook_manager.apply_after_tool_call(
            task=pending.task,
            cycle_index=pending.cycle_index,
            call=call,
            context=context,
            result=tool_result,
        )

        behavior_reason = ToolCallRunner._apply_tool_use_behavior(
            task=pending.task,
            call=call,
            result=tool_result,
        )

        raw_result = deepcopy(source.raw_result)
        raw_result.shared_state = deepcopy(context.shared_state)
        for cycle in raw_result.cycles:
            if cycle.index == approval.cycle_index:
                replaced = False
                for index, existing in enumerate(cycle.tool_results):
                    if (
                        existing.tool_call_id == pending.call.id
                        and existing.metadata.get("approval_interruption_id") == pending.interruption_id
                    ):
                        cycle.tool_results[index] = tool_result
                        replaced = True
                        break
                if not replaced:
                    cycle.tool_results.append(tool_result)
                break
        tool_message = tool_result.to_tool_message()
        raw_result.messages = [
            message for message in raw_result.messages if not (message.role == "tool" and message.tool_call_id == pending.call.id)
        ]
        raw_result.messages.append(tool_message)
        if effective_config.session is not None:
            effective_config.session.add_items([tool_message])

        resume_budget_events: list[RunEvent] = []
        if resume_budget_evaluator is not None:
            exhaustion = resume_budget_evaluator.tool_batch_complete()
            raw_result.budget_usage = resume_budget_evaluator.snapshot()
            raw_result.budget_exhaustion = exhaustion
            if exhaustion is not None:
                raw_result.status = AgentStatus.FAILED
                raw_result.completion_reason = CompletionReason.BUDGET_EXHAUSTED
                raw_result.completion_tool_name = None
                raw_result.partial_output = _last_assistant_output(raw_result.cycles)
                raw_result.final_answer = None
                raw_result.wait_reason = None
                raw_result.error = "Run budget exhausted."
                budget_event: RunEvent = BudgetExhaustedEvent(
                    run_id=resumed_run_id,
                    trace_id=source.trace_id,
                    agent_name=source.agent_name,
                    session_id=cls._resolve_event_session_id(effective_config),
                    cycle_index=approval.cycle_index,
                    enforcement_boundary=BudgetEnforcementBoundary.TOOL_BATCH_COMPLETE,
                    budget_usage=raw_result.budget_usage,
                    budget_exhaustion=exhaustion,
                )
            else:
                budget_event = BudgetSnapshotEvent(
                    run_id=resumed_run_id,
                    trace_id=source.trace_id,
                    agent_name=source.agent_name,
                    session_id=cls._resolve_event_session_id(effective_config),
                    cycle_index=approval.cycle_index,
                    enforcement_boundary=BudgetEnforcementBoundary.TOOL_BATCH_COMPLETE,
                    budget_usage=raw_result.budget_usage,
                )
            resume_budget_events.append(budget_event)
            cls._emit_chain_event(budget_event, run_config=effective_config, event_sink=None)

        if raw_result.completion_reason != CompletionReason.BUDGET_EXHAUSTED and tool_result.directive == ToolDirective.CONTINUE:
            tracing = dict(resume_context.run_config.tracing) if isinstance(resume_context.run_config.tracing, dict) else {}
            tracing["trace_id"] = source.trace_id
            config = replace(
                resume_context.run_config,
                initial_messages=deepcopy(raw_result.messages),
                shared_state=deepcopy(raw_result.shared_state),
                tracing=tracing,
                metadata=cls._metadata_with_initial_budget_usage(
                    resume_context.run_config.metadata,
                    raw_result.budget_usage,
                ),
            )
            continued = resume_context.runner._run(
                resume_context.agent,
                resume_context.input,
                run_config=config,
            )
            continued.metadata.update(
                {
                    "resumed": True,
                    "approved_interruption_id": approval.interruption_id,
                }
            )
            if resume_budget_events:
                continued.events = [*source.events, *resume_budget_events, *continued.events]
            return continued

        if raw_result.completion_reason == CompletionReason.BUDGET_EXHAUSTED:
            pass
        elif tool_result.directive == ToolDirective.WAIT_USER:
            raw_result.completion_tool_name = call.name
            raw_result.error = None
            wait_reason = tool_result.metadata.get("question") if isinstance(tool_result.metadata, dict) else None
            if not wait_reason:
                wait_reason = tool_result.content
            raw_result.status = AgentStatus.WAIT_USER
            raw_result.completion_reason = CompletionReason.WAIT_USER
            raw_result.partial_output = _last_assistant_output(raw_result.cycles)
            raw_result.final_answer = None
            raw_result.wait_reason = str(wait_reason)
        else:
            raw_result.completion_tool_name = call.name
            raw_result.error = None
            raw_result.status = AgentStatus.COMPLETED
            raw_result.completion_reason = behavior_reason or CompletionReason.TOOL_FINISH
            raw_result.partial_output = None
            raw_result.final_answer = AgentRuntime._extract_final_message(tool_result)
            raw_result.wait_reason = None

        guardrail_context = context.run_context
        if not isinstance(guardrail_context, RunContext):
            guardrail_context = RunContext(
                context=effective_config.context,
                run_id=source.run_id,
                agent_name=source.agent_name,
                model=source.resolved_model.model_id if source.resolved_model is not None else None,
                workspace=effective_config.workspace,
                metadata={**resume_context.agent.metadata, **effective_config.metadata},
            )
        final_output, output_coercion_error = cls._postprocess_output(
            agent=resume_context.agent,
            run_context=guardrail_context,
            raw_result=raw_result,
            final_output=raw_result.final_answer or raw_result.wait_reason,
            cancellation_token=effective_config.cancellation_token,
        )
        terminal_event = cls._terminal_event(
            result=raw_result,
            final_output=final_output,
            run_id=resumed_run_id,
            trace_id=source.trace_id,
            agent_name=source.agent_name,
            session_id=cls._resolve_event_session_id(effective_config),
            cancellation_token=effective_config.cancellation_token,
        )
        cls._emit_chain_event(terminal_event, run_config=effective_config, event_sink=None)

        resumed = RunResult(
            input=source.input,
            new_items=[
                *[
                    message
                    for message in source.new_items
                    if not (message.role == "tool" and message.tool_call_id == pending.call.id)
                ],
                tool_message,
            ],
            final_output=final_output,
            status=raw_result.status,
            raw_result=raw_result,
            events=[*source.events, *resume_budget_events, terminal_event],
            token_usage=raw_result.token_usage,
            trace_id=source.trace_id,
            run_id=resumed_run_id,
            metadata={**source.metadata, "resumed": True, "approved_interruption_id": approval.interruption_id},
            agent_name=source.agent_name,
            resolved_model=source.resolved_model,
            _resume_context=resume_context,
        )
        if output_coercion_error is not None:
            raise output_coercion_error
        handoff_request = cls._extract_handoff(resumed)
        if handoff_request is None:
            return resumed

        legacy_event = HandoffEvent(
            run_id=resumed.run_id,
            trace_id=resumed.trace_id,
            source_agent=handoff_request.source_agent,
            target_agent=handoff_request.target_agent,
            tool_call_id=handoff_request.tool_call_id,
            cycle_index=handoff_request.cycle_index,
            session_id=cls._resolve_event_session_id(effective_config),
            metadata=dict(handoff_request.metadata),
        )
        resumed.events.append(legacy_event)
        cls._emit_chain_event(legacy_event, run_config=effective_config, event_sink=None)
        return resume_context.runner._run(
            resume_context.agent,
            resume_context.input,
            run_config=resume_context.run_config,
            initial_result=resumed,
        )

    @classmethod
    def _cancelled_approval_resume_result(
        cls,
        source: RunResult,
        resume_context: _RunResumeContext,
        approval: ApprovalSnapshot,
        *,
        effective_config: RunConfig,
    ) -> RunResult:
        cancellation_token = effective_config.cancellation_token
        assert cancellation_token is not None and cancellation_token.cancelled

        raw_result = deepcopy(source.raw_result)
        raw_result.status = AgentStatus.FAILED
        raw_result.final_answer = None
        raw_result.wait_reason = None
        raw_result.error = cancellation_token.reason or "Operation was cancelled"
        raw_result.completion_reason = CompletionReason.CANCELLED
        raw_result.completion_tool_name = None
        cls._normalize_completion_observation(raw_result, cancellation_token=cancellation_token)

        resumed_run_id = f"run_{uuid.uuid4().hex}"
        terminal_event = cls._terminal_event(
            result=raw_result,
            final_output=raw_result.error,
            run_id=resumed_run_id,
            trace_id=source.trace_id,
            agent_name=source.agent_name,
            session_id=cls._resolve_event_session_id(effective_config),
            cancellation_token=cancellation_token,
        )
        cls._emit_chain_event(terminal_event, run_config=effective_config, event_sink=None)
        return RunResult(
            input=source.input,
            new_items=deepcopy(source.new_items),
            final_output=raw_result.error,
            status=raw_result.status,
            raw_result=raw_result,
            events=[*source.events, terminal_event],
            token_usage=raw_result.token_usage,
            trace_id=source.trace_id,
            run_id=resumed_run_id,
            metadata={**source.metadata, "resumed": True, "approved_interruption_id": approval.interruption_id},
            agent_name=source.agent_name,
            resolved_model=source.resolved_model,
            _resume_context=resume_context,
        )

    @staticmethod
    def _approval_snapshot_matches(
        source: RunResult,
        approval: ApprovalSnapshot,
        pending: _PendingToolApproval,
    ) -> bool:
        if (
            approval.cycle_index != pending.cycle_index
            or approval.tool_call_id != pending.call.id
            or approval.tool_name != pending.call.name
            or approval.arguments != pending.call.arguments
        ):
            return False
        return any(
            cycle.index == pending.cycle_index
            and any(call == pending.call for call in cycle.tool_calls)
            and any(
                result.tool_call_id == pending.call.id
                and result.metadata.get("approval_interruption_id") == pending.interruption_id
                and result.metadata.get("tool_name") == pending.call.name
                and result.metadata.get("arguments") == pending.call.arguments
                for result in cycle.tool_results
            )
            for cycle in source.raw_result.cycles
        )

    @staticmethod
    def _effective_run_config(
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
            "continue",
        )
        effective_max_cycles = _validate_bounded_int(configured_max_cycles, "max_cycles", minimum=1)
        effective_max_handoffs = _validate_bounded_int(configured_max_handoffs, "max_handoffs", minimum=0)
        assert effective_max_cycles is not None
        assert effective_max_handoffs is not None
        model_settings = (
            ModelSettings().resolve(defaults.model_settings).resolve(agent.model_settings).resolve(config.model_settings)
        )
        shared_state = None
        if defaults.shared_state is not None or config.shared_state is not None:
            shared_state = {**(defaults.shared_state or {}), **(config.shared_state or {})}

        def prefer_run(name: str) -> Any:
            value = getattr(config, name)
            return value if value is not None else getattr(defaults, name)

        settings_file = config.settings_file or defaults.settings_file or DEFAULT_SETTINGS_FILE
        timeout_seconds = config.timeout_seconds
        if timeout_seconds is None:
            timeout_seconds = defaults.timeout_seconds
        if timeout_seconds is None:
            timeout_seconds = DEFAULT_TIMEOUT_SECONDS

        return replace(
            config,
            model=model,
            model_provider=config.model_provider or defaults.model_provider,
            model_settings=model_settings,
            workspace=prefer_run("workspace"),
            workspace_backend=prefer_run("workspace_backend"),
            session=prefer_run("session"),
            max_cycles=effective_max_cycles,
            max_handoffs=effective_max_handoffs,
            no_tool_policy=configured_no_tool_policy,
            tool_policy=merge_tool_policy_layers(agent.tool_policy, defaults.tool_policy, config.tool_policy),
            execution_backend=prefer_run("execution_backend"),
            cancellation_token=prefer_run("cancellation_token"),
            approval_provider=prefer_run("approval_provider"),
            approval_timeout_seconds=prefer_run("approval_timeout_seconds"),
            approval_broker=prefer_run("approval_broker"),
            event_store=prefer_run("event_store"),
            event_store_fail_closed=defaults.event_store_fail_closed or config.event_store_fail_closed,
            stream=prefer_run("stream"),
            hooks=[*defaults.hooks, *config.hooks],
            tracing=prefer_run("tracing"),
            context=prefer_run("context"),
            context_providers=[*defaults.context_providers, *config.context_providers],
            max_context_chars=prefer_run("max_context_chars"),
            memory_providers=[*defaults.memory_providers, *config.memory_providers],
            metadata={**defaults.metadata, **config.metadata},
            settings_file=settings_file,
            default_backend=prefer_run("default_backend"),
            llm_builder=prefer_run("llm_builder"),
            timeout_seconds=timeout_seconds,
            tool_registry_factory=prefer_run("tool_registry_factory"),
            log_preview_chars=prefer_run("log_preview_chars"),
            debug_dump_dir=prefer_run("debug_dump_dir"),
            shared_state=shared_state,
            initial_messages=prefer_run("initial_messages"),
            before_cycle_messages=prefer_run("before_cycle_messages"),
            interruption_messages=prefer_run("interruption_messages"),
            sub_task_manager=prefer_run("sub_task_manager"),
            runtime_log_handler=prefer_run("runtime_log_handler"),
            runtime_stream_callback=prefer_run("runtime_stream_callback"),
            budget_limits=prefer_run("budget_limits"),
            host_cost_meter=prefer_run("host_cost_meter"),
        )

    @classmethod
    def _run(
        cls,
        agent: Agent,
        input: str,
        *,
        run_config: RunConfig,
        event_sink: Callable[[RunEvent], None] | None = None,
        resume_runner: RunHandleRunner | None = None,
        runner_defaults: RunConfig | None = None,
        initial_result: RunResult | None = None,
        _compiled_invocation: _CompiledTaskInvocation | None = None,
    ) -> RunResult:
        base_config = run_config
        defaults = runner_defaults or RunConfig()
        current_agent = agent
        current_input = input
        chain_events: list[RunEvent] = []
        pending: _PendingHandoff | None = None
        next_result = initial_result
        next_compiled_invocation = _compiled_invocation
        handoff_count = 0
        max_handoffs = cls._effective_run_config(
            agent,
            base_config,
            runner_defaults=defaults,
        ).max_handoffs
        assert max_handoffs is not None

        while True:
            effective_config = cls._effective_run_config(
                current_agent,
                base_config,
                runner_defaults=defaults,
            )
            if next_result is not None:
                result = next_result
                next_result = None
            else:
                try:
                    result = cls._run_single_agent(
                        current_agent,
                        current_input,
                        run_config=effective_config,
                        event_sink=event_sink,
                        resume_runner=resume_runner,
                        resume_config=base_config,
                        _compiled_invocation=next_compiled_invocation,
                    )
                    next_compiled_invocation = None
                except Exception as exc:
                    if pending is not None:
                        completed = HandoffCompletedEvent(
                            run_id=pending.source_run_id,
                            trace_id=pending.source_trace_id,
                            source_agent=pending.request.source_agent,
                            target_agent=pending.request.target_agent,
                            tool_call_id=pending.request.tool_call_id,
                            status=AgentStatus.FAILED.value,
                            child_session_id=pending.session_id,
                            cycle_index=pending.request.cycle_index,
                            session_id=pending.session_id,
                            metadata={
                                **pending.request.metadata,
                                "chain_continues": False,
                                "error": str(exc),
                            },
                        )
                        chain_events.append(completed)
                        cls._emit_chain_event(completed, run_config=pending.event_config, event_sink=event_sink)
                    raise

            chain_events.extend(result.events)
            if _INITIAL_BUDGET_USAGE_METADATA_KEY in base_config.metadata:
                base_config = replace(
                    base_config,
                    metadata={
                        key: value
                        for key, value in base_config.metadata.items()
                        if key != _INITIAL_BUDGET_USAGE_METADATA_KEY
                    },
                )
            request = cls._extract_handoff(result)
            if pending is not None:
                completed = HandoffCompletedEvent(
                    run_id=pending.source_run_id,
                    trace_id=pending.source_trace_id,
                    source_agent=pending.request.source_agent,
                    target_agent=pending.request.target_agent,
                    tool_call_id=pending.request.tool_call_id,
                    status=result.status.value,
                    child_session_id=cls._resolve_event_session_id(effective_config),
                    child_run_id=result.run_id,
                    cycle_index=pending.request.cycle_index,
                    session_id=pending.session_id,
                    metadata={
                        **pending.request.metadata,
                        "chain_continues": request is not None,
                    },
                )
                chain_events.append(completed)
                cls._emit_chain_event(completed, run_config=pending.event_config, event_sink=event_sink)
                pending = None

            if request is None:
                result.events = chain_events
                return result
            if handoff_count >= max_handoffs:
                raise RuntimeError("maximum handoff depth exceeded")

            transfer = next(
                (
                    candidate
                    for candidate in current_agent.handoffs
                    if candidate.tool_name == request.tool_name and candidate.agent.name == request.target_agent
                ),
                None,
            )
            if transfer is None:
                raise RuntimeError(f"handoff target {request.target_agent!r} is not registered on agent {current_agent.name!r}")

            session_id = cls._resolve_event_session_id(effective_config)
            started = HandoffStartedEvent(
                run_id=result.run_id,
                trace_id=result.trace_id,
                source_agent=request.source_agent,
                target_agent=request.target_agent,
                tool_call_id=request.tool_call_id,
                child_session_id=session_id,
                cycle_index=request.cycle_index,
                session_id=session_id,
                metadata=dict(request.metadata),
            )
            chain_events.append(started)
            cls._emit_chain_event(started, run_config=effective_config, event_sink=event_sink)
            pending = _PendingHandoff(
                request=request,
                source_run_id=result.run_id,
                source_trace_id=result.trace_id,
                session_id=session_id,
                event_config=effective_config,
            )
            handoff_count += 1
            base_config = replace(
                base_config,
                shared_state=deepcopy(result.raw_result.shared_state),
            )
            current_agent = transfer.agent
            current_input = request.input

    @staticmethod
    def _emit_chain_event(
        event: RunEvent,
        *,
        run_config: RunConfig,
        event_sink: Callable[[RunEvent], None] | None,
    ) -> None:
        if run_config.event_store is not None:
            try:
                run_config.event_store.append(event)
            except Exception as exc:
                if run_config.event_store_fail_closed:
                    raise
                warnings.warn(f"Run event store append failed: {exc}", RuntimeWarning, stacklevel=2)
        if event_sink is not None:
            event_sink(event)
        Runner._notify_observer(run_config.stream, event, label="Run event stream observer")

    @staticmethod
    def _notify_observer(callback: Callable[[Any], None] | None, payload: Any, *, label: str) -> None:
        if callback is None:
            return
        try:
            callback(payload)
        except BaseException as exc:
            warnings.warn(f"{label} failed: {exc}", RuntimeWarning, stacklevel=2)

    @classmethod
    def _run_single_agent(
        cls,
        agent: Agent,
        input: str,
        *,
        run_config: RunConfig,
        event_sink: Callable[[RunEvent], None] | None = None,
        resume_runner: RunHandleRunner | None = None,
        resume_config: RunConfig | None = None,
        _compiled_invocation: _CompiledTaskInvocation | None = None,
    ) -> RunResult:
        if run_config.approval_provider is not None and run_config.approval_broker is None:
            run_config = replace(run_config, approval_broker=ApprovalBroker())
        run_id = f"run_{uuid.uuid4().hex}"
        trace_id = cls._resolve_trace_id(run_config)
        trace = _RunTrace(
            processors=cls._trace_processors(run_config),
            trace_id=trace_id,
            agent_name=agent.name,
            workflow_name=cls._workflow_name(run_config),
        )
        trace_metadata: dict[str, Any] = {"status": "failed", "error": "run aborted"}
        try:
            result, trace_metadata = cls._run_single_agent_inner(
                agent,
                input,
                run_config=run_config,
                run_id=run_id,
                trace_id=trace_id,
                trace=trace,
                event_sink=event_sink,
                resume_runner=resume_runner,
                resume_config=resume_config,
                _compiled_invocation=_compiled_invocation,
            )
            return result
        except BaseException as exc:
            trace_metadata = {"status": "failed", "error": str(exc)}
            raise
        finally:
            ended_run_span = trace.finish(trace_metadata)
            if "result" in locals():
                result.metadata["run_span"] = ended_run_span.to_dict()

    @classmethod
    def _run_single_agent_inner(
        cls,
        agent: Agent,
        input: str,
        *,
        run_config: RunConfig,
        run_id: str,
        trace_id: str,
        trace: _RunTrace,
        event_sink: Callable[[RunEvent], None] | None = None,
        resume_runner: RunHandleRunner | None = None,
        resume_config: RunConfig | None = None,
        _compiled_invocation: _CompiledTaskInvocation | None = None,
    ) -> tuple[RunResult, dict[str, Any]]:
        event_session_id = cls._resolve_event_session_id(run_config)
        collected_events: list[RunEvent] = []
        user_input = cls._normalize_input(input)
        initial_budget_usage = cls._initial_budget_usage(run_config)

        def capture_event(event: RunEvent | None) -> None:
            if event is None:
                return
            collected_events.append(event)
            trace.on_event(event)
            if run_config.event_store is not None:
                try:
                    run_config.event_store.append(event)
                except Exception as exc:
                    if run_config.event_store_fail_closed:
                        raise RuntimeError(f"run event store append failed: {exc}") from exc
                    warnings.warn(f"Run event store append failed: {exc}", RuntimeWarning, stacklevel=2)
            if event_sink is not None:
                event_sink(event)
            cls._notify_observer(run_config.stream, event, label="Run event stream observer")

        def log_handler(event: str, payload: dict[str, Any]) -> None:
            if event not in _RUNTIME_TERMINAL_LOG_EVENTS:
                capture_event(
                    event_from_runtime_log(
                        event,
                        payload,
                        run_id=run_id,
                        trace_id=trace_id,
                        agent_name=agent.name,
                        user_input=user_input,
                        session_id=event_session_id,
                    )
                )
            if run_config.runtime_log_handler is not None:
                run_config.runtime_log_handler(event, payload)

        def stream_callback(payload: dict[str, Any]) -> None:
            canonical_child = isinstance(payload, _CanonicalSubAgentStreamPayload)
            capture_event(
                event_from_stream_payload(
                    payload,
                    run_id=(str(payload["run_id"]) if canonical_child else run_id),
                    trace_id=(str(payload["trace_id"]) if canonical_child else trace_id),
                    agent_name=(str(payload["agent_name"]) if canonical_child else agent.name),
                    session_id=(str(payload["session_id"]) if canonical_child else event_session_id),
                    parent_run_id=(str(payload.get("parent_run_id") or "") or None if canonical_child else None),
                    preserve_metadata=canonical_child,
                )
            )
            cls._notify_observer(
                run_config.runtime_stream_callback,
                payload,
                label="Runtime stream observer",
            )

        context_model = run_config.model
        if isinstance(context_model, ModelRef):
            context_model = context_model.model()
        elif context_model is not None:
            context_model = getattr(context_model, "model_id", context_model)
        guardrail_context = RunContext(
            context=run_config.context,
            run_id=run_id,
            agent_name=agent.name,
            model=context_model,
            workspace=run_config.workspace,
            metadata={**agent.metadata, **run_config.metadata},
        )
        input_result = cls._apply_input_guardrails(agent=agent, run_context=guardrail_context, user_input=user_input)
        if input_result.outcome in {"block", "require_approval"}:
            message = input_result.message or "Input blocked by guardrail."
            failed_event = RunFailedEvent(
                run_id=run_id,
                trace_id=trace_id,
                error=message,
                completion_reason=CompletionReason.FAILED,
                agent_name=agent.name,
                session_id=event_session_id,
            )
            capture_event(failed_event)
            raw_result = AgentResult(
                status=AgentStatus.FAILED,
                completion_reason=CompletionReason.FAILED,
                messages=[],
                cycles=[],
                error=message,
            )
            trace_metadata = {"status": "failed", "error": message}
            result = RunResult(
                input=user_input,
                new_items=[],
                final_output=message,
                status=AgentStatus.FAILED,
                raw_result=raw_result,
                events=collected_events,
                token_usage=raw_result.token_usage,
                trace_id=trace_id,
                run_id=run_id,
                metadata={},
                agent_name=agent.name,
                resolved_model=None,
            )
            return result, trace_metadata
        if input_result.outcome == "rewrite":
            user_input = str(input_result.value)

        llm_client, resolved = cls._resolve_model(agent=agent, run_config=run_config)
        guardrail_context.model = resolved.model_id
        provider_settings = cls._provider_default_settings(run_config=run_config, resolved=resolved)
        resolved_model_settings = provider_settings.resolve(run_config.model_settings)
        if run_config.debug_dump_dir:
            cast(Any, llm_client).debug_dump_dir = run_config.debug_dump_dir

        registry = cls._build_tool_registry(agent=agent, run_config=run_config)
        runtime = AgentRuntime(
            llm_client=llm_client,
            tool_registry=registry,
            default_workspace=cls._resolve_workspace(run_config.workspace),
            log_handler=log_handler,
            log_preview_chars=run_config.log_preview_chars,
            settings_file=run_config.settings_file or DEFAULT_SETTINGS_FILE,
            default_backend=run_config.default_backend,
            llm_builder=run_config.llm_builder,
            tool_registry_factory=run_config.tool_registry_factory,
            execution_backend=run_config.execution_backend,
            workspace_backend=run_config.workspace_backend,
            hooks=[*agent.hooks, *run_config.hooks],
        )
        task = (
            deepcopy(_compiled_invocation.task)
            if _compiled_invocation is not None
            else AgentCompiler().compile(
                agent=agent,
                input=user_input,
                run_config=run_config,
                resolved=resolved,
                trace_id=trace_id,
                run_id=run_id,
            )
        )
        task.metadata.pop(_INITIAL_BUDGET_USAGE_METADATA_KEY, None)
        policy = run_config.tool_policy
        policy_metadata = _tool_policy_metadata(policy)
        for key in _TOOL_POLICY_METADATA_KEYS:
            task.metadata.pop(key, None)
        for key in ("_vv_agent_allowed_tools", "_vv_agent_disallowed_tools"):
            if key in policy_metadata:
                task.metadata[key] = policy_metadata[key]
        task.extra_tool_names = [
            *task.extra_tool_names,
            *[name for name in registry.list_planner_extra_tool_names() if name not in task.extra_tool_names],
        ]
        initial_messages = (
            list(run_config.initial_messages)
            if run_config.initial_messages is not None
            else cls._session_initial_messages(run_config)
        )
        runtime_metadata = dict(run_config.metadata)
        runtime_metadata.pop(_INITIAL_BUDGET_USAGE_METADATA_KEY, None)
        for key in _TOOL_POLICY_METADATA_KEYS:
            runtime_metadata.pop(key, None)
        runtime_metadata.update(policy_metadata)
        ctx = ExecutionContext(
            cancellation_token=run_config.cancellation_token,
            stream_callback=stream_callback,
            metadata={
                **runtime_metadata,
                "_vv_agent_agent_name": agent.name,
                "_vv_agent_emit_event": capture_event,
                "trace_id": trace_id,
                "_vv_agent_run_id": run_id,
                "_vv_agent_trace_id": trace_id,
                "_vv_agent_input": user_input,
                "_vv_agent_model_settings": resolved_model_settings,
                "_vv_agent_model_provider": run_config.model_provider,
                "_vv_agent_run_context": guardrail_context,
                "_vv_agent_session": run_config.session,
                "_vv_agent_session_id": event_session_id,
                "_vv_agent_memory_providers": list(run_config.memory_providers),
                "_vv_agent_approval_provider": run_config.approval_provider,
                "_vv_agent_approval_broker": run_config.approval_broker,
                "_vv_agent_approval_timeout_seconds": run_config.approval_timeout_seconds,
            },
        )

        raw_result = runtime.run(
            task,
            workspace=cls._resolve_workspace(run_config.workspace),
            shared_state=run_config.shared_state,
            initial_messages=initial_messages,
            user_message=user_input,
            ctx=ctx,
            before_cycle_messages=run_config.before_cycle_messages,
            interruption_messages=run_config.interruption_messages,
            sub_task_manager=run_config.sub_task_manager,
            budget_limits=run_config.budget_limits,
            host_cost_meter=run_config.host_cost_meter,
            initial_budget_usage=initial_budget_usage,
        )
        final_output = raw_result.final_answer or raw_result.wait_reason or raw_result.error
        final_output, output_coercion_error = cls._postprocess_output(
            agent=agent,
            run_context=guardrail_context,
            raw_result=raw_result,
            final_output=final_output,
            cancellation_token=run_config.cancellation_token,
        )

        new_items = cls._new_session_items(initial_messages=initial_messages, result=raw_result)
        if run_config.session is not None:
            run_config.session.add_items(cls._session_items_for_persistence(new_items, raw_result))
            capture_event(
                SessionPersistedEvent(
                    run_id=run_id,
                    trace_id=trace_id,
                    agent_name=agent.name,
                    session_id=event_session_id or "",
                )
            )
        capture_event(
            cls._terminal_event(
                result=raw_result,
                final_output=final_output,
                run_id=run_id,
                trace_id=trace_id,
                agent_name=agent.name,
                session_id=event_session_id,
                cancellation_token=run_config.cancellation_token,
            )
        )
        trace_metadata = {
            "status": raw_result.status.value,
            "final_output": final_output,
            "completion_reason": raw_result.completion_reason.value if raw_result.completion_reason is not None else None,
            "completion_tool_name": raw_result.completion_tool_name,
            "partial_output": raw_result.partial_output,
        }
        if output_coercion_error is not None:
            raise output_coercion_error
        result = RunResult(
            input=user_input,
            new_items=new_items,
            final_output=final_output,
            status=raw_result.status,
            raw_result=raw_result,
            events=collected_events,
            token_usage=raw_result.token_usage,
            trace_id=trace_id,
            run_id=run_id,
            metadata={"resolved_model": resolved.model_id, "backend": resolved.backend},
            agent_name=agent.name,
            resolved_model=resolved,
            _resume_context=_RunResumeContext(
                agent=agent,
                input=user_input,
                run_config=resume_config or run_config,
                runner=resume_runner or cast(RunHandleRunner, cls),
                effective_run_config=run_config,
                pending_tool_approval=ctx._pending_tool_approval,
            ),
        )
        return result, trace_metadata

    @staticmethod
    def _apply_input_guardrails(*, agent: Agent, run_context: RunContext[Any], user_input: str) -> GuardrailResult:
        current_input = user_input
        for guardrail in agent.input_guardrails:
            result = guardrail(run_context, current_input)
            if result.outcome == "rewrite":
                current_input = str(result.value)
                continue
            if result.outcome != "allow":
                return result
        if current_input != user_input:
            return GuardrailResult.rewrite(current_input)
        return GuardrailResult.allow()

    @staticmethod
    def _apply_output_guardrails(*, agent: Agent, run_context: RunContext[Any], final_output: Any) -> GuardrailResult:
        current_output = final_output
        for guardrail in agent.output_guardrails:
            result = guardrail(run_context, current_output)
            if result.outcome == "rewrite":
                current_output = result.value
                continue
            if result.outcome != "allow":
                return result
        if current_output != final_output:
            return GuardrailResult.rewrite(current_output)
        return GuardrailResult.allow()

    @classmethod
    def _postprocess_output(
        cls,
        *,
        agent: Agent,
        run_context: RunContext[Any],
        raw_result: AgentResult,
        final_output: Any,
        cancellation_token: Any | None,
    ) -> tuple[Any, Exception | None]:
        cls._normalize_completion_observation(raw_result, cancellation_token=cancellation_token)
        if raw_result.completion_reason in {CompletionReason.CANCELLED, CompletionReason.BUDGET_EXHAUSTED}:
            return final_output, None
        output_result = cls._apply_output_guardrails(
            agent=agent,
            run_context=run_context,
            final_output=final_output,
        )
        if output_result.outcome == "rewrite":
            final_output = output_result.value
            cls._replace_raw_result_output(raw_result, final_output)
        elif output_result.outcome in {"block", "require_approval"}:
            final_output = output_result.message or "Output blocked by guardrail."
            raw_result.status = AgentStatus.FAILED
            raw_result.completion_reason = CompletionReason.FAILED
            raw_result.completion_tool_name = None
            raw_result.partial_output = raw_result.partial_output or _last_assistant_output(raw_result.cycles)
            raw_result.final_answer = None
            raw_result.wait_reason = None
            raw_result.error = final_output

        cls._normalize_completion_observation(raw_result, cancellation_token=cancellation_token)
        output_coercion_error: Exception | None = None
        if raw_result.status == AgentStatus.COMPLETED:
            try:
                final_output = cls._coerce_output_type(agent=agent, final_output=final_output)
            except Exception as exc:
                output_coercion_error = ValueError(f"failed to validate final output: {exc}")
        return final_output, output_coercion_error

    @staticmethod
    def _replace_raw_result_output(result: AgentResult, output: Any) -> None:
        value = output if isinstance(output, str) or output is None else json.dumps(output, ensure_ascii=False)
        if result.status == AgentStatus.COMPLETED:
            result.final_answer = value
        elif result.status == AgentStatus.WAIT_USER:
            result.wait_reason = value
        else:
            result.error = value

    @staticmethod
    def _terminal_event(
        *,
        result: AgentResult,
        final_output: Any,
        run_id: str,
        trace_id: str,
        agent_name: str,
        session_id: str | None,
        cancellation_token: Any | None,
    ) -> RunEvent:
        cycle_index = len(result.cycles) or None
        cancelled = bool(result.status == AgentStatus.FAILED and cancellation_token is not None and cancellation_token.cancelled)
        if cancelled:
            return RunCancelledEvent(
                run_id=run_id,
                trace_id=trace_id,
                agent_name=agent_name,
                session_id=session_id,
                cycle_index=cycle_index,
                reason=result.error or cancellation_token.reason or "run cancelled",
                completion_reason=CompletionReason.CANCELLED,
                partial_output=result.partial_output,
                budget_usage=result.budget_usage,
                budget_exhaustion=result.budget_exhaustion,
            )
        if result.status in {AgentStatus.FAILED, AgentStatus.MAX_CYCLES}:
            return RunFailedEvent(
                run_id=run_id,
                trace_id=trace_id,
                agent_name=agent_name,
                session_id=session_id,
                cycle_index=cycle_index,
                error=result.error or result.status.value,
                status=(result.status.value if result.budget_usage is not None else None),
                completion_reason=result.completion_reason,
                completion_tool_name=result.completion_tool_name,
                partial_output=result.partial_output,
                budget_usage=result.budget_usage,
                budget_exhaustion=result.budget_exhaustion,
            )
        event_output = final_output
        if event_output is not None and not isinstance(event_output, str):
            event_output = json.dumps(
                RunResult._serializable_output(event_output),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        return RunCompletedEvent(
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            cycle_index=cycle_index,
            final_output=event_output,
            status=result.status.value,
            completion_reason=result.completion_reason,
            completion_tool_name=result.completion_tool_name,
            partial_output=result.partial_output,
            budget_usage=result.budget_usage,
            budget_exhaustion=result.budget_exhaustion,
        )

    @staticmethod
    def _normalize_completion_observation(
        result: AgentResult,
        *,
        cancellation_token: Any | None,
    ) -> None:
        cancelled = bool(result.status == AgentStatus.FAILED and cancellation_token is not None and cancellation_token.cancelled)
        if cancelled:
            result.completion_reason = CompletionReason.CANCELLED
            result.completion_tool_name = None
        elif result.status == AgentStatus.WAIT_USER:
            result.completion_reason = result.completion_reason or CompletionReason.WAIT_USER
        elif result.status == AgentStatus.MAX_CYCLES:
            result.completion_reason = CompletionReason.MAX_CYCLES
            result.completion_tool_name = None
        elif result.status == AgentStatus.FAILED:
            if result.completion_reason != CompletionReason.BUDGET_EXHAUSTED:
                result.completion_reason = CompletionReason.FAILED
            result.completion_tool_name = None

        if result.status == AgentStatus.COMPLETED:
            result.partial_output = None
        else:
            result.partial_output = result.partial_output or _last_assistant_output(result.cycles)

    @staticmethod
    def _coerce_output_type(*, agent: Agent, final_output: Any) -> Any:
        output_type = agent.output_type
        if output_type is None or final_output is None:
            return final_output
        if output_type is str:
            return str(final_output)

        payload = final_output
        if isinstance(final_output, str):
            payload = json.loads(final_output)

        if output_type is dict:
            if not isinstance(payload, dict):
                raise ValueError("Expected final output JSON object for output_type=dict.")
            return payload
        if output_type is list:
            if not isinstance(payload, list):
                raise ValueError("Expected final output JSON array for output_type=list.")
            return payload
        if isinstance(output_type, type) and is_dataclass(output_type):
            if not isinstance(payload, dict):
                raise ValueError("Expected final output JSON object for dataclass output_type.")
            field_names = {item.name for item in fields(output_type)}
            return output_type(**{key: value for key, value in payload.items() if key in field_names})

        model_validate = getattr(output_type, "model_validate", None)
        if callable(model_validate):
            return model_validate(payload)
        return final_output

    @classmethod
    def _resolve_model(cls, *, agent: Agent, run_config: RunConfig) -> tuple[LLMClient, ResolvedModelConfig]:
        if run_config.model_provider is not None:
            provider = cast(Any, run_config.model_provider)
            if hasattr(provider, "resolve") and hasattr(provider, "client"):
                model = run_config.model or agent.model
                if model is None:
                    default_model_ref = getattr(provider, "default_model_ref", None)
                    model = default_model_ref() if callable(default_model_ref) else None
                if model is None:
                    raise ValueError("Agent.model or RunConfig.model is required when a model provider is configured.")
                resolved = provider.resolve(ModelRef.coerce(model))
                return provider.client(resolved), resolved
            return provider(agent, run_config)
        model = run_config.model or agent.model
        if hasattr(model, "complete"):
            model_id = getattr(model, "model_id", "direct")
            return cast(LLMClient, model), cls._direct_resolved(str(model_id))
        if model is None:
            raise ValueError("Agent.model or RunConfig.model is required when no model_provider is configured.")
        settings_file = run_config.settings_file or DEFAULT_SETTINGS_FILE
        timeout_seconds = run_config.timeout_seconds if run_config.timeout_seconds is not None else DEFAULT_TIMEOUT_SECONDS
        backend = run_config.default_backend or cls._infer_backend(settings_file, str(model))
        llm_builder = run_config.llm_builder or build_openai_llm_from_local_settings
        return llm_builder(
            settings_file,
            backend=backend,
            model=str(model),
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _provider_default_settings(*, run_config: RunConfig, resolved: ResolvedModelConfig) -> ModelSettings:
        default_settings = getattr(run_config.model_provider, "default_settings", None)
        if callable(default_settings):
            return default_settings(resolved)
        return ModelSettings()

    @classmethod
    def _build_tool_registry(cls, *, agent: Agent, run_config: RunConfig) -> ToolRegistry:
        registry = (
            run_config.tool_registry_factory() if run_config.tool_registry_factory is not None else build_default_registry()
        )
        for candidate in agent.tools:
            if is_tool_executor(candidate):
                executor = cast(ToolExecutor, candidate)
                is_model_visible = executor.exposure != ToolExposure.HIDDEN
                registry.register_executor(
                    executor,
                    expose_to_model=is_model_visible,
                    planner_extra=is_model_visible,
                )
                continue
            tool = candidate if isinstance(candidate, FunctionTool) else adapt_tool(cast(Tool, candidate))
            if not cls._tool_is_enabled(tool=tool, agent=agent, run_config=run_config):
                continue
            if not isinstance(candidate, FunctionTool):
                is_model_visible = tool.exposure != ToolExposure.HIDDEN
                registry.register_executor(
                    tool.to_executor(),
                    expose_to_model=is_model_visible,
                    planner_extra=is_model_visible,
                )
                continue
            if tool.exposure != ToolExposure.HIDDEN:
                registry.register_schema(tool.name, tool.to_openai_schema())

            def handler(context: ToolContext, arguments: dict[str, Any], *, _tool: FunctionTool = tool) -> ToolExecutionResult:
                return cls._execute_function_tool(_tool, context=context, arguments=arguments, run_config=run_config)

            registry.register(ToolSpec(name=tool.name, handler=handler))
            registry.mark_policy_managed_by_handler(tool.name)
        for transfer in agent.handoffs:
            if not transfer.tool_name:
                continue
            registry.register_schema(
                transfer.tool_name,
                {
                    "type": "function",
                    "function": {
                        "name": transfer.tool_name,
                        "description": transfer.description or f"Transfer control to {transfer.agent.name}.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "Input or handoff summary for the target agent.",
                                    "minLength": 1,
                                }
                            },
                            "required": ["input"],
                            "additionalProperties": False,
                        },
                    },
                },
            )

            def handoff_handler(
                context: ToolContext,
                arguments: dict[str, Any],
                *,
                _target: Agent = transfer.agent,
                _tool_name: str = transfer.tool_name,
                _metadata: tuple[tuple[str, Any], ...] = tuple(transfer.metadata.items()),
            ) -> ToolExecutionResult:
                input_value = arguments.get("input")
                if set(arguments) != {"input"} or not isinstance(input_value, str) or not input_value.strip():
                    return ToolExecutionResult(
                        tool_call_id=context.tool_call_id,
                        content=json.dumps(
                            {
                                "ok": False,
                                "error": "handoff requires a non-empty input string and no additional arguments",
                                "error_code": "invalid_handoff_arguments",
                            },
                            ensure_ascii=False,
                        ),
                        status="error",
                        status_code=ToolResultStatus.ERROR,
                        error_code="invalid_handoff_arguments",
                    )
                handoff_input = input_value.strip()
                metadata = dict(_metadata)
                metadata.update(
                    {
                        "mode": "handoff",
                        "handoff_from": agent.name,
                        "handoff_to": _target.name,
                        "handoff_input": handoff_input,
                        "handoff_tool_name": _tool_name,
                    }
                )
                return ToolExecutionResult(
                    tool_call_id=context.tool_call_id,
                    content=json.dumps(
                        {
                            "ok": True,
                            "handoff": True,
                            "from_agent": agent.name,
                            "to_agent": _target.name,
                            "input": handoff_input,
                        },
                        ensure_ascii=False,
                    ),
                    directive=ToolDirective.FINISH,
                    metadata=metadata,
                )

            registry.register(ToolSpec(name=transfer.tool_name, handler=handoff_handler))
        return registry

    @classmethod
    def _execute_function_tool(
        cls,
        tool: FunctionTool,
        *,
        context: ToolContext,
        arguments: dict[str, Any],
        run_config: RunConfig,
    ) -> ToolExecutionResult:
        run_config = cls._tool_run_config_from_context(context=context, fallback=run_config)
        denied_result = cls._policy_denial_result(
            tool,
            context=context,
            arguments=arguments,
            run_config=run_config,
        )
        if denied_result is not None:
            return denied_result
        approval_result = cls._approval_result(tool, context=context, arguments=arguments, run_config=run_config)
        if approval_result is not None:
            return approval_result
        return cls._invoke_function_tool(tool, context=context, arguments=arguments, run_config=run_config)

    @classmethod
    def _invoke_function_tool(
        cls,
        tool: FunctionTool,
        *,
        context: ToolContext,
        arguments: dict[str, Any],
        run_config: RunConfig,
    ) -> ToolExecutionResult:
        if isinstance(tool, BackgroundAgentTask):
            handle = tool.start(
                cls,
                context,
                arguments,
                run_config=cls._child_run_config(run_config, context=context),
            )
            snapshot = handle.snapshot()
            return ToolExecutionResult(
                tool_call_id="",
                content=json.dumps(snapshot.to_dict(), ensure_ascii=False),
                metadata={
                    "agent": tool.agent.name,
                    "mode": "background_task",
                    "task_id": handle.task_id,
                    "child_status": snapshot.status.value,
                },
            )
        if tool.metadata.get("mode") == "agent_as_tool" and isinstance(tool.metadata.get("agent"), Agent):
            child_agent = tool.metadata["agent"]
            child = cls._run_child_agent(
                child_agent,
                arguments=arguments,
                parent_config=run_config,
                context=context,
            )
            return ToolExecutionResult(
                tool_call_id="",
                content=child.final_output or "",
                metadata={
                    "agent": child_agent.name,
                    "mode": tool.metadata.get("mode"),
                    "child_status": child.status.value,
                    "child_run_id": child.run_id,
                },
            )
        return tool.to_tool_execution_result(tool.invoke(context, arguments))

    @staticmethod
    def _tool_run_config_from_context(*, context: ToolContext, fallback: RunConfig) -> RunConfig:
        runtime_metadata = context.ctx.metadata if context.ctx is not None else {}
        task_metadata = context.task_metadata if isinstance(context.task_metadata, dict) else {}
        is_sub_task = task_metadata.get("is_sub_task") is True
        policy_keys_present = any(key in runtime_metadata for key in _TOOL_POLICY_METADATA_KEYS)

        if is_sub_task or policy_keys_present:
            allowed = runtime_metadata.get("_vv_agent_allowed_tools")
            disallowed = runtime_metadata.get("_vv_agent_disallowed_tools")
            can_use_tool = runtime_metadata.get("_vv_agent_tool_policy_can_use_tool")
            approval = runtime_metadata.get("_vv_agent_tool_policy_approval")
            tool_policy = None
            if policy_keys_present:
                tool_policy = ToolPolicy(
                    allowed_tools=([name for name in allowed if isinstance(name, str)] if isinstance(allowed, list) else None),
                    disallowed_tools=(
                        [name for name in disallowed if isinstance(name, str)] if isinstance(disallowed, list) else []
                    ),
                    approval=(approval if approval in {"always", "never", "on_request"} else "default"),
                    can_use_tool=can_use_tool if callable(can_use_tool) else None,
                )
        else:
            tool_policy = fallback.tool_policy

        def runtime_capability(key: str, fallback_value: Any) -> Any:
            return runtime_metadata.get(key, fallback_value)

        return replace(
            fallback,
            tool_policy=tool_policy,
            approval_provider=runtime_capability(
                "_vv_agent_approval_provider",
                fallback.approval_provider,
            ),
            approval_broker=runtime_capability(
                "_vv_agent_approval_broker",
                fallback.approval_broker,
            ),
            approval_timeout_seconds=runtime_capability(
                "_vv_agent_approval_timeout_seconds",
                fallback.approval_timeout_seconds,
            ),
        )

    @staticmethod
    def _tool_is_enabled(*, tool: FunctionTool, agent: Agent, run_config: RunConfig) -> bool:
        if callable(tool.is_enabled):
            run_context = RunContext(
                context=run_config.context,
                metadata={**agent.metadata, **run_config.metadata},
            )
            return bool(tool.is_enabled(run_context, agent))
        return bool(tool.is_enabled)

    @staticmethod
    def _policy_denial_result(
        tool: FunctionTool,
        *,
        context: ToolContext,
        arguments: dict[str, Any],
        run_config: RunConfig,
    ) -> ToolExecutionResult | None:
        policy = run_config.tool_policy
        policy_source: str | None = None
        if policy is not None:
            if policy.allowed_tools is not None and tool.name not in policy.allowed_tools:
                policy_source = "allowed_tools"
            elif tool.name in policy.disallowed_tools:
                policy_source = "disallowed_tools"
            elif policy.can_use_tool is not None and not bool(policy.can_use_tool(tool.name, dict(arguments))):
                policy_source = "can_use_tool"

        planned_tool_names = context.metadata.get(_PLANNED_TOOL_NAMES_METADATA_KEY)
        if (
            policy_source is None
            and isinstance(planned_tool_names, (frozenset, set, list, tuple))
            and tool.name not in planned_tool_names
        ):
            policy_source = "planned_name"

        if policy_source is None:
            return None

        message = f"Tool {tool.name} is not allowed for these arguments."
        return ToolExecutionResult(
            tool_call_id="",
            content=json.dumps(
                {
                    "ok": False,
                    "error": message,
                    "error_code": "tool_not_allowed",
                    "tool_name": tool.name,
                },
                ensure_ascii=False,
            ),
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="tool_not_allowed",
            metadata={
                "mode": "permission_denied",
                "policy_source": policy_source,
                "tool_name": tool.name,
                "arguments": dict(arguments),
                "message": message,
            },
        )

    @staticmethod
    def _approval_result(
        tool: FunctionTool,
        *,
        context: ToolContext,
        arguments: dict[str, Any],
        run_config: RunConfig,
    ) -> ToolExecutionResult | None:
        if context.ctx is not None:
            approved = getattr(context.ctx, "_approved_tool_approval", None)
            approved_call = getattr(approved, "call", None)
            if (
                approved_call is not None
                and approved_call.id == context.tool_call_id
                and approved_call.name == tool.name
                and approved_call.arguments == arguments
            ):
                return None
        policy = run_config.tool_policy
        if policy is not None and policy.approval == "never":
            return None
        needs_approval = policy is not None and policy.approval == "always"
        if callable(tool.needs_approval):
            needs_approval = needs_approval or bool(tool.needs_approval(context, arguments))
        else:
            needs_approval = needs_approval or bool(tool.needs_approval)
        if not needs_approval:
            return None
        if run_config.approval_provider is not None:
            return Runner._brokered_approval_result(
                tool,
                context=context,
                arguments=arguments,
                run_config=run_config,
            )
        message = f"Approval required for tool {tool.name}."
        runtime_metadata = context.ctx.metadata if context.ctx is not None else {}
        run_id = str(runtime_metadata.get("_vv_agent_run_id") or "run")
        interruption_id = ApprovalRequest.create(
            tool_name=tool.name,
            tool_call_id=context.tool_call_id,
            arguments=arguments,
            run_id=run_id,
            trace_id=str(runtime_metadata.get("_vv_agent_trace_id") or runtime_metadata.get("trace_id") or ""),
            agent_name=str(runtime_metadata.get("_vv_agent_agent_name") or ""),
            cycle_index=context.cycle_index,
        ).request_id
        return ToolExecutionResult(
            tool_call_id="",
            content=message,
            status_code=ToolResultStatus.WAIT_RESPONSE,
            directive=ToolDirective.WAIT_USER,
            error_code="tool_approval_required",
            metadata={
                "mode": "approval_requested",
                "approval_required": True,
                "approval_interruption_id": interruption_id,
                "request_id": interruption_id,
                "tool_name": tool.name,
                "arguments": dict(arguments),
                "message": message,
            },
        )

    @staticmethod
    def _brokered_approval_result(
        tool: FunctionTool,
        *,
        context: ToolContext,
        arguments: dict[str, Any],
        run_config: RunConfig,
    ) -> ToolExecutionResult | None:
        provider = run_config.approval_provider
        broker = run_config.approval_broker
        runtime_metadata = context.ctx.metadata if context.ctx is not None else {}
        run_id = str(runtime_metadata.get("_vv_agent_run_id") or "")
        trace_id = str(runtime_metadata.get("_vv_agent_trace_id") or runtime_metadata.get("trace_id") or "")
        agent_name = str(runtime_metadata.get("_vv_agent_agent_name") or "")
        request = ApprovalRequest.create(
            tool_name=tool.name,
            tool_call_id=context.tool_call_id,
            arguments=arguments,
            run_id=run_id,
            trace_id=trace_id,
            agent_name=agent_name,
            cycle_index=context.cycle_index,
            metadata={
                "tool_metadata": dict(tool.metadata),
                "session_id": str(runtime_metadata.get("session_id") or runtime_metadata.get("_vv_agent_session_id") or ""),
            },
        )

        def check_cancelled() -> None:
            if context.ctx is not None:
                context.ctx.check_cancelled()

        if provider is None:
            return None
        if broker is None:
            broker = ApprovalBroker()
        try:
            session_allowed = broker.is_session_allowed(tool.name)
        except Exception as exc:
            raise ApprovalError(str(exc)) from exc
        if session_allowed:
            return None
        try:
            should_request = provider.should_request(request)
        except Exception as exc:
            raise ApprovalError(str(exc)) from exc
        check_cancelled()
        if not should_request:
            return None
        try:
            broker.register(request)
        except Exception as exc:
            broker.discard(request.request_id)
            raise ApprovalError(str(exc)) from exc
        bind_request_cancellation(broker, request.request_id, context.ctx.cancellation_token if context.ctx else None)

        def emit(event: RunEvent) -> None:
            event_sink = runtime_metadata.get("_vv_agent_emit_event")
            if callable(event_sink):
                event_sink(event)

        message = f"Approval required for tool {tool.name}."
        emit(
            ApprovalRequestedEvent(
                run_id=run_id,
                trace_id=trace_id,
                agent_name=agent_name,
                session_id=str(request.metadata.get("session_id") or "") or None,
                cycle_index=context.cycle_index,
                request_id=request.request_id,
                tool_name=tool.name,
                tool_call_id=context.tool_call_id,
                message=message,
                metadata={
                    "arguments": dict(arguments),
                    "tool_name": tool.name,
                },
            )
        )

        try:
            decision = provider.decide(request)
        except Exception as exc:
            broker.discard(request.request_id)
            raise ApprovalError(str(exc)) from exc
        check_cancelled()
        try:
            if decision is None:
                decision = broker.wait(request.request_id, timeout=run_config.approval_timeout_seconds)
            else:
                broker.resolve(request.request_id, decision)
                decision = broker.wait(request.request_id, timeout=0)
        except Exception as exc:
            broker.discard(request.request_id)
            raise ApprovalError(str(exc)) from exc
        approved = decision.action in {"allow", "allow_session"}
        emit(
            ApprovalResolvedEvent(
                run_id=run_id,
                trace_id=trace_id,
                agent_name=agent_name,
                session_id=str(request.metadata.get("session_id") or "") or None,
                cycle_index=context.cycle_index,
                request_id=request.request_id,
                tool_name=tool.name,
                tool_call_id=context.tool_call_id,
                action=decision.action,
                approved=approved,
                metadata={
                    "action": decision.action,
                    "reason": decision.reason,
                    "decision_metadata": dict(decision.metadata),
                },
            )
        )
        check_cancelled()
        if approved:
            return None
        if decision.action == "timeout":
            return Runner._approval_error_result(
                tool=tool,
                context=context,
                arguments=arguments,
                request_id=request.request_id,
                error_code="tool_approval_timeout",
                action=decision.action,
                message=decision.reason or "Approval request timed out.",
            )
        return Runner._approval_error_result(
            tool=tool,
            context=context,
            arguments=arguments,
            request_id=request.request_id,
            error_code="tool_approval_denied",
            action=decision.action,
            message=decision.reason or f"Approval denied for tool {tool.name}.",
        )

    @staticmethod
    def _approval_error_result(
        *,
        tool: FunctionTool,
        context: ToolContext,
        arguments: dict[str, Any],
        request_id: str,
        error_code: str,
        action: str,
        message: str,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_call_id=context.tool_call_id,
            content=json.dumps(
                {
                    "ok": False,
                    "error": message,
                    "error_code": error_code,
                    "tool_name": tool.name,
                },
                ensure_ascii=False,
            ),
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code=error_code,
            metadata={
                "mode": "approval_resolved",
                "request_id": request_id,
                "tool_name": tool.name,
                "arguments": dict(arguments),
                "action": action,
                "message": message,
            },
        )

    @staticmethod
    def _extract_handoff(result: RunResult) -> _HandoffRequest | None:
        if result.status != AgentStatus.COMPLETED:
            return None
        for cycle in result.raw_result.cycles:
            for tool_result in cycle.tool_results:
                metadata = tool_result.metadata if isinstance(tool_result.metadata, dict) else {}
                if metadata.get("mode") != "handoff":
                    continue
                source_agent = metadata.get("handoff_from")
                target_agent = metadata.get("handoff_to")
                tool_name = metadata.get("handoff_tool_name")
                handoff_input = metadata.get("handoff_input")
                if not isinstance(source_agent, str) or not source_agent:
                    raise RuntimeError("handoff tool result is missing canonical control metadata")
                if not isinstance(target_agent, str) or not target_agent:
                    raise RuntimeError("handoff tool result is missing canonical control metadata")
                if not isinstance(tool_name, str) or not tool_name:
                    raise RuntimeError("handoff tool result is missing canonical control metadata")
                if not isinstance(handoff_input, str) or not handoff_input:
                    raise RuntimeError("handoff tool result is missing canonical control metadata")
                return _HandoffRequest(
                    source_agent=source_agent,
                    target_agent=target_agent,
                    tool_name=tool_name,
                    tool_call_id=tool_result.tool_call_id,
                    input=handoff_input,
                    cycle_index=cycle.index,
                    metadata=dict(metadata),
                )
        return None

    @classmethod
    def _run_child_agent(
        cls,
        child_agent: Agent,
        *,
        arguments: dict[str, Any],
        parent_config: RunConfig,
        context: ToolContext | None = None,
    ) -> RunResult:
        child_input = cls._child_agent_prompt(arguments=arguments, context=context)
        return cls.run_sync(
            child_agent,
            child_input,
            run_config=cls._child_run_config(parent_config, context=context),
        )

    @staticmethod
    def _child_run_config(parent_config: RunConfig, *, context: ToolContext | None = None) -> RunConfig:
        cancellation_token = parent_config.cancellation_token
        if cancellation_token is not None:
            cancellation_token = cancellation_token.child()
        return replace(
            parent_config,
            model=None,
            model_settings=None,
            session=None,
            stream=None,
            shared_state=deepcopy(context.shared_state) if context is not None else None,
            initial_messages=None,
            before_cycle_messages=None,
            interruption_messages=None,
            sub_task_manager=None,
            runtime_log_handler=None,
            runtime_stream_callback=None,
            cancellation_token=cancellation_token,
            host_cost_meter=None,
            metadata={
                key: value
                for key, value in parent_config.metadata.items()
                if key != _INITIAL_BUDGET_USAGE_METADATA_KEY
            },
        )

    @staticmethod
    def _child_agent_prompt(*, arguments: dict[str, Any], context: ToolContext | None) -> str:
        task_description = next(
            (
                value.strip()
                for key in ("task_description", "task", "input", "prompt")
                if isinstance((value := arguments.get(key)), str) and value.strip()
            ),
            "",
        )
        if not task_description:
            raise ValueError("agent tool requires task_description")
        output_requirements = arguments.get("output_requirements")
        if isinstance(output_requirements, str) and output_requirements.strip():
            task_description += f"\n\n<Output Requirements>\n{output_requirements.strip()}\n</Output Requirements>"
        if arguments.get("include_main_summary") is True and context is not None:
            runtime_metadata = context.ctx.metadata if context.ctx is not None else {}
            parent_summary = context.shared_state.get("main_task_summary")
            if not isinstance(parent_summary, str) or not parent_summary.strip():
                parent_summary = runtime_metadata.get("_vv_agent_input")
            if isinstance(parent_summary, str) and parent_summary.strip():
                task_description += f"\n\n<Main Task Summary>\n{parent_summary.strip()}\n</Main Task Summary>"
        return task_description

    @staticmethod
    def _session_initial_messages(run_config: RunConfig) -> list[Message] | None:
        if run_config.session is None:
            return None
        items = run_config.session.get_items()
        return list(items) or None

    @staticmethod
    def _new_session_items(*, initial_messages: list[Message] | None, result: AgentResult) -> list[Message]:
        history = list(initial_messages or [])
        result_messages = list(result.messages)
        prefix_length = len(history)
        if not history or history[0].role != "system":
            prefix_length += 1
        if prefix_length > len(result_messages):
            return []
        return deepcopy(result_messages[prefix_length:])

    @staticmethod
    def _session_items_for_persistence(items: list[Message], result: AgentResult) -> list[Message]:
        if result.status != AgentStatus.WAIT_USER:
            return items
        approval_call_ids = {
            tool_result.tool_call_id
            for cycle in result.cycles
            for tool_result in cycle.tool_results
            if tool_result.error_code == "tool_approval_required"
        }
        if not approval_call_ids:
            return items
        return [item for item in items if not (item.role == "tool" and item.tool_call_id in approval_call_ids)]

    @staticmethod
    def _normalize_input(input: str) -> str:
        return str(input)

    @staticmethod
    def _initial_budget_usage(run_config: RunConfig) -> BudgetUsageSnapshot | None:
        value = run_config.metadata.get(_INITIAL_BUDGET_USAGE_METADATA_KEY)
        if value is None:
            return None
        if isinstance(value, BudgetUsageSnapshot):
            return value
        if isinstance(value, dict):
            return BudgetUsageSnapshot.from_dict(value)
        raise TypeError(f"RunConfig.metadata[{_INITIAL_BUDGET_USAGE_METADATA_KEY!r}] must be an object")

    @staticmethod
    def _metadata_with_initial_budget_usage(
        metadata: dict[str, Any],
        usage: BudgetUsageSnapshot | None,
    ) -> dict[str, Any]:
        merged = dict(metadata)
        if usage is None:
            merged.pop(_INITIAL_BUDGET_USAGE_METADATA_KEY, None)
        else:
            merged[_INITIAL_BUDGET_USAGE_METADATA_KEY] = usage.to_dict()
        return merged

    @staticmethod
    def _resolve_workspace(workspace: Any | None) -> Path | None:
        if workspace is None or hasattr(workspace, "read_text"):
            return None
        return Path(workspace).expanduser().resolve()

    @staticmethod
    def _resolve_trace_id(run_config: RunConfig) -> str:
        tracing = run_config.tracing or {}
        candidate = tracing.get("trace_id") if isinstance(tracing, dict) else None
        return str(candidate) if candidate else new_trace_id()

    @staticmethod
    def _resolve_event_session_id(run_config: RunConfig) -> str | None:
        session = getattr(run_config.session, "session_id", None)
        candidate = session if session is not None else run_config.metadata.get("session_id")
        if candidate is None:
            return None
        normalized = str(candidate).strip()
        return normalized or None

    @staticmethod
    def _workflow_name(run_config: RunConfig) -> str | None:
        tracing = run_config.tracing or {}
        candidate = tracing.get("workflow_name") if isinstance(tracing, dict) else None
        return str(candidate) if candidate else None

    @staticmethod
    def _trace_processors(run_config: RunConfig) -> list[TraceProcessor]:
        tracing = run_config.tracing or {}
        raw_processors = tracing.get("processors") if isinstance(tracing, dict) else None
        if not isinstance(raw_processors, list):
            return []
        processors: list[TraceProcessor] = []
        for processor in raw_processors:
            if callable(getattr(processor, "on_span_start", None)) and callable(getattr(processor, "on_span_end", None)):
                processors.append(processor)
        return processors

    @staticmethod
    def _infer_backend(settings_file: str | Path, model: str) -> str:
        settings = load_llm_settings_from_file(settings_file)
        providers = settings.get("providers")
        if not isinstance(providers, dict):
            providers = settings.get("backends")
        matches: list[str] = []
        if isinstance(providers, dict):
            for backend, config in providers.items():
                models = config.get("models") if isinstance(config, dict) else None
                if isinstance(models, dict) and model in models:
                    matches.append(str(backend))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"Cannot infer backend for model {model!r}; set RunConfig.default_backend.")
        raise ValueError(f"Model {model!r} exists in multiple backends {matches}; set RunConfig.default_backend.")

    @staticmethod
    def _direct_resolved(model: str) -> ResolvedModelConfig:
        endpoint = EndpointConfig(endpoint_id="direct", api_key="", api_base="")
        return ResolvedModelConfig(
            backend="direct",
            requested_model=model,
            selected_model=model,
            model_id=model,
            endpoint_options=[EndpointOption(endpoint=endpoint, model_id=model)],
        )


@dataclass(frozen=True, slots=True)
class ConfiguredRunner:
    default_run_config: RunConfig = field(default_factory=RunConfig)

    def run_sync(self, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunResult:
        return Runner._run(
            agent,
            input,
            run_config=run_config or RunConfig(),
            resume_runner=self,
            runner_defaults=self.default_run_config,
        )

    def stream_sync(self, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> Iterator[RunEvent]:
        handle = self.start(agent, input, run_config=run_config)
        yield from handle.events()
        handle.result()

    def start(self, agent: Agent, input: str, *, run_config: RunConfig | None = None) -> RunHandle:
        config = run_config or RunConfig()
        control_config = replace(
            config,
            cancellation_token=config.cancellation_token or self.default_run_config.cancellation_token,
            approval_broker=config.approval_broker or self.default_run_config.approval_broker,
        )
        return RunHandle._start_worker(agent=agent, input=input, run_config=control_config, runner=self)

    def resume(self, state: RunState, *, input: str | None = None) -> RunResult:
        return Runner.resume(state, input=input)

    def _run(
        self,
        agent: Agent,
        input: str,
        *,
        run_config: RunConfig,
        event_sink: Callable[[RunEvent], None] | None = None,
        initial_result: RunResult | None = None,
        _compiled_invocation: _CompiledTaskInvocation | None = None,
    ) -> RunResult:
        return Runner._run(
            agent,
            input,
            run_config=run_config,
            event_sink=event_sink,
            resume_runner=self,
            runner_defaults=self.default_run_config,
            initial_result=initial_result,
            _compiled_invocation=_compiled_invocation,
        )
