from __future__ import annotations

import json
import time
from collections.abc import Callable, Collection, Iterable
from dataclasses import replace
from typing import Any, cast

from vv_agent.approval import ApprovalBroker, ApprovalError, ApprovalProvider, ApprovalRequest, bind_request_cancellation
from vv_agent.events import (
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    RunEvent,
    ToolCallCompletedEvent,
    ToolCallPlannedEvent,
    ToolCallStartedEvent,
)
from vv_agent.tools.argument_validation import invalid_tool_arguments_result
from vv_agent.tools.base import ToolContext, is_tool_call_preapproved
from vv_agent.tools.dispatcher import _needs_tool_call_id, _parse_arguments
from vv_agent.tools.executor import (
    RegistryToolExecutor,
    ToolExecutor,
    get_executor_tool_metadata,
    is_tool_executor,
)
from vv_agent.tools.function import FunctionTool, Tool, adapt_tool
from vv_agent.tools.metadata import metadata_policy_denial_source
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus

ToolEventSink = Callable[[RunEvent], None]
ToolResultFinalizer = Callable[[ToolCall, ToolContext, ToolExecutionResult], ToolExecutionResult]
_PLANNED_TOOL_NAMES_METADATA_KEY = "_vv_agent_planned_tool_names"
_TOOL_DISPATCH_CALLBACK_METADATA_KEY = "_vv_agent_tool_dispatch_callback"
_TOOL_DISPATCH_STARTED_METADATA_KEY = "_vv_agent_tool_dispatch_started"
_TOOL_DISPATCH_STARTED_AT_METADATA_KEY = "_vv_agent_tool_dispatch_started_at_ns"
_TOOL_DISPATCH_DURATION_MS_METADATA_KEY = "_vv_agent_tool_dispatch_duration_ms"
_TOOL_TYPED_METADATA_KEY = "_vv_agent_tool_metadata"
_JSON_SAFE_INTEGER_MAX = (1 << 53) - 1


class ToolOrchestrator:
    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        executors: Iterable[ToolExecutor] = (),
    ) -> None:
        self.registry = registry
        self._executors = {executor.name: executor for executor in executors}

    @classmethod
    def from_tools(cls, tools: Iterable[Tool | ToolExecutor]) -> ToolOrchestrator:
        executors: list[ToolExecutor] = []
        for tool in tools:
            if isinstance(tool, FunctionTool):
                executors.append(tool.to_executor())
            elif is_tool_executor(tool):
                executors.append(cast(ToolExecutor, tool))
            else:
                executors.append(adapt_tool(cast(Tool, tool)).to_executor())
        return cls(executors=executors)

    @classmethod
    def from_registry(cls, registry: ToolRegistry) -> ToolOrchestrator:
        return cls(registry=registry)

    def run_one(
        self,
        call: ToolCall,
        *,
        context: ToolContext,
        allowed_tool_names: Collection[str] | None = None,
        event_sink: ToolEventSink | None = None,
        _precomputed_result: ToolExecutionResult | None = None,
        _result_finalizer: ToolResultFinalizer | None = None,
    ) -> ToolExecutionResult:
        arguments, parse_error = _parse_arguments(call.id, call.arguments)
        if parse_error is not None:
            return parse_error
        normalized_call = ToolCall(id=call.id, name=call.name, arguments=arguments, extra_content=call.extra_content)
        context_metadata = dict(context.metadata)
        if allowed_tool_names is not None:
            context_metadata[_PLANNED_TOOL_NAMES_METADATA_KEY] = frozenset(allowed_tool_names)
        call_context = replace(
            context,
            tool_call_id=normalized_call.id,
            tool_name=normalized_call.name,
            arguments=dict(arguments),
            metadata=context_metadata,
        )

        executor = self._resolve_executor(normalized_call.name)
        tool_metadata = get_executor_tool_metadata(executor) if executor is not None else None
        if tool_metadata is not None:
            call_context.metadata[_TOOL_TYPED_METADATA_KEY] = tool_metadata
        self._emit_planned(
            normalized_call,
            executor=executor,
            context=call_context,
            event_sink=event_sink,
        )
        if _precomputed_result is not None:
            result = _precomputed_result
        elif executor is None:
            policy_result = self._policy_denial_result(
                None,
                call=normalized_call,
                context=call_context,
                allowed_tool_names=allowed_tool_names,
            )
            if policy_result is not None:
                result = policy_result
            else:
                result = self._error_result(
                    call.id,
                    f"Unknown tool: {call.name}",
                    error_code="tool_not_found",
                )
        else:
            try:
                validation_result = invalid_tool_arguments_result(
                    tool_call_id=normalized_call.id,
                    schema=executor.params_json_schema,
                    arguments=normalized_call.arguments,
                )
                if validation_result is not None:
                    result = validation_result
                else:
                    policy_result = self._policy_denial_result(
                        executor,
                        call=normalized_call,
                        context=call_context,
                        allowed_tool_names=allowed_tool_names,
                    )
                    if policy_result is not None:
                        result = policy_result
                    else:
                        approval_result = self._approval_result(executor, call=normalized_call, context=call_context)
                        if approval_result is not None:
                            result = approval_result
                        else:
                            if not executor.metadata.get("policy_managed_by_handler"):
                                mark_external_tool_execution_started(
                                    normalized_call,
                                    context=call_context,
                                    event_sink=event_sink,
                                )
                            result = executor.execute(normalized_call, call_context)
            except ApprovalError:
                raise
            except Exception as exc:
                if _is_cancelled_error(exc):
                    raise
                message = (
                    executor.failure_error_function(exc)
                    if executor.failure_error_function is not None
                    else f"Tool execution failed ({normalized_call.name}): {exc}"
                )
                result = self._error_result(
                    normalized_call.id,
                    message,
                    error_code="tool_execution_failed",
                )

        self._capture_execution_duration(call_context)

        if _result_finalizer is not None:
            result = _result_finalizer(normalized_call, call_context, result)

        if _needs_tool_call_id(result.tool_call_id):
            result.tool_call_id = normalized_call.id

        if result.directive == ToolDirective.WAIT_USER and result.status_code == ToolResultStatus.SUCCESS:
            result.status_code = ToolResultStatus.WAIT_RESPONSE

        self._emit_completed(normalized_call, result=result, context=call_context, event_sink=event_sink)
        return result

    @staticmethod
    def _capture_execution_duration(context: ToolContext) -> None:
        if context.metadata.get(_TOOL_DISPATCH_STARTED_METADATA_KEY) is not True:
            return
        if _TOOL_DISPATCH_DURATION_MS_METADATA_KEY in context.metadata:
            return
        started_at_ns = context.metadata.get(_TOOL_DISPATCH_STARTED_AT_METADATA_KEY)
        if not isinstance(started_at_ns, int):
            return
        elapsed_ns = max(time.monotonic_ns() - started_at_ns, 0)
        context.metadata[_TOOL_DISPATCH_DURATION_MS_METADATA_KEY] = min(
            elapsed_ns // 1_000_000,
            _JSON_SAFE_INTEGER_MAX,
        )

    def _resolve_executor(self, name: str) -> ToolExecutor | None:
        executor = self._executors.get(name)
        if executor is not None:
            return executor
        if self.registry is None or not self.registry.has_tool(name):
            return None
        if self.registry.has_executor(name):
            return self.registry.get_executor(name)
        spec = self.registry.get(name)
        schema = self.registry.get_schema(name) if self.registry.has_schema(name) else None
        return RegistryToolExecutor(
            name=spec.name,
            handler=spec.handler,
            schema=schema,
            tool_metadata=spec.tool_metadata,
        )

    @staticmethod
    def _approval_result(
        executor: ToolExecutor,
        *,
        call: ToolCall,
        context: ToolContext,
    ) -> ToolExecutionResult | None:
        if executor.metadata.get("policy_managed_by_handler"):
            return None
        if is_tool_call_preapproved(
            context,
            tool_call_id=call.id,
            tool_name=call.name,
            arguments=dict(call.arguments),
        ):
            return None
        metadata = _runtime_metadata(context)
        approval_mode = str(metadata.get("_vv_agent_tool_policy_approval") or "default")
        if approval_mode == "never":
            return None
        needs_approval = approval_mode == "always"
        if not needs_approval:
            needs_approval = executor.requires_approval(context, call.arguments)
        if not needs_approval:
            return None
        request = ApprovalRequest.create(
            tool_name=executor.name,
            tool_call_id=call.id,
            arguments=call.arguments,
            run_id=str(metadata.get("_vv_agent_run_id") or ""),
            trace_id=str(metadata.get("_vv_agent_trace_id") or metadata.get("trace_id") or ""),
            agent_name=str(metadata.get("_vv_agent_agent_name") or ""),
            cycle_index=context.cycle_index,
            metadata={
                "tool_metadata": dict(executor.metadata),
                "session_id": str(metadata.get("session_id") or metadata.get("_vv_agent_session_id") or ""),
            },
        )
        provider = metadata.get("_vv_agent_approval_provider")
        if provider is not None:
            broker = metadata.get("_vv_agent_approval_broker")
            return ToolOrchestrator._brokered_approval_result(
                executor,
                call=call,
                context=context,
                request=request,
                provider=cast(ApprovalProvider, provider),
                broker=broker if isinstance(broker, ApprovalBroker) else None,
                timeout=metadata.get("_vv_agent_approval_timeout_seconds"),
            )
        message = f"Approval required for tool {executor.name}."
        event_sink = _runtime_event_sink(context)
        if event_sink is not None:
            event_sink(
                ApprovalRequestedEvent(
                    run_id=request.run_id,
                    trace_id=request.trace_id,
                    agent_name=request.agent_name,
                    cycle_index=context.cycle_index,
                    request_id=request.request_id,
                    tool_name=executor.name,
                    tool_call_id=call.id,
                    message=message,
                    metadata={"arguments": dict(call.arguments), "tool_name": executor.name},
                )
            )
        return ToolExecutionResult(
            tool_call_id=call.id,
            content=message,
            status_code=ToolResultStatus.WAIT_RESPONSE,
            directive=ToolDirective.WAIT_USER,
            error_code="tool_approval_required",
            metadata={
                "mode": "approval_requested",
                "approval_required": True,
                "approval_interruption_id": request.request_id,
                "request_id": request.request_id,
                "tool_name": executor.name,
                "arguments": dict(call.arguments),
                "message": message,
            },
        )

    @staticmethod
    def _policy_denial_result(
        executor: ToolExecutor | None,
        *,
        call: ToolCall,
        context: ToolContext,
        allowed_tool_names: Collection[str] | None,
    ) -> ToolExecutionResult | None:
        if executor is not None and executor.metadata.get("policy_managed_by_handler"):
            return None
        metadata = _runtime_metadata(context)
        allowed_tools = metadata.get("_vv_agent_allowed_tools")
        disallowed_tools = metadata.get("_vv_agent_disallowed_tools")
        can_use_tool = metadata.get("_vv_agent_tool_policy_can_use_tool")

        policy_source: str | None = None
        if isinstance(allowed_tools, list) and call.name not in allowed_tools:
            policy_source = "allowed_tools"
        elif isinstance(disallowed_tools, list) and call.name in disallowed_tools:
            policy_source = "disallowed_tools"
        elif callable(can_use_tool) and not bool(can_use_tool(call.name, dict(call.arguments))):
            policy_source = "can_use_tool"
        elif executor is not None:
            policy_source = metadata_policy_denial_source(
                get_executor_tool_metadata(executor),
                denied_side_effects=metadata.get("_vv_agent_denied_side_effects", []),
                denied_capability_tags=metadata.get(
                    "_vv_agent_denied_capability_tags",
                    [],
                ),
                deny_terminal_tools=metadata.get("_vv_agent_deny_terminal_tools", False),
                denied_cost_dimensions=metadata.get(
                    "_vv_agent_denied_cost_dimensions",
                    [],
                ),
            )
        elif allowed_tool_names is not None and call.name not in allowed_tool_names:
            policy_source = "planned_name"

        if policy_source is None and allowed_tool_names is not None and call.name not in allowed_tool_names:
            policy_source = "planned_name"

        if policy_source is None:
            return None
        message = f"Tool {call.name} is not allowed for these arguments."
        return ToolExecutionResult(
            tool_call_id=call.id,
            content=json.dumps(
                {
                    "ok": False,
                    "error": message,
                    "error_code": "tool_not_allowed",
                    "tool_name": call.name,
                },
                ensure_ascii=False,
            ),
            status_code=ToolResultStatus.ERROR,
            error_code="tool_not_allowed",
            metadata={
                "mode": "permission_denied",
                "policy_source": policy_source,
                "tool_name": call.name,
                "arguments": dict(call.arguments),
                "message": message,
            },
        )

    @staticmethod
    def _brokered_approval_result(
        executor: ToolExecutor,
        *,
        call: ToolCall,
        context: ToolContext,
        request: ApprovalRequest,
        provider: ApprovalProvider,
        broker: ApprovalBroker | None,
        timeout: Any,
    ) -> ToolExecutionResult | None:
        def check_cancelled() -> None:
            if context.ctx is not None:
                context.ctx.check_cancelled()

        broker = broker or ApprovalBroker()
        try:
            session_allowed = broker.is_session_allowed(executor.name)
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
        message = f"Approval required for tool {executor.name}."
        ToolOrchestrator._emit_approval_requested(executor, call=call, context=context, request=request, message=message)

        try:
            decision = provider.decide(request)
        except Exception as exc:
            broker.discard(request.request_id)
            raise ApprovalError(str(exc)) from exc
        check_cancelled()
        try:
            if decision is None:
                decision = broker.wait(request.request_id, timeout=timeout if isinstance(timeout, (int, float)) else None)
            else:
                broker.resolve(request.request_id, decision)
                decision = broker.wait(request.request_id, timeout=0)
        except Exception as exc:
            broker.discard(request.request_id)
            raise ApprovalError(str(exc)) from exc

        approved = decision.action in {"allow", "allow_session"}
        ToolOrchestrator._emit_approval_resolved(
            executor,
            call=call,
            context=context,
            request=request,
            action=decision.action,
            reason=decision.reason,
            decision_metadata=dict(decision.metadata),
        )
        check_cancelled()
        if approved:
            return None
        if decision.action == "timeout":
            return ToolOrchestrator._approval_error_result(
                executor=executor,
                call=call,
                request_id=request.request_id,
                error_code="tool_approval_timeout",
                action=decision.action,
                message=decision.reason or "Approval request timed out.",
            )
        return ToolOrchestrator._approval_error_result(
            executor=executor,
            call=call,
            request_id=request.request_id,
            error_code="tool_approval_denied",
            action=decision.action,
            message=decision.reason or f"Approval denied for tool {executor.name}.",
        )

    @staticmethod
    def _approval_error_result(
        *,
        executor: ToolExecutor,
        call: ToolCall,
        request_id: str,
        error_code: str,
        action: str,
        message: str,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_call_id=call.id,
            content=json.dumps(
                {
                    "ok": False,
                    "error": message,
                    "error_code": error_code,
                    "tool_name": executor.name,
                },
                ensure_ascii=False,
            ),
            status_code=ToolResultStatus.ERROR,
            error_code=error_code,
            metadata={
                "mode": "approval_resolved",
                "request_id": request_id,
                "tool_name": executor.name,
                "arguments": dict(call.arguments),
                "action": action,
                "message": message,
            },
        )

    @staticmethod
    def _emit_approval_requested(
        executor: ToolExecutor,
        *,
        call: ToolCall,
        context: ToolContext,
        request: ApprovalRequest,
        message: str,
    ) -> None:
        event_sink = _runtime_event_sink(context)
        if event_sink is None:
            return
        event_sink(
            ApprovalRequestedEvent(
                run_id=request.run_id,
                trace_id=request.trace_id,
                agent_name=request.agent_name,
                cycle_index=context.cycle_index,
                request_id=request.request_id,
                tool_name=executor.name,
                tool_call_id=call.id,
                message=message,
                metadata={"arguments": dict(call.arguments), "tool_name": executor.name},
            )
        )

    @staticmethod
    def _emit_approval_resolved(
        executor: ToolExecutor,
        *,
        call: ToolCall,
        context: ToolContext,
        request: ApprovalRequest,
        action: str,
        reason: str,
        decision_metadata: dict[str, Any],
    ) -> None:
        event_sink = _runtime_event_sink(context)
        if event_sink is None:
            return
        event_sink(
            ApprovalResolvedEvent(
                run_id=request.run_id,
                trace_id=request.trace_id,
                agent_name=request.agent_name,
                cycle_index=context.cycle_index,
                request_id=request.request_id,
                tool_name=executor.name,
                tool_call_id=call.id,
                action=action,
                metadata={
                    "reason": reason,
                    "decision_metadata": decision_metadata,
                },
            )
        )

    @staticmethod
    def _error_result(tool_call_id: str, message: str, *, error_code: str | None = None) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_call_id=tool_call_id,
            status_code=ToolResultStatus.ERROR,
            error_code=error_code,
            content=json.dumps({"ok": False, "error": message, "error_code": error_code}, ensure_ascii=False),
        )

    @staticmethod
    def _emit_planned(
        call: ToolCall,
        *,
        executor: ToolExecutor | None,
        context: ToolContext,
        event_sink: ToolEventSink | None,
    ) -> None:
        if event_sink is None:
            return
        metadata = _runtime_metadata(context)
        session_id = str(metadata.get("_vv_agent_session_id") or metadata.get("session_id") or "") or None
        event_sink(
            ToolCallPlannedEvent(
                run_id=str(metadata.get("_vv_agent_run_id") or ""),
                trace_id=str(metadata.get("_vv_agent_trace_id") or metadata.get("trace_id") or ""),
                agent_name=str(metadata.get("_vv_agent_agent_name") or ""),
                session_id=session_id,
                cycle_index=context.cycle_index,
                tool_name=call.name,
                tool_call_id=call.id,
                arguments=dict(call.arguments),
                tool_metadata=(get_executor_tool_metadata(executor) if executor is not None else None),
                metadata={
                    "tool_name": call.name,
                    "tool_call_id": call.id,
                    "tool_arguments": dict(call.arguments),
                },
            )
        )

    @staticmethod
    def _emit_started(call: ToolCall, *, context: ToolContext, event_sink: ToolEventSink | None) -> None:
        if event_sink is None:
            return
        metadata = _runtime_metadata(context)
        session_id = str(metadata.get("_vv_agent_session_id") or metadata.get("session_id") or "") or None
        event_sink(
            ToolCallStartedEvent(
                run_id=str(metadata.get("_vv_agent_run_id") or ""),
                trace_id=str(metadata.get("_vv_agent_trace_id") or metadata.get("trace_id") or ""),
                agent_name=str(metadata.get("_vv_agent_agent_name") or ""),
                session_id=session_id,
                cycle_index=context.cycle_index,
                tool_name=call.name,
                tool_call_id=call.id,
                arguments=dict(call.arguments),
                tool_metadata=context.metadata.get(_TOOL_TYPED_METADATA_KEY),
                metadata={
                    "tool_name": call.name,
                    "tool_call_id": call.id,
                    "tool_arguments": dict(call.arguments),
                },
            )
        )

    @staticmethod
    def _emit_completed(
        call: ToolCall,
        *,
        result: ToolExecutionResult,
        context: ToolContext,
        event_sink: ToolEventSink | None,
    ) -> None:
        if event_sink is None:
            return
        metadata = _runtime_metadata(context)
        execution_started = context.metadata.get(_TOOL_DISPATCH_STARTED_METADATA_KEY) is True
        duration_ms = context.metadata.get(_TOOL_DISPATCH_DURATION_MS_METADATA_KEY)
        if not execution_started or not isinstance(duration_ms, int):
            duration_ms = None
        session_id = str(metadata.get("_vv_agent_session_id") or metadata.get("session_id") or "") or None
        event_sink(
            ToolCallCompletedEvent(
                run_id=str(metadata.get("_vv_agent_run_id") or ""),
                trace_id=str(metadata.get("_vv_agent_trace_id") or metadata.get("trace_id") or ""),
                agent_name=str(metadata.get("_vv_agent_agent_name") or ""),
                session_id=session_id,
                cycle_index=context.cycle_index,
                tool_name=call.name,
                tool_call_id=result.tool_call_id,
                status=result.status_code.value.lower(),
                directive=result.directive.value,
                error_code=result.error_code,
                execution_started=execution_started,
                duration_ms=duration_ms,
                tool_metadata=context.metadata.get(_TOOL_TYPED_METADATA_KEY),
                metadata={
                    "tool_name": call.name,
                    "tool_call_id": result.tool_call_id,
                    "tool_arguments": dict(call.arguments),
                    "status": result.status_code.value.lower(),
                    "directive": result.directive.value,
                    "error_code": result.error_code,
                    "content": result.content,
                    "metadata": dict(result.metadata),
                },
            )
        )


def _runtime_metadata(context: ToolContext) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if isinstance(context.metadata, dict):
        metadata.update(context.metadata)
    if context.ctx is not None and isinstance(context.ctx.metadata, dict):
        metadata.update(context.ctx.metadata)
    return metadata


def _runtime_event_sink(context: ToolContext) -> ToolEventSink | None:
    if context.ctx is None:
        return None
    return context.ctx.event_handler


def mark_external_tool_execution_started(
    call: ToolCall,
    *,
    context: ToolContext,
    event_sink: ToolEventSink | None = None,
) -> None:
    """Mark the single boundary immediately before an executor can cause effects."""

    if context.metadata.get(_TOOL_DISPATCH_STARTED_METADATA_KEY) is True:
        return
    context.metadata[_TOOL_DISPATCH_STARTED_METADATA_KEY] = True
    context.metadata[_TOOL_DISPATCH_STARTED_AT_METADATA_KEY] = time.monotonic_ns()
    callback = context.metadata.get(_TOOL_DISPATCH_CALLBACK_METADATA_KEY)
    if callable(callback):
        callback(call)
    if event_sink is None:
        event_sink = _runtime_event_sink(context)
    ToolOrchestrator._emit_started(call, context=context, event_sink=event_sink)


def _is_cancelled_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "CancelledError" and exc.__class__.__module__ == "vv_agent.runtime.cancellation"
