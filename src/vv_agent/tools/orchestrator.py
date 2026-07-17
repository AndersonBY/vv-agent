from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterable
from dataclasses import replace
from typing import Any, cast

from vv_agent.approval import ApprovalBroker, ApprovalError, ApprovalProvider, ApprovalRequest, bind_request_cancellation
from vv_agent.events import ApprovalRequestedEvent, ApprovalResolvedEvent, RunEvent, ToolCallCompletedEvent, ToolCallStartedEvent
from vv_agent.tools.base import ToolContext, is_tool_call_preapproved
from vv_agent.tools.dispatcher import _needs_tool_call_id, _parse_arguments
from vv_agent.tools.executor import RegistryToolExecutor, ToolExecutor, is_tool_executor
from vv_agent.tools.function import FunctionTool, Tool, adapt_tool
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus

ToolEventSink = Callable[[RunEvent], None]
_PLANNED_TOOL_NAMES_METADATA_KEY = "_vv_agent_planned_tool_names"
_TOOL_DISPATCH_CALLBACK_METADATA_KEY = "_vv_agent_tool_dispatch_callback"
_TOOL_DISPATCH_STARTED_METADATA_KEY = "_vv_agent_tool_dispatch_started"


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
        if executor is None:
            policy_result = self._policy_denial_result(
                None,
                call=normalized_call,
                context=call_context,
                allowed_tool_names=allowed_tool_names,
            )
            if policy_result is not None:
                return policy_result
            return self._error_result(call.id, f"Unknown tool: {call.name}", error_code="tool_not_found")

        try:
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

        if _needs_tool_call_id(result.tool_call_id):
            result.tool_call_id = normalized_call.id

        if result.directive == ToolDirective.WAIT_USER and result.status_code == ToolResultStatus.SUCCESS:
            result.status_code = ToolResultStatus.WAIT_RESPONSE

        self._emit_completed(normalized_call, result=result, context=call_context, event_sink=event_sink)
        return result

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
        return RegistryToolExecutor(name=spec.name, handler=spec.handler, schema=schema)

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
        event_sink = metadata.get("_vv_agent_emit_event")
        if callable(event_sink):
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
        elif allowed_tool_names is not None and call.name not in allowed_tool_names:
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
            status="error",
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
            approved=approved,
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
            status="error",
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
        event_sink = _runtime_metadata(context).get("_vv_agent_emit_event")
        if not callable(event_sink):
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
        approved: bool,
        action: str,
        reason: str,
        decision_metadata: dict[str, Any],
    ) -> None:
        event_sink = _runtime_metadata(context).get("_vv_agent_emit_event")
        if not callable(event_sink):
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
                approved=approved,
                metadata={
                    "action": action,
                    "reason": reason,
                    "decision_metadata": decision_metadata,
                },
            )
        )

    @staticmethod
    def _error_result(tool_call_id: str, message: str, *, error_code: str | None = None) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_call_id=tool_call_id,
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code=error_code,
            content=json.dumps({"ok": False, "error": message, "error_code": error_code}, ensure_ascii=False),
        )

    @staticmethod
    def _emit_started(call: ToolCall, *, context: ToolContext, event_sink: ToolEventSink | None) -> None:
        if event_sink is None:
            return
        metadata = _runtime_metadata(context)
        event_sink(
            ToolCallStartedEvent(
                run_id=str(metadata.get("_vv_agent_run_id") or ""),
                trace_id=str(metadata.get("_vv_agent_trace_id") or metadata.get("trace_id") or ""),
                agent_name=str(metadata.get("_vv_agent_agent_name") or ""),
                cycle_index=context.cycle_index,
                tool_name=call.name,
                tool_call_id=call.id,
                metadata={"arguments": dict(call.arguments)},
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
        event_sink(
            ToolCallCompletedEvent(
                run_id=str(metadata.get("_vv_agent_run_id") or ""),
                trace_id=str(metadata.get("_vv_agent_trace_id") or metadata.get("trace_id") or ""),
                agent_name=str(metadata.get("_vv_agent_agent_name") or ""),
                cycle_index=context.cycle_index,
                tool_name=call.name,
                tool_call_id=result.tool_call_id,
                status=result.status_code.value if result.status_code else str(result.status or ""),
                metadata={
                    "arguments": dict(call.arguments),
                    "directive": result.directive.value,
                    "error_code": result.error_code,
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
    callback = context.metadata.get(_TOOL_DISPATCH_CALLBACK_METADATA_KEY)
    if callable(callback):
        callback(call)
    ToolOrchestrator._emit_started(call, context=context, event_sink=event_sink)


def _is_cancelled_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "CancelledError" and exc.__class__.__module__ == "vv_agent.runtime.cancellation"
