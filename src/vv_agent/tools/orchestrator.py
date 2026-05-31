from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterable
from dataclasses import replace
from typing import Any, cast

from vv_agent.approval import ApprovalBroker, ApprovalProvider, ApprovalRequest
from vv_agent.events import ApprovalRequestedEvent, ApprovalResolvedEvent, RunEvent, ToolCallCompletedEvent, ToolCallStartedEvent
from vv_agent.tools.base import ToolContext
from vv_agent.tools.dispatcher import _needs_tool_call_id, _parse_arguments
from vv_agent.tools.executor import RegistryToolExecutor, ToolExecutor
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus

ToolEventSink = Callable[[RunEvent], None]


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
    def from_tools(cls, tools: Iterable[FunctionTool | ToolExecutor]) -> ToolOrchestrator:
        executors: list[ToolExecutor] = []
        for tool in tools:
            if isinstance(tool, FunctionTool):
                executors.append(tool.to_executor())
            else:
                executors.append(tool)
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
        if allowed_tool_names is not None and call.name not in allowed_tool_names:
            return self._error_result(
                call.id,
                f"Tool not allowed: {call.name}",
                error_code="tool_not_allowed",
            )

        arguments, parse_error = _parse_arguments(call.id, call.arguments)
        if parse_error is not None:
            return parse_error
        normalized_call = ToolCall(id=call.id, name=call.name, arguments=arguments, extra_content=call.extra_content)
        call_context = replace(
            context,
            tool_call_id=normalized_call.id,
            tool_name=normalized_call.name,
            arguments=dict(arguments),
        )

        executor = self._resolve_executor(normalized_call.name)
        if executor is None:
            return self._error_result(call.id, f"Unknown tool: {call.name}", error_code="tool_not_found")

        self._emit_started(normalized_call, context=call_context, event_sink=event_sink)
        try:
            approval_result = self._approval_result(executor, call=normalized_call, context=call_context)
            result = approval_result if approval_result is not None else executor.execute(normalized_call, call_context)
        except Exception as exc:
            if _is_cancelled_error(exc):
                raise
            result = self._error_result(
                normalized_call.id,
                f"Tool execution failed ({normalized_call.name}): {exc}",
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
        if not executor.requires_approval(context, call.arguments):
            return None
        metadata = _runtime_metadata(context)
        request = ApprovalRequest.create(
            tool_name=executor.name,
            tool_call_id=call.id,
            arguments=call.arguments,
            run_id=str(metadata.get("_vv_agent_run_id") or ""),
            trace_id=str(metadata.get("_vv_agent_trace_id") or metadata.get("trace_id") or ""),
            agent_name=str(metadata.get("_vv_agent_agent_name") or ""),
            cycle_index=context.cycle_index,
            metadata={"tool_metadata": dict(executor.metadata)},
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
            metadata={
                "mode": "approval_requested",
                "request_id": request.request_id,
                "tool_name": executor.name,
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

        should_request = provider.should_request(request)
        check_cancelled()
        if not should_request:
            return None

        broker = broker or ApprovalBroker()
        broker.register(request)
        message = f"Approval required for tool {executor.name}."
        ToolOrchestrator._emit_approval_requested(executor, call=call, context=context, request=request, message=message)

        decision = provider.decide(request)
        check_cancelled()
        if decision is None:
            decision = broker.wait(request.request_id, timeout=timeout if isinstance(timeout, (int, float)) else None)
        else:
            broker.resolve(request.request_id, decision)
            decision = broker.wait(request.request_id, timeout=0)

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
                message=decision.reason or "Approval request timed out.",
            )
        return ToolOrchestrator._approval_error_result(
            executor=executor,
            call=call,
            request_id=request.request_id,
            error_code="tool_approval_denied",
            message=decision.reason or f"Approval denied for tool {executor.name}.",
        )

    @staticmethod
    def _approval_error_result(
        *,
        executor: ToolExecutor,
        call: ToolCall,
        request_id: str,
        error_code: str,
        message: str,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_call_id=call.id,
            content=json.dumps(
                {
                    "ok": False,
                    "error": message,
                    "error_code": error_code,
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


def _is_cancelled_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "CancelledError" and exc.__class__.__module__ == "vv_agent.runtime.cancellation"
