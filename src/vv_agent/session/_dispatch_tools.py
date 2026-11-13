"""SDK executor/context and notification leaves for core preflight admission."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast

from vv_agent.approval import ApprovalError
from vv_agent.kernel import _driver_dispatch as dispatch
from vv_agent.kernel import _tool_admission as admission
from vv_agent.kernel._token_count import _StopIterationTransport
from vv_agent.tools.base import ToolContext
from vv_agent.tools.orchestrator import (
    ToolOrchestrator,
    _is_cancelled_error,
    _ToolAdmissionHost,
    mark_external_tool_execution_started,
)
from vv_agent.types import ToolExecutionResult

from .providers import DispatchReady, _ready


class PreflightHost:
    def __init__(self, orchestrator: ToolOrchestrator, context: ToolContext):
        self.orchestrator = orchestrator
        self.context = context
        self.call_context = None
        self.executor = None
        self.admission: _ToolAdmissionHost
        self.active = False

    def handle(self, request: object) -> Any:
        if isinstance(request, dispatch.PreflightHandler):
            executor = self.orchestrator._resolve_executor(request.plan.payload["request"]["name"])
            return executor is not None and executor.metadata.get("policy_managed_by_handler")
        if isinstance(request, dispatch.BeginPreflight):
            self.context.metadata["_vv_agent_tool_dispatch_callback"] = _ready
            self.active = True
            return self.context.metadata["session_tool_names"]
        if request is dispatch.EndPreflight.CLEANUP:
            self.close()
            return None
        if isinstance(request, dispatch.PrepareAdmission):
            self.call_context, self.executor = self.orchestrator._prepare_call(
                request.call,
                request.arguments,
                context=self.context,
                allowed_tool_names=request.allowed_tool_names,
                event_sink=None,
            )
            self.admission = _ToolAdmissionHost(self.orchestrator, request.call, self.executor, self.call_context)
            return self.executor is not None
        if isinstance(request, dispatch.AdmissionException):
            return isinstance(request.error, ApprovalError) or _is_cancelled_error(request.error)
        if isinstance(request, dispatch.FormatAdmissionException):
            executor = self.executor
            assert executor is not None
            message = (
                executor.failure_error_function(request.error)
                if executor.failure_error_function is not None
                else f"Tool execution failed ({request.call.name}): {request.error}"
            )
            return self.orchestrator._error_result(request.call.id, message, error_code="tool_execution_failed")
        context = self.call_context
        assert context is not None
        if isinstance(request, dispatch.InvokePreflight):
            assert self.executor is not None
            try:
                if request.mark_started:
                    mark_external_tool_execution_started(request.call, context=context, event_sink=None)
                result = self.executor.execute(request.call, context)
            except DispatchReady:
                return True, None
            return False, result
        if isinstance(request, dispatch.FinishPreflight):
            result = self.orchestrator._finish_call(request.call, request.result, context, event_sink=None, result_finalizer=None)
            if not isinstance(result, ToolExecutionResult):
                raise ValueError("preflight handler bypassed the dispatch boundary")
            return result
        return self.admission.respond(cast(admission.AdmissionRequest, request))

    def close(self) -> None:
        if self.active:
            self.active = False
            self.context.metadata.pop("_vv_agent_tool_dispatch_callback", None)

    def run[T](self, steps: Generator[object, Any, T]) -> T:
        response = None
        error = None
        try:
            while True:
                try:
                    request = steps.throw(error) if error is not None else steps.send(response)
                except StopIteration as done:
                    return done.value
                except _StopIterationTransport as transport:
                    original = transport.original
                    context = original.__context__
                    try:
                        raise original
                    finally:
                        original.__context__ = context
                response, error = None, None
                try:
                    response = self.handle(request)
                except Exception as exc:
                    error = _StopIterationTransport(exc) if isinstance(exc, StopIteration) else exc
        finally:
            self.close()
            steps.close()
