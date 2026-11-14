"""Concrete SDK observations and effects for core dispatch and finalization."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from vv_agent.kernel import _child_delegation as delegation
from vv_agent.kernel import _child_summary as summary
from vv_agent.kernel import _driver_dispatch as core
from vv_agent.kernel import _tool_admission as admission
from vv_agent.kernel._token_count import TokenCount
from vv_agent.memory.manager import MemoryManager
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import builtin_error
from vv_agent.tools.orchestrator import ToolOrchestrator
from vv_agent.types import Message, ToolCall, ToolExecutionResult

from ._dispatch_context import begin_context, finish_context
from ._dispatch_delegation import ChildCompilationHost
from ._dispatch_model import invoke_model
from ._dispatch_tools import PreflightHost
from ._driver_scope import _invoke
from .children import ChildTool, SummaryChildTool
from .compaction import manager_for, tokens
from .delegation import admitted_result
from .providers import Definitive, Provider
from .records import Record
from .reducer import Attempt
from .runtime import EndpointUnavailable

if TYPE_CHECKING:
    from .kernel import _Driver


class _DispatchHost:
    def __init__(self, driver: _Driver, *, context=None, binding=None):
        self.driver = driver
        self.context = cast(ToolContext, context)
        self.binding = binding
        self.provider: Provider | None = None
        self.prepared_client = None
        self.preflight = None
        self.manager: MemoryManager | None = None
        self.child_compiler: ChildCompilationHost | None = None
        self.after_tool: Callable[..., ToolExecutionResult]
        self.after_tool_arguments: dict[str, Any] = {}

    def run[T](self, steps: Generator[object, Any, T]) -> T:
        from ._recovery_phase import run

        return run(self.driver, steps, dispatch_host=self)

    def _handle(self, request):
        driver = self.driver
        if isinstance(request, core.PrepareContext):
            self.context_task, self.context_plan = request.task, request.plan
            self.context_shared = driver.shared_state()
            self.prepared_context = begin_context(
                driver.runtime,
                request.task,
                request.plan,
                driver.scope.token,
                approved=request.approved,
                shared_state=self.context_shared,
            )
            return None
        if isinstance(request, core.FinishContext):
            self.context = finish_context(
                driver.runtime,
                self.prepared_context,
                self.context_task,
                self.context_plan,
                request.facts,
                shared_state=self.context_shared,
                store=driver.store,
                session_id=driver.sid,
            )
            return self.context.metadata
        if request is core.DispatchRead.HAS_HOOKS:
            return driver.runtime.hooks.has_hooks()
        if request is core.DispatchRead.CONTEXT_CYCLE:
            return self.context.cycle_index
        if isinstance(request, core.SetContextCycle):
            self.context.cycle_index = request.cycle
        elif request is core.DispatchRead.CONTEXT_SHARED:
            return self.context.shared_state
        elif request is core.DispatchRead.DURABLE_CONTEXT:
            return driver.runtime.durable_state(self.context.shared_state)
        elif request is core.DispatchRead.HANDLER_VERSION:
            return driver.runtime.handler_version
        elif request is core.DispatchRead.POLL_LEASE:
            return driver.scope.poll(driver.store)
        elif request is core.DispatchRead.CANCELLED:
            return driver.scope.token.cancelled
        elif isinstance(request, core.BeforeTool):
            return driver.runtime.hooks.apply_before_tool_call(
                task=request.task,
                cycle_index=self.context.cycle_index,
                call=request.call,
                context=self.context,
            )
        elif isinstance(request, core.BindPreparedState):
            return driver.runtime.bind_state(request.shared, request.task)
        elif isinstance(request, core.ApprovalDenied):
            executor = driver.runtime.functions.orchestrator._resolve_executor(request.plan._payload["request"]["name"])
            assert executor is not None
            return ToolOrchestrator._approval_error_result(
                executor=executor,
                call=ToolCall.from_dict(request.plan.payload["request"]),
                request_id=request.answer["request_id"],
                error_code="tool_approval_timeout" if request.action == "timeout" else "tool_approval_denied",
                action=request.action,
                message=request.answer.get("reason")
                or (
                    "Approval request timed out." if request.action == "timeout" else f"Approval denied for tool {executor.name}."
                ),
            )
        elif isinstance(request, core.ApproveSessionCall):
            assert self.context.ctx is not None
            self.context.ctx._approved_tool_approval = SimpleNamespace(call=request.call)
        elif isinstance(request, core.PrepareModel):
            try:
                self.prepared_client = driver.runtime.prepare_model(
                    request.plan._payload["request"], request.plan._payload["purpose"], request.plan.attempt or 1
                )
            except EndpointUnavailable:
                return False
            return True
        elif isinstance(request, core.SelectProvider):
            self.provider = driver.provider(request.plan)
            return self.provider is not None
        elif isinstance(request, core.SubmitInteraction):
            return driver.runtime.functions.submit(request.plan, context=self.context)
        elif isinstance(request, core.InteractionFailure):
            return builtin_error(request.message, request.code)
        elif isinstance(request, core.ChildBinding):
            self.binding = driver.runtime.children.get(request.name)
            return isinstance(self.binding, SummaryChildTool)
        elif isinstance(request, core.HasChild):
            return request.name in driver.runtime.children
        elif isinstance(request, core.HasDelegation):
            return request.name in driver.runtime.delegated_tools
        elif isinstance(request, core.AdmitChildren):
            driver.commit(request.records, guarded=True, prepare=lambda tx: self.run(core.admit_children(driver, tx, request)))
        elif isinstance(request, core.CallChildAdmission):
            binding = self.binding
            if isinstance(binding, SummaryChildTool):
                return binding.admit(request.plan, request.summary)
            if isinstance(binding, ChildTool):
                return binding.admit(request.plan)
            return binding(request.plan) if binding is not None else request.child_specs
        elif isinstance(request, core.BackgroundAdmissionResult):
            return admitted_result(driver.store, request.plan, request.handles)
        elif isinstance(request, core.InvokeOperation):
            return _invoke(
                driver.scope,
                (lambda: cast(Provider, self.provider).submit(request.plan, context=self.context))
                if request.has_provider
                else (
                    lambda: invoke_model(driver, request.plan, request.task, self.context, request.started, self.prepared_client)
                ),
                driver.runtime.model_timeout if request.plan._payload["op_kind"] == "model" else driver.runtime.tool_timeout,
            )
        elif isinstance(request, core.DispatchHook):
            driver.runtime.hook(request.point, request.plan)
        elif isinstance(request, core.CommitDispatch):
            driver.commit(request.records, guarded=request.guarded)
        elif request is core.DispatchRead.PREPARE_AFTER_TOOL:
            self.after_tool = driver.runtime.hooks.apply_after_tool_call
        elif isinstance(request, core.PrepareAfterTool):
            call = request.call
            self.after_tool_arguments = dict(
                task=request.task,
                cycle_index=self.context.cycle_index,
                call=call,
                context=replace(self.context, tool_call_id=call.id, tool_name=call.name, arguments=dict(call.arguments)),
            )
        elif isinstance(request, core.AfterTool):
            return self.after_tool(**self.after_tool_arguments, result=request.result)
        elif isinstance(request, core.PreflightHandler):
            self.preflight = PreflightHost(driver.runtime.functions.orchestrator, self.context)
            return self.preflight.handle(request)
        elif isinstance(
            request,
            (
                core.BeginPreflight,
                core.PrepareAdmission,
                core.AdmissionException,
                core.FormatAdmissionException,
                core.InvokePreflight,
                core.FinishPreflight,
                core.EndPreflight,
                admission.Read,
                admission.ApprovedCall,
                admission.PolicyPredicate,
                admission.RequiresApproval,
                admission.RequestApproval,
                admission.SchemaCheck,
            ),
        ):
            assert self.preflight is not None
            return self.preflight.handle(request)
        elif isinstance(request, delegation.ChildBindingMode):
            self.child_compiler = ChildCompilationHost(driver.runtime, self.context)
            return self.child_compiler.handle(request)
        elif isinstance(request, delegation.ChildRequest):
            assert self.child_compiler is not None
            return self.child_compiler.handle(request)
        elif isinstance(request, summary.RequiresSummary):
            return cast(SummaryChildTool, self.binding).requires_summary(request.plan)
        elif request is summary.SummaryRead.PREPARE_MEMORY:
            self.manager = manager_for(driver)
            self.manager.protected_start = None
        elif request is summary.SummaryRead.PLAN_SETTINGS:
            manager = cast(MemoryManager, self.manager)
            return {
                "protected_start": manager.protected_start,
                "language": manager.language,
                "event_limit": manager.summary_event_limit,
            }
        elif request is summary.SummaryRead.ACCEPT_SETTINGS:
            manager = cast(MemoryManager, self.manager)
            return {
                "protected_start": manager.protected_start,
                "recovery_tool_available": manager.recovery_tool_available,
                "effective_context_window": manager.effective_context_window,
                "model": manager.model,
            }
        elif request is summary.SummaryRead.MODEL_CONTEXT_WINDOW:
            return cast(MemoryManager, self.manager).model_context_window
        elif request is summary.SummaryRead.MODEL_ROUTE:
            _, resolved = driver.runtime.model_route("compaction")
            return resolved.model_id
        elif isinstance(request, summary.CountSummaryPrompt):
            return tokens(cast(MemoryManager, self.manager), request.messages)
        elif isinstance(request, summary.CountPrefix):
            messages = request.request.messages
            messages = [messages] if isinstance(messages, Message) else messages
            return request.count_messages([m.to_openai_message() for m in messages], model=request.request.model)
        elif isinstance(request, TokenCount):
            return (
                cast(MemoryManager, self.manager)._estimate_message_tokens(request.messages)
                if isinstance(request.messages, Message)
                else cast(MemoryManager, self.manager)._calculate_message_length(request.messages)
            )
        else:
            raise TypeError(f"unsupported dispatch request: {type(request).__name__}")
        return None


def dispatch(self: _Driver, attempt: Attempt) -> None:
    return _DispatchHost(self).run(core.dispatch(self, attempt))


def finalize_tool(self: _Driver, plan: Record, context, outcome: Definitive) -> Definitive:
    return _DispatchHost(self, context=context).run(core.finalize_tool(self, plan, outcome))
