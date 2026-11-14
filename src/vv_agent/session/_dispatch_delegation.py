"""SDK parser, Agent compiler and resource leaves for child admission."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from vv_agent.kernel import _child_delegation as core
from vv_agent.kernel._driver_inputs import DefinitionFacts
from vv_agent.kernel._token_count import _StopIterationTransport
from vv_agent.prompt.builder import build_raw_system_prompt_bundle, build_system_prompt_bundle
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.sub_agents import create_sub_task
from vv_agent.workspace import DiscoveryFilteredWorkspaceBackend

from .delegation import child_agent_prompt, child_config, target_agent

if TYPE_CHECKING:
    from .runtime import Runtime


class ChildCompilationHost:
    def __init__(self, runtime: Runtime, context: ToolContext):
        self.runtime, self.context = runtime, context
        self.transfer = None
        self.agent: Any = None
        self.config: Any = None
        self.child_runtime: Runtime

    def handle(self, request: core.ChildRequest) -> Any:
        runtime, context = self.runtime, self.context
        if isinstance(request, core.ChildBindingMode):
            executor = runtime.registry.get_executor(request.name) if runtime.registry.has_executor(request.name) else None
            mode = executor.metadata.get("mode") if executor else None
            self.transfer = next((t for t in runtime.agent.handoffs if t.tool_name == request.name), None)
            return mode, bool(self.transfer)
        if isinstance(request, core.ParseConfiguredChildren):
            return create_sub_task(replace(context, sub_task_runner=request.collect), request.arguments)
        if isinstance(request, core.ParentInputMetadata):
            if context.ctx is not None:
                context.ctx.metadata.setdefault("_vv_agent_input", request.user_prompt)
        elif isinstance(request, core.ChildPrompt):
            return child_agent_prompt(arguments=request.arguments, context=context)
        elif request is core.ChildRead.MAX_HANDOFFS:
            return runtime.config.max_handoffs
        elif isinstance(request, core.SelectChildAgent):
            self.agent = target_agent(runtime, request.selector, request.mode, request.sub)
        elif request is core.ChildRead.AGENT_NAME:
            return self.agent.name
        elif isinstance(request, core.CompileChild):
            config = child_config(runtime, context, self.agent, request.mode, request.sub)
            if request.request.exclude_files_pattern:
                config = replace(
                    config,
                    workspace_backend=DiscoveryFilteredWorkspaceBackend(
                        context.workspace_backend, request.request.exclude_files_pattern
                    ),
                )
            self.child_runtime = runtime.for_agent(self.agent, config)
            self.config = config
            return self.child_runtime.compile(request.request.task_description, request.turn_id)
        elif request is core.ChildRead.WORKSPACE:
            return str(context.workspace)
        elif isinstance(request, core.ChildPromptBundle):
            task, sub = request.task, request.sub
            if sub.system_prompt:
                return build_raw_system_prompt_bundle(sub.system_prompt)
            return build_system_prompt_bundle(
                sub.description,
                language=str(task.metadata.get("language", "zh-CN")),
                allow_interruption=False,
                use_workspace=task.use_workspace,
                enable_todo_management=True,
                agent_type=task.agent_type,
                available_skills=task.metadata.get("available_skills"),
                workspace=context.workspace,
            )
        elif request is core.ChildRead.TODO_LIST:
            return context.shared_state.get("todo_list")
        elif isinstance(request, core.ChildDefinition):
            return DefinitionFacts(
                self.child_runtime._definitions,
                self.child_runtime._definition_bindings(request.task),
                self.child_runtime.registry,
            )
        elif request is core.ChildRead.BUDGET_LIMITS:
            return self.config.budget_limits
        elif request is core.ChildRead.HANDLER_VERSION:
            return runtime.handler_version
        else:
            assert request is core.ChildRead.TRANSFER_METADATA
            assert self.transfer is not None
            return self.transfer.metadata
        return None

    def run[T](self, steps: Generator[core.ChildRequest, Any, T]) -> T:
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
            steps.close()
