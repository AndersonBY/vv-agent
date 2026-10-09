"""Task-oriented test assembly for the sole session driver."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from vv_agent import Agent, RunConfig, Runner, ToolPolicy
from vv_agent.model import ScriptedModelProvider
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.engine import AgentRuntime as _LegacyAssembly
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import AgentResult, AgentTask


def start_runner(driver: SessionDriver, session_id: str, agent: Agent, content: str, *, run_config=None):
    with (
        patch("vv_agent.session.surfaces.SessionDriver", return_value=driver),
        patch("vv_agent.runner.uuid.uuid4", return_value=SimpleNamespace(hex=session_id)),
    ):
        return Runner.start(agent, content, run_config=run_config)


class KernelRuntime(_LegacyAssembly):
    # Pure memory/sub-agent construction helpers remain until F3b extracts them.
    def run(self, task: AgentTask, **kwargs: Any) -> AgentResult:
        if kwargs.get("checkpoint_controller") is not None or kwargs.get("sub_task_manager") is not None:
            raise ValueError("kernel tests use session records and child admission")
        context: ExecutionContext = kwargs.get("ctx") or ExecutionContext()
        workspace = Path(kwargs.get("workspace") or self.default_workspace or ".").resolve()
        provider = self.model_provider or ScriptedModelProvider(backend="test", default_model=task.model, llm=self.llm_client)
        callbacks = [cb for cb in (self.event_handler, context.event_handler) if cb is not None]

        def stream(event):
            for callback in callbacks:
                callback(event)

        allowed = task.metadata.get("_vv_agent_allowed_tools")
        policy = ToolPolicy(
            allowed_tools=allowed,
            disallowed_tools=task.metadata.get("_vv_agent_disallowed_tools", []),
            can_use_tool=context.metadata.get("_vv_agent_tool_policy_can_use_tool"),
        )
        config = RunConfig(
            model=task.model,
            model_provider=provider,
            model_settings=task.model_settings,
            workspace=workspace,
            workspace_backend=self._workspace_backend,
            tool_registry_factory=lambda: self.tool_registry,
            hooks=list(self.hook_manager.hooks),
            after_cycle_hooks=list(self.after_cycle_hook_manager.hooks),
            cancellation_token=context.cancellation_token,
            stream=stream,
            log_preview_chars=self.log_preview_chars,
            tool_policy=policy,
            before_cycle_messages=kwargs.get("before_cycle_messages"),
            interruption_messages=kwargs.get("interruption_messages"),
            budget_limits=kwargs.get("budget_limits"),
            host_cost_meter=kwargs.get("host_cost_meter"),
            approval_provider=context.metadata.get("_vv_agent_approval_provider"),
            approval_broker=context.metadata.get("_vv_agent_approval_broker"),
        )
        messages = kwargs.get("prepared_initial_messages") or kwargs.get("initial_messages") or task.initial_messages or []
        seed = {
            "messages": [m.to_dict() for m in messages],
            "shared_state": kwargs.get("shared_state") or task.initial_shared_state or {},
        }
        task = replace(task, initial_messages=None, initial_shared_state=None)
        agent = Agent(task.metadata.get("agent_name", "test"), task.prompt_bundle, model=task.model, sub_agents=task.sub_agents)
        driver = SessionDriver()
        try:
            sid = task.task_id
            task.metadata["session_id"] = sid
            if task.metadata.get("session_memory_enabled"):
                from vv_agent.session.memory import session_memory

                memory = session_memory(task, workspace if task.use_workspace else None)
                memory.load()
                task.metadata.setdefault("vv_session", {})["memory_initial_state"] = memory.state.to_dict()
            driver.create(sid, str(workspace), {"seed": seed})
            handle = driver.start(sid, agent, config, kwargs.get("user_message") or task.user_prompt, task=task, autostart=False)
            handle.runtime.memory_manager = self._build_memory_manager(
                task=task,
                workspace_path=workspace,
                workspace_backend=self._workspace_backend,
                ctx=context,
            )
            handle.start()
            return handle.result().raw_result
        finally:
            driver.close()
