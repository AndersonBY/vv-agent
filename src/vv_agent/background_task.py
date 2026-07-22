from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from vv_agent.run_config import RunConfig
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.outputs import ToolOutputJson
from vv_agent.types import AgentStatus

if TYPE_CHECKING:
    from vv_agent.agent import Agent
    from vv_agent.result import RunResult
    from vv_agent.run_handle import RunHandle
    from vv_agent.tools.base import ToolContext


@dataclass(frozen=True, slots=True)
class BackgroundAgentTaskSnapshot:
    task_id: str
    agent_name: str
    status: AgentStatus
    final_output: Any | None = None
    error: str | None = None

    @property
    def done(self) -> bool:
        return self.status not in {AgentStatus.PENDING, AgentStatus.RUNNING}

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "final_output": self.final_output,
            "error": self.error,
        }


class BackgroundAgentTaskHandle:
    def __init__(self, *, task_id: str, agent_name: str, run_handle: RunHandle) -> None:
        self.task_id = task_id
        self.agent_name = agent_name
        self._run_handle = run_handle

    @property
    def status(self) -> AgentStatus:
        return self.snapshot().status

    def poll(self) -> BackgroundAgentTaskSnapshot:
        return self.snapshot()

    def snapshot(self) -> BackgroundAgentTaskSnapshot:
        state = self._run_handle.state()
        if not state.done:
            return BackgroundAgentTaskSnapshot(
                task_id=self.task_id,
                agent_name=self.agent_name,
                status=AgentStatus.RUNNING,
            )

        try:
            result = self._run_handle.result(timeout=0)
        except TimeoutError:
            return BackgroundAgentTaskSnapshot(
                task_id=self.task_id,
                agent_name=self.agent_name,
                status=AgentStatus.RUNNING,
            )
        except BaseException as exc:
            return BackgroundAgentTaskSnapshot(
                task_id=self.task_id,
                agent_name=self.agent_name,
                status=AgentStatus.FAILED,
                error=str(exc),
            )
        return self._snapshot_from_result(result)

    def wait(self, timeout: float | None = None) -> BackgroundAgentTaskSnapshot:
        try:
            result = self._run_handle.result(timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError(f"Background agent task {self.task_id} was not ready before timeout.") from exc
        except BaseException as exc:
            return BackgroundAgentTaskSnapshot(
                task_id=self.task_id,
                agent_name=self.agent_name,
                status=AgentStatus.FAILED,
                error=str(exc),
            )
        return self._snapshot_from_result(result)

    def _snapshot_from_result(self, result: RunResult) -> BackgroundAgentTaskSnapshot:
        return BackgroundAgentTaskSnapshot(
            task_id=self.task_id,
            agent_name=self.agent_name,
            status=result.status,
            final_output=result.final_output,
            error=result.raw_result.error,
        )


class BackgroundAgentTask(FunctionTool):
    __slots__ = ("_handles", "_handles_lock", "agent")

    def __init__(
        self,
        agent: Agent,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        tool_name = name or f"{agent.name}_background_task"
        if not tool_name.strip():
            raise ValueError("background task tool name cannot be empty")
        self.agent = agent
        self._handles: dict[str, BackgroundAgentTaskHandle] = {}
        self._handles_lock = threading.Lock()
        super().__init__(
            name=tool_name,
            description=description or f"Start the {agent.name} agent as a background task.",
            params_json_schema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Task for the background agent.",
                    }
                },
                "required": ["task_description"],
                "additionalProperties": False,
            },
            on_invoke=self._requested_output,
            metadata={"agent": agent, "mode": "background_task"},
        )

    def start(
        self,
        runner: Any,
        context: ToolContext | None,
        raw_arguments: dict[str, Any],
        *,
        run_config: RunConfig | None = None,
    ) -> BackgroundAgentTaskHandle:
        task_description = self._input_from_arguments(raw_arguments)
        config = self._inherited_run_config(context, run_config)
        run_handle = runner.start(self.agent, task_description, run_config=config)
        handle = BackgroundAgentTaskHandle(
            task_id=f"bg_agent_{uuid.uuid4().hex[:12]}",
            agent_name=self.agent.name,
            run_handle=run_handle,
        )
        with self._handles_lock:
            self._handles[handle.task_id] = handle
        return handle

    def get_handle(self, task_id: str) -> BackgroundAgentTaskHandle:
        with self._handles_lock:
            handle = self._handles.get(task_id)
        if handle is None:
            raise KeyError(f"Unknown background agent task: {task_id}")
        return handle

    def _requested_output(self, _context: ToolContext | None, arguments: dict[str, Any]) -> ToolOutputJson:
        return ToolOutputJson(
            data={
                "agent_name": self.agent.name,
                "status": "background_task_requested",
                "task_description": self._input_from_arguments(arguments),
            }
        )

    @staticmethod
    def _inherited_run_config(context: ToolContext | None, run_config: RunConfig | None) -> RunConfig:
        config = run_config or RunConfig()
        if context is None:
            return config

        runtime_metadata = context.ctx.metadata if context.ctx is not None else {}
        cancellation_token = config.cancellation_token
        if cancellation_token is None and context.ctx is not None and context.ctx.cancellation_token is not None:
            cancellation_token = context.ctx.cancellation_token.child()

        shared_state = {**context.shared_state, **(config.shared_state or {})}
        metadata = dict(context.task_metadata or context.metadata)
        for key in (
            "agent_name",
            "session_id",
            "approved_tool_interruption_ids",
            "_vv_agent_run_id",
            "_vv_agent_trace_id",
            "_vv_agent_agent_name",
            "_vv_agent_input",
            "_vv_agent_session_id",
            "_vv_agent_tool_use_behavior",
            "_vv_agent_stop_at_tool_names",
        ):
            metadata.pop(key, None)
        metadata.update(config.metadata)

        parent_context = getattr(context.run_context, "context", None)
        return replace(
            config,
            workspace=config.workspace if config.workspace is not None else context.workspace,
            workspace_backend=(config.workspace_backend if config.workspace_backend is not None else context.workspace_backend),
            model_provider=(
                config.model_provider if config.model_provider is not None else runtime_metadata.get("_vv_agent_model_provider")
            ),
            execution_backend=(
                config.execution_backend if config.execution_backend is not None else runtime_metadata.get("execution_backend")
            ),
            context=config.context if config.context is not None else parent_context,
            cancellation_token=cancellation_token,
            shared_state=shared_state,
            metadata=metadata,
        )

    @staticmethod
    def _input_from_arguments(raw_arguments: dict[str, Any]) -> str:
        if not isinstance(raw_arguments, dict):
            raise ValueError("background task arguments must be an object")
        value = raw_arguments.get("task_description")
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("background task requires task_description")


__all__ = [
    "BackgroundAgentTask",
    "BackgroundAgentTaskHandle",
    "BackgroundAgentTaskSnapshot",
]
