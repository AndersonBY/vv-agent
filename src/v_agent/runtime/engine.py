from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v_agent.constants import TASK_FINISH_TOOL_NAME
from v_agent.llm.base import LLMClient
from v_agent.memory import MemoryManager
from v_agent.runtime.cycle_runner import CycleRunner
from v_agent.runtime.tool_call_runner import ToolCallRunner
from v_agent.tools import ToolContext, ToolRegistry
from v_agent.types import AgentResult, AgentStatus, AgentTask, CycleRecord, Message, ToolDirective, ToolExecutionResult


class AgentRuntime:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        default_workspace: str | Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.default_workspace = Path(default_workspace).resolve() if default_workspace else None
        self.cycle_runner = CycleRunner(llm_client=llm_client, tool_registry=tool_registry)
        self.tool_call_runner = ToolCallRunner(tool_registry=tool_registry)

    def run(
        self,
        task: AgentTask,
        *,
        workspace: str | Path | None = None,
        shared_state: dict[str, Any] | None = None,
    ) -> AgentResult:
        workspace_path = self._prepare_workspace(workspace)
        shared = dict(shared_state or {})
        shared.setdefault("todo_list", [])

        messages: list[Message] = [
            Message(role="system", content=task.system_prompt),
            Message(role="user", content=task.user_prompt),
        ]
        cycles: list[CycleRecord] = []

        memory_manager = MemoryManager(threshold_chars=task.memory_threshold_chars)

        for cycle_index in range(1, task.max_cycles + 1):
            try:
                messages, cycle_record = self.cycle_runner.run_cycle(
                    task=task,
                    messages=messages,
                    cycle_index=cycle_index,
                    memory_manager=memory_manager,
                )
            except Exception as exc:
                return AgentResult(
                    status=AgentStatus.FAILED,
                    messages=messages,
                    cycles=cycles,
                    error=f"LLM call failed in cycle {cycle_index}: {exc}",
                    shared_state=shared,
                )

            if cycle_record.tool_calls:
                context = ToolContext(workspace=workspace_path, shared_state=shared, cycle_index=cycle_index)
                tool_result = self.tool_call_runner.run(
                    tool_calls=cycle_record.tool_calls,
                    context=context,
                    messages=messages,
                    cycle_record=cycle_record,
                )
                cycles.append(cycle_record)

                if tool_result and tool_result.directive == ToolDirective.WAIT_USER:
                    wait_reason = tool_result.metadata.get("question") if isinstance(tool_result.metadata, dict) else None
                    if not wait_reason:
                        wait_reason = tool_result.content
                    return AgentResult(
                        status=AgentStatus.WAIT_USER,
                        messages=messages,
                        cycles=cycles,
                        wait_reason=str(wait_reason),
                        shared_state=shared,
                    )

                if tool_result and tool_result.directive == ToolDirective.FINISH:
                    final_answer = self._extract_final_message(tool_result)
                    return AgentResult(
                        status=AgentStatus.COMPLETED,
                        messages=messages,
                        cycles=cycles,
                        final_answer=final_answer,
                        shared_state=shared,
                    )

                continue

            cycles.append(cycle_record)
            if task.no_tool_policy == "finish":
                return AgentResult(
                    status=AgentStatus.COMPLETED,
                    messages=messages,
                    cycles=cycles,
                    final_answer=cycle_record.assistant_message,
                    shared_state=shared,
                )

            if task.no_tool_policy == "wait_user":
                return AgentResult(
                    status=AgentStatus.WAIT_USER,
                    messages=messages,
                    cycles=cycles,
                    wait_reason=cycle_record.assistant_message or "No tool call and runtime is waiting for user.",
                    shared_state=shared,
                )

            if cycle_index < task.max_cycles:
                messages.append(Message(role="user", content=self._build_continue_hint()))

        return AgentResult(
            status=AgentStatus.MAX_CYCLES,
            messages=messages,
            cycles=cycles,
            final_answer="Reached max cycles without finish signal.",
            shared_state=shared,
        )

    def _prepare_workspace(self, workspace: str | Path | None) -> Path:
        target = Path(workspace) if workspace else self.default_workspace
        if target is None:
            target = Path.cwd() / ".v-agent-workspace"
        target = target.resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _build_continue_hint() -> str:
        return (
            "No tool call was produced. "
            f"Continue the task and call `{TASK_FINISH_TOOL_NAME}` "
            "when all todo items are done."
        )

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
