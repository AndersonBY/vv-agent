from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v_agent.constants import TASK_FINISH_TOOL_NAME
from v_agent.llm.base import LLMClient
from v_agent.memory import MemoryManager
from v_agent.runtime.tool_planner import plan_tool_schemas
from v_agent.tools import ToolContext, ToolRegistry
from v_agent.tools.dispatcher import dispatch_tool_call
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
                messages, memory_compacted = memory_manager.compact(messages)
                memory_usage_percentage = self._estimate_memory_usage_percentage(messages, task.memory_threshold_chars)
                tool_schemas = plan_tool_schemas(
                    registry=self.tool_registry,
                    task=task,
                    memory_usage_percentage=memory_usage_percentage,
                )
                llm_response = self.llm_client.complete(
                    model=task.model,
                    messages=messages,
                    tools=tool_schemas,
                )
            except Exception as exc:
                return AgentResult(
                    status=AgentStatus.FAILED,
                    messages=messages,
                    cycles=cycles,
                    error=f"LLM call failed in cycle {cycle_index}: {exc}",
                    shared_state=shared,
                )

            messages.append(Message(role="assistant", content=llm_response.content))
            cycle_record = CycleRecord(
                index=cycle_index,
                assistant_message=llm_response.content,
                tool_calls=llm_response.tool_calls,
                memory_compacted=memory_compacted,
            )

            if llm_response.tool_calls:
                context = ToolContext(workspace=workspace_path, shared_state=shared, cycle_index=cycle_index)
                tool_result = self._run_tools(llm_response.tool_calls, context, messages, cycle_record)
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
                    final_answer=llm_response.content,
                    shared_state=shared,
                )

            if task.no_tool_policy == "wait_user":
                return AgentResult(
                    status=AgentStatus.WAIT_USER,
                    messages=messages,
                    cycles=cycles,
                    wait_reason=llm_response.content or "No tool call and runtime is waiting for user.",
                    shared_state=shared,
                )

            if cycle_index < task.max_cycles:
                messages.append(
                    Message(
                        role="user",
                        content=(
                            "No tool call was produced. "
                            f"Continue the task and call `{TASK_FINISH_TOOL_NAME}` "
                            "when all todo items are done."
                        ),
                    )
                )

        return AgentResult(
            status=AgentStatus.MAX_CYCLES,
            messages=messages,
            cycles=cycles,
            final_answer="Reached max cycles without finish signal.",
            shared_state=shared,
        )

    def _run_tools(
        self,
        tool_calls,
        context: ToolContext,
        messages: list[Message],
        cycle_record: CycleRecord,
    ) -> ToolExecutionResult | None:
        latest_directive_result: ToolExecutionResult | None = None

        for call in tool_calls:
            result = dispatch_tool_call(
                registry=self.tool_registry,
                context=context,
                call=call,
            )

            cycle_record.tool_results.append(result)
            messages.append(result.to_tool_message())

            if result.directive in (ToolDirective.WAIT_USER, ToolDirective.FINISH):
                latest_directive_result = result
                break

        return latest_directive_result

    def _prepare_workspace(self, workspace: str | Path | None) -> Path:
        target = Path(workspace) if workspace else self.default_workspace
        if target is None:
            target = Path.cwd() / ".v-agent-workspace"
        target = target.resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _estimate_memory_usage_percentage(messages: list[Message], threshold_chars: int) -> int:
        if threshold_chars <= 0:
            return 0
        used_chars = sum(len(message.content) for message in messages)
        return int((used_chars / threshold_chars) * 100)

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
