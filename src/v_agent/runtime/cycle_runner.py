from __future__ import annotations

import json
from typing import Any

from v_agent.llm.base import LLMClient
from v_agent.memory import MemoryManager
from v_agent.runtime.tool_planner import plan_tool_schemas
from v_agent.tools import ToolRegistry
from v_agent.types import AgentTask, CycleRecord, Message, ToolCall


class CycleRunner:
    def __init__(self, *, llm_client: LLMClient, tool_registry: ToolRegistry) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry

    def run_cycle(
        self,
        *,
        task: AgentTask,
        messages: list[Message],
        cycle_index: int,
        memory_manager: MemoryManager,
    ) -> tuple[list[Message], CycleRecord]:
        compacted_messages, memory_compacted = memory_manager.compact(messages)
        memory_usage_percentage = self._estimate_memory_usage_percentage(compacted_messages, task.memory_threshold_chars)
        tool_schemas = plan_tool_schemas(
            registry=self.tool_registry,
            task=task,
            memory_usage_percentage=memory_usage_percentage,
        )

        llm_response = self.llm_client.complete(
            model=task.model,
            messages=compacted_messages,
            tools=tool_schemas,
        )

        next_messages = list(compacted_messages)
        serialized_tool_calls = self._serialize_tool_calls(llm_response.tool_calls)
        raw_reasoning = llm_response.raw.get("reasoning_content")
        reasoning_content = raw_reasoning if isinstance(raw_reasoning, str) and raw_reasoning else None
        next_messages.append(
            Message(
                role="assistant",
                content=llm_response.content,
                tool_calls=serialized_tool_calls or None,
                reasoning_content=reasoning_content,
            )
        )

        cycle_record = CycleRecord(
            index=cycle_index,
            assistant_message=llm_response.content,
            tool_calls=llm_response.tool_calls,
            memory_compacted=memory_compacted,
        )
        return next_messages, cycle_record

    @staticmethod
    def _estimate_memory_usage_percentage(messages: list[Message], threshold_chars: int) -> int:
        if threshold_chars <= 0:
            return 0
        used_chars = sum(len(message.content) for message in messages)
        return int((used_chars / threshold_chars) * 100)

    @staticmethod
    def _serialize_tool_calls(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            serialized.append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                    },
                }
            )
        return serialized
