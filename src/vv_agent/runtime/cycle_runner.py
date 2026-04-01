from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from vv_agent.llm.base import LLMClient
from vv_agent.memory import CompactionExhaustedError, MemoryManager
from vv_agent.runtime.hooks import RuntimeHookManager
from vv_agent.runtime.token_usage import normalize_token_usage
from vv_agent.runtime.tool_planner import plan_tool_schemas
from vv_agent.tools import ToolRegistry
from vv_agent.types import AgentTask, CycleRecord, Message, ToolCall

if TYPE_CHECKING:
    from vv_agent.runtime.context import ExecutionContext

MAX_PTL_RETRIES = 3
_PTL_ERROR_PATTERNS = (
    "prompt is too long",
    "prompt_too_long",
    "context_length_exceeded",
    "maximum context length",
    "request too large",
    "too many tokens",
)


class CycleRunner:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        hook_manager: RuntimeHookManager | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.hook_manager = hook_manager or RuntimeHookManager()

    def run_cycle(
        self,
        *,
        task: AgentTask,
        messages: list[Message],
        cycle_index: int,
        memory_manager: MemoryManager,
        previous_prompt_tokens: int | None = None,
        recent_tool_call_ids: set[str] | None = None,
        shared_state: dict[str, Any] | None = None,
        ctx: ExecutionContext | None = None,
    ) -> tuple[list[Message], CycleRecord]:
        shared = shared_state or {}
        if ctx is not None:
            ctx.check_cancelled()
        pre_compact_messages = self.hook_manager.apply_before_memory_compact(
            task=task,
            cycle_index=cycle_index,
            messages=messages,
            shared_state=shared,
        )
        pre_compact_messages = memory_manager.apply_session_memory_context(pre_compact_messages)
        preemptively_microcompacted = False
        compact_total_tokens = previous_prompt_tokens
        if memory_manager.should_preemptive_microcompact(
            pre_compact_messages,
            total_tokens=previous_prompt_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
        ):
            pre_compact_messages, cleared = memory_manager.microcompact_messages(
                pre_compact_messages,
                cycle_index=cycle_index,
            )
            if cleared > 0:
                preemptively_microcompacted = True
                compact_total_tokens = None
        compacted_messages, memory_compacted = memory_manager.compact(
            pre_compact_messages,
            cycle_index=cycle_index,
            total_tokens=compact_total_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
        )
        memory_compacted = memory_compacted or preemptively_microcompacted
        ptl_retries = 0
        request_messages = compacted_messages
        request_tool_schemas: list[dict[str, Any]] = []
        llm_response = None

        while True:
            compacted_messages = memory_manager.apply_session_memory_context(compacted_messages)
            memory_usage_percentage = memory_manager.estimate_memory_usage_percentage(compacted_messages)
            tool_schemas = plan_tool_schemas(
                registry=self.tool_registry,
                task=task,
                memory_usage_percentage=memory_usage_percentage,
            )
            request_messages, request_tool_schemas = self.hook_manager.apply_before_llm(
                task=task,
                cycle_index=cycle_index,
                messages=compacted_messages,
                tool_schemas=tool_schemas,
                shared_state=shared,
            )

            if ctx is not None:
                ctx.check_cancelled()

            stream_callback = ctx.stream_callback if ctx is not None else None
            try:
                llm_response = self.llm_client.complete(
                    model=task.model,
                    messages=request_messages,
                    tools=request_tool_schemas,
                    stream_callback=stream_callback,
                )
                break
            except Exception as exc:
                if not self._is_prompt_too_long_error(exc):
                    raise

                ptl_retries += 1
                if ptl_retries > MAX_PTL_RETRIES:
                    raise CompactionExhaustedError(ptl_retries, exc) from exc

                if ptl_retries == 1:
                    compacted_messages, _ = memory_manager.compact(
                        compacted_messages,
                        cycle_index=cycle_index,
                        total_tokens=None,
                        recent_tool_call_ids=recent_tool_call_ids,
                        force=True,
                    )
                else:
                    compacted_messages = memory_manager.emergency_compact(
                        compacted_messages,
                        cycle_index=cycle_index,
                        drop_ratio=min(0.2 * ptl_retries, 0.95),
                    )
                memory_compacted = True

        if ctx is not None:
            ctx.check_cancelled()
        llm_response = self.hook_manager.apply_after_llm(
            task=task,
            cycle_index=cycle_index,
            messages=request_messages,
            tool_schemas=request_tool_schemas,
            response=llm_response,
            shared_state=shared,
        )

        next_messages = list(request_messages)
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
            token_usage=normalize_token_usage(llm_response.raw.get("usage")),
        )
        return next_messages, cycle_record

    @staticmethod
    def _is_prompt_too_long_error(error: Exception) -> bool:
        visited: set[int] = set()
        stack: list[Any] = [error]
        while stack:
            current = stack.pop()
            identifier = id(current)
            if identifier in visited:
                continue
            visited.add(identifier)

            current_text = str(current).lower()
            if any(pattern in current_text for pattern in _PTL_ERROR_PATTERNS):
                return True

            cause = getattr(current, "__cause__", None)
            context = getattr(current, "__context__", None)
            if cause is not None:
                stack.append(cause)
            if context is not None:
                stack.append(context)
            args = getattr(current, "args", ())
            if isinstance(args, tuple):
                stack.extend(arg for arg in args if isinstance(arg, BaseException))
        return False

    @staticmethod
    def _serialize_tool_calls(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            payload = {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                },
            }
            if tool_call.extra_content is not None:
                payload["extra_content"] = tool_call.extra_content
            serialized.append(payload)
        return serialized
