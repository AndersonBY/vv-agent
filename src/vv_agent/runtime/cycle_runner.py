from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vv_agent.events import RunEvent
from vv_agent.llm.base import LLMClient
from vv_agent.memory import CompactionExhaustedError, MemoryManager
from vv_agent.memory.provider import MemoryCompactCompleted, MemoryCompactStarted, MemoryProvider, MemoryProviderResult
from vv_agent.memory.token_utils import count_messages_tokens
from vv_agent.model_settings import ModelSettings
from vv_agent.runtime.hooks import RuntimeHookManager
from vv_agent.runtime.token_usage import normalize_token_usage
from vv_agent.runtime.tool_planner import plan_tool_schemas
from vv_agent.tools import ToolRegistry
from vv_agent.types import AgentTask, CycleRecord, LLMResponse, Message, ToolCall

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
        estimated_compact_tokens = self._estimate_compact_tokens(
            pre_compact_messages,
            memory_manager=memory_manager,
            total_tokens=previous_prompt_tokens,
        )
        should_preemptive_microcompact = memory_manager.should_preemptive_microcompact(
            pre_compact_messages,
            total_tokens=previous_prompt_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
        )
        compact_lifecycle_started = False
        if should_preemptive_microcompact or self._should_full_compact(
            estimated_compact_tokens,
            memory_manager=memory_manager,
        ):
            self._emit_memory_compact_started(
                ctx=ctx,
                cycle_index=cycle_index,
                messages=pre_compact_messages,
                estimated_tokens=estimated_compact_tokens,
            )
            compact_lifecycle_started = True
        if should_preemptive_microcompact:
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
        if compact_lifecycle_started:
            self._emit_memory_compact_completed(
                ctx=ctx,
                cycle_index=cycle_index,
                before_messages=pre_compact_messages,
                after_messages=compacted_messages,
                memory_manager=memory_manager,
            )
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
                llm_response = self._complete_llm(
                    model=task.model,
                    messages=request_messages,
                    tools=request_tool_schemas,
                    stream_callback=stream_callback,
                    model_settings=self._model_settings_from_context(ctx),
                )
                break
            except Exception as exc:
                if not self._is_prompt_too_long_error(exc):
                    raise

                ptl_retries += 1
                if ptl_retries > MAX_PTL_RETRIES:
                    raise CompactionExhaustedError(ptl_retries, exc) from exc

                if ptl_retries == 1:
                    before_retry_compact = compacted_messages
                    self._emit_memory_compact_started(
                        ctx=ctx,
                        cycle_index=cycle_index,
                        messages=before_retry_compact,
                        estimated_tokens=self._estimate_compact_tokens(
                            before_retry_compact,
                            memory_manager=memory_manager,
                            total_tokens=None,
                        ),
                    )
                    compacted_messages, _ = memory_manager.compact(
                        compacted_messages,
                        cycle_index=cycle_index,
                        total_tokens=None,
                        recent_tool_call_ids=recent_tool_call_ids,
                        force=True,
                    )
                    self._emit_memory_compact_completed(
                        ctx=ctx,
                        cycle_index=cycle_index,
                        before_messages=before_retry_compact,
                        after_messages=compacted_messages,
                        memory_manager=memory_manager,
                    )
                else:
                    before_retry_compact = compacted_messages
                    self._emit_memory_compact_started(
                        ctx=ctx,
                        cycle_index=cycle_index,
                        messages=before_retry_compact,
                        estimated_tokens=self._estimate_compact_tokens(
                            before_retry_compact,
                            memory_manager=memory_manager,
                            total_tokens=None,
                        ),
                    )
                    compacted_messages = memory_manager.emergency_compact(
                        compacted_messages,
                        cycle_index=cycle_index,
                        drop_ratio=min(0.2 * ptl_retries, 0.95),
                    )
                    self._emit_memory_compact_completed(
                        ctx=ctx,
                        cycle_index=cycle_index,
                        before_messages=before_retry_compact,
                        after_messages=compacted_messages,
                        memory_manager=memory_manager,
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

    def _complete_llm(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Any,
        model_settings: ModelSettings | None,
    ) -> LLMResponse:
        return self.llm_client.complete(
            model=model,
            messages=messages,
            tools=tools,
            stream_callback=stream_callback,
            model_settings=model_settings,
        )

    @staticmethod
    def _model_settings_from_context(ctx: ExecutionContext | None) -> ModelSettings | None:
        metadata = getattr(ctx, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("_vv_agent_model_settings")
        return value if isinstance(value, ModelSettings) else None

    @staticmethod
    def _estimate_compact_tokens(
        messages: list[Message],
        *,
        memory_manager: MemoryManager,
        total_tokens: int | None,
    ) -> int | None:
        if isinstance(total_tokens, int) and total_tokens >= 0:
            return total_tokens
        try:
            payload = [message.to_openai_message() for message in messages]
            return count_messages_tokens(payload, model=memory_manager.model)
        except Exception:
            return None

    @staticmethod
    def _should_full_compact(
        estimated_tokens: int | None,
        *,
        memory_manager: MemoryManager,
    ) -> bool:
        return estimated_tokens is not None and estimated_tokens > memory_manager.autocompact_threshold

    def _emit_memory_compact_started(
        self,
        *,
        ctx: ExecutionContext | None,
        cycle_index: int,
        messages: list[Message],
        estimated_tokens: int | None,
    ) -> None:
        providers = self._memory_providers_from_context(ctx)
        emit_event = self._event_emitter_from_context(ctx)
        if not providers and emit_event is None:
            return
        event = MemoryCompactStarted(
            **self._memory_event_context(ctx),
            cycle_index=cycle_index,
            message_count=len(messages),
            estimated_tokens=estimated_tokens,
        )
        metadata = self._call_before_memory_providers(providers, event)
        if metadata:
            event = MemoryCompactStarted(
                **self._memory_event_context(ctx),
                cycle_index=cycle_index,
                message_count=len(messages),
                estimated_tokens=estimated_tokens,
                event_id=event.event_id,
                created_at=event.created_at,
                metadata=metadata,
            )
        if emit_event is not None:
            emit_event(event)

    def _emit_memory_compact_completed(
        self,
        *,
        ctx: ExecutionContext | None,
        cycle_index: int,
        before_messages: list[Message],
        after_messages: list[Message],
        memory_manager: MemoryManager,
    ) -> None:
        providers = self._memory_providers_from_context(ctx)
        emit_event = self._event_emitter_from_context(ctx)
        if not providers and emit_event is None:
            return
        event = MemoryCompactCompleted(
            **self._memory_event_context(ctx),
            cycle_index=cycle_index,
            before_count=len(before_messages),
            after_count=len(after_messages),
            summary_tokens=self._estimate_compact_tokens(
                after_messages,
                memory_manager=memory_manager,
                total_tokens=None,
            ),
        )
        metadata = self._call_after_memory_providers(providers, event)
        if metadata:
            event = MemoryCompactCompleted(
                **self._memory_event_context(ctx),
                cycle_index=cycle_index,
                before_count=len(before_messages),
                after_count=len(after_messages),
                summary_tokens=event.summary_tokens,
                event_id=event.event_id,
                created_at=event.created_at,
                metadata=metadata,
            )
        if emit_event is not None:
            emit_event(event)

    def _call_before_memory_providers(
        self,
        providers: list[MemoryProvider],
        event: MemoryCompactStarted,
    ) -> dict[str, Any]:
        results: dict[str, dict[str, Any]] = {}
        errors: list[dict[str, str]] = []
        for index, provider in enumerate(providers):
            provider_name = self._memory_provider_name(provider, index=index, existing=results)
            try:
                result = provider.before_compact(event)
            except Exception as exc:
                self._record_memory_provider_error(
                    provider_name=provider_name,
                    stage="before_compact",
                    error=exc,
                    errors=errors,
                )
                continue
            if isinstance(result, MemoryProviderResult) and result.metadata:
                results[provider_name] = dict(result.metadata)
        return self._memory_provider_metadata(results=results, errors=errors)

    def _call_after_memory_providers(
        self,
        providers: list[MemoryProvider],
        event: MemoryCompactCompleted,
    ) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        for index, provider in enumerate(providers):
            provider_name = self._memory_provider_name(provider, index=index, existing={})
            try:
                provider.after_compact(event)
            except Exception as exc:
                self._record_memory_provider_error(
                    provider_name=provider_name,
                    stage="after_compact",
                    error=exc,
                    errors=errors,
                )
        return self._memory_provider_metadata(results={}, errors=errors)

    @staticmethod
    def _record_memory_provider_error(
        *,
        provider_name: str,
        stage: str,
        error: Exception,
        errors: list[dict[str, str]],
    ) -> None:
        warnings.warn(
            f"Memory provider {provider_name} {stage} failed: {error}",
            RuntimeWarning,
            stacklevel=3,
        )
        errors.append(
            {
                "provider": provider_name,
                "stage": stage,
                "error": str(error),
                "error_type": type(error).__name__,
            }
        )

    @staticmethod
    def _memory_provider_metadata(
        *,
        results: dict[str, dict[str, Any]],
        errors: list[dict[str, str]],
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if results:
            metadata["memory_provider_results"] = results
        if errors:
            metadata["memory_provider_errors"] = errors
        return metadata

    @staticmethod
    def _memory_provider_name(
        provider: MemoryProvider,
        *,
        index: int,
        existing: dict[str, Any],
    ) -> str:
        base_name = provider.__class__.__name__
        if base_name not in existing:
            return base_name
        return f"{base_name}#{index + 1}"

    @staticmethod
    def _memory_providers_from_context(ctx: ExecutionContext | None) -> list[MemoryProvider]:
        metadata = getattr(ctx, "metadata", None)
        if not isinstance(metadata, dict):
            return []
        providers = metadata.get("_vv_agent_memory_providers")
        if providers is None:
            return []
        if isinstance(providers, list | tuple):
            return list(providers)
        return []

    @staticmethod
    def _event_emitter_from_context(ctx: ExecutionContext | None) -> Callable[[RunEvent], None] | None:
        metadata = getattr(ctx, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        emitter = metadata.get("_vv_agent_emit_event")
        return emitter if callable(emitter) else None

    @staticmethod
    def _memory_event_context(ctx: ExecutionContext | None) -> dict[str, Any]:
        metadata = getattr(ctx, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "run_id": str(metadata.get("_vv_agent_run_id") or ""),
            "trace_id": str(metadata.get("_vv_agent_trace_id") or metadata.get("trace_id") or ""),
            "agent_name": metadata.get("_vv_agent_agent_name"),
            "session_id": metadata.get("_vv_agent_session_id"),
        }

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
