from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from vv_agent.events import MemoryCompactTrigger, RunEvent, _project_provider_stream_payload
from vv_agent.llm.base import LLMClient, LlmRequest, complete_llm_request
from vv_agent.memory import CompactionExhaustedError, MemoryManager
from vv_agent.memory.manager import CompactionMode
from vv_agent.memory.provider import MemoryCompactCompleted, MemoryCompactStarted, MemoryProvider, MemoryProviderResult
from vv_agent.memory.token_utils import count_messages_tokens
from vv_agent.model_settings import ModelSettings
from vv_agent.runtime.checkpoint_resume import CheckpointResumeController
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
        lifecycle_before_messages = pre_compact_messages
        preemptively_microcompacted = False
        preemptive_mode: CompactionMode = "none"
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
        should_full_compact = self._should_full_compact(
            estimated_compact_tokens,
            memory_manager=memory_manager,
        )
        if should_preemptive_microcompact or should_full_compact:
            self._emit_memory_compact_started(
                ctx=ctx,
                cycle_index=cycle_index,
                messages=pre_compact_messages,
                estimated_tokens=estimated_compact_tokens,
                trigger="full_threshold" if should_full_compact else "micro_threshold",
                memory_manager=memory_manager,
            )
            compact_lifecycle_started = True
        if should_preemptive_microcompact:
            pre_compact_messages, cleared = memory_manager.microcompact_messages(
                pre_compact_messages,
                cycle_index=cycle_index,
            )
            if cleared > 0:
                preemptively_microcompacted = True
                preemptive_mode = "micro"
                compact_total_tokens = None
        compaction_result = memory_manager.compact_with_result(
            pre_compact_messages,
            cycle_index=cycle_index,
            total_tokens=compact_total_tokens,
            recent_tool_call_ids=recent_tool_call_ids,
        )
        compacted_messages = compaction_result.messages
        memory_compacted = compaction_result.changed or preemptively_microcompacted
        if compact_lifecycle_started:
            changed = lifecycle_before_messages != compacted_messages
            mode = self._strongest_compaction_mode(preemptive_mode, compaction_result.mode)
            if not changed:
                mode = "none"
            self._emit_memory_compact_completed(
                ctx=ctx,
                cycle_index=cycle_index,
                before_messages=lifecycle_before_messages,
                after_messages=compacted_messages,
                memory_manager=memory_manager,
                mode=mode,
                changed=changed,
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

            stream_callback = None
            if ctx is not None and ctx.event_handler is not None:
                metadata = ctx.metadata
                run_id = str(metadata.get("_vv_agent_run_id") or task.task_id)
                trace_id = str(metadata.get("_vv_agent_trace_id") or metadata.get("trace_id") or run_id)
                agent_name = str(metadata.get("_vv_agent_agent_name") or task.metadata.get("agent_name") or task.task_id)
                session_id = self._optional_identity(metadata.get("_vv_agent_session_id"))
                parent_run_id = self._optional_identity(metadata.get("_vv_agent_parent_run_id"))
                event_handler = ctx.event_handler

                def stream_callback(
                    payload: dict[str, Any],
                    _cycle_index: int = cycle_index,
                    _run_id: str = run_id,
                    _trace_id: str = trace_id,
                    _agent_name: str = agent_name,
                    _session_id: str | None = session_id,
                    _parent_run_id: str | None = parent_run_id,
                    _event_handler: Callable[[RunEvent], None] = event_handler,
                ) -> None:
                    event = _project_provider_stream_payload(
                        {**payload, "cycle": _cycle_index},
                        run_id=_run_id,
                        trace_id=_trace_id,
                        agent_name=_agent_name,
                        session_id=_session_id,
                        parent_run_id=_parent_run_id,
                    )
                    if event is not None:
                        _event_handler(event)

            try:
                llm_response = self._complete_llm(
                    cycle_index=cycle_index,
                    operation_slot=f"main:{ptl_retries + 1}",
                    model=task.model,
                    messages=request_messages,
                    tools=request_tool_schemas,
                    metadata=dict(task.metadata),
                    stream_callback=stream_callback,
                    model_settings=self._effective_model_settings(task, ctx),
                    ctx=ctx,
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
                        trigger="prompt_too_long",
                        memory_manager=memory_manager,
                    )
                    forced_result = memory_manager.compact_with_result(
                        compacted_messages,
                        cycle_index=cycle_index,
                        total_tokens=None,
                        recent_tool_call_ids=recent_tool_call_ids,
                        force=True,
                    )
                    compacted_messages = forced_result.messages
                    self._emit_memory_compact_completed(
                        ctx=ctx,
                        cycle_index=cycle_index,
                        before_messages=before_retry_compact,
                        after_messages=compacted_messages,
                        memory_manager=memory_manager,
                        mode=forced_result.mode,
                        changed=before_retry_compact != compacted_messages,
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
                        trigger="prompt_too_long",
                        memory_manager=memory_manager,
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
                        mode=("emergency" if before_retry_compact != compacted_messages else "none"),
                        changed=before_retry_compact != compacted_messages,
                    )
                memory_compacted = True

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
            tool_calls=deepcopy(llm_response.tool_calls),
            memory_compacted=memory_compacted,
            token_usage=normalize_token_usage(
                llm_response.raw.get("usage"),
                usage_source=llm_response.raw.get("usage_source"),
                cache_status=llm_response.raw.get("cache_status"),
            ),
            _planned_tool_names=tuple(
                name
                for schema in request_tool_schemas
                if isinstance(schema, dict)
                for function in [schema.get("function")]
                if isinstance(function, dict)
                for name in [function.get("name")]
                if isinstance(name, str)
            ),
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
        cycle_index: int,
        operation_slot: str,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        metadata: dict[str, Any],
        stream_callback: Any,
        model_settings: ModelSettings | None,
        ctx: ExecutionContext | None,
    ) -> LLMResponse:
        request = LlmRequest(
            model=model,
            messages=messages,
            tools=tools,
            metadata=metadata,
            model_settings=model_settings,
        )
        checkpoint_controller = ctx.metadata.get("_vv_agent_checkpoint_controller") if ctx is not None else None

        def invoke() -> LLMResponse:
            return complete_llm_request(
                self.llm_client,
                request,
                stream_callback=stream_callback,
            )

        if isinstance(checkpoint_controller, CheckpointResumeController):
            return checkpoint_controller.complete_model(
                cycle_index=cycle_index,
                operation_slot=operation_slot,
                request=request,
                invoke=invoke,
            )
        return invoke()

    @staticmethod
    def _optional_identity(value: Any) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    @staticmethod
    def _effective_model_settings(task: AgentTask, ctx: ExecutionContext | None) -> ModelSettings | None:
        metadata = getattr(ctx, "metadata", None)
        if isinstance(metadata, dict):
            value = metadata.get("_vv_agent_model_settings")
            if isinstance(value, ModelSettings):
                return value
        return task.model_settings

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

    @staticmethod
    def _strongest_compaction_mode(*modes: CompactionMode) -> CompactionMode:
        rank = {"none": 0, "micro": 1, "structural": 2, "summary": 3, "emergency": 4}
        return max(modes, key=rank.__getitem__, default="none")

    def _emit_memory_compact_started(
        self,
        *,
        ctx: ExecutionContext | None,
        cycle_index: int,
        messages: list[Message],
        estimated_tokens: int | None,
        trigger: MemoryCompactTrigger,
        memory_manager: MemoryManager,
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
            trigger=trigger,
            configured_threshold=memory_manager.compact_threshold,
            effective_threshold=memory_manager.autocompact_threshold,
            microcompact_threshold=memory_manager.microcompact_trigger_threshold,
            model_context_window=memory_manager.model_context_window,
            model_max_output_tokens=memory_manager.model_max_output_tokens,
            reserved_output_tokens=memory_manager.reserved_output_tokens,
            reserved_output_source=memory_manager.reserved_output_source,
            autocompact_buffer_tokens=memory_manager.autocompact_buffer_tokens,
        )
        provider_event = MemoryCompactStarted(
            **self._memory_event_context(ctx),
            cycle_index=cycle_index,
            message_count=len(messages),
            estimated_tokens=estimated_tokens,
            trigger=trigger,
            configured_threshold=memory_manager.compact_threshold,
            effective_threshold=memory_manager.autocompact_threshold,
            microcompact_threshold=memory_manager.microcompact_trigger_threshold,
            model_context_window=memory_manager.model_context_window,
            model_max_output_tokens=memory_manager.model_max_output_tokens,
            reserved_output_tokens=memory_manager.reserved_output_tokens,
            reserved_output_source=memory_manager.reserved_output_source,
            autocompact_buffer_tokens=memory_manager.autocompact_buffer_tokens,
            event_id=event.event_id,
            created_at=event.created_at,
            metadata={"messages": list(messages)},
        )
        metadata = self._call_before_memory_providers(providers, provider_event)
        if metadata:
            event = MemoryCompactStarted(
                **self._memory_event_context(ctx),
                cycle_index=cycle_index,
                message_count=len(messages),
                estimated_tokens=estimated_tokens,
                trigger=trigger,
                configured_threshold=memory_manager.compact_threshold,
                effective_threshold=memory_manager.autocompact_threshold,
                microcompact_threshold=memory_manager.microcompact_trigger_threshold,
                model_context_window=memory_manager.model_context_window,
                model_max_output_tokens=memory_manager.model_max_output_tokens,
                reserved_output_tokens=memory_manager.reserved_output_tokens,
                reserved_output_source=memory_manager.reserved_output_source,
                autocompact_buffer_tokens=memory_manager.autocompact_buffer_tokens,
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
        mode: CompactionMode,
        changed: bool,
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
            mode=mode,
            changed=changed,
        )
        metadata = self._call_after_memory_providers(providers, event)
        if metadata:
            event = MemoryCompactCompleted(
                **self._memory_event_context(ctx),
                cycle_index=cycle_index,
                before_count=len(before_messages),
                after_count=len(after_messages),
                summary_tokens=event.summary_tokens,
                mode=mode,
                changed=changed,
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
        return ctx.event_handler if ctx is not None else None

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
                    "arguments": json.dumps(
                        tool_call.arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
            if tool_call.extra_content is not None:
                payload["extra_content"] = tool_call.extra_content
            serialized.append(payload)
        return serialized
