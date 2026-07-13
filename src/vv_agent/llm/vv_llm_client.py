from __future__ import annotations

import json
import logging
import random
import re
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam
from vv_llm.chat_clients import create_chat_client, format_messages, get_token_counts
from vv_llm.settings import Settings
from vv_llm.types import APIConnectionError, APIStatusError, BackendType

from vv_agent.llm.anthropic_prompt_cache import (
    PROMPT_CACHE_ENABLED_KEY,
    SYSTEM_PROMPT_SECTIONS_KEY,
    apply_claude_prompt_cache,
)
from vv_agent.llm.base import LLMClient, LlmRequest
from vv_agent.model_settings import ModelSettings, ResponseFormat, ToolChoice
from vv_agent.prompt import CacheBreakTracker, hash_system_prompt_sections, hash_tool_payload
from vv_agent.runtime.context import StreamCallback
from vv_agent.types import LLMResponse, Message, ToolCall

_REASONING_CHAIN_PROVIDERS = {
    "deepseek",
    "minimax",
    "moonshot",
}

_REASONING_CHAIN_MODEL_PREFIXES = (
    "deepseek-",
    "minimax-",
    "kimi-",
    "moonshot-",
)

_QWEN_THINKING_KEEP_SUFFIX_MODELS = (
    "qwen3-next-80b-a3b-thinking",
    "qwen3-vl-235b-a22b-thinking",
    "qwen3-vl-32b-thinking",
    "qwen3-vl-30b-a3b-thinking",
    "qwen3-vl-8b-thinking",
)
_QWEN_THINKING_KEEP_SUFFIX_MODELS_LOWER = {item.lower() for item in _QWEN_THINKING_KEEP_SUFFIX_MODELS}

_TOOL_CALL_INCREMENTAL_ENDPOINT_PREFIXES = (
    "openai",
    "moonshot",
    "anthropic",
    "deepseek",
    "minimax",
    "qwen",
    "zhipuai",
)


@dataclass(slots=True)
class _RequestOptions:
    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    timeout_seconds: float | None = None
    thinking: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None
    extra_args: dict[str, Any] | None = None
    max_attempts: int = 3
    backoff_seconds: float = 2.0
    is_gemini_3_model: bool = False
    tool_call_incremental: bool = True


@dataclass(slots=True)
class EndpointTarget:
    endpoint_id: str
    api_key: str
    api_base: str
    endpoint_type: str = "default"
    model_id: str | None = None


@dataclass(slots=True)
class VVLlmClient(LLMClient):
    endpoint_targets: list[EndpointTarget]
    backend: str = "openai"
    selected_model: str | None = None
    settings: Settings | None = None
    timeout_seconds: float = 90.0
    max_retries_per_endpoint: int = 3
    backoff_seconds: float = 2.0
    randomize_endpoints: bool = True
    debug_dump_dir: str | None = None
    _preferred_endpoint_id: str | None = field(default=None, init=False, repr=False)
    _request_counter: int = field(default=0, init=False, repr=False)
    _prompt_cache_tracker: CacheBreakTracker = field(default_factory=CacheBreakTracker, init=False, repr=False)

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: StreamCallback | None = None,
        model_settings: ModelSettings | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not self.endpoint_targets:
            raise RuntimeError("No endpoint targets configured")
        if model_settings is not None and model_settings.extra_headers:
            raise ValueError(
                "ModelSettings.extra_headers is not supported by the vv-llm adapter; "
                "configure headers on the provider endpoint instead"
            )
        if model_settings is not None and model_settings.extra_args:
            raise ValueError(
                "ModelSettings.extra_args is not supported by the vv-llm adapter; "
                "use extra_body or a custom model client instead"
            )

        backend_type = self._resolve_backend_type(self.backend)
        model_name = self._current_model_name(model)
        settings = self._ensure_settings(model)
        extracted_metadata = self._extract_request_metadata(messages)
        request_metadata = {**extracted_metadata, **dict(request_metadata or {})}
        message_payload = self._build_message_payload(
            messages,
            preserve_reasoning_chain=self._should_preserve_reasoning_chain(model),
        )
        tool_payload = self._build_tool_payload(tools)

        ordered_targets = self._ordered_targets()
        errors: list[str] = []
        last_error: Exception | None = None

        for target in ordered_targets:
            selected_model_id = target.model_id or model
            should_stream = self._should_use_stream(selected_model_id)
            request_options = self._resolve_request_options(
                selected_model_id,
                stream=should_stream,
                endpoint_type=target.endpoint_type,
                model_settings=model_settings,
            )
            request_tool_payload, request_options.tool_choice = self._apply_tool_choice(
                tool_payload,
                request_options.tool_choice,
            )
            request_messages = self._prepare_messages_for_model(message_payload, request_options.model)
            formatted_messages = self._format_messages_for_request(
                settings=settings,
                backend_type=backend_type,
                endpoint_type=target.endpoint_type,
                model_name=model_name,
                messages=request_messages,
            )
            request_messages_payload, request_tool_payload, request_extra_body = apply_claude_prompt_cache(
                endpoint_type=target.endpoint_type,
                model=request_options.model,
                messages=formatted_messages,
                tools=request_tool_payload,
                extra_body=request_options.extra_body,
                metadata=request_metadata,
            )
            self._track_prompt_cache_state(
                endpoint_type=target.endpoint_type,
                model=request_options.model,
                metadata=request_metadata,
                tool_payload=request_tool_payload,
            )
            request_options = _RequestOptions(
                model=request_options.model,
                temperature=request_options.temperature,
                top_p=request_options.top_p,
                max_tokens=request_options.max_tokens,
                tool_choice=request_options.tool_choice,
                parallel_tool_calls=request_options.parallel_tool_calls,
                response_format=request_options.response_format,
                timeout_seconds=request_options.timeout_seconds,
                thinking=request_options.thinking,
                reasoning_effort=request_options.reasoning_effort,
                extra_body=request_extra_body,
                extra_args=request_options.extra_args,
                max_attempts=request_options.max_attempts,
                backoff_seconds=request_options.backoff_seconds,
                is_gemini_3_model=request_options.is_gemini_3_model,
                tool_call_incremental=request_options.tool_call_incremental,
            )
            self._dump_request_messages(request_messages_payload, model_name=model_name)

            for attempt in range(1, request_options.max_attempts + 1):
                try:
                    chat_client = create_chat_client(
                        backend=backend_type,
                        model=model_name,
                        stream=should_stream,
                        random_endpoint=False,
                        endpoint_id=target.endpoint_id,
                        settings=settings,
                    )
                    if should_stream:
                        response = self._stream_completion(
                            chat_client=chat_client,
                            options=request_options,
                            messages=request_messages_payload,
                            model_name=model_name,
                            tool_payload=request_tool_payload,
                            stream_callback=stream_callback,
                        )
                    else:
                        response = self._non_stream_completion(
                            chat_client=chat_client,
                            options=request_options,
                            messages=request_messages_payload,
                            model_name=model_name,
                            tool_payload=request_tool_payload,
                        )

                    response.raw["used_endpoint_id"] = target.endpoint_id
                    response.raw["used_model_id"] = request_options.model
                    response.raw["stream_mode"] = should_stream
                    self._preferred_endpoint_id = target.endpoint_id
                    return response
                except APIConnectionError as exc:
                    last_error = exc
                    errors.append(f"{target.endpoint_id}: network timeout/connection error (attempt {attempt})")
                    if attempt < request_options.max_attempts:
                        self._sleep_backoff(attempt, request_options.backoff_seconds)
                        continue
                    break
                except APIStatusError as exc:
                    last_error = exc
                    status = exc.status_code
                    detail = getattr(exc, 'message', '') or str(getattr(exc, 'body', ''))
                    errors.append(f"{target.endpoint_id}: status {status} - {detail} (attempt {attempt})")
                    if status in {429, 500, 502, 503, 504, 408} and attempt < request_options.max_attempts:
                        self._sleep_backoff(attempt, request_options.backoff_seconds)
                        continue
                    if status in {400, 413, 422}:
                        raise
                    break
                except Exception:
                    raise

        details = "; ".join(errors) if errors else "no attempts made"
        raise RuntimeError(f"All endpoints failed: {details}") from last_error

    def complete_request(
        self,
        request: LlmRequest,
        *,
        stream_callback: StreamCallback | None = None,
    ) -> LLMResponse:
        return self.complete(
            model=request.model,
            messages=request.messages,
            tools=request.tools,
            stream_callback=stream_callback,
            model_settings=request.model_settings,
            request_metadata=request.metadata,
        )

    def _ensure_settings(self, model: str) -> Settings:
        if self.settings is not None:
            return self.settings

        backend = self.backend.strip().lower()
        model_name = self._current_model_name(model)
        endpoint_options = [
            {
                "endpoint_id": target.endpoint_id,
                "model_id": target.model_id or model,
            }
            for target in self.endpoint_targets
        ]
        settings_data = {
            "VERSION": "2",
            "backends": {
                backend: {
                    "default_endpoint": self.endpoint_targets[0].endpoint_id,
                    "models": {
                        model_name: {
                            "id": model,
                            "endpoints": endpoint_options,
                            "function_call_available": True,
                            "native_multimodal": True,
                        }
                    },
                }
            },
            "endpoints": [
                {
                    "id": target.endpoint_id,
                    "api_key": target.api_key,
                    "api_base": target.api_base,
                    "endpoint_type": target.endpoint_type,
                }
                for target in self.endpoint_targets
            ],
        }
        self.settings = Settings(**settings_data)
        return self.settings

    def _build_message_payload(
        self,
        messages: list[Message],
        *,
        preserve_reasoning_chain: bool = False,
    ) -> list[ChatCompletionMessageParam]:
        last_assistant_index = max(
            (index for index, message in enumerate(messages) if message.role == "assistant"),
            default=-1,
        )

        payload: list[ChatCompletionMessageParam] = []
        for index, message in enumerate(messages):
            include_reasoning = message.role == "assistant" and (
                preserve_reasoning_chain or index == last_assistant_index
            )
            item = message.to_openai_message(include_reasoning_content=include_reasoning)
            if preserve_reasoning_chain and message.role == "assistant" and "reasoning_content" not in item:
                # Moonshot/DeepSeek/MiniMax reasoning tool-call flows require this field.
                item["reasoning_content"] = message.reasoning_content or ""
            payload.append(cast(ChatCompletionMessageParam, item))
        return payload

    @staticmethod
    def _extract_request_metadata(messages: list[Message]) -> dict[str, Any]:
        for message in messages:
            if message.role != "system":
                continue
            return dict(message.metadata)
        return {}

    def _track_prompt_cache_state(
        self,
        *,
        endpoint_type: str,
        model: str,
        metadata: dict[str, Any],
        tool_payload: list[dict[str, Any]],
    ) -> None:
        normalized_endpoint = str(endpoint_type or "").strip().lower()
        normalized_model = str(model or "").strip().lower()
        if normalized_endpoint not in {"anthropic", "anthropic_vertex"}:
            return
        if not normalized_model.startswith("claude"):
            return
        if metadata.get(PROMPT_CACHE_ENABLED_KEY, True) is False:
            return

        system_hash = hash_system_prompt_sections(metadata.get(SYSTEM_PROMPT_SECTIONS_KEY))
        tool_hash = hash_tool_payload(tool_payload)
        self._prompt_cache_tracker.check(system_hash=system_hash, tool_hash=tool_hash)

    def _should_preserve_reasoning_chain(self, requested_model: str) -> bool:
        if self._is_reasoning_chain_provider(self.backend):
            return True
        if any(self._is_reasoning_chain_provider(target.endpoint_type) for target in self.endpoint_targets):
            return True
        for candidate in self._iter_reasoning_model_candidates(requested_model):
            normalized = candidate.strip().lower()
            if not normalized:
                continue
            if normalized.startswith(_REASONING_CHAIN_MODEL_PREFIXES):
                return True
        return False

    @staticmethod
    def _is_reasoning_chain_provider(value: str) -> bool:
        normalized = value.strip().lower()
        return any(
            normalized == provider or normalized.startswith(f"{provider}-") or normalized.startswith(f"{provider}_")
            for provider in _REASONING_CHAIN_PROVIDERS
        )

    def _uses_deepseek_model(self, *, model: str, endpoint_type: str | None) -> bool:
        normalized_model = model.strip().lower()
        if normalized_model.startswith("deepseek-"):
            return True
        normalized_backend = self.backend.strip().lower()
        if self._is_reasoning_chain_provider(normalized_backend) and normalized_backend.startswith("deepseek"):
            return True
        if endpoint_type is None:
            return False
        normalized_endpoint = endpoint_type.strip().lower()
        return self._is_reasoning_chain_provider(normalized_endpoint) and normalized_endpoint.startswith("deepseek")

    def _iter_reasoning_model_candidates(self, requested_model: str) -> list[str]:
        candidates: list[str] = [requested_model]
        if self.selected_model:
            candidates.append(self.selected_model)
        for target in self.endpoint_targets:
            if target.model_id:
                candidates.append(target.model_id)
        return candidates

    @staticmethod
    def _resolve_backend_type(raw_backend: str) -> BackendType:
        normalized = raw_backend.strip().lower()
        try:
            return BackendType(normalized)
        except ValueError as exc:
            raise RuntimeError(f"Unsupported backend for vv-llm: {raw_backend!r}") from exc

    def _current_model_name(self, fallback: str) -> str:
        if self.selected_model and self.selected_model.strip():
            return self.selected_model.strip()
        return fallback

    def _ordered_targets(self) -> list[EndpointTarget]:
        targets = list(self.endpoint_targets)

        if self._preferred_endpoint_id:
            preferred_index = None
            for index, target in enumerate(targets):
                if target.endpoint_id == self._preferred_endpoint_id:
                    preferred_index = index
                    break
            if preferred_index is not None:
                preferred = targets.pop(preferred_index)
                if self.randomize_endpoints:
                    random.shuffle(targets)
                targets.insert(0, preferred)
                return targets

        if self.randomize_endpoints:
            random.shuffle(targets)
        return targets

    def _format_messages_for_request(
        self,
        *,
        settings: Settings,
        backend_type: BackendType,
        endpoint_type: str,
        model_name: str,
        messages: list[ChatCompletionMessageParam],
    ) -> list[dict[str, Any]]:
        try:
            backend_settings = settings.get_backend(backend_type)
            model_settings = backend_settings.get_model_setting(model_name)
            format_backend = BackendType.OpenAI if endpoint_type.startswith("openai") else backend_type
            formatted = format_messages(
                messages=messages,
                backend=format_backend,
                native_multimodal=model_settings.native_multimodal,
                function_call_available=model_settings.function_call_available,
            )
            return cast(list[dict[str, Any]], formatted)
        except Exception:
            return [cast(dict[str, Any], dict(message)) for message in messages]

    @staticmethod
    def _build_tool_payload(tools: list[dict[str, object]]) -> list[dict[str, Any]]:
        if not tools:
            return []

        payload: list[dict[str, Any]] = []
        for schema in tools:
            schema_type = schema.get("type")
            schema_function = schema.get("function")
            if schema_type == "function" and isinstance(schema_function, dict):
                payload.append(cast(dict[str, Any], schema))
            else:
                payload.append({"type": "function", "function": schema})
        return payload

    @staticmethod
    def _should_use_stream(_model: str) -> bool:
        return True

    @staticmethod
    def _prepare_messages_for_model(
        messages: list[ChatCompletionMessageParam],
        model: str,
    ) -> list[ChatCompletionMessageParam]:
        # MiniMax endpoints reject requests with multiple system-role turns.
        minimax_strict_system = model.lower().startswith("minimax")
        prepared: list[ChatCompletionMessageParam] = []
        seen_system = False

        for raw_message in messages:
            message = dict(raw_message)
            role = message.get("role")
            if role == "system":
                if not seen_system:
                    seen_system = True
                    prepared.append(cast(ChatCompletionMessageParam, message))
                    continue
                if minimax_strict_system:
                    summary_content = message.get("content")
                    summary_text = summary_content if isinstance(summary_content, str) else ""
                    prefix = "[memory_summary]\n" if message.get("name") == "memory_summary" else ""
                    prepared.append(
                        cast(
                            ChatCompletionMessageParam,
                            {
                                "role": "user",
                                "content": f"{prefix}{summary_text}".strip(),
                            },
                        )
                    )
                    continue

            prepared.append(cast(ChatCompletionMessageParam, message))

        return prepared

    def _resolve_request_options(
        self,
        model: str,
        *,
        stream: bool,
        endpoint_type: str | None,
        model_settings: ModelSettings | None = None,
    ) -> _RequestOptions:
        resolved_model = model
        normalized_model = resolved_model.lower()
        temperature: float | None = None
        top_p: float | None = None
        max_tokens: int | None = None
        tool_choice: str | dict[str, Any] | None = None
        parallel_tool_calls: bool | None = None
        response_format: dict[str, Any] | None = None
        timeout_seconds: float | None = None
        thinking: dict[str, Any] | None = None
        reasoning_effort: str | None = None
        extra_body: dict[str, Any] | None = None
        extra_args: dict[str, Any] | None = None
        max_attempts = max(1, int(self.max_retries_per_endpoint))
        backoff_seconds = max(0.0, float(self.backoff_seconds))

        if self._uses_deepseek_model(model=normalized_model, endpoint_type=endpoint_type):
            extra_body = {"thinking": {"type": "enabled"}}
            reasoning_effort = "max"
        elif normalized_model.startswith("claude") and normalized_model.endswith("-thinking"):
            resolved_model = self._remove_suffix_case_insensitive(resolved_model, "-thinking")
            normalized_model = resolved_model.lower()
            thinking = {"type": "enabled", "budget_tokens": 16000}
            temperature = 1.0
            max_tokens = 20000

        if normalized_model in ("o3-mini-high", "o4-mini-high"):
            reasoning_effort = "high"
            resolved_model = self._remove_suffix_case_insensitive(resolved_model, "-high")
            normalized_model = resolved_model.lower()

        if normalized_model.startswith("gpt-5") and normalized_model.endswith("-high"):
            reasoning_effort = "high"
            resolved_model = self._remove_suffix_case_insensitive(resolved_model, "-high")
            normalized_model = resolved_model.lower()

        if stream and normalized_model.startswith("qwen3"):
            if normalized_model.endswith("-thinking"):
                if normalized_model not in _QWEN_THINKING_KEEP_SUFFIX_MODELS_LOWER:
                    resolved_model = self._remove_suffix_case_insensitive(resolved_model, "-thinking")
                    normalized_model = resolved_model.lower()
                extra_body = {"enable_thinking": True}
            else:
                extra_body = {"enable_thinking": False}

        if normalized_model.startswith(("glm-4.", "glm-5")) and normalized_model.endswith("-thinking"):
            resolved_model = self._remove_suffix_case_insensitive(resolved_model, "-thinking")
            normalized_model = resolved_model.lower()
            extra_body = {"thinking": {"type": "enabled"}, "tool_stream": True} if stream else {"thinking": {"type": "enabled"}}

        if normalized_model.startswith("gemini-2.5"):
            reasoning_effort = None
            extra_body = {
                "extra_body": {
                    "google": {
                        "thinking_config": {
                            "thinkingBudget": -1,
                            "include_thoughts": True,
                        }
                    }
                }
            }

        is_gemini_3_model = normalized_model.startswith("gemini-3")
        if is_gemini_3_model:
            if temperature is None:
                temperature = 1.0
            if normalized_model in {"gemini-3-pro", "gemini-3-flash"}:
                resolved_model = f"{resolved_model}-preview"
                normalized_model = resolved_model.lower()
            reasoning_effort = None
            extra_body = {
                "extra_body": {
                    "google": {
                        "thinking_config": {
                            "thinkingLevel": "high",
                            "include_thoughts": True,
                        }
                    }
                }
            }

        if model_settings is not None:
            if model_settings.temperature is not None:
                temperature = model_settings.temperature
            if model_settings.top_p is not None:
                top_p = model_settings.top_p
            if model_settings.max_tokens is not None:
                max_tokens = model_settings.max_tokens
            if model_settings.tool_choice is not None:
                tool_choice = cast(ToolChoice, model_settings.tool_choice).to_wire()
            if model_settings.parallel_tool_calls is not None:
                parallel_tool_calls = model_settings.parallel_tool_calls
            if model_settings.response_format is not None:
                response_format = cast(ResponseFormat, model_settings.response_format).to_wire()
            if model_settings.timeout_seconds is not None:
                timeout_seconds = model_settings.timeout_seconds
            if model_settings.reasoning is not None:
                reasoning = dict(model_settings.reasoning)
                effort = reasoning.pop("effort", reasoning.pop("reasoning_effort", None))
                if effort is not None:
                    reasoning_effort = str(effort)
                if reasoning:
                    thinking = reasoning
            if model_settings.extra_body is not None:
                merged_extra_body = dict(extra_body or {})
                merged_extra_body.update(model_settings.extra_body)
                extra_body = merged_extra_body
            if model_settings.retry is not None:
                max_attempts = max(1, int(model_settings.retry.max_attempts))
                backoff_seconds = max(0.0, float(model_settings.retry.backoff_seconds))

        return _RequestOptions(
            model=resolved_model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            response_format=response_format,
            timeout_seconds=timeout_seconds,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            extra_args=extra_args,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            is_gemini_3_model=is_gemini_3_model,
            tool_call_incremental=self._tool_call_incremental_enabled(
                model=resolved_model,
                endpoint_type=endpoint_type,
            ),
        )

    @staticmethod
    def _tool_call_incremental_enabled(*, model: str, endpoint_type: str | None) -> bool:
        normalized_endpoint = (endpoint_type or "").strip().lower()
        if not normalized_endpoint or normalized_endpoint == "default":
            return True

        return normalized_endpoint.startswith(_TOOL_CALL_INCREMENTAL_ENDPOINT_PREFIXES)

    @staticmethod
    def _remove_suffix_case_insensitive(value: str, suffix: str) -> str:
        if value.lower().endswith(suffix.lower()):
            return value[: -len(suffix)]
        return value

    def _build_request_payload(
        self,
        *,
        options: _RequestOptions,
        messages: list[dict[str, Any]],
        model_name: str,
        tool_payload: list[dict[str, Any]],
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "model": model_name,
            "stream": stream,
            "skip_cutoff": True,
            "timeout": options.timeout_seconds if options.timeout_seconds is not None else self.timeout_seconds,
        }
        if tool_payload:
            payload["tools"] = tool_payload
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if options.temperature is not None:
            payload["temperature"] = options.temperature
        if options.top_p is not None:
            payload["top_p"] = options.top_p
        if options.max_tokens is not None:
            payload["max_tokens"] = options.max_tokens
        if options.tool_choice is not None:
            payload["tool_choice"] = options.tool_choice
        if options.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = options.parallel_tool_calls
        if options.response_format is not None:
            payload["response_format"] = options.response_format
        if options.thinking is not None:
            payload["thinking"] = options.thinking
        if options.reasoning_effort is not None:
            payload["reasoning_effort"] = options.reasoning_effort
        if options.extra_body is not None:
            payload["extra_body"] = options.extra_body
        if options.extra_args is not None:
            payload.update(options.extra_args)
        return payload

    @staticmethod
    def _apply_tool_choice(
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if tool_choice == "none":
            return [], None
        if not isinstance(tool_choice, dict):
            return list(tools), tool_choice

        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        selected = [tool for tool in tools if VVLlmClient._tool_payload_name(tool) == name]
        if not selected:
            raise ValueError(f"tool_choice refers to unknown tool: {name}")
        return selected, "required"

    @staticmethod
    def _tool_payload_name(tool: dict[str, Any]) -> str | None:
        function = tool.get("function")
        if not isinstance(function, dict):
            return None
        name = function.get("name")
        return name if isinstance(name, str) else None

    def _non_stream_completion(
        self,
        *,
        chat_client: Any,
        options: _RequestOptions,
        messages: list[dict[str, Any]],
        model_name: str,
        tool_payload: list[dict[str, Any]],
    ) -> LLMResponse:
        payload = self._build_request_payload(
            options=options,
            messages=messages,
            model_name=model_name,
            tool_payload=tool_payload,
            stream=False,
        )

        response = chat_client.create_completion(**payload)
        parsed_tool_calls = self._parse_non_stream_tool_calls(self._read_field(response, "tool_calls"))
        reasoning_content = self._extract_reasoning_content(self._read_field(response, "reasoning_content"))
        if not reasoning_content:
            reasoning_content = self._extract_reasoning_content(self._read_field(response, "reasoning"))

        content = self._extract_content(self._read_field(response, "content"))
        usage_dump = self._usage_to_dict(self._read_field(response, "usage"))
        if not usage_dump:
            usage_dump = self._estimate_usage(messages=messages, content=content, model=options.model)

        raw_payload: dict[str, Any] = {
            "usage": usage_dump,
            "stream_collected": False,
        }
        if reasoning_content:
            raw_payload["reasoning_content"] = reasoning_content

        raw_content = self._read_field(response, "raw_content")
        if raw_content is not None:
            raw_payload["raw_content"] = raw_content

        return LLMResponse(
            content=content,
            tool_calls=parsed_tool_calls,
            raw=raw_payload,
        )

    def _stream_completion(
        self,
        *,
        chat_client: Any,
        options: _RequestOptions,
        messages: list[dict[str, Any]],
        model_name: str,
        tool_payload: list[dict[str, Any]],
        stream_callback: StreamCallback | None = None,
    ) -> LLMResponse:
        payload = self._build_request_payload(
            options=options,
            messages=messages,
            model_name=model_name,
            tool_payload=tool_payload,
            stream=True,
        )

        stream = cast(Iterable[Any], chat_client.create_completion(**payload))

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        content_chars = 0
        reasoning_chars = 0
        complete_raw_content: list[dict[str, Any]] = []
        tool_call_parts: dict[str, dict[str, Any]] = {}
        last_active_tool_call_id: str | None = None
        usage_dump: dict[str, Any] | None = None

        for chunk in stream:
            usage = self._read_field(chunk, "usage")
            usage_candidate = self._usage_to_dict(usage)
            if usage_candidate:
                usage_dump = usage_candidate

            chunk_reasoning = self._extract_reasoning_content(self._read_field(chunk, "reasoning_content"))
            if chunk_reasoning:
                reasoning_parts.append(chunk_reasoning)
                reasoning_chars += len(chunk_reasoning)
                self._emit_stream_event(
                    stream_callback,
                    {
                        "event": "reasoning_delta",
                        "reasoning_delta": chunk_reasoning,
                        "reasoning_chars": reasoning_chars,
                        "estimated_tokens": self._estimate_stream_tokens(reasoning_chars),
                    },
                )

            raw_content = self._read_field(chunk, "raw_content")
            if raw_content is not None:
                self._collect_raw_content(complete_raw_content, raw_content)

            text = self._extract_content(self._read_field(chunk, "content"))
            if text:
                content_parts.append(text)
                content_chars += len(text)
                self._emit_stream_event(
                    stream_callback,
                    {
                        "event": "assistant_delta",
                        "content_delta": text,
                        "content_chars": content_chars,
                        "estimated_tokens": self._estimate_stream_tokens(content_chars),
                    },
                )

            for tool_call_index, tool_delta in enumerate(self._read_field(chunk, "tool_calls") or []):
                previous_active_tool_call_id = last_active_tool_call_id
                last_active_tool_call_id = self._accumulate_tool_call_delta(
                    tool_call_parts=tool_call_parts,
                    tool_delta=tool_delta,
                    default_index=tool_call_index,
                    last_active_tool_call_id=last_active_tool_call_id,
                    incremental=options.tool_call_incremental,
                    keep_extra_content=options.is_gemini_3_model,
                )
                self._emit_tool_call_stream_events(
                    stream_callback=stream_callback,
                    tool_call_parts=tool_call_parts,
                    tool_delta=tool_delta,
                    default_index=tool_call_index,
                    previous_active_tool_call_id=previous_active_tool_call_id,
                    active_tool_call_id=last_active_tool_call_id,
                )

        parsed_tool_calls: list[ToolCall] = []
        tool_call_extra_content: dict[str, Any] = {}
        for _, slot in sorted(tool_call_parts.items(), key=self._tool_call_sort_key):
            name = str(slot.get("name", "")).strip()
            if not name:
                continue
            raw_arguments = str(slot.get("arguments", ""))
            tool_id = str(slot.get("id") or f"call_{uuid.uuid4().hex[:12]}")
            raw_extra_content = slot.get("extra_content")
            parsed_tool_calls.append(
                ToolCall(
                    id=tool_id,
                    name=name,
                    arguments=self._parse_arguments(raw_arguments),
                    extra_content=raw_extra_content if isinstance(raw_extra_content, dict) else None,
                )
            )
            if options.is_gemini_3_model and "extra_content" in slot:
                tool_call_extra_content[tool_id] = slot["extra_content"]

        normalized = self._normalize_tool_calls(parsed_tool_calls)
        final_usage = usage_dump or self._estimate_usage(
            messages=messages,
            content="".join(content_parts),
            model=options.model,
        )
        raw_payload: dict[str, Any] = {
            "usage": final_usage,
            "stream_collected": True,
        }
        if reasoning_parts:
            raw_payload["reasoning_content"] = "".join(reasoning_parts)
        if complete_raw_content:
            raw_payload["raw_content"] = complete_raw_content
        if tool_call_extra_content:
            raw_payload["tool_call_extra_content"] = tool_call_extra_content

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=normalized,
            raw=raw_payload,
        )

    @staticmethod
    def _emit_stream_event(stream_callback: StreamCallback | None, event: dict[str, Any]) -> None:
        if stream_callback is not None:
            stream_callback(event)

    @staticmethod
    def _estimate_stream_tokens(text_length: int) -> int:
        if text_length <= 0:
            return 0
        return (text_length + 3) // 4

    def _emit_tool_call_stream_events(
        self,
        *,
        stream_callback: StreamCallback | None,
        tool_call_parts: dict[str, dict[str, Any]],
        tool_delta: Any,
        default_index: int,
        previous_active_tool_call_id: str | None,
        active_tool_call_id: str | None,
    ) -> None:
        if stream_callback is None or not active_tool_call_id or active_tool_call_id not in tool_call_parts:
            return

        function = self._read_field(tool_delta, "function")
        if function is None:
            return

        name_raw = self._read_field(function, "name")
        has_name = isinstance(name_raw, str) and bool(name_raw.strip())
        arguments_raw = self._read_field(function, "arguments")
        has_arguments_delta = isinstance(arguments_raw, str) and bool(arguments_raw)
        slot = tool_call_parts[active_tool_call_id]

        started_emitted = bool(slot.get("_stream_started_emitted"))
        if has_name and not started_emitted:
            arguments_chars = len(str(slot.get("arguments") or ""))
            self._emit_stream_event(
                stream_callback,
                {
                    "event": "tool_call_started",
                    "tool_call_id": str(slot.get("id") or ""),
                    "tool_call_index": self._resolve_tool_call_index(tool_delta, default_index),
                    "function_name": str(slot.get("name") or ""),
                    "arguments_chars": arguments_chars,
                    "estimated_tokens": self._estimate_stream_tokens(arguments_chars),
                },
            )
            slot["_stream_started_emitted"] = True

        if has_arguments_delta:
            arguments_chars = len(str(slot.get("arguments") or ""))
            self._emit_stream_event(
                stream_callback,
                {
                    "event": "tool_call_progress",
                    "tool_call_id": str(slot.get("id") or ""),
                    "tool_call_index": self._resolve_tool_call_index(tool_delta, default_index),
                    "function_name": str(slot.get("name") or ""),
                    "arguments_chars": arguments_chars,
                    "estimated_tokens": self._estimate_stream_tokens(arguments_chars),
                },
            )

    def _resolve_tool_call_index(self, tool_delta: Any, default_index: int) -> int:
        index_raw = self._read_field(tool_delta, "index")
        return index_raw if isinstance(index_raw, int) else default_index

    @staticmethod
    def _tool_call_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        call_id, _ = item
        index_text, _, _ = call_id.partition("_")
        try:
            index = int(index_text)
        except ValueError:
            index = 10**9
        return index, call_id

    @classmethod
    def _collect_raw_content(cls, complete_raw_content: list[dict[str, Any]], chunk_raw_content: Any) -> None:
        if isinstance(chunk_raw_content, list):
            for item in chunk_raw_content:
                cls._collect_raw_content(complete_raw_content, item)
            return

        if not isinstance(chunk_raw_content, dict):
            return

        chunk_type_raw = chunk_raw_content.get("type")
        chunk_type = chunk_type_raw if isinstance(chunk_type_raw, str) else ""

        if chunk_type == "thinking_delta":
            thinking_block = cls._find_or_create_raw_block(
                complete_raw_content,
                block_type="thinking",
                defaults={"thinking": "", "signature": ""},
            )
            thinking_block["thinking"] = f"{thinking_block.get('thinking', '')}{chunk_raw_content.get('thinking', '')}"
            return

        if chunk_type == "signature_delta":
            thinking_block = cls._find_or_create_raw_block(
                complete_raw_content,
                block_type="thinking",
                defaults={"thinking": "", "signature": ""},
            )
            thinking_block["signature"] = f"{thinking_block.get('signature', '')}{chunk_raw_content.get('signature', '')}"
            return

        if chunk_type == "text_delta":
            text_block = cls._find_or_create_raw_block(
                complete_raw_content,
                block_type="text",
                defaults={"text": ""},
            )
            text_block["text"] = f"{text_block.get('text', '')}{chunk_raw_content.get('text', '')}"
            return

        if chunk_type == "input_json_delta":
            # input_json_delta is already aggregated via tool call deltas.
            return

        if chunk_type in {"thinking", "text", "tool_use"}:
            if not cls._raw_block_exists(complete_raw_content, chunk_raw_content):
                complete_raw_content.append(dict(chunk_raw_content))
            return

        complete_raw_content.append(dict(chunk_raw_content))

    @staticmethod
    def _find_or_create_raw_block(
        blocks: list[dict[str, Any]],
        *,
        block_type: str,
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == block_type:
                return block
        new_block = {"type": block_type, **defaults}
        blocks.append(new_block)
        return new_block

    @staticmethod
    def _raw_block_exists(blocks: list[dict[str, Any]], candidate: dict[str, Any]) -> bool:
        candidate_type = candidate.get("type")
        candidate_id = candidate.get("id")
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") != candidate_type:
                continue
            if candidate_id is not None and block.get("id") == candidate_id:
                return True
            if candidate_id is None and block == candidate:
                return True
        return False

    def _accumulate_tool_call_delta(
        self,
        *,
        tool_call_parts: dict[str, dict[str, Any]],
        tool_delta: Any,
        default_index: int,
        last_active_tool_call_id: str | None,
        incremental: bool,
        keep_extra_content: bool,
    ) -> str | None:
        function = self._read_field(tool_delta, "function")
        if function is None:
            return last_active_tool_call_id

        index_raw = self._read_field(tool_delta, "index")
        index = index_raw if isinstance(index_raw, int) else default_index

        name_raw = self._read_field(function, "name")
        name = name_raw.strip() if isinstance(name_raw, str) else ""
        arguments_raw = self._read_field(function, "arguments")
        arguments = arguments_raw if isinstance(arguments_raw, str) else ""

        delta_id_raw = self._read_field(tool_delta, "id")
        delta_id = delta_id_raw.strip() if isinstance(delta_id_raw, str) else ""

        extra_content = self._read_field(tool_delta, "extra_content") if keep_extra_content else None

        if name:
            tool_id = delta_id or f"generated_{index}_{len(tool_call_parts)}"
            unique_id = f"{index}_{tool_id}"

            if unique_id in tool_call_parts:
                slot = tool_call_parts[unique_id]
                if arguments:
                    slot["arguments"] = self._merge_tool_arguments(
                        existing=str(slot.get("arguments", "")),
                        incoming=arguments,
                        incremental=incremental,
                    )
                if delta_id and (not slot.get("id") or str(slot["id"]).startswith("generated_")):
                    slot["id"] = delta_id
                if keep_extra_content and extra_content:
                    slot["extra_content"] = extra_content
            else:
                tool_call_parts[unique_id] = {
                    "id": tool_id,
                    "name": name,
                    "arguments": arguments,
                }
                if keep_extra_content and extra_content:
                    tool_call_parts[unique_id]["extra_content"] = extra_content

            return unique_id

        if not arguments and not delta_id:
            return last_active_tool_call_id

        target_id = last_active_tool_call_id
        if target_id is None and delta_id:
            for existing_id, slot in tool_call_parts.items():
                if slot.get("id") == delta_id:
                    target_id = existing_id
                    break

        if target_id and target_id in tool_call_parts:
            slot = tool_call_parts[target_id]
            if arguments:
                slot["arguments"] = self._merge_tool_arguments(
                    existing=str(slot.get("arguments", "")),
                    incoming=arguments,
                    incremental=incremental,
                )
            if delta_id and (not slot.get("id") or str(slot["id"]).startswith("generated_")):
                slot["id"] = delta_id

        return target_id

    @staticmethod
    def _merge_tool_arguments(*, existing: str, incoming: str, incremental: bool) -> str:
        if not existing:
            return incoming
        if incremental:
            return existing + incoming
        return incoming

    def _parse_non_stream_tool_calls(self, tool_calls_raw: Any) -> list[ToolCall]:
        parsed_tool_calls: list[ToolCall] = []
        for call in tool_calls_raw or []:
            function_call = self._read_field(call, "function")
            if function_call is None:
                continue

            name_raw = self._read_field(function_call, "name")
            if not isinstance(name_raw, str) or not name_raw.strip():
                continue

            arguments_raw = self._read_field(function_call, "arguments")
            call_id_raw = self._read_field(call, "id")
            extra_content_raw = self._read_field(call, "extra_content")
            parsed_tool_calls.append(
                ToolCall(
                    id=call_id_raw if isinstance(call_id_raw, str) and call_id_raw else f"call_{uuid.uuid4().hex[:12]}",
                    name=name_raw,
                    arguments=self._parse_arguments(arguments_raw if isinstance(arguments_raw, str) else None),
                    extra_content=extra_content_raw if isinstance(extra_content_raw, dict) else None,
                )
            )

        return self._normalize_tool_calls(parsed_tool_calls)

    @staticmethod
    def _normalize_tool_calls(tool_calls: list[ToolCall]) -> list[ToolCall]:
        normalized: list[ToolCall] = []
        for tool_call in tool_calls:
            tool_id = tool_call.id or f"call_{uuid.uuid4().hex[:12]}"
            tool_name = tool_call.name.replace(" ", "")
            normalized.append(
                ToolCall(
                    id=tool_id,
                    name=tool_name,
                    arguments=tool_call.arguments,
                    extra_content=tool_call.extra_content,
                )
            )
        return normalized

    @staticmethod
    def _parse_arguments(raw_arguments: str | None) -> dict[str, Any]:
        if not raw_arguments:
            return {}
        try:
            payload = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {"_raw": raw_arguments}
        if isinstance(payload, dict):
            return payload
        return {"_value": payload}

    @staticmethod
    def _extract_content(content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                text: str | None = None
                if isinstance(block, dict):
                    value = block.get("text")
                    if isinstance(value, str):
                        text = value
                else:
                    value = getattr(block, "text", None)
                    if isinstance(value, str):
                        text = value
                if text:
                    parts.append(text)
            return "\n".join(parts)

        if content is None:
            return ""

        return str(content)

    @staticmethod
    def _extract_reasoning_content(content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, dict):
            for key in ("reasoning_content", "reasoning", "thinking", "text", "content"):
                value = content.get(key)
                if isinstance(value, str) and value:
                    return value
            return ""

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    if item:
                        parts.append(item)
                    continue

                if isinstance(item, dict):
                    for key in ("reasoning_content", "reasoning", "thinking", "text", "content"):
                        value = item.get(key)
                        if isinstance(value, str) and value:
                            parts.append(value)
                            break
                    continue

                for key in ("reasoning_content", "reasoning", "thinking", "text", "content"):
                    value = getattr(item, key, None)
                    if isinstance(value, str) and value:
                        parts.append(value)
                        break

            return "".join(parts)

        for key in ("reasoning_content", "reasoning", "thinking", "text", "content"):
            value = getattr(content, key, None)
            if isinstance(value, str) and value:
                return value

        return ""

    @staticmethod
    def _read_field(source: Any, key: str) -> Any:
        if isinstance(source, dict):
            return source.get(key)
        return getattr(source, key, None)

    @staticmethod
    def _usage_to_dict(usage: Any) -> dict[str, Any]:
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return dict(usage)
        model_dump = getattr(usage, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        return {}

    def _estimate_usage(self, *, messages: list[dict[str, Any]], content: str, model: str) -> dict[str, Any]:
        prompt_payload = json.dumps(messages, ensure_ascii=False, default=str)
        prompt_tokens = self._estimate_token_count(prompt_payload, model)
        completion_tokens = self._estimate_token_count(content, model)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    @staticmethod
    def _estimate_token_count(text: str, model: str) -> int:
        if not text:
            return 0
        try:
            count = get_token_counts(text, model=model, use_token_server_first=False)
            if isinstance(count, int) and count >= 0:
                return count
        except Exception:
            pass

        # Heuristic fallback when tokenizer is unavailable for this model.
        return max(len(text) // 4, 1)

    @staticmethod
    def _sleep_backoff(attempt: int, backoff_seconds: float) -> None:
        jitter = random.uniform(0.0, 0.5)
        sleep_seconds = backoff_seconds * attempt + jitter
        time.sleep(sleep_seconds)

    def _dump_request_messages(self, messages: list[dict[str, Any]], *, model_name: str) -> None:
        if not self.debug_dump_dir:
            return
        self._request_counter += 1
        dump_dir = Path(self.debug_dump_dir)
        try:
            dump_dir.mkdir(parents=True, exist_ok=True)
            safe_model_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name).strip("_") or "model"
            filename = f"request_{self._request_counter:03d}_{safe_model_name}.json"
            payload = {
                "request_index": self._request_counter,
                "model": model_name,
                "message_count": len(messages),
                "messages": messages,
            }
            (dump_dir / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logging.getLogger(__name__).debug("Failed to dump request messages", exc_info=True)


VvLlmClient = VVLlmClient
