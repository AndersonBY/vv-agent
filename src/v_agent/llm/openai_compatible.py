from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, cast

from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam

from v_agent.llm.base import LLMClient
from v_agent.types import LLMResponse, Message, ToolCall

_STREAM_MODEL_PREFIXES = (
    "qwen3",
    "claude",
    "gemini",
    "kimi",
    "glm-4.",
    "glm-5",
    "gpt-5",
    "MiniMax",
)

_STREAM_MODEL_EXACT = {
    "deepseek-reasoner",
    "deepseek-r1-tools",
}

_DEEPSEEK_REASONING_MODELS = (
    "deepseek-reasoner",
    "deepseek-r1-tools",
)

_CLAUDE_THINKING_MODELS = (
    "claude-3-7-sonnet-thinking",
    "claude-opus-4-20250514-thinking",
    "claude-opus-4-1-20250805-thinking",
    "claude-sonnet-4-20250514-thinking",
    "claude-sonnet-4-5-20250929-thinking",
    "claude-opus-4-5-20251101-thinking",
    "claude-opus-4-6-thinking",
)

_QWEN_THINKING_KEEP_SUFFIX_MODELS = (
    "qwen3-next-80b-a3b-thinking",
    "qwen3-vl-235b-a22b-thinking",
    "qwen3-vl-32b-thinking",
    "qwen3-vl-30b-a3b-thinking",
    "qwen3-vl-8b-thinking",
)

_TOOL_CALL_INCREMENTAL_MODELS = {
    "qwen3-coder-plus",
    "qwen3-coder-flash",
    "qwen3-max",
    "qwen3-max-preview",
    "qwen3-next-80b-a3b-thinking",
    "qwen3-next-80b-a3b-instruct",
    "qwen3-235b-a22b-instruct-2507",
    "qwen3-coder-480b-a35b-instruct",
    "qwen3-235b-a22b",
    "qwen3-235b-a22b-thinking",
    "qwen3-32b",
    "qwen3-32b-thinking",
    "qwen3-30b-a3b",
    "qwen3-30b-a3b-thinking",
    "qwen3-14b",
    "qwen3-14b-thinking",
    "qwen3-8b",
    "qwen3-8b-thinking",
    "qwen3-4b",
    "qwen3-4b-thinking",
    "qwen3-1.7b",
    "qwen3-1.7b-thinking",
    "qwen3-0.6b",
    "qwen3-0.6b-thinking",
    "mixtral-8x7b",
}

_TOOL_CALL_INCREMENTAL_ENDPOINT_PREFIXES = (
    "openai",
    "moonshot",
    "anthropic",
    "deepseek",
    "minimax",
    "zhipuai",
)


@dataclass(slots=True)
class _RequestOptions:
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    thinking: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None
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
class OpenAICompatibleLLM(LLMClient):
    endpoint_targets: list[EndpointTarget]
    timeout_seconds: float = 90.0
    max_retries_per_endpoint: int = 3
    backoff_seconds: float = 2.0
    randomize_endpoints: bool = True
    _preferred_endpoint_id: str | None = field(default=None, init=False, repr=False)

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
    ) -> LLMResponse:
        if not self.endpoint_targets:
            raise RuntimeError("No endpoint targets configured")

        message_payload = cast(list[ChatCompletionMessageParam], [msg.to_openai_message() for msg in messages])
        tool_payload = self._build_tool_payload(tools)

        ordered_targets = self._ordered_targets()
        errors: list[str] = []
        last_error: Exception | None = None

        for target in ordered_targets:
            selected_model = target.model_id or model
            should_stream = self._should_use_stream(selected_model)
            request_options = self._resolve_request_options(
                selected_model,
                stream=should_stream,
                endpoint_type=target.endpoint_type,
            )
            request_messages = self._prepare_messages_for_model(message_payload, request_options.model)

            for attempt in range(1, self.max_retries_per_endpoint + 1):
                try:
                    client = OpenAI(api_key=target.api_key, base_url=target.api_base, timeout=self.timeout_seconds)
                    if should_stream:
                        response = self._stream_completion(
                            client=client,
                            options=request_options,
                            messages=request_messages,
                            tool_payload=tool_payload,
                        )
                    else:
                        response = self._non_stream_completion(
                            client=client,
                            options=request_options,
                            messages=request_messages,
                            tool_payload=tool_payload,
                        )

                    response.raw["used_endpoint_id"] = target.endpoint_id
                    response.raw["used_model_id"] = request_options.model
                    response.raw["stream_mode"] = should_stream
                    self._preferred_endpoint_id = target.endpoint_id
                    return response

                except AuthenticationError as exc:
                    last_error = exc
                    errors.append(f"{target.endpoint_id}: authentication failed")
                    break
                except RateLimitError as exc:
                    last_error = exc
                    errors.append(f"{target.endpoint_id}: rate limited")
                    self._sleep_backoff(attempt)
                    break
                except (APITimeoutError, APIConnectionError) as exc:
                    last_error = exc
                    errors.append(f"{target.endpoint_id}: network timeout/connection error (attempt {attempt})")
                    if attempt < self.max_retries_per_endpoint:
                        self._sleep_backoff(attempt)
                        continue
                    break
                except APIStatusError as exc:
                    last_error = exc
                    status = exc.status_code
                    errors.append(f"{target.endpoint_id}: status {status} (attempt {attempt})")
                    if status in {429, 500, 502, 503, 504, 408} and attempt < self.max_retries_per_endpoint:
                        self._sleep_backoff(attempt)
                        continue
                    break
                except Exception as exc:
                    last_error = exc
                    errors.append(f"{target.endpoint_id}: unexpected {type(exc).__name__} (attempt {attempt})")
                    if attempt < self.max_retries_per_endpoint:
                        self._sleep_backoff(attempt)
                        continue
                    break

        details = "; ".join(errors) if errors else "no attempts made"
        raise RuntimeError(f"All endpoints failed: {details}") from last_error

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
                continue
            payload.append({"type": "function", "function": schema})
        return payload

    @staticmethod
    def _should_use_stream(model: str) -> bool:
        if model in _STREAM_MODEL_EXACT:
            return True
        return model.startswith(_STREAM_MODEL_PREFIXES)

    @staticmethod
    def _prepare_messages_for_model(
        messages: list[ChatCompletionMessageParam],
        model: str,
    ) -> list[ChatCompletionMessageParam]:
        # MiniMax OpenAI-compatible endpoint validates message roles strictly and
        # rejects requests containing multiple system-role turns.
        minimax_strict_system = model.startswith("MiniMax")
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

    def _resolve_request_options(self, model: str, *, stream: bool, endpoint_type: str | None) -> _RequestOptions:
        resolved_model = model
        temperature: float | None = None
        max_tokens: int | None = None
        thinking: dict[str, Any] | None = None
        reasoning_effort: str | None = None
        extra_body: dict[str, Any] | None = None

        if resolved_model in _DEEPSEEK_REASONING_MODELS:
            temperature = 0.6
        elif resolved_model in _CLAUDE_THINKING_MODELS:
            resolved_model = resolved_model.removesuffix("-thinking")
            thinking = {"type": "enabled", "budget_tokens": 16000}
            temperature = 1.0
            max_tokens = 20000

        if resolved_model in ("o3-mini-high", "o4-mini-high"):
            reasoning_effort = "high"
            resolved_model = resolved_model.removesuffix("-high")

        if resolved_model.startswith("gpt-5") and resolved_model.endswith("-high"):
            reasoning_effort = "high"
            resolved_model = resolved_model.removesuffix("-high")

        if stream and resolved_model.startswith("qwen3"):
            if resolved_model.endswith("-thinking"):
                if resolved_model not in _QWEN_THINKING_KEEP_SUFFIX_MODELS:
                    resolved_model = resolved_model.removesuffix("-thinking")
                extra_body = {"enable_thinking": True}
            else:
                extra_body = {"enable_thinking": False}

        if resolved_model.startswith(("glm-4.", "glm-5")) and resolved_model.endswith("-thinking"):
            resolved_model = resolved_model.removesuffix("-thinking")
            extra_body = {"thinking": {"type": "enabled"}, "tool_stream": True} if stream else {"thinking": {"type": "enabled"}}

        if resolved_model.startswith("gemini-2.5"):
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

        is_gemini_3_model = resolved_model.startswith("gemini-3")
        if is_gemini_3_model:
            if temperature is None:
                temperature = 1.0
            if resolved_model in {"gemini-3-pro", "gemini-3-flash"}:
                resolved_model = f"{resolved_model}-preview"
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

        return _RequestOptions(
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            is_gemini_3_model=is_gemini_3_model,
            tool_call_incremental=self._tool_call_incremental_enabled(
                model=resolved_model,
                endpoint_type=endpoint_type,
            ),
        )

    @staticmethod
    def _tool_call_incremental_enabled(*, model: str, endpoint_type: str | None) -> bool:
        if model in _TOOL_CALL_INCREMENTAL_MODELS or model.startswith("qwen3"):
            return True

        normalized_endpoint = (endpoint_type or "").strip().lower()
        if not normalized_endpoint or normalized_endpoint == "default":
            return True

        return normalized_endpoint.startswith(_TOOL_CALL_INCREMENTAL_ENDPOINT_PREFIXES)

    @staticmethod
    def _build_request_payload(
        *,
        options: _RequestOptions,
        messages: list[ChatCompletionMessageParam],
        tool_payload: list[dict[str, Any]],
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": options.model,
            "messages": messages,
        }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        if tool_payload:
            payload["tools"] = tool_payload
            payload["tool_choice"] = "auto"

        if options.temperature is not None:
            payload["temperature"] = options.temperature
        if options.max_tokens is not None:
            payload["max_tokens"] = options.max_tokens
        if options.thinking is not None:
            payload["thinking"] = options.thinking
        if options.reasoning_effort is not None:
            payload["reasoning_effort"] = options.reasoning_effort
        if options.extra_body is not None:
            payload["extra_body"] = options.extra_body

        return payload

    def _non_stream_completion(
        self,
        *,
        client: OpenAI,
        options: _RequestOptions,
        messages: list[ChatCompletionMessageParam],
        tool_payload: list[dict[str, Any]],
    ) -> LLMResponse:
        payload = self._build_request_payload(
            options=options,
            messages=messages,
            tool_payload=tool_payload,
            stream=False,
        )

        response = cast(Any, client.chat.completions.create)(**payload)
        choice = response.choices[0]
        assistant_message = choice.message

        parsed_tool_calls = self._parse_non_stream_tool_calls(assistant_message.tool_calls)
        reasoning_content = self._extract_reasoning_content(getattr(assistant_message, "reasoning_content", None))
        if not reasoning_content:
            reasoning_content = self._extract_reasoning_content(getattr(assistant_message, "reasoning", None))

        raw_payload = response.model_dump(exclude_none=True)
        if reasoning_content:
            raw_payload["reasoning_content"] = reasoning_content

        return LLMResponse(
            content=self._extract_content(assistant_message.content),
            tool_calls=parsed_tool_calls,
            raw=raw_payload,
        )

    def _stream_completion(
        self,
        *,
        client: OpenAI,
        options: _RequestOptions,
        messages: list[ChatCompletionMessageParam],
        tool_payload: list[dict[str, Any]],
    ) -> LLMResponse:
        payload = self._build_request_payload(
            options=options,
            messages=messages,
            tool_payload=tool_payload,
            stream=True,
        )

        stream = cast(Any, client.chat.completions.create)(**payload)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_call_parts: dict[str, dict[str, Any]] = {}
        last_active_tool_call_id: str | None = None
        usage_dump: dict[str, Any] | None = None

        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None and hasattr(usage, "model_dump"):
                usage_dump = usage.model_dump(exclude_none=True)

            chunk_reasoning = self._extract_reasoning_content(getattr(chunk, "reasoning_content", None))
            if chunk_reasoning:
                reasoning_parts.append(chunk_reasoning)

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            for choice in choices:
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                text = self._extract_content(getattr(delta, "content", None))
                if text:
                    content_parts.append(text)

                reasoning_text = self._extract_reasoning_content(getattr(delta, "reasoning_content", None))
                if not reasoning_text:
                    reasoning_text = self._extract_reasoning_content(getattr(delta, "reasoning", None))
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)

                for tool_call_index, tool_delta in enumerate(getattr(delta, "tool_calls", None) or []):
                    last_active_tool_call_id = self._accumulate_tool_call_delta(
                        tool_call_parts=tool_call_parts,
                        tool_delta=tool_delta,
                        default_index=tool_call_index,
                        last_active_tool_call_id=last_active_tool_call_id,
                        incremental=options.tool_call_incremental,
                        keep_extra_content=options.is_gemini_3_model,
                    )

        parsed_tool_calls: list[ToolCall] = []
        tool_call_extra_content: dict[str, Any] = {}
        for _, slot in sorted(tool_call_parts.items(), key=self._tool_call_sort_key):
            name = str(slot.get("name", "")).strip()
            if not name:
                continue
            raw_arguments = str(slot.get("arguments", ""))
            tool_id = str(slot.get("id") or f"call_{uuid.uuid4().hex[:12]}")
            parsed_tool_calls.append(
                ToolCall(
                    id=tool_id,
                    name=name,
                    arguments=self._parse_arguments(raw_arguments),
                )
            )
            if options.is_gemini_3_model and "extra_content" in slot:
                tool_call_extra_content[tool_id] = slot["extra_content"]

        normalized = self._normalize_tool_calls(parsed_tool_calls)
        raw_payload: dict[str, Any] = {"usage": usage_dump or {}, "stream_collected": True}
        if reasoning_parts:
            raw_payload["reasoning_content"] = "".join(reasoning_parts)
        if tool_call_extra_content:
            raw_payload["tool_call_extra_content"] = tool_call_extra_content

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=normalized,
            raw=raw_payload,
        )

    @staticmethod
    def _tool_call_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        call_id, _ = item
        index_text, _, _ = call_id.partition("_")
        try:
            index = int(index_text)
        except ValueError:
            index = 10**9
        return index, call_id

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
        function = getattr(tool_delta, "function", None)
        if function is None:
            return last_active_tool_call_id

        index_raw = getattr(tool_delta, "index", None)
        index = index_raw if isinstance(index_raw, int) else default_index

        name_raw = getattr(function, "name", None)
        name = name_raw.strip() if isinstance(name_raw, str) else ""
        arguments_raw = getattr(function, "arguments", None)
        arguments = arguments_raw if isinstance(arguments_raw, str) else ""

        delta_id_raw = getattr(tool_delta, "id", None)
        delta_id = delta_id_raw.strip() if isinstance(delta_id_raw, str) else ""

        extra_content = getattr(tool_delta, "extra_content", None) if keep_extra_content else None

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

        if not arguments:
            return last_active_tool_call_id

        target_id = last_active_tool_call_id
        if target_id is None and delta_id:
            for existing_id, slot in tool_call_parts.items():
                if slot.get("id") == delta_id:
                    target_id = existing_id
                    break

        if target_id and target_id in tool_call_parts:
            slot = tool_call_parts[target_id]
            slot["arguments"] = self._merge_tool_arguments(
                existing=str(slot.get("arguments", "")),
                incoming=arguments,
                incremental=incremental,
            )

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
            function_call = getattr(call, "function", None)
            if function_call is None:
                continue

            name = getattr(function_call, "name", None)
            if not isinstance(name, str) or not name.strip():
                continue

            arguments_raw = getattr(function_call, "arguments", None)
            parsed_tool_calls.append(
                ToolCall(
                    id=getattr(call, "id", None) or f"call_{uuid.uuid4().hex[:12]}",
                    name=name,
                    arguments=self._parse_arguments(arguments_raw if isinstance(arguments_raw, str) else None),
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

    def _sleep_backoff(self, attempt: int) -> None:
        jitter = random.uniform(0.0, 0.5)
        sleep_seconds = self.backoff_seconds * attempt + jitter
        time.sleep(sleep_seconds)
