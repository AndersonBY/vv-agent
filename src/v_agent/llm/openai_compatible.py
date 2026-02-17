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
        should_stream = self._should_use_stream(model)

        ordered_targets = self._ordered_targets()
        errors: list[str] = []
        last_error: Exception | None = None

        for target in ordered_targets:
            selected_model = target.model_id or model

            for attempt in range(1, self.max_retries_per_endpoint + 1):
                try:
                    client = OpenAI(api_key=target.api_key, base_url=target.api_base, timeout=self.timeout_seconds)
                    if should_stream:
                        response = self._stream_completion(
                            client=client,
                            model=selected_model,
                            messages=message_payload,
                            tool_payload=tool_payload,
                        )
                    else:
                        response = self._non_stream_completion(
                            client=client,
                            model=selected_model,
                            messages=message_payload,
                            tool_payload=tool_payload,
                        )

                    response.raw["used_endpoint_id"] = target.endpoint_id
                    response.raw["used_model_id"] = selected_model
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

    def _non_stream_completion(
        self,
        *,
        client: OpenAI,
        model: str,
        messages: list[ChatCompletionMessageParam],
        tool_payload: list[dict[str, Any]],
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tool_payload:
            payload["tools"] = tool_payload
            payload["tool_choice"] = "auto"

        response = cast(Any, client.chat.completions.create)(**payload)
        choice = response.choices[0]
        assistant_message = choice.message

        parsed_tool_calls = self._parse_non_stream_tool_calls(assistant_message.tool_calls)
        reasoning_content = self._extract_reasoning_content(getattr(assistant_message, "reasoning_content", None))
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
        model: str,
        messages: list[ChatCompletionMessageParam],
        tool_payload: list[dict[str, Any]],
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tool_payload:
            payload["tools"] = tool_payload
            payload["tool_choice"] = "auto"

        stream = cast(Any, client.chat.completions.create)(**payload)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_call_parts: dict[int, dict[str, Any]] = {}
        usage_dump: dict[str, Any] | None = None

        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None and hasattr(usage, "model_dump"):
                usage_dump = usage.model_dump(exclude_none=True)

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            delta = getattr(choices[0], "delta", None)
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

            for tool_delta in getattr(delta, "tool_calls", None) or []:
                index = getattr(tool_delta, "index", None)
                if index is None:
                    index = len(tool_call_parts)

                slot = tool_call_parts.setdefault(
                    int(index),
                    {
                        "id": "",
                        "name": "",
                        "arguments": [],
                    },
                )

                delta_id = getattr(tool_delta, "id", None)
                if isinstance(delta_id, str) and delta_id:
                    slot["id"] = delta_id

                function = getattr(tool_delta, "function", None)
                if function is not None:
                    name = getattr(function, "name", None)
                    if isinstance(name, str) and name:
                        slot["name"] = name
                    arguments = getattr(function, "arguments", None)
                    if isinstance(arguments, str) and arguments:
                        slot["arguments"].append(arguments)

        parsed_tool_calls: list[ToolCall] = []
        for _, slot in sorted(tool_call_parts.items(), key=lambda item: item[0]):
            name = str(slot.get("name", "")).strip()
            if not name:
                continue
            raw_arguments = "".join(slot.get("arguments", []))
            parsed_tool_calls.append(
                ToolCall(
                    id=str(slot.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                    name=name,
                    arguments=self._parse_arguments(raw_arguments),
                )
            )

        normalized = self._normalize_tool_calls(parsed_tool_calls)
        raw_payload: dict[str, Any] = {"usage": usage_dump or {}, "stream_collected": True}
        if reasoning_parts:
            raw_payload["reasoning_content"] = "".join(reasoning_parts)
        return LLMResponse(
            content="".join(content_parts),
            tool_calls=normalized,
            raw=raw_payload,
        )

    def _parse_non_stream_tool_calls(self, tool_calls_raw: Any) -> list[ToolCall]:
        parsed_tool_calls: list[ToolCall] = []
        for call in tool_calls_raw or []:
            function_call = getattr(call, "function", None)
            if function_call is None:
                continue

            parsed_tool_calls.append(
                ToolCall(
                    id=getattr(call, "id", None) or f"call_{uuid.uuid4().hex[:12]}",
                    name=function_call.name,
                    arguments=self._parse_arguments(function_call.arguments),
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
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
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
            return "".join(parts)
        return ""

    def _sleep_backoff(self, attempt: int) -> None:
        jitter = random.uniform(0.0, 0.5)
        sleep_seconds = self.backoff_seconds * attempt + jitter
        time.sleep(sleep_seconds)
