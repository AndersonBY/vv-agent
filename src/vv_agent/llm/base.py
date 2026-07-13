from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from vv_agent.model_settings import ModelSettings
from vv_agent.types import LLMResponse, Message


@dataclass(slots=True)
class LlmRequest:
    model: str
    messages: list[Message]
    tools: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    model_settings: ModelSettings | None = None


class LlmError(RuntimeError):
    """Base error raised by model clients."""


class ScriptExhaustedError(LlmError):
    pass


class LlmRequestError(LlmError):
    pass


class LLMClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
        model_settings: ModelSettings | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        ...


def complete_llm_request(
    client: LLMClient,
    request: LlmRequest,
    *,
    stream_callback: Callable[[dict[str, Any]], None] | None = None,
) -> LLMResponse:
    complete_request = getattr(client, "complete_request", None)
    if callable(complete_request):
        return complete_request(request, stream_callback=stream_callback)
    return client.complete(
        model=request.model,
        messages=request.messages,
        tools=request.tools,
        stream_callback=stream_callback,
        model_settings=request.model_settings,
        request_metadata=request.metadata,
    )
