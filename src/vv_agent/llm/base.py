from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from vv_agent.model_settings import ModelSettings
from vv_agent.prompt import PromptBundle
from vv_agent.types import LLMResponse, Message


@dataclass(slots=True)
class LlmRequest:
    model: str
    messages: list[Message]
    tools: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    model_settings: ModelSettings | None = None
    prompt_bundle: PromptBundle | None = None


class LlmError(RuntimeError):
    """Base error raised by model clients."""


class ScriptExhaustedError(LlmError):
    pass


class LlmRequestError(LlmError):
    pass


class LLMClient(Protocol):
    def complete(self, request: LlmRequest) -> LLMResponse:
        ...

    def complete_with_stream(
        self,
        request: LlmRequest,
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> LLMResponse:
        ...
