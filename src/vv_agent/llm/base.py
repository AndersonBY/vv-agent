from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from vv_agent.model_settings import ModelSettings
from vv_agent.types import LLMResponse, Message


class LLMClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
        model_settings: ModelSettings | None = None,
    ) -> LLMResponse:
        ...
