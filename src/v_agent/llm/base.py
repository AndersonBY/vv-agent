from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from v_agent.types import LLMResponse, Message


class LLMClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        ...
