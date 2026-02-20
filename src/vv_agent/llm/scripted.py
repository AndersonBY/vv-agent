from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from vv_agent.llm.base import LLMClient
from vv_agent.types import LLMResponse, Message

ScriptStep = LLMResponse | Callable[[str, list[Message]], LLMResponse]


@dataclass(slots=True)
class ScriptedLLM(LLMClient):
    steps: list[ScriptStep] = field(default_factory=list)

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        del tools
        if not self.steps:
            raise RuntimeError("No scripted LLM steps left.")
        step = self.steps.pop(0)
        if callable(step):
            return step(model, messages)
        return step
