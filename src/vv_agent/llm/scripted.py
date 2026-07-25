from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from vv_agent.llm.base import LLMClient, LlmRequest, ScriptExhaustedError
from vv_agent.types import LLMResponse

ScriptStep = LLMResponse | Callable[[LlmRequest], LLMResponse]


@dataclass(slots=True)
class ScriptedLLM(LLMClient):
    steps: list[ScriptStep] = field(default_factory=list)

    def complete(self, request: LlmRequest) -> LLMResponse:
        if not self.steps:
            raise ScriptExhaustedError("No scripted LLM steps left.")
        step = self.steps.pop(0)
        if isinstance(step, LLMResponse):
            return step
        return step(request)

    def complete_with_stream(self, request: LlmRequest, stream_callback=None) -> LLMResponse:
        del stream_callback
        return self.complete(request)
