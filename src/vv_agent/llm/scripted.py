from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from vv_agent.llm.base import LLMClient, LlmRequest, ScriptExhaustedError
from vv_agent.model_settings import ModelSettings
from vv_agent.types import LLMResponse, Message

ScriptStep = LLMResponse | Callable[[LlmRequest], LLMResponse]


@dataclass(slots=True)
class ScriptedLLM(LLMClient):
    steps: list[ScriptStep] = field(default_factory=list)

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback=None,
        model_settings: ModelSettings | None = None,
        request_metadata: dict[str, object] | None = None,
    ) -> LLMResponse:
        return self.complete_request(
            LlmRequest(
                model=model,
                messages=list(messages),
                tools=list(tools),
                metadata=dict(request_metadata or {}),
                model_settings=model_settings,
            ),
            stream_callback=stream_callback,
        )

    def complete_request(self, request: LlmRequest, *, stream_callback=None) -> LLMResponse:
        del stream_callback
        if not self.steps:
            raise ScriptExhaustedError("No scripted LLM steps left.")
        step = self.steps.pop(0)
        if isinstance(step, LLMResponse):
            return step
        return step(request)
