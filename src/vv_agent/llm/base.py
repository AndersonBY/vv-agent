from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from vv_agent.model_settings import ModelSettings
from vv_agent.prompt import PromptBundle
from vv_agent.types import LLMResponse, Message

if TYPE_CHECKING:
    from vv_agent.kernel._content import Principal


@dataclass(slots=True)
class LlmRequest:
    model: str
    messages: list[Message]
    tools: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    model_settings: ModelSettings | None = None
    prompt_bundle: PromptBundle | None = None
    principal: Principal | None = None

    def __post_init__(self) -> None:
        from vv_agent.kernel._content import Principal

        if self.principal is not None and not isinstance(self.principal, Principal):
            raise TypeError("principal must be a Principal or None")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "principal" and hasattr(self, name):
            raise AttributeError("principal is read-only")
        object.__setattr__(self, name, value)


class LlmError(RuntimeError):
    """Base error raised by model clients."""


class ScriptExhaustedError(LlmError):
    pass


class LlmRequestError(LlmError):
    pass


class LLMClient(Protocol):
    def complete(self, request: LlmRequest) -> LLMResponse: ...

    def complete_with_stream(
        self,
        request: LlmRequest,
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> LLMResponse: ...
