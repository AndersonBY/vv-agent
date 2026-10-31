from typing import TYPE_CHECKING

from vv_agent.llm.base import LLMClient, LlmError, LlmRequest, LlmRequestError, ScriptExhaustedError
from vv_agent.llm.scripted import ScriptedLLM

if TYPE_CHECKING:
    from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient

__all__ = [
    "EndpointTarget",
    "LLMClient",
    "LlmError",
    "LlmRequest",
    "LlmRequestError",
    "ScriptExhaustedError",
    "ScriptedLLM",
    "VvLlmClient",
]


def __getattr__(name: str):
    if name in {"EndpointTarget", "VvLlmClient"}:
        from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient

        globals().update(EndpointTarget=EndpointTarget, VvLlmClient=VvLlmClient)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
