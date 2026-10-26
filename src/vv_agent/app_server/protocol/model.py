from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelListRequest:
    agent_key: str | None = None
    provider: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload: dict[str, str] = {}
        if self.agent_key is not None:
            payload["agentKey"] = self.agent_key
        if self.provider is not None:
            payload["provider"] = self.provider
        return payload


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSummary:
    id: str
    provider: str | None = None
    display_name: str | None = None
    context_length: int | None = None
    supports_tools: bool | None = None
    max_output_tokens: int | None = None
    supports_vision: bool | None = None
    input_modalities: list[str] | None = None
    supports_reasoning: bool | None = None
    supported_reasoning_efforts: list[str] | None = None
    default_reasoning_effort: str | None = None
    supports_prompt_cache: bool | None = None
    supports_continuation: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "supportsTools": self.supports_tools,
            "contextLength": self.context_length,
            "maxOutputTokens": self.max_output_tokens,
            "supportsVision": self.supports_vision,
            "inputModalities": self.input_modalities,
            "supportsReasoning": self.supports_reasoning,
            "supportedReasoningEfforts": self.supported_reasoning_efforts,
            "defaultReasoningEffort": self.default_reasoning_effort,
            "supportsPromptCache": self.supports_prompt_cache,
            "supportsContinuation": self.supports_continuation,
        }
        if self.provider is not None:
            payload["provider"] = self.provider
        if self.display_name is not None:
            payload["displayName"] = self.display_name
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class ModelListResponse:
    models: list[ModelSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"models": [model.to_dict() for model in self.models]}
