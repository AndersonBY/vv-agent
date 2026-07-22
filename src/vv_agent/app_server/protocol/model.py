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
    supports_tools: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.id, "supportsTools": self.supports_tools}
        if self.provider is not None:
            payload["provider"] = self.provider
        if self.display_name is not None:
            payload["displayName"] = self.display_name
        if self.context_length is not None:
            payload["contextLength"] = self.context_length
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class ModelListResponse:
    models: list[ModelSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"models": [model.to_dict() for model in self.models]}
