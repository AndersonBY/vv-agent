from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelListRequest:
    agent_key: str | None = None

    def to_dict(self) -> dict[str, str]:
        if self.agent_key is None:
            return {}
        return {"agentKey": self.agent_key}


@dataclass(frozen=True, slots=True)
class ModelSummary:
    id: str
    provider: str
    display_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.id, "provider": self.provider}
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
