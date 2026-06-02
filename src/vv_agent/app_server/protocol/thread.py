from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ThreadStartParams:
    agent_key: str
    cwd: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"agentKey": self.agent_key}
        if self.cwd is not None:
            payload["cwd"] = self.cwd
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload
