from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Handoff:
    agent: Any
    description: str = ""
    tool_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tool_name is None:
            object.__setattr__(self, "tool_name", f"transfer_to_{_slugify(getattr(self.agent, 'name', 'agent'))}")


def handoff(
    *,
    agent: Any,
    description: str = "",
    tool_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Handoff:
    return Handoff(agent=agent, description=description, tool_name=tool_name, metadata=dict(metadata or {}))


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_").lower()
    return normalized or "agent"
