from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SkillProperties:
    """Metadata fields defined by the Agent Skills specification."""

    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "description": self.description,
        }
        if self.license is not None:
            payload["license"] = self.license
        if self.compatibility is not None:
            payload["compatibility"] = self.compatibility
        if self.allowed_tools is not None:
            payload["allowed-tools"] = self.allowed_tools
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class LoadedSkill:
    """Fully loaded skill record used by runtime activation."""

    properties: SkillProperties
    skill_md_path: Path
    instructions: str

    @property
    def name(self) -> str:
        return self.properties.name
