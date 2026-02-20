from __future__ import annotations

from typing import Protocol


class SkillIntegration(Protocol):
    def enabled(self) -> bool:
        ...
