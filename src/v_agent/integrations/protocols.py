from __future__ import annotations

from typing import Protocol


class DocumentIntegration(Protocol):
    def enabled(self) -> bool:
        ...


class WorkflowIntegration(Protocol):
    def enabled(self) -> bool:
        ...


class SkillIntegration(Protocol):
    def enabled(self) -> bool:
        ...
