from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from vv_agent.events import MemoryCompactCompleted, MemoryCompactStarted


@dataclass(frozen=True, slots=True)
class MemorySearchRequest:
    query: str
    limit: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    content: str = ""
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemorySaveRequest:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemorySaveResult:
    memory_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryProviderResult:
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryProvider(Protocol):
    def search(self, request: MemorySearchRequest) -> list[MemorySearchResult]:
        ...

    def save(self, request: MemorySaveRequest) -> MemorySaveResult:
        ...

    def before_compact(self, event: MemoryCompactStarted) -> MemoryProviderResult:
        ...

    def after_compact(self, event: MemoryCompactCompleted) -> None:
        ...
