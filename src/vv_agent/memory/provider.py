from __future__ import annotations

import warnings
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
    def search(self, request: MemorySearchRequest) -> list[MemorySearchResult]: ...

    def save(self, request: MemorySaveRequest) -> MemorySaveResult: ...

    def before_compact(self, event: MemoryCompactStarted) -> MemoryProviderResult: ...

    def after_compact(self, event: MemoryCompactCompleted) -> None: ...


def _call_before_memory_providers(
    providers: list[MemoryProvider],
    event: MemoryCompactStarted,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for index, provider in enumerate(providers):
        provider_name = _memory_provider_name(provider, index=index, existing=results)
        try:
            result = provider.before_compact(event)
        except Exception as exc:
            _record_memory_provider_error(
                provider_name=provider_name,
                stage="before_compact",
                error=exc,
                errors=errors,
            )
            continue
        if isinstance(result, MemoryProviderResult) and result.metadata:
            results[provider_name] = dict(result.metadata)
    return _memory_provider_metadata(results=results, errors=errors)


def _call_after_memory_providers(
    providers: list[MemoryProvider],
    event: MemoryCompactCompleted,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    for index, provider in enumerate(providers):
        provider_name = _memory_provider_name(provider, index=index, existing={})
        try:
            provider.after_compact(event)
        except Exception as exc:
            _record_memory_provider_error(
                provider_name=provider_name,
                stage="after_compact",
                error=exc,
                errors=errors,
            )
    return _memory_provider_metadata(results={}, errors=errors)


def _record_memory_provider_error(
    *,
    provider_name: str,
    stage: str,
    error: Exception,
    errors: list[dict[str, str]],
) -> None:
    warnings.warn(
        f"Memory provider {provider_name} {stage} failed: {error}",
        RuntimeWarning,
        stacklevel=3,
    )
    errors.append(
        {
            "provider": provider_name,
            "stage": stage,
            "error": str(error),
            "error_type": type(error).__name__,
        }
    )


def _memory_provider_metadata(
    *,
    results: dict[str, dict[str, Any]],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if results:
        metadata["memory_provider_results"] = results
    if errors:
        metadata["memory_provider_errors"] = errors
    return metadata


def _memory_provider_name(
    provider: MemoryProvider,
    *,
    index: int,
    existing: dict[str, Any],
) -> str:
    base_name = provider.__class__.__name__
    if base_name not in existing:
        return base_name
    return f"{base_name}#{index + 1}"
