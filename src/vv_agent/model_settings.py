from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class RetrySettings:
    max_attempts: int = 3
    backoff_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class ModelSettings:
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tool_choice: Literal["auto", "required", "none"] | str | None = None
    parallel_tool_calls: bool | None = None
    reasoning: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    timeout_seconds: float | None = None
    retry: RetrySettings | None = None
    extra_headers: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None
    extra_args: dict[str, Any] | None = None

    def resolve(self, override: ModelSettings | None) -> ModelSettings:
        if override is None:
            return self

        values: dict[str, Any] = {}
        for item in fields(self):
            current = getattr(self, item.name)
            incoming = getattr(override, item.name)
            if item.name in {"extra_headers", "extra_body", "extra_args"}:
                values[item.name] = self._merge_dicts(current, incoming)
            else:
                values[item.name] = incoming if incoming is not None else current
        return ModelSettings(**values)

    @staticmethod
    def _merge_dicts(base: dict[Any, Any] | None, override: dict[Any, Any] | None) -> dict[Any, Any] | None:
        if base is None and override is None:
            return None
        merged: dict[Any, Any] = {}
        if base:
            merged.update(base)
        if override:
            merged.update(override)
        return merged
