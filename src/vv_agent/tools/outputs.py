from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolOutputText:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolOutputJson:
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolOutputImage:
    url: str | None = None
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolOutputFile:
    path: str
    mime_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolOutputError:
    message: str
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


ToolOutput = ToolOutputText | ToolOutputJson | ToolOutputImage | ToolOutputFile | ToolOutputError
