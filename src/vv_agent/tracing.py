from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: f"span_{uuid.uuid4().hex}")
    parent_id: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, metadata: dict[str, Any] | None = None) -> Span:
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return replace(self, ended_at=time.time(), metadata=merged)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "started_at": self.started_at,
            "metadata": dict(self.metadata),
        }
        if self.parent_id is not None:
            payload["parent_id"] = self.parent_id
        if self.ended_at is not None:
            payload["ended_at"] = self.ended_at
        return payload


class TraceProcessor(Protocol):
    def on_span_start(self, span: Span) -> None:
        ...

    def on_span_end(self, span: Span) -> None:
        ...
