from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
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


class TraceSink(TraceProcessor, Protocol):
    def flush(self) -> None:
        ...


class JsonlTraceExporter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._lock = Lock()

    def _write(self, event: str, span: Span) -> None:
        payload = {"event": event, "timestamp": time.time(), "span": span.to_dict()}
        with self._lock:
            self._file.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    def on_span_start(self, span: Span) -> None:
        self._write("span_start", span)

    def on_span_end(self, span: Span) -> None:
        self._write("span_end", span)

    def flush(self) -> None:
        with self._lock:
            self._file.flush()
