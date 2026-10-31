from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

_OMITTED = object()


@dataclass(frozen=True, slots=True)
class ToolResultRef:
    session_id: str
    turn_id: str
    operation_id: str
    attempt: int
    tool_call_id: str
    seq: int
    record_digest: str

    def to_dict(self):
        return dict(
            zip(
                ("sessionId", "turnId", "operationId", "attempt", "toolCallId", "seq", "recordDigest"),
                asdict(self).values(),
                strict=True,
            )
        )

    @classmethod
    def from_dict(cls, value):
        names = ("sessionId", "turnId", "operationId", "attempt", "toolCallId", "seq", "recordDigest")
        if not isinstance(value, dict) or set(value) != set(names):
            raise ValueError("invalid_result_ref")
        for key in ("sessionId", "turnId", "operationId", "toolCallId", "recordDigest"):
            if not isinstance(value[key], str) or not value[key] or len(value[key].encode()) > 4096:
                raise ValueError("invalid_result_ref")
        for key in ("seq", "attempt"):
            if type(value[key]) is not int or not 1 <= value[key] <= 2**53 - 1:
                raise ValueError("invalid_result_ref")
        if len(value["recordDigest"]) != 64 or any(c not in "0123456789abcdef" for c in value["recordDigest"]):
            raise ValueError("invalid_result_ref")
        return cls(*(value[name] for name in names))


@dataclass(frozen=True, slots=True)
class ToolResultSummary:
    kind: str
    preview: str
    byte_length: int
    truncated: bool
    ref: ToolResultRef
    resources: list[dict[str, Any]]
    context: str = "normal"
    value: Any = _OMITTED
    cursor: dict[str, Any] | None = None

    def to_dict(self):
        result = {
            "kind": self.kind,
            "preview": self.preview,
            "byteLength": self.byte_length,
            "truncated": self.truncated,
            "ref": self.ref.to_dict(),
            "resources": self.resources,
            "context": self.context,
        }
        if self.value is not _OMITTED:
            result["value"] = self.value
        if self.cursor is not None:
            result["cursor"] = self.cursor
        return result


@dataclass(frozen=True, slots=True)
class ToolResultReadParams:
    thread_id: str
    ref: ToolResultRef
    part: str = "content"
    offset: int = 0
    limit: int = 65_536

    def to_dict(self):
        return {
            "threadId": self.thread_id,
            "ref": self.ref.to_dict(),
            "part": self.part,
            "offset": self.offset,
            "limit": self.limit,
        }


@dataclass(frozen=True, slots=True)
class ToolResultReadResponse:
    ref: ToolResultRef
    kind: str
    context: str
    part: str
    data: str
    offset: int
    next_offset: int
    total_bytes: int
    sha256: str
    eof: bool
    encoding: str = "base64"

    def to_dict(self):
        return {
            "ref": self.ref.to_dict(),
            "kind": self.kind,
            "context": self.context,
            "part": self.part,
            "encoding": self.encoding,
            "data": self.data,
            "offset": self.offset,
            "nextOffset": self.next_offset,
            "totalBytes": self.total_bytes,
            "sha256": self.sha256,
            "eof": self.eof,
        }
