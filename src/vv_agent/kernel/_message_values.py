"""Message values, artifact references and compaction evidence validation."""

from __future__ import annotations

import json
import posixpath
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from ._canonical_json import MAX_WIRE_INTEGER, canonical_json_bytes, validate_sha256

Role = Literal["system", "user", "assistant", "tool"]


_MESSAGE_FIELDS = frozenset(
    {
        "role",
        "content",
        "name",
        "tool_call_id",
        "tool_calls",
        "reasoning_content",
        "image_url",
        "metadata",
        "artifact_ref",
    }
)


COMPACTION_METADATA_KEY = "_vv_agent_compaction"


def validate_compaction_metadata(metadata: dict[str, Any]) -> None:
    if COMPACTION_METADATA_KEY not in metadata:
        return
    manifest = metadata[COMPACTION_METADATA_KEY]
    if not isinstance(manifest, dict) or set(manifest) != {"artifacts", "cursors"}:
        raise ValueError("invalid compaction evidence manifest")
    for key, pointer, decoder in (
        ("artifacts", "artifact_ref", ToolArtifactRef.from_dict),
        ("cursors", "cursor", ToolResultCursor.from_dict),
    ):
        records = manifest[key]
        if not isinstance(records, list):
            raise ValueError("compaction evidence must be arrays")
        for record in records:
            if not isinstance(record, dict) or set(record) != {"tool_call_id", "tool_name", "arguments", pointer}:
                raise ValueError("invalid compaction evidence record fields")
            for identity in ("tool_call_id", "tool_name"):
                if not isinstance(record[identity], str) or not record[identity]:
                    raise ValueError("invalid compaction evidence identity")
            raw = record["arguments"]
            if not isinstance(raw, str):
                raise ValueError("compaction evidence arguments must be object JSON")
            arguments = json.loads(raw)
            if not isinstance(arguments, dict) or canonical_json_bytes(arguments).decode() != raw:
                raise ValueError("compaction evidence arguments must be canonical object JSON")
            decoder(record[pointer])


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None
    image_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_ref: ToolArtifactRef | None = None

    def to_openai_message(self, *, include_reasoning_content: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.role == "assistant" and self.tool_calls:
            payload["tool_calls"] = self.tool_calls
            if not self.content:
                payload["content"] = None
        if include_reasoning_content and self.role == "assistant" and self.reasoning_content:
            payload["reasoning_content"] = self.reasoning_content
        if self.role == "user" and self.image_url:
            content_blocks: list[dict[str, Any]] = []
            if self.content:
                content_blocks.append({"type": "text", "text": self.content})
            content_blocks.append({"type": "image_url", "image_url": {"url": self.image_url}})
            payload["content"] = content_blocks
        return payload

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content
        if self.image_url is not None:
            d["image_url"] = self.image_url
        validate_compaction_metadata(self.metadata)
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        if self.artifact_ref is not None:
            d["artifact_ref"] = self.artifact_ref.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        from ._message_codec import _decode_canonical_message

        if not isinstance(data, dict):
            raise TypeError("Message payload must be a dict")
        return _decode_canonical_message(data)


_ARTIFACT_PATH_RE = re.compile(r"^\.vv-agent/artifacts/(?:[A-Za-z0-9][A-Za-z0-9._-]{0,127}/)*[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ToolArtifactRef:
    path: str
    media_type: str
    encoding: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        try:
            path_bytes = self.path.encode("utf-8") if isinstance(self.path, str) else b""
        except UnicodeEncodeError as exc:
            raise ValueError("artifact_path_invalid") from exc
        if not path_bytes or len(path_bytes) > 512 or not _ARTIFACT_PATH_RE.fullmatch(self.path):
            raise ValueError("artifact_path_invalid")
        if self.media_type != "text/plain" or self.encoding != "utf-8":
            raise ValueError("tool_result_invalid: artifact must use text/plain with utf-8 encoding")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or not 0 <= self.size_bytes <= MAX_WIRE_INTEGER
        ):
            raise ValueError("tool_result_invalid: artifact size_bytes must be a JSON-safe non-negative integer")
        try:
            validate_sha256(self.sha256, "tool artifact sha256")
        except ValueError as exc:
            raise ValueError("tool_result_invalid: artifact sha256 is invalid") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "media_type": self.media_type,
            "encoding": self.encoding,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ToolArtifactRef:
        if not isinstance(payload, dict) or set(payload) != {"path", "media_type", "encoding", "size_bytes", "sha256"}:
            raise ValueError("tool_result_invalid: artifact fields do not match the current wire")
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ValueError("tool_result_invalid: artifact fields are invalid") from exc


@dataclass(frozen=True, slots=True)
class ToolResultCursor:
    kind: str
    path: str
    offset_chars: int
    sha256: str

    def __post_init__(self) -> None:
        if self.kind != "read_file":
            raise ValueError("tool_result_invalid: cursor kind must be read_file")
        try:
            path_bytes = self.path.encode("utf-8") if isinstance(self.path, str) else b""
        except UnicodeEncodeError as exc:
            raise ValueError("tool_result_invalid: cursor path must be a normalized caller-visible path") from exc
        if not path_bytes or "\x00" in self.path or "\\" in self.path or posixpath.normpath(self.path) != self.path:
            raise ValueError("tool_result_invalid: cursor path must be a normalized caller-visible path")
        if (
            isinstance(self.offset_chars, bool)
            or not isinstance(self.offset_chars, int)
            or not 0 <= self.offset_chars <= MAX_WIRE_INTEGER
        ):
            raise ValueError("tool_result_invalid: cursor offset_chars must be a JSON-safe non-negative integer")
        try:
            validate_sha256(self.sha256, "tool result cursor sha256")
        except ValueError as exc:
            raise ValueError("tool_result_invalid: cursor sha256 is invalid") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "offset_chars": self.offset_chars,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ToolResultCursor:
        if not isinstance(payload, dict) or set(payload) != {"kind", "path", "offset_chars", "sha256"}:
            raise ValueError("tool_result_invalid: cursor fields do not match the current wire")
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ValueError("tool_result_invalid: cursor fields are invalid") from exc
