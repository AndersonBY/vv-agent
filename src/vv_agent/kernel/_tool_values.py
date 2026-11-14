"""Tool calls, directives, statuses and complete tool-result validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ._canonical_json import MAX_WIRE_INTEGER, canonical_json_bytes, validate_sha256
from ._message_values import Message, ToolArtifactRef, ToolResultCursor


class ToolDirective(StrEnum):
    CONTINUE = "continue"
    WAIT_USER = "wait_user"
    FINISH = "finish"


class ToolResultStatus(StrEnum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    WAIT_RESPONSE = "WAIT_RESPONSE"
    RUNNING = "RUNNING"
    PENDING_COMPRESS = "PENDING_COMPRESS"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    extra_content: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "arguments": dict(self.arguments),
        }
        if self.extra_content is not None:
            payload["extra_content"] = deepcopy(self.extra_content)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        raw_extra_content = data.get("extra_content")
        extra_content = deepcopy(raw_extra_content) if isinstance(raw_extra_content, dict) else None
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=dict(data.get("arguments", {})),
            extra_content=extra_content,
        )


_TOOL_RESULT_FORBIDDEN_METADATA_KEYS = frozenset({"content", "instructions", "output", "stderr", "stdout"})


@dataclass(slots=True)
class ToolExecutionResult:
    tool_call_id: str
    content: str
    status_code: ToolResultStatus = ToolResultStatus.SUCCESS
    directive: ToolDirective = ToolDirective.CONTINUE
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    image_url: str | None = None
    image_path: str | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None
    original_bytes: int | None = None
    visible_bytes: int | None = None
    artifact: ToolArtifactRef | None = None
    cursor: ToolResultCursor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status_code, ToolResultStatus):
            self.status_code = ToolResultStatus(self.status_code)
        if not isinstance(self.directive, ToolDirective):
            self.directive = ToolDirective(self.directive)
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.tool_call_id, str) or not isinstance(self.content, str):
            raise ValueError("tool_result_invalid: tool_call_id and content must be strings")
        for value, name in (
            (self.error_code, "error_code"),
            (self.image_url, "image_url"),
            (self.image_path, "image_path"),
        ):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"tool_result_invalid: {name} must be a string when present")
        if not isinstance(self.metadata, dict) or not all(isinstance(key, str) for key in self.metadata):
            raise ValueError("tool_result_invalid: metadata must be an object with string keys")
        if "request_compaction" in self.metadata and self.metadata["request_compaction"] is not True:
            raise ValueError("tool_result_invalid: request_compaction must be true when present")
        forbidden_metadata = sorted(set(self.metadata).intersection(_TOOL_RESULT_FORBIDDEN_METADATA_KEYS))
        if forbidden_metadata:
            raise ValueError(f"tool_result_invalid: metadata contains model-visible payload keys: {forbidden_metadata}")
        if "vv_artifact" in self.metadata:
            retained = self.metadata["vv_artifact"]
            ref_keys = {"path", "media_type", "encoding", "size_bytes", "sha256"}
            if not isinstance(retained, dict) or set(retained) != ref_keys | {
                "source",
                "source_complete",
                "capture_incomplete_reason",
            }:
                raise ValueError("tool_result_invalid: invalid vv_artifact fields")
            ref = ToolArtifactRef.from_dict({key: retained[key] for key in ref_keys})
            if ref != self.artifact:
                raise ValueError("tool_result_invalid: vv_artifact does not match artifact")
            source = retained["source"]
            if (
                not isinstance(source, dict)
                or set(source) != {"operation_id", "attempt", "tool_call_id", "arguments_sha256"}
                or not isinstance(source["operation_id"], str)
                or not source["operation_id"].strip()
                or type(source["attempt"]) is not int
                or not 1 <= source["attempt"] <= MAX_WIRE_INTEGER
                or source["tool_call_id"] != self.tool_call_id
            ):
                raise ValueError("tool_result_invalid: invalid vv_artifact source")
            validate_sha256(source["arguments_sha256"], "artifact arguments_sha256")
            complete, reason = retained["source_complete"], retained["capture_incomplete_reason"]
            if type(complete) is not bool or (
                (complete and reason is not None) or (not complete and (not isinstance(reason, str) or not reason.strip()))
            ):
                raise ValueError("tool_result_invalid: invalid artifact capture completeness")
        try:
            canonical_json_bytes(self.metadata, "tool result metadata")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ValueError("tool_result_invalid: metadata must be RFC 8785 I-JSON") from exc
        if self.status_code is ToolResultStatus.SUCCESS and self.error_code is not None:
            raise ValueError("tool_result_invalid: SUCCESS results cannot contain error_code")
        recovery_values = (
            self.truncation_reason,
            self.original_bytes,
            self.visible_bytes,
            self.artifact,
            self.cursor,
        )
        if self.truncated is None:
            if any(value is not None for value in recovery_values):
                raise ValueError("tool_result_invalid: ordinary results cannot contain recovery fields")
            return
        if self.truncated is not True:
            raise ValueError("tool_result_invalid: truncated must be omitted or true")
        if self.truncation_reason not in {"output_limit", "read_limit"}:
            raise ValueError("tool_result_invalid: truncated result has an invalid truncation_reason")
        for value, name in ((self.original_bytes, "original_bytes"), (self.visible_bytes, "visible_bytes")):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_WIRE_INTEGER:
                raise ValueError(f"tool_result_invalid: {name} must be a JSON-safe non-negative integer")
        visible_bytes = len(self.content.encode("utf-8"))
        if self.visible_bytes != visible_bytes or self.original_bytes is None or self.original_bytes < visible_bytes:
            raise ValueError("tool_result_invalid: truncated result byte sizes do not match content")
        if self.truncation_reason == "output_limit":
            if not isinstance(self.artifact, ToolArtifactRef) or self.cursor is not None:
                raise ValueError("tool_result_invalid: output_limit requires only an artifact recovery pointer")
        elif not isinstance(self.cursor, ToolResultCursor) or self.artifact is not None:
            raise ValueError("tool_result_invalid: read_limit requires only a cursor recovery pointer")

    def to_tool_message(self) -> Message:
        self._validate()
        content = self.content
        if self.truncated is True:
            recovery = {
                key: value
                for key, value in self.to_dict().items()
                if key
                in {
                    "truncated",
                    "truncation_reason",
                    "original_bytes",
                    "visible_bytes",
                    "artifact",
                    "cursor",
                }
            }
            recovery_json = canonical_json_bytes({"vv_agent_recovery": recovery}, "tool result recovery").decode("utf-8")
            content = f"{content}\n{recovery_json}"
        return Message(
            role="tool",
            content=content,
            tool_call_id=self.tool_call_id,
            artifact_ref=self.artifact,
        )

    def to_dict(self) -> dict[str, Any]:
        self._validate()
        d: dict[str, Any] = {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "status_code": self.status_code.value,
            "directive": self.directive.value,
        }
        if self.error_code is not None:
            d["error_code"] = self.error_code
        if self.metadata:
            d["metadata"] = self.metadata
        if self.image_url is not None:
            d["image_url"] = self.image_url
        if self.image_path is not None:
            d["image_path"] = self.image_path
        if self.truncated is not None:
            d["truncated"] = self.truncated
        if self.truncation_reason is not None:
            d["truncation_reason"] = self.truncation_reason
        if self.original_bytes is not None:
            d["original_bytes"] = self.original_bytes
        if self.visible_bytes is not None:
            d["visible_bytes"] = self.visible_bytes
        if self.artifact is not None:
            d["artifact"] = self.artifact.to_dict()
        if self.cursor is not None:
            d["cursor"] = self.cursor.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolExecutionResult:
        if not isinstance(data, dict):
            raise TypeError("ToolExecutionResult payload must be a dict")
        allowed = {
            "tool_call_id",
            "content",
            "status_code",
            "directive",
            "error_code",
            "metadata",
            "image_url",
            "image_path",
            "truncated",
            "truncation_reason",
            "original_bytes",
            "visible_bytes",
            "artifact",
            "cursor",
        }
        unknown = sorted(set(data) - allowed)
        missing = sorted({"tool_call_id", "content", "status_code", "directive"} - set(data))
        if missing or unknown:
            raise ValueError(
                "tool_result_invalid: "
                f"ToolExecutionResult fields do not match the current wire: missing={missing}, unknown={unknown}"
            )
        try:
            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("tool_result_invalid: metadata must be an object")
            optional_strings: dict[str, str | None] = {}
            for key in ("error_code", "image_url", "image_path"):
                value = data.get(key)
                if key in data and not isinstance(value, str):
                    raise ValueError(f"tool_result_invalid: {key} must be a string")
                optional_strings[key] = value
            truncated = data.get("truncated")
            if "truncated" in data and truncated is not True:
                raise ValueError("tool_result_invalid: truncated must be omitted or true")
            for key in ("truncation_reason", "original_bytes", "visible_bytes", "artifact", "cursor"):
                if key in data and data[key] is None:
                    raise ValueError(f"tool_result_invalid: {key} must be omitted instead of null")
            artifact = data.get("artifact")
            cursor = data.get("cursor")
            if artifact is not None and not isinstance(artifact, dict):
                raise ValueError("tool_result_invalid: artifact must be an object")
            if cursor is not None and not isinstance(cursor, dict):
                raise ValueError("tool_result_invalid: cursor must be an object")
            return cls(
                tool_call_id=data["tool_call_id"],
                content=data["content"],
                status_code=ToolResultStatus(data["status_code"]),
                directive=ToolDirective(data["directive"]),
                error_code=optional_strings["error_code"],
                metadata=dict(metadata),
                image_url=optional_strings["image_url"],
                image_path=optional_strings["image_path"],
                truncated=truncated,
                truncation_reason=data.get("truncation_reason"),
                original_bytes=data.get("original_bytes"),
                visible_bytes=data.get("visible_bytes"),
                artifact=ToolArtifactRef.from_dict(artifact) if artifact is not None else None,
                cursor=ToolResultCursor.from_dict(cursor) if cursor is not None else None,
            )
        except (TypeError, ValueError) as exc:
            if str(exc) == "artifact_path_invalid":
                raise ValueError("artifact_path_invalid") from exc
            raise ValueError("tool_result_invalid") from exc
