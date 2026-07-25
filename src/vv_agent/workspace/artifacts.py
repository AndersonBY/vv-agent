from __future__ import annotations

import errno
import hashlib
import uuid
from dataclasses import dataclass

from vv_agent.types import ToolArtifactRef
from vv_agent.workspace.base import WorkspaceBackend, _normalize_workspace_path

BOUNDED_TEXT_CHARS = 12_000
PREVIEW_HEAD_CHARS = 6_000
PREVIEW_TAIL_CHARS = 5_953
PREVIEW_MARKER = "\n... output omitted; full text in artifact ...\n"


class ArtifactPathInvalidError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BoundedTextPreview:
    content: str
    original_bytes: int
    visible_bytes: int
    truncated: bool


def bounded_text_preview(text: str) -> BoundedTextPreview:
    original_bytes = len(text.encode("utf-8"))
    if len(text) <= BOUNDED_TEXT_CHARS:
        return BoundedTextPreview(
            content=text,
            original_bytes=original_bytes,
            visible_bytes=original_bytes,
            truncated=False,
        )

    content = f"{text[:PREVIEW_HEAD_CHARS]}{PREVIEW_MARKER}{text[-PREVIEW_TAIL_CHARS:]}"
    if len(content) != BOUNDED_TEXT_CHARS:
        raise AssertionError("bounded text preview constants do not total 12,000 characters")
    return BoundedTextPreview(
        content=content,
        original_bytes=original_bytes,
        visible_bytes=len(content.encode("utf-8")),
        truncated=True,
    )


def persist_text_artifact(
    backend: WorkspaceBackend,
    task_id: str,
    tool_call_id: str,
    text: str,
) -> ToolArtifactRef:
    task_segment = _artifact_segment(task_id, "task")
    call_segment = _artifact_segment(tool_call_id, "call")
    data = text.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    last_collision: FileExistsError | None = None

    for _ in range(32):
        suffix = uuid.uuid4().hex
        path = f".vv-agent/artifacts/{task_segment}/{call_segment}-{suffix}.txt"
        try:
            written = backend.write_text_exclusive(path, text)
        except FileExistsError as exc:
            last_collision = exc
            continue
        if isinstance(written, bool) or not isinstance(written, int) or written != len(data):
            raise OSError(
                errno.EIO,
                f"artifact write reported {written} of {len(data)} bytes",
            )
        return ToolArtifactRef(
            path=path,
            media_type="text/plain",
            encoding="utf-8",
            size_bytes=len(data),
            sha256=digest,
        )

    if last_collision is not None:
        raise last_collision
    raise FileExistsError(errno.EEXIST, "could not allocate an exclusive artifact path")


def is_reserved_artifact_path(path: str) -> bool:
    normalized = _normalize_workspace_path(path)
    return normalized.startswith(".vv-agent/artifacts/")


def _artifact_segment(value: str, fallback: str) -> str:
    segment: list[str] = []
    for character in str(value):
        if len(segment) >= 64:
            break
        if character.isascii() and (character.isalnum() or character in "._-"):
            segment.append(character)
        elif not segment or segment[-1] != "-":
            segment.append("-")
    while segment and not segment[0].isalnum():
        segment.pop(0)
    return "".join(segment) or fallback
