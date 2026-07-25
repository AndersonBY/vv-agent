from __future__ import annotations

import errno
import hashlib
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from vv_agent.types import ToolArtifactRef
from vv_agent.workspace.base import WorkspaceBackend, _normalize_workspace_path

BOUNDED_TEXT_CHARS = 12_000
PREVIEW_HEAD_CHARS = 6_000
PREVIEW_TAIL_CHARS = 5_953
PREVIEW_MARKER = "\n... output omitted; full text in artifact ...\n"
_CAPTURE_CHUNK_CHARS = 64 * 1024


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


def bounded_captured_text_preview(path: Path) -> BoundedTextPreview:
    """Build a bounded preview without loading an entire process capture."""
    first_chars: list[str] = []
    first_count = 0
    tail = ""
    total_chars = 0
    original_bytes = 0

    for chunk in _iter_captured_text_chunks(path):
        total_chars += len(chunk)
        original_bytes += len(chunk.encode("utf-8"))
        if first_count < BOUNDED_TEXT_CHARS:
            visible_chunk = chunk[: BOUNDED_TEXT_CHARS - first_count]
            first_chars.append(visible_chunk)
            first_count += len(visible_chunk)
        tail = (tail + chunk)[-PREVIEW_TAIL_CHARS:]

    first = "".join(first_chars)
    if total_chars <= BOUNDED_TEXT_CHARS:
        return BoundedTextPreview(
            content=first,
            original_bytes=original_bytes,
            visible_bytes=original_bytes,
            truncated=False,
        )

    content = f"{first[:PREVIEW_HEAD_CHARS]}{PREVIEW_MARKER}{tail}"
    if len(content) != BOUNDED_TEXT_CHARS:
        raise AssertionError("bounded capture preview constants do not total 12,000 characters")
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


def persist_captured_text_artifact(
    backend: WorkspaceBackend,
    task_id: str,
    tool_call_id: str,
    capture_path: Path,
) -> ToolArtifactRef:
    """Persist a completed capture through the backend's exclusive chunk writer."""
    task_segment = _artifact_segment(task_id, "task")
    call_segment = _artifact_segment(tool_call_id, "call")
    last_collision: FileExistsError | None = None

    for _ in range(32):
        suffix = uuid.uuid4().hex
        path = f".vv-agent/artifacts/{task_segment}/{call_segment}-{suffix}.txt"
        digest = hashlib.sha256()
        size_bytes = 0

        def chunks(current_digest=digest) -> Iterator[str]:
            nonlocal size_bytes
            for chunk in _iter_captured_text_chunks(capture_path):
                data = chunk.encode("utf-8")
                current_digest.update(data)
                size_bytes += len(data)
                yield chunk

        try:
            written = backend.write_text_chunks_exclusive(path, chunks())
        except FileExistsError as exc:
            last_collision = exc
            continue
        if isinstance(written, bool) or not isinstance(written, int) or written != size_bytes:
            raise OSError(
                errno.EIO,
                f"artifact write reported {written} of {size_bytes} bytes",
            )
        return ToolArtifactRef(
            path=path,
            media_type="text/plain",
            encoding="utf-8",
            size_bytes=size_bytes,
            sha256=digest.hexdigest(),
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


def _iter_captured_text_chunks(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while chunk := handle.read(_CAPTURE_CHUNK_CHARS):
            yield chunk
