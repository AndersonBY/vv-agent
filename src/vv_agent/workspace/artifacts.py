from __future__ import annotations

import errno
import hashlib
import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from vv_agent.types import ToolArtifactRef
from vv_agent.workspace.base import WorkspaceBackend, _normalize_workspace_path
from vv_agent.workspace.streaming import scan_text

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
    json_bytes: int


def bounded_text_preview(text: str, *, artifact_path: str | None = None) -> BoundedTextPreview:
    original_bytes = sum(len(chunk.encode("utf-8")) for chunk in _text_chunks(text))
    json_bytes = 2 + sum(len(json.dumps(chunk, ensure_ascii=False)[1:-1].encode("utf-8")) for chunk in _text_chunks(text))
    if len(text) <= BOUNDED_TEXT_CHARS:
        return BoundedTextPreview(
            content=text,
            original_bytes=original_bytes,
            visible_bytes=original_bytes,
            truncated=False,
            json_bytes=json_bytes,
        )

    marker = PREVIEW_MARKER
    if artifact_path is not None:
        shown = BOUNDED_TEXT_CHARS
        while True:
            marker = (
                f"\n... {shown} of {len(text)} source characters shown; full text in {artifact_path}; "
                "read a range with read_file(start_line, end_line) or search with search_files(pattern) ...\n"
            )
            remaining = BOUNDED_TEXT_CHARS - len(marker)
            if remaining == shown:
                break
            shown = remaining
    tail_chars = BOUNDED_TEXT_CHARS - PREVIEW_HEAD_CHARS - len(marker)
    if tail_chars <= 0:
        raise ValueError("artifact recovery marker exceeds preview budget")
    content = f"{text[:PREVIEW_HEAD_CHARS]}{marker}{text[-tail_chars:]}"
    if len(content) != BOUNDED_TEXT_CHARS:
        raise AssertionError("bounded text preview constants do not total 12,000 characters")
    return BoundedTextPreview(
        content=content,
        original_bytes=original_bytes,
        visible_bytes=len(content.encode("utf-8")),
        truncated=True,
        json_bytes=json_bytes,
    )


def bounded_captured_text_preview(path: Path) -> BoundedTextPreview:
    """Build a bounded preview without loading an entire process capture."""
    first_chars: list[str] = []
    first_count = 0
    tail = ""
    total_chars = 0
    original_bytes = 0
    json_bytes = 2

    for chunk in _iter_captured_text_chunks(path):
        total_chars += len(chunk)
        original_bytes += len(chunk.encode("utf-8"))
        json_bytes += len(json.dumps(chunk, ensure_ascii=False)[1:-1].encode("utf-8"))
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
            json_bytes=json_bytes,
        )

    content = f"{first[:PREVIEW_HEAD_CHARS]}{PREVIEW_MARKER}{tail}"
    if len(content) != BOUNDED_TEXT_CHARS:
        raise AssertionError("bounded capture preview constants do not total 12,000 characters")
    return BoundedTextPreview(
        content=content,
        original_bytes=original_bytes,
        visible_bytes=len(content.encode("utf-8")),
        truncated=True,
        json_bytes=json_bytes,
    )


def _text_chunks(text: str) -> Iterator[str]:
    for start in range(0, len(text), _CAPTURE_CHUNK_CHARS):
        yield text[start : start + _CAPTURE_CHUNK_CHARS]


def text_artifact_ref(task_id: str, tool_call_id: str, text: str, *, identity: str | None = None) -> ToolArtifactRef:
    """Describe deterministic content reuse or one immutable completion identity."""
    digest = hashlib.sha256()
    size = 0
    for chunk in _text_chunks(text):
        data = chunk.encode("utf-8")
        digest.update(data)
        size += len(data)
    suffix = digest.hexdigest() if identity is None else hashlib.sha256(identity.encode("utf-8")).hexdigest()
    path = f".vv-agent/artifacts/{_artifact_segment(task_id, 'task')}/{_artifact_segment(tool_call_id, 'call')[:59]}-{suffix}.txt"
    return ToolArtifactRef(path, "text/plain", "utf-8", size, digest.hexdigest())


def verify_text_artifact(backend: WorkspaceBackend, artifact: ToolArtifactRef) -> None:
    scanned = scan_text(backend, artifact.path, lambda _text: None)
    if not scanned.valid_utf8 or scanned.size_bytes != artifact.size_bytes or scanned.sha256 != artifact.sha256:
        raise OSError(errno.EIO, "existing artifact differs from selected text")


def persist_text_artifact(
    backend: WorkspaceBackend,
    task_id: str,
    tool_call_id: str,
    text: str,
    *,
    reuse_existing: bool = False,
    identity: str | None = None,
) -> ToolArtifactRef:
    expected = text_artifact_ref(task_id, tool_call_id, text, identity=identity)
    last_collision: FileExistsError | None = None

    for _ in range(32):
        artifact = expected
        if not reuse_existing and identity is None:
            task_segment = _artifact_segment(task_id, "task")
            call_segment = _artifact_segment(tool_call_id, "call")
            path = f".vv-agent/artifacts/{task_segment}/{call_segment}-{uuid.uuid4().hex}.txt"
            artifact = ToolArtifactRef(path, "text/plain", "utf-8", expected.size_bytes, expected.sha256)
        try:
            written = backend.write_text_chunks_exclusive(artifact.path, _text_chunks(text))
        except FileExistsError as exc:
            if not reuse_existing and identity is None:
                last_collision = exc
                continue
            # Rebuilding a model request must retain the same immutable reference.
            verify_text_artifact(backend, artifact)
            return artifact
        if isinstance(written, bool) or not isinstance(written, int) or written != artifact.size_bytes:
            raise OSError(
                errno.EIO,
                f"artifact write reported {written} of {artifact.size_bytes} bytes",
            )
        file_info = getattr(backend, "file_info", None)
        if callable(file_info):
            info = file_info(artifact.path)
            if info is None or not info.is_file or info.size != artifact.size_bytes:
                raise OSError(errno.EIO, "artifact file size differs from selected text")
        return artifact

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
    return normalized.startswith((".vv-agent/artifacts/", ".vv-agent/resources/"))


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
