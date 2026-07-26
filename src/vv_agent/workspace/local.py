from __future__ import annotations

import errno
import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from vv_agent.workspace.artifacts import ArtifactPathInvalidError, is_reserved_artifact_path
from vv_agent.workspace.base import FileInfo, _exclusive_workspace_path_segments, _normalize_workspace_path

_PRIVATE_ARTIFACT_ROOT_ENV = "VV_AGENT_PRIVATE_ARTIFACT_ROOT"
_PRIVATE_ARTIFACT_ROOT_NAME = "vv-agent-artifacts"


def _glob_match(path: str, pattern: str) -> bool:
    """Match a posix path against a glob pattern supporting ``**``."""
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern[i : i + 3] == "**/":
            parts.append("(?:.+/)?")
            i += 3
        elif pattern[i : i + 2] == "**":
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    regex = "^" + "".join(parts) + "$"
    return re.match(regex, path) is not None


class LocalWorkspaceBackend:
    __slots__ = ("_allow_outside_root", "_artifact_root", "_root")

    def __init__(self, root: Path, *, allow_outside_root: bool = False) -> None:
        self._root = root.resolve()
        self._allow_outside_root = bool(allow_outside_root)
        self._artifact_root = _private_artifact_root(self._root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def allow_outside_root(self) -> bool:
        return self._allow_outside_root

    def _resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        target = candidate.resolve() if candidate.is_absolute() else (self._root / candidate).resolve()
        if not self._allow_outside_root and target != self._root and self._root not in target.parents:
            raise ValueError(f"Path escapes workspace: {path}")
        return target

    def _artifact_segments(self, path: str) -> tuple[str, ...] | None:
        if not isinstance(path, str) or path.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", path):
            return None
        normalized = _normalize_workspace_path(path)
        if not is_reserved_artifact_path(normalized):
            return None
        return _exclusive_workspace_path_segments(normalized)

    def _resolve_artifact(self, path: str) -> tuple[Path, tuple[str, ...]] | None:
        segments = self._artifact_segments(path)
        if segments is None:
            return None

        target = self._artifact_root
        for segment in segments:
            target /= segment
            try:
                target_stat = target.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(target_stat.st_mode):
                raise ArtifactPathInvalidError("artifact_path_invalid")
        return target, segments

    def _resolve_read_target(self, path: str) -> tuple[Path, str | None]:
        artifact = self._resolve_artifact(path)
        if artifact is not None:
            target, _segments = artifact
            return target, _normalize_workspace_path(path)
        return self._resolve(path), None

    def _to_output_path(self, path: Path) -> str:
        try:
            rel = path.relative_to(self._root).as_posix()
            return rel or "."
        except ValueError:
            return str(path)

    def list_files(self, base: str, glob: str) -> list[str]:
        if self._artifact_segments(base) is not None:
            return []
        root = self._resolve(base)
        if not root.exists() or not root.is_dir():
            return []

        pattern = str(glob or "**/*")
        files: list[str] = []
        for current_root, dirs, filenames in os.walk(root, topdown=True, onerror=lambda _e: None, followlinks=False):
            dirs.sort(key=str.lower)
            filenames.sort(key=str.lower)
            current = Path(current_root)
            for filename in filenames:
                candidate = current / filename
                try:
                    rel_from_base = candidate.relative_to(root).as_posix()
                    if not _glob_match(rel_from_base, pattern):
                        continue
                    rel = self._to_output_path(candidate)
                    if is_reserved_artifact_path(rel):
                        continue
                except (OSError, ValueError):
                    continue
                files.append(rel)
        files.sort()
        return files

    def read_text(self, path: str) -> str:
        target, _logical_path = self._resolve_read_target(path)
        return target.read_text(encoding="utf-8", errors="replace")

    def read_bytes(self, path: str) -> bytes:
        target, _logical_path = self._resolve_read_target(path)
        return target.read_bytes()

    def write_text(self, path: str, content: str, *, append: bool = False) -> int:
        if is_reserved_artifact_path(path):
            raise PermissionError("artifact paths are immutable")
        target = self._resolve(path)
        if self._is_reserved_target(target):
            raise PermissionError("artifact paths are immutable")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        mode = "ab" if append else "wb"
        with target.open(mode) as fh:
            return fh.write(data)

    def write_text_exclusive(self, path: str, content: str) -> int:
        return self.write_text_chunks_exclusive(path, (content,))

    def write_text_chunks_exclusive(self, path: str, chunks: Iterable[str]) -> int:
        segments = _exclusive_workspace_path_segments(path)
        artifact = self._resolve_artifact(path)
        root = self._artifact_root if artifact is not None else self._root
        root.mkdir(parents=True, exist_ok=True)
        if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
            return _write_exclusive_chunks_at(root, segments, chunks)
        return _write_exclusive_chunks_portable(root, segments, chunks)

    def file_info(self, path: str) -> FileInfo | None:
        target, logical_path = self._resolve_read_target(path)
        if not target.exists():
            return None
        stat = target.stat()
        return FileInfo(
            path=logical_path or self._to_output_path(target),
            is_file=target.is_file(),
            is_dir=target.is_dir(),
            size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            suffix=target.suffix,
        )

    def exists(self, path: str) -> bool:
        try:
            target, _logical_path = self._resolve_read_target(path)
            return target.exists()
        except (OSError, RuntimeError, ValueError):
            return False

    def is_file(self, path: str) -> bool:
        try:
            target, _logical_path = self._resolve_read_target(path)
            return target.is_file()
        except (OSError, RuntimeError, ValueError):
            return False

    def mkdir(self, path: str) -> None:
        if is_reserved_artifact_path(path):
            raise PermissionError("artifact paths are immutable")
        self._resolve(path).mkdir(parents=True, exist_ok=True)

    def _is_reserved_target(self, target: Path) -> bool:
        try:
            relative = target.relative_to(self._root).as_posix()
        except ValueError:
            return False
        return is_reserved_artifact_path(relative)


def _private_artifact_root(workspace_root: Path) -> Path:
    configured_root = os.environ.get(_PRIVATE_ARTIFACT_ROOT_ENV)
    base = Path(configured_root).expanduser() if configured_root else Path(tempfile.gettempdir()) / _PRIVATE_ARTIFACT_ROOT_NAME
    base.mkdir(parents=True, exist_ok=True)
    if base.is_symlink() or not base.is_dir():
        raise ArtifactPathInvalidError("artifact_path_invalid")
    with suppress(OSError):
        base.chmod(0o700)

    digest = hashlib.sha256(os.fsencode(str(workspace_root))).hexdigest()
    artifact_root = base / digest
    artifact_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ArtifactPathInvalidError("artifact_path_invalid")
    return artifact_root.resolve()


def _write_all(file_descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(file_descriptor, remaining)
        if written <= 0:
            raise OSError("exclusive workspace write made no progress")
        remaining = remaining[written:]


def _write_exclusive_chunks_at(root: Path, segments: tuple[str, ...], chunks: Iterable[str]) -> int:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(root, directory_flags)
    try:
        for segment in segments[:-1]:
            try:
                os.mkdir(segment, mode=0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            else:
                os.fsync(directory_fd)
            segment_stat = os.stat(segment, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(segment_stat.st_mode):
                raise ArtifactPathInvalidError("artifact_path_invalid")
            if not stat.S_ISDIR(segment_stat.st_mode):
                raise NotADirectoryError(segment)
            try:
                next_fd = os.open(segment, directory_flags, dir_fd=directory_fd)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ArtifactPathInvalidError("artifact_path_invalid") from exc
                raise
            os.close(directory_fd)
            directory_fd = next_fd

        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        file_fd = os.open(segments[-1], file_flags, 0o600, dir_fd=directory_fd)
        try:
            total = _write_chunks(file_fd, chunks)
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        os.fsync(directory_fd)
        return total
    finally:
        os.close(directory_fd)


def _write_exclusive_chunks_portable(root: Path, segments: tuple[str, ...], chunks: Iterable[str]) -> int:
    parent = root
    for segment in segments[:-1]:
        parent /= segment
        try:
            parent.lstat()
        except FileNotFoundError:
            parent.mkdir(mode=0o700)
            continue
        if parent.is_symlink():
            raise ArtifactPathInvalidError("artifact_path_invalid")
        if not parent.is_dir():
            raise NotADirectoryError(parent)

    target = parent / segments[-1]
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    file_fd = os.open(target, flags, 0o600)
    try:
        total = _write_chunks(file_fd, chunks)
        os.fsync(file_fd)
    finally:
        os.close(file_fd)
    return total


def _write_chunks(file_descriptor: int, chunks: Iterable[str]) -> int:
    total = 0
    for chunk in chunks:
        if not isinstance(chunk, str):
            raise TypeError("artifact text chunks must be strings")
        data = chunk.encode("utf-8")
        _write_all(file_descriptor, data)
        total += len(data)
    return total
