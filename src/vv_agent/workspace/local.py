from __future__ import annotations

import errno
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path

from vv_agent.workspace.artifacts import ArtifactPathInvalidError, is_reserved_artifact_path
from vv_agent.workspace.base import FileInfo, _exclusive_workspace_path_segments


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
    __slots__ = ("_allow_outside_root", "_root")

    def __init__(self, root: Path, *, allow_outside_root: bool = False) -> None:
        self._root = root.resolve()
        self._allow_outside_root = bool(allow_outside_root)

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

    def _to_output_path(self, path: Path) -> str:
        try:
            rel = path.relative_to(self._root).as_posix()
            return rel or "."
        except ValueError:
            return str(path)

    def list_files(self, base: str, glob: str) -> list[str]:
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
                except (OSError, ValueError):
                    continue
                files.append(rel)
        files.sort()
        return files

    def read_text(self, path: str) -> str:
        return self._resolve(path).read_text(encoding="utf-8", errors="replace")

    def read_bytes(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

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
        segments = _exclusive_workspace_path_segments(path)
        self._root.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
            return _write_exclusive_at(self._root, segments, data)
        return _write_exclusive_portable(self._root, segments, data)

    def file_info(self, path: str) -> FileInfo | None:
        target = self._resolve(path)
        if not target.exists():
            return None
        stat = target.stat()
        return FileInfo(
            path=self._to_output_path(target),
            is_file=target.is_file(),
            is_dir=target.is_dir(),
            size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            suffix=target.suffix,
        )

    def exists(self, path: str) -> bool:
        try:
            return self._resolve(path).exists()
        except (OSError, RuntimeError, ValueError):
            return False

    def is_file(self, path: str) -> bool:
        try:
            return self._resolve(path).is_file()
        except (OSError, RuntimeError, ValueError):
            return False

    def mkdir(self, path: str) -> None:
        self._resolve(path).mkdir(parents=True, exist_ok=True)

    def _is_reserved_target(self, target: Path) -> bool:
        try:
            relative = target.relative_to(self._root).as_posix()
        except ValueError:
            return False
        return is_reserved_artifact_path(relative)


def _write_all(file_descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(file_descriptor, remaining)
        if written <= 0:
            raise OSError("exclusive workspace write made no progress")
        remaining = remaining[written:]


def _write_exclusive_at(root: Path, segments: tuple[str, ...], data: bytes) -> int:
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

        file_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        file_fd = os.open(segments[-1], file_flags, 0o600, dir_fd=directory_fd)
        try:
            _write_all(file_fd, data)
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        os.fsync(directory_fd)
        return len(data)
    finally:
        os.close(directory_fd)


def _write_exclusive_portable(root: Path, segments: tuple[str, ...], data: bytes) -> int:
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
        _write_all(file_fd, data)
        os.fsync(file_fd)
    finally:
        os.close(file_fd)
    return len(data)
