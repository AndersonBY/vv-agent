from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from hashlib import sha256
from threading import RLock
from uuid import uuid4

from vv_agent.workspace.artifacts import is_reserved_artifact_path
from vv_agent.workspace.base import FileInfo, _exclusive_workspace_path_segments, _normalize_workspace_path


def _glob_match(path: str, pattern: str) -> bool:
    """Match a posix path against a glob pattern supporting ``**``."""
    # Convert glob pattern to regex: ** matches any path segments, * matches within a segment
    import re as _re

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
            parts.append(_re.escape(pattern[i]))
            i += 1
    regex = "^" + "".join(parts) + "$"
    return _re.match(regex, path) is not None


class MemoryWorkspaceBackend:
    __slots__ = ("_dirs", "_files", "_lock", "_retained_workspace_id")

    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}
        self._dirs: set[str] = {""}
        self._lock = RLock()
        self._retained_workspace_id = sha256(uuid4().bytes).hexdigest()

    @staticmethod
    def _norm(path: str) -> str:
        return _normalize_workspace_path(path)

    def list_files(self, base: str, glob: str) -> list[str]:
        base_n = self._norm(base)
        if is_reserved_artifact_path(base_n):
            return []
        pattern = f"{base_n}/{glob}" if base_n else glob
        with self._lock:
            files = [p for p in self._files if not is_reserved_artifact_path(p) and _glob_match(p, pattern)]
        files.sort()
        return files

    def read_text(self, path: str) -> str:
        key = self._norm(path)
        with self._lock:
            if key not in self._files:
                raise FileNotFoundError(path)
            return self._files[key].decode("utf-8", errors="replace")

    def read_bytes(self, path: str) -> bytes:
        key = self._norm(path)
        with self._lock:
            if key not in self._files:
                raise FileNotFoundError(path)
            return self._files[key]

    def _read_bytes_chunks(self, path: str) -> Iterable[bytes]:
        key = self._norm(path)
        with self._lock:
            if key not in self._files:
                raise FileNotFoundError(path)
            data = self._files[key]
        for offset in range(0, len(data), 65_536):
            yield data[offset : offset + 65_536]

    def write_text(self, path: str, content: str, *, append: bool = False) -> int:
        key = self._norm(path)
        if is_reserved_artifact_path(key):
            raise PermissionError("artifact paths are immutable")
        data = content.encode("utf-8")
        with self._lock:
            if append and key in self._files:
                self._files[key] += data
            else:
                self._files[key] = data
            self._ensure_parents(key)
        return len(data)

    def write_text_exclusive(self, path: str, content: str) -> int:
        return self.write_text_chunks_exclusive(path, (content,))

    def write_text_chunks_exclusive(self, path: str, chunks: Iterable[str]) -> int:
        return self.write_bytes_chunks_exclusive(path, (chunk.encode("utf-8") for chunk in chunks))

    def write_bytes_chunks_exclusive(self, path: str, chunks: Iterable[bytes]) -> int:
        key = "/".join(_exclusive_workspace_path_segments(path))
        data = b"".join(chunks)
        with self._lock:
            if key in self._files or key in self._dirs:
                raise FileExistsError(path)
            parts = key.split("/")
            for index in range(1, len(parts)):
                parent = "/".join(parts[:index])
                if parent in self._files:
                    raise NotADirectoryError(parent)
            self._files[key] = data
            self._ensure_parents(key)
        return len(data)

    @property
    def retained_workspace_id(self) -> str:
        return self._retained_workspace_id

    def retain_output(self, path, *, expected):
        from vv_agent.workspace.resources import copy_retained_output

        return copy_retained_output(self, path, expected=expected)

    def _read_output_chunks(self, path):
        _exclusive_workspace_path_segments(path)
        return self._read_bytes_chunks(path)

    def open_retained(self, ref):
        from vv_agent.workspace.resources import validate_retained_workspace

        return self._read_output_chunks(validate_retained_workspace(ref, self.retained_workspace_id))

    def file_info(self, path: str) -> FileInfo | None:
        key = self._norm(path)
        with self._lock:
            if key in self._files:
                suffix = ""
                dot = key.rfind(".")
                if dot != -1 and "/" not in key[dot:]:
                    suffix = key[dot:]
                return FileInfo(
                    path=key,
                    is_file=True,
                    is_dir=False,
                    size=len(self._files[key]),
                    modified_at=datetime.now(tz=UTC).isoformat(),
                    suffix=suffix,
                )
            if key in self._dirs:
                return FileInfo(
                    path=key or ".",
                    is_file=False,
                    is_dir=True,
                    size=0,
                    modified_at=datetime.now(tz=UTC).isoformat(),
                    suffix="",
                )
            return None

    def exists(self, path: str) -> bool:
        key = self._norm(path)
        with self._lock:
            return key in self._files or key in self._dirs

    def is_file(self, path: str) -> bool:
        with self._lock:
            return self._norm(path) in self._files

    def mkdir(self, path: str) -> None:
        key = self._norm(path)
        with self._lock:
            self._dirs.add(key)
            self._ensure_parents(key)

    def _ensure_parents(self, key: str) -> None:
        parts = key.split("/")
        for i in range(1, len(parts)):
            self._dirs.add("/".join(parts[:i]))
