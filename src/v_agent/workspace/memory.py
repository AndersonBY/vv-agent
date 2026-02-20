from __future__ import annotations

import posixpath
from datetime import UTC, datetime

from v_agent.workspace.base import FileInfo


def _glob_match(path: str, pattern: str) -> bool:
    """Match a posix path against a glob pattern supporting ``**``."""
    # Convert glob pattern to regex: ** matches any path segments, * matches within a segment
    import re as _re

    parts: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern[i:i + 3] == "**/":
            parts.append("(?:.+/)?")
            i += 3
        elif pattern[i:i + 2] == "**":
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
    __slots__ = ("_dirs", "_files")

    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}
        self._dirs: set[str] = {""}

    @staticmethod
    def _norm(path: str) -> str:
        return posixpath.normpath(path).lstrip("/")

    def list_files(self, base: str, glob: str) -> list[str]:
        base_n = self._norm(base)
        pattern = posixpath.join(base_n, glob) if base_n and base_n != "." else glob
        files = [p for p in self._files if _glob_match(p, pattern)]
        files.sort()
        return files

    def read_text(self, path: str) -> str:
        key = self._norm(path)
        if key not in self._files:
            raise FileNotFoundError(path)
        return self._files[key].decode("utf-8", errors="replace")

    def read_bytes(self, path: str) -> bytes:
        key = self._norm(path)
        if key not in self._files:
            raise FileNotFoundError(path)
        return self._files[key]

    def write_text(self, path: str, content: str, *, append: bool = False) -> int:
        key = self._norm(path)
        data = content.encode("utf-8")
        if append and key in self._files:
            self._files[key] += data
        else:
            self._files[key] = data
        self._ensure_parents(key)
        return len(content)

    def file_info(self, path: str) -> FileInfo | None:
        key = self._norm(path)
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
                path=key,
                is_file=False,
                is_dir=True,
                size=0,
                modified_at=datetime.now(tz=UTC).isoformat(),
                suffix="",
            )
        return None

    def exists(self, path: str) -> bool:
        key = self._norm(path)
        return key in self._files or key in self._dirs

    def is_file(self, path: str) -> bool:
        return self._norm(path) in self._files

    def mkdir(self, path: str) -> None:
        key = self._norm(path)
        self._dirs.add(key)
        self._ensure_parents(key)

    def _ensure_parents(self, key: str) -> None:
        parts = key.split("/")
        for i in range(1, len(parts)):
            self._dirs.add("/".join(parts[:i]))
