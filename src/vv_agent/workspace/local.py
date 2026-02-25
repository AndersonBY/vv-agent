from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from pathlib import Path

from vv_agent.workspace.base import FileInfo


def _glob_match(path: str, pattern: str) -> bool:
    """Match a posix path against a glob pattern supporting ``**``."""
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
            parts.append(re.escape(pattern[i]))
            i += 1
    regex = "^" + "".join(parts) + "$"
    return re.match(regex, path) is not None


class LocalWorkspaceBackend:
    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, path: str) -> Path:
        target = (self._root / path).resolve()
        if target != self._root and self._root not in target.parents:
            raise ValueError(f"Path escapes workspace: {path}")
        return target

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
                    rel = candidate.relative_to(self._root).as_posix()
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
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8") as fh:
            fh.write(content)
        return len(content)

    def file_info(self, path: str) -> FileInfo | None:
        target = self._resolve(path)
        if not target.exists():
            return None
        stat = target.stat()
        return FileInfo(
            path=target.relative_to(self._root).as_posix(),
            is_file=target.is_file(),
            is_dir=target.is_dir(),
            size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            suffix=target.suffix,
        )

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def is_file(self, path: str) -> bool:
        return self._resolve(path).is_file()

    def mkdir(self, path: str) -> None:
        self._resolve(path).mkdir(parents=True, exist_ok=True)
