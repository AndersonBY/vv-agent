from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

INVALID_EXCLUDE_FILES_PATTERN_CODE = "invalid_exclude_files_pattern"
INVALID_EXCLUDE_FILES_PATTERN_MESSAGE = "exclude_files_pattern must be a valid portable regular expression"


def _normalize_workspace_path(path: str) -> str:
    """Normalize a virtual workspace path without allowing root escape."""
    parts: list[str] = []
    for part in str(path).replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


@dataclass(slots=True)
class FileInfo:
    path: str
    is_file: bool
    is_dir: bool
    size: int
    modified_at: str
    suffix: str


class WorkspaceBackend(Protocol):
    def list_files(self, base: str, glob: str) -> list[str]: ...
    def read_text(self, path: str) -> str: ...
    def read_bytes(self, path: str) -> bytes: ...
    def write_text(self, path: str, content: str, *, append: bool = False) -> int: ...
    def file_info(self, path: str) -> FileInfo | None: ...
    def exists(self, path: str) -> bool: ...
    def is_file(self, path: str) -> bool: ...
    def mkdir(self, path: str) -> None: ...


class InvalidPortableRegexError(ValueError):
    pass


def compile_portable_workspace_regex(pattern: str) -> re.Pattern[str]:
    if not isinstance(pattern, str):
        raise InvalidPortableRegexError(INVALID_EXCLUDE_FILES_PATTERN_MESSAGE)

    portable_letter_escapes = frozenset("AabBdDfnrsStvwWx")
    index = 0
    in_character_class = False
    while index < len(pattern):
        current = pattern[index]
        if current == "\\":
            if index + 1 >= len(pattern):
                raise InvalidPortableRegexError(INVALID_EXCLUDE_FILES_PATTERN_MESSAGE)
            escaped = pattern[index + 1]
            if escaped.isascii() and escaped.isalnum() and escaped not in portable_letter_escapes:
                raise InvalidPortableRegexError(INVALID_EXCLUDE_FILES_PATTERN_MESSAGE)
            if escaped == "x" and index + 2 < len(pattern) and pattern[index + 2] == "{":
                raise InvalidPortableRegexError(INVALID_EXCLUDE_FILES_PATTERN_MESSAGE)
            if in_character_class and escaped in {"A", "b", "B"}:
                raise InvalidPortableRegexError(INVALID_EXCLUDE_FILES_PATTERN_MESSAGE)
            index += 2
            continue
        if current == "[":
            if in_character_class:
                raise InvalidPortableRegexError(INVALID_EXCLUDE_FILES_PATTERN_MESSAGE)
            in_character_class = True
            index += 1
            continue
        if current == "]" and in_character_class:
            in_character_class = False
            index += 1
            continue
        if in_character_class:
            if pattern.startswith(("&&", "--", "~~"), index):
                raise InvalidPortableRegexError(INVALID_EXCLUDE_FILES_PATTERN_MESSAGE)
            index += 1
            continue
        if pattern.startswith("{,", index):
            raise InvalidPortableRegexError(INVALID_EXCLUDE_FILES_PATTERN_MESSAGE)
        if (
            current == "("
            and index + 1 < len(pattern)
            and pattern[index + 1] == "?"
            and not pattern.startswith("(?:", index)
        ):
            raise InvalidPortableRegexError(INVALID_EXCLUDE_FILES_PATTERN_MESSAGE)
        if current in {"*", "+", "?", "}"} and index + 1 < len(pattern) and pattern[index + 1] == "+":
            raise InvalidPortableRegexError(INVALID_EXCLUDE_FILES_PATTERN_MESSAGE)
        if current in {"*", "+", "?"} and index + 1 < len(pattern) and pattern[index + 1] in {"*", "+"}:
            raise InvalidPortableRegexError(INVALID_EXCLUDE_FILES_PATTERN_MESSAGE)
        index += 1

    try:
        return re.compile(pattern, flags=re.ASCII)
    except re.error as exc:
        raise InvalidPortableRegexError(INVALID_EXCLUDE_FILES_PATTERN_MESSAGE) from exc


class DiscoveryFilteredWorkspaceBackend:
    """Hide matching paths from discovery while preserving direct path access."""

    __slots__ = ("_backend", "_exclude_pattern", "_regex")

    def __init__(self, backend: WorkspaceBackend, exclude_pattern: str) -> None:
        self._backend = backend
        self._exclude_pattern = exclude_pattern
        self._regex = compile_portable_workspace_regex(exclude_pattern)

    @property
    def backend(self) -> WorkspaceBackend:
        return self._backend

    @property
    def exclude_pattern(self) -> str:
        return self._exclude_pattern

    def list_files(self, base: str, glob: str) -> list[str]:
        return [
            path
            for path in self._backend.list_files(base, glob)
            if not self._regex.search(_normalize_workspace_path(path))
        ]

    def read_text(self, path: str) -> str:
        return self._backend.read_text(path)

    def read_bytes(self, path: str) -> bytes:
        return self._backend.read_bytes(path)

    def write_text(self, path: str, content: str, *, append: bool = False) -> int:
        return self._backend.write_text(path, content, append=append)

    def file_info(self, path: str) -> FileInfo | None:
        return self._backend.file_info(path)

    def exists(self, path: str) -> bool:
        return self._backend.exists(path)

    def is_file(self, path: str) -> bool:
        return self._backend.is_file(path)

    def mkdir(self, path: str) -> None:
        self._backend.mkdir(path)
