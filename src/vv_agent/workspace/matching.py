"""Interruptible matching with one CPU budget across discovery and content."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from time import thread_time

import regex

MATCH_CPU_SECONDS = 0.25
_request_deadline: ContextVar[float | None] = ContextVar("workspace_match_deadline", default=None)


@contextmanager
def matching_request() -> Iterator[None]:
    if _request_deadline.get() is not None:
        yield
        return
    token = _request_deadline.set(thread_time() + MATCH_CPU_SECONDS)
    try:
        yield
    finally:
        _request_deadline.reset(token)


def check_matching_budget() -> None:
    deadline = _request_deadline.get()
    if deadline is not None and thread_time() >= deadline:
        raise TimeoutError("Regular expression too slow; aborted")


def compile_pattern(pattern: str, flags: int = 0) -> regex.Pattern:
    compiled = regex.compile(pattern, flags | regex.V0)
    if compiled.flags & regex.REVERSE:
        raise ValueError("REVERSE regular expression mode is not supported")
    return compiled


def compile_glob(pattern: str) -> regex.Pattern:
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
            parts.append(regex.escape(pattern[i]))
            i += 1
    return compile_pattern("^" + "".join(parts) + "$")


class RegexBudget:
    def __init__(self, pattern: regex.Pattern) -> None:
        if pattern.flags & regex.REVERSE:
            raise ValueError("REVERSE regular expression mode is not supported")
        self.pattern = pattern
        deadline = _request_deadline.get()
        self.deadline = thread_time() + MATCH_CPU_SECONDS if deadline is None else deadline

    def _remaining(self) -> float:
        remaining = self.deadline - thread_time()
        if remaining <= 0:
            raise TimeoutError("Regular expression too slow; aborted")
        return remaining

    def finditer(self, text: str, pos: int = 0) -> Iterator[regex.Match]:
        yield from self.pattern.finditer(text, pos, timeout=self._remaining())

    def match(self, text: str) -> regex.Match | None:
        return self.pattern.match(text, timeout=self._remaining())

    def search(self, text: str) -> regex.Match | None:
        return self.pattern.search(text, timeout=self._remaining())
