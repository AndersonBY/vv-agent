"""Provider error classification shared by model callers and compaction."""

from __future__ import annotations

from typing import Any

MAX_PTL_RETRIES = 3


_PTL_ERROR_PATTERNS = (
    "prompt is too long",
    "prompt_too_long",
    "context_length_exceeded",
    "maximum context length",
    "request too large",
    "too many tokens",
)


def is_prompt_too_long_error(error: Exception) -> bool:
    visited: set[int] = set()
    stack: list[Any] = [error]
    while stack:
        current = stack.pop()
        identifier = id(current)
        if identifier in visited:
            continue
        visited.add(identifier)

        current_text = str(current).lower()
        if any(pattern in current_text for pattern in _PTL_ERROR_PATTERNS):
            return True

        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if cause is not None:
            stack.append(cause)
        if context is not None:
            stack.append(context)
        args = getattr(current, "args", ())
        if isinstance(args, tuple):
            stack.extend(arg for arg in args if isinstance(arg, BaseException))
    return False
