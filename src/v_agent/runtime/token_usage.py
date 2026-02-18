from __future__ import annotations

from typing import Any

from v_agent.types import TaskTokenUsage, TokenUsage


def normalize_token_usage(raw_usage: Any) -> TokenUsage:
    if not isinstance(raw_usage, dict):
        return TokenUsage()

    prompt_tokens = _read_int(raw_usage.get("prompt_tokens"))
    completion_tokens = _read_int(raw_usage.get("completion_tokens"))
    input_tokens = _read_int(raw_usage.get("input_tokens"))
    output_tokens = _read_int(raw_usage.get("output_tokens"))

    if input_tokens is None:
        input_tokens = prompt_tokens
    if output_tokens is None:
        output_tokens = completion_tokens

    total_tokens = _read_int(raw_usage.get("total_tokens"))
    if total_tokens is None:
        total_tokens = (prompt_tokens or input_tokens or 0) + (completion_tokens or output_tokens or 0)

    cached_tokens = _read_nested_int(
        raw_usage,
        ("prompt_tokens_details", "cached_tokens"),
        ("input_tokens_details", "cached_tokens"),
        ("cache_read_input_tokens",),
        ("cache_read_tokens",),
    )
    reasoning_tokens = _read_nested_int(
        raw_usage,
        ("completion_tokens_details", "reasoning_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
        ("reasoning_tokens",),
    )
    cache_creation_tokens = _read_nested_int(
        raw_usage,
        ("input_tokens_details", "cache_creation_tokens"),
        ("prompt_tokens_details", "cache_creation_tokens"),
        ("cache_creation_input_tokens",),
        ("cache_creation_tokens",),
    )

    return TokenUsage(
        prompt_tokens=prompt_tokens or input_tokens or 0,
        completion_tokens=completion_tokens or output_tokens or 0,
        total_tokens=total_tokens or 0,
        cached_tokens=cached_tokens or 0,
        reasoning_tokens=reasoning_tokens or 0,
        input_tokens=input_tokens or 0,
        output_tokens=output_tokens or 0,
        cache_creation_tokens=cache_creation_tokens or 0,
        raw=dict(raw_usage),
    )


def summarize_task_token_usage(cycles: list[Any]) -> TaskTokenUsage:
    summary = TaskTokenUsage()
    for cycle in cycles:
        usage = getattr(cycle, "token_usage", None)
        if isinstance(usage, TokenUsage):
            summary.add_cycle(cycle.index, usage)
    return summary


def _read_nested_int(source: dict[str, Any], *path_options: tuple[str, ...]) -> int | None:
    for path in path_options:
        current: Any = source
        matched = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                matched = False
                break
            current = current[key]
        if not matched:
            continue
        value = _read_int(current)
        if value is not None:
            return value
    return None


def _read_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None
