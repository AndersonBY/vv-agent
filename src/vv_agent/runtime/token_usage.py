from __future__ import annotations

from typing import Any

from vv_agent.types import CacheUsage, CacheUsageStatus, ModelCallRecord, TaskTokenUsage, TokenUsage, UsageSource


def normalize_token_usage(
    raw_usage: Any,
    *,
    usage_source: UsageSource | str | None = None,
    cache_status: CacheUsageStatus | str | None = None,
) -> TokenUsage:
    if not isinstance(raw_usage, dict):
        normalized_cache_status = _cache_status(cache_status)
        return TokenUsage(
            usage_source=_usage_source(usage_source, {}),
            cache_usage=CacheUsage(
                status=normalized_cache_status,
                source="adapter_capability" if normalized_cache_status is CacheUsageStatus.UNSUPPORTED else None,
            ),
        )

    prompt_tokens = _read_non_negative_int(raw_usage.get("prompt_tokens"))
    completion_tokens = _read_non_negative_int(raw_usage.get("completion_tokens"))
    native_input_tokens = _read_non_negative_int(raw_usage.get("input_tokens"))
    output_tokens = _read_non_negative_int(raw_usage.get("output_tokens"))
    if native_input_tokens is None:
        native_input_tokens = prompt_tokens
    if output_tokens is None:
        output_tokens = completion_tokens

    cache_read_input_tokens = _read_nested_non_negative_int(
        raw_usage,
        ("cache_read_input_tokens",),
        ("cache_read_tokens",),
        ("prompt_tokens_details", "cached_tokens"),
        ("input_tokens_details", "cached_tokens"),
    )
    reasoning_tokens = _read_nested_int(
        raw_usage,
        ("completion_tokens_details", "reasoning_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
        ("reasoning_tokens",),
    )
    cache_write_input_tokens = _read_nested_non_negative_int(
        raw_usage,
        ("cache_write_input_tokens",),
        ("cache_creation_input_tokens",),
        ("cache_write_tokens",),
        ("input_tokens_details", "cache_creation_tokens"),
        ("prompt_tokens_details", "cache_creation_tokens"),
    )
    uncached_input_tokens = _read_nested_non_negative_int(raw_usage, ("uncached_input_tokens",))

    anthropic_native = (
        prompt_tokens is None
        and raw_usage.get("total_tokens") is None
        and uncached_input_tokens is None
        and _has_any_key(raw_usage, "cache_read_input_tokens", "cache_creation_input_tokens")
    )
    input_tokens = native_input_tokens
    if anthropic_native and native_input_tokens is not None:
        input_tokens = native_input_tokens + (cache_read_input_tokens or 0) + (cache_write_input_tokens or 0)

    total_tokens = _read_non_negative_int(raw_usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    observed_cache_metric = any(
        value is not None
        for value in (
            cache_read_input_tokens,
            cache_write_input_tokens,
            uncached_input_tokens,
        )
    )
    normalized_cache_status = CacheUsageStatus.PROVIDER_REPORTED if observed_cache_metric else _cache_status(cache_status)
    if normalized_cache_status is CacheUsageStatus.PROVIDER_REPORTED and uncached_input_tokens is None:
        if anthropic_native and native_input_tokens is not None:
            uncached_input_tokens = native_input_tokens + (cache_write_input_tokens or 0)
        elif input_tokens is not None and cache_read_input_tokens is not None:
            uncached_input_tokens = max(input_tokens - cache_read_input_tokens, 0)

    cache_usage = CacheUsage(
        status=normalized_cache_status,
        read_input_tokens=(cache_read_input_tokens if normalized_cache_status is CacheUsageStatus.PROVIDER_REPORTED else None),
        write_input_tokens=(cache_write_input_tokens if normalized_cache_status is CacheUsageStatus.PROVIDER_REPORTED else None),
        uncached_input_tokens=(uncached_input_tokens if normalized_cache_status is CacheUsageStatus.PROVIDER_REPORTED else None),
        source=(
            "provider_usage"
            if normalized_cache_status is CacheUsageStatus.PROVIDER_REPORTED
            else "adapter_capability"
            if normalized_cache_status is CacheUsageStatus.UNSUPPORTED
            else None
        ),
    )

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        usage_source=_usage_source(usage_source, raw_usage),
        cache_usage=cache_usage,
        provider_usage=dict(raw_usage),
    )


def summarize_task_token_usage(model_calls: list[ModelCallRecord]) -> TaskTokenUsage:
    summary = TaskTokenUsage()
    for model_call in model_calls:
        summary.add_model_call(model_call)
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
        value = _read_non_negative_int(current)
        if value is not None:
            return value
    return None


def _read_nested_non_negative_int(source: dict[str, Any], *path_options: tuple[str, ...]) -> int | None:
    for path in path_options:
        current: Any = source
        for key in path:
            if not isinstance(current, dict) or key not in current:
                break
            current = current[key]
        else:
            value = _read_non_negative_int(current)
            if value is not None:
                return value
    return None


def _read_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 and value.is_integer() else None
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _usage_source(value: UsageSource | str | None, raw_usage: dict[str, Any]) -> UsageSource:
    if value is not None:
        return value if isinstance(value, UsageSource) else UsageSource(value)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        if key in raw_usage and _read_non_negative_int(raw_usage.get(key)) is not None:
            return UsageSource.PROVIDER_REPORTED
    return UsageSource.ACCOUNTING_MISSING


def _cache_status(value: CacheUsageStatus | str | None) -> CacheUsageStatus:
    if value is None:
        return CacheUsageStatus.ACCOUNTING_MISSING
    return value if isinstance(value, CacheUsageStatus) else CacheUsageStatus(value)


def _has_any_key(source: dict[str, Any], *keys: str) -> bool:
    return any(key in source for key in keys)
