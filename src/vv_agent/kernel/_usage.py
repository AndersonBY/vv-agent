"""Pure provider-usage normalization and the shared model-attempt ledger."""

from __future__ import annotations

from typing import Any

from ._execution_values import (
    CacheUsage,
    CacheUsageStatus,
    ModelCallOperation,
    ModelCallRecord,
    ModelCallStatus,
    TaskTokenUsage,
    TokenUsage,
    UsageSource,
    _aggregate_cache_usage,
)
from ._state import ExecutionState


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
        ("cache_creation_tokens",),
        ("input_tokens_details", "cache_creation_tokens"),
        ("prompt_tokens_details", "cache_creation_tokens"),
        ("prompt_tokens_details", "cache_write_tokens"),
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
    identities: set[str] = set()
    totals: dict[str, int | None] = dict.fromkeys(("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens"), 0)
    for record in model_calls:
        if not isinstance(record, ModelCallRecord):
            raise TypeError("model_call must be a ModelCallRecord")
        if record.call_id in identities:
            raise ValueError("model_call_id_duplicate")
        identities.add(record.call_id)
        for name, total in totals.items():
            value = getattr(record.usage, name)
            totals[name] = None if total is None or value is None else total + value
    return TaskTokenUsage(
        input_tokens=totals["input_tokens"],
        output_tokens=totals["output_tokens"],
        total_tokens=totals["total_tokens"],
        reasoning_tokens=totals["reasoning_tokens"],
        cache_usage=_aggregate_cache_usage([record.usage.cache_usage for record in model_calls]),
        model_calls=list(model_calls),
    )


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


def model_usage(value: dict[str, Any] | None):
    # Framework receipt annotations are not provider token-usage fields.
    return normalize_token_usage({k: v for k, v in (value or {}).items() if not k.startswith("session_")})


def project_model_usage(state: ExecutionState, turn_id: str) -> tuple[TaskTokenUsage, list[dict[str, Any]]]:
    """Project every started model attempt in retained operation/attempt order."""
    turn = state.turns[turn_id]
    calls: list[ModelCallRecord] = []
    ledger: list[dict[str, Any]] = []
    for oid, op in state.operations.items():
        if op.turn_id != turn_id or op.kind != "model":
            continue
        for number, attempt in op.attempts.items():
            if not attempt.started:
                continue
            purpose = attempt.plan._payload["purpose"]
            cycle = (
                attempt.plan._payload["request"]["metadata"]
                .get("vv_session", {})
                .get("cycle_index", int(oid.rsplit("/", 1)[1]) if purpose in {"primary", "output_repair"} else 1)
            )
            ledger.append(
                {
                    "purpose": purpose,
                    "operation_id": oid,
                    "attempt": number,
                    "usage": model_usage(attempt.result._payload["usage"] if attempt.result else None).to_dict(),
                    "status": "completed"
                    if attempt.result and not attempt.result._payload["result"].get("error_code")
                    else "failed"
                    if attempt.result
                    else "ambiguous",
                }
            )
            error_code = attempt.result._payload["result"].get("error_code") if attempt.result else "model_outcome_unknown"
            calls.append(
                ModelCallRecord(
                    call_id=f"{oid}/{number}",
                    operation_id=oid,
                    attempt=number,
                    operation={
                        "compaction": ModelCallOperation.MEMORY_COMPACTION,
                        "session_memory": ModelCallOperation.SESSION_MEMORY,
                        "output_repair": ModelCallOperation.OUTPUT_REPAIR,
                    }.get(purpose, ModelCallOperation.AGENT_CYCLE),
                    cycle_index=cycle,
                    backend=attempt.plan._payload["request"]
                    .get("metadata", {})
                    .get("vv_session", {})
                    .get(
                        "model_binding",
                        turn.start._payload["definition"]["model_binding"]
                        .get("internal", {})
                        .get(purpose, turn.start._payload["definition"]["model_binding"]),
                    )["backend"],
                    model=attempt.plan._payload["request"]["model"],
                    status=ModelCallStatus.FAILED
                    if attempt.result and error_code
                    else ModelCallStatus.COMPLETED
                    if attempt.result
                    else ModelCallStatus.AMBIGUOUS,
                    usage=model_usage(attempt.result.payload["usage"] if attempt.result else None),
                    error_code=error_code,
                )
            )
    return summarize_task_token_usage(calls), ledger
