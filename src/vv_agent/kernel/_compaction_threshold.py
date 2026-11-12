"""Pure compaction capacity calculation and integer coercion."""

from __future__ import annotations

from typing import Any


def compute_compaction_threshold(
    *,
    configured_threshold: Any,
    model_context_window: Any,
    reserved_output_tokens: Any,
    autocompact_buffer_tokens: Any,
) -> int:
    """Resolve the effective compaction threshold.

    The user-configured threshold remains the primary ceiling. Model-derived
    limits only lower that ceiling when the active model has a smaller safe
    prompt budget.
    """
    configured = _coerce_positive_int(configured_threshold) or 0
    context_window = _coerce_positive_int(model_context_window) or 0
    reserved = max(_coerce_int_like(reserved_output_tokens), 0)
    buffer_tokens = max(_coerce_int_like(autocompact_buffer_tokens), 0)

    derived = 0
    if context_window > 0:
        effective_window = max(context_window - reserved, 0)
        derived = max(effective_window - buffer_tokens, 0)

    if context_window > 0:
        return min(configured, derived) if configured > 0 else derived
    return configured


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        coerced = int(value)
        return coerced if coerced > 0 else None
    if isinstance(value, str):
        try:
            coerced = int(value.strip())
        except ValueError:
            return None
        return coerced if coerced > 0 else None
    return None


def _coerce_int_like(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0
