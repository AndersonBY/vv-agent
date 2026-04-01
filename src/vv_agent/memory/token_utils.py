from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


def count_tokens(text: str | dict[str, Any], model: str = "") -> int:
    """Count tokens for a text payload with vv-llm fallback handling.

    Prefer passing normalized strings. Structured payloads may be serialized
    differently by vendor tokenizers versus the local JSON fallback path.
    """
    if not text:
        return 0

    try:
        from vv_llm.chat_clients.utils import get_token_counts

        count = get_token_counts(text, model=model, use_token_server_first=False)
        if isinstance(count, int) and count > 0:
            return count
    except Exception:
        logger.debug("vv-llm token counting unavailable; using local estimate", exc_info=True)

    raw = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    return _estimate_tokens(raw)


def count_messages_tokens(
    messages: list[dict[str, Any]],
    *,
    model: str = "",
    tools: Any = None,
    native_multimodal: bool = True,
) -> int:
    """Count tokens for OpenAI-compatible messages."""
    if not messages:
        return 0

    try:
        from vv_llm.chat_clients.utils import get_message_token_counts

        kwargs: dict[str, Any] = {
            "messages": messages,
            "model": model or "gpt-4o",
            "native_multimodal": native_multimodal,
        }
        if tools is not None:
            kwargs["tools"] = tools

        count = get_message_token_counts(**kwargs)
        if isinstance(count, int) and count >= 0:
            return count
    except Exception:
        logger.debug("vv-llm message token counting unavailable; using local estimate", exc_info=True)

    raw = json.dumps(messages, ensure_ascii=False, default=str)
    return _estimate_tokens(raw)


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

    if configured > 0 and derived > 0:
        return min(configured, derived)
    if configured > 0:
        return configured
    return derived


@lru_cache(maxsize=256)
def resolve_model_token_limits(model: str) -> tuple[int | None, int | None]:
    """Resolve context window and output budget from vv-llm model defaults."""
    normalized_model = str(model or "").strip()
    if not normalized_model:
        return None, None

    try:
        from vv_llm.types import defaults as vv_defaults
    except Exception:
        logger.debug("vv-llm model defaults unavailable", exc_info=True)
        return None, None

    for name in dir(vv_defaults):
        if not name.endswith("_MODELS"):
            continue
        models = getattr(vv_defaults, name, None)
        if not isinstance(models, dict):
            continue

        setting = models.get(normalized_model)
        if not isinstance(setting, dict):
            setting = next(
                (
                    candidate
                    for candidate in models.values()
                    if isinstance(candidate, dict) and candidate.get("id") == normalized_model
                ),
                None,
            )
        if isinstance(setting, dict):
            return _coerce_positive_int(setting.get("context_length")), _coerce_positive_int(
                setting.get("max_output_tokens")
            )

    return None, None


def _estimate_tokens(text: str) -> int:
    """Local CJK-aware token estimate aligned with vv-llm fallback behavior."""
    if not text:
        return 0

    cjk_chars = sum(
        1
        for char in text
        if "\u4e00" <= char <= "\u9fff" or "\u3000" <= char <= "\u303f" or "\uff00" <= char <= "\uffef"
    )
    ascii_chars = len(text) - cjk_chars
    return max(1, int(cjk_chars * 1.5 + ascii_chars * 0.25))


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
