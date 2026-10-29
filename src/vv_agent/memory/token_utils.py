from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from threading import Lock
from typing import Any, cast

logger = logging.getLogger(__name__)

# Only digests, tokenizer identities and integers survive a call, never message bodies.
_TOKEN_CACHE_LIMIT = 4096
_TOKEN_CACHE: OrderedDict[tuple[Any, ...], tuple[int, int]] = OrderedDict()
_TOKEN_CACHE_LOCK = Lock()
_VERSION_FALLBACK_LOGGED: set[str] = set()
_VERIFIED_DEEPSEEK_VERSIONS = frozenset({("0.8.3", "0.3.0"), ("0.8.4", "0.3.0")})
_VERIFIED_TIKTOKEN_VERSIONS = frozenset({("0.8.3", "0.14.0"), ("0.8.4", "0.14.0")})
# vv-llm 0.8.4 get_token_counts: use its no-endpoint fallback, never its provider branch.
_PROVIDER_LOCAL_FALLBACKS = (
    (("abab", "MiniMax"), "int(len(text) / 1.33)"),
    (("moonshot", "kimi", "gemini", "stepfun", "glm"), "cl100k_base"),
    (("claude",), "o200k_base"),
)


def _cached_token_data(text: str, identity: tuple[Any, ...], calculate: Callable[[], tuple[int, int]]) -> tuple[int, int]:
    key = (*identity, len(text), sha256(text.encode("utf-8", errors="surrogatepass")).digest())
    with _TOKEN_CACHE_LOCK:
        cached = _TOKEN_CACHE.get(key)
        if cached is not None:
            _TOKEN_CACHE.move_to_end(key)
            return cached
    result = calculate()
    with _TOKEN_CACHE_LOCK:
        _TOKEN_CACHE[key] = result
        _TOKEN_CACHE.move_to_end(key)
        while len(_TOKEN_CACHE) > _TOKEN_CACHE_LIMIT:
            _TOKEN_CACHE.popitem(last=False)
    return result


@lru_cache(maxsize=1)
def _deepseek_versions() -> tuple[str, str]:
    return version("vv-llm"), version("deepseek-tokenizer")


@lru_cache(maxsize=1)
def _tiktoken_versions() -> tuple[str, str]:
    return version("vv-llm"), version("tiktoken")


def _local_tokenizer_family(model: str) -> str:
    for prefixes, fallback in _PROVIDER_LOCAL_FALLBACKS:
        if model.startswith(prefixes):
            return fallback
    if model.startswith("deepseek"):
        return "deepseek"
    if model.startswith("qwen"):
        return "qwen"
    # vv-llm routes every remaining name to o200k, except this exact spelling.
    return "cl100k_base" if model == "gpt-3.5-turbo" else "o200k_base"


def incremental_token_counting_status(model: str) -> tuple[bool, str]:
    """Return incremental eligibility and local method; unsafe boundaries still use a whole count."""
    family = _local_tokenizer_family(model)
    if family in ("int(len(text) / 1.33)", "qwen"):
        return False, f"local {family}; whole-conversation counting"
    package = "deepseek-tokenizer" if family == "deepseek" else "tiktoken"
    try:
        versions = _deepseek_versions() if family == "deepseek" else _tiktoken_versions()
    except PackageNotFoundError:
        return False, f"local {family}; token counting dependency is not installed"
    verified = _VERIFIED_DEEPSEEK_VERSIONS if family == "deepseek" else _VERIFIED_TIKTOKEN_VERSIONS
    if versions not in verified:
        return False, f"unverified versions: vv-llm={versions[0]}, {package}={versions[1]}; whole-conversation local {family}"
    from vv_llm.chat_clients import utils

    if utils.get_message_token_counts.__module__ != utils.__name__ or utils.get_token_counts.__module__ != utils.__name__:
        return False, f"local {family}; token counting functions have been replaced"
    return True, f"verified {family} newline boundaries"


def _count_local_text(text: str, model: str) -> int:
    from vv_llm.chat_clients import utils

    for prefixes, fallback in _PROVIDER_LOCAL_FALLBACKS:
        if model.startswith(prefixes):
            if fallback == "int(len(text) / 1.33)":
                return int(len(text) / 1.33)
            encoding = utils.get_gpt_35_encoding() if fallback == "cl100k_base" else utils.get_gpt_4o_encoding()
            return len(encoding.encode(text))
    return utils.get_token_counts(text, model, use_token_server_first=False)


def _message_token_parts(messages: list[dict[str, Any]], *, native_multimodal: bool) -> tuple[list[str], int]:
    from vv_llm.chat_clients import utils
    from vv_llm.types.enums import BackendType

    formatted = utils.format_messages(cast(Any, messages), backend=BackendType.OpenAI, native_multimodal=native_multimodal)
    texts = []
    images = 0
    for message in formatted:
        content = message["content"]
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item["type"] == "text":
                    texts.append(item["text"])
                elif isinstance(item, dict) and item["type"].startswith("image"):
                    images += 1
    return texts, images


def _count_incremental_messages(messages: list[dict[str, Any]], *, model: str, tools: Any, native_multimodal: bool) -> int | None:
    """Reuse exact newline-terminated counts; unsupported boundaries use a whole count."""
    enabled, reason = incremental_token_counting_status(model)
    family = _local_tokenizer_family(model)
    if not enabled:
        if reason.startswith("unverified versions:"):
            with _TOKEN_CACHE_LOCK:
                first = family not in _VERSION_FALLBACK_LOGGED
                _VERSION_FALLBACK_LOGGED.add(family)
            if first:
                logger.warning("Incremental token counting disabled for %s; using whole conversation: %s", family, reason)
        return None
    from vv_llm.chat_clients import utils

    if family == "deepseek":
        from deepseek_tokenizer import deepseek_tokenizer

        encode = deepseek_tokenizer.encode
    else:
        encoding = utils.get_gpt_35_encoding() if family == "cl100k_base" else utils.get_gpt_4o_encoding()
        encode = encoding.encode
    texts, images = _message_token_parts(messages, native_multimodal=native_multimodal)
    if any(not isinstance(text, str) or not text for text in texts):
        return None
    # DeepSeek stops at the first newline run; tiktoken can span whitespace after it.
    if family == "deepseek" and any(text[0] in "\r\n" for text in texts):
        return None
    # o200k's punctuation suffix can consume slashes after the separator as well.
    if family != "deepseek" and any(text[0].isspace() or (family == "o200k_base" and text[0] == "/") for text in texts):
        return None
    identity = (family, model, utils.get_token_counts, encode, utils.get_message_token_counts)
    total = 0
    for index, text in enumerate(texts):
        plain, terminated = _cached_token_data(
            text,
            identity,
            lambda text=text: (
                _count_local_text(text, model),
                _count_local_text(text + "\n", model),
            ),
        )
        # The verified pre-tokenizers cannot cross these guarded newline cuts.
        total += plain if index == len(texts) - 1 else terminated
    if images:
        total += images * (utils.calculate_image_tokens(2048, 2048, model) if native_multimodal else 1)
    if tools:
        tool_text = str(tools)
        total += _cached_token_data(tool_text, (*identity, "tools"), lambda: (_count_local_text(tool_text, model), 0))[0]
    return total


def _count_whole_local_messages(messages: list[dict[str, Any]], *, model: str, tools: Any, native_multimodal: bool) -> int:
    """Use local text methods with vv-llm's message formatting and image estimate."""
    from vv_llm.chat_clients import utils

    texts, images = _message_token_parts(messages, native_multimodal=native_multimodal)
    combined_text = "\n".join(texts)
    total = _count_local_text(combined_text, model) if combined_text else 0
    if images:
        total += images * (utils.calculate_image_tokens(2048, 2048, model) if native_multimodal else 1)
    if tools:
        total += _count_local_text(str(tools), model)
    return total


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
    """Estimate compaction context size locally, independent of provider endpoints."""
    if not messages:
        return 0

    try:
        from vv_llm.chat_clients import utils

        kwargs: dict[str, Any] = {
            "messages": messages,
            "model": model or "gpt-4o",
            "native_multimodal": native_multimodal,
        }
        if tools is not None:
            kwargs["tools"] = tools

        count = _count_incremental_messages(messages, model=kwargs["model"], tools=tools, native_multimodal=native_multimodal)
        if count is None:
            # Preserve an explicitly replaced message counter instead of bypassing it.
            if utils.get_message_token_counts.__module__ == utils.__name__:
                count = _count_whole_local_messages(
                    messages, model=kwargs["model"], tools=tools, native_multimodal=native_multimodal
                )
            else:
                count = utils.get_message_token_counts(**kwargs)
        if isinstance(count, int) and count >= 0:
            return count
    except Exception:
        logger.debug("vv-llm message token counting unavailable; using local estimate", exc_info=True)

    # json.dumps(list) inserts two brackets and ", " between individual dumps.
    cjk_chars, ascii_chars = 0, 2 * len(messages)
    for message in messages:
        raw = json.dumps(message, ensure_ascii=False, default=str)
        cjk, ascii_ = _cached_token_data(raw, ("json-estimate",), lambda raw=raw: _estimate_characters(raw))
        cjk_chars += cjk
        ascii_chars += ascii_
    return max(1, int(cjk_chars * 1.5 + ascii_chars * 0.25))


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
            return _coerce_positive_int(setting.get("context_length")), _coerce_positive_int(setting.get("max_output_tokens"))

    return None, None


def _estimate_tokens(text: str) -> int:
    """Local CJK-aware token estimate aligned with vv-llm fallback behavior."""
    if not text:
        return 0

    cjk_chars, ascii_chars = _estimate_characters(text)
    return max(1, int(cjk_chars * 1.5 + ascii_chars * 0.25))


def _estimate_characters(text: str) -> tuple[int, int]:
    cjk_chars = sum(
        1 for char in text if "\u4e00" <= char <= "\u9fff" or "\u3000" <= char <= "\u303f" or "\uff00" <= char <= "\uffef"
    )
    return cjk_chars, len(text) - cjk_chars


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
