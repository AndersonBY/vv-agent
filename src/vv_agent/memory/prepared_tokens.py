"""Prepare real local counting dependencies before compaction or replay."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from . import token_utils


@dataclass(frozen=True, slots=True)
class PreparedTokenCounter:
    """One model's bound dependencies; message counting uses token_utils' algorithm.

    The no-initialization-I/O domain is the Message.to_openai_message() I-JSON
    payloads produced by compaction, with JSON tools. Arbitrary user objects and vv-llm's VectorVein
    attachment loader are not part of that domain.
    """

    model: str
    family: str
    count_text: Callable[[str], int]
    format_messages: Callable[..., list[Any]]
    image_tokens: int
    incremental_status: tuple[bool, str]
    cache_identity: tuple[Any, ...]

    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: Any = None,
        native_multimodal: bool = True,
    ) -> int:
        effective_model = self.model if model is None else model or "gpt-4o"
        if effective_model != self.model:
            raise ValueError("prepared token counter model does not match")
        return token_utils._count_messages_tokens(
            messages,
            model=self.model,
            tools=tools,
            native_multimodal=native_multimodal,
            _prepared=self,
        )


def prepare_token_counter(model: str = "") -> PreparedTokenCounter:
    """Opt in to bound dependencies; preparation failures remain retryable.

    Call in the outer assembly, in the original model-use order. In particular,
    Qwen aliases bind vv-llm's shared selected object, never a per-model factory.
    Replaced counters require a trusted prepared equivalent and are not captured
    as if merely retaining their callable could establish CPU-only counting.
    """
    from vv_llm.chat_clients import utils
    from vv_llm.types.enums import BackendType

    if any(
        function.__module__ != utils.__name__ or hasattr(function, "__wrapped__")
        for function in (utils.get_token_counts, utils.get_message_token_counts, utils.format_messages)
    ):
        raise ValueError("replaced token counting functions require a prepared equivalent")
    effective_model = model or "gpt-4o"
    family = token_utils._local_tokenizer_family(effective_model)
    status = token_utils.incremental_token_counting_status(effective_model)
    encode: Callable[[str], Any] | None = None
    if family == "int(len(text) / 1.33)":
        count_text = token_utils._count_minimax_text
    else:
        if family == "deepseek":
            from deepseek_tokenizer import deepseek_tokenizer

            encode = deepseek_tokenizer.encode
        elif family == "qwen":
            encode = utils._get_qwen_tokenizer(effective_model).encode
        else:
            encoding = utils.get_gpt_35_encoding() if family == "cl100k_base" else utils.get_gpt_4o_encoding()
            # Encoding.encode can import regex on its first special-token check.
            import regex  # noqa: F401

            encode = encoding.encode

        def count_text(text: str) -> int:
            return len(encode(text))

    token_utils._warn_incremental_fallback(family, status[1])
    return PreparedTokenCounter(
        model=effective_model,
        family=family,
        count_text=count_text,
        format_messages=partial(utils.format_messages, backend=BackendType.OpenAI),
        image_tokens=utils.calculate_image_tokens(2048, 2048, effective_model),
        incremental_status=status,
        cache_identity=(family, effective_model, utils.get_token_counts, encode, utils.get_message_token_counts),
    )
