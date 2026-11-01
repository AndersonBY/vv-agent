"""Exact counts and bounded tokenizer work, measured without a wall clock."""

import gc
import json
import random
import weakref
from collections import defaultdict
from functools import wraps
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
import pytest
import regex
from deepseek_tokenizer import deepseek_tokenizer
from support.token_counting import PROVIDER_MODELS, provider_counting_settings
from support.token_counting import legacy_count as _legacy_count
from vv_llm.chat_clients import utils
from vv_llm.settings import Server

from vv_agent.memory import token_utils


def legacy_count(messages, **kwargs):
    """The local reference is vv-llm's original whole count with no token server."""
    with patch.object(utils.settings, "token_server", None):
        return _legacy_count(messages, **kwargs)


@pytest.fixture(autouse=True, params=[False, True], ids=["no-server", "global-server"])
def token_server_settings(request, monkeypatch):
    from vv_llm.settings import settings

    assert utils.settings is settings
    server = Server(host="token-server.invalid", port=8188, url="https://token-server.invalid") if request.param else None
    monkeypatch.setattr(settings, "token_server", server)

    def unexpected_http(request):
        pytest.fail(f"Local counting attempted HTTP: {request.method} {request.url}")

    with httpx2.Client(transport=httpx2.MockTransport(unexpected_http)) as client:

        def post(*args, proxy=None, **kwargs):
            assert proxy is None
            return client.post(*args, **kwargs)

        monkeypatch.setattr(utils.httpx2, "post", post)
        yield
    assert settings.token_server is server


@pytest.fixture(autouse=True)
def empty_token_cache():
    token_utils._TOKEN_CACHE.clear()
    utils._qwen_tokenizer_cache.clear()
    yield
    token_utils._TOKEN_CACHE.clear()
    utils._qwen_tokenizer_cache.clear()


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-v4-pro",
        "deepseek-flash",
        "gpt-3.5-turbo",
        "gpt-4o",
        "o1-mini",
        "o3-mini",
        "gpt-5.4",
        "unknown-model",
        "qwen-plus",
        "qwen3.6-27b",
        "MiniMax-M2",
        "abab",
        "moonshot-v1",
        "kimi-k3",
        "gemini-3-pro",
        "claude-sonnet-4",
        "stepfun",
        "glm-4-plus",
    ],
)
@pytest.mark.parametrize("native_multimodal", [True, False])
def test_totals_match_original_across_every_vv_llm_family(monkeypatch, model, native_multimodal):
    # Exercise the installed implementations with endpointless settings, never credentials or I/O.
    backend = SimpleNamespace(models=defaultdict(lambda: SimpleNamespace(endpoints=[], id=model)))
    monkeypatch.setattr(type(utils.settings), "get_backend", lambda self, _: backend)
    if model.startswith(("MiniMax", "abab", "moonshot", "kimi", "gemini", "claude", "stepfun", "glm")):
        # This suite compares local algorithms; remote server behavior has its own test.
        monkeypatch.setattr(utils, "settings", SimpleNamespace(token_server=None, get_backend=lambda _: backend))
    messages = [
        {"role": "system", "content": "Rules and 中文。"},
        {"role": "user", "content": "hello!\n"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "empty", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "empty", "content": "done"},
        {
            "role": "assistant",
            "content": "Inspect.",
            "tool_calls": [{"id": "a", "type": "function", "function": {"name": "read", "arguments": '{"x":"中文"}'}}],
        },
        {"role": "tool", "tool_call_id": "a", "content": "结果\uff01\nline two\n"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}},
                {"type": "image_url", "image_url": {"url": "https://example.invalid/b.png"}},
                {"type": "text", "text": "again"},
                {"type": "input_audio", "input_audio": {"data": "unused", "format": "wav"}},
            ],
        },
    ]
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    for end in (1, len(messages)):
        expected = legacy_count(messages[:end], model=model, tools=tools, native_multimodal=native_multimodal)
        assert (
            token_utils.count_messages_tokens(messages[:end], model=model, tools=tools, native_multimodal=native_multimodal)
            == expected
        )
    for content in ("", "\n", "\r\n", [], [{"type": "image_url", "image_url": {"url": "https://example.invalid/c.png"}}]):
        boundary_messages = [{"role": "user", "content": content}, *messages]
        assert token_utils.count_messages_tokens(
            boundary_messages, model=model, tools=tools, native_multimodal=native_multimodal
        ) == legacy_count(boundary_messages, model=model, tools=tools, native_multimodal=native_multimodal)


def test_plain_message_counts_are_not_additive_and_have_no_fixed_separator_cost():
    for model in ("deepseek-v4-pro", "gpt-4o", "gpt-3.5-turbo", "qwen-plus"):
        plain = [{"role": "user", "content": s} for s in ("hello", "world")]
        punctuated = [{"role": "user", "content": s} for s in ("hello!", "world")]
        delta_plain = legacy_count(plain, model=model) - sum(legacy_count([m], model=model) for m in plain)
        delta_punctuation = legacy_count(punctuated, model=model) - sum(legacy_count([m], model=model) for m in punctuated)
        assert delta_plain == 1 and delta_punctuation == 0


def test_deepseek_exact_newline_boundaries_with_adversarial_unicode():
    rng = random.Random(20261029)
    vocabulary = [
        "hello",
        "World",
        "plain text",
        "你好",
        "123",
        " ",
        "\n",
        "\t",
        "\r\n",
        "!",
        ".",
        "'s",
        "<\uff5cUser\uff5c>",
        "\u0301",
        "😀",
    ]
    for _ in range(100):
        messages = [{"role": "user", "content": "".join(rng.choices(vocabulary, k=10)).lstrip("\r\n") or "x"} for _ in range(3)]
        assert token_utils.count_messages_tokens(messages, model="deepseek-v4-pro") == legacy_count(
            messages, model="deepseek-v4-pro"
        )


@pytest.mark.parametrize("boundary", ["", "\n", "\r\nleading", "\n\n", " ", "\t", "\u0301"])
def test_deepseek_unusual_boundaries_remain_exact(boundary):
    messages = [{"role": "user", "content": s} for s in ("hello!\n", boundary, "world")]
    assert token_utils.count_messages_tokens(messages, model="deepseek-v4-pro") == legacy_count(messages, model="deepseek-v4-pro")


def test_only_new_or_changed_deepseek_text_is_tokenized(monkeypatch):
    encoded = []
    encode = deepseek_tokenizer.encode

    @wraps(encode)
    def counted(text, *args, **kwargs):
        encoded.append(len(text))
        return encode(text, *args, **kwargs)

    monkeypatch.setattr(deepseek_tokenizer, "encode", counted)
    messages = [{"role": "user", "content": "Mixed 中文 facts. " * 4000}]
    token_utils.count_messages_tokens(messages, model="deepseek-v4-pro")
    for _ in range(4):
        encoded.clear()
        new = {"role": "user", "content": "new 信息"}
        messages.append(new)
        token_utils.count_messages_tokens(messages, model="deepseek-v4-pro")
        assert sum(encoded) <= 2 * len(new["content"]) + 1
    encoded.clear()
    messages[0] = {"role": "user", "content": "changed"}
    token_utils.count_messages_tokens(messages, model="deepseek-v4-pro")
    assert sum(encoded) == 2 * len("changed") + 1


def test_tokenizer_identity_and_model_change_invalidate_cached_counts(monkeypatch):
    messages = [{"role": "user", "content": "identity"}]
    original = token_utils.count_messages_tokens(messages, model="deepseek-v4-pro")
    encode = deepseek_tokenizer.encode

    def changed(*args, **kwargs):
        return [*encode(*args, **kwargs), 1]

    monkeypatch.setattr(deepseek_tokenizer, "encode", changed)
    assert token_utils.count_messages_tokens(messages, model="deepseek-v4-pro") == original + 1
    assert token_utils.count_messages_tokens(messages, model="deepseek-v3") == original + 1


@pytest.mark.parametrize("model", ["deepseek-flash", "gpt-3.5-turbo", "gpt-4o", "qwen-plus"])
def test_replaced_functions_and_unreviewed_versions_keep_the_whole_local_path(monkeypatch, model):
    seen = []

    def count(text, model, *, use_token_server_first=True):
        assert not use_token_server_first
        seen.append(text)
        return len(text) // 3

    monkeypatch.setattr(utils, "get_token_counts", count)
    messages = [{"role": "user", "content": "hello"}, {"role": "user", "content": "world"}]
    if model != "qwen-plus":
        enabled, reason = token_utils.incremental_token_counting_status(model)
        assert not enabled and "token counting functions have been replaced" in reason
    assert token_utils.count_messages_tokens(messages, model=model) == len("hello\nworld") // 3
    assert seen == ["hello\nworld"]
    seen.clear()
    monkeypatch.setattr(token_utils, "_deepseek_versions", lambda: ("future", "0.3.0"))
    monkeypatch.setattr(token_utils, "_tiktoken_versions", lambda: ("future", "0.14.0"))
    assert token_utils.count_messages_tokens(messages, model=model) == len("hello\nworld") // 3
    assert seen == ["hello\nworld"]


def test_public_status_and_version_fallback_warning_once_per_family(monkeypatch, caplog):
    from vv_agent.memory import incremental_token_counting_status

    for model in ("deepseek-flash", "gpt-4o", "gpt-3.5-turbo"):
        enabled, reason = incremental_token_counting_status(model)
        assert enabled, reason
    assert incremental_token_counting_status("qwen-plus") == (False, "local qwen; whole-conversation counting")
    monkeypatch.setattr(token_utils, "_deepseek_versions", lambda: ("future", "0.3.0"))
    monkeypatch.setattr(token_utils, "_VERSION_FALLBACK_LOGGED", set())
    enabled, reason = incremental_token_counting_status("deepseek-flash")
    assert not enabled and "future" in reason and "0.3.0" in reason
    for model in ("deepseek-flash", "deepseek-v4-pro", "deepseek-flash"):
        messages = [{"role": "user", "content": s} for s in ("hello", "world")]
        assert token_utils.count_messages_tokens(messages, model=model) == legacy_count(messages, model=model)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "whole conversation" in caplog.text


@pytest.mark.parametrize("model", ["gpt-3.5-turbo", "gpt-4o", "unknown-model"])
def test_tiktoken_only_new_or_changed_text_is_tokenized(monkeypatch, model):
    encoded = []
    count = utils.get_token_counts

    @wraps(count)
    def counted(text, *args, **kwargs):
        encoded.append(len(text))
        return count(text, *args, **kwargs)

    monkeypatch.setattr(utils, "get_token_counts", counted)
    assert token_utils.incremental_token_counting_status(model)[0]
    messages = [{"role": "user", "content": "Mixed 中文 facts. " * 4000}]
    token_utils.count_messages_tokens(messages, model=model)
    for _ in range(4):
        encoded.clear()
        messages.append({"role": "user", "content": "new 信息"})
        token_utils.count_messages_tokens(messages, model=model)
        assert sum(encoded) <= 2 * len("new 信息") + 1
    encoded.clear()
    messages[0] = {"role": "user", "content": "changed"}
    token_utils.count_messages_tokens(messages, model=model)
    assert sum(encoded) == 2 * len("changed") + 1


@pytest.mark.parametrize("model", ["gpt-3.5-turbo", "gpt-4o"])
def test_tiktoken_boundary_proof_matches_pre_tokens_and_token_ids(model):
    encoding = utils.get_gpt_35_encoding() if model == "gpt-3.5-turbo" else utils.get_gpt_4o_encoding()
    assert all(not any(c.isspace() for c in text) for text in encoding.special_tokens_set)
    rng = random.Random(2026102903)
    alphabet = [chr(rng.randrange(0x110000)) for _ in range(500)]
    alphabet = [c for c in alphabet if not 0xD800 <= ord(c) <= 0xDFFF]
    vocabulary = [*alphabet, "a", "中文", "\n", "\r\n", " \t", "x\n \ny", "'s", "e\u0301", "/!", "123", "\u2028", "\u0085"]
    for _ in range(500):
        parts = ["".join(rng.choices(vocabulary, k=12)) for _ in range(3)]
        parts = [p if p and not p[0].isspace() and p[0] != "/" else "x" + p for p in parts]
        chunks = [p + "\n" for p in parts[:-1]] + parts[-1:]
        text = "\n".join(parts)
        assert regex.findall(encoding._pat_str, text) == [s for p in chunks for s in regex.findall(encoding._pat_str, p)]
        assert encoding.encode(text) == [t for p in chunks for t in encoding.encode(p)]
        messages = [{"role": "user", "content": p} for p in parts]
        assert token_utils.count_messages_tokens(messages, model=model) == legacy_count(messages, model=model)
    for special in encoding.special_tokens_set:
        messages = [{"role": "user", "content": s} for s in ("hello", special, "world")]
        assert token_utils.count_messages_tokens(messages, model=model) == legacy_count(messages, model=model)


@pytest.mark.parametrize("model", ["gpt-3.5-turbo", "gpt-4o"])
@pytest.mark.parametrize("boundary", ["", "\n", " \n", " \t", "\r", "/", "/\n", "\u2028", "\u0085"])
def test_tiktoken_unsafe_boundaries_use_the_original_whole_call(monkeypatch, model, boundary):
    messages = [{"role": "user", "content": s} for s in ("hello!\n", boundary, "world")]
    expected = legacy_count(messages, model=model)
    count = utils.get_token_counts
    calls = []

    @wraps(count)
    def counted(text, *args, **kwargs):
        calls.append(text)
        assert kwargs["use_token_server_first"] is False
        return count(text, *args, **kwargs)

    monkeypatch.setattr(utils, "get_token_counts", counted)
    assert token_utils.count_messages_tokens(messages, model=model) == expected
    unsafe = not boundary or boundary[0].isspace() or (model == "gpt-4o" and boundary[0] == "/")
    if unsafe:
        assert calls == ["\n".join(m["content"] for m in messages)]
    else:
        assert len(calls) == 6


def test_tiktoken_unverified_versions_warn_once_for_each_encoder(monkeypatch, caplog):
    monkeypatch.setattr(token_utils, "_tiktoken_versions", lambda: ("0.8.4", "0.15.0"))
    monkeypatch.setattr(token_utils, "_VERSION_FALLBACK_LOGGED", set())
    for model in ("gpt-4o", "o3-mini", "unknown", "gpt-3.5-turbo", "gpt-3.5-turbo"):
        enabled, reason = token_utils.incremental_token_counting_status(model)
        assert not enabled and "tiktoken=0.15.0" in reason
        messages = [{"role": "user", "content": s} for s in ("hello", "world")]
        assert token_utils.count_messages_tokens(messages, model=model) == legacy_count(messages, model=model)
    assert len(caplog.records) == 2
    assert all(r.levelname == "WARNING" for r in caplog.records)


def test_deepseek_boundary_proof_matches_token_ids_and_pre_tokens():
    tokenizer = deepseek_tokenizer._tokenizer
    assert not tokenizer.byte_level_add_prefix_space and not tokenizer.byte_level_use_regex
    assert all(not any(c.isspace() for c in text) for text in tokenizer.added_token_ids)
    rng = random.Random(2026102902)
    alphabet = [chr(rng.randrange(0x110000)) for _ in range(400)]
    alphabet = [c for c in alphabet if not 0xD800 <= ord(c) <= 0xDFFF]
    vocabulary = [*alphabet, " \t", "\r\n", "\n\n", "a\n \nb", "e\u0301", "中文", "123", "'s", "/!"]
    for _ in range(200):
        parts = ["".join(rng.choices(vocabulary, k=8)).lstrip("\r\n") or "x" for _ in range(3)]
        combined = "\n".join(parts)
        chunks = [part + "\n" for part in parts[:-1]] + parts[-1:]
        assert tokenizer._pre_tokenize(combined) == [p for chunk in chunks for p in tokenizer._pre_tokenize(chunk)]
        assert deepseek_tokenizer.encode(combined) == [token for chunk in chunks for token in deepseek_tokenizer.encode(chunk)]
    # Added tokens are split before pre-tokenization; exercise every one and partial spellings.
    for text in tokenizer.added_token_ids:
        for parts in ([text, "x"], ["x!", text], [text[: len(text) // 2], text[len(text) // 2 :]]):
            assert deepseek_tokenizer.encode("\n".join(parts)) == [
                *deepseek_tokenizer.encode(parts[0] + "\n"),
                *deepseek_tokenizer.encode(parts[1]),
            ]


@pytest.mark.parametrize("model", PROVIDER_MODELS)
@pytest.mark.parametrize("unverified", [False, True])
@pytest.mark.parametrize("native_multimodal", [False, True])
def test_provider_families_use_the_no_endpoint_count_without_http(monkeypatch, model, unverified, native_multimodal):
    backend, setting, endpoint = provider_counting_settings(model)
    monkeypatch.setattr(type(utils.settings), "get_backend", lambda self, _: backend)
    monkeypatch.setattr(type(utils.settings), "get_endpoint", lambda self, _: endpoint)
    messages = [
        {"role": "user", "content": "Mixed 中文 facts. " * 400},
        {
            "role": "assistant",
            "content": "Read.",
            "tool_calls": [{"id": "a", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "a", "content": "结果"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "https://image.invalid/a.png"}},
            ],
        },
    ]
    tools = [{"type": "function", "function": {"name": "read"}}]
    with patch.object(utils.settings, "token_server", None):
        expected = utils.get_message_token_counts(messages, model=model, tools=tools, native_multimodal=native_multimodal)
    if unverified:
        monkeypatch.setattr(token_utils, "_tiktoken_versions", lambda: ("future", "0.14.0"))
    setting.endpoints = ["fake-enabled"]

    def unexpected_http(request):
        pytest.fail(f"Provider counting attempted HTTP: {request.method} {request.url}")

    def unexpected_client(*args, **kwargs):
        pytest.fail("Compaction initialized a provider SDK")

    # Covers raw HTTP and SDK-created clients; BaseException failures cannot hide in estimates/retries.
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", lambda self, request: unexpected_http(request))
    for name in ("Anthropic", "AnthropicVertex", "get_token_with_cache"):
        monkeypatch.setattr(utils, name, unexpected_client)
    with httpx2.Client(transport=httpx2.MockTransport(unexpected_http)) as client:

        def post(*args, proxy=None, **kwargs):
            assert proxy is None
            return client.post(*args, **kwargs)

        monkeypatch.setattr(utils.httpx2, "post", post)
        for _ in range(2):
            assert (
                token_utils.count_messages_tokens(messages, model=model, tools=tools, native_multimodal=native_multimodal)
                == expected
            )
    assert setting.endpoints == ["fake-enabled"] and endpoint.enabled


@pytest.mark.parametrize("model", PROVIDER_MODELS)
def test_provider_local_method_status_and_incremental_work(monkeypatch, model):
    family = token_utils._local_tokenizer_family(model)
    enabled, reason = token_utils.incremental_token_counting_status(model)
    assert family in reason
    assert enabled is (not model.startswith(("MiniMax", "abab")))
    if not enabled:
        assert reason == "local int(len(text) / 1.33); whole-conversation counting"
        return
    encoding = utils.get_gpt_35_encoding() if family == "cl100k_base" else utils.get_gpt_4o_encoding()
    encoded = []
    original = type(encoding).encode

    def counted(self, text, *args, **kwargs):
        encoded.append(len(text))
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(type(encoding), "encode", counted)
    messages = [{"role": "user", "content": "Mixed 中文 facts. " * 4000}]
    token_utils.count_messages_tokens(messages, model=model)
    for content in ("new 信息", "another result"):
        encoded.clear()
        messages.append({"role": "user", "content": content})
        token_utils.count_messages_tokens(messages, model=model)
        assert sum(encoded) == 2 * len(content) + 1


def test_minimax_rounds_the_whole_text_and_tools_separately():
    messages = [{"role": "user", "content": "ab"}, {"role": "user", "content": "cd"}]
    assert token_utils.count_messages_tokens(messages, model="MiniMax-M2.5", tools=["x"]) == (
        int(len("ab\ncd") / 1.33) + int(len(str(["x"])) / 1.33)
    )


@pytest.mark.parametrize(
    ("model", "method"),
    [
        ("abab", "int(len(text) / 1.33)"),
        ("MiniMax-M2.5", "int(len(text) / 1.33)"),
        ("moonshot-v1", "cl100k_base"),
        ("kimi-k3", "cl100k_base"),
        ("gemini-3-pro", "cl100k_base"),
        ("claude-sonnet-4", "o200k_base"),
        ("stepfun", "cl100k_base"),
        ("glm-4-flash", "cl100k_base"),
    ],
)
def test_mapping_matches_installed_vv_llm_no_endpoint_branches(monkeypatch, model, method):
    backend, _, _ = provider_counting_settings(model)
    monkeypatch.setattr(type(utils.settings), "get_backend", lambda self, _: backend)
    assert token_utils._local_tokenizer_family(model) == method
    for text in ("", "a", "Mixed 中文 facts.\n", " \r\n/leading", "😀e\u0301"):
        if method == "int(len(text) / 1.33)":
            expected = int(len(text) / 1.33)
        else:
            encoding = utils.get_gpt_35_encoding() if method == "cl100k_base" else utils.get_gpt_4o_encoding()
            expected = len(encoding.encode(text))
        assert utils.get_token_counts(text, model, use_token_server_first=False) == expected
        assert token_utils._count_local_text(text, model) == expected


def test_fallback_rounds_once_and_scans_only_new_serialized_messages(monkeypatch):
    def unavailable(**_kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(utils, "get_message_token_counts", unavailable)
    messages = [{"role": "user", "content": "中文 a\uff01" * 4000}]
    expected = legacy_count(messages)
    assert token_utils.count_messages_tokens(messages) == expected
    visited = []
    count = token_utils._estimate_characters

    def scanned(text):
        visited.append(len(text))
        return count(text)

    monkeypatch.setattr(token_utils, "_estimate_characters", scanned)
    messages.append({"role": "user", "content": "你a"})
    expected = legacy_count(messages)
    visited.clear()
    assert token_utils.count_messages_tokens(messages) == expected
    assert visited == [len(json.dumps(messages[-1], ensure_ascii=False, default=str))]
    assert token_utils.count_messages_tokens([]) == 0


def test_fallback_matches_whole_json_for_nested_and_non_json_values(monkeypatch):
    monkeypatch.setattr(utils, "get_message_token_counts", lambda **_: -1)
    messages = [{"role": "user", "content": "你"}, {"nested": ["a", "\uff01", {"bytes": b"data", "set": {1, 2}}]}]
    for end in range(1, len(messages) + 1):
        assert token_utils.count_messages_tokens(messages[:end]) == legacy_count(messages[:end])


def test_fallback_keeps_the_old_estimate_for_lone_surrogates(monkeypatch):
    monkeypatch.setattr(utils, "get_message_token_counts", lambda **_: -1)
    messages = [{"role": "user", "content": "old \ud800 value 中文"}, {"nested": ["\udfff"]}]
    assert token_utils.count_messages_tokens(messages) == legacy_count(messages)


def test_cache_is_bounded_and_does_not_retain_message_bodies(monkeypatch):
    class Body(str):
        pass

    monkeypatch.setattr(token_utils, "_TOKEN_CACHE_LIMIT", 4)
    body = Body("unique body 中文")
    reference = weakref.ref(body)
    messages = [{"role": "user", "content": body}]
    token_utils.count_messages_tokens(messages, model="deepseek-v4-pro")
    del body, messages
    gc.collect()
    assert reference() is None
    for index in range(8):
        token_utils.count_messages_tokens([{"role": "user", "content": f"message {index}"}], model="deepseek-v4-pro")
    assert len(token_utils._TOKEN_CACHE) == 4
    assert all(isinstance(item, int) for value in token_utils._TOKEN_CACHE.values() for item in value)
