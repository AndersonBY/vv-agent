from __future__ import annotations

import json

from vv_agent.memory import token_utils


def test_count_tokens_uses_vv_llm_when_available(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_get_token_counts(text: str, *, model: str = "", use_token_server_first: bool = True) -> int:
        calls["text"] = text
        calls["model"] = model
        calls["use_token_server_first"] = use_token_server_first
        return 123

    monkeypatch.setattr("vv_llm.chat_clients.utils.get_token_counts", fake_get_token_counts)

    assert token_utils.count_tokens("hello", model="gpt-5.4") == 123
    assert calls == {
        "text": "hello",
        "model": "gpt-5.4",
        "use_token_server_first": False,
    }


def test_count_tokens_falls_back_to_cjk_aware_estimate(monkeypatch) -> None:
    def fake_get_token_counts(*_args, **_kwargs) -> int:
        raise RuntimeError("tokenizer unavailable")

    monkeypatch.setattr("vv_llm.chat_clients.utils.get_token_counts", fake_get_token_counts)

    text = "你好hello"
    assert token_utils.count_tokens(text, model="gpt-5.4") == token_utils._estimate_tokens(text)


def test_count_tokens_accepts_dict_payload(monkeypatch) -> None:
    def fake_get_token_counts(*_args, **_kwargs) -> int:
        raise RuntimeError("tokenizer unavailable")

    monkeypatch.setattr("vv_llm.chat_clients.utils.get_token_counts", fake_get_token_counts)

    payload = {"role": "user", "content": "hello"}
    expected = token_utils._estimate_tokens(json.dumps(payload, ensure_ascii=False, default=str))

    assert token_utils.count_tokens(payload, model="gpt-5.4") == expected


def test_count_messages_tokens_uses_vv_llm_when_available(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_get_message_token_counts(**kwargs: object) -> int:
        calls.update(kwargs)
        return 77

    monkeypatch.setattr("vv_llm.chat_clients.utils.get_message_token_counts", fake_get_message_token_counts)

    messages = [{"role": "user", "content": "hello"}]
    assert token_utils.count_messages_tokens(messages, model="gpt-5.4") == 77
    assert calls["messages"] == messages
    assert calls["model"] == "gpt-5.4"
    assert calls["native_multimodal"] is True


def test_count_messages_tokens_falls_back_to_json_estimate(monkeypatch) -> None:
    def fake_get_message_token_counts(**_kwargs: object) -> int:
        raise RuntimeError("tokenizer unavailable")

    monkeypatch.setattr("vv_llm.chat_clients.utils.get_message_token_counts", fake_get_message_token_counts)

    messages = [{"role": "user", "content": "你好"}]
    expected = token_utils._estimate_tokens(json.dumps(messages, ensure_ascii=False, default=str))

    assert token_utils.count_messages_tokens(messages, model="gpt-5.4") == expected


def test_count_tokens_returns_zero_for_empty_text() -> None:
    assert token_utils.count_tokens("", model="gpt-5.4") == 0


def test_estimate_tokens_handles_cjk_and_ascii_mix() -> None:
    assert token_utils._estimate_tokens("你好") == 3
    assert token_utils._estimate_tokens("hello") == 1
    assert token_utils._estimate_tokens("你好hello") == 4


def test_resolve_model_token_limits_reads_vv_llm_defaults() -> None:
    token_utils.resolve_model_token_limits.cache_clear()

    context_window, max_output_tokens = token_utils.resolve_model_token_limits("gpt-5.4")

    assert isinstance(context_window, int)
    assert context_window > 0
    assert isinstance(max_output_tokens, int)
    assert max_output_tokens > 0


def test_resolve_model_token_limits_returns_none_for_unknown_model() -> None:
    token_utils.resolve_model_token_limits.cache_clear()

    assert token_utils.resolve_model_token_limits("definitely-not-a-real-model") == (None, None)


def test_resolve_model_token_limits_returns_none_for_empty_model() -> None:
    token_utils.resolve_model_token_limits.cache_clear()

    assert token_utils.resolve_model_token_limits("") == (None, None)
