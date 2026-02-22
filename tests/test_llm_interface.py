from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

from openai.types.chat import ChatCompletionMessageParam

from vv_agent.constants import TODO_WRITE_TOOL_NAME
from vv_agent.llm.vv_llm_client import EndpointTarget, VVLlmClient
from vv_agent.types import Message


class _FakeUsage:
    def __init__(self, *, prompt_tokens: int = 3, completion_tokens: int = 2) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def model_dump(self, *, exclude_none: bool = True) -> dict[str, int]:
        del exclude_none
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


class _FakeChatClient:
    behavior_by_endpoint: ClassVar[dict[str, Callable[[dict[str, Any]], Any] | Any]] = {}
    seen_calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, endpoint_id: str):
        self.endpoint_id = endpoint_id

    def create_completion(self, **kwargs: Any) -> Any:
        _FakeChatClient.seen_calls.append({"endpoint_id": self.endpoint_id, **kwargs})
        behavior = self.behavior_by_endpoint[self.endpoint_id]
        if callable(behavior):
            callable_behavior = cast(Callable[[dict[str, Any]], Any], behavior)
            return callable_behavior(kwargs)
        return behavior


def _fake_create_chat_client(*, endpoint_id: str, **kwargs: Any) -> _FakeChatClient:
    del kwargs
    return _FakeChatClient(endpoint_id=endpoint_id)


def _passthrough_format_messages(*, messages: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
    del kwargs
    return list(messages)


def test_llm_failover_to_next_endpoint(monkeypatch) -> None:
    def failing_call(kwargs: dict[str, Any]) -> Any:
        del kwargs
        raise RuntimeError("first endpoint down")

    _FakeChatClient.behavior_by_endpoint = {
        "first": failing_call,
        "second": SimpleNamespace(content="ok from backup", tool_calls=[], reasoning_content=None, usage=_FakeUsage()),
    }
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)

    llm = VVLlmClient(
        endpoint_targets=[
            EndpointTarget(endpoint_id="first", api_key="k1", api_base="https://first.example/v1"),
            EndpointTarget(endpoint_id="second", api_key="k2", api_base="https://second.example/v1"),
        ],
        backend="openai",
        selected_model="gpt-4o-mini",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    response = llm.complete(model="gpt-4o-mini", messages=[Message(role="user", content="hello")], tools=[])

    assert response.content == "ok from backup"
    assert response.raw["used_endpoint_id"] == "second"


def test_llm_stream_aggregates_tool_calls(monkeypatch) -> None:
    chunk_1 = SimpleNamespace(
        usage=None,
        content="hello ",
        reasoning_content=None,
        tool_calls=[
            SimpleNamespace(
                index=0,
                id="tc1",
                function=SimpleNamespace(
                    name=TODO_WRITE_TOOL_NAME,
                    arguments='{"todos":[{"title":"a","status":"pending",',
                ),
            )
        ],
    )
    chunk_2 = SimpleNamespace(
        usage=_FakeUsage(),
        content="world",
        reasoning_content=None,
        tool_calls=[
            SimpleNamespace(
                index=0,
                id=None,
                function=SimpleNamespace(name=None, arguments='"priority":"medium"}]}'),
            )
        ],
    )

    def stream_call(kwargs: dict[str, Any]) -> Any:
        assert kwargs["stream"] is True
        return [chunk_1, chunk_2]

    _FakeChatClient.behavior_by_endpoint = {"stream": stream_call}
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)

    llm = VVLlmClient(
        endpoint_targets=[EndpointTarget(endpoint_id="stream", api_key="k", api_base="https://stream.example/v1")],
        backend="moonshot",
        selected_model="kimi-k2.5",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    response = llm.complete(model="kimi-k2.5", messages=[Message(role="user", content="hi")], tools=[])

    assert response.content == "hello world"
    assert response.raw["stream_collected"] is True
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == TODO_WRITE_TOOL_NAME
    assert response.tool_calls[0].arguments["todos"][0]["title"] == "a"


def test_llm_stream_collects_reasoning_content(monkeypatch) -> None:
    chunk_1 = SimpleNamespace(
        usage=None,
        content="",
        reasoning_content="step-1",
        tool_calls=[],
    )
    chunk_2 = SimpleNamespace(
        usage=_FakeUsage(),
        content="final",
        reasoning_content="|step-2",
        tool_calls=[],
    )

    def stream_call(kwargs: dict[str, Any]) -> Any:
        assert kwargs["stream"] is True
        return [chunk_1, chunk_2]

    _FakeChatClient.behavior_by_endpoint = {"stream-reasoning": stream_call}
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)

    llm = VVLlmClient(
        endpoint_targets=[
            EndpointTarget(
                endpoint_id="stream-reasoning",
                api_key="k",
                api_base="https://stream-reasoning.example/v1",
            )
        ],
        backend="moonshot",
        selected_model="kimi-k2.5",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    response = llm.complete(model="kimi-k2.5", messages=[Message(role="user", content="hi")], tools=[])
    assert response.content == "final"
    assert response.raw["reasoning_content"] == "step-1|step-2"


def test_resolve_request_options_aligns_claude_thinking_profile() -> None:
    llm = VVLlmClient(endpoint_targets=[])
    options = llm._resolve_request_options("claude-opus-4-6-thinking", stream=True, endpoint_type="openai")
    assert options.model == "claude-opus-4-6"
    assert options.temperature == 1.0
    assert options.max_tokens == 20000
    assert options.thinking == {"type": "enabled", "budget_tokens": 16000}


def test_resolve_request_options_aligns_gemini3_profile() -> None:
    llm = VVLlmClient(endpoint_targets=[])
    options = llm._resolve_request_options("gemini-3-pro", stream=True, endpoint_type="openai")
    assert options.model == "gemini-3-pro-preview"
    assert options.temperature == 1.0
    assert options.is_gemini_3_model is True
    assert options.extra_body == {
        "extra_body": {
            "google": {
                "thinking_config": {
                    "thinkingLevel": "high",
                    "include_thoughts": True,
                }
            }
        }
    }


def test_llm_stream_request_payload_aligns_qwen_thinking(monkeypatch) -> None:
    chunk = SimpleNamespace(
        usage=_FakeUsage(),
        content="done",
        reasoning_content=None,
        tool_calls=[],
    )

    def stream_call(kwargs: dict[str, Any]) -> Any:
        assert kwargs["stream"] is True
        assert kwargs["extra_body"] == {"enable_thinking": True}
        return [chunk]

    _FakeChatClient.behavior_by_endpoint = {"qwen": stream_call}
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)

    llm = VVLlmClient(
        endpoint_targets=[EndpointTarget(endpoint_id="qwen", api_key="k", api_base="https://qwen.example/v1")],
        backend="qwen",
        selected_model="qwen3-32b-thinking",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    response = llm.complete(model="qwen3-32b-thinking", messages=[Message(role="user", content="hi")], tools=[])
    assert response.content == "done"


def test_llm_stream_aggregates_tool_calls_without_index(monkeypatch) -> None:
    chunk_1 = SimpleNamespace(
        usage=None,
        content="",
        reasoning_content=None,
        tool_calls=[
            SimpleNamespace(
                index=0,
                id="tc_missing_index",
                function=SimpleNamespace(
                    name=TODO_WRITE_TOOL_NAME,
                    arguments='{"todos":[{"title":"x",',
                ),
            )
        ],
    )
    chunk_2 = SimpleNamespace(
        usage=_FakeUsage(),
        content="",
        reasoning_content=None,
        tool_calls=[
            SimpleNamespace(
                index=None,
                id=None,
                function=SimpleNamespace(name=None, arguments='"status":"pending"}]}'),
            )
        ],
    )

    def stream_call(kwargs: dict[str, Any]) -> Any:
        assert kwargs["stream"] is True
        return [chunk_1, chunk_2]

    _FakeChatClient.behavior_by_endpoint = {"missing-index": stream_call}
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)

    llm = VVLlmClient(
        endpoint_targets=[
            EndpointTarget(
                endpoint_id="missing-index",
                api_key="k",
                api_base="https://missing-index.example/v1",
            )
        ],
        backend="moonshot",
        selected_model="kimi-k2.5",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    response = llm.complete(model="kimi-k2.5", messages=[Message(role="user", content="hi")], tools=[])
    assert response.tool_calls[0].arguments["todos"][0]["title"] == "x"


def test_prepare_messages_for_minimax_converts_extra_system_messages() -> None:
    messages = [
        {"role": "system", "content": "base system"},
        {"role": "system", "name": "memory_summary", "content": "summary"},
        {"role": "assistant", "content": "next"},
    ]

    prepared = VVLlmClient._prepare_messages_for_model(cast(list[ChatCompletionMessageParam], messages), "MiniMax-M2.5")
    assert prepared[0]["role"] == "system"
    assert prepared[1]["role"] == "user"
    assert prepared[1]["content"] == "[memory_summary]\nsummary"
    assert prepared[2]["role"] == "assistant"


def test_prepare_messages_for_lowercase_minimax_converts_extra_system_messages() -> None:
    messages = [
        {"role": "system", "content": "base system"},
        {"role": "system", "name": "memory_summary", "content": "summary"},
    ]

    prepared = VVLlmClient._prepare_messages_for_model(
        cast(list[ChatCompletionMessageParam], messages),
        "minimax-m2.5",
    )
    assert prepared[0]["role"] == "system"
    assert prepared[1]["role"] == "user"
    assert prepared[1]["content"] == "[memory_summary]\nsummary"


def test_prepare_messages_for_non_minimax_keeps_multi_system_messages() -> None:
    messages = [
        {"role": "system", "content": "base system"},
        {"role": "system", "name": "memory_summary", "content": "summary"},
    ]
    prepared = VVLlmClient._prepare_messages_for_model(
        cast(list[ChatCompletionMessageParam], messages),
        "kimi-k2.5",
    )
    assert prepared[0]["role"] == "system"
    assert prepared[1]["role"] == "system"


def test_should_use_stream_is_case_insensitive_for_minimax() -> None:
    assert VVLlmClient._should_use_stream("MiniMax-M2.5") is True
    assert VVLlmClient._should_use_stream("minimax-m2.5") is True


def test_build_message_payload_keeps_reasoning_only_for_last_assistant_by_default() -> None:
    llm = VVLlmClient(endpoint_targets=[], backend="openai", selected_model="gpt-4o")
    payload = llm._build_message_payload(
        [
            Message(role="system", content="sys"),
            Message(role="assistant", content="first", reasoning_content="old-thought"),
            Message(role="tool", content="result", tool_call_id="call_1"),
            Message(role="assistant", content="second", reasoning_content="latest-thought"),
            Message(role="user", content="continue"),
        ],
        preserve_reasoning_chain=False,
    )
    assert "reasoning_content" not in payload[1]
    assert payload[3]["reasoning_content"] == "latest-thought"


def test_build_message_payload_preserves_reasoning_chain_for_reasoning_models() -> None:
    llm = VVLlmClient(endpoint_targets=[], backend="moonshot", selected_model="kimi-k2.5")
    payload = llm._build_message_payload(
        [
            Message(role="system", content="sys"),
            Message(role="assistant", content="", tool_calls=[], reasoning_content="old-thought"),
            Message(role="tool", content="result", tool_call_id="call_1"),
            Message(role="assistant", content="", tool_calls=[], reasoning_content=None),
        ],
        preserve_reasoning_chain=True,
    )
    assert payload[1]["reasoning_content"] == "old-thought"
    assert payload[3]["reasoning_content"] == ""


def test_moonshot_request_preserves_reasoning_for_all_assistant_turns(monkeypatch) -> None:
    chunk = SimpleNamespace(
        usage=_FakeUsage(),
        content="done",
        reasoning_content=None,
        tool_calls=[],
    )

    def stream_call(kwargs: dict[str, Any]) -> Any:
        assistant_messages = [msg for msg in kwargs["messages"] if msg.get("role") == "assistant"]
        assert len(assistant_messages) == 2
        assert assistant_messages[0]["reasoning_content"] == "old-thought"
        assert assistant_messages[1]["reasoning_content"] == "latest-thought"
        return [chunk]

    _FakeChatClient.behavior_by_endpoint = {"moonshot-reasoning": stream_call}
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)

    llm = VVLlmClient(
        endpoint_targets=[
            EndpointTarget(
                endpoint_id="moonshot-reasoning",
                api_key="k",
                api_base="https://moonshot.example/v1",
            )
        ],
        backend="moonshot",
        selected_model="kimi-k2.5",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    response = llm.complete(
        model="kimi-k2.5",
        messages=[
            Message(role="system", content="sys"),
            Message(role="assistant", content="first", reasoning_content="old-thought"),
            Message(role="user", content="continue"),
            Message(role="assistant", content="second", reasoning_content="latest-thought"),
            Message(role="user", content="next turn"),
        ],
        tools=[],
    )
    assert response.content == "done"


def test_llm_estimates_usage_when_backend_missing_usage(monkeypatch) -> None:
    response_without_usage = SimpleNamespace(content="done", tool_calls=[], reasoning_content=None, usage=None)

    _FakeChatClient.behavior_by_endpoint = {"usage-missing": response_without_usage}
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.get_token_counts", lambda text, model, use_token_server_first=False: 10)

    llm = VVLlmClient(
        endpoint_targets=[EndpointTarget(endpoint_id="usage-missing", api_key="k", api_base="https://u.example/v1")],
        backend="openai",
        selected_model="gpt-4o-mini",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    result = llm.complete(model="gpt-4o-mini", messages=[Message(role="user", content="hello")], tools=[])
    assert result.raw["usage"]["prompt_tokens"] == 10
    assert result.raw["usage"]["completion_tokens"] == 10
    assert result.raw["usage"]["total_tokens"] == 20


def test_llm_stream_collects_raw_content_blocks(monkeypatch) -> None:
    chunk_1 = SimpleNamespace(
        usage=None,
        content="",
        reasoning_content=None,
        raw_content={"type": "thinking_delta", "thinking": "step-1"},
        tool_calls=[],
    )
    chunk_2 = SimpleNamespace(
        usage=None,
        content="",
        reasoning_content=None,
        raw_content={"type": "signature_delta", "signature": "sig-1"},
        tool_calls=[],
    )
    chunk_3 = SimpleNamespace(
        usage=_FakeUsage(),
        content="done",
        reasoning_content=None,
        raw_content={"type": "text_delta", "text": "visible text"},
        tool_calls=[],
    )

    def stream_call(kwargs: dict[str, Any]) -> Any:
        assert kwargs["stream"] is True
        return [chunk_1, chunk_2, chunk_3]

    _FakeChatClient.behavior_by_endpoint = {"stream-raw-content": stream_call}
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)

    llm = VVLlmClient(
        endpoint_targets=[
            EndpointTarget(
                endpoint_id="stream-raw-content",
                api_key="k",
                api_base="https://stream-raw-content.example/v1",
            )
        ],
        backend="moonshot",
        selected_model="kimi-k2.5",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    result = llm.complete(model="kimi-k2.5", messages=[Message(role="user", content="hello")], tools=[])
    raw_content = result.raw["raw_content"]
    assert result.content == "done"
    assert isinstance(raw_content, list)
    assert raw_content[0]["type"] == "thinking"
    assert raw_content[0]["thinking"] == "step-1"
    assert raw_content[0]["signature"] == "sig-1"
    assert raw_content[1]["type"] == "text"
    assert raw_content[1]["text"] == "visible text"


def test_llm_debug_dump_writes_request_messages(monkeypatch, tmp_path: Path) -> None:
    response = SimpleNamespace(content="ok", tool_calls=[], reasoning_content=None, usage=_FakeUsage())
    _FakeChatClient.behavior_by_endpoint = {"dump-endpoint": response}
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)

    llm = VVLlmClient(
        endpoint_targets=[
            EndpointTarget(
                endpoint_id="dump-endpoint",
                api_key="k",
                api_base="https://dump.example/v1",
                model_id="gpt/4o-mini",
            )
        ],
        backend="openai",
        selected_model="gpt/4o-mini",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
        debug_dump_dir=str(tmp_path / "dumps"),
    )

    result = llm.complete(model="gpt/4o-mini", messages=[Message(role="user", content="hello")], tools=[])
    assert result.content == "ok"

    dump_files = sorted((tmp_path / "dumps").glob("request_*.json"))
    assert len(dump_files) == 1
    assert dump_files[0].name == "request_001_gpt_4o-mini.json"
    payload = dump_files[0].read_text(encoding="utf-8")
    assert '"request_index": 1' in payload
    assert '"message_count": 1' in payload
