from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import httpx
import pytest
from openai.types.chat import ChatCompletionMessageParam
from openai.types.completion_usage import PromptTokensDetails
from vv_llm.types import APIConnectionError, APIStatusError
from vv_llm.types.llm_parameters import Usage

from vv_agent import constants as constants_module
from vv_agent.llm.anthropic_prompt_cache import apply_claude_prompt_cache
from vv_agent.llm.vv_llm_client import EndpointTarget, VVLlmClient
from vv_agent.model_settings import ModelSettings, RetrySettings, ToolChoice
from vv_agent.runtime.token_usage import normalize_token_usage
from vv_agent.types import Message

TASK_LIST_TOOL_NAME = getattr(constants_module, "".join(("TO", "DO")) + "_WRITE_TOOL_NAME")


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
        raise APIConnectionError(request=httpx.Request("POST", "https://first.example/v1/chat/completions"))

    _FakeChatClient.behavior_by_endpoint = {
        "first": failing_call,
        "second": SimpleNamespace(content="ok from backup", tool_calls=[], reasoning_content=None, usage=_FakeUsage()),
    }
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)
    monkeypatch.setattr(VVLlmClient, "_should_use_stream", staticmethod(lambda model: False))

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


def test_llm_bridge_preserves_provider_reported_zero_cache_usage(monkeypatch) -> None:
    usage = Usage(
        prompt_tokens=11,
        completion_tokens=7,
        total_tokens=18,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
    )
    _FakeChatClient.behavior_by_endpoint = {
        "moonshot": SimpleNamespace(
            content="ok",
            tool_calls=[],
            reasoning_content=None,
            usage=usage,
        )
    }
    _FakeChatClient.seen_calls = []
    monkeypatch.setattr(
        "vv_agent.llm.vv_llm_client.create_chat_client",
        _fake_create_chat_client,
    )
    monkeypatch.setattr(
        "vv_agent.llm.vv_llm_client.format_messages",
        _passthrough_format_messages,
    )
    monkeypatch.setattr(
        VVLlmClient,
        "_should_use_stream",
        staticmethod(lambda model: False),
    )

    llm = VVLlmClient(
        endpoint_targets=[
            EndpointTarget(
                endpoint_id="moonshot",
                api_key="test-key",
                api_base="https://api.moonshot.cn/v1",
            )
        ],
        backend="moonshot",
        selected_model="kimi-k2.6",
        randomize_endpoints=False,
    )
    response = llm.complete(
        model="kimi-k2.6",
        messages=[Message(role="user", content="hello")],
        tools=[],
    )

    normalized = normalize_token_usage(
        response.raw["usage"],
        usage_source=response.raw["usage_source"],
    )
    assert response.raw["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert normalized.cache_usage.read_tokens == 0
    assert normalized.cache_usage.uncached_input_tokens == 11
    assert normalized.usage_source.value == "provider_reported"


def test_llm_retries_transient_status_on_the_same_endpoint(monkeypatch) -> None:
    attempts = 0

    def flaky_call(kwargs: dict[str, Any]) -> Any:
        nonlocal attempts
        del kwargs
        attempts += 1
        if attempts == 1:
            request = httpx.Request("POST", "https://first.example/v1/chat/completions")
            response = httpx.Response(429, request=request)
            raise APIStatusError("rate limited", response=response, body={"error": "rate limited"})
        return SimpleNamespace(content="ok", tool_calls=[], reasoning_content=None, usage=_FakeUsage())

    _FakeChatClient.behavior_by_endpoint = {"first": flaky_call}
    _FakeChatClient.seen_calls = []
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)
    monkeypatch.setattr(VVLlmClient, "_should_use_stream", staticmethod(lambda model: False))
    llm = VVLlmClient(
        endpoint_targets=[EndpointTarget(endpoint_id="first", api_key="k", api_base="https://first.example/v1")],
        randomize_endpoints=False,
        max_retries_per_endpoint=2,
        backoff_seconds=0.0,
    )

    response = llm.complete(model="demo", messages=[Message(role="user", content="hello")], tools=[])

    assert response.content == "ok"
    assert attempts == 2


def test_llm_auth_status_fails_over_without_same_endpoint_retry(monkeypatch) -> None:
    def unauthorized_call(kwargs: dict[str, Any]) -> Any:
        del kwargs
        request = httpx.Request("POST", "https://first.example/v1/chat/completions")
        response = httpx.Response(401, request=request)
        raise APIStatusError("unauthorized", response=response, body={"error": "unauthorized"})

    _FakeChatClient.behavior_by_endpoint = {
        "first": unauthorized_call,
        "second": SimpleNamespace(content="backup", tool_calls=[], reasoning_content=None, usage=_FakeUsage()),
    }
    _FakeChatClient.seen_calls = []
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)
    monkeypatch.setattr(VVLlmClient, "_should_use_stream", staticmethod(lambda model: False))
    llm = VVLlmClient(
        endpoint_targets=[
            EndpointTarget(endpoint_id="first", api_key="bad", api_base="https://first.example/v1"),
            EndpointTarget(endpoint_id="second", api_key="good", api_base="https://second.example/v1"),
        ],
        randomize_endpoints=False,
        max_retries_per_endpoint=3,
        backoff_seconds=0.0,
    )

    response = llm.complete(model="demo", messages=[Message(role="user", content="hello")], tools=[])

    assert response.content == "backup"
    assert [call["endpoint_id"] for call in _FakeChatClient.seen_calls] == ["first", "second"]


@pytest.mark.parametrize("failure", ["bad_request", "program_error"])
def test_llm_aborts_non_retryable_errors_without_failover(monkeypatch, failure: str) -> None:
    def failing_call(kwargs: dict[str, Any]) -> Any:
        del kwargs
        if failure == "bad_request":
            request = httpx.Request("POST", "https://first.example/v1/chat/completions")
            response = httpx.Response(400, request=request)
            raise APIStatusError("invalid request", response=response, body={"error": "invalid request"})
        raise RuntimeError("programming error")

    _FakeChatClient.behavior_by_endpoint = {
        "first": failing_call,
        "second": SimpleNamespace(content="must not run", tool_calls=[], reasoning_content=None, usage=_FakeUsage()),
    }
    _FakeChatClient.seen_calls = []
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)
    monkeypatch.setattr(VVLlmClient, "_should_use_stream", staticmethod(lambda model: False))
    llm = VVLlmClient(
        endpoint_targets=[
            EndpointTarget(endpoint_id="first", api_key="k1", api_base="https://first.example/v1"),
            EndpointTarget(endpoint_id="second", api_key="k2", api_base="https://second.example/v1"),
        ],
        randomize_endpoints=False,
        max_retries_per_endpoint=3,
        backoff_seconds=0.0,
    )

    with pytest.raises((APIStatusError, RuntimeError)):
        llm.complete(model="demo", messages=[Message(role="user", content="hello")], tools=[])

    assert [call["endpoint_id"] for call in _FakeChatClient.seen_calls] == ["first"]


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
                    name=TASK_LIST_TOOL_NAME,
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
    assert response.raw["usage_source"] == "provider_reported"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == TASK_LIST_TOOL_NAME
    assert response.tool_calls[0].arguments["todos"][0]["title"] == "a"


def test_llm_stream_emits_tool_call_progress_events(monkeypatch) -> None:
    chunk_1 = SimpleNamespace(
        usage=None,
        content="hello ",
        reasoning_content=None,
        tool_calls=[
            SimpleNamespace(
                index=0,
                id="tc1",
                function=SimpleNamespace(
                    name=TASK_LIST_TOOL_NAME,
                    arguments='{"todos":[{"title":"a",',
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
                function=SimpleNamespace(name=None, arguments='"status":"pending"}]}'),
            )
        ],
    )

    def stream_call(kwargs: dict[str, Any]) -> Any:
        assert kwargs["stream"] is True
        return [chunk_1, chunk_2]

    _FakeChatClient.behavior_by_endpoint = {"stream-events": stream_call}
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)

    llm = VVLlmClient(
        endpoint_targets=[EndpointTarget(endpoint_id="stream-events", api_key="k", api_base="https://stream.example/v1")],
        backend="moonshot",
        selected_model="kimi-k2.5",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )
    stream_events: list[dict[str, Any]] = []

    response = llm.complete(
        model="kimi-k2.5",
        messages=[Message(role="user", content="hi")],
        tools=[],
        stream_callback=stream_events.append,
    )

    assert response.content == "hello world"
    assert [event["event"] for event in stream_events] == [
        "assistant_delta",
        "tool_call_started",
        "tool_call_progress",
        "assistant_delta",
        "tool_call_progress",
    ]
    assert stream_events[1] == {
        "event": "tool_call_started",
        "tool_call_id": "tc1",
        "tool_call_index": 0,
        "function_name": TASK_LIST_TOOL_NAME,
        "arguments_chars": 23,
        "estimated_tokens": 6,
    }
    assert stream_events[-1]["tool_call_id"] == "tc1"
    assert stream_events[-1]["arguments_chars"] == len('{"todos":[{"title":"a","status":"pending"}]}')
    assert stream_events[-1]["estimated_tokens"] == 11


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


def test_llm_stream_preserves_gemini3_tool_call_extra_content(monkeypatch) -> None:
    chunk_1 = SimpleNamespace(
        usage=None,
        content="",
        reasoning_content=None,
        tool_calls=[
            SimpleNamespace(
                index=0,
                id="tc1",
                function=SimpleNamespace(
                    name="default_api:find_files",
                    arguments='{"path": "."}',
                ),
                extra_content={"google": {"thought_signature": "sig_123"}},
            )
        ],
    )
    chunk_2 = SimpleNamespace(
        usage=_FakeUsage(),
        content="done",
        reasoning_content=None,
        tool_calls=[],
    )

    def stream_call(kwargs: dict[str, Any]) -> Any:
        assert kwargs["stream"] is True
        return [chunk_1, chunk_2]

    _FakeChatClient.behavior_by_endpoint = {"gemini-stream": stream_call}
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)

    llm = VVLlmClient(
        endpoint_targets=[
            EndpointTarget(
                endpoint_id="gemini-stream",
                api_key="k",
                api_base="https://gemini.example/v1",
            )
        ],
        backend="gemini",
        selected_model="gemini-3-pro-preview",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    response = llm.complete(
        model="gemini-3-pro-preview",
        messages=[Message(role="user", content="hi")],
        tools=[],
    )

    assert response.content == "done"
    assert response.tool_calls[0].extra_content == {"google": {"thought_signature": "sig_123"}}
    assert response.raw["tool_call_extra_content"] == {
        "tc1": {"google": {"thought_signature": "sig_123"}}
    }


def test_resolve_request_options_aligns_claude_thinking_profile() -> None:
    llm = VVLlmClient(endpoint_targets=[])
    options = llm._resolve_request_options("claude-opus-4-7-thinking", stream=True, endpoint_type="openai")
    assert options.model == "claude-opus-4-7"
    assert options.temperature == 1.0
    assert options.max_tokens == 20000
    assert options.thinking == {"type": "enabled", "budget_tokens": 16000}


def test_llm_rejects_per_request_extra_headers() -> None:
    llm = VVLlmClient(
        endpoint_targets=[EndpointTarget(endpoint_id="demo", api_key="k", api_base="https://example.test/v1")]
    )

    with pytest.raises(ValueError, match="extra_headers is not supported"):
        llm.complete(
            model="demo-model",
            messages=[Message(role="user", content="hello")],
            tools=[],
            model_settings=ModelSettings(extra_headers={"x-request": "value"}),
        )


def test_llm_rejects_per_request_extra_args() -> None:
    llm = VVLlmClient(
        endpoint_targets=[EndpointTarget(endpoint_id="demo", api_key="k", api_base="https://example.test/v1")]
    )

    with pytest.raises(ValueError, match="extra_args is not supported"):
        llm.complete(
            model="demo-model",
            messages=[Message(role="user", content="hello")],
            tools=[],
            model_settings=ModelSettings(extra_args={"extra_query": {"region": "test"}}),
        )


def test_tool_choice_semantics_reach_the_real_provider_request(monkeypatch) -> None:
    response = SimpleNamespace(content="ok", tool_calls=[], reasoning_content=None, usage=_FakeUsage())
    _FakeChatClient.behavior_by_endpoint = {"tool-choice": response}
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)
    monkeypatch.setattr(VVLlmClient, "_should_use_stream", staticmethod(lambda model: False))

    llm = VVLlmClient(
        endpoint_targets=[
            EndpointTarget(endpoint_id="tool-choice", api_key="k", api_base="https://example.test/v1")
        ],
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )
    tools: list[dict[str, object]] = [
        {
            "type": "function",
            "function": {"name": name, "description": name, "parameters": {"type": "object"}},
        }
        for name in ("lookup", "write")
    ]

    llm.complete(
        model="demo-model",
        messages=[Message(role="user", content="hello")],
        tools=tools,
        model_settings=ModelSettings(tool_choice=ToolChoice.tool("lookup")),
    )
    named_call = _FakeChatClient.seen_calls[-1]
    assert named_call["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in named_call["tools"]] == ["lookup"]

    llm.complete(
        model="demo-model",
        messages=[Message(role="user", content="hello")],
        tools=tools,
        model_settings=ModelSettings(tool_choice=ToolChoice.none()),
    )
    none_call = _FakeChatClient.seen_calls[-1]
    assert "tool_choice" not in none_call
    assert "tools" not in none_call

    with pytest.raises(ValueError, match="unknown tool: missing"):
        llm.complete(
            model="demo-model",
            messages=[Message(role="user", content="hello")],
            tools=tools,
            model_settings=ModelSettings(tool_choice=ToolChoice.tool("missing")),
        )


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


def test_request_retry_settings_do_not_leak_into_the_next_request() -> None:
    llm = VVLlmClient(
        endpoint_targets=[],
        max_retries_per_endpoint=3,
        backoff_seconds=2.0,
    )

    overridden = llm._resolve_request_options(
        "gpt-4o",
        stream=False,
        endpoint_type="openai",
        model_settings=ModelSettings(retry=RetrySettings(max_attempts=7, backoff_seconds=0.25)),
    )
    defaulted = llm._resolve_request_options(
        "gpt-4o",
        stream=False,
        endpoint_type="openai",
        model_settings=None,
    )

    assert (overridden.max_attempts, overridden.backoff_seconds) == (7, 0.25)
    assert (defaulted.max_attempts, defaulted.backoff_seconds) == (3, 2.0)
    assert (llm.max_retries_per_endpoint, llm.backoff_seconds) == (3, 2.0)


def test_provider_timeout_default_and_model_override_reach_the_request(monkeypatch) -> None:
    response = SimpleNamespace(content="ok", tool_calls=[], reasoning_content=None, usage=_FakeUsage())
    _FakeChatClient.behavior_by_endpoint = {"timeout": response}
    _FakeChatClient.seen_calls = []
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)
    monkeypatch.setattr(VVLlmClient, "_should_use_stream", staticmethod(lambda model: False))
    llm = VVLlmClient(
        endpoint_targets=[EndpointTarget(endpoint_id="timeout", api_key="k", api_base="https://example.test/v1")],
        timeout_seconds=12.5,
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
    )

    llm.complete(model="demo-model", messages=[Message(role="user", content="default")], tools=[])
    assert _FakeChatClient.seen_calls[-1]["timeout"] == 12.5

    llm.complete(
        model="demo-model",
        messages=[Message(role="user", content="override")],
        tools=[],
        model_settings=ModelSettings(timeout_seconds=3.25),
    )
    assert _FakeChatClient.seen_calls[-1]["timeout"] == 3.25


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


def test_tool_call_incremental_uses_qwen_endpoint_prefix_for_new_models() -> None:
    assert (
        VVLlmClient._tool_call_incremental_enabled(
            model="qwen-next-custom",
            endpoint_type="qwen",
        )
        is True
    )


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
                    name=TASK_LIST_TOOL_NAME,
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


def test_should_use_stream_defaults_all_models_to_stream() -> None:
    assert VVLlmClient._should_use_stream("MiniMax-M2.5") is True
    assert VVLlmClient._should_use_stream("deepseek-v5-pro") is True
    assert VVLlmClient._should_use_stream("gpt-4o") is True
    assert VVLlmClient._should_use_stream("custom-enterprise-model") is True


def test_vv_llm_request_options_apply_public_model_settings() -> None:
    llm = VVLlmClient(endpoint_targets=[], backend="openai", selected_model="gpt-4o")

    options = llm._resolve_request_options(
        "gpt-4o",
        stream=False,
        endpoint_type="openai",
        model_settings=ModelSettings(
            temperature=0.3,
            max_tokens=512,
            reasoning={"effort": "low"},
            extra_body={"seed": 7},
            retry=RetrySettings(max_attempts=5, backoff_seconds=0.25),
        ),
    )

    assert options.temperature == 0.3
    assert options.max_tokens == 512
    assert options.reasoning_effort == "low"
    assert options.extra_body == {"seed": 7}
    assert options.max_attempts == 5
    assert options.backoff_seconds == 0.25
    assert llm.max_retries_per_endpoint == 3
    assert llm.backoff_seconds == 2.0


def test_deepseek_provider_defaults_new_models_to_reasoning_options() -> None:
    llm = VVLlmClient(endpoint_targets=[], backend="deepseek", selected_model="deepseek-v5-pro")

    options = llm._resolve_request_options(
        "deepseek-v5-pro",
        stream=False,
        endpoint_type="deepseek",
    )

    assert options.temperature is None
    assert options.thinking is None
    assert options.extra_body == {"thinking": {"type": "enabled"}}
    assert options.reasoning_effort == "max"


def test_kimi_k3_request_options_enforce_provider_profile_after_public_settings() -> None:
    llm = VVLlmClient(endpoint_targets=[], backend="moonshot", selected_model="kimi-k3")

    options = llm._resolve_request_options(
        "kimi-k3",
        stream=True,
        endpoint_type="moonshot",
        model_settings=ModelSettings(
            temperature=0.3,
            top_p=0.7,
            max_tokens=4096,
            reasoning={"effort": "low", "type": "enabled"},
            extra_body={
                "temperature": 0.4,
                "top_p": 0.8,
                "n": 2,
                "presence_penalty": 1,
                "frequency_penalty": 1,
                "thinking": {"type": "enabled"},
                "reasoning_effort": "low",
                "max_tokens": 1024,
                "max_completion_tokens": 2048,
                "provider_option": "kept",
            },
        ),
    )

    assert options.temperature is None
    assert options.top_p is None
    assert options.max_tokens is None
    assert options.max_completion_tokens == 4096
    assert options.thinking is None
    assert options.reasoning_effort == "max"
    assert options.extra_body == {"provider_option": "kept"}


def test_minimax_provider_defaults_new_models_to_reasoning_chain() -> None:
    llm = VVLlmClient(endpoint_targets=[], backend="minimax", selected_model="minimax-m3")

    assert llm._should_preserve_reasoning_chain("minimax-m3") is True


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
    first_assistant = cast(dict[str, Any], payload[1])
    second_assistant = cast(dict[str, Any], payload[3])
    assert "reasoning_content" not in first_assistant
    assert second_assistant["reasoning_content"] == "latest-thought"


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
    first_assistant = cast(dict[str, Any], payload[1])
    second_assistant = cast(dict[str, Any], payload[3])
    assert first_assistant["reasoning_content"] == "old-thought"
    assert second_assistant["reasoning_content"] == ""


def test_deepseek_request_preserves_reasoning_for_all_assistant_turns(monkeypatch) -> None:
    response = SimpleNamespace(
        usage=_FakeUsage(),
        content="done",
        reasoning_content=None,
        tool_calls=[],
    )

    def completion_call(kwargs: dict[str, Any]) -> Any:
        assistant_messages = [msg for msg in kwargs["messages"] if msg.get("role") == "assistant"]
        assert len(assistant_messages) == 2
        assert assistant_messages[0]["reasoning_content"] == "old-thought"
        assert assistant_messages[1]["reasoning_content"] == "latest-thought"
        return [response]

    _FakeChatClient.behavior_by_endpoint = {"deepseek-default": completion_call}
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)

    llm = VVLlmClient(
        endpoint_targets=[EndpointTarget(endpoint_id="deepseek-default", api_key="k", api_base="https://deepseek.example/v1")],
        backend="deepseek",
        selected_model="deepseek-v5-pro",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )
    llm.complete(
        model="deepseek-v5-pro",
        messages=[
            Message(role="system", content="sys"),
            Message(role="assistant", content="", reasoning_content="old-thought"),
            Message(role="tool", content="result", tool_call_id="call_1"),
            Message(role="assistant", content="", reasoning_content="latest-thought"),
            Message(role="user", content="continue"),
        ],
        tools=[],
    )


def test_moonshot_provider_defaults_new_models_to_reasoning_chain(monkeypatch) -> None:
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

    _FakeChatClient.behavior_by_endpoint = {"moonshot-new-reasoning": stream_call}
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)

    llm = VVLlmClient(
        endpoint_targets=[
            EndpointTarget(
                endpoint_id="moonshot-new-reasoning",
                api_key="k",
                api_base="https://moonshot.example/v1",
            )
        ],
        backend="moonshot",
        selected_model="kimi-k3",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    response = llm.complete(
        model="kimi-k3",
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


def test_kimi_k3_real_request_uses_fixed_profile_and_completion_limit(monkeypatch) -> None:
    chunk = SimpleNamespace(
        usage=_FakeUsage(),
        content="done",
        reasoning_content=None,
        tool_calls=[],
    )

    def stream_call(kwargs: dict[str, Any]) -> Any:
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "max_tokens" not in kwargs
        assert "thinking" not in kwargs
        assert kwargs["reasoning_effort"] == "max"
        assert kwargs["max_completion_tokens"] == 4096
        assert kwargs["extra_body"] == {"provider_option": "kept"}
        return [chunk]

    _FakeChatClient.behavior_by_endpoint = {"moonshot-k3-profile": stream_call}
    _FakeChatClient.seen_calls = []
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)

    llm = VVLlmClient(
        endpoint_targets=[
            EndpointTarget(
                endpoint_id="moonshot-k3-profile",
                api_key="k",
                api_base="https://moonshot.example/v1",
            )
        ],
        backend="moonshot",
        selected_model="kimi-k3",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )
    response = llm.complete(
        model="kimi-k3",
        messages=[Message(role="user", content="hi")],
        tools=[],
        model_settings=ModelSettings(
            temperature=0.3,
            top_p=0.7,
            max_tokens=4096,
            reasoning={"effort": "low", "type": "enabled"},
            extra_body={"thinking": {"type": "enabled"}, "provider_option": "kept"},
        ),
    )

    assert response.content == "done"


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

    _FakeChatClient.behavior_by_endpoint = {"usage-missing": [response_without_usage]}
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
    assert result.raw["usage_source"] == "estimated"


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
    _FakeChatClient.behavior_by_endpoint = {"dump-endpoint": [response]}
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


def test_claude_direct_request_adds_explicit_breakpoints(monkeypatch) -> None:
    chunk = SimpleNamespace(
        usage=_FakeUsage(),
        content="ok",
        reasoning_content=None,
        tool_calls=[],
    )

    def stream_call(kwargs: dict[str, Any]) -> Any:
        assert kwargs["stream"] is True
        return [chunk]

    _FakeChatClient.behavior_by_endpoint = {"anthropic-direct": stream_call}
    _FakeChatClient.seen_calls = []

    monkeypatch.setattr("vv_agent.llm.vv_llm_client.create_chat_client", _fake_create_chat_client)
    monkeypatch.setattr("vv_agent.llm.vv_llm_client.format_messages", _passthrough_format_messages)

    llm = VVLlmClient(
        endpoint_targets=[
            EndpointTarget(
                endpoint_id="anthropic-direct",
                api_key="k",
                api_base="https://api.anthropic.com",
                endpoint_type="anthropic",
            )
        ],
        backend="anthropic",
        selected_model="claude-sonnet-4-5-20250929",
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    long_section = "stable context " * 400
    messages = [
        Message(
            role="system",
            content="system",
            metadata={
                "anthropic_prompt_cache_enabled": True,
                "system_prompt_sections": [
                    {"id": "core_identity", "text": long_section, "stable": True},
                    {"id": "tool_runtime_contract", "text": "tool contract", "stable": True},
                ],
            },
        ),
        Message(role="user", content="hello"),
    ]
    tools: list[dict[str, object]] = [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search docs " * 200,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    result = llm.complete(
        model="claude-sonnet-4-5-20250929",
        messages=messages,
        tools=tools,
    )

    assert result.content == "ok"
    call = _FakeChatClient.seen_calls[-1]
    assert "extra_body" not in call
    assert call["messages"][0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert call["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert call["messages"][1]["content"][0]["cache_control"] == {"type": "ephemeral"}

    llm.complete(
        model="claude-sonnet-4-5-20250929",
        messages=messages,
        tools=tools,
        request_metadata={"anthropic_prompt_cache_enabled": False},
    )
    disabled_call = _FakeChatClient.seen_calls[-1]
    assert all("cache_control" not in block for message in disabled_call["messages"] for block in message["content"])
    assert all("cache_control" not in tool for tool in disabled_call["tools"])


def test_apply_claude_prompt_cache_vertex_marks_history_boundary_and_skips_thinking() -> None:
    planned_messages, planned_tools, planned_extra_body = apply_claude_prompt_cache(
        endpoint_type="anthropic_vertex",
        model="claude-sonnet-4-5-20250929",
        messages=[
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "assistant reply"},
                    {"type": "thinking", "thinking": "private chain"},
                ],
            },
            {"role": "user", "content": "latest user turn " * 300},
        ],
        tools=[],
        extra_body=None,
        metadata={
            "anthropic_prompt_cache_enabled": True,
            "system_prompt_sections": [
                {"id": "core_identity", "text": "stable section " * 400, "stable": True},
            ],
        },
    )

    assert planned_extra_body is None
    assert planned_tools == []
    assert planned_messages[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert planned_messages[1]["content"][0].get("cache_control") is None
    assert planned_messages[1]["content"][1].get("cache_control") is None
    assert planned_messages[2]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_apply_claude_prompt_cache_uses_sonnet_4_6_threshold() -> None:
    planned_messages, planned_tools, planned_extra_body = apply_claude_prompt_cache(
        endpoint_type="anthropic",
        model="claude-sonnet-4-6",
        messages=[
            {
                "role": "system",
                    "content": "stable system " * 350,
            },
            {
                "role": "user",
                "content": "latest user turn " * 40,
            },
        ],
        tools=[],
        extra_body=None,
        metadata={"anthropic_prompt_cache_enabled": True},
    )

    assert planned_extra_body is None
    assert planned_tools == []
    assert planned_messages[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert planned_messages[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
