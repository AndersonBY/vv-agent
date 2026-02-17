from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import ClassVar, cast

from v_agent.constants import TODO_WRITE_TOOL_NAME
from v_agent.llm.openai_compatible import EndpointTarget, OpenAICompatibleLLM
from v_agent.types import Message


class _FakeUsage:
    def model_dump(self, *, exclude_none: bool = True) -> dict[str, int]:
        del exclude_none
        return {"prompt_tokens": 3, "completion_tokens": 2}


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))]

    def model_dump(self, *, exclude_none: bool = True) -> dict[str, object]:
        del exclude_none
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}


class _FakeOpenAI:
    behavior_by_base_url: ClassVar[dict[str, Callable[[dict[str, object]], object] | object]] = {}

    def __init__(self, *, api_key: str, base_url: str, timeout: float):
        del api_key, timeout
        behavior = self.behavior_by_base_url[base_url]

        def create(**kwargs):
            if callable(behavior):
                callable_behavior = cast(Callable[[dict[str, object]], object], behavior)
                return callable_behavior(kwargs)
            return behavior

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def test_llm_failover_to_next_endpoint(monkeypatch) -> None:
    def failing_call(kwargs):
        del kwargs
        raise RuntimeError("first endpoint down")

    _FakeOpenAI.behavior_by_base_url = {
        "https://first.example/v1": failing_call,
        "https://second.example/v1": _FakeResponse("ok from backup"),
    }

    monkeypatch.setattr("v_agent.llm.openai_compatible.OpenAI", _FakeOpenAI)

    llm = OpenAICompatibleLLM(
        endpoint_targets=[
            EndpointTarget(endpoint_id="first", api_key="k1", api_base="https://first.example/v1"),
            EndpointTarget(endpoint_id="second", api_key="k2", api_base="https://second.example/v1"),
        ],
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    response = llm.complete(model="test-model", messages=[Message(role="user", content="hello")], tools=[])

    assert response.content == "ok from backup"
    assert response.raw["used_endpoint_id"] == "second"


def test_llm_stream_aggregates_tool_calls(monkeypatch) -> None:
    chunk_1 = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content="hello ",
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
            )
        ],
    )
    chunk_2 = SimpleNamespace(
        usage=_FakeUsage(),
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content="world",
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id=None,
                            function=SimpleNamespace(name=None, arguments='"priority":"medium"}]}'),
                        )
                    ],
                )
            )
        ],
    )

    def stream_call(kwargs):
        assert kwargs["stream"] is True
        return [chunk_1, chunk_2]

    _FakeOpenAI.behavior_by_base_url = {
        "https://stream.example/v1": stream_call,
    }

    monkeypatch.setattr("v_agent.llm.openai_compatible.OpenAI", _FakeOpenAI)

    llm = OpenAICompatibleLLM(
        endpoint_targets=[EndpointTarget(endpoint_id="stream", api_key="k", api_base="https://stream.example/v1")],
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    response = llm.complete(model="kimi-k2-thinking", messages=[Message(role="user", content="hi")], tools=[])

    assert response.content == "hello world"
    assert response.raw["stream_collected"] is True
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == TODO_WRITE_TOOL_NAME
    assert response.tool_calls[0].arguments["todos"][0]["title"] == "a"


def test_llm_stream_collects_reasoning_content(monkeypatch) -> None:
    chunk_1 = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content="",
                    reasoning_content="step-1",
                    tool_calls=[],
                )
            )
        ],
    )
    chunk_2 = SimpleNamespace(
        usage=_FakeUsage(),
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content="final",
                    reasoning_content="|step-2",
                    tool_calls=[],
                )
            )
        ],
    )

    def stream_call(kwargs):
        assert kwargs["stream"] is True
        return [chunk_1, chunk_2]

    _FakeOpenAI.behavior_by_base_url = {
        "https://stream-reasoning.example/v1": stream_call,
    }

    monkeypatch.setattr("v_agent.llm.openai_compatible.OpenAI", _FakeOpenAI)

    llm = OpenAICompatibleLLM(
        endpoint_targets=[
            EndpointTarget(
                endpoint_id="stream-reasoning",
                api_key="k",
                api_base="https://stream-reasoning.example/v1",
            )
        ],
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    response = llm.complete(model="kimi-k2-thinking", messages=[Message(role="user", content="hi")], tools=[])
    assert response.content == "final"
    assert response.raw["reasoning_content"] == "step-1|step-2"


def test_resolve_request_options_aligns_claude_thinking_profile() -> None:
    llm = OpenAICompatibleLLM(endpoint_targets=[])
    options = llm._resolve_request_options("claude-opus-4-6-thinking", stream=True, endpoint_type="openai")
    assert options.model == "claude-opus-4-6"
    assert options.temperature == 1.0
    assert options.max_tokens == 20000
    assert options.thinking == {"type": "enabled", "budget_tokens": 16000}


def test_resolve_request_options_aligns_gemini3_profile() -> None:
    llm = OpenAICompatibleLLM(endpoint_targets=[])
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
        choices=[SimpleNamespace(delta=SimpleNamespace(content="done", tool_calls=[]))],
    )

    def stream_call(kwargs):
        assert kwargs["stream"] is True
        assert kwargs["model"] == "qwen3-32b"
        assert kwargs["extra_body"] == {"enable_thinking": True}
        return [chunk]

    _FakeOpenAI.behavior_by_base_url = {
        "https://qwen.example/v1": stream_call,
    }
    monkeypatch.setattr("v_agent.llm.openai_compatible.OpenAI", _FakeOpenAI)

    llm = OpenAICompatibleLLM(
        endpoint_targets=[EndpointTarget(endpoint_id="qwen", api_key="k", api_base="https://qwen.example/v1")],
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    response = llm.complete(model="qwen3-32b-thinking", messages=[Message(role="user", content="hi")], tools=[])
    assert response.content == "done"


def test_llm_stream_aggregates_tool_calls_without_index(monkeypatch) -> None:
    chunk_1 = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content="",
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
            )
        ],
    )
    chunk_2 = SimpleNamespace(
        usage=_FakeUsage(),
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            index=None,
                            id=None,
                            function=SimpleNamespace(name=None, arguments='"status":"pending"}]}'),
                        )
                    ],
                )
            )
        ],
    )

    def stream_call(kwargs):
        assert kwargs["stream"] is True
        return [chunk_1, chunk_2]

    _FakeOpenAI.behavior_by_base_url = {
        "https://missing-index.example/v1": stream_call,
    }
    monkeypatch.setattr("v_agent.llm.openai_compatible.OpenAI", _FakeOpenAI)

    llm = OpenAICompatibleLLM(
        endpoint_targets=[
            EndpointTarget(
                endpoint_id="missing-index",
                api_key="k",
                api_base="https://missing-index.example/v1",
            )
        ],
        randomize_endpoints=False,
        max_retries_per_endpoint=1,
        backoff_seconds=0.0,
    )

    response = llm.complete(model="kimi-k2-thinking", messages=[Message(role="user", content="hi")], tools=[])
    assert response.tool_calls[0].arguments["todos"][0]["title"] == "x"
