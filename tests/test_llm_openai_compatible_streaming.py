from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.chat import ChatCompletionChunk
from vv_llm.chat_clients.openai_client import OpenAIChatClient
from vv_llm.settings import Settings

from vv_agent.llm.vv_llm_client import VvLlmClient
from vv_agent.model_settings import ModelSettings


@pytest.mark.parametrize("interleaved", [True, False], ids=["interleaved-real-index", "sequential-all-index-zero"])
def test_openai_compatible_stream_preserves_arguments_and_progress(interleaved: bool) -> None:
    model, endpoint = "gpt-4o", "offline-openai"
    settings = Settings.load_from_dict(
        {
            "rate_limit": {"enabled": False, "backend": "memory", "default_rpm": 60, "default_tpm": 1000000},
            "endpoints": [{"id": endpoint, "api_base": "https://provider.invalid/v1", "api_key": "test-key"}],
            "backends": {
                "openai": {
                    "models": {
                        model: {
                            "id": model,
                            "endpoints": [endpoint],
                            "context_length": 8192,
                            "max_output_tokens": 1024,
                            "function_call_available": True,
                        }
                    }
                }
            },
        }
    )
    deltas = [
        {"index": 0, "id": "ca", "type": "function", "function": {"name": "a", "arguments": '{"x":'}},
        {"index": 1 if interleaved else 0, "id": "cb", "type": "function", "function": {"name": "b", "arguments": '{"y":'}},
        {"index": 0, "function": {"arguments": "1}"}},
        {"index": 1 if interleaved else 0, "function": {"arguments": "2}"}},
    ]
    order = [0, 1, 2, 3] if interleaved else [0, 2, 1, 3]
    chunks = [
        ChatCompletionChunk.model_validate(
            {
                "id": f"chunk-{position}",
                "created": 0,
                "model": model,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"tool_calls": [deltas[position]]}, "finish_reason": None}],
            }
        )
        for position in order
    ]
    chunks.append(
        ChatCompletionChunk.model_validate(
            {
                "id": "usage",
                "created": 0,
                "model": model,
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
    )
    requests: list[dict[str, Any]] = []

    def create(**kwargs: Any):
        requests.append(kwargs)
        assert kwargs["stream"] is True
        assert kwargs["model"] == model
        return iter(chunks)

    chat_client = OpenAIChatClient(model=model, stream=True, settings=settings)
    chat_client.endpoint, chat_client.model_id = settings.get_endpoint(endpoint), model
    chat_client.__dict__["raw_client"] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = VvLlmClient(endpoint_targets=[], backend="openai", selected_model=model)
    options = client._resolve_request_options(
        model, stream=True, endpoint_type="openai", model_settings=ModelSettings(max_tokens=16)
    )
    events: list[dict[str, Any]] = []
    response = client._stream_completion(
        chat_client=chat_client,
        options=options,
        messages=[{"role": "user", "content": "Call both tools."}],
        model_name=model,
        tool_payload=[],
        stream_callback=events.append,
    )

    assert len(requests) == 1
    assert [(call.id, call.name, call.arguments) for call in response.tool_calls] == [
        ("ca", "a", {"x": 1}),
        ("cb", "b", {"y": 2}),
    ]
    assert response.raw["usage"] == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    by_delta = [
        [("tool_call_started", "ca", 0, "a", 5), ("tool_call_progress", "ca", 0, "a", 5)],
        [
            ("tool_call_started", "cb", 1 if interleaved else 0, "b", 5),
            ("tool_call_progress", "cb", 1 if interleaved else 0, "b", 5),
        ],
        [("tool_call_progress", "ca", 0, "a", 7)],
        [("tool_call_progress", "cb", 1 if interleaved else 0, "b", 7)],
    ]
    assert [
        (event["event"], event["tool_call_id"], event["tool_call_index"], event["function_name"], event["arguments_chars"])
        for event in events
    ] == [event for position in order for event in by_delta[position]]
