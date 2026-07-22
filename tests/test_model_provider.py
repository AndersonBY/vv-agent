from __future__ import annotations

import json
from pathlib import Path

import pytest

from vv_agent import (
    Agent,
    ModelRef,
    ModelSettings,
    RunConfig,
    Runner,
    ScriptedModelProvider,
)
from vv_agent.config import ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.llm.base import complete_llm_request
from vv_agent.model import ModelError
from vv_agent.types import LLMResponse, Message, ToolCall


def test_scripted_model_provider_runs_through_the_shared_request_contract(tmp_path: Path) -> None:
    requests: list[LlmRequest] = []

    def respond(request: LlmRequest) -> LLMResponse:
        requests.append(request)
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="finish",
                    name=TASK_FINISH_TOOL_NAME,
                    arguments={"message": "done"},
                )
            ],
        )

    provider = ScriptedModelProvider.from_callback("scripted", "demo-model", respond).with_default_settings(
        ModelSettings(temperature=0.1, max_tokens=100)
    )
    result = Runner.run_sync(
        Agent(
            name="assistant",
            instructions="Finish.",
            model=ModelRef.named("demo-model"),
            model_settings=ModelSettings(temperature=0.2),
        ),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=provider,
            model_settings=ModelSettings(max_tokens=300),
        ),
    )

    assert result.final_output == "done"
    assert len(requests) == 1
    assert requests[0].model == "demo-model"
    assert requests[0].model_settings == ModelSettings(temperature=0.2, max_tokens=300)
    assert requests[0].metadata["trace_id"] == result.trace_id
    assert requests[0].metadata["system_prompt_sections"]


def test_scripted_llm_callback_receives_the_complete_request() -> None:
    requests: list[LlmRequest] = []
    llm = ScriptedLLM(steps=[lambda request: requests.append(request) or LLMResponse(content="done")])
    request = LlmRequest(
        model="demo-model",
        messages=[Message(role="user", content="hello")],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        metadata={"trace_id": "trace-1"},
        model_settings=ModelSettings(temperature=0.2),
    )

    response = llm.complete_request(request)

    assert response.content == "done"
    assert requests == [request]
    assert requests[0].tools[0]["function"] == {"name": "lookup"}
    assert requests[0].metadata == {"trace_id": "trace-1"}
    assert requests[0].model_settings == ModelSettings(temperature=0.2)


def test_kwargs_llm_fallback_preserves_request_metadata() -> None:
    seen_metadata: list[dict[str, object]] = []

    class KwargsLLM:
        def complete(
            self,
            *,
            model,
            messages,
            tools,
            stream_callback=None,
            model_settings=None,
            request_metadata=None,
        ) -> LLMResponse:
            del model, messages, tools, stream_callback, model_settings
            seen_metadata.append(dict(request_metadata or {}))
            return LLMResponse(content="done")

    response = complete_llm_request(
        KwargsLLM(),
        LlmRequest(model="demo", messages=[], metadata={"trace_id": "trace-1"}),
    )

    assert response.content == "done"
    assert seen_metadata == [{"trace_id": "trace-1"}]


def test_scripted_provider_default_model_is_runner_fallback(tmp_path: Path) -> None:
    provider = ScriptedModelProvider.new(
        "scripted",
        "provider-default",
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            )
        ],
    )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Finish."),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=provider),
    )

    assert result.resolved is not None
    assert result.resolved.model_id == "provider-default"
    assert result.resolved.function_call_available is True
    assert result.resolved.response_format_available is True


def test_model_ref_and_provider_backend_validation_match_rust_contract() -> None:
    model = ModelRef.backend("other", "demo")
    provider = ScriptedModelProvider.new("scripted", "demo", [])

    assert model.model() == "demo"
    assert model.backend_name() == "other"
    with pytest.raises(ModelError, match="backend mismatch"):
        provider.resolve(model)


def test_model_ref_wire_matches_shared_closed_contract() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "parity" / "model_ref.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    refs = [ModelRef.from_dict(payload) for payload in fixture["valid"]]
    assert [ref.to_dict() for ref in refs] == fixture["valid"]
    for payload in fixture["invalid"]:
        with pytest.raises((TypeError, ValueError)):
            ModelRef.from_dict(payload)


def test_resolved_model_ref_cannot_serialize_credentials() -> None:
    resolved = _resolved_for_wire_test()

    with pytest.raises(ValueError, match="process-local"):
        ModelRef.resolved(resolved).to_dict()


def _resolved_for_wire_test() -> ResolvedModelConfig:
    from vv_agent.config import EndpointConfig, EndpointOption

    endpoint = EndpointConfig(
        endpoint_id="private",
        api_key="must-not-serialize",
        api_base="https://example.invalid/v1",
    )
    return ResolvedModelConfig(
        backend="private",
        requested_model="demo",
        selected_model="demo",
        model_id="demo",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="demo")],
    )
