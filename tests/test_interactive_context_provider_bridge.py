from __future__ import annotations

from typing import Any

from vv_agent import AgentSessionOptions, InteractiveAgentClient, InteractiveAgentDefinition
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.context_providers import ContextFragment, ContextRequest
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse, Message


def _resolved(*, backend: str = "test", model: str = "test-model") -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend=backend,
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id=model)],
    )


class _StaticProvider:
    def fragments(self, request: ContextRequest) -> list[ContextFragment]:
        assert request.agent_name == "inline"
        return [
            ContextFragment(
                id="runtime_context",
                text="Runtime context from provider.",
                stable=True,
                source="test",
            )
        ]


def test_interactive_session_options_pass_context_providers_to_run_config(tmp_path) -> None:
    seen_prompts: list[str] = []

    def respond(_model: str, messages: list[Message]) -> LLMResponse:
        seen_prompts.extend(message.content for message in messages if message.role == "system")
        return LLMResponse(content="answer", tool_calls=[])

    def llm_builder(*_: Any, **__: Any):
        return ScriptedLLM(steps=[respond]), _resolved()

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="test",
            workspace=tmp_path,
            llm_builder=llm_builder,
            context_providers=[_StaticProvider()],
        )
    )
    session = client.create_session(
        session_id="session_1",
        agent=InteractiveAgentDefinition(description="assistant", model="test-model", system_prompt="assistant"),
    )

    session.prompt("hello", auto_follow_up=False)

    assert seen_prompts == ["assistant\n\nRuntime context from provider."]


def test_interactive_agent_definition_passes_session_context_providers_to_run_config(tmp_path) -> None:
    seen_prompts: list[str] = []

    def respond(_model: str, messages: list[Message]) -> LLMResponse:
        seen_prompts.extend(message.content for message in messages if message.role == "system")
        return LLMResponse(content="answer", tool_calls=[])

    def llm_builder(*_: Any, **__: Any):
        return ScriptedLLM(steps=[respond]), _resolved()

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="test",
            workspace=tmp_path,
            llm_builder=llm_builder,
        )
    )
    session = client.create_session(
        session_id="session_1",
        agent=InteractiveAgentDefinition(
            description="assistant",
            model="test-model",
            system_prompt="assistant",
            context_providers=[_StaticProvider()],
        ),
    )

    session.prompt("hello", auto_follow_up=False)

    assert seen_prompts == ["assistant\n\nRuntime context from provider."]
