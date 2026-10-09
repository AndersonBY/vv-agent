from __future__ import annotations

from support import FixedModelProvider

from vv_agent import AgentSessionOptions, InteractiveAgentClient, InteractiveAgentDefinition
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.llm import ScriptedLLM
from vv_agent.memory.provider import (
    MemoryCompactCompleted,
    MemoryCompactStarted,
    MemoryProviderResult,
    MemorySaveRequest,
    MemorySaveResult,
    MemorySearchRequest,
    MemorySearchResult,
)
from vv_agent.types import LLMResponse


class _MemoryProvider:
    def search(self, request: MemorySearchRequest) -> list[MemorySearchResult]:
        del request
        return []

    def save(self, request: MemorySaveRequest) -> MemorySaveResult:
        del request
        return MemorySaveResult()

    def before_compact(self, event: MemoryCompactStarted) -> MemoryProviderResult:
        del event
        return MemoryProviderResult()

    def after_compact(self, event: MemoryCompactCompleted) -> None:
        del event


def _resolved(*, backend: str = "test", model: str = "test-model") -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend=backend,
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id=model)],
    )


def test_interactive_session_options_pass_memory_providers_to_run_config(monkeypatch, tmp_path):
    from vv_agent.session.surfaces import SessionDriver

    provider = _MemoryProvider()
    seen_configs = []
    original = SessionDriver.runtime

    def capture(self, agent, config, task=None):
        seen_configs.append(config)
        return original(self, agent, config, task)

    monkeypatch.setattr(SessionDriver, "runtime", capture)
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=FixedModelProvider(ScriptedLLM([LLMResponse("done")]), _resolved()),
            workspace=tmp_path,
            memory_providers=[provider],
        )
    )
    session = client.create_session(
        agent=InteractiveAgentDefinition(description="assistant", model="test-model"), session_id="memory-options"
    )
    try:
        assert session.prompt("hello").final_output == "done"
        assert seen_configs[0].memory_providers == [provider]
    finally:
        client.driver.close()


def test_interactive_agent_definition_passes_memory_providers_to_run_config(monkeypatch, tmp_path):
    from vv_agent.session.surfaces import SessionDriver

    provider = _MemoryProvider()
    seen_configs = []
    original = SessionDriver.runtime

    def capture(self, agent, config, task=None):
        seen_configs.append(config)
        return original(self, agent, config, task)

    monkeypatch.setattr(SessionDriver, "runtime", capture)
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=FixedModelProvider(ScriptedLLM([LLMResponse("done")]), _resolved()), workspace=tmp_path
        )
    )
    session = client.create_session(
        agent=InteractiveAgentDefinition(description="assistant", model="test-model", memory_providers=[provider]),
        session_id="memory-definition",
    )
    try:
        assert session.prompt("hello").final_output == "done"
        assert seen_configs[0].memory_providers == [provider]
    finally:
        client.driver.close()
