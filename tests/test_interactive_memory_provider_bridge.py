from __future__ import annotations

from typing import Any

import vv_agent.interactive as interactive_mod
from vv_agent import AgentSessionOptions, InteractiveAgentClient, InteractiveAgentDefinition
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.memory.provider import (
    MemoryCompactCompleted,
    MemoryCompactStarted,
    MemoryProviderResult,
    MemorySaveRequest,
    MemorySaveResult,
    MemorySearchRequest,
    MemorySearchResult,
)
from vv_agent.result import RunResult
from vv_agent.run_config import RunConfig
from vv_agent.types import AgentResult, AgentStatus, Message


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


def _run_result() -> RunResult:
    raw_result = AgentResult(
        status=AgentStatus.COMPLETED,
        messages=[Message(role="assistant", content="done")],
        cycles=[],
        final_answer="done",
    )
    return RunResult(
        input="hello",
        new_items=[],
        final_output="done",
        status=AgentStatus.COMPLETED,
        raw_result=raw_result,
    )


def test_interactive_session_options_pass_memory_providers_to_run_config(monkeypatch, tmp_path) -> None:
    provider = _MemoryProvider()
    seen_configs: list[RunConfig] = []

    class _Handle:
        def events(self) -> list[Any]:
            return []

        def result(self) -> RunResult:
            return _run_result()

    def fake_start(_agent: Any, _input: str, *, task: Any, run_config: RunConfig) -> _Handle:
        del task
        seen_configs.append(run_config)
        return _Handle()

    monkeypatch.setattr(interactive_mod.Runner, "_start_compiled", fake_start)

    def llm_builder(*_: Any, **__: Any):
        return object(), _resolved()

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="test",
            workspace=tmp_path,
            llm_builder=llm_builder,
            memory_providers=[provider],
        )
    )
    session = client.create_session(
        session_id="session_1",
        agent=InteractiveAgentDefinition(description="assistant", model="test-model"),
    )

    session.prompt("hello", auto_follow_up=False)

    assert seen_configs[0].memory_providers == [provider]


def test_interactive_agent_definition_passes_memory_providers_to_run_config(monkeypatch, tmp_path) -> None:
    provider = _MemoryProvider()
    seen_configs: list[RunConfig] = []

    class _Handle:
        def events(self) -> list[Any]:
            return []

        def result(self) -> RunResult:
            return _run_result()

    def fake_start(_agent: Any, _input: str, *, task: Any, run_config: RunConfig) -> _Handle:
        del task
        seen_configs.append(run_config)
        return _Handle()

    monkeypatch.setattr(interactive_mod.Runner, "_start_compiled", fake_start)

    def llm_builder(*_: Any, **__: Any):
        return object(), _resolved()

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
            memory_providers=[provider],
        ),
    )

    session.prompt("hello", auto_follow_up=False)

    assert seen_configs[0].memory_providers == [provider]
