from __future__ import annotations

from pathlib import Path

import pytest

from v_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from v_agent.constants import TASK_FINISH_TOOL_NAME
from v_agent.llm import ScriptedLLM
from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from v_agent.tools import build_default_registry
from v_agent.types import AgentStatus, LLMResponse, SubAgentConfig, ToolCall


def _fake_resolved(*, backend: str, model: str) -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    option = EndpointOption(endpoint=endpoint, model_id=model)
    return ResolvedModelConfig(
        backend=backend,
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[option],
    )


def test_sdk_client_runs_named_agent(tmp_path: Path) -> None:
    builder_calls: list[tuple[str, str, str]] = []

    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del timeout_seconds
        builder_calls.append((str(settings_path), backend, model))
        llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="done",
                    tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
                )
            ]
        )
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
        agents={
            "demo": AgentDefinition(
                description="you are helper",
                model="kimi-k2.5",
            )
        },
    )

    run = client.run_agent(agent_name="demo", prompt="say ok")
    assert run.result.status == AgentStatus.COMPLETED
    assert run.result.final_answer == "ok"
    assert run.resolved.backend == "moonshot"
    assert builder_calls == [("local_settings.py", "moonshot", "kimi-k2.5")]


def test_sdk_prepare_task_supports_sub_agent_configs(tmp_path: Path) -> None:
    client = AgentSDKClient(
        options=AgentSDKOptions(settings_file=Path("local_settings.py"), default_backend="moonshot", workspace=tmp_path),
        agents={
            "orchestrator": AgentDefinition(
                description="orchestrate",
                model="kimi-k2.5",
                sub_agents={
                    "research-sub": SubAgentConfig(
                        model="kimi-k2.5",
                        description="collect data",
                    )
                },
            )
        },
    )

    task = client.prepare_task(
        agent_name="orchestrator",
        prompt="plan work",
        resolved_model_id="kimi-k2.5",
    )
    assert task.sub_agents_enabled is True
    assert task.sub_agents["research-sub"].description == "collect data"
    assert task.metadata["sub_agent_names"] == ["research-sub"]


def test_sdk_client_unknown_agent_raises_clear_error(tmp_path: Path) -> None:
    client = AgentSDKClient(
        options=AgentSDKOptions(settings_file=Path("local_settings.py"), default_backend="moonshot", workspace=tmp_path),
        agents={"demo": AgentDefinition(description="d", model="m")},
    )
    with pytest.raises(ValueError, match="Unknown agent: missing"):
        client.run_agent(agent_name="missing", prompt="hi")
