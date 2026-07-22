from __future__ import annotations

from pathlib import Path

from support import FixedModelProvider

from vv_agent import (
    Agent,
    AgentStartedEvent,
    CycleStartedEvent,
    LLMStartedEvent,
    RunConfig,
    Runner,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse, ToolCall


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="m",
        selected_model="m",
        model_id="m",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="m")],
    )


def test_runner_emits_tool_call_started_and_completed_events(tmp_path: Path) -> None:
    model_provider = FixedModelProvider(
        ScriptedLLM(
            steps=[
                LLMResponse(
                    content="done",
                    tool_calls=[
                        ToolCall(
                            id="finish",
                            name=TASK_FINISH_TOOL_NAME,
                            arguments={"message": "ok"},
                        )
                    ],
                )
            ]
        ),
        _resolved(),
    )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Finish.", model="m"),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    started = [event for event in result.events if isinstance(event, ToolCallStartedEvent)]
    finished = [event for event in result.events if isinstance(event, ToolCallCompletedEvent)]
    assert len(started) == 1
    assert started[0].tool_name == TASK_FINISH_TOOL_NAME
    assert started[0].tool_call_id == "finish"
    assert len(finished) == 1
    assert finished[0].tool_name == TASK_FINISH_TOOL_NAME
    assert finished[0].tool_call_id == "finish"


def test_runner_emits_cycle_and_llm_started_events(tmp_path: Path) -> None:
    model_provider = FixedModelProvider(
        ScriptedLLM(
            steps=[
                LLMResponse(
                    content="done",
                    tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
                )
            ]
        ),
        _resolved(),
    )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Finish.", model="m"),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert [event.type for event in result.events[:4]] == [
        "run_started",
        "agent_started",
        "cycle_started",
        "llm_started",
    ]
    assert isinstance(result.events[1], AgentStartedEvent)
    assert isinstance(result.events[2], CycleStartedEvent)
    assert isinstance(result.events[3], LLMStartedEvent)
    assert result.events[3].to_dict()["model"] == "m"
