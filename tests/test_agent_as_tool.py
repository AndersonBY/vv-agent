from __future__ import annotations

from pathlib import Path

from vv_agent import Agent, RunConfig, Runner
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse, ToolCall


def _resolved(agent_name: str) -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model=agent_name,
        selected_model=agent_name,
        model_id=agent_name,
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id=agent_name)],
    )


def test_agent_as_tool_returns_child_output_to_parent(tmp_path: Path) -> None:
    researcher = Agent(name="researcher", instructions="Collect facts.", model="researcher")
    writer = Agent(
        name="writer",
        instructions="Write with supplied research.",
        model="writer",
        tools=[researcher.as_tool(name="research", description="Collect facts.")],
    )
    provider_calls: list[str] = []

    def model_provider(agent: Agent, run_config: RunConfig):
        del run_config
        provider_calls.append(agent.name)
        if agent.name == "researcher":
            return (
                ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="research done",
                            tool_calls=[
                                ToolCall(
                                    id="child-finish",
                                    name=TASK_FINISH_TOOL_NAME,
                                    arguments={"message": "facts from child"},
                                )
                            ],
                        )
                    ]
                ),
                _resolved(agent.name),
            )
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="need research",
                        tool_calls=[ToolCall(id="research-call", name="research", arguments={"input": "find facts"})],
                    ),
                    LLMResponse(
                        content="final",
                        tool_calls=[
                            ToolCall(
                                id="parent-finish",
                                name=TASK_FINISH_TOOL_NAME,
                                arguments={"message": "final with facts"},
                            )
                        ],
                    ),
                ]
            ),
            _resolved(agent.name),
        )

    result = Runner.run_sync(writer, "write report", run_config=RunConfig(workspace=tmp_path, model_provider=model_provider))

    assert result.final_output == "final with facts"
    assert provider_calls == ["writer", "researcher"]
    first_cycle = result.raw_result.cycles[0]
    assert first_cycle.tool_results[0].content == "facts from child"
    assert first_cycle.tool_results[0].metadata["agent"] == "researcher"
    assert first_cycle.tool_results[0].metadata["mode"] == "agent_as_tool"
