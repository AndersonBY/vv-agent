from __future__ import annotations

from pathlib import Path

from vv_agent import Agent, HandoffEvent, RunConfig, Runner, handoff
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.types import AgentStatus, LLMResponse, ToolCall


def _resolved(agent_name: str) -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model=agent_name,
        selected_model=agent_name,
        model_id=agent_name,
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id=agent_name)],
    )


def test_handoff_transfers_control_and_finishes_with_target_output(tmp_path: Path) -> None:
    writer = Agent(name="writer", instructions="Write the answer.", model="writer")
    triage = Agent(
        name="triage",
        instructions="Transfer writing tasks.",
        model="triage",
        handoffs=[handoff(agent=writer, description="Use for writing.")],
    )
    provider_calls: list[str] = []

    def model_provider(agent: Agent, run_config: RunConfig):
        del run_config
        provider_calls.append(agent.name)
        if agent.name == "writer":
            return (
                ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="writer done",
                            tool_calls=[
                                ToolCall(
                                    id="writer-finish",
                                    name=TASK_FINISH_TOOL_NAME,
                                    arguments={"message": "written by target"},
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
                        content="transfer",
                        tool_calls=[
                            ToolCall(
                                id="handoff-call",
                                name="transfer_to_writer",
                                arguments={"input": "write this"},
                            )
                        ],
                    )
                ]
            ),
            _resolved(agent.name),
        )

    result = Runner.run_sync(triage, "please write", run_config=RunConfig(workspace=tmp_path, model_provider=model_provider))

    assert result.status == AgentStatus.COMPLETED
    assert result.final_output == "written by target"
    assert provider_calls == ["triage", "writer"]
    handoff_events = [event for event in result.events if isinstance(event, HandoffEvent)]
    assert len(handoff_events) == 1
    assert handoff_events[0].source_agent == "triage"
    assert handoff_events[0].target_agent == "writer"
