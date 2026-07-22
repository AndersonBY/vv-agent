from __future__ import annotations

import pytest

from vv_agent import Agent, Handoff, ModelSettings, ToolOutputError, function_tool, handoff


def test_agent_exposes_instructions_and_tools() -> None:
    @function_tool
    def echo(text: str) -> str:
        """Echo text."""
        return text

    agent = Agent(
        name="ops",
        instructions="Check facts first.",
        model="kimi-k2.6",
        model_settings=ModelSettings(temperature=0.2),
        tools=[echo],
        metadata={"team": "ops"},
    )

    assert agent.name == "ops"
    assert agent.instructions == "Check facts first."
    assert agent.model == "kimi-k2.6"
    assert agent.tools == [echo]
    assert agent.metadata == {"team": "ops"}


def test_agent_as_tool_wraps_child_agent() -> None:
    child = Agent(name="researcher", instructions="Research facts.", model="m")
    tool = child.as_tool(name="research", description="Ask the researcher.")

    assert tool.name == "research"
    assert tool.description == "Ask the researcher."
    assert tool.params_json_schema["required"] == ["task_description"]
    assert tool.metadata["agent"] is child
    assert tool.metadata["mode"] == "agent_as_tool"


@pytest.mark.parametrize("legacy_field", ["task", "input", "prompt"])
def test_agent_as_tool_rejects_removed_task_description_aliases(legacy_field: str) -> None:
    child = Agent(name="researcher", instructions="Research facts.", model="m")
    tool = child.as_tool(name="research")

    output = tool.invoke(None, {legacy_field: "research the current contract"})

    assert isinstance(output, ToolOutputError)
    assert "requires task_description" in output.message


def test_handoff_declares_control_transfer_contract() -> None:
    target = Agent(name="writer", instructions="Write the final answer.", model="m")
    transfer = handoff(agent=target, description="Transfer to writer.")

    assert isinstance(transfer, Handoff)
    assert transfer.agent is target
    assert transfer.description == "Transfer to writer."
    assert transfer.tool_name == "transfer_to_writer"
