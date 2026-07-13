from __future__ import annotations

import vv_agent
from vv_agent import Agent, Handoff, ModelSettings, function_tool, handoff


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


def test_handoff_declares_control_transfer_contract() -> None:
    target = Agent(name="writer", instructions="Write the final answer.", model="m")
    transfer = handoff(agent=target, description="Transfer to writer.")

    assert isinstance(transfer, Handoff)
    assert transfer.agent is target
    assert transfer.description == "Transfer to writer."
    assert transfer.tool_name == "transfer_to_writer"


def test_top_level_public_api_does_not_export_legacy_entry_points() -> None:
    legacy_names = {
        "AgentDefinition",
        "AgentSDKClient",
        "AgentSDKOptions",
        "AgentTask",
        "AgentRuntime",
    }

    assert legacy_names.isdisjoint(set(vv_agent.__all__))
    for name in legacy_names:
        assert not hasattr(vv_agent, name)


def test_sdk_package_no_longer_exports_legacy_entry_points() -> None:
    import vv_agent.sdk as sdk

    legacy_names = {
        "AgentDefinition",
        "AgentSDKClient",
        "AgentSDKOptions",
    }

    assert legacy_names.isdisjoint(set(getattr(sdk, "__all__", ())))
    for name in legacy_names:
        assert not hasattr(sdk, name)
