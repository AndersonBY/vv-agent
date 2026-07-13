from __future__ import annotations

from vv_agent import Agent, RunConfig, ToolPolicy, function_tool, handoff
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.runtime.compiler import AgentCompiler, RuntimeTask


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="requested",
        selected_model="selected",
        model_id="model-id",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="model-id")],
    )


def test_agent_compiler_builds_runtime_task_from_public_contract() -> None:
    @function_tool
    def lookup(order_id: str) -> str:
        """Lookup order."""
        return order_id

    writer = Agent(name="writer", instructions="Write.", model="writer")
    agent = Agent(
        name="ops",
        instructions="Check facts.",
        model="agent-model",
        tools=[lookup],
        handoffs=[handoff(agent=writer, description="Transfer to writer.")],
        metadata={"team": "ops"},
    )

    task = AgentCompiler().compile(
        agent=agent,
        input="analyze order",
        run_config=RunConfig(
            model="override-model",
            max_cycles=12,
            tool_policy=ToolPolicy(allowed_tools=[TASK_FINISH_TOOL_NAME, "lookup", "transfer_to_writer"]),
            metadata={"request_id": "r1"},
        ),
        resolved=_resolved(),
        trace_id="trace-1",
    )

    assert isinstance(task, RuntimeTask)
    assert task.task_id.startswith("ops_")
    assert task.model == "model-id"
    assert task.system_prompt == "Check facts."
    assert task.user_prompt == "analyze order"
    assert task.max_cycles == 12
    assert task.extra_tool_names == ["lookup", "transfer_to_writer"]
    assert task.metadata["team"] == "ops"
    assert task.metadata["request_id"] == "r1"
    assert task.metadata["_vv_agent_allowed_tools"] == [TASK_FINISH_TOOL_NAME, "lookup", "transfer_to_writer"]
    assert task.metadata["trace_id"] == "trace-1"
    assert not hasattr(task, "runtime_metadata")
