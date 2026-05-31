from __future__ import annotations

from pathlib import Path

from vv_agent import Agent, RunConfig, Runner, ToolApprovalRequestedEvent, ToolPolicy, function_tool
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.llm import ScriptedLLM
from vv_agent.types import AgentStatus, LLMResponse, ToolCall


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="m",
        selected_model="m",
        model_id="m",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="m")],
    )


def test_tool_needs_approval_interrupts_before_invocation(tmp_path: Path) -> None:
    invoked = False

    @function_tool(needs_approval=True)
    def delete_file(path: str) -> str:
        nonlocal invoked
        invoked = True
        return f"deleted {path}"

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="delete",
                        tool_calls=[ToolCall(id="delete-call", name="delete_file", arguments={"path": "a.txt"})],
                    )
                ]
            ),
            _resolved(),
        )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Use tools.", model="m", tools=[delete_file]),
        "delete a file",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert invoked is False
    assert result.status == AgentStatus.WAIT_USER
    assert "delete_file" in (result.final_output or "")
    approval_events = [event for event in result.events if isinstance(event, ToolApprovalRequestedEvent)]
    assert len(approval_events) == 1
    assert approval_events[0].tool_name == "delete_file"
    assert approval_events[0].tool_call_id == "delete-call"


def test_tool_policy_never_approval_allows_tool_invocation(tmp_path: Path) -> None:
    invoked = False

    @function_tool(needs_approval=True)
    def safe_echo(text: str) -> str:
        nonlocal invoked
        invoked = True
        return text

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="echo",
                        tool_calls=[ToolCall(id="echo-call", name="safe_echo", arguments={"text": "ok"})],
                    )
                ]
            ),
            _resolved(),
        )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Use tools.", model="m", tools=[safe_echo], tool_use_behavior="stop_on_first_tool"),
        "echo",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            tool_policy=ToolPolicy(approval="never"),
        ),
    )

    assert invoked is True
    assert result.status == AgentStatus.COMPLETED
    assert result.final_output == "ok"
