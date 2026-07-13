from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from vv_agent import Agent, ApprovalPolicy, RunConfig, Runner, ToolPolicy, function_tool
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.model_settings import ModelSettings
from vv_agent.runtime import BaseRuntimeHook, BeforeLLMEvent
from vv_agent.types import AgentStatus, LLMResponse, Message, ToolCall


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="m",
        selected_model="m",
        model_id="m",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="m")],
    )


def test_agent_rejects_empty_name_and_static_instructions() -> None:
    with pytest.raises(ValueError, match="agent name cannot be empty"):
        Agent(name=" ", instructions="Valid instructions.")

    with pytest.raises(ValueError, match="agent instructions cannot be empty"):
        Agent(name="assistant", instructions=" ")


def test_dynamic_instructions_receive_agent_and_complete_run_context(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def instructions(context, current_agent: Agent) -> str:
        observed.update(
            {
                "same_agent": current_agent is agent,
                "run_id": context.run_id,
                "agent_name": context.agent_name,
                "model": context.model,
                "workspace": context.workspace,
                "metadata": dict(context.metadata),
                "app_state": context.app_state,
            }
        )
        return f"tenant={context.metadata['tenant']} agent={current_agent.name} run={context.run_id}"

    def finish(request: LlmRequest) -> LLMResponse:
        _model, messages = request.model, request.messages
        assert "tenant=acme agent=assistant run=run_" in messages[0].content
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
        )

    agent = Agent(
        name="assistant",
        instructions=instructions,
        model=ScriptedLLM(steps=[finish]),
        metadata={"tenant": "acme"},
    )
    result = Runner.run_sync(
        agent,
        "go",
        run_config=RunConfig(workspace=tmp_path, context={"request_id": "req-1"}),
    )

    assert result.final_output == "done"
    assert observed["same_agent"] is True
    assert str(observed["run_id"]).startswith("run_")
    assert observed["agent_name"] == "assistant"
    assert observed["model"] == "direct"
    assert observed["workspace"] == tmp_path
    assert cast(dict[str, object], observed["metadata"])["tenant"] == "acme"
    assert observed["app_state"] == {"request_id": "req-1"}


def test_agent_hooks_run_before_per_run_hooks(tmp_path: Path) -> None:
    order: list[str] = []

    class OrderedHook(BaseRuntimeHook):
        def __init__(self, name: str) -> None:
            self.name = name

        def before_llm(self, event: BeforeLLMEvent):
            del event
            order.append(self.name)
            return None

    agent = Agent(
        name="assistant",
        instructions="Finish.",
        model=ScriptedLLM(
            steps=[
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
                )
            ]
        ),
        hooks=[OrderedHook("agent")],
    )

    result = Runner.run_sync(
        agent,
        "go",
        run_config=RunConfig(workspace=tmp_path, hooks=[OrderedHook("run")]),
    )

    assert result.final_output == "done"
    assert order == ["agent", "run"]


def test_agent_max_cycles_applies_when_run_config_only_supplies_provider() -> None:
    calls = 0

    class NoToolLLM:
        def complete(
            self,
            *,
            model: str,
            messages: list[Message],
            tools: list[dict[str, object]],
            stream_callback=None,
            model_settings: ModelSettings | None = None,
            request_metadata: dict[str, object] | None = None,
        ) -> LLMResponse:
            del model, messages, tools, stream_callback, model_settings, request_metadata
            nonlocal calls
            calls += 1
            return LLMResponse(content=f"cycle {calls}")

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return NoToolLLM(), _resolved()

    result = Runner.run_sync(
        Agent(name="bounded", instructions="Try twice.", model="m", max_cycles=2),
        "go",
        run_config=RunConfig(model_provider=model_provider),
    )

    assert result.status == AgentStatus.MAX_CYCLES
    assert calls == 2


def test_run_config_max_cycles_overrides_agent_default() -> None:
    llm = ScriptedLLM(steps=[LLMResponse(content="one"), LLMResponse(content="two")])

    result = Runner.run_sync(
        Agent(name="bounded", instructions="Try more.", model=llm, max_cycles=3),
        "go",
        run_config=RunConfig(max_cycles=1),
    )

    assert result.status == AgentStatus.MAX_CYCLES
    assert len(result.raw_result.cycles) == 1


def test_explicit_framework_default_max_cycles_still_overrides_agent_default() -> None:
    llm = ScriptedLLM(steps=[LLMResponse(content=f"cycle {index}") for index in range(10)])

    result = Runner.run_sync(
        Agent(name="bounded", instructions="Try more.", model=llm, max_cycles=3),
        "go",
        run_config=RunConfig(max_cycles=10),
    )

    assert result.status == AgentStatus.MAX_CYCLES
    assert len(result.raw_result.cycles) == 10


def test_agent_tool_policy_is_default_and_run_config_policy_overrides_it() -> None:
    @function_tool
    def lookup() -> str:
        """Look up data."""
        return "data"

    class CapturingLLM:
        def __init__(self) -> None:
            self.tool_names: list[str] = []

        def complete(
            self,
            *,
            model: str,
            messages: list[Message],
            tools: list[dict[str, object]],
            stream_callback=None,
            model_settings: ModelSettings | None = None,
            request_metadata: dict[str, object] | None = None,
        ) -> LLMResponse:
            del model, messages, stream_callback, model_settings, request_metadata
            self.tool_names = [str(cast(dict[str, object], tool["function"])["name"]) for tool in tools]
            return LLMResponse(
                content="finish",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            )

    agent_llm = CapturingLLM()
    agent = Agent(
        name="policy",
        instructions="Use policy.",
        model=agent_llm,
        tools=[lookup],
        tool_policy=ToolPolicy(allowed_tools=[TASK_FINISH_TOOL_NAME]),
    )
    Runner.run_sync(agent, "agent policy")
    assert agent_llm.tool_names == [TASK_FINISH_TOOL_NAME]

    run_llm = CapturingLLM()
    agent.model = run_llm
    Runner.run_sync(
        agent,
        "run policy",
        run_config=RunConfig(tool_policy=ToolPolicy(allowed_tools=[TASK_FINISH_TOOL_NAME, "lookup"])),
    )
    assert run_llm.tool_names == [TASK_FINISH_TOOL_NAME, "lookup"]


def test_agent_and_run_tool_policies_merge_each_policy_dimension() -> None:
    agent_policy = ToolPolicy(
        allowed_tools=["agent-only"],
        disallowed_tools=["agent-blocked", "shared-blocked"],
        approval="always",
        can_use_tool=lambda name, arguments: name != "agent-denied" and bool(arguments.get("agent_ok")),
    )
    run_policy = ToolPolicy(
        allowed_tools=["run-only"],
        disallowed_tools=["shared-blocked", "run-blocked"],
        can_use_tool=lambda name, arguments: name != "run-denied" and bool(arguments.get("run_ok")),
    )
    agent = Agent(name="policy", instructions="Use policy.", tool_policy=agent_policy)

    effective = Runner._effective_run_config(agent, RunConfig(tool_policy=run_policy))
    merged = effective.tool_policy

    assert merged is not None
    assert merged.allowed_tools == ["run-only"]
    assert merged.disallowed_tools == ["agent-blocked", "shared-blocked", "run-blocked"]
    assert merged.approval == "always"
    assert merged.can_use_tool is not None
    assert merged.can_use_tool("allowed", {"agent_ok": True, "run_ok": True}) is True
    assert merged.can_use_tool("agent-denied", {"agent_ok": True, "run_ok": True}) is False
    assert merged.can_use_tool("run-denied", {"agent_ok": True, "run_ok": True}) is False


def test_run_explicit_approval_policy_overrides_agent_explicit_policy() -> None:
    agent = Agent(
        name="policy",
        instructions="Use policy.",
        tool_policy=ToolPolicy(approval="always"),
    )

    effective = Runner._effective_run_config(
        agent,
        RunConfig(tool_policy=ToolPolicy(approval="never")),
    )

    assert effective.tool_policy is not None
    assert effective.tool_policy.approval == "never"


@pytest.mark.parametrize(
    ("runner_approval", "agent_approval", "run_approval", "expected"),
    [
        ("always", "on_request", "default", "on_request"),
        ("never", "always", "on_request", "on_request"),
    ],
)
def test_on_request_is_an_explicit_approval_policy_override(
    runner_approval: ApprovalPolicy,
    agent_approval: ApprovalPolicy,
    run_approval: ApprovalPolicy,
    expected: ApprovalPolicy,
) -> None:
    effective = Runner._effective_run_config(
        Agent(
            name="policy",
            instructions="Use policy.",
            tool_policy=ToolPolicy(approval=agent_approval),
        ),
        RunConfig(tool_policy=ToolPolicy(approval=run_approval)),
        runner_defaults=RunConfig(
            tool_policy=ToolPolicy(approval=runner_approval),
        ),
    )

    assert effective.tool_policy is not None
    assert effective.tool_policy.approval == expected


def test_approval_policy_is_public_and_rejects_unknown_runtime_values() -> None:
    policy: ApprovalPolicy = "on_request"
    assert ToolPolicy(approval=policy).approval == "on_request"

    with pytest.raises(
        ValueError,
        match="approval must be one of: always, default, never, on_request",
    ):
        ToolPolicy(approval=cast(ApprovalPolicy, "sometimes"))
