from __future__ import annotations

from contextlib import suppress

import pytest

from vv_agent import Agent, AgentStatus, ApprovalRequestedEvent, RunConfig, Runner, function_tool
from vv_agent.approval import ApprovalDecision, ApprovalProvider, ApprovalRequest
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime.cancellation import CancelledError
from vv_agent.types import LLMResponse, ToolCall


def _resolved_model(model: str = "test-model") -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id=model)],
    )


class AlwaysAskApprovalProvider(ApprovalProvider):
    def should_request(self, request: ApprovalRequest) -> bool:
        return True

    def decide(self, request: ApprovalRequest) -> ApprovalDecision | None:
        return None


class DenyApprovalProvider(ApprovalProvider):
    def should_request(self, request: ApprovalRequest) -> bool:
        return True

    def decide(self, request: ApprovalRequest) -> ApprovalDecision | None:
        return ApprovalDecision.deny("not safe")


def _finish_response(message: str = "finished") -> LLMResponse:
    return LLMResponse(
        content=message,
        tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": message})],
    )


def test_approval_request_pauses_tool_until_handle_approves() -> None:
    calls: list[str] = []

    @function_tool(needs_approval=True)
    def dangerous() -> str:
        calls.append("ran")
        return "allowed"

    agent = Agent(name="assistant", instructions="Use tool.", model="test-model", tools=[dangerous])
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="dangerous", arguments={})]),
            _finish_response(),
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        return llm, _resolved_model()

    handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=model_provider,
            approval_provider=AlwaysAskApprovalProvider(),
        ),
    )

    request_id = ""
    for event in handle.events():
        if isinstance(event, ApprovalRequestedEvent):
            request_id = event.request_id
            assert calls == []
            handle.approve(request_id, ApprovalDecision.allow())
        if event.type == "run_completed":
            break

    assert request_id
    assert calls == ["ran"]
    assert handle.result().final_output == "finished"


def test_approval_denial_returns_tool_error_without_running_tool() -> None:
    calls: list[str] = []

    @function_tool(needs_approval=True)
    def dangerous() -> str:
        calls.append("ran")
        return "allowed"

    agent = Agent(name="assistant", instructions="Use tool.", model="test-model", tools=[dangerous])
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="dangerous", arguments={})]),
            _finish_response(),
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        return llm, _resolved_model()

    result = Runner.run_sync(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=model_provider,
            approval_provider=DenyApprovalProvider(),
        ),
    )

    tool_result = result.raw_result.cycles[0].tool_results[0]
    assert calls == []
    assert tool_result.error_code == "tool_approval_denied"
    assert result.final_output == "finished"


def test_approval_timeout_returns_tool_error_without_running_tool() -> None:
    calls: list[str] = []

    @function_tool(needs_approval=True)
    def dangerous() -> str:
        calls.append("ran")
        return "allowed"

    agent = Agent(name="assistant", instructions="Use tool.", model="test-model", tools=[dangerous])
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="dangerous", arguments={})]),
            _finish_response(),
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        return llm, _resolved_model()

    result = Runner.run_sync(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=model_provider,
            approval_provider=AlwaysAskApprovalProvider(),
            approval_timeout_seconds=0.01,
        ),
    )

    tool_result = result.raw_result.cycles[0].tool_results[0]
    assert calls == []
    assert tool_result.error_code == "tool_approval_timeout"
    assert result.final_output == "finished"


def test_approval_rejects_unknown_request_id_without_storing_decision() -> None:
    agent = Agent(name="assistant", instructions="Answer.", model="test-model")
    llm = ScriptedLLM(steps=[_finish_response("done")])

    def model_provider(agent: Agent, run_config: RunConfig):
        return llm, _resolved_model()

    handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(model_provider=model_provider),
    )
    assert handle.result(timeout=2).final_output == "done"

    with pytest.raises(KeyError, match="Unknown approval request"):
        handle.approve("approval_missing", ApprovalDecision.allow())


def test_approval_rejects_stale_request_id_after_timeout() -> None:
    @function_tool(needs_approval=True)
    def dangerous() -> str:
        return "allowed"

    agent = Agent(name="assistant", instructions="Use tool.", model="test-model", tools=[dangerous])
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="dangerous", arguments={})]),
            _finish_response(),
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        return llm, _resolved_model()

    handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=model_provider,
            approval_provider=AlwaysAskApprovalProvider(),
            approval_timeout_seconds=0.01,
        ),
    )

    request_id = ""
    for event in handle.events():
        if isinstance(event, ApprovalRequestedEvent):
            request_id = event.request_id
        if event.type == "run_completed":
            break

    assert request_id
    assert handle.result(timeout=2).final_output == "finished"
    with pytest.raises(KeyError, match="Unknown approval request"):
        handle.approve(request_id, ApprovalDecision.allow())


def test_approval_provider_does_not_change_default_no_tool_policy() -> None:
    agent = Agent(name="assistant", instructions="Answer.", model="test-model")
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="first", tool_calls=[]),
            LLMResponse(content="second", tool_calls=[]),
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        return llm, _resolved_model()

    result = Runner.run_sync(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=model_provider,
            approval_provider=AlwaysAskApprovalProvider(),
            max_cycles=2,
        ),
    )

    assert result.status == AgentStatus.MAX_CYCLES
    assert result.final_output == "Reached max cycles without finish signal."


def test_cancel_unblocks_pending_approval_without_running_tool() -> None:
    calls: list[str] = []

    @function_tool(needs_approval=True)
    def dangerous() -> str:
        calls.append("ran")
        return "allowed"

    agent = Agent(name="assistant", instructions="Use tool.", model="test-model", tools=[dangerous])
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="dangerous", arguments={})]),
            _finish_response(),
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        return llm, _resolved_model()

    handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=model_provider,
            approval_provider=AlwaysAskApprovalProvider(),
        ),
    )

    request_id = ""
    for event in handle.events():
        if isinstance(event, ApprovalRequestedEvent):
            request_id = event.request_id
            assert handle.cancel()
            break

    assert request_id
    caught: BaseException | None = None
    try:
        handle.result(timeout=0.2)
    except BaseException as exc:
        caught = exc

    if isinstance(caught, TimeoutError):
        handle.approve(request_id, ApprovalDecision.deny("cleanup"))
        with suppress(CancelledError):
            handle.result(timeout=2)

    assert not isinstance(caught, TimeoutError)
    assert isinstance(caught, CancelledError)
    assert calls == []
