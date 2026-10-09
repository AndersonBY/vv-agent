from __future__ import annotations

from threading import Event

import pytest
from support import FixedModelProvider

from vv_agent import (
    Agent,
    AgentStatus,
    ApprovalRequestedEvent,
    RunConfig,
    Runner,
    ToolPolicy,
    build_default_registry,
    function_tool,
)
from vv_agent.approval import ApprovalBroker, ApprovalDecision, ApprovalProvider, ApprovalRequest
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.session.store import Conflict
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


class FailingDecisionApprovalProvider(ApprovalProvider):
    def __init__(self) -> None:
        self.request_id = ""

    def should_request(self, request: ApprovalRequest) -> bool:
        self.request_id = request.request_id
        return True

    def decide(self, request: ApprovalRequest) -> ApprovalDecision | None:
        raise RuntimeError("approval provider unavailable")


class BlockingShouldRequestApprovalProvider(ApprovalProvider):
    def __init__(self, *, should_request_result: bool = True) -> None:
        self.entered = Event()
        self.proceed = Event()
        self.request_id = ""
        self.should_request_result = should_request_result

    def should_request(self, request: ApprovalRequest) -> bool:
        self.request_id = request.request_id
        self.entered.set()
        self.proceed.wait(timeout=2)
        return self.should_request_result

    def decide(self, request: ApprovalRequest) -> ApprovalDecision | None:
        return None


def _finish_response(message: str = "finished") -> LLMResponse:
    return LLMResponse(content=message)


def test_approval_provider_failure_fails_run_without_faking_a_denial() -> None:
    calls: list[str] = []

    @function_tool(needs_approval=True)
    def dangerous() -> str:
        calls.append("ran")
        return "allowed"

    provider = FailingDecisionApprovalProvider()
    broker = ApprovalBroker()
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="dangerous", arguments={})]),
        ]
    )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Use tool.", model="test-model", tools=[dangerous]),
        "go",
        run_config=RunConfig(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            approval_provider=provider,
            approval_broker=broker,
            max_cycles=1,
        ),
    )

    assert result.status == AgentStatus.FAILED
    assert result.raw_result.error == {
        "code": "agent_failed",
        "message": "approval provider unavailable",
        "retryable": False,
    }
    assert calls == []
    assert provider.request_id
    assert broker.pending_request(provider.request_id) is None
    assert [event.type for event in result.events if event.type in {"approval_requested", "approval_resolved", "run_failed"}] == [
        "approval_requested",
        "run_failed",
    ]


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

    handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            approval_provider=AlwaysAskApprovalProvider(),
        ),
    )

    request_id = ""
    for event in handle.result().events:
        if isinstance(event, ApprovalRequestedEvent):
            request_id = event.request_id
            assert calls == []
            handle.approve(request_id, ApprovalDecision.allow())
        if event.type == "run_completed":
            break

    assert request_id
    assert handle.result().status is AgentStatus.WAIT_USER
    assert handle.resume().final_output == "finished"
    assert calls == ["ran"]


def test_function_tool_approval_policy_always_requests_once() -> None:
    calls: list[str] = []

    @function_tool
    def guarded_function() -> str:
        calls.append("ran")
        return "allowed"

    agent = Agent(name="assistant", instructions="Use tool.", model="test-model", tools=[guarded_function])
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="guarded_function", arguments={})]),
            _finish_response(),
        ]
    )

    handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            approval_provider=AlwaysAskApprovalProvider(),
            tool_policy=ToolPolicy(approval="always"),
        ),
    )

    request_ids: list[str] = []
    for event in handle.result().events:
        if isinstance(event, ApprovalRequestedEvent):
            if event.tool_name == "guarded_function":
                request_ids.append(event.request_id)
                assert calls == []
            handle.approve(event.request_id, ApprovalDecision.allow())
        if event.type == "run_completed":
            break

    assert request_ids
    assert len(request_ids) == 1
    assert handle.result().status is AgentStatus.WAIT_USER
    assert handle.resume().final_output == "finished"
    assert calls == ["ran"]


def test_executor_registered_tool_approval_can_be_approved_from_run_handle() -> None:
    calls: list[str] = []

    @function_tool(needs_approval=True)
    def dangerous_executor() -> str:
        calls.append("ran")
        return "allowed"

    def registry_factory():
        registry = build_default_registry()
        registry.register_executor(dangerous_executor.to_executor())
        return registry

    agent = Agent(name="assistant", instructions="Use tool.", model="test-model")
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="dangerous_executor", arguments={})]),
            _finish_response(),
        ]
    )

    handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            approval_provider=AlwaysAskApprovalProvider(),
            tool_registry_factory=registry_factory,
        ),
    )

    request_id = ""
    for event in handle.result().events:
        if isinstance(event, ApprovalRequestedEvent):
            if event.tool_name == "dangerous_executor":
                request_id = event.request_id
                assert calls == []
            handle.approve(event.request_id, ApprovalDecision.allow())
        if event.type == "run_completed":
            break

    assert request_id
    assert handle.result().status is AgentStatus.WAIT_USER
    assert handle.resume().final_output == "finished"
    assert calls == ["ran"]


def test_executor_approval_policy_never_skips_executor_approval() -> None:
    calls: list[str] = []

    @function_tool(needs_approval=True)
    def safe_executor() -> str:
        calls.append("ran")
        return "allowed"

    def registry_factory():
        registry = build_default_registry()
        registry.register_executor(safe_executor.to_executor())
        return registry

    agent = Agent(name="assistant", instructions="Use tool.", model="test-model")
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="safe_executor", arguments={})]),
            _finish_response(),
        ]
    )

    result = Runner.run_sync(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            tool_registry_factory=registry_factory,
            tool_policy=ToolPolicy(approval="never"),
        ),
    )

    assert calls == ["ran"]
    assert result.final_output == "finished"


def test_executor_approval_policy_always_requests_executor_approval() -> None:
    calls: list[str] = []

    @function_tool
    def policy_executor() -> str:
        calls.append("ran")
        return "allowed"

    def registry_factory():
        registry = build_default_registry()
        registry.register_executor(policy_executor.to_executor())
        return registry

    agent = Agent(name="assistant", instructions="Use tool.", model="test-model")
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="policy_executor", arguments={})]),
            _finish_response(),
        ]
    )

    handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            approval_provider=AlwaysAskApprovalProvider(),
            tool_registry_factory=registry_factory,
            tool_policy=ToolPolicy(approval="always"),
        ),
    )

    request_id = ""
    for event in handle.result().events:
        if isinstance(event, ApprovalRequestedEvent):
            if event.tool_name == "policy_executor":
                request_id = event.request_id
                assert calls == []
            handle.approve(event.request_id, ApprovalDecision.allow())
        if event.type == "run_completed":
            break

    assert request_id
    assert handle.result().status is AgentStatus.WAIT_USER
    assert handle.resume().final_output == "finished"
    assert calls == ["ran"]


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

    result = Runner.run_sync(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            approval_provider=DenyApprovalProvider(),
        ),
    )

    tool_result = result.raw_result.cycles[0].tool_results[0]
    assert calls == []
    assert tool_result.error_code == "tool_approval_denied"
    assert result.final_output == "finished"


def test_approval_timeout_returns_tool_error_without_running_tool():
    calls = []

    @function_tool(needs_approval=True)
    def dangerous():
        calls.append("ran")
        return "allowed"

    handle = Runner.start(
        Agent("assistant", "Use tool.", model="test-model", tools=[dangerous]),
        "go",
        run_config=RunConfig(
            model_provider=FixedModelProvider(
                ScriptedLLM([LLMResponse("calling", [ToolCall("call_1", "dangerous", {})]), _finish_response()]),
                _resolved_model(),
            ),
            approval_provider=AlwaysAskApprovalProvider(),
        ),
    )
    assert handle.result().status is AgentStatus.WAIT_USER
    request = next(e for e in handle.result().events if isinstance(e, ApprovalRequestedEvent))
    handle.approve(request.request_id, ApprovalDecision.timeout("deadline reached"))
    result = handle.resume()
    assert calls == []
    assert result.raw_result.cycles[0].tool_results[0].error_code == "tool_approval_timeout"
    assert result.final_output == "finished"
    with pytest.raises(Conflict, match="different bytes"):
        handle.approve(request.request_id, ApprovalDecision.allow())


def test_approval_rejects_unknown_request_id_without_storing_decision() -> None:
    agent = Agent(name="assistant", instructions="Answer.", model="test-model")
    llm = ScriptedLLM(steps=[_finish_response("done")])

    handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(model_provider=FixedModelProvider(llm, _resolved_model())),
    )
    assert handle.result(timeout=2).final_output == "done"

    with pytest.raises(KeyError, match="Unknown approval request"):
        handle.approve("approval_missing", ApprovalDecision.allow())


def test_approval_provider_does_not_change_default_no_tool_policy() -> None:
    agent = Agent(name="assistant", instructions="Answer.", model="test-model")
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="first", tool_calls=[]),
            LLMResponse(content="second", tool_calls=[]),
        ]
    )

    result = Runner.run_sync(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            approval_provider=AlwaysAskApprovalProvider(),
            max_cycles=2,
        ),
    )

    assert result.status == AgentStatus.COMPLETED
    assert result.final_output == "first"
    assert len(result.raw_result.cycles) == 1


@pytest.mark.parametrize("via_token", [False, True])
def test_cancel_pending_approval_without_running_tool(via_token):
    calls = []
    token = CancellationToken()

    @function_tool(needs_approval=True)
    def dangerous():
        calls.append("ran")
        return "allowed"

    handle = Runner.start(
        Agent("assistant", "Use tool.", model="test-model", tools=[dangerous]),
        "go",
        run_config=RunConfig(
            model_provider=FixedModelProvider(
                ScriptedLLM([LLMResponse("calling", [ToolCall("call_1", "dangerous", {})])]), _resolved_model()
            ),
            approval_provider=AlwaysAskApprovalProvider(),
            cancellation_token=token,
        ),
    )
    assert handle.result().status is AgentStatus.WAIT_USER
    if via_token:
        token.cancel()
    else:
        assert handle.cancel()
    result = handle.resume()
    assert result.status is AgentStatus.FAILED
    assert result.completion_reason is not None and result.completion_reason.value == "cancelled"
    assert calls == []
    assert sum(e.type == "run_cancelled" for e in result.events) == 1


def test_cancel_after_approval_is_queued_prevents_tool_side_effect():
    calls = []

    @function_tool(needs_approval=True)
    def dangerous():
        calls.append("ran")
        return "allowed"

    handle = Runner.start(
        Agent("assistant", "Use tool.", model="test-model", tools=[dangerous]),
        "go",
        run_config=RunConfig(
            model_provider=FixedModelProvider(
                ScriptedLLM([LLMResponse("calling", [ToolCall("call_1", "dangerous", {})])]), _resolved_model()
            ),
            approval_provider=AlwaysAskApprovalProvider(),
        ),
    )
    waiting = handle.result()
    request = next(e for e in waiting.events if isinstance(e, ApprovalRequestedEvent))
    handle.approve(request.request_id, ApprovalDecision.allow())
    assert handle.cancel()
    result = handle.resume()
    assert result.completion_reason is not None and result.completion_reason.value == "cancelled"
    assert calls == []


@pytest.mark.parametrize("should_request", [False, True])
def test_cancel_during_should_request_prevents_tool_side_effect(should_request):
    calls = []
    provider = BlockingShouldRequestApprovalProvider(should_request_result=should_request)

    @function_tool(needs_approval=True)
    def dangerous():
        calls.append("ran")
        return "allowed"

    handle = Runner.start(
        Agent("assistant", "Use tool.", model="test-model", tools=[dangerous]),
        "go",
        run_config=RunConfig(
            model_provider=FixedModelProvider(
                ScriptedLLM([LLMResponse("calling", [ToolCall("call_1", "dangerous", {})])]), _resolved_model()
            ),
            approval_provider=provider,
        ),
    )
    assert provider.entered.wait(2)
    assert handle.cancel()
    provider.proceed.set()
    handle.result(2)
    result = handle.resume()
    assert result.raw_result.error is not None and result.raw_result.error["code"] == "cancel_requested"
    assert calls == []
