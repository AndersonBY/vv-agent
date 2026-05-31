from __future__ import annotations

from contextlib import suppress
from threading import Event

import pytest

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
from vv_agent.approval import ApprovalDecision, ApprovalProvider, ApprovalRequest
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime.cancellation import CancellationToken, CancelledError
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

    def model_provider(agent: Agent, run_config: RunConfig):
        return llm, _resolved_model()

    handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=model_provider,
            approval_provider=AlwaysAskApprovalProvider(),
            tool_registry_factory=registry_factory,
        ),
    )

    request_id = ""
    for event in handle.events():
        if isinstance(event, ApprovalRequestedEvent):
            if event.tool_name == "dangerous_executor":
                request_id = event.request_id
                assert calls == []
            handle.approve(event.request_id, ApprovalDecision.allow())
        if event.type == "run_completed":
            break

    assert request_id
    assert calls == ["ran"]
    assert handle.result().final_output == "finished"


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

    def model_provider(agent: Agent, run_config: RunConfig):
        return llm, _resolved_model()

    result = Runner.run_sync(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=model_provider,
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

    def model_provider(agent: Agent, run_config: RunConfig):
        return llm, _resolved_model()

    handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(
            model_provider=model_provider,
            approval_provider=AlwaysAskApprovalProvider(),
            tool_registry_factory=registry_factory,
            tool_policy=ToolPolicy(approval="always"),
        ),
    )

    request_id = ""
    for event in handle.events():
        if isinstance(event, ApprovalRequestedEvent):
            if event.tool_name == "policy_executor":
                request_id = event.request_id
                assert calls == []
            handle.approve(event.request_id, ApprovalDecision.allow())
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


def test_direct_cancellation_token_unblocks_pending_approval_without_running_tool() -> None:
    calls: list[str] = []
    token = CancellationToken()

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
            cancellation_token=token,
        ),
    )

    request_id = ""
    for event in handle.events():
        if isinstance(event, ApprovalRequestedEvent):
            request_id = event.request_id
            token.cancel()
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


def test_cancel_after_approval_resolved_prevents_tool_side_effect() -> None:
    calls: list[str] = []
    token = CancellationToken()

    @function_tool(needs_approval=True)
    def dangerous() -> str:
        calls.append("ran")
        return "allowed"

    def stream(event) -> None:
        if event.type == "approval_resolved":
            token.cancel()

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
            cancellation_token=token,
            stream=stream,
        ),
    )

    request_id = ""
    for event in handle.events():
        if isinstance(event, ApprovalRequestedEvent):
            request_id = event.request_id
            handle.approve(request_id, ApprovalDecision.allow())
            break

    assert request_id
    with pytest.raises(CancelledError):
        handle.result(timeout=2)
    assert calls == []


def test_cancel_during_should_request_does_not_lose_cancellation() -> None:
    calls: list[str] = []
    provider = BlockingShouldRequestApprovalProvider()

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
            approval_provider=provider,
        ),
    )

    assert provider.entered.wait(timeout=2)
    assert handle.cancel()
    provider.proceed.set()

    caught: BaseException | None = None
    try:
        handle.result(timeout=0.6)
    except BaseException as exc:
        caught = exc

    if isinstance(caught, TimeoutError):
        handle.approve(provider.request_id, ApprovalDecision.deny("cleanup"))
        with suppress(CancelledError):
            handle.result(timeout=2)

    assert not isinstance(caught, TimeoutError)
    assert isinstance(caught, CancelledError)
    assert calls == []


def test_cancel_during_should_request_false_prevents_tool_side_effect() -> None:
    calls: list[str] = []
    provider = BlockingShouldRequestApprovalProvider(should_request_result=False)

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
            approval_provider=provider,
        ),
    )

    assert provider.entered.wait(timeout=2)
    assert handle.cancel()
    provider.proceed.set()

    caught: BaseException | None = None
    try:
        handle.result(timeout=0.6)
    except BaseException as exc:
        caught = exc

    assert isinstance(caught, CancelledError)
    assert calls == []
