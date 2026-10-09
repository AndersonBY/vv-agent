from __future__ import annotations

from support import FixedModelProvider

from vv_agent import Agent, AgentStatus, ApprovalRequestedEvent, RunConfig, Runner, build_default_registry, function_tool
from vv_agent.approval import ApprovalBroker, ApprovalDecision, ApprovalProvider, ApprovalRequest
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse, ToolCall


class AlwaysAsk(ApprovalProvider):
    def __init__(self) -> None:
        self.requested_tools: list[str] = []

    def should_request(self, request: ApprovalRequest) -> bool:
        self.requested_tools.append(request.tool_name)
        return True

    def decide(self, request: ApprovalRequest) -> ApprovalDecision | None:
        del request
        return None


def _request(request_id: str, tool_name: str) -> ApprovalRequest:
    return ApprovalRequest(
        request_id=request_id,
        tool_name=tool_name,
        tool_call_id=f"call-{request_id}",
    )


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="m",
        selected_model="m",
        model_id="m",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="m")],
    )


def _finish() -> LLMResponse:
    return LLMResponse(content="done")


def test_broker_only_grants_session_access_for_allow_session() -> None:
    broker = ApprovalBroker()

    broker.register(_request("allow", "alpha"))
    assert broker.resolve("allow", ApprovalDecision.allow())
    assert broker.is_session_allowed("alpha") is False

    broker.register(_request("deny", "alpha"))
    assert broker.resolve("deny", ApprovalDecision.deny())
    assert broker.is_session_allowed("alpha") is False

    broker.register(_request("timeout", "alpha"))
    assert broker.resolve("timeout", ApprovalDecision.timeout())
    assert broker.is_session_allowed("alpha") is False

    broker.register(_request("session", "alpha"))
    assert broker.resolve("session", ApprovalDecision.allow_session())
    assert broker.is_session_allowed("alpha") is True
    assert broker.is_session_allowed("beta") is False
    assert broker.session_allowed_tools() == frozenset({"alpha"})


def test_allow_session_skips_same_function_tool_but_other_tool_still_requests() -> None:
    calls: list[str] = []

    @function_tool(needs_approval=True)
    def alpha() -> str:
        calls.append("alpha")
        return "alpha"

    @function_tool(needs_approval=True)
    def beta() -> str:
        calls.append("beta")
        return "beta"

    provider = AlwaysAsk()
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="call guarded tools",
                tool_calls=[
                    ToolCall(id="alpha-1", name="alpha", arguments={}),
                    ToolCall(id="alpha-2", name="alpha", arguments={}),
                    ToolCall(id="beta-1", name="beta", arguments={}),
                ],
            ),
            _finish(),
        ]
    )

    handle = Runner.start(
        Agent(name="approval", instructions="Call tools.", model="m", tools=[alpha, beta]),
        "go",
        run_config=RunConfig(model_provider=FixedModelProvider(llm, _resolved()), approval_provider=provider),
    )
    requested = []
    result = handle.result()
    while result.status is AgentStatus.WAIT_USER:
        for entry in result.metadata["session_waits"]:
            requested.extend(entry["scope"])
            decision = ApprovalDecision.allow_session() if entry["scope"] == ["alpha"] else ApprovalDecision.allow()
            handle.approve(entry["request_id"], decision)
        result = handle.resume()
    assert result.final_output == "done"
    assert requested == ["alpha", "beta"]
    assert provider.requested_tools == ["alpha", "beta"]
    assert calls == ["alpha", "alpha", "beta"]


def test_allow_session_is_honored_by_executor_orchestrator_path() -> None:
    calls: list[str] = []

    @function_tool(needs_approval=True)
    def guarded_executor() -> str:
        calls.append("ran")
        return "ok"

    def registry_factory():
        registry = build_default_registry()
        registry.register_executor(guarded_executor.to_executor())
        return registry

    provider = AlwaysAsk()
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="call twice",
                tool_calls=[
                    ToolCall(id="guarded-1", name="guarded_executor", arguments={}),
                    ToolCall(id="guarded-2", name="guarded_executor", arguments={}),
                ],
            ),
            _finish(),
        ]
    )

    handle = Runner.start(
        Agent(name="approval", instructions="Call executor.", model="m"),
        "go",
        run_config=RunConfig(
            model_provider=FixedModelProvider(llm, _resolved()),
            approval_provider=provider,
            tool_registry_factory=registry_factory,
        ),
    )
    waiting = handle.result()
    requests = [e for e in waiting.events if isinstance(e, ApprovalRequestedEvent)]
    assert len(requests) == 1
    handle.approve(requests[0].request_id, ApprovalDecision.allow_session())
    assert handle.resume().final_output == "done"
    assert provider.requested_tools == ["guarded_executor"]
    assert calls == ["ran", "ran"]
