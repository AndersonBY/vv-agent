from __future__ import annotations

from threading import Event, Thread

from support import FixedModelProvider

from vv_agent import (
    AgentSessionOptions,
    ApprovalDecision,
    ApprovalProvider,
    ApprovalRequest,
    InteractiveAgentClient,
    InteractiveAgentDefinition,
    ToolPolicy,
    build_default_registry,
    function_tool,
)
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.llm import ScriptedLLM
from vv_agent.tools.executor import FunctionToolExecutor
from vv_agent.types import LLMResponse, ToolCall

_TEST_APPROVAL_TIMEOUT_SECONDS = 10.0


class AlwaysAskApprovalProvider(ApprovalProvider):
    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []

    def should_request(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return request.tool_name == "dangerous"

    def decide(self, request: ApprovalRequest) -> ApprovalDecision | None:
        return None


def _resolved_model(model: str = "test-model") -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id=model)],
    )


def test_interactive_session_routes_approval_to_active_run_handle(tmp_path) -> None:
    calls: list[str] = []

    @function_tool(needs_approval=False)
    def dangerous() -> str:
        calls.append("ran")
        return "allowed"

    registry = build_default_registry()
    registry.register_executor(FunctionToolExecutor(dangerous))
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="dangerous", arguments={})]),
            LLMResponse(content="finished"),
        ]
    )

    provider = AlwaysAskApprovalProvider()
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            workspace=tmp_path,
            tool_registry_factory=lambda: registry,
            approval_provider=provider,
            approval_timeout_seconds=_TEST_APPROVAL_TIMEOUT_SECONDS,
            tool_policy=ToolPolicy(approval="always"),
        )
    )
    session = client.create_session(
        agent=InteractiveAgentDefinition(description="Use the tool.", model="test-model", extra_tool_names=["dangerous"]),
        session_id="session-a",
    )

    request_seen = Event()
    request_id = ""
    run_error: list[BaseException] = []

    def listener(event: str, payload: dict[str, object]) -> None:
        nonlocal request_id
        if event != "approval_requested":
            return
        request_id = str(payload.get("request_id") or "")
        request_seen.set()

    def run_prompt() -> None:
        try:
            session.prompt("go")
        except BaseException as exc:  # pragma: no cover - asserted after join
            run_error.append(exc)

    session.subscribe(listener)
    thread = Thread(target=run_prompt)
    thread.start()

    assert request_seen.wait(timeout=_TEST_APPROVAL_TIMEOUT_SECONDS)
    assert calls == []
    thread.join(timeout=_TEST_APPROVAL_TIMEOUT_SECONDS)
    session.approve(request_id, "allow")
    from vv_agent import Runner

    turn_id = client.driver.store.read_state(session.session_id)[0].active_turn_id
    assert turn_id is not None
    assert Runner.resume(session.session_id, turn_id).final_output == "finished"
    assert not thread.is_alive()
    assert run_error == []
    assert calls == ["ran"]
    assert provider.requests[0].metadata["session_id"] == "session-a"


def test_allow_session_persists_across_automatic_follow_up(tmp_path) -> None:
    calls: list[str] = []

    @function_tool(needs_approval=False)
    def dangerous() -> str:
        calls.append("ran")
        return "allowed"

    registry = build_default_registry()
    registry.register_executor(FunctionToolExecutor(dangerous))
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="first call", tool_calls=[ToolCall(id="call_1", name="dangerous", arguments={})]),
            LLMResponse(content="first"),
            LLMResponse(content="second call", tool_calls=[ToolCall(id="call_2", name="dangerous", arguments={})]),
            LLMResponse(content="second"),
        ]
    )

    provider = AlwaysAskApprovalProvider()
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            workspace=tmp_path,
            tool_registry_factory=lambda: registry,
            approval_provider=provider,
            approval_timeout_seconds=_TEST_APPROVAL_TIMEOUT_SECONDS,
            tool_policy=ToolPolicy(approval="always"),
        )
    )
    session = client.create_session(
        agent=InteractiveAgentDefinition(description="Use the tool twice.", model="test-model", extra_tool_names=["dangerous"]),
        session_id="session-allow",
    )
    waiting = session.prompt("run it", auto_follow_up=False)
    assert waiting.status.value == "wait_user"
    request_id = waiting.metadata["session_waits"][0]["request_id"]
    session.approve(request_id, "allow_session")
    from vv_agent import Runner

    assert Runner.resume(session.session_id, waiting.run_id).final_output == "first"
    result = session.prompt("run it again")
    assert result.final_output == "second"
    assert calls == ["ran", "ran"]
    assert [request.tool_name for request in provider.requests].count("dangerous") == 1


def test_interactive_session_exposes_active_run_handle_lifecycle(tmp_path) -> None:
    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="finished"),
        ]
    )

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            workspace=tmp_path,
        )
    )
    session = client.create_session(
        agent=InteractiveAgentDefinition(description="Finish immediately.", model="test-model"),
        session_id="session-a",
    )
    active_handles: list[object | None] = []

    def listener(event: str, payload: dict[str, object]) -> None:
        if event == "session_active_run_handle_changed":
            active_handles.append(payload.get("handle"))

    session.subscribe(listener)
    session.prompt("go")

    assert len(active_handles) >= 2
    assert active_handles[0] is not None
    assert active_handles[-1] is None
    assert session.active_run_handle is None
