from __future__ import annotations

from threading import Event, Thread

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
from vv_agent.constants import TASK_FINISH_TOOL_NAME
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
            LLMResponse(
                content="finished",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "finished"})],
            ),
        ]
    )

    def model_provider(settings_file, **kwargs):
        del settings_file, kwargs
        return llm, _resolved_model()

    provider = AlwaysAskApprovalProvider()
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="test",
            llm_builder=model_provider,
            tool_registry_factory=lambda: registry,
            approval_provider=provider,
            approval_timeout_seconds=_TEST_APPROVAL_TIMEOUT_SECONDS,
            tool_policy=ToolPolicy(approval="always"),
        )
    )
    session = client.create_session(
        agent=InteractiveAgentDefinition(description="Use the tool.", model="test-model"),
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
    session.approve(request_id, "allow")
    thread.join(timeout=_TEST_APPROVAL_TIMEOUT_SECONDS)

    assert not thread.is_alive()
    assert run_error == []
    assert calls == ["ran"]
    assert provider.requests[0].metadata["session_id"] == "session-a"
