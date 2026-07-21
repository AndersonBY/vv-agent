from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

import vv_agent
from vv_agent import (
    Agent,
    AgentSessionOptions,
    AgentSessionRun,
    AgentStatus,
    InteractiveAgentClient,
    InteractiveAgentDefinition,
    Message,
    ModelSettings,
    ScriptedModelProvider,
    create_agent_session,
    function_tool,
    handoff,
    input_guardrail,
    output_guardrail,
)
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import CREATE_SUB_TASK_TOOL_NAME, TASK_FINISH_TOOL_NAME
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm import LlmRequest
from vv_agent.runtime import BaseRuntimeHook, BeforeLLMEvent
from vv_agent.types import AgentResult, LLMResponse, SubAgentConfig, ToolCall


def _resolved() -> ResolvedModelConfig:
    return ResolvedModelConfig(
        backend="moonshot",
        requested_model="kimi-k2.6",
        selected_model="kimi-k2.6",
        model_id="kimi-k2.6",
        endpoint_options=[
            EndpointOption(
                endpoint=EndpointConfig(
                    endpoint_id="default",
                    api_key="test-key",
                    api_base="https://example.test/v1",
                ),
                model_id="kimi-k2.6",
            )
        ],
    )


def _completed_run(
    *,
    agent_name: str = "inline",
    prompt: str = "hello",
    shared_state: dict[str, Any] | None = None,
) -> AgentSessionRun:
    state = dict(shared_state or {})
    state.setdefault("todo_list", [])
    return AgentSessionRun(
        agent_name=agent_name,
        result=AgentResult(
            status=AgentStatus.COMPLETED,
            messages=[
                Message(role="user", content=prompt),
                Message(role="assistant", content=f"answer: {prompt}"),
            ],
            cycles=[],
            final_answer=f"answer: {prompt}",
            shared_state=state,
        ),
        resolved=_resolved(),
    )


def test_top_level_public_api_exports_interactive_session_names() -> None:
    expected = {
        "AgentSession",
        "AgentSessionOptions",
        "AgentSessionRun",
        "AgentSessionState",
        "InteractiveAgentClient",
        "InteractiveAgentDefinition",
        "create_agent_session",
    }

    assert expected.issubset(set(vv_agent.__all__))
    for name in expected:
        assert hasattr(vv_agent, name)


def test_agent_session_preserves_session_id_messages_shared_state_and_events(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def execute_run(**kwargs: Any) -> AgentSessionRun:
        calls.append(kwargs)
        return _completed_run(
            agent_name=kwargs["task_name"],
            prompt=kwargs["prompt"],
            shared_state={**kwargs["shared_state"], "last_prompt": kwargs["prompt"]},
        )

    definition = InteractiveAgentDefinition(description="desktop agent", model="kimi-k2.6")
    session = create_agent_session(
        execute_run=execute_run,
        session_id="desktop-session-1",
        agent_name="desktop",
        definition=definition,
        workspace=tmp_path,
        shared_state={"todo_list": [{"title": "existing", "status": "pending"}]},
    )
    events: list[tuple[str, dict[str, Any]]] = []
    unsubscribe = session.subscribe(lambda event, payload: events.append((event, payload)))

    run = session.prompt("hello")
    unsubscribe()

    assert session.session_id == "desktop-session-1"
    assert run.result.final_answer == "answer: hello"
    assert session.messages[-1].content == "answer: hello"
    assert session.shared_state["last_prompt"] == "hello"
    assert calls[0]["session_id"] == "desktop-session-1"
    assert calls[0]["initial_messages"] is None
    assert calls[0]["cancellation_token"] is not None
    assert events[0][0] == "session_run_start"
    assert events[-1][0] == "session_run_end"


def test_agent_session_requires_execute_run_to_accept_cancellation_token(tmp_path: Path) -> None:
    def execute_run(
        *,
        prompt: str,
        session_id: str,
        agent: InteractiveAgentDefinition,
        task_name: str,
        workspace: Path,
        shared_state: dict[str, Any],
        initial_messages: list[Message],
        before_cycle_messages: Any,
        interruption_messages: Any,
        log_handler: Any,
    ) -> AgentSessionRun:
        del session_id, agent, task_name, workspace, initial_messages, before_cycle_messages, interruption_messages, log_handler
        return _completed_run(prompt=prompt, shared_state=shared_state)

    session = create_agent_session(
        execute_run=execute_run,
        session_id="desktop-session-requires-cancel",
        agent_name="desktop",
        definition=InteractiveAgentDefinition(description="desktop agent", model="kimi-k2.6"),
        workspace=tmp_path,
    )

    with pytest.raises(TypeError, match="cancellation_token"):
        session.prompt("hello")


def test_agent_session_queues_steering_and_follow_up_prompts(tmp_path: Path) -> None:
    prompts: list[str] = []

    def execute_run(**kwargs: Any) -> AgentSessionRun:
        prompts.append(kwargs["prompt"])
        before_cycle_messages = kwargs["before_cycle_messages"](1, [], {})
        prompts.extend(message.content for message in before_cycle_messages)
        return _completed_run(prompt=kwargs["prompt"], shared_state=kwargs["shared_state"])

    session = create_agent_session(
        execute_run=execute_run,
        session_id="desktop-session-2",
        agent_name="desktop",
        definition=InteractiveAgentDefinition(description="desktop agent", model="kimi-k2.6"),
        workspace=tmp_path,
    )

    session.steer("interrupt with context")
    session.steer("second steering message")
    session.follow_up("continue with next turn")
    session.prompt("start")

    assert prompts == [
        "start",
        "interrupt with context",
        "second steering message",
        "continue with next turn",
    ]


def test_interactive_client_prepare_task_maps_definition_to_runtime_task(tmp_path: Path) -> None:
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="moonshot",
            workspace=tmp_path,
        )
    )
    definition = InteractiveAgentDefinition(
        description="Use the computer.",
        model="kimi-k2.6",
        backend="moonshot",
        max_cycles=5,
        memory_compact_threshold=2048,
        memory_threshold_percentage=80,
        no_tool_policy="finish",
        native_multimodal=True,
        extra_tool_names=["read_image"],
        exclude_tools=["ask_user"],
        bash_shell="/bin/bash",
        windows_shell_priority=["pwsh", "powershell"],
        bash_env={"VCLAW_SESSION_ID": "sid-1"},
        metadata={"session_id": "sid-1"},
        system_prompt="custom system prompt",
    )

    task = client.prepare_task(
        prompt="open browser",
        resolved_model_id="kimi-k2.6",
        resolved_context_length=1_048_576,
        resolved_max_output_tokens=1_048_576,
        agent=definition,
        task_name="desktop",
        workspace=tmp_path,
        session_id="sid-1",
    )

    assert task.model == "kimi-k2.6"
    assert task.system_prompt == "custom system prompt"
    assert task.user_prompt == "open browser"
    assert task.max_cycles == 5
    assert task.memory_compact_threshold == 2048
    assert task.memory_threshold_percentage == 80
    assert task.no_tool_policy == "finish"
    assert task.native_multimodal is True
    assert task.extra_tool_names == ["read_image"]
    assert task.exclude_tools == ["ask_user"]
    assert task.metadata["session_id"] == "sid-1"
    assert task.metadata["bash_shell"] == "/bin/bash"
    assert task.metadata["windows_shell_priority"] == ["pwsh", "powershell"]
    assert task.metadata["bash_env"] == {"VCLAW_SESSION_ID": "sid-1"}
    assert task.metadata["model_context_window"] == 1_048_576
    assert task.metadata["model_max_output_tokens"] == 1_048_576
    assert "reserved_output_tokens" not in task.metadata


def test_interactive_definition_uses_contract_memory_threshold_default(tmp_path: Path) -> None:
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="moonshot",
            workspace=tmp_path,
        )
    )
    definition = InteractiveAgentDefinition(description="Use the computer.", model="kimi-k3")

    task = client.prepare_task(
        prompt="inspect",
        resolved_model_id="kimi-k3",
        agent=definition,
    )

    assert definition.memory_compact_threshold == 250_000
    assert task.memory_compact_threshold == 250_000


def test_interactive_definition_preserves_explicit_zero_memory_threshold(tmp_path: Path) -> None:
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="moonshot",
            workspace=tmp_path,
        )
    )
    definition = InteractiveAgentDefinition(
        description="Use the computer.",
        model="kimi-k3",
        memory_compact_threshold=0,
    )

    task = client.prepare_task(
        prompt="inspect",
        resolved_model_id="kimi-k3",
        agent=definition,
    )

    assert task.memory_compact_threshold == 0


def test_interactive_client_create_session_preserves_caller_session_id(tmp_path: Path) -> None:
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="moonshot",
            workspace=tmp_path,
        )
    )

    session = client.create_session(
        agent=InteractiveAgentDefinition(description="desktop agent", model="kimi-k2.6"),
        workspace=tmp_path,
        session_id="caller-session-id",
    )

    assert session.session_id == "caller-session-id"


def test_interactive_client_preserves_complete_public_agent(tmp_path: Path) -> None:
    dynamic_contexts: list[tuple[str, str, Path, dict[str, Any]]] = []
    hook_calls: list[str] = []
    tool_calls: list[str] = []
    guardrail_calls: list[str] = []
    output_guardrail_calls: list[str] = []
    requests: list[LlmRequest] = []

    @function_tool
    def remember(value: str) -> str:
        """Remember a value."""
        tool_calls.append(value)
        return value

    @input_guardrail
    def record_guardrail(context, value: str) -> GuardrailResult:
        guardrail_calls.append(f"{context.agent_name}:{value}")
        return GuardrailResult.allow()

    @output_guardrail
    def rewrite_output(context, value: str) -> GuardrailResult:
        output_guardrail_calls.append(f"{context.agent_name}:{value}")
        return GuardrailResult.rewrite('{"status":"guarded"}')

    class RecordHook(BaseRuntimeHook):
        def before_llm(self, event: BeforeLLMEvent):
            hook_calls.append(event.task.task_id)
            return None

    def instructions(context, current_agent) -> str:
        dynamic_contexts.append((context.agent_name, str(context.model), Path(context.workspace), dict(context.metadata)))
        assert current_agent is agent
        return "Dynamic instructions."

    configured_child = SubAgentConfig(model="child-model", description="Research the request.")
    agent = Agent(
        name="interactive-agent",
        instructions=instructions,
        model="parent-model",
        model_settings=ModelSettings(temperature=0.25, max_tokens=321),
        tools=[remember],
        input_guardrails=[record_guardrail],
        output_guardrails=[rewrite_output],
        output_type=dict,
        hooks=[RecordHook()],
        metadata={"agent_marker": "kept"},
        sub_agents={"researcher": configured_child},
    )

    def capture_request(request: LlmRequest) -> LLMResponse:
        requests.append(request)
        tool_names = [str(cast(dict[str, object], item["function"])["name"]) for item in request.tools]
        assert "remember" in tool_names
        assert CREATE_SUB_TASK_TOOL_NAME in tool_names
        if len(requests) == 1:
            return LLMResponse(
                content="remember",
                tool_calls=[ToolCall(id="remember-call", name="remember", arguments={"value": "kept"})],
            )
        return LLMResponse(
            content="finish",
            tool_calls=[
                ToolCall(
                    id="finish-call",
                    name=TASK_FINISH_TOOL_NAME,
                    arguments={"message": '{"status":"ok"}'},
                )
            ],
        )

    provider = ScriptedModelProvider.from_callback("test", "parent-model", capture_request)
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="test",
            workspace=tmp_path,
            llm_builder=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("builder must not run")),
        )
    )
    agent.model = provider.client(provider.resolve(provider.default_model_ref()))
    session = client.create_session(agent=agent, session_id="public-agent-session")

    run = session.prompt("preserve everything")

    assert session.agent is agent
    assert session.definition is None
    assert session.agent_name == "interactive-agent"
    assert run.agent_name == "interactive-agent"
    assert run.final_output == {"status": "guarded"}
    assert tool_calls == ["kept"]
    assert guardrail_calls == ["interactive-agent:preserve everything"]
    assert output_guardrail_calls == ['interactive-agent:{"status":"ok"}']
    assert len(hook_calls) == 2
    assert len(dynamic_contexts) == 1
    agent_name, model, workspace, metadata = dynamic_contexts[0]
    assert (agent_name, model, workspace) == ("interactive-agent", "direct", tmp_path.resolve())
    assert metadata["agent_marker"] == "kept"
    assert metadata["session_id"] == "public-agent-session"
    assert metadata["trace_id"]
    assert requests[0].model_settings == ModelSettings(temperature=0.25, max_tokens=321)
    assert requests[0].metadata["session_id"] == "public-agent-session"
    assert requests[0].metadata["system_prompt_sources"] == {
        "agent_instructions": "agent.instructions",
        "configured_sub_agents": "agent.sub_agents",
    }
    assert configured_child.description == "Research the request."


def test_interactive_client_preserves_public_agent_handoff(tmp_path: Path) -> None:
    shared_llm = ScriptedModelProvider.new(
        "test",
        "shared-model",
        [
            LLMResponse(
                content="transfer",
                tool_calls=[
                    ToolCall(
                        id="handoff-call",
                        name="transfer_to_writer",
                        arguments={"input": "write it"},
                    )
                ],
            ),
            LLMResponse(
                content="written",
                tool_calls=[
                    ToolCall(
                        id="writer-finish",
                        name=TASK_FINISH_TOOL_NAME,
                        arguments={"message": "writer result"},
                    )
                ],
            ),
        ],
    ).llm
    writer = Agent(name="writer", instructions="Write.", model=shared_llm)
    triage = Agent(
        name="triage",
        instructions="Transfer.",
        model=shared_llm,
        handoffs=[handoff(agent=writer, description="Write the result.")],
    )
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="test",
            workspace=tmp_path,
        )
    )

    session = client.create_session(agent=triage)
    run = session.prompt("route this")

    assert session.agent is triage
    assert run.agent_name == "writer"
    assert run.final_output == "writer result"
    assert [event.type for event in run.events if event.type.startswith("handoff_")] == [
        "handoff_started",
        "handoff_completed",
    ]


def test_interactive_definition_rejects_ignored_system_prompt_template() -> None:
    with pytest.raises(ValueError, match="system_prompt_template is not supported"):
        InteractiveAgentDefinition(
            description="desktop agent",
            model="kimi-k2.6",
            system_prompt_template="ignored {description}",
        )


def test_interactive_client_requires_debug_dump_capable_llm(tmp_path: Path) -> None:
    class NoDebugDumpLLM:
        __slots__ = ()

    def llm_builder(*_: Any, **__: Any) -> tuple[NoDebugDumpLLM, ResolvedModelConfig]:
        return NoDebugDumpLLM(), _resolved()

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=cast(Any, llm_builder),
            debug_dump_dir=str(tmp_path / "debug"),
        )
    )

    with pytest.raises(AttributeError):
        client._execute(
            prompt="hello",
            agent=InteractiveAgentDefinition(description="desktop agent", model="kimi-k2.6"),
            workspace=tmp_path,
        )


def test_public_sub_agent_session_registry_wrappers() -> None:
    from vv_agent.runtime.engine import (
        get_sub_agent_session,
        register_sub_agent_session,
        unregister_sub_agent_session,
    )

    session = object()
    register_sub_agent_session("sub-session", session)
    try:
        assert get_sub_agent_session(session_id="sub-session") is session
    finally:
        unregister_sub_agent_session("sub-session", session)

    assert get_sub_agent_session(session_id="sub-session") is None
