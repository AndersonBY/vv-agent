from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from support import FixedModelProvider

import vv_agent
from vv_agent import (
    Agent,
    AgentSessionOptions,
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
from vv_agent.constants import CREATE_SUB_TASK_TOOL_NAME
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm import LlmRequest
from vv_agent.runtime.hooks import BaseRuntimeHook, BeforeLLMEvent
from vv_agent.types import LLMResponse, SubAgentConfig, ToolCall


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


def _empty_model_provider() -> ScriptedModelProvider:
    return ScriptedModelProvider.new("moonshot", "kimi-k2.6", [])


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


def test_agent_session_preserves_session_id_messages_shared_state_and_events(tmp_path):
    @function_tool
    def remember(context: vv_agent.ToolContext, value: str) -> str:
        context.shared_state["last_prompt"] = value
        return "answer: " + value

    provider = ScriptedModelProvider.new("test", "m", [LLMResponse("", [ToolCall("remember", "remember", {"value": "hello"})])])
    session = create_agent_session(
        agent=Agent("desktop", "Remember.", tools=[remember], tool_use_behavior="stop_on_first_tool"),
        options=AgentSessionOptions(model_provider=provider, workspace=tmp_path),
        session_id="desktop-session-1",
        shared_state={"todo_list": [{"title": "existing", "status": "pending"}]},
    )
    events = []
    unsubscribe = session.subscribe(lambda event, payload: events.append((event, payload)))
    try:
        run = session.prompt("hello")
        unsubscribe()
        assert session.session_id == run.raw_result.session_id == "desktop-session-1"
        assert run.final_output == "answer: hello"
        assert session.messages[-1].content == "answer: hello"
        assert session.shared_state["last_prompt"] == "hello"
        assert events[0][0] == "session_run_start"
        assert events[-1][0] == "session_run_end"
    finally:
        session.driver.close()


def test_interactive_client_prepare_task_maps_definition_to_runtime_task(tmp_path: Path) -> None:
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=_empty_model_provider(),
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
    assert task.prompt_bundle.flatten() == "custom system prompt"
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
            model_provider=_empty_model_provider(),
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


def test_interactive_task_uses_resolved_context_when_metadata_is_non_positive(
    tmp_path: Path,
) -> None:
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=_empty_model_provider(),
            workspace=tmp_path,
        )
    )
    definition = InteractiveAgentDefinition(
        description="Use the computer.",
        model="capacity-model",
        metadata={"model_context_window": 0},
    )

    task = client.prepare_task(
        prompt="inspect",
        resolved_model_id="capacity-model",
        resolved_context_length=64_000,
        resolved_max_output_tokens=8_192,
        agent=definition,
    )

    assert task.metadata["model_context_window"] == 64_000
    assert task.metadata["model_max_output_tokens"] == 8_192


def test_interactive_definition_preserves_explicit_zero_memory_threshold(tmp_path: Path) -> None:
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=_empty_model_provider(),
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


def test_interactive_client_create_session_preserves_caller_session_id(surface, tmp_path: Path) -> None:
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=_empty_model_provider(),
            workspace=tmp_path,
        )
    )

    session = client.create_session(
        agent=InteractiveAgentDefinition(description="desktop agent", model="kimi-k2.6"),
        workspace=tmp_path,
        session_id="caller-session-id",
    )

    assert session.session_id == "caller-session-id"


def test_kernel_session_creation_seed_is_durable_and_read_only(surface, tmp_path):
    history = [Message("user", "retained history")]
    provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse("done")])
    client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, workspace=tmp_path))
    session = client.create_session(
        agent=Agent("assistant", "Work.", model="m"),
        session_id="seeded",
        session={"messages": [m.to_dict() for m in history], "shared_state": {"nested": {"value": 2}}},
    )
    assert session.messages == history
    assert session.shared_state == {"nested": {"value": 2}}
    session.shared_state["nested"]["value"] = 99
    assert session.shared_state == {"nested": {"value": 2}}
    for name in ("replace_messages", "replace_shared_state", "clear_queues", "session"):
        assert name not in dir(session) and not hasattr(session, name)
    run = session.prompt("go")
    assert run.result.shared_state == {"nested": {"value": 2}}
    assert [m.content for m in session.messages if m.role == "user"] == ["retained history", "go"]


def test_interactive_client_preserves_complete_public_agent(surface, tmp_path: Path) -> None:
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
        return LLMResponse(content='{"status":"ok"}')

    provider = ScriptedModelProvider.from_callback("test", "parent-model", capture_request)
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=provider,
            workspace=tmp_path,
        )
    )
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
    assert (agent_name, model, workspace) == ("interactive-agent", "parent-model", tmp_path.resolve())
    assert metadata["agent_marker"] == "kept"
    assert metadata["session_id"] == "public-agent-session"
    assert metadata["trace_id"]
    from vv_agent.model_settings import RetrySettings

    assert requests[0].model_settings == ModelSettings(
        temperature=0.25, max_tokens=321, retry=RetrySettings(max_attempts=1, backoff_seconds=0)
    )
    assert requests[0].metadata["session_id"] == "public-agent-session"
    assert requests[0].prompt_bundle is not None
    assert [(section.id, section.source) for section in requests[0].prompt_bundle.sections] == [
        ("agent_instructions", "agent.instructions"),
        ("configured_sub_agents", "agent.sub_agents"),
    ]
    assert configured_child.description == "Research the request."


def test_interactive_client_preserves_public_agent_handoff(surface, tmp_path: Path) -> None:
    provider = ScriptedModelProvider.new(
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
            LLMResponse(content="writer result"),
        ],
    )
    writer = Agent(name="writer", instructions="Write.", model="shared-model")
    triage = Agent(
        name="triage",
        instructions="Transfer.",
        model="shared-model",
        handoffs=[handoff(agent=writer, description="Write the result.")],
    )
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=provider,
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


def test_interactive_client_requires_debug_dump_capable_llm(tmp_path: Path) -> None:
    class NoDebugDumpLLM:
        __slots__ = ()

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=FixedModelProvider(cast(Any, NoDebugDumpLLM()), _resolved()),
            workspace=tmp_path,
            debug_dump_dir=str(tmp_path / "debug"),
        )
    )

    with pytest.raises(AttributeError):
        client._execute(
            prompt="hello",
            agent=InteractiveAgentDefinition(description="desktop agent", model="kimi-k2.6"),
            workspace=tmp_path,
        )


def test_interactive_real_same_turn_user_reply(surface, tmp_path):
    provider = ScriptedModelProvider.new(
        "test",
        "m",
        [
            LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Choose a value"})]),
            LLMResponse("resumed"),
        ],
    )
    client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, workspace=tmp_path))
    session = client.create_session(agent=Agent("assistant", "Ask.", model="m"), session_id="interactive-wait")
    first = session.prompt("go", auto_follow_up=False)
    assert first.status == AgentStatus.WAIT_USER
    second = session.continue_run("value")
    assert second.final_output == "resumed"
    assert second.run_id == first.run_id
    state, records, _ = surface.store.read_state(session.session_id)
    assert len(state.turns) == 1
    assert sum(r.record.kind == "turn_ended" for r in records) == 1


def test_interactive_live_steer_and_durable_follow_up(surface, tmp_path):
    import threading

    ready, release = threading.Event(), threading.Event()
    requests = []

    def first(request):
        ready.set()
        assert release.wait(3)
        return LLMResponse("", [ToolCall("todo", "todo_write", {"todos": []})])

    def model(request):
        requests.append([m.content for m in request.messages if m.role == "user"])
        return LLMResponse("done")

    callbacks = iter([first, model, model])
    provider = ScriptedModelProvider.from_callback("test", "m", lambda request: next(callbacks)(request))
    client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, workspace=tmp_path))
    session = client.create_session(agent=Agent("assistant", "Work.", model="m"), session_id="interactive-controls")
    result, errors = [], []

    def prompt():
        try:
            result.append(session.prompt("first"))
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=prompt)
    worker.start()
    assert ready.wait(3)
    session.steer("steering")
    session.follow_up("next")
    assert session.state().pending_follow_ups == 1
    release.set()
    worker.join(5)
    assert not worker.is_alive() and not errors, errors
    assert result[0].final_output == "done"
    assert requests[0] == ["first", "steering"]
    assert requests[1] == ["first", "steering", "next"]
    state, records, _ = surface.store.read_state(session.session_id)
    assert len(state.turns) == 2
    assert any(r.record.kind == "input_applied" and r.record._payload["input"]["kind"] == "steer" for r in records)
    assert any(r.record.kind == "input_applied" and r.record._payload["input"]["kind"] == "follow_up" for r in records)


def test_interactive_child_wait_reply_keeps_child_identity(surface, tmp_path):
    from vv_agent.types import ToolDirective, ToolExecutionResult

    @function_tool
    def wait_child() -> ToolExecutionResult:
        return ToolExecutionResult(
            "", "Need child input", directive=ToolDirective.WAIT_USER, metadata={"question": "Need child input"}
        )

    child = Agent("worker", "Work.", model="m", tools=[wait_child])
    provider = ScriptedModelProvider.new(
        "test",
        "m",
        [
            LLMResponse("", [ToolCall("child", "worker", {"task_description": "work"})]),
            LLMResponse("", [ToolCall("wait", "wait_child", {})]),
            LLMResponse("child done"),
            LLMResponse("parent done"),
        ],
    )
    client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, workspace=tmp_path))
    session = client.create_session(agent=Agent("assistant", "Delegate.", model="m", tools=[child.as_tool()]))
    first = session.prompt("go", auto_follow_up=False)
    assert first.status == AgentStatus.WAIT_USER
    wait = first.metadata["session_waits"][0]
    assert wait["question"] == "Need child input" and wait["session_id"] != session.session_id
    final = session.continue_run("child answer")
    assert final.run_id == first.run_id and final.final_output == "parent done"
    child_state = surface.store.read_state(wait["session_id"])[0]
    assert len(child_state.turns) == 1 and child_state.active_turn_id is None


def test_interactive_sequential_children_complete_in_one_prompt(surface, tmp_path):
    child = Agent("worker", "Work.", model="m")
    provider = ScriptedModelProvider.new(
        "test",
        "m",
        [
            LLMResponse("", [ToolCall("first", "worker", {"task_description": "one"})]),
            LLMResponse("one done"),
            LLMResponse("", [ToolCall("second", "worker", {"task_description": "two"})]),
            LLMResponse("two done"),
            LLMResponse("parent done"),
        ],
    )
    client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, workspace=tmp_path))
    session = client.create_session(agent=Agent("parent", "Delegate.", model="m", tools=[child.as_tool()]))
    assert session.prompt("go").final_output == "parent done"
    ids = surface.store.list_sessions()
    assert len(ids) == 3
    assert all(surface.store.read_state(sid)[0].active_turn_id is None for sid in ids)


def test_interactive_background_child_runs_independently(surface, tmp_path):
    import threading

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def model(request):
        if any(m.role == "user" and m.content == "background work" for m in request.messages):
            entered.set()
            assert release.wait(5)
            finished.set()
            return LLMResponse("child done")
        if not any(m.role == "tool" for m in request.messages):
            return LLMResponse("", [ToolCall("bg", "worker_background_task", {"task_description": "background work"})])
        return LLMResponse("parent done")

    child = Agent("worker", "Work.", model="m")
    provider = ScriptedModelProvider.from_callback("test", "m", model)
    client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, workspace=tmp_path))
    session = client.create_session(agent=Agent("parent", "Delegate.", model="m", tools=[child.as_background_task()]))
    try:
        result = session.prompt("go", auto_follow_up=False)
        assert result.final_output == "parent done" and entered.wait(3)
        assert not finished.is_set()
    finally:
        release.set()
    assert finished.wait(3)
    for handle in client.driver.handles:
        handle.join(3)
    children = [sid for sid in surface.store.list_sessions() if sid != session.session_id]
    assert len(children) == 1
    state, records, _ = surface.store.read_state(children[0])
    assert state.active_turn_id is None
    assert any(r.record.kind == "turn_ended" and r.record._payload["result"] == "child done" for r in records)


def test_interactive_close_during_model_call_is_idempotent(surface, tmp_path):
    import threading

    entered, release = threading.Event(), threading.Event()

    def model(request):
        entered.set()
        assert release.wait(5)
        return LLMResponse("too late")

    provider = ScriptedModelProvider.from_callback("test", "m", model)
    client = InteractiveAgentClient(options=AgentSessionOptions(model_provider=provider, workspace=tmp_path))
    session = client.create_session(agent=Agent("parent", "Work.", model="m"))
    results = []
    worker = threading.Thread(target=lambda: results.append(session.prompt("go")))
    worker.start()
    assert entered.wait(3)
    assert session.close()
    assert not session.close()
    release.set()
    worker.join(5)
    assert not worker.is_alive() and session.closed
    state, records, _ = surface.store.read_state(session.session_id)
    assert state.closed and len(state.turns) == 1
    assert sum(r.record.kind == "turn_ended" for r in records) == 1
    assert next(r.record for r in records if r.record.kind == "turn_ended")._payload["status"] == "cancelled"
