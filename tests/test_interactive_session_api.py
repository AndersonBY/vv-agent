from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

import vv_agent
from vv_agent import (
    AgentSessionOptions,
    AgentSessionRun,
    AgentStatus,
    InteractiveAgentClient,
    InteractiveAgentDefinition,
    Message,
    create_agent_session,
)
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.types import AgentResult


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
    assert calls[0]["initial_messages"] == []
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
    session.follow_up("continue with next turn")
    session.prompt("start")

    assert prompts == ["start", "interrupt with context", "continue with next turn"]


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
