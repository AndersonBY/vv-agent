from __future__ import annotations

import json
from pathlib import Path

import pytest
from support import ModelMapProvider

from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import CREATE_SUB_TASK_TOOL_NAME, TASK_FINISH_TOOL_NAME
from vv_agent.events import (
    AssistantDeltaEvent,
    ModelToolCallProgressEvent,
    ModelToolCallStartedEvent,
    RunEvent,
    SubRunCompletedEvent,
    SubRunStartedEvent,
)
from vv_agent.llm import LLMClient, ScriptedLLM
from vv_agent.model import ModelRef
from vv_agent.runtime import AgentRuntime
from vv_agent.runtime.backends.inline import InlineBackend
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.engine import (
    _register_sub_agent_session,
    _unregister_sub_agent_session,
    get_sub_agent_session,
    steer_sub_agent_session,
    subscribe_sub_agent_session,
)
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentStatus, AgentTask, LLMResponse, SubAgentConfig, SubTaskRequest, ToolCall


def _fake_resolved(*, backend: str, model: str) -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake-endpoint", api_key="k", api_base="https://example.invalid/v1")
    option = EndpointOption(endpoint=endpoint, model_id=model)
    return ResolvedModelConfig(
        backend=backend,
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[option],
    )


def _shared_model_provider(
    *,
    parent_llm: LLMClient,
    child_llm: LLMClient,
    child_model: str = "kimi-k2.5",
    child_backend: str = "moonshot",
) -> ModelMapProvider:
    return ModelMapProvider(
        routes={
            "parent-model": (parent_llm, _fake_resolved(backend="moonshot", model="parent-model")),
            child_model: (child_llm, _fake_resolved(backend=child_backend, model=child_model)),
        },
        default_model="parent-model",
    )


def _parent_client(provider: ModelMapProvider) -> LLMClient:
    return provider.client(provider.resolve(ModelRef.named("parent-model")))


def test_sub_agent_session_helpers_expose_registered_session() -> None:
    captured_events: list[tuple[str, dict[str, int]]] = []

    class _DummySession:
        def __init__(self) -> None:
            self.listener = None

        def subscribe(self, listener):
            self.listener = listener

            def _unsubscribe() -> None:
                self.listener = None

            return _unsubscribe

    session = _DummySession()
    _register_sub_agent_session("sub-session-1", session)
    try:
        assert get_sub_agent_session(session_id="sub-session-1") is session
        unsubscribe = subscribe_sub_agent_session(
            session_id="sub-session-1",
            listener=lambda event, payload: captured_events.append((event, dict(payload))),
        )
        assert callable(unsubscribe)
        assert session.listener is not None
        session.listener("cycle_started", {"cycle": 1})
        assert captured_events == [("cycle_started", {"cycle": 1})]
        unsubscribe()
        assert session.listener is None
    finally:
        _unregister_sub_agent_session("sub-session-1", session)


def test_sub_agent_session_helpers_raise_for_registered_non_session() -> None:
    bad_session = object()
    _register_sub_agent_session("bad-sub-session", bad_session)
    try:
        with pytest.raises(AttributeError, match="subscribe"):
            subscribe_sub_agent_session(session_id="bad-sub-session", listener=lambda *_args: None)
        with pytest.raises(AttributeError, match="steer"):
            steer_sub_agent_session(session_id="bad-sub-session", prompt="focus")
    finally:
        _unregister_sub_agent_session("bad-sub-session", bad_session)


def test_create_sub_task_executes_configured_sub_agent(tmp_path: Path) -> None:
    parent_llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="delegate",
                tool_calls=[
                    ToolCall(
                        id="p1",
                        name=CREATE_SUB_TASK_TOOL_NAME,
                        arguments={
                            "agent_id": "research-sub",
                            "task_description": "Collect core facts",
                            "output_requirements": "Return short bullet list",
                        },
                    )
                ],
            ),
            LLMResponse(
                content="finish parent",
                tool_calls=[ToolCall(id="p2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "parent done"})],
            ),
        ]
    )

    sub_llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="sub done",
                tool_calls=[ToolCall(id="s1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "sub-result"})],
            )
        ]
    )
    provider = _shared_model_provider(parent_llm=parent_llm, child_llm=sub_llm)

    runtime = AgentRuntime(
        llm_client=_parent_client(provider),
        model_provider=provider,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        tool_registry_factory=build_default_registry,
    )
    task = AgentTask(
        task_id="parent",
        model="parent-model",
        system_prompt="sys",
        user_prompt="run parent task",
        max_cycles=4,
        sub_agents={
            "research-sub": SubAgentConfig(
                model="kimi-k2.5",
                backend="moonshot",
                description="collect facts",
            )
        },
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "parent done"
    assert provider.resolved_models == ["parent-model", "kimi-k2.5"]

    first_tool_payload = json.loads(result.cycles[0].tool_results[0].content)
    assert first_tool_payload["status"] == "completed"
    assert first_tool_payload["final_answer"] == "sub-result"
    assert first_tool_payload["resolved"]["backend"] == "moonshot"


def test_create_sub_task_batch_aggregates_sub_agent_results(tmp_path: Path) -> None:
    parent_llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="batch delegate",
                tool_calls=[
                    ToolCall(
                        id="p1",
                        name=CREATE_SUB_TASK_TOOL_NAME,
                        arguments={
                            "agent_id": "writer-sub",
                            "tasks": [
                                {"task_description": "Write section A"},
                                {"task_description": "Write section B"},
                            ],
                        },
                    )
                ],
            ),
            LLMResponse(
                content="finish parent",
                tool_calls=[ToolCall(id="p2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "batch done"})],
            ),
        ]
    )

    sub_llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="sub done",
                tool_calls=[ToolCall(id="s1", name=TASK_FINISH_TOOL_NAME, arguments={"message": answer})],
            )
            for answer in ("sub-A", "sub-B")
        ]
    )
    provider = _shared_model_provider(parent_llm=parent_llm, child_llm=sub_llm)

    runtime = AgentRuntime(
        llm_client=_parent_client(provider),
        model_provider=provider,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        tool_registry_factory=build_default_registry,
    )
    task = AgentTask(
        task_id="parent_batch",
        model="parent-model",
        system_prompt="sys",
        user_prompt="run parent batch task",
        max_cycles=4,
        sub_agents={
            "writer-sub": SubAgentConfig(
                model="kimi-k2.5",
                backend="moonshot",
                description="write sections",
            )
        },
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "batch done"
    assert provider.resolved_models == ["parent-model", "kimi-k2.5", "kimi-k2.5"]

    batch_payload = json.loads(result.cycles[0].tool_results[0].content)
    assert batch_payload["summary"] == {"total": 2, "completed": 2, "failed": 0}
    assert batch_payload["results"][0]["final_answer"] == "sub-A"
    assert batch_payload["results"][1]["final_answer"] == "sub-B"


def test_create_sub_task_batch_uses_execution_backend_parallel_map(tmp_path: Path) -> None:
    class _TrackingInlineBackend(InlineBackend):
        def __init__(self) -> None:
            super().__init__()
            self.parallel_map_calls = 0

        def parallel_map(self, fn, items):
            self.parallel_map_calls += 1
            return super().parallel_map(fn, items)

    backend = _TrackingInlineBackend()
    parent_llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="batch delegate",
                tool_calls=[
                    ToolCall(
                        id="p1",
                        name=CREATE_SUB_TASK_TOOL_NAME,
                        arguments={
                            "agent_id": "writer-sub",
                            "tasks": [
                                {"task_description": "Write section A"},
                                {"task_description": "Write section B"},
                            ],
                        },
                    )
                ],
            ),
            LLMResponse(
                content="finish parent",
                tool_calls=[ToolCall(id="p2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
        ]
    )

    sub_llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="sub done",
                tool_calls=[ToolCall(id="s1", name=TASK_FINISH_TOOL_NAME, arguments={"message": answer})],
            )
            for answer in ("sub-A", "sub-B")
        ]
    )
    provider = _shared_model_provider(parent_llm=parent_llm, child_llm=sub_llm)

    runtime = AgentRuntime(
        llm_client=_parent_client(provider),
        model_provider=provider,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        tool_registry_factory=build_default_registry,
        execution_backend=backend,
    )
    task = AgentTask(
        task_id="parent_batch_parallel",
        model="parent-model",
        system_prompt="sys",
        user_prompt="run parent batch task",
        max_cycles=4,
        sub_agents={
            "writer-sub": SubAgentConfig(
                model="kimi-k2.5",
                backend="moonshot",
                description="write sections",
            )
        },
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert backend.parallel_map_calls == 1
    assert provider.resolved_models == ["parent-model", "kimi-k2.5", "kimi-k2.5"]


def test_sub_task_metadata_contains_isolated_browser_scope(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    parent_task = AgentTask(
        task_id="parent",
        model="parent-model",
        system_prompt="sys",
        user_prompt="run parent task",
        max_cycles=4,
        metadata={"language": "zh-CN"},
    )
    sub_agent = SubAgentConfig(
        model="kimi-k2.5",
        backend="moonshot",
        description="collect facts",
    )
    request = SubTaskRequest(
        agent_name="research-sub",
        task_description="Collect one fact",
        metadata={
            "task_id": "user-overridden-task-id",
            "session_id": "user-overridden-session-id",
            "browser_scope_key": "user-overridden-browser-scope",
        },
    )

    sub_task = runtime._build_sub_agent_task(
        parent_task=parent_task,
        sub_task_id="sub-task-1",
        sub_session_id="sub-session-1",
        sub_agent_name="research-sub",
        sub_agent=sub_agent,
        resolved_model_id="kimi-k2.5",
        child_run_id="child-run",
        trace_id="child-trace",
        parent_run_id="",
        parent_tool_call_id="",
        request=request,
        parent_shared_state={},
        workspace_path=tmp_path,
    )

    assert sub_task.metadata["task_id"] == "sub-task-1"
    assert sub_task.metadata["session_id"] == "sub-session-1"
    assert sub_task.metadata["browser_scope_key"] == "sub-session-1"


def test_sub_task_metadata_inherits_sub_agent_prompt_cache_metadata(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    parent_task = AgentTask(
        task_id="parent",
        model="parent-model",
        system_prompt="sys",
        user_prompt="run parent task",
        max_cycles=4,
        metadata={"language": "zh-CN"},
    )
    sub_agent = SubAgentConfig(
        model="claude-sonnet-4-5-20250929",
        backend="anthropic",
        description="collect facts",
        metadata={
            "anthropic_prompt_cache_enabled": True,
            "system_prompt_sections": [
                {"id": "core_identity", "text": "stable section", "stable": True},
            ],
        },
    )

    sub_task = runtime._build_sub_agent_task(
        parent_task=parent_task,
        sub_task_id="sub-task-cache",
        sub_session_id="sub-session-cache",
        sub_agent_name="research-sub",
        sub_agent=sub_agent,
        resolved_model_id="claude-sonnet-4-5-20250929",
        child_run_id="child-run-cache",
        trace_id="child-trace",
        parent_run_id="",
        parent_tool_call_id="",
        request=SubTaskRequest(agent_name="research-sub", task_description="Collect one fact"),
        parent_shared_state={},
        workspace_path=tmp_path,
    )

    assert sub_task.metadata["anthropic_prompt_cache_enabled"] is True
    assert sub_task.metadata["system_prompt_sections"][0]["id"] == "core_identity"


def test_sub_task_metadata_generates_prompt_cache_sections_for_default_prompt(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    parent_task = AgentTask(
        task_id="parent",
        model="parent-model",
        system_prompt="sys",
        user_prompt="run parent task",
        max_cycles=4,
        metadata={"language": "zh-CN"},
    )
    sub_agent = SubAgentConfig(
        model="claude-sonnet-4-5-20250929",
        backend="anthropic",
        description="collect facts",
    )

    sub_task = runtime._build_sub_agent_task(
        parent_task=parent_task,
        sub_task_id="sub-task-cache-default",
        sub_session_id="sub-session-cache-default",
        sub_agent_name="research-sub",
        sub_agent=sub_agent,
        resolved_model_id="claude-sonnet-4-5-20250929",
        child_run_id="child-run-cache-default",
        trace_id="child-trace",
        parent_run_id="",
        parent_tool_call_id="",
        request=SubTaskRequest(agent_name="research-sub", task_description="Collect one fact"),
        parent_shared_state={},
        workspace_path=tmp_path,
    )

    assert sub_task.metadata["system_prompt_sections"][0]["id"] == "agent_definition"
    assert sub_task.metadata["system_prompt_sections"][-1]["id"] == "current_time"
    assert sub_task.metadata["system_prompt_sections"][-1]["stable"] is False


def test_sub_task_session_events_include_task_and_session_identifiers(tmp_path: Path) -> None:
    captured_events: list[RunEvent] = []

    parent_llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="delegate",
                tool_calls=[
                    ToolCall(
                        id="p1",
                        name=CREATE_SUB_TASK_TOOL_NAME,
                        arguments={
                            "agent_id": "research-sub",
                            "task_description": "Collect one fact",
                        },
                    )
                ],
            ),
            LLMResponse(
                content="finish parent",
                tool_calls=[ToolCall(id="p2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
        ]
    )

    sub_llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="sub finish",
                tool_calls=[ToolCall(id="s1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "sub done"})],
            )
        ]
    )
    provider = _shared_model_provider(parent_llm=parent_llm, child_llm=sub_llm)

    runtime = AgentRuntime(
        llm_client=_parent_client(provider),
        model_provider=provider,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        tool_registry_factory=build_default_registry,
        event_handler=captured_events.append,
    )
    task = AgentTask(
        task_id="parent_session_events",
        model="parent-model",
        system_prompt="sys",
        user_prompt="run parent task",
        max_cycles=4,
        sub_agents={
            "research-sub": SubAgentConfig(
                model="kimi-k2.5",
                backend="moonshot",
                description="collect facts",
            )
        },
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert provider.resolved_models == ["parent-model", "kimi-k2.5"]

    payload = json.loads(result.cycles[0].tool_results[0].content)
    task_id = payload["task_id"]
    session_id = payload["session_id"]
    assert task_id
    assert session_id == task_id

    child_lifecycle = [event for event in captured_events if isinstance(event, SubRunStartedEvent | SubRunCompletedEvent)]
    assert [event.type for event in child_lifecycle] == ["sub_run_started", "sub_run_completed"]
    assert all(event.task_id == task_id for event in child_lifecycle)
    assert all(event.session_id == session_id for event in child_lifecycle)


def test_sub_agent_stream_callback_forwards_event_objects(tmp_path: Path) -> None:
    contract = json.loads(
        (Path(__file__).parent / "fixtures" / "parity" / "configured_sub_agent.json").read_text(encoding="utf-8")
    )
    assert "event_sink" in contract["capability_projection"]["inherited"]
    assert contract["stream_forwarding"]["raw_callback"] is False
    parent_llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="delegate",
                tool_calls=[
                    ToolCall(
                        id="p1",
                        name=CREATE_SUB_TASK_TOOL_NAME,
                        arguments={"agent_id": "research-sub", "task_description": "Collect core facts"},
                    )
                ],
            ),
            LLMResponse(
                content="finish parent",
                tool_calls=[ToolCall(id="p2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "parent done"})],
            ),
        ]
    )

    class StreamingSubLLM:
        def complete(
            self,
            *,
            model,
            messages,
            tools,
            stream_callback=None,
            model_settings=None,
            request_metadata=None,
        ):
            del model, messages, tools, model_settings, request_metadata
            if stream_callback is not None:
                stream_callback(
                    {
                        "event": "assistant_delta",
                        "content_delta": "checking",
                        "task_id": "spoofed-task",
                        "session_id": "spoofed-session",
                        "sub_agent_name": "spoofed-agent",
                    }
                )
                stream_callback(
                    {
                        "event": "tool_call_started",
                        "tool_call_id": "sub-tool-1",
                        "tool_call_index": 0,
                        "function_name": "bash",
                        "arguments_chars": 0,
                        "estimated_tokens": 0,
                    }
                )
                stream_callback(
                    {
                        "event": "tool_call_progress",
                        "tool_call_id": "sub-tool-1",
                        "tool_call_index": 0,
                        "function_name": "bash",
                        "arguments_chars": 48,
                        "estimated_tokens": 12,
                    }
                )
            return LLMResponse(
                content="sub done",
                tool_calls=[ToolCall(id="s1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "sub done"})],
            )

    parent_events: list[RunEvent] = []
    provider = _shared_model_provider(parent_llm=parent_llm, child_llm=StreamingSubLLM())
    runtime = AgentRuntime(
        llm_client=_parent_client(provider),
        model_provider=provider,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        tool_registry_factory=build_default_registry,
    )
    task = AgentTask(
        task_id="parent_stream_events",
        model="parent-model",
        system_prompt="sys",
        user_prompt="run parent task",
        max_cycles=4,
        sub_agents={
            "research-sub": SubAgentConfig(
                model="kimi-k2.5",
                backend="moonshot",
                description="collect facts",
            )
        },
    )

    result = runtime.run(task, ctx=ExecutionContext(event_handler=parent_events.append))

    assert result.status == AgentStatus.COMPLETED
    assert provider.resolved_models == ["parent-model", "kimi-k2.5"]
    child_stream_events = [
        event
        for event in parent_events
        if isinstance(event, AssistantDeltaEvent | ModelToolCallStartedEvent | ModelToolCallProgressEvent)
    ]
    assert [event.type for event in child_stream_events] == [
        "assistant_delta",
        "model_tool_call_started",
        "model_tool_call_progress",
    ]
    started = next(event for event in parent_events if isinstance(event, SubRunStartedEvent))
    assert all(event.run_id == started.run_id for event in child_stream_events)
    assert all(event.session_id == started.session_id for event in child_stream_events)
    assert all(event.agent_name == "research-sub" for event in child_stream_events)
    delta = next(event for event in child_stream_events if isinstance(event, AssistantDeltaEvent))
    assert delta.delta == "checking"
    assert delta.metadata == {}
    progress = next(event for event in child_stream_events if isinstance(event, ModelToolCallProgressEvent))
    assert progress.tool_call_id == "sub-tool-1"
    assert progress.tool_name == "bash"
    assert progress.estimated_tokens == 12


def test_create_sub_task_reports_error_without_sub_agent_model_resolution(tmp_path: Path) -> None:
    parent_llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="delegate",
                tool_calls=[
                    ToolCall(
                        id="p1",
                        name=CREATE_SUB_TASK_TOOL_NAME,
                        arguments={"agent_id": "research-sub", "task_description": "Collect core facts"},
                    )
                ],
            )
        ]
    )
    runtime = AgentRuntime(
        llm_client=parent_llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="parent_err",
        model="parent-model",
        system_prompt="sys",
        user_prompt="run parent task",
        max_cycles=1,
        sub_agents={
            "research-sub": SubAgentConfig(
                model="kimi-k2.5",
                backend="moonshot",
                description="collect facts",
            )
        },
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.MAX_CYCLES
    tool_result = result.cycles[0].tool_results[0]
    assert tool_result.error_code == "sub_task_failed"
    payload = json.loads(tool_result.content)
    assert payload["error"] == "Sub-agent model resolution requires model_provider when backend is explicit."


def test_steer_sub_agent_session_targets_registered_session() -> None:
    class _DummySession:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def steer(self, prompt: str) -> None:
            self.messages.append(prompt)

    session_id = "sub-steer-test"
    dummy = _DummySession()
    _register_sub_agent_session(session_id, dummy)
    try:
        assert steer_sub_agent_session(session_id=session_id, prompt="focus github") is True
        assert dummy.messages == ["focus github"]
    finally:
        _unregister_sub_agent_session(session_id, dummy)

    assert steer_sub_agent_session(session_id=session_id, prompt="after cleanup") is False
