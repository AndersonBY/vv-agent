from __future__ import annotations

import json
from pathlib import Path

import pytest
from support import ModelMapProvider
from support.kernel_runtime import KernelRuntime as AgentRuntime

from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import CREATE_SUB_TASK_TOOL_NAME
from vv_agent.events import (
    AssistantDeltaEvent,
    RunEvent,
    SubRunCompletedEvent,
    SubRunStartedEvent,
)
from vv_agent.llm import LLMClient, ScriptedLLM
from vv_agent.model import ModelRef
from vv_agent.prompt import build_raw_system_prompt_bundle
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
            LLMResponse(content="parent done"),
        ]
    )

    sub_llm = ScriptedLLM(steps=[LLMResponse(content="sub-result")])
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
        prompt_bundle=build_raw_system_prompt_bundle("sys"),
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
    assert set(provider.resolved_models) == {"parent-model", "kimi-k2.5"}

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
            LLMResponse(content="batch done"),
        ]
    )

    sub_llm = ScriptedLLM(steps=[LLMResponse(content=answer) for answer in ("sub-A", "sub-B")])
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
        prompt_bundle=build_raw_system_prompt_bundle("sys"),
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
    assert set(provider.resolved_models) == {"parent-model", "kimi-k2.5"}

    batch_payload = json.loads(result.cycles[0].tool_results[0].content)
    assert batch_payload["summary"] == {"total": 2, "completed": 2, "failed": 0}
    assert batch_payload["results"][0]["final_answer"] == "sub-A"
    assert batch_payload["results"][1]["final_answer"] == "sub-B"


def test_sub_task_metadata_contains_isolated_browser_scope(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    parent_task = AgentTask(
        task_id="parent",
        model="parent-model",
        prompt_bundle=build_raw_system_prompt_bundle("sys"),
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


def test_sub_task_uses_prompt_bundle_instead_of_prompt_section_metadata(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    parent_task = AgentTask(
        task_id="parent",
        model="parent-model",
        prompt_bundle=build_raw_system_prompt_bundle("sys"),
        user_prompt="run parent task",
        max_cycles=4,
        metadata={"language": "zh-CN"},
    )
    sub_agent = SubAgentConfig(
        model="claude-sonnet-4-5-20250929",
        backend="anthropic",
        description="collect facts",
        metadata={"anthropic_prompt_cache_enabled": True},
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
    assert "system_prompt_sections" not in sub_task.metadata
    assert [section.id for section in sub_task.prompt_bundle.sections] == ["agent_definition", "tools", "current_time"]


def test_sub_task_metadata_generates_prompt_cache_sections_for_default_prompt(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    parent_task = AgentTask(
        task_id="parent",
        model="parent-model",
        prompt_bundle=build_raw_system_prompt_bundle("sys"),
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

    assert "system_prompt_sections" not in sub_task.metadata
    assert sub_task.prompt_bundle.sections[0].id == "agent_definition"
    assert sub_task.prompt_bundle.sections[-1].id == "current_time"
    assert sub_task.prompt_bundle.sections[-1].stable is False


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
            LLMResponse(content="done"),
        ]
    )

    sub_llm = ScriptedLLM(steps=[LLMResponse(content="sub done")])
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
        prompt_bundle=build_raw_system_prompt_bundle("sys"),
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
    assert set(provider.resolved_models) == {"parent-model", "kimi-k2.5"}

    payload = json.loads(result.cycles[0].tool_results[0].content)
    task_id = payload["task_id"]
    session_id = payload["session_id"]
    assert task_id
    assert session_id == task_id

    child_lifecycle = [event for event in captured_events if isinstance(event, SubRunStartedEvent | SubRunCompletedEvent)]
    assert [event.type for event in child_lifecycle] == ["sub_run_started", "sub_run_completed"]
    assert all(event.child_session_id == task_id for event in child_lifecycle)
    assert all(event.child_session_id == session_id for event in child_lifecycle)


def test_sub_agent_stream_callback_retains_child_identity(tmp_path):
    from support.kernel_runtime import start_runner

    from vv_agent import Agent, RunConfig
    from vv_agent.session.surfaces import SessionDriver

    class StreamingSubLLM:
        def complete(self, request):
            return LLMResponse("child done")

        def complete_with_stream(self, request, stream_callback=None):
            assert stream_callback is not None
            stream_callback({"event": "assistant_delta", "content_delta": "checking", "session_id": "spoofed"})
            return self.complete(request)

    parent = ScriptedLLM(
        [
            LLMResponse(
                "",
                [
                    ToolCall(
                        "delegate",
                        "create_sub_task",
                        {"agent_id": "worker", "task_description": "work", "wait_for_completion": False},
                    )
                ],
            ),
            LLMResponse("parent done"),
        ]
    )
    provider = _shared_model_provider(parent_llm=parent, child_llm=StreamingSubLLM())
    driver = SessionDriver()
    try:
        handle = start_runner(
            driver,
            "parent",
            Agent(
                "parent",
                "Delegate.",
                model="parent-model",
                sub_agents={"worker": SubAgentConfig(model="kimi-k2.5", backend="moonshot", description="Work.")},
            ),
            "go",
            run_config=RunConfig(model_provider=provider, workspace=tmp_path),
        )
        assert handle.result().final_output == "parent done"
        child = next(h for h in driver.handles if h.session_id != "parent")
        child.join(3)
        deltas = [e for e in child.events() if isinstance(e, AssistantDeltaEvent)]
        assert len(deltas) == 1 and deltas[0].delta == "checking"
        assert deltas[0].session_id == child.session_id and deltas[0].run_id == child.run_id
        assert deltas[0].agent_name == "worker"
    finally:
        driver.close()


def test_sub_agent_admission_rejects_unresolved_model_before_child_creation(tmp_path):
    from support.kernel_runtime import start_runner

    from vv_agent import Agent, RunConfig, ScriptedModelProvider
    from vv_agent.model import ModelError
    from vv_agent.session.surfaces import SessionDriver

    driver = SessionDriver()
    try:
        handle = start_runner(
            driver,
            "parent",
            Agent(
                "parent",
                "Delegate.",
                model="m",
                sub_agents={"worker": SubAgentConfig(model="child", backend="unavailable", description="Work.")},
            ),
            "go",
            run_config=RunConfig(
                workspace=tmp_path,
                model_provider=ScriptedModelProvider.from_steps(
                    "test",
                    "m",
                    [
                        LLMResponse(
                            "", [ToolCall("delegate", "create_sub_task", {"agent_id": "worker", "task_description": "work"})]
                        )
                    ],
                ),
            ),
        )
        with pytest.raises(ModelError, match="backend mismatch"):
            handle.result()
        assert driver.store.list_sessions() == ("parent",)
        assert not any(r.record.kind == "op_parked" for r in driver.store.read_state("parent")[1])
    finally:
        driver.close()
