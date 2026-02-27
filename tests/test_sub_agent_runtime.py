from __future__ import annotations

import json
from pathlib import Path

from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import BATCH_SUB_TASKS_TOOL_NAME, CREATE_SUB_TASK_TOOL_NAME, TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime import AgentRuntime
from vv_agent.runtime.backends.inline import InlineBackend
from vv_agent.runtime.engine import (
    _register_sub_agent_session,
    _unregister_sub_agent_session,
    steer_sub_agent_session,
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
                            "agent_name": "research-sub",
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

    builder_calls: list[tuple[str, str, str]] = []
    settings_file = tmp_path / "local_settings.py"
    settings_file.write_text("LLM_SETTINGS = {}", encoding="utf-8")

    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del timeout_seconds
        builder_calls.append((str(settings_path), backend, model))
        sub_llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="sub done",
                    tool_calls=[ToolCall(id="s1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "sub-result"})],
                )
            ]
        )
        return sub_llm, _fake_resolved(backend=backend, model=model)

    runtime = AgentRuntime(
        llm_client=parent_llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        settings_file=settings_file,
        default_backend="moonshot",
        llm_builder=fake_llm_builder,
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
    assert len(builder_calls) == 1
    assert builder_calls[0] == (str(settings_file.resolve()), "moonshot", "kimi-k2.5")

    first_tool_payload = json.loads(result.cycles[0].tool_results[0].content)
    assert first_tool_payload["status"] == "completed"
    assert first_tool_payload["final_answer"] == "sub-result"
    assert first_tool_payload["resolved"]["backend"] == "moonshot"


def test_batch_sub_tasks_aggregates_sub_agent_results(tmp_path: Path) -> None:
    parent_llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="batch delegate",
                tool_calls=[
                    ToolCall(
                        id="p1",
                        name=BATCH_SUB_TASKS_TOOL_NAME,
                        arguments={
                            "agent_name": "writer-sub",
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

    settings_file = tmp_path / "local_settings.py"
    settings_file.write_text("LLM_SETTINGS = {}", encoding="utf-8")
    sub_answers = iter(["sub-A", "sub-B"])
    builder_calls: list[int] = []

    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        builder_calls.append(1)
        sub_llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="sub done",
                    tool_calls=[
                        ToolCall(
                            id="s1",
                            name=TASK_FINISH_TOOL_NAME,
                            arguments={"message": next(sub_answers)},
                        )
                    ],
                )
            ]
        )
        return sub_llm, _fake_resolved(backend=backend, model=model)

    runtime = AgentRuntime(
        llm_client=parent_llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        settings_file=settings_file,
        default_backend="moonshot",
        llm_builder=fake_llm_builder,
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
    assert len(builder_calls) == 2

    batch_payload = json.loads(result.cycles[0].tool_results[0].content)
    assert batch_payload["summary"] == {"total": 2, "completed": 2, "failed": 0}
    assert batch_payload["results"][0]["final_answer"] == "sub-A"
    assert batch_payload["results"][1]["final_answer"] == "sub-B"


def test_batch_sub_tasks_uses_execution_backend_parallel_map(tmp_path: Path) -> None:
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
                        name=BATCH_SUB_TASKS_TOOL_NAME,
                        arguments={
                            "agent_name": "writer-sub",
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

    settings_file = tmp_path / "local_settings.py"
    settings_file.write_text("LLM_SETTINGS = {}", encoding="utf-8")
    sub_answers = iter(["sub-A", "sub-B"])

    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        sub_llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="sub done",
                    tool_calls=[
                        ToolCall(
                            id="s1",
                            name=TASK_FINISH_TOOL_NAME,
                            arguments={"message": next(sub_answers)},
                        )
                    ],
                )
            ]
        )
        return sub_llm, _fake_resolved(backend=backend, model=model)

    runtime = AgentRuntime(
        llm_client=parent_llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        settings_file=settings_file,
        default_backend="moonshot",
        llm_builder=fake_llm_builder,
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
        request=request,
        parent_shared_state={},
        workspace_path=tmp_path,
    )

    assert sub_task.metadata["task_id"] == "sub-task-1"
    assert sub_task.metadata["session_id"] == "sub-session-1"
    assert sub_task.metadata["browser_scope_key"] == "sub-session-1"


def test_sub_task_session_events_include_task_and_session_identifiers(tmp_path: Path) -> None:
    captured_events: list[tuple[str, dict[str, object]]] = []

    parent_llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="delegate",
                tool_calls=[
                    ToolCall(
                        id="p1",
                        name=CREATE_SUB_TASK_TOOL_NAME,
                        arguments={
                            "agent_name": "research-sub",
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

    settings_file = tmp_path / "local_settings.py"
    settings_file.write_text("LLM_SETTINGS = {}", encoding="utf-8")

    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        sub_llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="sub finish",
                    tool_calls=[ToolCall(id="s1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "sub done"})],
                )
            ]
        )
        return sub_llm, _fake_resolved(backend=backend, model=model)

    runtime = AgentRuntime(
        llm_client=parent_llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        settings_file=settings_file,
        default_backend="moonshot",
        llm_builder=fake_llm_builder,
        tool_registry_factory=build_default_registry,
        log_handler=lambda event, payload: captured_events.append((event, dict(payload))),
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

    payload = json.loads(result.cycles[0].tool_results[0].content)
    task_id = payload["task_id"]
    session_id = payload["session_id"]
    assert task_id
    assert session_id == task_id

    session_created = [item for item in captured_events if item[0] == "sub_agent_session_created"]
    assert session_created
    assert session_created[0][1]["task_id"] == task_id
    assert session_created[0][1]["session_id"] == session_id

    cycle_events = [event for event in captured_events if event[0] == "sub_agent_cycle_started"]
    assert cycle_events
    assert all(event_payload.get("task_id") == task_id for _, event_payload in cycle_events)
    assert all(event_payload.get("session_id") == session_id for _, event_payload in cycle_events)

    session_run_events = [event for event in captured_events if event[0] == "sub_agent_session_run_start"]
    assert session_run_events
    assert session_run_events[0][1]["task_id"] == task_id
    assert session_run_events[0][1]["session_id"] == session_id


def test_create_sub_task_reports_error_without_sub_agent_model_resolution(tmp_path: Path) -> None:
    parent_llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="delegate",
                tool_calls=[
                    ToolCall(
                        id="p1",
                        name=CREATE_SUB_TASK_TOOL_NAME,
                        arguments={"agent_name": "research-sub", "task_description": "Collect core facts"},
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
    assert "requires runtime settings_file" in payload["error"]


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
