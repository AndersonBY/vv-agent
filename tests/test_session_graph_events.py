from __future__ import annotations

from pathlib import Path

from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import CREATE_SUB_TASK_TOOL_NAME, TASK_FINISH_TOOL_NAME
from vv_agent.events import (
    HandoffCompletedEvent,
    HandoffStartedEvent,
    RunEvent,
    SubRunCompletedEvent,
    SubRunStartedEvent,
    event_from_dict,
)
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime import AgentRuntime
from vv_agent.runtime.context import ExecutionContext
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentStatus, AgentTask, LLMResponse, SubAgentConfig, ToolCall


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


def test_sub_run_event_carries_parent_lineage() -> None:
    event = SubRunStartedEvent(
        run_id="run_child",
        trace_id="trace_1",
        session_id="session_child",
        parent_run_id="run_parent",
        parent_tool_call_id="call_create_sub_task",
        agent_name="researcher",
    )

    payload = event.to_dict()

    assert payload["type"] == "sub_run_started"
    assert payload["parent_run_id"] == "run_parent"
    assert payload["parent_tool_call_id"] == "call_create_sub_task"
    assert payload["session_id"] == "session_child"


def test_session_graph_events_round_trip_from_dict() -> None:
    events = [
        SubRunStartedEvent(
            run_id="run_child",
            trace_id="trace_1",
            session_id="session_child",
            child_session_id="session_child",
            parent_run_id="run_parent",
            parent_tool_call_id="call_create_sub_task",
            agent_name="researcher",
            task_id="task_child",
            metadata={"kind": "sub"},
        ),
        SubRunCompletedEvent(
            run_id="run_child",
            trace_id="trace_1",
            session_id="session_child",
            child_session_id="session_child",
            parent_run_id="run_parent",
            parent_tool_call_id="call_create_sub_task",
            agent_name="researcher",
            task_id="task_child",
            status="completed",
            final_output="done",
            token_usage={"total_tokens": 12},
        ),
        HandoffStartedEvent(
            run_id="run_parent",
            trace_id="trace_1",
            source_agent="planner",
            target_agent="researcher",
            tool_call_id="call_handoff",
            child_session_id="session_child",
        ),
        HandoffCompletedEvent(
            run_id="run_parent",
            trace_id="trace_1",
            source_agent="planner",
            target_agent="researcher",
            tool_call_id="call_handoff",
            child_session_id="session_child",
            status="completed",
        ),
    ]

    for event in events:
        replayed = event_from_dict(event.to_dict())
        assert type(replayed) is type(event)
        assert replayed.to_dict() == event.to_dict()


def test_create_sub_task_emits_sub_run_events_with_parent_tool_call_lineage(tmp_path: Path) -> None:
    parent_llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="delegate",
                tool_calls=[
                    ToolCall(
                        id="call_create_sub_task",
                        name=CREATE_SUB_TASK_TOOL_NAME,
                        arguments={
                            "agent_name": "research-sub",
                            "task_description": "Collect core facts",
                        },
                    )
                ],
            ),
            LLMResponse(
                content="finish parent",
                tool_calls=[
                    ToolCall(id="call_finish_parent", name=TASK_FINISH_TOOL_NAME, arguments={"message": "parent done"})
                ],
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
                    content="sub done",
                    tool_calls=[ToolCall(id="sub_finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "sub done"})],
                )
            ]
        )
        return sub_llm, _fake_resolved(backend=backend, model=model)

    emitted: list[RunEvent] = []
    ctx = ExecutionContext(
        metadata={
            "_vv_agent_agent_name": "planner",
            "_vv_agent_emit_event": emitted.append,
            "_vv_agent_run_id": "run_parent",
            "_vv_agent_trace_id": "trace_1",
            "_vv_agent_session_id": "session_parent",
        }
    )
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

    result = runtime.run(task, ctx=ctx)

    assert result.status == AgentStatus.COMPLETED
    started_events = [event for event in emitted if isinstance(event, SubRunStartedEvent)]
    completed_events = [event for event in emitted if isinstance(event, SubRunCompletedEvent)]
    assert len(started_events) == 1
    assert len(completed_events) == 1

    started = started_events[0]
    completed = completed_events[0]
    assert started.parent_run_id == "run_parent"
    assert started.parent_tool_call_id == "call_create_sub_task"
    assert started.agent_name == "research-sub"
    assert started.session_id
    assert started.child_session_id == started.session_id
    assert completed.run_id == started.run_id
    assert completed.session_id == started.session_id
    assert completed.child_session_id == started.session_id
    assert completed.parent_run_id == "run_parent"
    assert completed.parent_tool_call_id == "call_create_sub_task"
    assert completed.status == AgentStatus.COMPLETED.value
    assert completed.final_output == "sub done"
