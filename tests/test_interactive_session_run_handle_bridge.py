from __future__ import annotations

import json
from typing import Any

from vv_agent import AgentSessionOptions, InteractiveAgentClient, InteractiveAgentDefinition
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import CREATE_SUB_TASK_TOOL_NAME, TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.types import AgentStatus, LLMResponse, Message, SubAgentConfig, ToolCall


def _resolved(*, backend: str = "test", model: str = "test-model") -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend=backend,
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id=model)],
    )


def _llm_builder(*_: Any, **__: Any):
    return ScriptedLLM(steps=[LLMResponse(content="answer", tool_calls=[])]), _resolved()


def test_interactive_session_emits_v1_events_from_run_handle(tmp_path) -> None:
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="test",
            workspace=tmp_path,
            llm_builder=_llm_builder,
        )
    )
    session = client.create_session(
        session_id="session_1",
        agent=InteractiveAgentDefinition(description="assistant", model="test-model"),
    )
    emitted: list[dict[str, Any]] = []
    session.subscribe(lambda event, payload: emitted.append({"event": event, **payload}))

    session.prompt("hello", auto_follow_up=False)

    typed_events = [payload for payload in emitted if payload.get("version") == "v1"]
    lifecycle_events = [payload for payload in typed_events if payload.get("type") in {"run_started", "run_completed"}]
    assert [payload["type"] for payload in lifecycle_events] == ["run_started", "run_completed"]
    assert [payload.get("session_id") for payload in lifecycle_events] == ["session_1", "session_1"]


def test_interactive_session_steering_queue_reaches_run_handle_runtime(tmp_path) -> None:
    seen_user_messages: list[list[str]] = []

    def respond(_: str, messages: list[Message]) -> LLMResponse:
        seen_user_messages.append([message.content for message in messages if message.role == "user"])
        return LLMResponse(
            content="answer",
            tool_calls=[ToolCall(id="finish-1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
        )

    def llm_builder(*_: Any, **__: Any):
        return ScriptedLLM(steps=[respond]), _resolved()

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="test",
            workspace=tmp_path,
            llm_builder=llm_builder,
        )
    )
    session = client.create_session(
        session_id="session_1",
        agent=InteractiveAgentDefinition(description="assistant", model="test-model"),
    )

    session.steer("queued context")
    run = session.prompt("hello", auto_follow_up=False)

    assert run.result.final_answer == "done"
    assert seen_user_messages == [["hello", "queued context"]]


def test_interactive_sub_agent_uses_bridge_llm_builder_for_child_model(tmp_path) -> None:
    builder_calls: list[tuple[str, str]] = []

    def llm_builder(*_: Any, backend: str, model: str, **__: Any):
        builder_calls.append((backend, model))
        if model == "parent-model":
            return (
                ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="delegate",
                            tool_calls=[
                                ToolCall(
                                    id="parent-call-1",
                                    name=CREATE_SUB_TASK_TOOL_NAME,
                                    arguments={
                                        "agent_name": "child",
                                        "task_description": "Handle child work",
                                        "output_requirements": "Return done",
                                    },
                                )
                            ],
                        ),
                        LLMResponse(
                            content="finish parent",
                            tool_calls=[
                                ToolCall(id="parent-call-2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "parent done"})
                            ],
                        ),
                    ]
                ),
                _resolved(backend=backend, model=model),
            )
        if model == "child-model":
            return (
                ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="finish child",
                            tool_calls=[
                                ToolCall(id="child-call-1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "child done"})
                            ],
                        )
                    ]
                ),
                _resolved(backend=backend, model=model),
            )
        raise AssertionError(f"Unexpected model: {model}")

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            settings_file=tmp_path / "settings.py",
            default_backend="test",
            workspace=tmp_path,
            llm_builder=llm_builder,
        )
    )
    session = client.create_session(
        session_id="session_1",
        agent=InteractiveAgentDefinition(
            description="parent",
            model="parent-model",
            max_cycles=4,
            enable_sub_agents=True,
            sub_agents={
                "child": SubAgentConfig(
                    description="child worker",
                    model="child-model",
                    backend="test",
                )
            },
        ),
    )

    run = session.prompt("delegate", auto_follow_up=False)

    assert run.result.status == AgentStatus.COMPLETED
    assert run.result.final_answer == "parent done"
    assert builder_calls == [("test", "parent-model"), ("test", "child-model")]
    sub_task_payload = json.loads(run.result.cycles[0].tool_results[0].content)
    assert sub_task_payload["status"] == "completed"
    assert sub_task_payload["final_answer"] == "child done"
