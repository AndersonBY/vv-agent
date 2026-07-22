from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any

from support import FixedModelProvider, ModelMapProvider

from vv_agent import AgentSessionOptions, InteractiveAgentClient, InteractiveAgentDefinition
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import CREATE_SUB_TASK_TOOL_NAME, TASK_FINISH_TOOL_NAME
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.runtime import CancellationToken
from vv_agent.types import AgentStatus, LLMResponse, SubAgentConfig, ToolCall


def _resolved(*, backend: str = "test", model: str = "test-model") -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend=backend,
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id=model)],
    )


def _model_provider() -> FixedModelProvider:
    return FixedModelProvider(
        ScriptedLLM(
            steps=[
                LLMResponse(
                    content="answer",
                    tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "answer"})],
                )
            ]
        ),
        _resolved(),
    )


def test_interactive_session_emits_current_events_from_run_handle(tmp_path) -> None:
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=_model_provider(),
            workspace=tmp_path,
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

    def respond(request: LlmRequest) -> LLMResponse:
        seen_user_messages.append([message.content for message in request.messages if message.role == "user"])
        return LLMResponse(
            content="answer",
            tool_calls=[ToolCall(id="finish-1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
        )

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=FixedModelProvider(ScriptedLLM(steps=[respond]), _resolved()),
            workspace=tmp_path,
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


def test_interactive_session_derives_each_run_from_host_cancellation_token(tmp_path) -> None:
    parent_token = CancellationToken()
    first_step_ready = threading.Event()
    release_first_step = threading.Event()

    def respond(_: LlmRequest) -> LLMResponse:
        first_step_ready.set()
        assert release_first_step.wait(timeout=3)
        return LLMResponse(content="continue", tool_calls=[])

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=FixedModelProvider(ScriptedLLM(steps=[respond]), _resolved()),
            workspace=tmp_path,
            cancellation_token=parent_token,
        )
    )
    session = client.create_session(agent=InteractiveAgentDefinition(description="assistant", model="test-model", max_cycles=2))
    result_queue: queue.Queue[Any] = queue.Queue()

    def run_prompt() -> None:
        try:
            result_queue.put(session.prompt("hello", auto_follow_up=False))
        except BaseException as exc:
            result_queue.put(exc)

    worker = threading.Thread(target=run_prompt, daemon=True)
    worker.start()
    assert first_step_ready.wait(timeout=3)

    parent_token.cancel("host shutdown")
    release_first_step.set()
    worker.join(timeout=3)

    assert not worker.is_alive()
    outcome = result_queue.get_nowait()
    assert not isinstance(outcome, BaseException)
    assert outcome.result.status == AgentStatus.FAILED
    assert "host shutdown" in (outcome.result.error or "")
    assert [event.type for event in outcome.events if event.type == "run_cancelled"] == ["run_cancelled"]


def test_active_run_handle_steer_queues_session_context(tmp_path) -> None:
    first_step_ready = threading.Event()
    first_step_can_finish = threading.Event()
    seen_user_messages: list[list[str]] = []
    steps: list[Any] = []

    def first_step(_: LlmRequest) -> LLMResponse:
        first_step_ready.set()
        assert first_step_can_finish.wait(timeout=3)
        return LLMResponse(content="continue", tool_calls=[])

    def second_step(request: LlmRequest) -> LLMResponse:
        seen_user_messages.append([message.content for message in request.messages if message.role == "user"])
        return LLMResponse(
            content="finish",
            tool_calls=[ToolCall(id="finish-1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
        )

    steps.extend([first_step, second_step])

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=FixedModelProvider(ScriptedLLM(steps=steps), _resolved()),
            workspace=tmp_path,
        )
    )
    session = client.create_session(
        session_id="session_1",
        agent=InteractiveAgentDefinition(description="assistant", model="test-model", max_cycles=3),
    )
    result_queue: queue.Queue[Any] = queue.Queue()
    worker = threading.Thread(
        target=lambda: result_queue.put(session.prompt("hello", auto_follow_up=False)),
        daemon=True,
    )
    worker.start()
    assert first_step_ready.wait(timeout=3)

    handle = _wait_for_active_handle(session)
    handle.steer("queued from handle")
    handle.steer("second queued from handle")
    first_step_can_finish.set()
    worker.join(timeout=3)

    assert not worker.is_alive()
    assert result_queue.get_nowait().result.final_answer == "done"
    assert seen_user_messages == [
        [
            "hello",
            "No tool call was produced. Continue the task and call `task_finish` when all todo items are done.",
            "queued from handle",
            "second queued from handle",
        ]
    ]


def test_active_run_handle_follow_up_queues_next_session_turn(tmp_path) -> None:
    first_step_ready = threading.Event()
    first_step_can_finish = threading.Event()
    seen_user_messages: list[list[str]] = []
    steps: list[Any] = []

    def first_step(_: LlmRequest) -> LLMResponse:
        first_step_ready.set()
        assert first_step_can_finish.wait(timeout=3)
        return LLMResponse(
            content="finish first",
            tool_calls=[ToolCall(id="finish-1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "first done"})],
        )

    def second_step(request: LlmRequest) -> LLMResponse:
        seen_user_messages.append([message.content for message in request.messages if message.role == "user"])
        return LLMResponse(
            content="finish second",
            tool_calls=[ToolCall(id="finish-2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "second done"})],
        )

    steps.extend([first_step, second_step])

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=FixedModelProvider(ScriptedLLM(steps=steps), _resolved()),
            workspace=tmp_path,
        )
    )
    session = client.create_session(
        session_id="session_1",
        agent=InteractiveAgentDefinition(description="assistant", model="test-model", max_cycles=2),
    )
    result_queue: queue.Queue[Any] = queue.Queue()
    worker = threading.Thread(target=lambda: result_queue.put(session.prompt("hello")), daemon=True)
    worker.start()
    assert first_step_ready.wait(timeout=3)

    handle = _wait_for_active_handle(session)
    handle.follow_up("continue from handle")
    first_step_can_finish.set()
    worker.join(timeout=3)

    assert not worker.is_alive()
    assert result_queue.get_nowait().result.final_answer == "second done"
    assert seen_user_messages == [["hello", "continue from handle"]]


def _wait_for_active_handle(session) -> Any:
    deadline = time.time() + 3
    while time.time() < deadline:
        handle = session.active_run_handle
        if handle is not None:
            return handle
        time.sleep(0.01)
    raise AssertionError("session did not expose an active run handle")


def test_interactive_sub_agent_uses_model_provider_for_child_model(tmp_path) -> None:
    parent_resolved = _resolved(model="parent-model")
    child_resolved = _resolved(model="child-model")
    model_provider = ModelMapProvider(
        routes={
            "parent-model": (
                ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="delegate",
                            tool_calls=[
                                ToolCall(
                                    id="parent-call-1",
                                    name=CREATE_SUB_TASK_TOOL_NAME,
                                    arguments={
                                        "agent_id": "child",
                                        "task_description": "Handle child work",
                                        "output_requirements": "Return done",
                                    },
                                )
                            ],
                        ),
                        LLMResponse(
                            content="finish parent",
                            tool_calls=[
                                ToolCall(
                                    id="parent-call-2",
                                    name=TASK_FINISH_TOOL_NAME,
                                    arguments={"message": "parent done"},
                                )
                            ],
                        ),
                    ]
                ),
                parent_resolved,
            ),
            "child-model": (
                ScriptedLLM(
                    steps=[
                        LLMResponse(
                            content="finish child",
                            tool_calls=[
                                ToolCall(
                                    id="child-call-1",
                                    name=TASK_FINISH_TOOL_NAME,
                                    arguments={"message": "child done"},
                                )
                            ],
                        )
                    ]
                ),
                child_resolved,
            ),
        },
        default_model="parent-model",
    )

    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=model_provider,
            workspace=tmp_path,
        )
    )
    session = client.create_session(
        session_id="session_1",
        agent=InteractiveAgentDefinition(
            description="parent",
            model="parent-model",
            max_cycles=4,
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
    assert model_provider.resolved_models[0] == "parent-model"
    assert model_provider.resolved_models[-1] == "child-model"
    sub_task_payload = json.loads(run.result.cycles[0].tool_results[0].content)
    assert sub_task_payload["status"] == "completed"
    assert sub_task_payload["final_answer"] == "child done"
