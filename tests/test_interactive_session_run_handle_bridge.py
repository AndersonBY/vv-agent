from __future__ import annotations

from typing import Any

from vv_agent import AgentSessionOptions, InteractiveAgentClient, InteractiveAgentDefinition
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse, Message, ToolCall


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="test-model",
        selected_model="test-model",
        model_id="test-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
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
    assert [payload["type"] for payload in typed_events if payload.get("type") in {"run_started", "run_completed"}] == [
        "run_started",
        "run_completed",
    ]


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
