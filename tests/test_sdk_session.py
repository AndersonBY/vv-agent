from __future__ import annotations

from pathlib import Path

import pytest

from v_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from v_agent.constants import ASK_USER_TOOL_NAME, TASK_FINISH_TOOL_NAME, TODO_WRITE_TOOL_NAME
from v_agent.llm import ScriptedLLM
from v_agent.sdk import AgentDefinition, AgentSDKClient, AgentSDKOptions
from v_agent.tools import build_default_registry
from v_agent.types import AgentStatus, LLMResponse, ToolCall


def _fake_resolved(*, backend: str, model: str) -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    option = EndpointOption(endpoint=endpoint, model_id=model)
    return ResolvedModelConfig(
        backend=backend,
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[option],
    )


def test_session_prompt_supports_follow_up_queue(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="finish-1",
                tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "first"})],
            ),
            LLMResponse(
                content="finish-2",
                tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "second"})],
            ),
        ]
    )

    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
        agent=AgentDefinition(description="helper", model="kimi-k2.5"),
    )

    session = client.create_session()
    session.follow_up("after first run")
    run = session.prompt("first run")

    assert run.result.status == AgentStatus.COMPLETED
    assert run.result.final_answer == "second"
    assert session.latest_run is not None
    assert session.latest_run.result.final_answer == "second"


def test_session_can_queue_steer_from_runtime_event(tmp_path: Path) -> None:
    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="two tool calls",
                    tool_calls=[
                        ToolCall(
                            id="a1",
                            name=TODO_WRITE_TOOL_NAME,
                            arguments={"todos": [{"title": "x", "status": "completed", "priority": "medium"}]},
                        ),
                        ToolCall(id="a2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "should be skipped"}),
                    ],
                ),
                LLMResponse(
                    content="final",
                    tool_calls=[ToolCall(id="a3", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
                ),
            ]
        )
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
        agent=AgentDefinition(description="helper", model="kimi-k2.5"),
    )
    session = client.create_session()
    triggered = {"done": False}

    def on_event(event: str, payload: dict[str, object]) -> None:
        if event == "tool_result" and payload.get("tool_name") == TODO_WRITE_TOOL_NAME and not triggered["done"]:
            triggered["done"] = True
            session.steer("switch strategy")

    session.subscribe(on_event)
    run = session.prompt("start")

    assert run.result.status == AgentStatus.COMPLETED
    assert run.result.final_answer == "done"
    assert run.result.cycles[0].tool_results[1].error_code == "skipped_due_to_steering"


def test_session_continue_after_wait_user_with_multiple_tool_calls(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="need user input",
                tool_calls=[
                    ToolCall(id="u1", name=ASK_USER_TOOL_NAME, arguments={"question": "pick style"}),
                    ToolCall(id="u2", name=ASK_USER_TOOL_NAME, arguments={"question": "pick output file"}),
                ],
            ),
            LLMResponse(
                content="finish",
                tool_calls=[ToolCall(id="u3", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
        ]
    )

    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
        agent=AgentDefinition(description="helper", model="kimi-k2.5"),
    )
    session = client.create_session()

    first = session.prompt("start", auto_follow_up=False)
    assert first.result.status == AgentStatus.WAIT_USER
    assert len(first.result.cycles[0].tool_results) == 2
    assert first.result.cycles[0].tool_results[1].error_code == "skipped_due_to_wait_user"

    second = session.continue_run("formal style, write to artifacts/result.md")
    assert second.result.status == AgentStatus.COMPLETED
    assert second.result.final_answer == "done"


def test_session_query_raises_when_not_completed(tmp_path: Path) -> None:
    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="ask user",
                    tool_calls=[ToolCall(id="c1", name=ASK_USER_TOOL_NAME, arguments={"question": "pick one"})],
                )
            ]
        )
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
        agent=AgentDefinition(description="helper", model="kimi-k2.5"),
    )
    session = client.create_session()

    with pytest.raises(RuntimeError, match="status=wait_user"):
        session.query("ask")


def test_session_emits_session_and_runtime_events(tmp_path: Path) -> None:
    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="finish",
                    tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
                )
            ]
        )
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
        agent=AgentDefinition(description="helper", model="kimi-k2.5"),
    )
    session = client.create_session()
    events: list[str] = []
    session.subscribe(lambda event, payload: (events.append(event), payload))

    run = session.prompt("run")
    assert run.result.status == AgentStatus.COMPLETED
    assert "session_run_start" in events
    assert "run_started" in events
    assert "run_completed" in events
    assert "session_run_end" in events
