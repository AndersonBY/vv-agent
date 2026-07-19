from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from vv_agent import (
    Agent,
    AgentStartedEvent,
    AssistantDeltaEvent,
    CycleStartedEvent,
    LLMStartedEvent,
    MemorySession,
    ModelSettings,
    RunCompletedEvent,
    RunConfig,
    Runner,
    RunResult,
)
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import READ_IMAGE_TOOL_NAME, TASK_FINISH_TOOL_NAME
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.tools import ToolExposure, function_tool
from vv_agent.types import AgentStatus, LLMResponse, Message, NoToolPolicy, ToolCall

ASSISTANT_REASONING_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "parity" / "assistant_reasoning_history_v1.json"
)


def _assistant_reasoning_contract() -> dict[str, object]:
    return json.loads(ASSISTANT_REASONING_FIXTURE_PATH.read_text(encoding="utf-8"))


def _assistant_reasoning_case(name: str) -> dict[str, object]:
    contract = _assistant_reasoning_contract()
    cases = cast(list[dict[str, object]], contract["cases"])
    return next(case for case in cases if case["name"] == name)


def _fake_resolved(
    *,
    backend: str = "test",
    model: str = "m",
    context_length: int | None = None,
    max_output_tokens: int | None = None,
    native_multimodal: bool = False,
) -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    option = EndpointOption(endpoint=endpoint, model_id=model)
    return ResolvedModelConfig(
        backend=backend,
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[option],
        context_length=context_length,
        max_output_tokens=max_output_tokens,
        native_multimodal=native_multimodal,
    )


def test_runner_run_sync_executes_agent_with_model_provider(tmp_path: Path) -> None:
    seen_settings: list[ModelSettings] = []

    def model_provider(agent: Agent, run_config: RunConfig):
        assert agent.name == "assistant"
        assert run_config.workspace == tmp_path
        seen_settings.append(agent.model_settings.resolve(run_config.model_settings))
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="done",
                        tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
                    )
                ]
            ),
            _fake_resolved(model="override-model"),
        )

    result = Runner.run_sync(
        Agent(
            name="assistant",
            instructions="Be concise.",
            model="base-model",
            model_settings=ModelSettings(temperature=0.1, max_tokens=100),
        ),
        "Say ok.",
        run_config=RunConfig(
            model="override-model",
            model_settings=ModelSettings(max_tokens=200),
            workspace=tmp_path,
            model_provider=model_provider,
        ),
    )

    assert isinstance(result, RunResult)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_output == "ok"
    assert result.input == "Say ok."
    assert result.raw_result.final_answer == "ok"
    assert seen_settings == [ModelSettings(temperature=0.1, max_tokens=200)]


def test_runner_preserves_reasoning_only_history_for_next_model_request(tmp_path: Path) -> None:
    contract = _assistant_reasoning_contract()
    runtime_case = cast(dict[str, object], contract["runtime_case"])
    first_response = cast(dict[str, object], runtime_case["first_response"])
    expected = cast(dict[str, object], runtime_case["expected"])
    reasoning_case = _assistant_reasoning_case("reasoning_only_assistant_is_preserved")
    reasoning_expected = cast(dict[str, object], reasoning_case["expected"])
    reasoning = cast(str, first_response["reasoning_content"])
    captured_requests: list[list[Message]] = []

    def capture_second_request(request: LlmRequest) -> LLMResponse:
        captured_requests.append(list(request.messages))
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="finish-reasoning", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})
            ],
        )

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content=cast(str, first_response["content"]),
                raw={
                    "reasoning_content": reasoning,
                    "usage": {
                        "completion_tokens": 7,
                        "total_tokens": 7,
                        "completion_tokens_details": {"reasoning_tokens": 7},
                    },
                },
            ),
            capture_second_request,
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return llm, _fake_resolved()

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Preserve private reasoning history.", model="m"),
        "continue the task",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            max_cycles=2,
            no_tool_policy=cast(NoToolPolicy, runtime_case["no_tool_policy"]),
        ),
    )

    assert result.status == AgentStatus.COMPLETED
    assert expected["next_model_request_contains_reasoning_turn"] is True
    assert len(captured_requests) == 1
    replayed = next(
        message
        for message in captured_requests[0]
        if message.role == "assistant" and message.reasoning_content == reasoning
    )
    assert replayed.content == expected["next_model_request_visible_content"]
    assert replayed.content == reasoning_expected["visible_content"]
    assert replayed.reasoning_content == reasoning_expected["reasoning_content"]
    assert replayed.to_openai_message() == reasoning_expected["openai_compatible_projection"]
    assert result.raw_result.cycles[0].token_usage.reasoning_tokens == 7


def test_runner_removes_fully_empty_assistant_before_next_model_request(tmp_path: Path) -> None:
    empty_case = _assistant_reasoning_case("fully_empty_assistant_is_removed")
    message = cast(dict[str, object], empty_case["message"])
    expected = cast(dict[str, object], empty_case["expected"])
    captured_requests: list[list[Message]] = []

    def capture_second_request(request: LlmRequest) -> LLMResponse:
        captured_requests.append(list(request.messages))
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(id="finish-empty", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
        )

    llm = ScriptedLLM(
        steps=[
            LLMResponse(content=cast(str, message["content"])),
            capture_second_request,
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return llm, _fake_resolved()

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Drop invalid empty history.", model="m"),
        "continue the task",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            max_cycles=2,
            no_tool_policy="continue",
        ),
    )

    assert result.status == AgentStatus.COMPLETED
    assert expected["retain_in_runtime_history"] is False
    assert len(captured_requests) == 1
    assert all(message.role != "assistant" for message in captured_requests[0])


def test_runner_passes_resolved_model_settings_to_llm_complete(tmp_path: Path) -> None:
    seen_settings: list[ModelSettings | None] = []

    class CapturingLLM:
        def complete(
            self,
            *,
            model: str,
            messages: list[Message],
            tools: list[dict[str, object]],
            stream_callback=None,
            model_settings: ModelSettings | None = None,
            request_metadata=None,
        ) -> LLMResponse:
            del model, messages, tools, stream_callback, request_metadata
            seen_settings.append(model_settings)
            return LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
            )

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return CapturingLLM(), _fake_resolved()

    Runner.run_sync(
        Agent(
            name="assistant",
            instructions="Use resolved settings.",
            model="m",
            model_settings=ModelSettings(temperature=0.2, max_tokens=100),
        ),
        "go",
        run_config=RunConfig(
            workspace=tmp_path,
            model_provider=model_provider,
            model_settings=ModelSettings(max_tokens=250, top_p=0.8),
        ),
    )

    assert seen_settings == [ModelSettings(temperature=0.2, top_p=0.8, max_tokens=250)]


def test_runner_keeps_hidden_tools_executable_but_out_of_model_schemas(tmp_path: Path) -> None:
    captured_tool_names: list[str] = []

    @function_tool(exposure=ToolExposure.HIDDEN)
    def internal_lookup() -> str:
        return "internal"

    class CapturingLLM:
        def complete(
            self,
            *,
            model: str,
            messages: list[Message],
            tools: list[dict[str, object]],
            stream_callback=None,
            model_settings: ModelSettings | None = None,
            request_metadata=None,
        ) -> LLMResponse:
            del model, messages, stream_callback, model_settings, request_metadata
            for schema in tools:
                function = cast(dict[str, object], schema["function"])
                captured_tool_names.append(str(function["name"]))
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            )

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return CapturingLLM(), _fake_resolved()

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Finish.", model="m", tools=[internal_lookup]),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert result.final_output == "done"
    assert "internal_lookup" not in captured_tool_names


def test_stop_on_first_tool_finishes_only_after_a_successful_tool(tmp_path: Path) -> None:
    calls: list[str] = []

    @function_tool(name="first")
    def first() -> str:
        calls.append("first")
        return "first output"

    @function_tool(name="second")
    def second() -> str:
        calls.append("second")
        return "second output"

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(id="c1", name="first", arguments={}),
                            ToolCall(id="c2", name="second", arguments={}),
                        ],
                    )
                ]
            ),
            _fake_resolved(),
        )

    result = Runner.run_sync(
        Agent(
            name="assistant",
            instructions="Use tools.",
            model="m",
            tools=[first, second],
            tool_use_behavior="stop_on_first_tool",
        ),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert result.status == AgentStatus.COMPLETED
    assert result.final_output == "first output"
    assert calls == ["first"]
    assert result.raw_result.cycles[0].tool_results[1].error_code == "skipped_due_to_finish"


def test_stop_on_first_tool_does_not_finish_on_a_no_tool_response(tmp_path: Path) -> None:
    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(content="draft without a tool"),
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})
                        ],
                    ),
                ]
            ),
            _fake_resolved(),
        )

    result = Runner.run_sync(
        Agent(
            name="assistant",
            instructions="Finish explicitly.",
            model="m",
            tool_use_behavior="stop_on_first_tool",
        ),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert result.status == AgentStatus.COMPLETED
    assert result.final_output == "done"
    assert len(result.raw_result.cycles) == 2


def test_stop_at_tool_names_ignores_other_successful_tools(tmp_path: Path) -> None:
    calls: list[str] = []

    @function_tool(name="prepare")
    def prepare() -> str:
        calls.append("prepare")
        return "prepared"

    @function_tool(name="publish")
    def publish() -> str:
        calls.append("publish")
        return "published"

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(id="c1", name="prepare", arguments={}),
                            ToolCall(id="c2", name="publish", arguments={}),
                        ],
                    )
                ]
            ),
            _fake_resolved(),
        )

    result = Runner.run_sync(
        Agent(
            name="assistant",
            instructions="Use tools.",
            model="m",
            tools=[prepare, publish],
            tool_use_behavior="stop_at_tool_names",
            stop_at_tool_names=["publish"],
        ),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert result.status == AgentStatus.COMPLETED
    assert result.final_output == "published"
    assert calls == ["prepare", "publish"]


def test_runner_exposes_read_image_for_resolved_multimodal_model(tmp_path: Path) -> None:
    seen_tool_names: set[str] = set()

    class CapturingLLM:
        def complete(
            self,
            *,
            model: str,
            messages: list[Message],
            tools: list[dict[str, object]],
            stream_callback=None,
            model_settings: ModelSettings | None = None,
            request_metadata=None,
        ) -> LLMResponse:
            del model, messages, stream_callback, model_settings, request_metadata
            for schema in tools:
                function = schema.get("function")
                if isinstance(function, dict):
                    typed_function = cast(dict[str, object], function)
                    seen_tool_names.add(str(typed_function.get("name")))
            return LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
            )

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return CapturingLLM(), _fake_resolved(native_multimodal=True)

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Inspect images.", model="m"),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert result.status == AgentStatus.COMPLETED
    assert READ_IMAGE_TOOL_NAME in seen_tool_names


def test_runner_prefers_resolved_catalog_token_limits_for_memory(tmp_path: Path, monkeypatch) -> None:
    from vv_agent.memory import MemoryManager

    manager_kwargs: list[dict[str, object]] = []

    def capture_memory_manager(**kwargs):
        manager_kwargs.append(kwargs)
        return MemoryManager(**kwargs)

    def reject_static_defaults(model: str):
        raise AssertionError(f"static defaults should not be queried for {model}")

    monkeypatch.setattr("vv_agent.runtime.engine.MemoryManager", capture_memory_manager)
    monkeypatch.setattr("vv_agent.runtime.engine.resolve_model_token_limits", reject_static_defaults)

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="done",
                        tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
                    )
                ]
            ),
            _fake_resolved(context_length=64_000, max_output_tokens=8_000),
        )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Use catalog limits.", model="m"),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert result.status == AgentStatus.COMPLETED
    assert manager_kwargs[0]["model_context_window"] == 64_000
    assert manager_kwargs[0]["reserved_output_tokens"] == 8_000


def test_runner_requires_llm_complete_to_accept_model_settings(tmp_path: Path) -> None:
    class LegacyLLM:
        def complete(
            self,
            *,
            model: str,
            messages: list[Message],
            tools: list[dict[str, object]],
            stream_callback=None,
            request_metadata=None,
        ):
            del model, messages, tools, stream_callback, request_metadata
            return LLMResponse(content="unused")

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return LegacyLLM(), _fake_resolved()

    result = Runner.run_sync(
        Agent(
            name="assistant",
            instructions="Use resolved settings.",
            model="m",
            model_settings=ModelSettings(temperature=0.2),
        ),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    error_text = result.raw_result.error or result.final_output or ""
    assert result.status == AgentStatus.FAILED
    assert "model_settings" in error_text


def test_runner_stream_sync_yields_typed_events(tmp_path: Path) -> None:
    class StreamingLLM:
        def complete(
            self,
            *,
            model: str,
            messages: list[Message],
            tools: list[dict[str, object]],
            stream_callback=None,
            model_settings: ModelSettings | None = None,
            request_metadata=None,
        ) -> LLMResponse:
            del model, messages, tools, model_settings, request_metadata
            if stream_callback is not None:
                stream_callback({"event": "assistant_delta", "content_delta": "hel"})
                stream_callback({"event": "assistant_delta", "content_delta": "lo"})
            return LLMResponse(
                content="hello",
                tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            )

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return StreamingLLM(), _fake_resolved()

    events = list(
        Runner.stream_sync(
            Agent(name="assistant", instructions="Stream.", model="m"),
            "go",
            run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
        )
    )

    assert [event.type for event in events] == [
        "run_started",
        "agent_started",
        "cycle_started",
        "llm_started",
        "assistant_delta",
        "assistant_delta",
        "tool_call_started",
        "tool_call_completed",
        "run_completed",
    ]
    assert [event.delta for event in events if isinstance(event, AssistantDeltaEvent)] == ["hel", "lo"]
    assert "cycle_llm_response" not in {event.type for event in events}
    completed = events[-1]
    assert isinstance(completed, RunCompletedEvent)
    assert completed.final_output == "done"
    assert completed.to_dict()["type"] == "run_completed"
    assert isinstance(events[1], AgentStartedEvent)
    assert isinstance(events[2], CycleStartedEvent)
    assert isinstance(events[3], LLMStartedEvent)


def test_runner_appends_session_items_across_runs(tmp_path: Path) -> None:
    session = MemorySession("thread-1")
    calls: list[list[Message]] = []

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config

        def respond(request: LlmRequest) -> LLMResponse:
            calls.append(list(request.messages))
            call_index = len(calls)
            return LLMResponse(
                content=f"{['first', 'second'][call_index - 1]} assistant",
                tool_calls=[
                    ToolCall(
                        id=f"finish_{call_index}",
                        name=TASK_FINISH_TOOL_NAME,
                        arguments={"message": f"{['first', 'second'][call_index - 1]} result"},
                    )
                ],
            )

        return ScriptedLLM(steps=[respond]), _fake_resolved()

    agent = Agent(name="assistant", instructions="Remember context.", model="m")
    config = RunConfig(workspace=tmp_path, session=session, model_provider=model_provider)

    first = Runner.run_sync(agent, "first input", run_config=config)
    second = Runner.run_sync(agent, "second input", run_config=config)

    assert first.final_output == "first result"
    assert second.final_output == "second result"
    persisted = session.get_items()
    fixture_path = Path(__file__).parent / "fixtures" / "parity" / "runner_session_messages_v1.jsonl"
    expected = [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines()]
    assert [item.to_dict() for item in persisted] == expected
    assert [message.content for message in calls[1] if message.role == "user"] == ["first input", "second input"]
    assert [message.role for message in calls[1]] == ["system", "user", "assistant", "tool", "user"]
    assert calls[1][2].tool_calls is not None
    assert calls[1][3].tool_call_id == "finish_1"


def test_runner_passes_tool_context_to_function_tool(tmp_path: Path) -> None:
    from vv_agent import ToolCallContext, function_tool

    @function_tool
    def report_context(context: ToolCallContext, value: str) -> str:
        """Report call context."""
        assert context.run_context is not None
        assert context.run_context.model == "m"
        context.shared_state["observed"] = value
        return (
            f"{context.run_id}:{context.agent_name}:{context.tool_name}:"
            f"{context.tool_call_id}:{context.raw_arguments['value']}:{context.app_state}:{value}"
        )

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="call",
                        tool_calls=[ToolCall(id="tool-call-1", name="report_context", arguments={"value": "42"})],
                    )
                ]
            ),
            _fake_resolved(),
        )

    result = Runner.run_sync(
        Agent(
            name="assistant",
            instructions="Use the tool.",
            model="m",
            tools=[report_context],
            tool_use_behavior="stop_on_first_tool",
        ),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider, context="app-state"),
    )

    assert result.final_output is not None
    assert result.final_output.endswith(":assistant:report_context:tool-call-1:42:app-state:42")
    assert result.raw_result.shared_state["observed"] == "42"


def test_runner_coerces_json_final_output_to_dataclass(tmp_path: Path) -> None:
    @dataclass
    class Summary:
        title: str
        count: int

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return (
            ScriptedLLM(
                steps=[
                    LLMResponse(
                        content="done",
                        tool_calls=[
                            ToolCall(
                                id="finish",
                                name=TASK_FINISH_TOOL_NAME,
                                arguments={"message": '{"title": "orders", "count": 3}'},
                            )
                        ],
                    )
                ]
            ),
            _fake_resolved(),
        )

    result = Runner.run_sync(
        Agent(name="assistant", instructions="Return JSON.", model="m", output_type=Summary),
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=model_provider),
    )

    assert result.final_output == Summary(title="orders", count=3)
