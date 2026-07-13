from __future__ import annotations

from threading import Event

import pytest

from vv_agent import Agent, RunConfig, Runner
from vv_agent.background_task import BackgroundAgentTask, BackgroundAgentTaskHandle
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.types import AgentStatus, LLMResponse, Message, ToolCall


def _resolved() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="background-model",
        selected_model="background-model",
        model_id="background-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="background-model")],
    )


def test_agent_background_task_starts_non_blocking_and_returns_pollable_handle() -> None:
    entered = Event()
    release = Event()

    class BlockingLLM:
        def complete(
            self,
            *,
            model: str,
            messages: list[Message],
            tools: list[dict[str, object]],
            stream_callback=None,
            model_settings=None,
            request_metadata=None,
        ) -> LLMResponse:
            del model, messages, tools, stream_callback, model_settings, request_metadata
            entered.set()
            assert release.wait(timeout=2)
            return LLMResponse(
                content="drafted",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "background draft"})],
            )

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        return BlockingLLM(), _resolved()

    agent = Agent(name="drafter", instructions="Draft the report.", model="background-model")
    task = agent.as_background_task(name="draft_report", description="Draft in the background.")

    assert isinstance(task, BackgroundAgentTask)
    assert task.params_json_schema["required"] == ["task_description"]
    handle = task.start(
        Runner,
        None,
        {"task_description": "draft the parity report"},
        run_config=RunConfig(model_provider=model_provider),
    )

    try:
        assert isinstance(handle, BackgroundAgentTaskHandle)
        assert handle.task_id.startswith("bg_agent_")
        assert handle.agent_name == "drafter"
        assert task.get_handle(handle.task_id) is handle
        assert entered.wait(timeout=1)
        assert handle.poll().status == AgentStatus.RUNNING
        assert handle.snapshot().done is False
        with pytest.raises(TimeoutError):
            handle.wait(timeout=0.01)
    finally:
        release.set()

    completed = handle.wait(timeout=2)
    assert completed.status == AgentStatus.COMPLETED
    assert completed.final_output == "background draft"
    assert completed.done is True
    assert handle.poll() == completed


def test_background_task_rejects_missing_task_description() -> None:
    task = Agent(name="worker", instructions="Work.", model="m").as_background_task()

    with pytest.raises(ValueError, match="requires task_description"):
        task.start(Runner, None, {})
