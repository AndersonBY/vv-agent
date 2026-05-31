from __future__ import annotations

from vv_agent import Agent, RunConfig, Runner
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.event_store import JsonlRunEventStore
from vv_agent.events import RunStartedEvent
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse, ToolCall


def test_jsonl_event_store_appends_and_replays_events(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    store = JsonlRunEventStore(path)
    event = RunStartedEvent(run_id="run_1", trace_id="trace_1", input="hi", session_id="s1")

    store.append(event)

    replayed = list(store.replay(run_id="run_1"))
    assert len(replayed) == 1
    assert replayed[0].type == "run_started"
    assert replayed[0].run_id == "run_1"
    assert replayed[0].session_id == "s1"


def test_runner_appends_events_to_configured_event_store(tmp_path) -> None:
    store = JsonlRunEventStore(tmp_path / "events.jsonl")
    agent = Agent(name="assistant", instructions="Answer.", model="test-model")
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            )
        ]
    )

    def model_provider(_agent: Agent, _run_config: RunConfig):
        endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
        return llm, ResolvedModelConfig(
            backend="test",
            requested_model="test-model",
            selected_model="test-model",
            model_id="test-model",
            endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
        )

    result = Runner.run_sync(
        agent,
        "go",
        run_config=RunConfig(model_provider=model_provider, event_store=store),
    )

    replayed = list(store.replay(run_id=result.run_id))
    assert replayed[0].type == "run_started"
    assert replayed[-1].type == "run_completed"
