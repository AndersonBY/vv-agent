from __future__ import annotations

from collections.abc import Iterator

import pytest

from vv_agent import Agent, RunConfig, Runner, event_from_dict
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.event_store import JsonlRunEventStore
from vv_agent.events import ApprovalResolvedEvent, RunEvent, RunStartedEvent
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse, ToolCall


def test_jsonl_event_store_appends_and_replays_events(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    store = JsonlRunEventStore(path)
    event = RunStartedEvent(
        run_id="run_1",
        trace_id="trace_1",
        input="hi",
        session_id="s1",
        parent_event_id="evt_parent",
        parent_run_id="run_parent",
        event_id="evt_started",
        created_at=123.45,
        metadata={"source": "test"},
    )
    approval = ApprovalResolvedEvent(
        run_id="run_1",
        trace_id="trace_1",
        tool_name="search",
        tool_call_id="call_1",
        approved=False,
        event_id="evt_approval",
        created_at=124.45,
    )

    store.append(event)
    store.append(approval)

    replayed = list(store.replay(run_id="run_1"))
    assert len(replayed) == 2
    assert replayed[0].type == "run_started"
    assert replayed[0].run_id == "run_1"
    assert replayed[0].session_id == "s1"
    assert replayed[0].event_id == "evt_started"
    assert replayed[0].created_at == 123.45
    assert replayed[0].parent_event_id == "evt_parent"
    assert replayed[0].parent_run_id == "run_parent"
    assert replayed[0].metadata == {"source": "test"}
    assert isinstance(replayed[1], ApprovalResolvedEvent)
    assert replayed[1].approved is False
    assert event_from_dict(replayed[1].to_dict()).event_id == "evt_approval"


def test_runner_appends_events_to_configured_event_store(tmp_path) -> None:
    store = JsonlRunEventStore(tmp_path / "events.jsonl")
    agent = Agent(name="assistant", instructions="Answer.", model="test-model")

    result = Runner.run_sync(
        agent,
        "go",
        run_config=RunConfig(model_provider=_model_provider(), event_store=store),
    )

    replayed = list(store.replay(run_id=result.run_id))
    assert replayed[0].type == "run_started"
    assert replayed[-1].type == "run_completed"


def test_event_store_append_failure_warns_and_run_continues_by_default() -> None:
    agent = Agent(name="assistant", instructions="Answer.", model="test-model")

    with pytest.warns(RuntimeWarning, match="Run event store append failed: store down"):
        result = Runner.run_sync(
            agent,
            "go",
            run_config=RunConfig(model_provider=_model_provider(), event_store=FailingRunEventStore()),
        )

    assert result.final_output == "done"


def test_event_store_append_failure_raises_when_fail_closed() -> None:
    agent = Agent(name="assistant", instructions="Answer.", model="test-model")

    with pytest.raises(RuntimeError, match="store down"):
        Runner.run_sync(
            agent,
            "go",
            run_config=RunConfig(
                model_provider=_model_provider(),
                event_store=FailingRunEventStore(),
                event_store_fail_closed=True,
            ),
        )


class FailingRunEventStore:
    def append(self, event: RunEvent) -> None:
        raise RuntimeError("store down")

    def replay(self, *, run_id: str) -> Iterator[RunEvent]:
        return iter(())


def _model_provider():
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            )
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del agent, run_config
        endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
        return llm, ResolvedModelConfig(
            backend="test",
            requested_model="test-model",
            selected_model="test-model",
            model_id="test-model",
            endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
        )

    return model_provider
