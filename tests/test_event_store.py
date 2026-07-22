from __future__ import annotations

from collections.abc import Iterator

import pytest
from support import FixedModelProvider

from vv_agent import Agent, RunConfig, Runner, event_from_dict
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.event_store import JsonlRunEventStore, RunEventReplayQuery
from vv_agent.events import ApprovalResolvedEvent, RunEvent, RunStartedEvent, SubRunCompletedEvent, SubRunStartedEvent
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
        request_id="request_1",
        action="deny",
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
    assert replayed[1].action == "deny"
    assert event_from_dict(replayed[1].to_dict()).event_id == "evt_approval"


def test_jsonl_event_store_parent_replay_includes_child_run_edges(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    store = JsonlRunEventStore(path)
    parent = RunStartedEvent(
        run_id="run_parent",
        trace_id="trace_1",
        input="parent input",
        event_id="evt_parent",
        created_at=1.0,
    )
    child_started = SubRunStartedEvent(
        run_id="run_child",
        trace_id="trace_1",
        session_id="session_child",
        parent_run_id="run_parent",
        parent_tool_call_id="call_create_sub_task",
        agent_name="researcher",
        event_id="evt_child_started",
        created_at=2.0,
    )
    child_completed = SubRunCompletedEvent(
        run_id="run_child",
        trace_id="trace_1",
        session_id="session_child",
        parent_run_id="run_parent",
        parent_tool_call_id="call_create_sub_task",
        agent_name="researcher",
        status="completed",
        event_id="evt_child_completed",
        created_at=3.0,
    )
    unrelated = SubRunStartedEvent(
        run_id="run_other_child",
        trace_id="trace_1",
        parent_run_id="run_other_parent",
        parent_tool_call_id="call_other",
        agent_name="other",
        event_id="evt_unrelated",
        created_at=4.0,
    )

    store.append(parent)
    store.append(child_started)
    store.append(child_completed)
    store.append(unrelated)

    parent_replay = list(store.replay(run_id="run_parent"))
    assert [event.event_id for event in parent_replay] == [
        "evt_parent",
        "evt_child_started",
        "evt_child_completed",
    ]
    assert parent_replay[1].run_id == "run_child"
    assert parent_replay[1].parent_run_id == "run_parent"

    child_replay = list(store.replay(run_id="run_child"))
    assert [event.event_id for event in child_replay] == ["evt_child_started", "evt_child_completed"]


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

    def replay(
        self,
        query: RunEventReplayQuery | None = None,
        *,
        run_id: str | None = None,
    ) -> Iterator[RunEvent]:
        del query, run_id
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

    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    resolved = ResolvedModelConfig(
        backend="test",
        requested_model="test-model",
        selected_model="test-model",
        model_id="test-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
    )
    return FixedModelProvider(llm, resolved)
