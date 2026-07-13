from __future__ import annotations

from pathlib import Path

import pytest

from vv_agent import EventStoreError, JsonlRunEventStore, RunEventReplayQuery, RunStartedEvent

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "event_store_replay_v1.jsonl"


def test_replay_query_includes_children_by_default_and_can_exclude_them(tmp_path: Path) -> None:
    store = JsonlRunEventStore(tmp_path / "events.jsonl")
    store.append(
        RunStartedEvent(
            run_id="run_parent",
            trace_id="trace_store",
            input="first",
            event_id="evt_parent_first",
            created_at=1.0,
        )
    )
    store.append(
        RunStartedEvent(
            run_id="run_child",
            trace_id="trace_store",
            input="child",
            parent_run_id="run_parent",
            event_id="evt_child",
            created_at=2.0,
        )
    )
    store.append(
        RunStartedEvent(
            run_id="run_parent",
            trace_id="trace_store",
            input="last",
            event_id="evt_parent_last",
            created_at=3.0,
        )
    )

    default_query = RunEventReplayQuery(run_id="run_parent")
    assert default_query.include_children is True
    assert [event.event_id for event in store.replay(default_query)] == [
        "evt_parent_first",
        "evt_child",
        "evt_parent_last",
    ]

    direct_only = RunEventReplayQuery(run_id="run_parent", include_children=False)
    assert [event.event_id for event in store.replay(query=direct_only)] == [
        "evt_parent_first",
        "evt_parent_last",
    ]


def test_jsonl_replay_is_lazy_and_stops_at_the_corrupt_contract_line() -> None:
    iterator = JsonlRunEventStore(FIXTURE_PATH).replay(RunEventReplayQuery.run("run_parent"))

    first = next(iterator)
    assert first.event_id == "evt_parent"

    with pytest.raises(EventStoreError) as caught:
        next(iterator)

    assert caught.value.code == "event_store_corrupt_line"
    assert caught.value.line_number == 2
    assert str(caught.value) == "event store corrupt line 2"
    with pytest.raises(StopIteration):
        next(iterator)


def test_jsonl_replay_returns_empty_iterator_for_missing_file(tmp_path: Path) -> None:
    store = JsonlRunEventStore(tmp_path / "missing.jsonl")

    assert list(store.replay(RunEventReplayQuery(run_id="run_missing"))) == []
