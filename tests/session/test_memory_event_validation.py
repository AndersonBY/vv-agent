"""Memory-boundary rejection leaves SQLite history, scheduling and caches unchanged."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel._content_publication import make_record_before_publication
from vv_agent.session import _recovery_phase
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import _Driver
from vv_agent.session.records import InboxItem
from vv_agent.session.reducer import TransitionError
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver

from .helpers import (
    fixture_content_reader,
    read_fixture_state,
)
from .test_recovery_matrix import runtime

FIXTURE = Path(__file__).parents[1] / "fixtures/parity/run_events.jsonl"


def wire(kind="memory_compact_started", *, run_id="t") -> dict[str, Any]:
    payload = next(json.loads(line) for line in FIXTURE.read_text().splitlines() if json.loads(line)["type"] == kind)
    return payload | {"run_id": run_id, "session_id": "s"}


def boundary(stage, key, payload, *, turn_id):
    return make_record_before_publication(
        "boundary_recorded",
        session_id="s",
        turn_id=turn_id,
        payload={
            "stage": stage,
            "boundary_id": key,
            "source_operation_id": None,
            "source_digest": None,
            "data": {"event": payload},
        },
    )


@pytest.fixture
def sqlite_session(tmp_path):
    path = tmp_path / "events.sqlite"
    with SQLiteStore.standalone(str(path)) as raw_store:
        sessions = SessionDriver(store=raw_store, content_key_provider=LocalKeyProvider(b"m" * 32))
        sessions.create(
            "s",
            "/workspace",
            attributes={"seed": {"unseen_input_protection": False}},
        )
        sessions.push("s", InboxItem("initial", "user", {"content": "go"}))
        rt = sessions.runtime(
            Agent("memory-events", "Validate memory boundaries."),
            RunConfig(model_provider=ScriptedModelProvider.new("scripted", "scripted", [])),
            session_id="s",
        )
        lease = sessions.store.acquire("s", owner="event-validation", ttl_ms=15000)
        assert lease is not None
        driver = _Driver(sessions.store, "s", rt, _Scope(lease, rt, now_ms=raw_store._now()))
        assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
        assert _recovery_phase.run(driver, inputs.start_turn(driver))
        turn_id = driver.state.active_turn_id
        assert turn_id is not None
        assert sessions.store.release(lease)
        yield sessions.store, path, turn_id


def commit_boundary_records(store, path, records, lease):
    rt = runtime(path, [], content_reader=fixture_content_reader(store, "s"))
    driver = _Driver(store, "s", rt, _Scope(lease, rt, now_ms=store._now()))
    driver.commit(list(records))


def sql_snapshot(store):
    return {
        table: tuple(store.connection.execute(f"SELECT * FROM {table} ORDER BY rowid"))
        for table in ("sk_session", "sk_record", "sk_commit", "sk_inbox", "sk_consumer")
    }


def rejection(case, turn_id):
    stage = "memory_started"
    payload: dict[str, Any] = wire(run_id=turn_id)
    error = ValueError
    if case == "valid_wrong_kind":
        payload = wire("agent_started", run_id=turn_id)
        error, message = TransitionError, "memory lifecycle type mismatch"
    elif case == "malformed_wrong_kind":
        payload = wire("diagnostic", run_id=turn_id) | {"level": "bad", "run_id": "wrong"}
        message = "Unsupported diagnostic level: 'bad'"
    elif case.startswith("cycle_"):
        payload = {
            "version": "v6",
            "type": "cycle_aborted",
            "run_id": "wrong",
            "session_id": "s",
            "trace_id": "trace",
            "event_id": "evt_cycle",
            "created_at": 0,
            "cycle_index": 0,
            "logical_cycle": 1.0 if case == "cycle_float" else True,
            "reason": "cancelled",
        }
        message = "cycle_aborted logical_cycle must equal cycle_index + 1"
        if case == "cycle_float":
            message = "cycle_aborted logical_cycle must equal cycle_index + 1"
        elif case == "cycle_metadata":
            payload["metadata"] = None
            message = "Run event metadata must be an object"
        elif case == "cycle_unknown":
            payload.update(metadata=None, future=1)
            message = "Run event cycle_aborted contains unknown fields: future"
    elif case == "bad_counter_and_identity":
        payload.update(message_count=-1, run_id="wrong")
        message = "Run event message_count must be a non-negative JSON-safe integer or null"
    elif case == "missing_and_malformed":
        payload.pop("trigger")
        payload["message_count"] = -1
        message = "Run event memory_compact_started is missing required fields: trigger"
    elif case == "unknown_and_missing":
        payload.pop("trigger")
        payload["future"] = 1
        message = "Run event memory_compact_started contains unknown fields: future"
    elif case in {"wrong_run", "wrong_session", "null_session"}:
        payload["run_id" if case == "wrong_run" else "session_id"] = None if case == "null_session" else "wrong"
        error, message = TransitionError, "memory lifecycle identity mismatch"
    elif case == "wrong_kind_and_identity":
        payload = wire("memory_compact_completed", run_id=turn_id) | {"run_id": "wrong"}
        error, message = TransitionError, "memory lifecycle type mismatch"
    elif case == "completion_without_start":
        stage, payload = "memory_completed", wire("memory_compact_completed", run_id=turn_id)
        error, message = TransitionError, "memory completion without start"
    else:
        raise AssertionError(case)
    return stage, payload, error, message


@pytest.mark.parametrize(
    "case",
    [
        "valid_wrong_kind",
        "malformed_wrong_kind",
        "cycle_bool",
        "cycle_float",
        "cycle_metadata",
        "cycle_unknown",
        "bad_counter_and_identity",
        "missing_and_malformed",
        "unknown_and_missing",
        "wrong_run",
        "wrong_session",
        "null_session",
        "wrong_kind_and_identity",
        "completion_without_start",
    ],
)
def test_rejected_batch_preserves_sqlite_state_and_allows_corrected_commit(sqlite_session, case):
    store, path, turn_id = sqlite_session
    stage, payload, error, message = rejection(case, turn_id)
    prefix = boundary("memory_started", "prefix", wire(run_id=turn_id), turn_id=turn_id)
    bad = boundary(stage, "case", payload, turn_id=turn_id)
    if case == "cycle_float":
        # Logical boundary values stay untouched until publication and trusted hydration.
        assert type(bad._payload["data"]["event"]["logical_cycle"]) is float
    lease = store.acquire("s", owner="memory-event", ttl_ms=15_000)
    assert lease is not None
    try:
        state, history, _ = store.read_state("s")
        assert not state.boundaries
        before_head = store.read("s", limit=1).head_seq
        before = sql_snapshot(store)
        with pytest.raises(error) as raised:
            commit_boundary_records(store, path, (prefix, bad), lease)
        assert str(raised.value) == message
        assert sql_snapshot(store) == before
        current, rows, _ = store.read_state("s")
        assert current.boundaries == state.boundaries == {}
        assert current.active_turn_id == state.active_turn_id == turn_id
        assert [row.record.encode() for row in rows] == [row.record.encode() for row in history]
        # Reuse the failed commit and boundary identities with valid bytes: neither was retained by the rejected batch.
        corrected = boundary("memory_completed", "prefix", wire("memory_compact_completed", run_id=turn_id), turn_id=turn_id)
        commit_boundary_records(store, path, (prefix, corrected), lease)
        assert store.read("s", limit=1).head_seq == before_head + 2
        accepted, _, _ = store.read_state("s")
        assert set(accepted.boundaries) == {
            (turn_id, "memory_started", "prefix"),
            (turn_id, "memory_completed", "prefix"),
        }
    finally:
        assert store.release(lease)


def test_memory_pair_keeps_existing_loose_identity_and_sparse_wire_rules(sqlite_session):
    store, path, turn_id = sqlite_session
    started = wire(run_id=turn_id) | {
        "event_id": "arbitrary-start",
        "trace_id": "unrelated-trace",
        "created_at": 0.25,
        "cycle_index": 2**53 - 1,
        "estimated_tokens": None,
        "metadata": {"future_extension": {"kept": True}},
    }
    completed = wire("memory_compact_completed", run_id=turn_id) | {
        "event_id": "arbitrary-finish",
        "trace_id": "different-trace",
        "created_at": 0,
        "cycle_index": None,
        "summary_tokens": None,
    }
    originals = deepcopy([started, completed])
    lease = store.acquire("s", owner="memory-event", ttl_ms=15_000)
    assert lease is not None
    commit_boundary_records(
        store,
        path,
        (
            boundary("memory_started", "pair", started, turn_id=turn_id),
            boundary("memory_completed", "pair", completed, turn_id=turn_id),
        ),
        lease,
    )
    assert store.release(lease)
    state, _, _ = read_fixture_state(store, "s")
    retained = [
        state.boundaries[(turn_id, stage, "pair")]._payload["data"]["event"] for stage in ("memory_started", "memory_completed")
    ]
    assert retained == originals
    assert "estimated_tokens" in retained[0] and retained[0]["estimated_tokens"] is None
    assert "summary_tokens" in retained[1] and retained[1]["summary_tokens"] is None
    assert [started, completed] == originals
