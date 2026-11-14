"""Memory-boundary rejection leaves SQLite history, scheduling and caches unchanged."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from vv_agent.session.records import SessionSpec, make_record
from vv_agent.session.reducer import TransitionError
from vv_agent.session.sqlite import SQLiteStore

from .helpers import record

FIXTURE = Path(__file__).parents[1] / "fixtures/parity/run_events.jsonl"


def wire(kind="memory_compact_started") -> dict[str, Any]:
    payload = next(json.loads(line) for line in FIXTURE.read_text().splitlines() if json.loads(line)["type"] == kind)
    return payload | {"run_id": "t", "session_id": "s"}


def boundary(stage, key, payload):
    return make_record(
        "boundary_recorded",
        session_id="s",
        turn_id="t",
        payload={
            "stage": stage,
            "boundary_id": key,
            "source_operation_id": None,
            "source_digest": None,
            "data": {"event": payload},
        },
    )


@pytest.fixture(params=["memory", "file"])
def sqlite_session(request, tmp_path):
    path = ":memory:" if request.param == "memory" else str(tmp_path / "events.sqlite")
    with SQLiteStore.standalone(path) as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("s", "p", "w", attributes={"seed": {"unseen_input_protection": False}}), consumers=("host",))
        lease = store.acquire("s", owner="event-validation", ttl_ms=15000)
        assert lease is not None
        with store.atomic() as tx:
            tx.append("s", lease=lease, expected_seq=1, commit_id="turn", records=(record("turn_started"),))
        yield store, lease


def sql_snapshot(store):
    return {
        table: tuple(store.connection.execute(f"SELECT * FROM {table} ORDER BY rowid"))
        for table in ("sk_session", "sk_record", "sk_commit", "sk_inbox", "sk_consumer")
    }


def rejection(case):
    stage = "memory_started"
    payload: dict[str, Any] = wire()
    error = ValueError
    if case == "valid_wrong_kind":
        payload = wire("agent_started")
        error, message = TransitionError, "memory lifecycle type mismatch"
    elif case == "malformed_wrong_kind":
        payload = wire("diagnostic") | {"level": "bad", "run_id": "wrong"}
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
            # Retained record canonicalization turns 1.0 into 1 before the fold validates the event.
            error, message = TransitionError, "memory lifecycle type mismatch"
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
        payload = wire("memory_compact_completed") | {"run_id": "wrong"}
        error, message = TransitionError, "memory lifecycle type mismatch"
    elif case == "completion_without_start":
        stage, payload = "memory_completed", wire("memory_compact_completed")
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
    store, lease = sqlite_session
    stage, payload, error, message = rejection(case)
    prefix = boundary("memory_started", "prefix", wire())
    bad = boundary(stage, "case", payload)
    if case == "cycle_float":
        assert type(bad._payload["data"]["event"]["logical_cycle"]) is int
    state, history, _ = store.read_state("s")
    assert not state.boundaries
    before = sql_snapshot(store)
    with pytest.raises(error) as raised, store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=2, commit_id="events", records=(prefix, bad))
    assert str(raised.value) == message
    assert sql_snapshot(store) == before
    current, rows, _ = store.read_state("s")
    assert current.boundaries == state.boundaries == {}
    assert current.active_turn_id == state.active_turn_id == "t"
    assert [row.record.encode() for row in rows] == [row.record.encode() for row in history]
    # Reuse the failed commit and boundary identities with valid bytes: neither was retained by the rejected batch.
    corrected = boundary("memory_completed", "prefix", wire("memory_compact_completed"))
    with store.atomic() as tx:
        receipt = tx.append("s", lease=lease, expected_seq=2, commit_id="events", records=(prefix, corrected))
    assert receipt.head_seq == 4
    accepted, _, _ = store.read_state("s")
    assert set(accepted.boundaries) == {("t", "memory_started", "prefix"), ("t", "memory_completed", "prefix")}


def test_memory_pair_keeps_existing_loose_identity_and_sparse_wire_rules(sqlite_session):
    store, lease = sqlite_session
    started = wire() | {
        "event_id": "arbitrary-start",
        "trace_id": "unrelated-trace",
        "created_at": 0.25,
        "cycle_index": 2**53 - 1,
        "estimated_tokens": None,
        "metadata": {"future_extension": {"kept": True}},
    }
    completed = wire("memory_compact_completed") | {
        "event_id": "arbitrary-finish",
        "trace_id": "different-trace",
        "created_at": 0,
        "cycle_index": None,
        "summary_tokens": None,
    }
    originals = deepcopy([started, completed])
    with store.atomic() as tx:
        tx.append(
            "s",
            lease=lease,
            expected_seq=2,
            commit_id="events",
            records=(boundary("memory_started", "pair", started), boundary("memory_completed", "pair", completed)),
        )
    state, _, _ = store.read_state("s")
    retained = [
        state.boundaries[("t", stage, "pair")]._payload["data"]["event"] for stage in ("memory_started", "memory_completed")
    ]
    assert retained == originals
    assert "estimated_tokens" in retained[0] and retained[0]["estimated_tokens"] is None
    assert "summary_tokens" in retained[1] and retained[1]["summary_tokens"] is None
    assert [started, completed] == originals
