"""M6 capacity bounds use the M5 PostgreSQL history and default lease settings."""

import time
from collections import Counter
from contextlib import suppress

import pytest
from support.s1_fixture_values import (
    append_fixture_records,
    content_ref,
    create_fixture_session,
    fixture_content_reader,
    read_fixture_state,
    wrap_fixture_store,
)

from vv_agent.budget import RunBudgetLimits
from vv_agent.kernel import _driver_recovery
from vv_agent.kernel._content_publication import make_record_before_publication
from vv_agent.session.kernel import drive
from vv_agent.session.records import digest
from vv_agent.types import LLMResponse, ToolExecutionResult

from . import test_recovery_matrix as recovery
from .conftest import open_store
from .helpers import record

DEFAULT_BUDGET = RunBudgetLimits().to_dict()


@pytest.fixture
def store(store):
    return wrap_fixture_store(store)


def runtime(store, database, steps, tools=(), *, sid="s", **kwargs):
    kwargs.setdefault("content_reader", fixture_content_reader(store, sid))
    return recovery.runtime(database, steps, tools, **kwargs)


def read_cold_state(database, sid="s"):
    with open_store(database) as store:
        return read_fixture_state(wrap_fixture_store(store), sid)


def turn_record(store, database, tid="t", text="go"):
    rt = runtime(store, database, [])
    rt.model_binding(rt.llm, rt.resolved)
    task = rt.compile(text, tid)
    definition = rt.definition(task)
    payload = record("turn_started", tid=tid).payload | {
        "definition": definition,
        "definition_digest": digest(definition),
        "budget": DEFAULT_BUDGET,
    }
    turn = make_record_before_publication("turn_started", session_id="s", turn_id=tid, payload=payload)
    object.__setattr__(turn, "_task_value", task)
    return turn


def seed_history(store, database, size):
    recovery.start(store)

    class ReceiptCut(BaseException):
        pass

    def hook(point, value):
        if point == "after_commit" and value.kind == "op_completed":
            raise ReceiptCut

    with suppress(ReceiptCut):
        drive(store, "s", runtime=runtime(store, database, [LLMResponse("retained answer")], hook=hook))
    state, rows, _ = read_fixture_state(store, "s")
    tid = state.active_turn_id
    assert tid is not None and state.phase == "active"
    probe = store.acquire("s", owner="benchmark-probe", ttl_ms=600000)
    assert probe is not None
    store.release(probe)
    publication_epoch = probe.epoch + 1
    pending = []
    # A bounded 1 KiB tool receipt; no growing model-request snapshots. This is a lower bound.
    while len(rows) + len(pending) + 3 <= size:
        oid = f"{tid}/benchmark/tool/{len(pending)}"
        request = {"id": oid, "name": "benchmark", "arguments": {}}
        result = ToolExecutionResult(tool_call_id=oid, content="x" * 1024).to_dict()
        pending.extend(
            [
                make_record_before_publication(
                    "op_planned",
                    session_id="s",
                    turn_id=tid,
                    operation_id=oid,
                    attempt=1,
                    payload=record("op_planned", tid=tid, oid=oid).payload
                    | {
                        "op_kind": "tool",
                        "purpose": None,
                        "request": request,
                        "request_digest": digest(request),
                        "budget_admission": {},
                    },
                ),
                record("op_started", tid=tid, oid=oid, epoch=publication_epoch),
                make_record_before_publication(
                    "op_completed",
                    session_id="s",
                    turn_id=tid,
                    operation_id=oid,
                    attempt=1,
                    payload=record("op_completed", tid=tid, oid=oid).payload
                    | {
                        "request_digest": digest(request),
                        "result": result,
                        "result_digest": digest(result),
                        "usage": content_ref(f"benchmark-usage-{oid}"),
                        "evidence": content_ref(f"benchmark-evidence-{oid}"),
                    },
                ),
            ]
        )
    for index in range(size - len(rows) - len(pending)):
        pending.append(
            make_record_before_publication(
                "usage_observed",
                session_id="s",
                turn_id=tid,
                payload=record("usage_observed", tid=tid, meter_id="padding", observation=index + 1).payload
                | {"usage": {}, "source": {}},
            )
        )
    append_fixture_records(store, "s", pending, commit_id="benchmark-seed")
    state, rows, _ = store.read_state("s")
    assert len(rows) == size and state.phase == "active"
    return tid, tuple(r.record for r in rows)


def test_capacity_2000_records(store, database):
    tid, _ = seed_history(store, database, 2000)
    with open_store(database) as cold:
        cold = wrap_fixture_store(cold)
        before = time.monotonic()
        drive(cold, "s", runtime=runtime(cold, database, []))
        elapsed = time.monotonic() - before
    state, rows, _ = read_fixture_state(store, "s")
    turn_rows = [row.record for row in rows if row.record.kind in {"turn_started", "turn_ended"}]
    assert elapsed <= 2, f"2000-record cold drive took {elapsed:.3f}s (limit 2s)"
    assert state.active_turn_id is None
    assert turn_rows[-1].kind == "turn_ended" and turn_rows[-1].turn_id == tid
    assert tid is not None


def test_recovery_hydrates_completed_tool_results_once(store, database, monkeypatch):
    _, history = seed_history(store, database, 120)
    expected = {
        operation_id
        for row in history
        if row.kind == "op_completed" and (operation_id := row.operation_id) is not None and "/benchmark/tool/" in operation_id
    }
    assert expected

    calls: Counter[str] = Counter()
    result = _driver_recovery._result

    def counted_result(driver, record):
        assert record.operation_id is not None
        calls[record.operation_id] += 1
        return result(driver, record)

    monkeypatch.setattr(_driver_recovery, "_result", counted_result)
    with open_store(database) as cold:
        drive(wrap_fixture_store(cold), "s", runtime=runtime(cold, database, []))

    observed = {operation_id: calls[operation_id] for operation_id in expected}
    assert observed == dict.fromkeys(expected, 1), observed


def seed_completed_prefix(store, database, size=5000):
    """A completed prior turn; current-turn cancellation/zombie assertions stay intact."""
    probe = store.acquire("s", owner="capacity-seed-probe", ttl_ms=15000)
    assert probe is not None
    store.release(probe)
    publication_epoch = probe.epoch + 1
    tid = "previous"
    rows = [turn_record(store, database, tid, "prior turn")]
    while len(rows) + 4 <= size - 1:
        oid = f"previous/tool/{len(rows)}"
        request = {"id": oid, "name": "benchmark", "arguments": {}}
        result = ToolExecutionResult(tool_call_id=oid, content="x" * 1024).to_dict()
        rows.extend(
            [
                make_record_before_publication(
                    "op_planned",
                    session_id="s",
                    turn_id=tid,
                    operation_id=oid,
                    attempt=1,
                    payload=record("op_planned", tid=tid, oid=oid).payload
                    | {
                        "op_kind": "tool",
                        "purpose": None,
                        "request": request,
                        "request_digest": digest(request),
                        "budget_admission": {},
                    },
                ),
                record("op_started", tid=tid, oid=oid, epoch=publication_epoch),
                make_record_before_publication(
                    "op_completed",
                    session_id="s",
                    turn_id=tid,
                    operation_id=oid,
                    attempt=1,
                    payload=record("op_completed", tid=tid, oid=oid).payload
                    | {
                        "request_digest": digest(request),
                        "result": result,
                        "result_digest": digest(result),
                        "usage": content_ref(f"benchmark-usage-{oid}"),
                        "evidence": content_ref(f"benchmark-evidence-{oid}"),
                    },
                ),
            ]
        )
    for index in range(size - 2 - len(rows)):
        rows.append(
            make_record_before_publication(
                "usage_observed",
                session_id="s",
                turn_id=tid,
                payload=record("usage_observed", tid=tid, meter_id="padding", observation=index + 1).payload
                | {"usage": {}, "source": {}},
            )
        )
    rows.append(
        make_record_before_publication(
            "turn_ended",
            session_id="s",
            turn_id=tid,
            payload=record("turn_ended", tid=tid).payload | {"budget": DEFAULT_BUDGET},
        )
    )
    # Publishing the large fixture can take longer than the normal setup lease under CI load.
    append_fixture_records(store, "s", rows, commit_id="capacity-prefix", setup_lease_ttl_ms=600_000)
    assert store.read("s").head_seq == size


def long_prefix(monkeypatch, database):
    original_start = recovery.start

    def start(store, sid="s"):
        original_start(store, sid)
        seed_completed_prefix(store, database)

    monkeypatch.setattr(recovery, "start", start)
    monkeypatch.setattr(
        recovery,
        "records_of",
        lambda store, kind: [r.record for r in read_fixture_state(store, "s")[1] if r.seq > 5000 and r.record.kind == kind],
    )


@pytest.mark.parametrize("cooperative", [False, True])
@pytest.mark.persistent_store
def test_long_log_cancel_5000(store, database, monkeypatch, cooperative, record_property):
    long_prefix(monkeypatch, database)
    original_receive = recovery.receive
    running = None

    def receive(pipe, expected, timeout=12):
        nonlocal running
        value = original_receive(pipe, expected, timeout)
        if expected == "tool_running":
            running = time.monotonic()
        elif expected == "token":
            assert running is not None
            record_property("cancel_latency_s", value - running)
        return value

    monkeypatch.setattr(recovery, "receive", receive)
    recovery.test_heartbeat_cancels_blocked_tool_under_two_seconds(store, database, cooperative)


@pytest.mark.parametrize("idempotent", [False, True])
@pytest.mark.persistent_store
def test_long_log_zombie_5000(store, database, monkeypatch, idempotent):
    long_prefix(monkeypatch, database)
    recovery.test_zombie_after_admission_is_fenced_but_external_call_can_still_happen(store, database, idempotent)


def test_prefix_cache_rollback_conflict_epoch_and_disposal(store, database):
    """A cache may observe provisional rows; binding checks must discard that branch."""
    from vv_agent.session.reducer import TransitionError, fold
    from vv_agent.session.store import LeaseLost

    create_fixture_session(store, "s")

    class Rollback(Exception):
        pass

    with pytest.raises(Rollback), store._transaction():
        append_fixture_records(store, "s", (turn_record(store, database),), commit_id="rolled-back")
        assert read_fixture_state(store, "s")[0].active_turn_id == "t"
        raise Rollback
    assert read_fixture_state(store, "s")[0].active_turn_id is None
    append_fixture_records(store, "s", (turn_record(store, database, "actual"),), commit_id="actual")
    lease = store.acquire("s", owner="cache-test", ttl_ms=15000)
    assert lease is not None
    with pytest.raises(TransitionError), store.atomic() as tx:
        tx.append(
            "s",
            lease=lease,
            expected_seq=2,
            commit_id="bad",
            records=(record("op_started", tid="actual", epoch=lease.epoch),),
        )
    state, rows, _ = read_fixture_state(store, "s")
    assert state == fold(r.record for r in rows)
    store._fold_cache = None
    assert read_fixture_state(store, "s")[0] == state
    store.release(lease)
    replacement = store.acquire("s", owner="replacement", ttl_ms=15000)
    assert replacement is not None and replacement.epoch > lease.epoch
    with pytest.raises(LeaseLost), store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=2, commit_id="zombie", records=())
    store.release(replacement)
    with open_store(database) as other:
        other = wrap_fixture_store(other)
        turn_ended = make_record_before_publication(
            "turn_ended",
            session_id="s",
            turn_id="actual",
            payload=record("turn_ended", tid="actual").payload | {"budget": DEFAULT_BUDGET},
        )
        append_fixture_records(other, "s", (turn_ended,), commit_id="tail")
    terminal = read_fixture_state(store, "s")
    store._fold_cache = None
    assert read_fixture_state(store, "s") == terminal
    assert terminal[0].active_turn_id is None


@pytest.mark.postgres
def test_prefix_cache_external_rollback_and_same_head_replacement(store, database):
    """An outer host rollback bypasses the store; a different same-length branch must miss."""
    create_fixture_session(store, "s")

    class Rollback(Exception):
        pass

    with pytest.raises(Rollback), store.connection.transaction():
        append_fixture_records(store, "s", (turn_record(store, database),), commit_id="provisional")
        raise Rollback
    cached = store._fold_cache
    # The current SQL host may either retain the provisional prefix or discard it
    # on rollback; a retained prefix must contain only the rows that were visible.
    assert cached is None or len(cached.records) == 2
    with open_store(database) as other:
        other = wrap_fixture_store(other)
        append_fixture_records(other, "s", (turn_record(other, database, "different"),), commit_id="different")
    assert read_fixture_state(store, "s")[0].active_turn_id == "different"


def test_read_state_and_append_payloads_cannot_poison_prefix(store, database):
    create_fixture_session(store, "s")
    start = turn_record(store, "unused.sqlite")
    append_fixture_records(store, "s", (start,), commit_id="start")
    start.payload["handler_version"] = "caller mutation"
    state, rows, _ = read_cold_state(database)
    state.turns["t"].cancelled = True
    rows[-1].record.payload["handler_version"] = "reader mutation"
    clean, _, _ = read_cold_state(database)
    assert not clean.turns["t"].cancelled
    assert clean.turns["t"].start.payload["handler_version"] == "1"
    turn_ended = make_record_before_publication(
        "turn_ended",
        session_id="s",
        turn_id="t",
        payload=record("turn_ended").payload | {"budget": DEFAULT_BUDGET},
    )
    append_fixture_records(store, "s", (turn_ended,), commit_id="end")
    store._fold_cache = None
    assert read_fixture_state(store, "s")[0].active_turn_id is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 2**60, "\ud800"])
def test_read_write_still_reject_non_ijson(value):
    from dataclasses import replace

    from vv_agent.session.records import Record, RecordError

    rec = replace(record("usage_observed"), payload=record("usage_observed").payload | {"usage": {"value": value}})
    with pytest.raises(RecordError):
        rec.encode()
    import json

    with pytest.raises(RecordError):
        Record.parse(json.dumps(rec.to_dict()).encode())


def test_drive_survives_cache_disposal_mid_run(store, database):
    from vv_agent.tools.function import function_tool
    from vv_agent.types import ToolCall

    effects = []

    @function_tool
    def effect() -> str:
        effects.append(1)
        return "ok"

    def hook(point, _record):
        if point == "after_commit":
            store._fold_cache = None

    recovery.start(store)
    drive(
        store,
        "s",
        runtime=runtime(
            store, database, [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")], [effect], hook=hook
        ),
    )
    assert effects == [1]
    assert read_fixture_state(store, "s")[0].active_turn_id is None
    assert next(r.record for r in read_fixture_state(store, "s")[1] if r.record.kind == "turn_ended").payload["result"] == "done"


def test_cached_records_match_stored_json_values(store, database):
    create_fixture_session(store, "s")
    rt = runtime(store, "unused.sqlite", [])
    rt.model_binding(rt.llm, rt.resolved)
    task = rt.compile("go", "t")
    task.metadata["nested"] = {"opaque": (1, 2), "one": 1.0}
    definition = rt.definition(task)
    payload = record("turn_started").payload | {
        "definition": definition,
        "definition_digest": digest(definition),
        "budget": DEFAULT_BUDGET,
    }
    turn = make_record_before_publication("turn_started", session_id="s", turn_id="t", payload=payload)
    object.__setattr__(turn, "_task_value", task)
    append_fixture_records(store, "s", (turn,), commit_id="turn")
    warm = read_cold_state(database)
    store._fold_cache = None
    assert warm == read_cold_state(database)
    stored = warm[1][-1].record.payload["definition"]["task"]["metadata"]["nested"]
    assert stored["opaque"] == [1, 2] and type(stored["one"]) is int
