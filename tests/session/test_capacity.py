"""M6 capacity bounds use the M5 PostgreSQL history and default lease settings."""

import time
from contextlib import suppress

import pytest

from vv_agent.session.kernel import drive, read_state
from vv_agent.session.records import digest
from vv_agent.types import LLMResponse, ToolExecutionResult

from . import test_recovery_matrix as recovery
from .conftest import open_store
from .helpers import record


def seed_history(store, database, size):
    recovery.start(store)

    class ReceiptCut(BaseException):
        pass

    def hook(point, value):
        if point == "after_commit" and value.kind == "op_completed":
            raise ReceiptCut

    with suppress(ReceiptCut):
        drive(store, "s", runtime=recovery.runtime(database, [LLMResponse("retained answer")], hook=hook))
    state, rows, _ = read_state(store, "s")
    tid = state.active_turn_id
    assert tid is not None and state.phase == "active"
    lease = store.acquire("s", owner="benchmark", ttl_ms=600000)
    assert lease is not None
    pending = []
    # A bounded 1 KiB tool receipt; no growing model-request snapshots. This is a lower bound.
    while len(rows) + len(pending) + 3 <= size:
        oid = f"{tid}/benchmark/tool/{len(pending)}"
        request = {"id": oid, "name": "benchmark", "arguments": {}}
        result = ToolExecutionResult(tool_call_id=oid, content="x" * 1024).to_dict()
        pending.extend(
            [
                record(
                    "op_planned", tid=tid, oid=oid, op_kind="tool", purpose=None, request=request, request_digest=digest(request)
                ),
                record("op_started", tid=tid, oid=oid, epoch=lease.epoch),
                record(
                    "op_completed", tid=tid, oid=oid, request_digest=digest(request), result=result, result_digest=digest(result)
                ),
            ]
        )
    for index in range(size - len(rows) - len(pending)):
        pending.append(record("usage_observed", tid=tid, meter_id="padding", observation=index + 1))
    with store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=len(rows), commit_id="benchmark-seed", records=tuple(pending))
    store.release(lease)
    state, rows, _ = read_state(store, "s")
    assert len(rows) == size and state.phase == "active"
    return tid, tuple(r.record for r in rows)


def test_capacity_2000_records(store, database):
    tid, _ = seed_history(store, database, 2000)
    with open_store(database) as cold:
        before = time.monotonic()
        drive(cold, "s", runtime=recovery.runtime(database, []))
        elapsed = time.monotonic() - before
    assert elapsed <= 2, f"2000-record cold drive took {elapsed:.3f}s (limit 2s)"
    assert read_state(store, "s")[0].active_turn_id is None
    assert tid is not None


def seed_completed_prefix(store, database, size=5000):
    """A completed prior turn; current-turn cancellation/zombie assertions stay intact."""
    lease = store.acquire("s", owner="capacity-seed", ttl_ms=15000)
    assert lease is not None
    tid = "previous"
    rt = recovery.runtime(database, [])
    definition = rt.definition(rt.compile("prior turn", tid))
    rows = [record("turn_started", tid=tid, definition=definition, definition_digest=digest(definition))]
    while len(rows) + 4 <= size - 1:
        oid = f"previous/tool/{len(rows)}"
        request = {"id": oid, "name": "benchmark", "arguments": {}}
        result = ToolExecutionResult(tool_call_id=oid, content="x" * 1024).to_dict()
        rows.extend(
            [
                record(
                    "op_planned", tid=tid, oid=oid, op_kind="tool", purpose=None, request=request, request_digest=digest(request)
                ),
                record("op_started", tid=tid, oid=oid, epoch=lease.epoch),
                record(
                    "op_completed", tid=tid, oid=oid, request_digest=digest(request), result=result, result_digest=digest(result)
                ),
            ]
        )
    for index in range(size - 2 - len(rows)):
        rows.append(record("usage_observed", tid=tid, meter_id="padding", observation=index + 1))
    rows.append(record("turn_ended", tid=tid))
    with store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=1, commit_id="capacity-prefix", records=tuple(rows))
    assert store.read("s").head_seq == size
    store.release(lease)


def long_prefix(monkeypatch, database):
    original_start = recovery.start

    def start(store, sid="s"):
        original_start(store, sid)
        seed_completed_prefix(store, database)

    monkeypatch.setattr(recovery, "start", start)
    monkeypatch.setattr(
        recovery,
        "records_of",
        lambda store, kind: [r.record for r in read_state(store, "s")[1] if r.seq > 5000 and r.record.kind == kind],
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
    from vv_agent.session.records import SessionSpec
    from vv_agent.session.reducer import TransitionError, fold
    from vv_agent.session.store import LeaseLost

    with store.atomic() as tx:
        tx.create(SessionSpec("s", "p", "w"), consumers=())
    lease = store.acquire("s", owner="cache-test", ttl_ms=15000)
    assert lease is not None

    class Rollback(Exception):
        pass

    with pytest.raises(Rollback), store._transaction():
        with store.atomic() as tx:
            tx.append("s", lease=lease, expected_seq=1, commit_id="rolled-back", records=(record("turn_started"),))
        assert read_state(store, "s")[0].active_turn_id == "t"
        raise Rollback
    assert read_state(store, "s")[0].active_turn_id is None
    with store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=1, commit_id="actual", records=(record("turn_started", tid="actual"),))
    with pytest.raises(TransitionError), store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=2, commit_id="bad", records=(record("op_started", tid="actual"),))
    state, rows, _ = read_state(store, "s")
    assert state == fold(r.record for r in rows)
    store._fold_cache = None
    assert read_state(store, "s")[0] == state
    store.release(lease)
    replacement = store.acquire("s", owner="replacement", ttl_ms=15000)
    assert replacement is not None and replacement.epoch > lease.epoch
    with pytest.raises(LeaseLost), store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=2, commit_id="zombie", records=())
    with open_store(database) as other, other.atomic() as tx:
        tx.append("s", lease=replacement, expected_seq=2, commit_id="tail", records=(record("turn_ended", tid="actual"),))
    terminal = read_state(store, "s")
    store._fold_cache = None
    assert read_state(store, "s") == terminal
    assert terminal[0].active_turn_id is None


@pytest.mark.postgres
def test_prefix_cache_external_rollback_and_same_head_replacement(store, database):
    """An outer host rollback bypasses the store; a different same-length branch must miss."""
    from vv_agent.session.records import SessionSpec

    with store.atomic() as tx:
        tx.create(SessionSpec("s", "p", "w"), consumers=())
    lease = store.acquire("s", owner="host", ttl_ms=15000)
    assert lease is not None

    class Rollback(Exception):
        pass

    with pytest.raises(Rollback), store.connection.transaction():
        with store.atomic() as tx:
            tx.append("s", lease=lease, expected_seq=1, commit_id="provisional", records=(record("turn_started"),))
        raise Rollback
    assert store._fold_cache is not None and len(store._fold_cache.records) == 2
    with open_store(database) as other, other.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=1, commit_id="different", records=(record("turn_started", tid="different"),))
    assert read_state(store, "s")[0].active_turn_id == "different"


def test_read_state_and_append_payloads_cannot_poison_prefix(store):
    from vv_agent.session.records import SessionSpec

    with store.atomic() as tx:
        tx.create(SessionSpec("s", "p", "w"), consumers=())
    lease = store.acquire("s", owner="test", ttl_ms=15000)
    assert lease is not None
    start = record("turn_started")
    with store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=1, commit_id="start", records=(start,))
    start.payload["handler_version"] = "caller mutation"
    state, rows, _ = read_state(store, "s")
    state.turns["t"].cancelled = True
    rows[-1].record.payload["handler_version"] = "reader mutation"
    clean, _, _ = read_state(store, "s")
    assert not clean.turns["t"].cancelled
    assert clean.turns["t"].start.payload["handler_version"] == "1"
    with store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=2, commit_id="end", records=(record("turn_ended"),))
    store._fold_cache = None
    assert read_state(store, "s")[0].active_turn_id is None


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
        runtime=recovery.runtime(
            database, [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")], [effect], hook=hook
        ),
    )
    assert effects == [1]
    assert read_state(store, "s")[0].active_turn_id is None
    assert recovery.records_of(store, "turn_ended")[0].payload["result"] == "done"


def test_cached_records_match_stored_json_values(store):
    from dataclasses import replace

    from vv_agent.session.records import SessionSpec

    with store.atomic() as tx:
        tx.create(SessionSpec("s", "p", "w"), consumers=())
    lease = store.acquire("s", owner="test", ttl_ms=15000)
    assert lease is not None
    usage = replace(record("usage_observed", usage={"opaque": (1, 2), "one": 1.0}), turn_id=None)
    with store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=1, commit_id="usage", records=(usage,))
    warm = read_state(store, "s")
    store._fold_cache = None
    assert warm == read_state(store, "s")
    stored = warm[1][-1].record.payload["usage"]
    assert stored["opaque"] == [1, 2] and type(stored["one"]) is int
