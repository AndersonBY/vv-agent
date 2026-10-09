from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from time import sleep

import pytest

from vv_agent.session.postgres import join_transaction
from vv_agent.session.records import SessionSpec, digest
from vv_agent.session.reducer import TransitionError, fold
from vv_agent.session.store import Conflict, LeaseLost, SequenceConflict

from .conftest import open_store
from .helpers import item, record


def create(store, sid="s", consumers=()):
    with store.atomic() as tx:
        return tx.create(SessionSpec(sid, "p", "w"), consumers=consumers)


def lease_for(store, sid="s"):
    lease = store.acquire(sid, owner="worker", ttl_ms=10000)
    assert lease is not None
    return lease


def append(store, lease, records, commit="c", expected=None, **kw):
    with store.atomic() as tx:
        return tx.append(
            lease.session_id,
            lease=lease,
            expected_seq=store.read(lease.session_id).head_seq if expected is None else expected,
            commit_id=commit,
            records=tuple(records),
            **kw,
        )


def counts(store):
    return (
        *tuple(
            store.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("sk_record", "sk_inbox", "sk_consumer")
        ),
        store.read("s").head_seq,
    )


def expire(store):
    store.connection.execute("UPDATE sk_session SET lease_until_ms = 0 WHERE session_id = 's'")


def test_create_idempotency(store):
    first = create(store, consumers=("host",))
    assert replace(create(store, consumers=("host",)), replayed=False) == first
    before = counts(store)
    with pytest.raises(Conflict), store.atomic() as tx:
        tx.create(SessionSpec("s", "different", "w"), consumers=("host",))
    with pytest.raises(Conflict):
        create(store, consumers=("other",))
    assert counts(store) == before == (1, 0, 1, 1)


def test_commit_replay_conflict_overlap_and_lost_ack(store):
    create(store)
    lease = lease_for(store)
    rec = record("turn_started")
    first = append(store, lease, [rec], expected=1)
    # The application discards the first receipt; only the durable store can recover it.
    replay = append(store, lease, [rec], expected=1)
    assert replay.replayed and replace(replay, replayed=False) == first
    before = counts(store)
    with pytest.raises(Conflict):
        append(store, lease, [record("turn_started", handler_version="other")], expected=1)
    assert counts(store) == before
    with pytest.raises(Conflict):
        append(store, lease, [record("op_planned"), record("turn_started", handler_version="bad")], commit="bad")
    assert counts(store) == before
    mixed = append(store, lease, [rec, record("op_planned")], commit="mixed")
    assert mixed.record_sequences == ((rec.record_id, 2), (record("op_planned").record_id, 3))
    assert replace(append(store, lease, [rec, record("op_planned")], commit="mixed", expected=2), replayed=False) == mixed
    overlap = append(store, lease, [rec], commit="overlap")
    empty = append(store, lease, [], commit="empty")
    append(store, lease, [record("op_started")], commit="later")
    assert replace(append(store, lease, [rec], commit="overlap", expected=0), replayed=False) == overlap
    assert replace(append(store, lease, [], commit="empty", expected=0), replayed=False) == empty
    assert counts(store) == (4, 0, 0, 4)


def test_cas_epoch_expiry_and_old_lease_replay(store):
    create(store)
    old = lease_for(store)
    first = append(store, old, [record("turn_started")], expected=1)
    with pytest.raises(SequenceConflict):
        append(store, old, [record("op_planned")], commit="stale", expected=1)
    expire(store)
    with pytest.raises(LeaseLost):
        store.renew(old, ttl_ms=10000)
    with pytest.raises(LeaseLost):
        append(store, old, [record("op_planned")], commit="expired")
    new = lease_for(store)
    assert new.epoch == old.epoch + 1
    assert not store.release(old)
    assert replace(append(store, old, [record("turn_started")], expected=1), replayed=False) == first
    with pytest.raises(LeaseLost):
        append(store, old, [record("op_planned")], commit="new")
    poll = store.renew(new, ttl_ms=10000)
    assert poll.lease.epoch == new.epoch and poll.db_now_ms < poll.lease.expires_at_ms
    assert store.release(poll.lease)
    assert counts(store) == (2, 0, 0, 2)


def test_push_consume_atomic_and_inbox_cas(store):
    create(store)
    incoming = item()
    with store.atomic() as tx:
        first = tx.push("s", incoming)
    with store.atomic() as tx:
        replay = tx.push("s", incoming)
        assert replay.replayed and replace(replay, replayed=False) == first
        with pytest.raises(Conflict):
            tx.push("s", replace(incoming, payload={"content": "other"}))
    assert first.input_seq == 1
    lease = lease_for(store)
    applied = record("input_applied")
    before = counts(store)
    with pytest.raises(SequenceConflict):
        append(store, lease, [applied, record("turn_started", input_ids=["i"])], consume_input_ids=("i",), expected_inbox_seq=0)
    with pytest.raises(TransitionError):
        append(store, lease, [applied, record("op_started")], consume_input_ids=("i",))
    assert counts(store) == before and len(store.peek_inbox("s")) == 1
    with pytest.raises(Conflict):
        append(store, lease, [], consume_input_ids=("i",))
    with pytest.raises(Conflict):
        append(store, lease, [applied])
    receipt = append(
        store, lease, [applied, record("turn_started", input_ids=["i"])], consume_input_ids=("i",), expected_inbox_seq=1
    )
    assert receipt.head_seq == 3 and store.peek_inbox("s") == ()
    assert store.connection.execute("SELECT consumed_seq FROM sk_inbox").fetchone()[0] == 2
    with store.atomic() as tx:
        assert replace(tx.push("s", incoming), replayed=False) == first
    with pytest.raises(Conflict):
        append(store, lease, [applied, record("turn_started", input_ids=["i"])], consume_input_ids=(), expected_inbox_seq=1)


@pytest.mark.postgres
def test_consumer_commit_boundary_host_atomicity_and_forged_ack(store, database):
    import psycopg

    create(store, consumers=("host",))
    lease = lease_for(store)
    append(store, lease, [record("turn_started"), record("op_planned"), record("op_started")])
    store.connection.execute("CREATE TABLE host_effect (id integer PRIMARY KEY)")
    with psycopg.connect(database, autocommit=True) as conn:
        with pytest.raises(RuntimeError), conn.transaction():
            tx = join_transaction(conn)
            first = tx.consumer_batch("s", "host", limit=1)
            assert first is not None and first.through_seq == 1
            tx.ack(first)
            batch = tx.consumer_batch("s", "host", limit=1)
            assert batch is not None and len(batch.records) == 3 and batch.through_seq == 4
            conn.execute("INSERT INTO host_effect VALUES (1)")
            tx.ack(batch)
            raise RuntimeError("host rollback")
        assert conn.execute("SELECT last_seq FROM sk_consumer").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM host_effect").fetchone()[0] == 0
        with conn.transaction():
            tx = join_transaction(conn)
            batch = tx.consumer_batch("s", "host", limit=100)
            assert batch is not None
            with pytest.raises(Conflict):
                tx.ack(replace(batch, through_seq=3))
            conn.execute("INSERT INTO host_effect VALUES (1)")
            tx.ack(batch)
        assert conn.execute("SELECT last_seq FROM sk_consumer").fetchone()[0] == 4
        assert conn.execute("SELECT count(*) FROM host_effect").fetchone()[0] == 1
        with pytest.raises(RuntimeError):
            join_transaction(conn)
    with store.atomic() as tx:
        assert tx.consumer_batch("s", "host") is None


def test_runnable_empty_inbox_future_user_wait_and_pagination(store):
    for sid in ("a", "b", "c", "d", "e"):
        create(store, sid, consumers=("host",))
    with store.atomic() as tx:
        tx.push("a", item())
    b = lease_for(store, "b")
    append(store, b, [record("turn_started", sid="b"), record("op_planned", sid="b"), record("op_started", sid="b")])
    store.connection.execute("UPDATE sk_session SET lease_until_ms=0 WHERE session_id='b'")
    c = lease_for(store, "c")
    future = store._now() + 60000
    append(store, c, [record("turn_started", sid="c"), record("op_planned", sid="c", not_before_ms=future)])
    store.release(c)
    d = lease_for(store, "d")
    append(
        store,
        d,
        [
            record("turn_started", sid="d"),
            record("op_planned", sid="d", op_kind="interaction", purpose=None),
            record(
                "op_parked", sid="d", phase="before_dispatch", handle={"kind": "user", "interaction_id": "q", "question": "?"}
            ),
        ],
    )
    store.release(d)
    all_work = store.list_runnable(limit=100)
    assert {w.session_id for w in all_work if w.kind == "drive"} == {"a", "b"}
    assert {w.session_id for w in all_work if w.kind == "project"} == {"a", "b", "c", "d", "e"}
    pages, after = [], None
    while page := store.list_runnable(limit=2, after=after):
        pages.extend(page)
        after = page[-1].cursor
    assert tuple(pages) == all_work
    with store.atomic() as tx:
        tx.push("d", item())
    assert "d" in {w.session_id for w in store.list_runnable() if w.kind == "drive"}


def test_due_time_reached_on_database_clock(store):
    create(store)
    lease = lease_for(store)
    now = store._now()
    append(store, lease, [record("turn_started"), record("op_planned", not_before_ms=now + 150)])
    store.release(lease)
    assert store.list_runnable() == ()
    sleep(0.16)
    assert [w.kind for w in store.list_runnable()] == ["drive"]


def test_schedule_rebuild_and_read_pages(store):
    create(store, consumers=("host",))
    lease = lease_for(store)
    append(
        store, lease, [record("turn_started"), record("op_planned"), record("op_started"), record("op_unknown", retry_at_ms=123)]
    )
    store.release(lease)
    before = store.list_runnable()
    page = store.read("s")
    state = fold(tuple(r.record for r in page.records))
    assert store.read("s", after_seq=1, through_seq=3, limit=1).records[0].seq == 2
    store.connection.execute("UPDATE sk_session SET phase='idle',next_drive_ms=NULL,active_turn_id=NULL,terminal_seq=0")
    store.rebuild_schedule("s")
    assert store.list_runnable() == before
    assert store.connection.execute("SELECT phase,next_drive_ms,active_turn_id,terminal_seq FROM sk_session").fetchone() == (
        state.phase,
        state.next_drive_ms,
        state.active_turn_id,
        state.terminal_seq,
    )


def test_renew_reports_control_and_future_inbox(store):
    create(store)
    lease = lease_for(store)
    with store.atomic() as tx:
        tx.push("s", item(kind="control", action="suspend"))
        tx.push("s", item("future", available_ms=9999999999999))
    poll = store.renew(lease, ttl_ms=1000)
    assert poll.inbox_seq == 2 and tuple(control.payload["action"] for control in poll.controls) == ("suspend",)
    assert len(store.peek_inbox("s")) == 1
    assert store.peek_inbox("s", through_input_seq=0) == ()


def test_concurrent_acquire_and_twenty_identical_pushes(store, database):
    create(store)
    barrier = Barrier(2)

    def acquire(n):
        with open_store(database) as other:
            barrier.wait(timeout=10)
            return other.acquire("s", owner=f"w{n}", ttl_ms=10000)

    with ThreadPoolExecutor(max_workers=2) as pool:
        leases = list(pool.map(acquire, range(2)))
    assert sum(lease is not None for lease in leases) == 1
    barrier = Barrier(20)

    def push(_):
        with open_store(database) as other:
            barrier.wait(timeout=10)
            with other.atomic() as tx:
                return tx.push("s", item())

    with ThreadPoolExecutor(max_workers=20) as pool:
        receipts = list(pool.map(push, range(20)))
    assert sum(not r.replayed for r in receipts) == 1
    assert len({replace(r, replayed=False) for r in receipts}) == 1
    assert counts(store) == (1, 1, 0, 1)


def test_host_push_rollback_and_illegal_append_no_writes_even_if_caught(store):
    create(store)
    lease = lease_for(store)
    with pytest.raises(RuntimeError), store.atomic() as tx:
        tx.push("s", item())
        raise RuntimeError("rollback")
    with store.atomic() as tx:
        before = counts(store)
        with pytest.raises(TransitionError):
            tx.append("s", lease=lease, expected_seq=1, commit_id="bad", records=(record("turn_started"), record("op_started")))
        assert counts(store) == before
    assert store.read("s").inbox_seq == 0


def test_suspended_and_terminal_schedule(store):
    create(store)
    lease = lease_for(store)
    append(store, lease, [record("turn_started")])
    control = item(kind="control", action="suspend", target_turn_id="t", generation=1)
    with store.atomic() as tx:
        tx.push("s", control)
    applied = record("input_applied", input=control.to_dict(), input_digest=digest(control.to_dict()))
    append(store, lease, [applied], commit="suspend", consume_input_ids=("i",))
    state = fold([r.record for r in store.read("s").records], consumed_inputs=(control,))
    assert state.phase == "suspended" and state.next_drive_ms is None
    store.release(lease)
    assert store.list_runnable() == ()


def test_lock_wait_uses_clock_after_acquiring_row(store, database):
    from threading import Event

    create(store)
    old = store.acquire("s", owner="old", ttl_ms=2000)
    assert old is not None
    entered = Event()

    def waiting_acquire():
        with open_store(database) as other:
            entered.set()
            return other.acquire("s", owner="new", ttl_ms=1000)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with store.atomic():
            store._lock("s")
            # The contender's transaction starts before expiry, but it cannot acquire the row until after expiry.
            store._rows("UPDATE sk_session SET lease_until_ms=%s", (store._now() + 150,))
            future = pool.submit(waiting_acquire)
            assert entered.wait(timeout=5)
            sleep(0.2)
        new = future.result(timeout=5)
    assert new is not None and new.epoch == old.epoch + 1


def test_expired_waiter_cannot_append_after_blocked_row(store, database):
    from threading import Event

    create(store)
    lease = lease_for(store)
    entered = Event()

    def waiting_append():
        with open_store(database) as other:
            entered.set()
            with pytest.raises(LeaseLost):
                append(other, lease, [record("turn_started")], expected=1)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with store.atomic():
            store._lock("s")
            store._rows("UPDATE sk_session SET lease_until_ms=%s", (store._now() + 150,))
            future = pool.submit(waiting_append)
            assert entered.wait(timeout=5)
            sleep(0.2)
        future.result(timeout=5)
    assert counts(store) == (1, 0, 0, 1)


def test_tx_cannot_escape_host_transaction(store):
    create(store, consumers=("host",))
    with store.atomic() as tx:
        batch = tx.consumer_batch("s", "host")
        assert batch is not None
    with store.atomic(), pytest.raises(RuntimeError):
        tx.ack(batch)
    assert store.connection.execute("SELECT last_seq FROM sk_consumer").fetchone()[0] == 0


def test_dispatch_epoch_and_not_before_are_atomic(store):
    create(store)
    lease = lease_for(store)
    before = counts(store)
    with pytest.raises(Conflict), store.atomic() as tx:
        tx.append(
            "s",
            lease=lease,
            expected_seq=1,
            commit_id="wrong-epoch",
            records=(record("turn_started"), record("op_planned"), record("op_started", epoch=lease.epoch + 1)),
        )
    assert counts(store) == before
    with pytest.raises(Conflict), store.atomic() as tx:
        tx.append(
            "s",
            lease=lease,
            expected_seq=1,
            commit_id="future",
            records=(record("turn_started"), record("op_planned", not_before_ms=9999999999999), record("op_started")),
        )
    assert counts(store) == before


def test_replayed_receipt_retains_original_timestamps(store):
    create(store)
    lease = lease_for(store)
    original = append(store, lease, [record("turn_started")])
    rows = store.read("s").records
    assert append(store, lease, [record("turn_started")]).replayed
    assert store.read("s").records == rows
    assert original.head_seq == 2


def test_commit_metadata_retains_references_without_another_record_copy(store):
    import json

    create(store)
    lease = lease_for(store)
    rec = record(
        "turn_started",
        definition={"large_frozen_request": "kept once"},
        definition_digest=digest({"large_frozen_request": "kept once"}),
    )
    append(store, lease, [rec])
    body = store.connection.execute("SELECT body FROM sk_commit WHERE commit_id='c'").fetchone()[0]
    assert json.loads(body) == {"record_ids": [rec.record_id], "consume_input_ids": []}
    commits = store.connection.execute("SELECT count(*) FROM sk_commit").fetchone()[0]
    with pytest.raises(Conflict):
        append(store, lease, [record("turn_started", handler_version="different")], commit="conflict")
    assert store.connection.execute("SELECT count(*) FROM sk_commit").fetchone()[0] == commits


def test_postgres_import_is_optional_and_no_django_dependency():
    import subprocess
    import sys

    script = """
import importlib.abc
import sys
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'psycopg', 'django'}:
            raise AssertionError(f'unexpected import: {fullname}')
sys.meta_path.insert(0, Forbidden())
import vv_agent.session
import vv_agent.session.postgres
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_late_callback_after_generation_replaced_is_consumed_as_audit(store):
    from vv_agent.session.records import InboxItem

    create(store)
    lease = lease_for(store)
    append(
        store,
        lease,
        [
            record("turn_started"),
            record("op_planned"),
            record("op_started"),
            record("turn_ended", unconfirmed_operations=["o"]),
            record("turn_started", tid="t2", generation=2),
        ],
    )
    callback = InboxItem(
        "callback",
        "provider_result",
        {
            "operation_id": "o",
            "attempt": 1,
            "request_digest": record("op_planned").payload["request_digest"],
            "provider_binding": "provider",
            "result": {"ok": True},
            "evidence": ["accepted"],
        },
        target_turn_id="t",
        generation=1,
    )
    with store.atomic() as tx:
        tx.push("s", callback)
    applied = record("input_applied", input=callback.to_dict(), input_digest=digest(callback.to_dict()), target_operation_id="o")
    append(store, lease, [applied, record("op_completed", context="audit")], commit="late", consume_input_ids=("callback",))
    assert store.peek_inbox("s") == ()
    state = store.rebuild_schedule("s")
    assert state.active_turn_id == "t2" and state.operations["o"].selected_attempt is None
    assert state.operations["o"].attempts[1].context == "audit"


def test_lease_poll_preserves_control_target_and_generation(store):
    create(store)
    lease = lease_for(store)
    control = item(kind="control", action="cancel", target_turn_id="old-turn", generation=7)
    with store.atomic() as tx:
        tx.push("s", control)
    assert store.renew(lease, ttl_ms=10000).controls == (control,)


def test_native_consumer_cursor_and_host_write_share_transaction(store):
    create(store, consumers=("host",))
    store.connection.execute("CREATE TABLE host_effect (id integer PRIMARY KEY)")
    with pytest.raises(RuntimeError), store.atomic() as tx:
        batch = tx.consumer_batch("s", "host")
        store.connection.execute("INSERT INTO host_effect VALUES (1)")
        tx.ack(batch)
        raise RuntimeError("kill before commit")
    assert store.connection.execute("SELECT count(*) FROM host_effect").fetchone()[0] == 0
    assert store.connection.execute("SELECT last_seq FROM sk_consumer").fetchone()[0] == 0
    with store.atomic() as tx:
        batch = tx.consumer_batch("s", "host")
        with pytest.raises(Conflict):
            tx.ack(replace(batch))
        store.connection.execute("INSERT INTO host_effect VALUES (1)")
        tx.ack(batch)
    assert store.connection.execute("SELECT last_seq FROM sk_consumer").fetchone()[0] == 1


@pytest.mark.sqlite_file
@pytest.mark.parametrize("version", [0, 2])
def test_sqlite_rejects_missing_or_unknown_database_version(store, database, version):
    store.connection.execute(f"PRAGMA user_version={version}")
    with pytest.raises(ValueError, match="schema version"), open_store(database):
        pass
