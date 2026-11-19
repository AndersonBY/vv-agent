import pytest

from vv_agent.kernel._content import Principal
from vv_agent.kernel._sql import LedgerIntegrityError, UnsupportedFeatureLevel
from vv_agent.session.export import export_session
from vv_agent.session.records import InboxItem, SessionSpec

from .conftest import open_store
from .helpers import content_ref, item, principal, record


def create(store, session_id: str = "s", *, consumers=(), parent=None):
    parent_session_id, parent_operation_id = parent if parent is not None else (None, None)
    spec = SessionSpec(
        session_id,
        principal(),
        content_ref("workspace"),
        parent_session_id=parent_session_id,
        parent_operation_id=parent_operation_id,
        attributes=content_ref(f"attributes/{session_id}"),
    )
    with store.atomic() as tx:
        return tx.create(spec, consumers=consumers)


def acquire(store, session_id: str = "s"):
    store._rows(
        "UPDATE sk_session SET next_drive_ms=%s WHERE tenant_id=%s AND session_id=%s",
        (store._now(), store.tenant_id, session_id),
    )
    lease = store.acquire(session_id, owner=f"worker/{session_id}", ttl_ms=10_000)
    assert lease is not None
    return lease


def append(store, lease, records, commit_id: str):
    with store.atomic() as tx:
        return tx.append(
            lease.session_id,
            lease=lease,
            expected_seq=store.read(lease.session_id).head_seq,
            commit_id=commit_id,
            records=tuple(records),
        )


def set_write_level(store, level: int) -> None:
    store._rows("UPDATE sk_settings SET write_level=%s WHERE singleton=1", (level,))


def session_level(store, session_id: str = "s") -> int:
    return store._rows(
        "SELECT level FROM sk_session WHERE tenant_id=%s AND session_id=%s",
        (store.tenant_id, session_id),
    )[0][0]


def level_two_turn(session_id: str = "s"):
    return record(
        "turn_started",
        sid=session_id,
        requirements={"level": 2, "handler_version": "1", "capabilities": []},
    )


def test_write_level_is_atomic_and_session_level_is_monotonic(store):
    create(store)
    lease = acquire(store)
    high = level_two_turn()

    with pytest.raises(UnsupportedFeatureLevel, match="write level 1"):
        append(store, lease, [high], "too-high")

    assert store._rows("SELECT count(*) FROM sk_record")[0][0] == 1
    assert store._rows("SELECT count(*) FROM sk_commit")[0][0] == 1
    assert session_level(store) == 1

    set_write_level(store, 2)
    with pytest.raises(UnsupportedFeatureLevel, match="session feature level 2 exceeds worker feature level 1"):
        append(store, lease, [high], "worker-not-upgraded")

    store.configure_worker(level=2)
    append(store, lease, [high], "promote")
    assert session_level(store) == 2

    append(store, lease, [record("turn_ended")], "finish")
    assert session_level(store) == 2
    store.release(lease)

    set_write_level(store, 1)
    store.configure_worker(level=1)
    assert store.acquire("s", owner="old", ttl_ms=10_000) is None
    store.configure_worker(level=2)
    low = acquire(store)
    with pytest.raises(UnsupportedFeatureLevel, match="feature level 2"):
        append(store, low, [record("usage_observed")], "lower-ceiling")
    assert session_level(store) == 2
    store.release(low)


def test_held_lease_renews_through_level_rise_until_unsafe_write(store):
    set_write_level(store, 2)
    create(store, consumers=("host",))
    with store.atomic() as tx:
        tx.push("s", item("before-rise", actor=principal("test")))
    store.configure_worker(level=1)
    lease = acquire(store)
    store._rows(
        "UPDATE sk_session SET level=2 WHERE tenant_id=%s AND session_id='s'",
        (store.tenant_id,),
    )
    counts = tuple(store._rows(f"SELECT count(*) FROM {table}")[0][0] for table in ("sk_record", "sk_commit", "sk_inbox"))
    mismatch = "session feature level 2 exceeds worker feature level 1"

    poll = store.renew(lease, ttl_ms=10_000)
    assert poll.lease == lease.__class__(
        lease.session_id,
        lease.owner,
        lease.epoch,
        poll.lease.expires_at_ms,
        lease.tenant_id,
    )
    assert poll.db_now_ms < poll.lease.expires_at_ms
    with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
        append(store, lease, [record("turn_ended")], "stale-writer")
    with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
        store.next_drive_delay_ms("s", lease=lease)
    with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
        store.defer_idle_drive(lease, poll_ms=1_000)
    with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
        store.defer_drive(lease, retry_after_ms=1_000)
    store.defer_projection("s", "host", retry_after_ms=1_000)

    assert tuple(store._rows(f"SELECT count(*) FROM {table}")[0][0] for table in ("sk_record", "sk_commit", "sk_inbox")) == counts
    assert store._rows("SELECT drive_retry_at_ms FROM sk_session WHERE session_id='s'")[0][0] == 0
    assert store._rows("SELECT project_retry_at_ms FROM sk_consumer WHERE session_id='s'")[0][0] > 0
    assert store.release(lease)

    store.configure_worker(level=2)
    store._rows("UPDATE sk_session SET level=1 WHERE tenant_id=%s AND session_id='s'", (store.tenant_id,))
    high = store.acquire("s", owner="upgraded", ttl_ms=10_000)
    assert high is not None
    store.release(high)


def test_high_worker_accepts_low_session(store):
    store.configure_worker(level=2)
    create(store)
    assert session_level(store) == 1
    lease = acquire(store)
    store.release(lease)


def test_host_push_and_commit_replay_do_not_require_worker_execution(store):
    set_write_level(store, 2)
    create(store, consumers=("host",))
    store.configure_worker(level=2)
    lease = acquire(store)
    high = level_two_turn()
    append(store, lease, [high], "promote")
    store.release(lease)
    store.configure_worker(level=1)

    with store.atomic() as tx:
        replayed_commit = tx.append(
            "s",
            lease=lease,
            expected_seq=1,
            commit_id="promote",
            records=(high,),
        )
    assert replayed_commit.replayed

    host_input = item("host-input")
    with store.atomic() as tx:
        first = tx.push("s", host_input)
    with store.atomic() as tx:
        replay = tx.push("s", host_input)
    assert not first.replayed
    assert replay.replayed
    assert replay.input_seq == first.input_seq
    assert session_level(store) == 2


def test_incompatible_executor_skips_drive_while_projection_uses_reader_level(store):
    create(store, consumers=("host",))
    store.configure_worker(level=2)
    lease = acquire(store)
    set_write_level(store, 2)
    append(store, lease, [level_two_turn()], "promote")
    store.release(lease)

    store.configure_worker(level=1)
    assert store.acquire("s", owner="old", ttl_ms=10_000) is None
    assert not store.is_runnable("s")
    claimed = store.claim_runnable(owner="projection", ttl_ms=10_000, limit=10)
    assert [(work.work.kind, work.lease) for work in claimed] == [("project", None)]
    store.defer_projection("s", "host", retry_after_ms=1_000)
    with store.atomic() as tx:
        batch = tx.consumer_batch("s", "host")
        assert batch is not None and batch.through_seq == 2
        tx.ack(batch)

    store.configure_worker(level=2)
    high = store.acquire("s", owner="new", ttl_ms=10_000)
    assert high is not None
    store.release(high)


def test_projection_ack_rechecks_level_inside_transaction(store):
    set_write_level(store, 2)
    create(store, consumers=("host",))
    store.configure_worker(level=2)
    lease = acquire(store)
    append(store, lease, [level_two_turn()], "promote")
    store.release(lease)

    with store.atomic() as tx:
        batch = tx.consumer_batch("s", "host")
        assert batch is not None
        store._rows(
            "UPDATE sk_session SET level=3 WHERE tenant_id=%s AND session_id='s'",
            (store.tenant_id,),
        )
        with pytest.raises(UnsupportedFeatureLevel, match="session feature level 3 exceeds reader feature level 2"):
            tx.ack(batch)

    assert store._rows("SELECT last_seq FROM sk_consumer WHERE session_id='s' AND consumer='host'")[0][0] == 0


def test_child_result_raises_parent_level_atomically(store):
    set_write_level(store, 2)
    create(store, "parent")
    create(store, "child", parent=("parent", "parent/op"))

    store.configure_worker(level=2)
    child_lease = acquire(store, "child")
    high = level_two_turn("child")
    append(store, child_lease, [high], "child-promote")
    store.release(child_lease)
    assert session_level(store, "child") == 2
    assert session_level(store, "parent") == 1

    child_result = InboxItem(
        "child/child/turn/ended",
        "child_result",
        {
            "session_id": "child",
            "turn_id": "child/turn/start",
            "operation_id": "parent/op",
            "attempt": 1,
            "result": None,
            "status": "completed",
            "terminal_seq": 2,
            "terminal_digest": high.digest,
        },
        target_turn_id="parent/turn",
        generation=1,
        actor=principal("child-delivery", "service"),
    )
    with store.atomic() as tx:
        tx.push("parent", child_result)
    assert session_level(store, "parent") == 2

    set_write_level(store, 1)
    with (
        pytest.raises(UnsupportedFeatureLevel, match="feature level 2"),
        store.atomic() as tx,
    ):
        tx.push("parent", item("low-again"))
    assert (
        store._rows(
            "SELECT count(*) FROM sk_inbox WHERE tenant_id=%s AND session_id='parent'",
            (store.tenant_id,),
        )[0][0]
        == 1
    )


def test_understated_level_cache_is_rejected_on_rebuild(store):
    set_write_level(store, 2)
    create(store)
    lease = acquire(store)
    store.configure_worker(level=2)
    append(store, lease, [level_two_turn()], "promote")
    store.release(lease)

    store._rows(
        "UPDATE sk_session SET level=1 WHERE tenant_id=%s AND session_id='s'",
        (store.tenant_id,),
    )
    store.configure_worker(level=2)
    with pytest.raises(LedgerIntegrityError, match="understated"):
        store.read_state("s")


def test_default_level_one_remains_runnable_and_acquirable(store):
    create(store, consumers=("host",))
    with store.atomic() as tx:
        tx.push("s", item("first"))

    assert session_level(store) == 1
    assert {work.kind for work in store.list_runnable()} >= {"drive", "project"}
    lease = acquire(store)
    assert lease is not None
    store.release(lease)


@pytest.mark.persistent_store
@pytest.mark.b8_unconfigured_storage
def test_unconfigured_store_rejects_writes_and_strong_reads(store, database):
    create(store, consumers=("host",))
    lease = acquire(store)
    chain = store._one(
        "SELECT chain_digest FROM sk_record WHERE tenant_id=%s AND session_id='s' AND seq=1",
        (store.tenant_id,),
    )[0]
    counts = tuple(
        store._rows(f"SELECT count(*) FROM {table}")[0][0] for table in ("sk_session", "sk_record", "sk_commit", "sk_inbox")
    )
    actor = Principal.from_dict(principal("forget"))
    message = "explicit signed signer or allow_unsigned hash-only"

    with open_store(database, allow_unsigned=False) as unconfigured:
        assert unconfigured._ledger_integrity_mode() == "unconfigured"
        assert unconfigured._one("SELECT storage_major,write_level FROM sk_settings WHERE singleton=1") == ("S1", 1)
        with pytest.raises(LedgerIntegrityError, match=message):
            unconfigured.read("s")

        with pytest.raises(LedgerIntegrityError, match=message):
            create(unconfigured, "blocked-create")
        with pytest.raises(LedgerIntegrityError, match=message), unconfigured.atomic() as tx:
            tx.push("s", item("blocked-push"))
        with pytest.raises(LedgerIntegrityError, match=message), unconfigured.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=1,
                commit_id="blocked-append",
                records=(record("usage_observed"),),
            )
        with pytest.raises(LedgerIntegrityError, match=message):
            unconfigured.acquire("s", owner="blocked-lease", ttl_ms=10_000)
        with pytest.raises(LedgerIntegrityError, match=message):
            unconfigured.begin_forget(
                "s",
                actor,
                bound_signer=None,
                verifier=None,
                allow_unsigned=True,
            )
        with pytest.raises(LedgerIntegrityError, match=message):
            unconfigured.complete_forget("s")
        with pytest.raises(LedgerIntegrityError, match=message):
            unconfigured.begin_purge(
                "s",
                actor,
                bound_signer=None,
                verifier=None,
                allow_unsigned=True,
            )
        with pytest.raises(LedgerIntegrityError, match=message):
            unconfigured.complete_purge("s")
        with pytest.raises(LedgerIntegrityError, match=message):
            unconfigured.read_state("s", expected_head=(1, chain))
        with pytest.raises(LedgerIntegrityError, match=message):
            unconfigured.verify_ledger("s")
        with pytest.raises(LedgerIntegrityError, match=message):
            export_session(unconfigured, "s")

    after = tuple(
        store._rows(f"SELECT count(*) FROM {table}")[0][0] for table in ("sk_session", "sk_record", "sk_commit", "sk_inbox")
    )
    assert after == counts
    assert store._ledger_integrity_mode() == "hash-only"
    assert export_session(store, "s").header.head_seq == 1
