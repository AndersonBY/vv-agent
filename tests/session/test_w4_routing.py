"""Contract-41 routing cache, worker compatibility and tenant-bound claim coverage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from threading import Event
from typing import Any, cast
from uuid import uuid4

import pytest

from tests.sim.world import SimClock, SimWorld
from vv_agent.kernel._requirements import SessionRequirements
from vv_agent.kernel._sql import InputAdmissionProofError, LedgerIntegrityError, UnsupportedFeatureLevel
from vv_agent.session import postgres as session_postgres
from vv_agent.session import sqlite as session_sqlite
from vv_agent.session.records import InboxItem, SessionSpec, digest, make_record
from vv_agent.session.store import Conflict, Lease
from vv_agent.session.worker import LocalWorker, WorkerSupport

from .conftest import open_store
from .helpers import content_ref, wrap_fixture_store


def _requirements(*capabilities: str, handler: str = "handler/h2", level: int = 2) -> SessionRequirements:
    return SessionRequirements.normalized(level=level, handler_version=handler, capabilities=capabilities)


def _principal(tenant_id: str) -> dict[str, Any]:
    return {
        "tenant_id": tenant_id,
        "subject_id": f"subject/{tenant_id}",
        "subject_kind": "user",
        "attributes": content_ref(f"principal/{tenant_id}"),
    }


def _create_session(
    store: Any,
    session_id: str,
    *,
    tenant_id: str = "default",
    bootstrap: SessionRequirements | None = None,
    consumers: tuple[str, ...] = (),
) -> Any:
    if bootstrap is not None:
        store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    bound = store.for_tenant(tenant_id)
    with bound.atomic() as tx:
        tx.create(
            SessionSpec(
                session_id,
                _principal(tenant_id),
                content_ref(f"workspace/{tenant_id}/{session_id}"),
                attributes=content_ref(f"attributes/{tenant_id}/{session_id}"),
                feature_level=2 if bootstrap is not None else 1,
                bootstrap=bootstrap,
            ),
            consumers=consumers,
        )
    return bound


def _push(store: Any, session_id: str, item: InboxItem) -> None:
    with store.atomic() as tx:
        tx.push(session_id, item)


def _input(
    input_id: str,
    kind: str,
    requirements: SessionRequirements,
    *,
    tenant_id: str = "default",
) -> InboxItem:
    payload = {"content": content_ref(f"input/{input_id}")}
    if kind == "model_selection":
        payload = {"selection": content_ref(f"selection/{input_id}")}
    return InboxItem(
        input_id,
        kind,
        payload,
        actor=_principal(tenant_id),
        feature_level=2,
        requirements=requirements,
    )


def _level_one_input(input_id: str, kind: str = "user") -> InboxItem:
    return InboxItem(
        input_id,
        kind,
        {"content": content_ref(f"input/{input_id}")},
        actor=_principal("default"),
    )


def _applied_input_record(session_id: str, turn_id: str | None, item: InboxItem, *, disposition: str) -> Any:
    return make_record(
        "input_applied",
        session_id=session_id,
        turn_id=turn_id,
        payload={
            "input": item.to_dict(),
            "input_digest": digest(item.to_dict()),
            "input_proof": None,
            "disposition": disposition,
            "reason": None,
            "target_operation_id": None,
            "target_wait_id": None,
            "position": "before_dispatch",
        },
    )


def _usage_record(session_id: str, meter_id: str, observation: int) -> Any:
    return make_record(
        "usage_observed",
        session_id=session_id,
        payload={
            "meter_id": meter_id,
            "observation": observation,
            "mode": "cumulative",
            "usage": content_ref(f"snapshot/{meter_id}/usage"),
            "source": content_ref(f"snapshot/{meter_id}/source"),
        },
    )


def _acquire(store: Any, session_id: str, owner: str) -> Lease:
    lease = store.acquire(session_id, owner=owner, ttl_ms=30_000)
    assert lease is not None
    return lease


def _append(
    store: Any,
    session_id: str,
    lease: Lease,
    commit_id: str,
    records: tuple[Any, ...],
    *,
    consume_input_ids: tuple[str, ...] = (),
) -> None:
    with store.atomic() as tx:
        tx.append(
            session_id,
            lease=lease,
            expected_seq=store.read(session_id).head_seq,
            commit_id=commit_id,
            records=records,
            consume_input_ids=consume_input_ids,
        )


def _lease_row(store: Any, session_id: str) -> tuple[int, str | None, int | None]:
    raw = store
    return cast(
        tuple[int, str | None, int | None],
        raw._one(
            "SELECT lease_epoch,lease_owner,lease_until_ms FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (store.tenant_id, session_id),
        ),
    )


def _cache(store: Any, session_id: str) -> tuple[int, str | None, tuple[str, ...], str | None]:
    raw = store
    row = raw._one(
        "SELECT head_seq,inbox_seq,lease_epoch,lease_owner,lease_until_ms,drive_retry_at_ms,level,"
        "deletion_state,handler_version,required_capabilities,active_turn_id "
        "FROM sk_session WHERE tenant_id=%s AND session_id=%s",
        (store.tenant_id, session_id),
    )
    cache = raw._routing_cache_from_row(row)
    return cache.requirements.level, cache.requirements.handler_version, cache.requirements.capabilities, cache.active_turn_id


class _HashSigner:
    key_id = "routing-test-key"

    def bind_key(self):
        return self

    def sign(self, message: bytes) -> bytes:
        return sha256(message).digest()

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        return key_id == self.key_id and signature == sha256(message).digest()


@pytest.mark.persistent_store
def test_heterogeneous_workers_filter_candidates_without_advancing_epoch(store) -> None:
    requirements = _requirements("family:agent", "model:old", "model:new", "tool:a")
    bound = _create_session(store, "heterogeneous", bootstrap=requirements)
    _push(bound, "heterogeneous", _input("work", "user", requirements))

    incompatible = (
        (1, ("handler/h2",), ("family:agent", "model:old", "model:new")),
        (2, ("handler/h2",), ("family:agent", "model:new", "tool:a")),
        (2, ("handler/other",), requirements.capabilities),
    )
    for index, (level, handlers, capabilities) in enumerate(incompatible):
        bound.configure_worker(level=level, handler_versions=handlers, capabilities=capabilities)
        assert "heterogeneous" in {work.session_id for work in bound.list_runnable()}
        assert bound.claim_runnable(owner=f"incompatible-{index}", ttl_ms=10_000, limit=1) == ()
        assert _lease_row(bound, "heterogeneous")[:2] == (0, None)

    bound.configure_worker(level=2, handler_versions=("handler/h2",), capabilities=requirements.capabilities)
    claimed = bound.claim_runnable(owner="compatible", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    assert claimed[0].lease.epoch == 1
    assert _lease_row(bound, "heterogeneous")[:2] == (1, "compatible")
    assert bound.release(claimed[0].lease)


@pytest.mark.persistent_store
def test_projection_claim_defer_ack_uses_reader_level_not_executor_support(store) -> None:
    requirements = _requirements("tool:future")
    bound = _create_session(store, "reader-projection", bootstrap=requirements, consumers=("host",))
    _push(bound, "reader-projection", _input("future", "user", requirements))
    bound.configure_reader(level=2)
    bound.configure_worker(level=1, handler_versions=("handler/h1",), capabilities=())

    assert bound.acquire("reader-projection", owner="executor", ttl_ms=10_000) is None
    claimed = bound.claim_runnable(owner="reader", ttl_ms=10_000, limit=10)
    assert [(work.work.kind, work.work.consumer, work.lease) for work in claimed] == [("project", "host", None)]
    assert _lease_row(bound, "reader-projection")[:2] == (0, None)

    bound.defer_projection("reader-projection", "host", retry_after_ms=1_000)
    with bound.atomic() as tx:
        batch = tx.consumer_batch("reader-projection", "host")
        assert batch is not None
        assert [stored.record.kind for stored in batch.records] == ["session_created"]
        tx.ack(batch)

    assert (
        bound._one(
            "SELECT last_seq FROM sk_consumer WHERE tenant_id=%s AND session_id=%s AND consumer=%s",
            (bound.tenant_id, "reader-projection", "host"),
        )[0]
        == 1
    )


@pytest.mark.persistent_store
def test_level_two_format_floor_blocks_level_one_worker_before_decode(store, monkeypatch) -> None:
    level_one = _requirements(handler="handler/h1", level=1)
    bound = _create_session(store, "format-floor", bootstrap=level_one, consumers=("host",))
    assert _cache(bound, "format-floor") == (2, "handler/h1", (), None)
    assert store.read("format-floor", limit=1).records[0].record.feature_level == 2

    bound.configure_worker(level=1, handler_versions=("handler/h1",), capabilities=())
    assert bound.acquire("format-floor", owner="level-one-worker", ttl_ms=10_000) is None
    assert bound.claim_runnable(owner="level-one-worker", ttl_ms=10_000, limit=1, work_kinds=("drive",)) == ()
    assert _lease_row(bound, "format-floor")[:2] == (0, None)

    bound.configure_reader(level=1)

    def decode_must_not_run(*_args, **_kwargs):
        raise AssertionError("L2 history was decoded after the L1 reader fence")

    monkeypatch.setattr(bound, "_stored", decode_must_not_run)
    monkeypatch.setattr(bound, "_stored_input", decode_must_not_run)
    mismatch = "exceeds reader feature level 1"
    with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
        bound.read("format-floor")
    with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
        bound.read_state("format-floor")
    with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
        bound.peek_inbox("format-floor")
    with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
        bound.read_input("format-floor", "missing")
    with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
        bound.control_records("format-floor", through_seq=1)
    with pytest.raises(UnsupportedFeatureLevel, match=mismatch):
        bound.projection_context("format-floor", ())
    with bound.atomic() as tx, pytest.raises(UnsupportedFeatureLevel, match=mismatch):
        tx.consumer_batch("format-floor", "host")


@pytest.mark.persistent_store
def test_future_pending_level_two_input_raises_floor_before_becoming_ready(store) -> None:
    bound = _create_session(store, "future-input")
    store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    future = replace(
        _input("future", "user", _requirements("tool:future", handler="handler/h2")),
        available_ms=store._now() + 60_000,
    )
    _push(bound, "future-input", future)
    assert _cache(bound, "future-input")[0] == 2

    bound.configure_worker(level=1, handler_versions=("handler/h2",), capabilities=("tool:future",))
    assert bound.acquire("future-input", owner="level-one-worker", ttl_ms=10_000) is None
    assert bound.claim_runnable(owner="level-one-worker", ttl_ms=10_000, limit=1) == ()
    assert _lease_row(bound, "future-input")[:2] == (0, None)


@pytest.mark.sqlite_file
def test_delayed_signed_l2_input_refreshes_routing_at_its_available_time(tmp_path: Path) -> None:
    clock = SimClock(1000)
    database = tmp_path / "future-routing.sqlite"
    with session_sqlite.SQLiteStore.standalone(str(database), clock=clock) as store:
        store.install_schema()
        store.configure_input_admission(_HashSigner())
        bootstrap = _requirements(handler="handler/h1", level=2)
        bound = _create_session(store, "future-route", bootstrap=bootstrap)
        delayed = replace(
            _input("future", "user", _requirements("tool:future", handler="handler/h2")),
            available_ms=5000,
        )
        _push(bound, "future-route", delayed)
        assert _cache(bound, "future-route") == (2, "handler/h1", (), None)

        clock.milliseconds = 6000
        bound.configure_worker(level=2, handler_versions=("handler/h2",), capabilities=("tool:future",))
        claimed = bound.claim_runnable(owner="future-h2", ttl_ms=10_000, limit=1)
        assert len(claimed) == 1 and claimed[0].lease is not None
        assert _cache(bound, "future-route") == (2, "handler/h2", ("tool:future",), None)
        assert bound.release(claimed[0].lease)


@pytest.mark.sqlite_file
def test_sim_seed17_delayed_signed_l2_input_is_converged_without_a_write(tmp_path: Path) -> None:
    with SimWorld(tmp_path / "seed17-routing.sqlite", seed=17, signed=True) as world:
        bound = _create_session(
            world.connection("host"),
            "seed17-future-route",
            bootstrap=_requirements(handler="handler/h1", level=2),
        )
        delayed = replace(
            _input("future", "user", _requirements("tool:future", handler="handler/h2")),
            available_ms=world.clock.now_ms() + 5000,
        )
        _push(bound, "seed17-future-route", delayed)
        assert _cache(bound, "seed17-future-route") == (2, "handler/h1", (), None)

        world.advance(5000)
        bound.configure_worker(level=2, handler_versions=("handler/h2",), capabilities=("tool:future",))
        claimed = bound.claim_runnable(owner="seed17-h2", ttl_ms=10_000, limit=1)
        assert len(claimed) == 1 and claimed[0].lease is not None
        assert _cache(bound, "seed17-future-route") == (2, "handler/h2", ("tool:future",), None)
        assert bound.release(claimed[0].lease)


@pytest.mark.parametrize("admission_mode", ["signed", "hash-only"])
@pytest.mark.sqlite_file
def test_delayed_routing_accepts_each_evidence_cut_and_repeated_availability(
    tmp_path: Path,
    admission_mode: str,
) -> None:
    clock = SimClock(1000)
    with session_sqlite.SQLiteStore.standalone(str(tmp_path / f"{admission_mode}.sqlite"), clock=clock) as store:
        store.install_schema()
        if admission_mode == "signed":
            store.configure_input_admission(_HashSigner())
        elif admission_mode == "hash-only":
            store.configure_input_admission(None, allow_unsigned=True)
        bound = _create_session(
            store,
            f"projection-{admission_mode}",
            bootstrap=_requirements(handler="handler/h1", level=2),
        )
        first = replace(
            _input("first", "user", _requirements("tool:first", handler="handler/h2")),
            available_ms=5000,
        )
        selection = replace(
            _input("selection", "model_selection", _requirements("model:new", handler="handler/h1")),
            available_ms=7000,
        )
        _push(bound, f"projection-{admission_mode}", first)
        _push(bound, f"projection-{admission_mode}", selection)
        assert _cache(bound, f"projection-{admission_mode}") == (2, "handler/h1", (), None)

        clock.milliseconds = 6000
        bound.configure_worker(level=2, handler_versions=("handler/h2",), capabilities=("tool:first",))
        first_claim = bound.claim_runnable(owner=f"{admission_mode}/first", ttl_ms=10_000, limit=1)
        assert len(first_claim) == 1 and first_claim[0].lease is not None
        assert _cache(bound, f"projection-{admission_mode}") == (2, "handler/h2", ("tool:first",), None)
        assert bound.release(first_claim[0].lease)

        clock.milliseconds = 8000
        bound.configure_worker(
            level=2,
            handler_versions=("handler/h2",),
            capabilities=("model:new", "tool:first"),
        )
        second_claim = bound.claim_runnable(owner=f"{admission_mode}/second", ttl_ms=10_000, limit=1)
        assert len(second_claim) == 1 and second_claim[0].lease is not None
        assert _cache(bound, f"projection-{admission_mode}") == (
            2,
            "handler/h2",
            ("model:new", "tool:first"),
            None,
        )
        assert bound.release(second_claim[0].lease)


@pytest.mark.parametrize("admission_mode", ["signed", "hash-only"])
@pytest.mark.sqlite_file
def test_malformed_routing_cache_is_rejected_before_epoch_in_each_admission_mode(
    tmp_path: Path,
    admission_mode: str,
) -> None:
    with session_sqlite.SQLiteStore.standalone(str(tmp_path / f"tampered-{admission_mode}.sqlite")) as store:
        store.install_schema()
        if admission_mode == "signed":
            store.configure_input_admission(_HashSigner())
        elif admission_mode == "hash-only":
            store.configure_input_admission(None, allow_unsigned=True)
        session_id = f"tampered-{admission_mode}"
        requirements = _requirements("tool:required", handler="handler/h2")
        bound = _create_session(store, session_id, bootstrap=requirements)
        _push(bound, session_id, _input("work", "user", requirements))
        store._rows(
            "UPDATE sk_session SET required_capabilities=%s WHERE tenant_id=%s AND session_id=%s",
            (b"[]", store.tenant_id, session_id),
        )
        bound.configure_worker(
            level=2,
            handler_versions=("handler/h2",),
            capabilities=requirements.capabilities,
        )

        with pytest.raises(LedgerIntegrityError, match="understates"):
            bound.claim_runnable(owner=f"tampered-{admission_mode}", ttl_ms=10_000, limit=1)
        assert _lease_row(bound, session_id)[:2] == (0, None)


@pytest.mark.parametrize("admission_mode", ["signed", "hash-only"])
@pytest.mark.sqlite_file
def test_late_received_ready_input_cannot_reuse_an_old_bootstrap_cache(
    tmp_path: Path,
    admission_mode: str,
) -> None:
    clock = SimClock(1000)
    with session_sqlite.SQLiteStore.standalone(str(tmp_path / f"late-{admission_mode}.sqlite"), clock=clock) as store:
        store.install_schema()
        if admission_mode == "signed":
            store.configure_input_admission(_HashSigner())
        elif admission_mode == "hash-only":
            store.configure_input_admission(None, allow_unsigned=True)
        session_id = f"late-{admission_mode}"
        bound = _create_session(
            store,
            session_id,
            bootstrap=_requirements(handler="handler/h1", level=2),
        )

        clock.milliseconds = 6000
        _push(
            bound,
            session_id,
            replace(
                _input("late-ready", "user", _requirements("tool:new", handler="handler/h2")),
                available_ms=5000,
            ),
        )
        store._rows(
            "UPDATE sk_session SET handler_version='handler/h1',required_capabilities=%s WHERE tenant_id=%s AND session_id=%s",
            (b"[]", store.tenant_id, session_id),
        )
        bound.configure_worker(level=2, handler_versions=("handler/h2",), capabilities=("tool:new",))

        with pytest.raises(LedgerIntegrityError, match="understates"):
            bound.claim_runnable(owner=f"late-{admission_mode}", ttl_ms=10_000, limit=1)
        assert _lease_row(bound, session_id)[:2] == (0, None)


@pytest.mark.parametrize("admission_mode", ["signed", "hash-only"])
@pytest.mark.postgres
def test_postgres_delayed_routing_refreshes_in_each_admission_mode(store, admission_mode: str) -> None:
    if admission_mode == "signed":
        store.configure_input_admission(_HashSigner())
    elif admission_mode == "hash-only":
        store.configure_input_admission(None, allow_unsigned=True)
    session_id = f"pg-projection-{admission_mode}"
    bound = _create_session(
        store,
        session_id,
        bootstrap=_requirements(handler="handler/h1", level=2),
    )
    delayed = replace(
        _input("future", "user", _requirements("tool:future", handler="handler/h2")),
        available_ms=store._now() + 400,
    )
    _push(bound, session_id, delayed)
    assert _cache(bound, session_id) == (2, "handler/h1", (), None)

    Event().wait(0.55)
    bound.configure_worker(level=2, handler_versions=("handler/h2",), capabilities=("tool:future",))
    claimed = bound.claim_runnable(owner=f"pg-{admission_mode}", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    assert _cache(bound, session_id) == (2, "handler/h2", ("tool:future",), None)
    assert bound.release(claimed[0].lease)


@pytest.mark.parametrize("admission_mode", ["signed", "hash-only"])
@pytest.mark.postgres
def test_postgres_late_ready_input_cannot_reuse_an_old_bootstrap_cache(store, admission_mode: str) -> None:
    if admission_mode == "signed":
        store.configure_input_admission(_HashSigner())
    elif admission_mode == "hash-only":
        store.configure_input_admission(None, allow_unsigned=True)
    session_id = f"pg-late-{admission_mode}"
    bound = _create_session(
        store,
        session_id,
        bootstrap=_requirements(handler="handler/h1", level=2),
    )
    _push(
        bound,
        session_id,
        replace(
            _input("late-ready", "user", _requirements("tool:new", handler="handler/h2")),
            available_ms=store._now() - 1000,
        ),
    )
    store._rows(
        "UPDATE sk_session SET handler_version='handler/h1',required_capabilities=%s WHERE tenant_id=%s AND session_id=%s",
        (b"[]", store.tenant_id, session_id),
    )
    bound.configure_worker(level=2, handler_versions=("handler/h2",), capabilities=("tool:new",))

    with pytest.raises(LedgerIntegrityError, match="understates"):
        bound.claim_runnable(owner=f"pg-late-{admission_mode}", ttl_ms=10_000, limit=1)
    assert _lease_row(bound, session_id)[:2] == (0, None)


@pytest.mark.persistent_store
def test_level_one_turn_end_restores_default_cache_before_late_wake(store) -> None:
    bound = _create_session(store, "seed17")
    bound.configure_worker(level=1, handler_versions=("1",), capabilities=())
    seed = _level_one_input("seed")
    _push(bound, "seed17", seed)
    lease = _acquire(bound, "seed17", "seed17-driver")
    try:
        _append(
            bound,
            "seed17",
            lease,
            "seed17-applied",
            (_applied_input_record("seed17", "turn-1", seed, disposition="applied"),),
            consume_input_ids=(seed.input_id,),
        )
        _append(
            bound,
            "seed17",
            lease,
            "seed17-start",
            (
                make_record(
                    "turn_started",
                    session_id="seed17",
                    turn_id="turn-1",
                    payload={
                        "input_ids": [seed.input_id],
                        "definition": content_ref("definition"),
                        "definition_digest": digest(content_ref("definition")),
                        "handler_version": "1",
                        "budget": content_ref("budget"),
                        "binding": "task",
                        "generation": 1,
                        "policy_digest": None,
                        "requirements": _requirements(handler="1", level=1).to_dict(),
                        "tokenizer_binding": None,
                        "environment": None,
                    },
                ),
            ),
        )
        _append(
            bound,
            "seed17",
            lease,
            "seed17-end",
            (
                make_record(
                    "turn_ended",
                    session_id="seed17",
                    turn_id="turn-1",
                    payload={
                        "status": "completed",
                        "reason": None,
                        "result": None,
                        "adopted_results": [],
                        "budget": content_ref("budget"),
                        "unconfirmed_operations": [],
                    },
                ),
            ),
        )
    finally:
        assert bound.release(lease)

    assert _cache(bound, "seed17") == (1, None, (), None)
    _push(bound, "seed17", _level_one_input("late"))
    claimed = bound.claim_runnable(owner="late-wake", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    assert claimed[0].lease.epoch == 2
    assert bound.release(claimed[0].lease)


@pytest.mark.persistent_store
def test_level_two_turn_end_and_idle_follow_up_never_lower_format_floor(store) -> None:
    level_one = _requirements(handler="handler/h1", level=1)
    bound = _create_session(store, "persistent-floor", bootstrap=level_one)
    bound.configure_worker(level=2, handler_versions=("handler/h1",), capabilities=())
    lease = _acquire(bound, "persistent-floor", "level-two-driver")
    try:
        _append(
            bound,
            "persistent-floor",
            lease,
            "floor-start",
            (
                make_record(
                    "turn_started",
                    session_id="persistent-floor",
                    turn_id="turn-1",
                    payload={
                        "input_ids": [],
                        "definition": content_ref("definition"),
                        "definition_digest": digest(content_ref("definition")),
                        "handler_version": "handler/h1",
                        "budget": content_ref("budget"),
                        "binding": "task",
                        "generation": 1,
                        "policy_digest": None,
                        "requirements": level_one.to_dict(),
                        "tokenizer_binding": None,
                        "environment": None,
                    },
                ),
            ),
        )
        _append(
            bound,
            "persistent-floor",
            lease,
            "floor-end",
            (
                make_record(
                    "turn_ended",
                    session_id="persistent-floor",
                    turn_id="turn-1",
                    payload={
                        "status": "completed",
                        "reason": None,
                        "result": None,
                        "adopted_results": [],
                        "budget": content_ref("budget"),
                        "unconfirmed_operations": [],
                    },
                ),
            ),
        )
    finally:
        assert bound.release(lease)

    assert _cache(bound, "persistent-floor") == (2, "handler/h1", (), None)
    _push(bound, "persistent-floor", _input("follow", "follow_up", level_one))
    assert _cache(bound, "persistent-floor") == (2, "handler/h1", (), None)

    bound.configure_worker(level=1, handler_versions=("handler/h1",), capabilities=())
    assert bound.acquire("persistent-floor", owner="level-one-worker", ttl_ms=10_000) is None
    assert bound.claim_runnable(owner="level-one-worker", ttl_ms=10_000, limit=1, work_kinds=("drive",)) == ()
    bound.configure_worker(level=2, handler_versions=("handler/h1",), capabilities=())
    claimed = bound.claim_runnable(owner="level-two-worker", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    assert bound.release(claimed[0].lease)


@pytest.mark.persistent_store
def test_idle_cache_uses_later_ready_user_before_existing_queued_follow_up(store) -> None:
    bootstrap = _requirements("family:agent", "tool:bootstrap")
    bound = _create_session(store, "idle-order", bootstrap=bootstrap)
    bound.configure_worker(
        level=3,
        handler_versions=("handler/h2", "handler/h3"),
        capabilities=(
            "family:agent",
            "tool:bootstrap",
            "tool:follow",
            "tool:user",
        ),
    )

    follow_requirements = _requirements("tool:follow", handler="handler/h2")
    follow = _input("follow", "follow_up", follow_requirements)
    _push(bound, "idle-order", follow)
    lease = _acquire(bound, "idle-order", "idle-order-driver")
    try:
        _append(
            bound,
            "idle-order",
            lease,
            "idle-follow-applied",
            (_applied_input_record("idle-order", "turn-1", follow, disposition="queued"),),
            consume_input_ids=(follow.input_id,),
        )
        _append(
            bound,
            "idle-order",
            lease,
            "idle-start",
            (
                make_record(
                    "turn_started",
                    session_id="idle-order",
                    turn_id="turn-1",
                    payload={
                        "input_ids": [follow.input_id],
                        "definition": content_ref("definition"),
                        "definition_digest": digest(content_ref("definition")),
                        "handler_version": "handler/h2",
                        "budget": content_ref("budget"),
                        "binding": "task",
                        "generation": 1,
                        "policy_digest": None,
                        "requirements": follow_requirements.to_dict(),
                        "tokenizer_binding": None,
                        "environment": None,
                    },
                ),
            ),
        )
        later_user = _input("later-user", "user", _requirements("tool:user", handler="handler/h3"))
        _push(bound, "idle-order", later_user)
        assert _cache(bound, "idle-order")[1] == "handler/h2"
        _append(
            bound,
            "idle-order",
            lease,
            "idle-end",
            (
                make_record(
                    "turn_ended",
                    session_id="idle-order",
                    turn_id="turn-1",
                    payload={
                        "status": "completed",
                        "reason": None,
                        "result": None,
                        "adopted_results": [],
                        "budget": content_ref("budget"),
                        "unconfirmed_operations": [],
                    },
                ),
            ),
        )
    finally:
        assert bound.release(lease)

    assert _cache(bound, "idle-order") == (2, "handler/h3", ("tool:user",), None)
    bound.configure_worker(level=3, handler_versions=("handler/h2",), capabilities=("tool:follow",))
    assert bound.claim_runnable(owner="old-route", ttl_ms=10_000, limit=1) == ()
    bound.configure_worker(level=3, handler_versions=("handler/h3",), capabilities=("tool:user",))
    claimed = bound.claim_runnable(owner="new-route", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    assert bound.release(claimed[0].lease)


@pytest.mark.persistent_store
def test_active_model_selection_unions_old_and_new_tags_and_rejects_handler_mismatch(store) -> None:
    bootstrap = _requirements("family:agent", "model:old", "tool:base")
    bound = _create_session(store, "selection", bootstrap=bootstrap)
    bound.configure_worker(
        level=2,
        handler_versions=("handler/h2",),
        capabilities=("family:agent", "model:new", "model:old", "tool:base"),
    )
    lease = _acquire(bound, "selection", "selection-driver")
    try:
        _append(
            bound,
            "selection",
            lease,
            "selection-start",
            (
                make_record(
                    "turn_started",
                    session_id="selection",
                    turn_id="turn-1",
                    payload={
                        "input_ids": [],
                        "definition": content_ref("definition"),
                        "definition_digest": digest(content_ref("definition")),
                        "handler_version": "handler/h2",
                        "budget": content_ref("budget"),
                        "binding": "task",
                        "generation": 1,
                        "policy_digest": None,
                        "requirements": bootstrap.to_dict(),
                        "tokenizer_binding": None,
                        "environment": None,
                    },
                ),
            ),
        )
        mismatched = _input("mismatch", "model_selection", _requirements("model:new", handler="handler/h1"))
        with pytest.raises(Conflict, match="handler"):
            _push(bound, "selection", mismatched)
        selected = _input("selected", "model_selection", _requirements("model:new", handler="handler/h2"))
        _push(bound, "selection", selected)
    finally:
        assert bound.release(lease)

    assert _cache(bound, "selection") == (
        2,
        "handler/h2",
        ("family:agent", "model:new", "model:old", "tool:base"),
        "turn-1",
    )
    bound.configure_worker(
        level=2,
        handler_versions=("handler/h2",),
        capabilities=(
            "family:agent",
            "model:old",
            "tool:base",
        ),
    )
    assert bound.claim_runnable(owner="old-only", ttl_ms=10_000, limit=1) == ()
    bound.configure_worker(
        level=2,
        handler_versions=("handler/h2",),
        capabilities=(
            "family:agent",
            "model:new",
            "model:old",
            "tool:base",
        ),
    )
    claimed = bound.claim_runnable(owner="both-models", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    assert bound.release(claimed[0].lease)


@pytest.mark.parametrize(
    ("sequence", "expected_capabilities"),
    [
        (("select", "user"), ("model:new", "tool:user")),
        (("select", "clear", "user"), ("tool:user",)),
        (("select", "user", "late"), ("model:new", "tool:user")),
    ],
)
@pytest.mark.persistent_store
def test_idle_selection_derivation_follows_publish_order(store, sequence, expected_capabilities) -> None:
    bootstrap = _requirements("family:agent", "tool:base", handler="handler/h1")
    bound = _create_session(store, "idle-selection", bootstrap=bootstrap)
    items = {
        "select": _input("select", "model_selection", _requirements("model:new", handler="handler/h1")),
        "clear": _input(
            "clear",
            "model_selection",
            _requirements("model:new", handler="handler/h1"),
        ),
        "user": _input("user", "user", _requirements("tool:user", handler="handler/h2")),
        "late": replace(
            _input("late", "user", _requirements("tool:late", handler="handler/h2")),
            target_turn_id="turn/old",
        ),
    }
    items["clear"] = replace(items["clear"], payload={"selection": None})

    for step in sequence:
        _push(bound, "idle-selection", items[step])

    level, handler, capabilities, active = _cache(bound, "idle-selection")
    assert level == 2
    assert handler == "handler/h2"
    assert capabilities == expected_capabilities
    assert active is None

    missing_capabilities = tuple(capability for capability in expected_capabilities if capability != "model:new")
    if missing_capabilities != expected_capabilities:
        bound.configure_worker(
            level=2,
            handler_versions=("handler/h2",),
            capabilities=missing_capabilities,
        )
        assert bound.claim_runnable(owner=f"missing-model-{'-'.join(sequence)}", ttl_ms=10_000, limit=1) == ()

    bound.configure_worker(
        level=2,
        handler_versions=("handler/h2",),
        capabilities=expected_capabilities,
    )
    claimed = bound.claim_runnable(owner=f"routing-{'-'.join(sequence)}", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    assert bound.release(claimed[0].lease)


@pytest.mark.persistent_store
def test_idle_selection_uses_queued_input_publication_sequence_across_turns(store) -> None:
    bootstrap = _requirements("family:agent", "tool:base", handler="handler/h1")
    bound = _create_session(store, "queued-selection", bootstrap=bootstrap)
    bound.configure_worker(
        level=2,
        handler_versions=("handler/h1",),
        capabilities=bootstrap.capabilities,
    )
    initial = _input("initial", "user", bootstrap)
    follow = _input("follow", "follow_up", _requirements("tool:follow", handler="handler/h2"))
    selection = _input("select", "model_selection", _requirements("model:new", handler="handler/h2"))

    _push(bound, "queued-selection", initial)
    lease = _acquire(bound, "queued-selection", "queued-selection-driver")
    try:
        initial_applied = _applied_input_record("queued-selection", None, initial, disposition="applied")
        _append(
            bound,
            "queued-selection",
            lease,
            "initial-applied",
            (initial_applied,),
            consume_input_ids=(initial.input_id,),
        )
        _append(
            bound,
            "queued-selection",
            lease,
            "previous-turn-start",
            (
                make_record(
                    "turn_started",
                    session_id="queued-selection",
                    turn_id="turn-previous",
                    payload={
                        "input_ids": [initial.input_id],
                        "definition": content_ref("definition"),
                        "definition_digest": digest(content_ref("definition")),
                        "handler_version": "handler/h1",
                        "budget": content_ref("budget"),
                        "binding": "task",
                        "generation": 1,
                        "policy_digest": None,
                        "requirements": bootstrap.to_dict(),
                        "tokenizer_binding": None,
                        "environment": None,
                    },
                ),
            ),
        )
        _append(
            bound,
            "queued-selection",
            lease,
            "previous-turn-usage",
            (
                _usage_record("queued-selection", "meter-1", 1),
                _usage_record("queued-selection", "meter-2", 1),
                _usage_record("queued-selection", "meter-3", 1),
            ),
        )

        _push(bound, "queued-selection", follow)
        queued_record = _applied_input_record("queued-selection", "turn-previous", follow, disposition="queued")
        with bound.atomic() as tx:
            queued_receipt = tx.append(
                "queued-selection",
                lease=lease,
                expected_seq=bound.read("queued-selection").head_seq,
                commit_id="follow-queued",
                records=(queued_record,),
                consume_input_ids=(follow.input_id,),
            )
        _append(
            bound,
            "queued-selection",
            lease,
            "previous-turn-end",
            (
                make_record(
                    "turn_ended",
                    session_id="queued-selection",
                    turn_id="turn-previous",
                    payload={
                        "status": "completed",
                        "reason": None,
                        "result": None,
                        "adopted_results": [],
                        "budget": content_ref("budget"),
                        "unconfirmed_operations": [],
                    },
                ),
            ),
        )
        _push(bound, "queued-selection", selection)
    finally:
        assert bound.release(lease)

    assert queued_receipt.record_sequences == ((queued_record.record_id, 7),)
    assert bound.read("queued-selection").inbox_seq == 3
    assert _cache(bound, "queued-selection") == (
        2,
        "handler/h2",
        ("model:new", "tool:follow"),
        None,
    )

    bound.configure_worker(
        level=2,
        handler_versions=("handler/h2",),
        capabilities=("tool:follow",),
    )
    assert bound.claim_runnable(owner="missing-model", ttl_ms=10_000, limit=1) == ()

    bound.configure_worker(
        level=2,
        handler_versions=("handler/h2",),
        capabilities=("model:new", "tool:follow"),
    )
    claimed = bound.claim_runnable(owner="with-model", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    assert bound.release(claimed[0].lease)


@pytest.mark.parametrize("tamper", ["record", "commit", "checkpoint", "inbox_proof"])
@pytest.mark.persistent_store
def test_append_revalidates_locked_prefix_before_signing(store, tamper: str) -> None:
    store.configure_input_admission(_HashSigner())
    session_id = f"locked-prefix-{tamper}"
    bound = _create_session(store, session_id)
    _push(bound, session_id, _level_one_input("pending"))
    lease = _acquire(bound, session_id, "locked-prefix-worker")
    guard_calls = 0

    def tamper_same_head_before_lock() -> Lease:
        nonlocal guard_calls
        guard_calls += 1
        if tamper == "record":
            bound._rows(
                "UPDATE sk_record SET body=%s WHERE tenant_id=%s AND session_id=%s AND seq=1",
                (b"tampered", bound.tenant_id, session_id),
            )
        elif tamper == "commit":
            bound._rows(
                "UPDATE sk_commit SET body=%s WHERE tenant_id=%s AND session_id=%s "
                "AND commit_id=(SELECT commit_id FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=1)",
                (b"tampered", bound.tenant_id, session_id, bound.tenant_id, session_id),
            )
        elif tamper == "checkpoint":
            bound._rows(
                "UPDATE sk_checkpoint SET chain_digest=%s WHERE tenant_id=%s AND session_id=%s AND seq=1",
                ("0" * 64, bound.tenant_id, session_id),
            )
        else:
            bound._rows(
                "UPDATE sk_inbox SET admission_signature=%s WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
                (b"tampered", bound.tenant_id, session_id, "pending"),
            )
        return lease

    try:
        with bound.atomic() as tx:
            tx.bind_lease_guard(tamper_same_head_before_lock)
            with pytest.raises((LedgerIntegrityError, InputAdmissionProofError)):
                tx.append(
                    session_id,
                    lease=lease,
                    expected_seq=1,
                    commit_id="must-not-sign",
                    records=(_usage_record(session_id, "tampered-prefix", 1),),
                )
    finally:
        bound.release(lease)

    assert guard_calls == 1
    assert (
        bound._one(
            "SELECT head_seq FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (bound.tenant_id, session_id),
        )[0]
        == 1
    )
    assert (
        bound._one(
            "SELECT count(*) FROM sk_commit WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
            (bound.tenant_id, session_id, "must-not-sign"),
        )[0]
        == 0
    )


@pytest.mark.persistent_store
@pytest.mark.parametrize("field", ["handler", "capabilities"])
def test_understated_cache_fails_before_epoch_grant(store, field: str) -> None:
    requirements = _requirements("family:agent", "tool:required")
    bound = _create_session(store, f"tampered-{field}", bootstrap=requirements)
    _push(bound, f"tampered-{field}", _input("work", "user", requirements))
    raw = bound
    if field == "handler":
        raw._rows(
            "UPDATE sk_session SET handler_version='handler/wrong' WHERE tenant_id=%s AND session_id=%s",
            (bound.tenant_id, f"tampered-{field}"),
        )
        bound.configure_worker(level=2, handler_versions=("handler/wrong",), capabilities=requirements.capabilities)
    else:
        raw._rows(
            "UPDATE sk_session SET required_capabilities=%s WHERE tenant_id=%s AND session_id=%s",
            (b"[]", bound.tenant_id, f"tampered-{field}"),
        )
        bound.configure_worker(level=2, handler_versions=("handler/h2",), capabilities=requirements.capabilities)

    with pytest.raises(LedgerIntegrityError, match="understates"):
        bound.claim_runnable(owner="tampered", ttl_ms=10_000, limit=1)
    assert _lease_row(bound, f"tampered-{field}")[:2] == (0, None)


@pytest.mark.sqlite_file
def test_routing_claim_never_hydrates_content(store, monkeypatch) -> None:
    requirements = _requirements("family:agent", "tool:required")
    bound = _create_session(store, "no-hydrate", bootstrap=requirements)
    _push(bound, "no-hydrate", _input("work", "user", requirements))
    wrapped = wrap_fixture_store(bound)
    wrapped.configure_worker(level=2, handler_versions=("handler/h2",), capabilities=requirements.capabilities)

    from vv_agent.kernel import _content_runtime

    def exploding_reader(*_args, **_kwargs):
        raise AssertionError("routing claim attempted content hydration")

    monkeypatch.setattr(_content_runtime, "durable_content_reader", exploding_reader)
    claimed = wrapped.claim_runnable(owner="no-reader", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    assert wrapped.release(claimed[0].lease)


@pytest.mark.sqlite_file
def test_signed_input_sequence_tamper_fails_public_input_reads(store) -> None:
    class Signer:
        key_id = "routing-test-key"

        def bind_key(self):
            return self

        def sign(self, message: bytes) -> bytes:
            return sha256(message).digest()

        def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
            return key_id == self.key_id and signature == sha256(message).digest()

    store.configure_input_admission(Signer())
    bound = _create_session(store, "signed-tamper")
    _push(bound, "signed-tamper", _level_one_input("signed"))
    store._rows(
        "UPDATE sk_inbox SET input_seq=2 WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
        (store.tenant_id, "signed-tamper", "signed"),
    )

    with pytest.raises(InputAdmissionProofError):
        bound.peek_inbox("signed-tamper", ready_only=False)
    with pytest.raises(InputAdmissionProofError):
        bound.read_input("signed-tamper", "signed")


@pytest.mark.persistent_store
def test_read_captured_head_excludes_concurrent_signed_append(store, database, monkeypatch) -> None:
    class Signer:
        key_id = "read-snapshot-key"

        def bind_key(self):
            return self

        def sign(self, message: bytes) -> bytes:
            return sha256(message).digest()

        def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
            return key_id == self.key_id and signature == sha256(message).digest()

    signer = Signer()
    store.configure_input_admission(signer)
    bound = _create_session(store, "read-snapshot", consumers=("events",))
    original_verified = store._verified_records
    captured = Event()
    resume = Event()

    def pause_after_head_capture(session_id: str, through: int):
        captured.set()
        assert resume.wait(10)
        return original_verified(session_id, through)

    monkeypatch.setattr(store, "_verified_records", pause_after_head_capture)
    with ThreadPoolExecutor(max_workers=1) as pool:
        read_future = pool.submit(store.read, "read-snapshot", limit=1)
        assert captured.wait(10)
        try:
            with open_store(database) as writer:
                writer.configure_input_admission(signer)
                lease = _acquire(writer, "read-snapshot", "concurrent-writer")
                try:
                    _append(
                        writer,
                        "read-snapshot",
                        lease,
                        "concurrent-append",
                        (_usage_record("read-snapshot", "snapshot", 1),),
                    )
                finally:
                    assert writer.release(lease)
        finally:
            resume.set()
        page = read_future.result(timeout=10)

    assert page.head_seq == 1
    assert [stored.seq for stored in page.records] == [1]
    assert bound.read("read-snapshot", limit=1).head_seq == 2


@pytest.mark.persistent_store
def test_read_state_uses_captured_inbox_watermark_during_level_raising_push(store, database, monkeypatch) -> None:
    bound = _create_session(store, "captured-inbox-race")
    store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    raw_bound = bound._bound
    original_verified = raw_bound._verified_records
    original_stored_input = raw_bound._stored_input
    decoded_inputs: list[int] = []
    injected = False

    def verify_then_push(session_id: str, head: int, *args, **kwargs):
        nonlocal injected
        verified = original_verified(session_id, head, *args, **kwargs)
        if session_id == "captured-inbox-race" and not injected:
            injected = True
            with open_store(database) as writer:
                _push(
                    writer.for_tenant("default"),
                    "captured-inbox-race",
                    _input(
                        "concurrent-l2",
                        "user",
                        _requirements("tool:concurrent", handler="handler/h2"),
                    ),
                )
        return verified

    def observe_stored_input(session_id: str, row, *args, **kwargs):
        decoded_inputs.append(row[1])
        return original_stored_input(session_id, row, *args, **kwargs)

    monkeypatch.setattr(raw_bound, "_verified_records", verify_then_push)
    monkeypatch.setattr(raw_bound, "_stored_input", observe_stored_input)
    _state, records, inbox_seq = bound.read_state("captured-inbox-race")

    assert injected is True
    assert decoded_inputs == []
    assert tuple(stored.seq for stored in records) == (1,)
    assert inbox_seq == 0
    assert _cache(bound, "captured-inbox-race")[0] == 2
    assert bound.peek_inbox("captured-inbox-race", ready_only=False)[0].item.input_id == "concurrent-l2"


@pytest.mark.persistent_store
def test_middle_record_of_multi_record_commit_is_a_valid_projection_boundary(store) -> None:
    bound = _create_session(store, "partial-commit", consumers=("events",))
    with bound.atomic() as tx:
        creation_batch = tx.consumer_batch("partial-commit", "events", limit=1)
        assert creation_batch is not None
        tx.ack(creation_batch)
    lease = _acquire(bound, "partial-commit", "partial-commit-writer")
    try:
        _append(
            bound,
            "partial-commit",
            lease,
            "partial-commit-batch",
            (
                _usage_record("partial-commit", "first", 1),
                _usage_record("partial-commit", "second", 2),
            ),
        )
    finally:
        assert bound.release(lease)

    page = bound.read("partial-commit")
    assert [stored.seq for stored in page.records] == [1, 2, 3]
    middle = page.records[1]
    assert middle.commit_id == "partial-commit-batch"
    assert bound.projection_context("partial-commit", (middle,)) == (middle,)
    assert bound.control_records("partial-commit", through_seq=middle.seq) == ()
    assert [stored.seq for stored in bound.read("partial-commit", through_seq=middle.seq).records] == [1, 2]

    with bound.atomic() as tx:
        batch = tx.consumer_batch("partial-commit", "events", limit=1)
        assert batch is not None
        assert [stored.seq for stored in batch.records] == [2, 3]
        tx.ack(batch)


@pytest.mark.persistent_store
def test_projection_context_verifies_captured_head_across_concurrent_append(store, database, monkeypatch) -> None:
    _create_session(store, "projection-snapshot")
    captured_record = store.read("projection-snapshot", limit=1).records[0]
    original_verified = store._verified_records
    captured = Event()
    resume = Event()

    def pause_after_head_capture(session_id: str, through: int):
        captured.set()
        assert resume.wait(10)
        return original_verified(session_id, through)

    monkeypatch.setattr(store, "_verified_records", pause_after_head_capture)
    with ThreadPoolExecutor(max_workers=1) as pool:
        projected = pool.submit(store.projection_context, "projection-snapshot", (captured_record,))
        assert captured.wait(10)
        try:
            with open_store(database) as writer:
                lease = _acquire(writer, "projection-snapshot", "projection-writer")
                try:
                    _append(
                        writer,
                        "projection-snapshot",
                        lease,
                        "projection-concurrent-append",
                        (_usage_record("projection-snapshot", "concurrent", 1),),
                    )
                finally:
                    assert writer.release(lease)
        finally:
            resume.set()
        assert projected.result(timeout=10) == (captured_record,)

    assert store.read("projection-snapshot", limit=2).head_seq == 2


@pytest.mark.persistent_store
def test_thin_reader_paths_work_without_execution_supports(store) -> None:
    requirements = _requirements("family:agent", "tool:executor-only")
    bound = _create_session(store, "thin-reader", bootstrap=requirements, consumers=("events",))
    _push(bound, "thin-reader", _input("pending", "user", requirements))
    bound.configure_reader(level=2)

    page = bound.read("thin-reader")
    state, records, head = bound.read_state("thin-reader")
    assert state.session_id == "thin-reader" and records == page.records and head == page.head_seq
    assert bound.projection_context("thin-reader", page.records) == page.records
    assert bound.peek_inbox("thin-reader")[0].item.input_id == "pending"

    with bound.atomic() as tx:
        batch = tx.consumer_batch("thin-reader", "events")
        assert batch is not None and batch.records
        tx.ack(batch)
    with bound.atomic() as tx:
        assert tx.consumer_batch("thin-reader", "events") is None

    assert "thin-reader" in {work.session_id for work in bound.list_runnable()}
    assert bound.claim_runnable(owner="thin-executor", ttl_ms=10_000, limit=1) == ()
    assert _lease_row(bound, "thin-reader")[:2] == (0, None)


@pytest.mark.persistent_store
def test_cross_tenant_worker_claim_keeps_tenant_and_cursor_order(store) -> None:
    for tenant_id in ("tenant-a", "tenant-b"):
        bound = _create_session(store, "shared", tenant_id=tenant_id)
        _push(
            bound,
            "shared",
            InboxItem(
                f"{tenant_id}/work",
                "user",
                {"content": content_ref(f"input/{tenant_id}")},
                actor=_principal(tenant_id),
            ),
        )
    tenant_b = _create_session(store, "b-only", tenant_id="tenant-b")
    _push(
        tenant_b,
        "b-only",
        InboxItem(
            "tenant-b/only",
            "user",
            {"content": content_ref("input/tenant-b-only")},
            actor=_principal("tenant-b"),
        ),
    )

    store.configure_worker(level=1, handler_versions=("1",), capabilities=(), all_tenants=True)
    listed = store.list_runnable(limit=2)
    assert [(work.tenant_id, work.session_id) for work in listed] == [
        ("tenant-a", "shared"),
        ("tenant-b", "b-only"),
    ]
    assert [work.tenant_id for work in store.list_runnable(limit=10, after=listed[-1].cursor)] == ["tenant-b"]
    tenant_a_view = store.for_tenant("tenant-a")
    assert {work.tenant_id for work in tenant_a_view.list_runnable()} == {"tenant-a"}

    first = store.claim_runnable(owner="cross-tenant", ttl_ms=10_000, limit=1)
    second = store.claim_runnable(owner="cross-tenant", ttl_ms=10_000, limit=1)
    assert len(first) == len(second) == 1
    assert first[0].work.tenant_id == "tenant-a" and first[0].lease is not None
    assert second[0].work.tenant_id == "tenant-b" and second[0].lease is not None
    try:
        assert first[0].lease.tenant_id == "tenant-a"
        assert second[0].lease.tenant_id == "tenant-b"
        with pytest.raises(Conflict):
            tenant_a_view.read("b-only")
        with pytest.raises(ValueError, match="cursor"):
            tenant_a_view.list_runnable(after=second[0].work.cursor)
    finally:
        assert store.for_tenant(first[0].lease.tenant_id).release(first[0].lease)
        assert store.for_tenant(second[0].lease.tenant_id).release(second[0].lease)


@pytest.mark.sqlite_file
def test_local_worker_returns_pre_handoff_claim_through_lease_tenant(store, database) -> None:
    tenant_store = _create_session(store, "local-tenant", tenant_id="tenant-a")
    _push(
        tenant_store,
        "local-tenant",
        InboxItem(
            "tenant-a/local",
            "user",
            {"content": content_ref("input/tenant-a-local")},
            actor=_principal("tenant-a"),
        ),
    )
    opens = 0

    @contextmanager
    def open_store():
        nonlocal opens
        opens += 1
        if opens == 2:
            raise RuntimeError("drive connection failed before handoff")
        with session_sqlite.SQLiteStore.standalone(str(database)) as opened:
            opened.configure_input_admission(allow_unsigned=True)
            yield opened

    worker = LocalWorker(
        WorkerSupport(
            open_store=open_store,
            drive=lambda *_args, **_kwargs: pytest.fail("drive callback must not receive a failed handoff"),
            all_tenants=True,
        ),
        scan_interval=1,
    )
    with pytest.raises(ExceptionGroup) as raised:
        worker._scan()
    assert len(raised.value.exceptions) == 1
    assert isinstance(raised.value.exceptions[0], RuntimeError)
    assert str(raised.value.exceptions[0]) == "drive connection failed before handoff"

    tenant_store = store.for_tenant("tenant-a")
    epoch, owner, expires_at_ms, retry_at_ms = cast(
        tuple[int, str | None, int | None, int],
        tenant_store._one(
            "SELECT lease_epoch,lease_owner,lease_until_ms,drive_retry_at_ms FROM sk_session "
            "WHERE tenant_id=%s AND session_id=%s",
            ("tenant-a", "local-tenant"),
        ),
    )
    assert epoch == 1
    assert owner is None and expires_at_ms is None
    assert retry_at_ms > 0


@pytest.mark.postgres
def test_real_postgres_worker_credential_scans_tenants_but_tenant_does_not(store, database: str) -> None:
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    suffix = uuid4().hex
    worker_role = f"vv_worker_{suffix}"
    tenant_role = f"vv_tenant_{suffix}"
    password = f"pw-{suffix}"
    roles_installed = False
    try:
        for tenant_id in ("tenant-a", "tenant-b"):
            bound = _create_session(store, "shared", tenant_id=tenant_id)
            _push(
                bound,
                "shared",
                InboxItem(
                    f"{tenant_id}/work",
                    "user",
                    {"content": content_ref(f"input/{tenant_id}")},
                    actor=_principal(tenant_id),
                ),
            )
        tenant_b = _create_session(store, "b-only", tenant_id="tenant-b")
        _push(
            tenant_b,
            "b-only",
            InboxItem(
                "tenant-b/only",
                "user",
                {"content": content_ref("input/tenant-b-only")},
                actor=_principal("tenant-b"),
            ),
        )
        store.install_worker_rls(worker_role, password)
        store.install_tenant_rls("tenant-a", tenant_role, password)
        roles_installed = True
        info = cast(Any, store).connection.info
        host = info.host if info.host and not info.host.startswith("/") else "127.0.0.1"
        worker_dsn = make_conninfo(
            "",
            user=worker_role,
            password=password,
            host=str(host),
            hostaddr=str(host),
            port=info.port or 5432,
            dbname=str(info.dbname),
        )
        tenant_dsn = make_conninfo(
            "",
            user=tenant_role,
            password=password,
            host=str(host),
            hostaddr=str(host),
            port=info.port or 5432,
            dbname=str(info.dbname),
        )

        with session_postgres.PostgresStore.worker_standalone(worker_dsn) as worker:
            worker.configure_input_admission(allow_unsigned=True)
            assert worker.runtime_mode == "worker"
            assert worker.credential_isolation is False
            assert worker.credential_tenant_id is None
            worker.configure_worker(level=1, handler_versions=("1",), capabilities=())
            with worker.heartbeat_store() as heartbeat:
                assert heartbeat.runtime_mode == "worker"
                assert heartbeat.credential_isolation is False
                assert heartbeat.credential_tenant_id is None
            listed = worker.list_runnable(limit=10)
            assert {(work.tenant_id, work.session_id) for work in listed} >= {
                ("tenant-a", "shared"),
                ("tenant-b", "shared"),
                ("tenant-b", "b-only"),
            }
            first = worker.claim_runnable(owner="pg-root", ttl_ms=10_000, limit=1)
            second = worker.claim_runnable(owner="pg-root", ttl_ms=10_000, limit=1)
            assert len(first) == len(second) == 1
            assert first[0].work.tenant_id == "tenant-a" and first[0].lease is not None
            assert second[0].work.tenant_id == "tenant-b" and second[0].lease is not None
            try:
                assert first[0].lease.tenant_id == "tenant-a"
                assert second[0].lease.tenant_id == "tenant-b"
            finally:
                assert worker.for_tenant(first[0].lease.tenant_id).release(first[0].lease)
                assert worker.for_tenant(second[0].lease.tenant_id).release(second[0].lease)

        with session_postgres.PostgresStore.tenant_standalone(tenant_dsn, tenant_id="tenant-a") as tenant:
            tenant.configure_input_admission(allow_unsigned=True)
            assert tenant.runtime_mode == "tenant"
            assert tenant.credential_isolation is True
            assert tenant.credential_tenant_id == "tenant-a"
            with tenant.heartbeat_store() as heartbeat:
                assert heartbeat.runtime_mode == "tenant"
                assert heartbeat.credential_isolation is True
                assert heartbeat.credential_tenant_id == "tenant-a"
            cast(Any, tenant).connection.execute("SELECT set_config('vv.tenant_id','tenant-a',false)")
            assert tenant.list_sessions() == ("shared",)
            with pytest.raises(Conflict):
                tenant.read("b-only")
            with pytest.raises(PermissionError, match="worker binding"):
                tenant.claim_runnable(owner="tenant-must-not-claim", ttl_ms=10_000, limit=1)
    finally:
        if roles_installed:
            with psycopg.connect(database, autocommit=True) as admin:
                for role in (tenant_role, worker_role):
                    policies = admin.execute(
                        "SELECT schemaname,tablename,policyname FROM pg_policies WHERE %s=ANY(roles)",
                        (role,),
                    ).fetchall()
                    for schema, table, policy in policies:
                        admin.execute(
                            sql.SQL("DROP POLICY {} ON {}.{}").format(
                                sql.Identifier(policy), sql.Identifier(schema), sql.Identifier(table)
                            )
                        )
                    schema_row = admin.execute("SELECT current_schema()").fetchone()
                    assert schema_row is not None
                    schema = schema_row[0]
                    admin.execute(
                        sql.SQL("REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA {} FROM {}").format(
                            sql.Identifier(schema), sql.Identifier(role)
                        )
                    )
                    admin.execute(
                        sql.SQL("REVOKE ALL PRIVILEGES ON SCHEMA {} FROM {}").format(sql.Identifier(schema), sql.Identifier(role))
                    )
                    admin.execute(
                        sql.SQL("REVOKE CONNECT ON DATABASE {} FROM {}").format(sql.Identifier(info.dbname), sql.Identifier(role))
                    )
                    admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
