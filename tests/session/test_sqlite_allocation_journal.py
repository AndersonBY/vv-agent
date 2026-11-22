from __future__ import annotations

import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from vv_agent.kernel._allocation_journal import AllocationOwner, SQLiteAllocationJournal
from vv_agent.kernel._content import ContentCleanupUnavailable, ContentUnavailable

_OWNER = AllocationOwner("tenant", "child", "content", "parent", "admission", "provider", "key")
_ETAG = '"marker-etag"'


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _id(number: int) -> bytes:
    return number.to_bytes(32)


def _bound(journal: SQLiteAllocationJournal, number: int, owner: AllocationOwner = _OWNER) -> None:
    assert journal.reserve(_id(number), pin_until_ms=_now_ms() + 10_000)
    assert journal.bind(_id(number), owner, reservation_etag=_ETAG)


def test_sidecar_changes_survive_primary_rollback(tmp_path: Path) -> None:
    path = tmp_path / "allocation.sqlite"
    identity = SQLiteAllocationJournal.provision(path, database_id="main-identity", namespace="bucket/prefix")
    journal = SQLiteAllocationJournal(path, identity)
    observer = SQLiteAllocationJournal(path, identity)
    with closing(sqlite3.connect(tmp_path / "primary.sqlite")) as primary:
        primary.execute("CREATE TABLE effects (value TEXT)")
        primary.execute("BEGIN IMMEDIATE")
        primary.execute("INSERT INTO effects VALUES ('uncommitted')")
        deadline = _now_ms() + 10_000
        assert journal.reserve(_id(1), pin_until_ms=deadline)
        reservation = observer.get(_id(1))
        assert reservation is not None
        assert (reservation.state, reservation.owner, reservation.reservation_etag) == ("unbound", None, None)
        assert not journal.reserve(_id(1), pin_until_ms=deadline + 1)
        assert journal.bind(_id(1), _OWNER, reservation_etag=_ETAG)
        assert not journal.bind(_id(1), replace(_OWNER, key_id="other"), reservation_etag=_ETAG)
        assert observer.claim_anonymous(_id(1)) is None
        primary.rollback()
        assert primary.execute("SELECT value FROM effects").fetchall() == []

    bound = SQLiteAllocationJournal(path, identity).get(_id(1))
    assert bound is not None
    assert (bound.state, bound.owner, bound.reservation_etag, bound.pin_until_ms) == ("bound", _OWNER, _ETAG, deadline)


def test_scoped_claim_and_finish_use_exact_binding(tmp_path: Path) -> None:
    path = tmp_path / "allocation.sqlite"
    identity = SQLiteAllocationJournal.provision(path, database_id="primary", namespace="bucket")
    journal = SQLiteAllocationJournal(path, identity)
    _bound(journal, 1)
    _bound(journal, 2, replace(_OWNER, admission_session_id="unrelated", admission_id="replacement", key_id="new"))
    _bound(journal, 3, replace(_OWNER, tenant_id="foreign"))
    assert journal.reserve(_id(4), pin_until_ms=_now_ms() + 10_000)

    (bound,) = journal.scoped_allocations("tenant", ("parent",))
    assert bound.allocation_id == _id(1)
    assert [row.allocation_id for row in journal.scoped_allocations("tenant", ("child", "parent"))] == [_id(1), _id(2)]
    assert journal.scoped_allocations("tenant", ()) == ()
    assert journal.claim_bound(replace(bound, owner=replace(_OWNER, provider_instance_id="foreign"))) is None
    assert journal.claim_bound(replace(bound, reservation_etag='"different-marker"')) is None
    assert journal.claim_bound(replace(bound, journal_id="different-journal")) is None

    claim = journal.claim_bound(bound)
    assert claim is not None and claim.state == "reaping"
    assert journal.claim_bound(bound) is None
    assert journal.claim_anonymous(_id(1)) is None
    with pytest.raises(ContentCleanupUnavailable):
        journal.finish_reap(claim, remote_terminal=False)
    reopened = SQLiteAllocationJournal(path, identity)
    assert reopened.scoped_allocations("tenant", ("parent",)) == (claim,)
    assert reopened.claim_bound(claim) == claim
    assert not reopened.finish_reap(replace(claim, owner=None), remote_terminal=True)
    assert not reopened.finish_reap(replace(claim, journal_id="different-journal"), remote_terminal=True)
    assert reopened.finish_reap(claim, remote_terminal=True)
    assert not reopened.finish_reap(claim, remote_terminal=True)
    assert reopened.get(_id(1)) is None
    assert reopened.get(_id(2)) is not None


def test_anonymous_claim_after_pin_expiry_excludes_bind_and_retains_unknown_marker(tmp_path: Path) -> None:
    path = tmp_path / "allocation.sqlite"
    identity = SQLiteAllocationJournal.provision(path, database_id="primary", namespace="bucket")
    writer = SQLiteAllocationJournal(path, identity, busy_timeout_ms=2000)
    sweeper = SQLiteAllocationJournal(path, identity, busy_timeout_ms=2000)
    assert not writer.reserve(_id(99), pin_until_ms=_now_ms() - 1)
    deadline = _now_ms() + 250
    assert writer.reserve(_id(1), pin_until_ms=deadline)
    assert writer.reserve(_id(2), pin_until_ms=deadline)
    assert sweeper.claim_anonymous(_id(1)) is None
    with closing(sqlite3.connect(path)) as blocker, ThreadPoolExecutor(2) as executor:
        blocker.execute("BEGIN IMMEDIATE")
        pending_bind = executor.submit(writer.bind, _id(1), _OWNER, reservation_etag=_ETAG)
        pending_claim = executor.submit(sweeper.claim_anonymous, _id(1))
        time.sleep(max(0, deadline - _now_ms() + 5) / 1000)
        blocker.commit()
        assert not pending_bind.result(timeout=2)
        claim = pending_claim.result(timeout=2)
    assert claim is not None and claim.owner is None and claim.state == "reaping"
    assert sweeper.anonymous_candidates(limit=1) == (claim,)
    next_page = sweeper.anonymous_candidates(limit=1, after=(claim.pin_until_ms, claim.allocation_id))
    assert [row.allocation_id for row in next_page] == [_id(2)]
    with pytest.raises(ContentCleanupUnavailable):
        sweeper.finish_reap(claim, remote_terminal=False)
    reopened = SQLiteAllocationJournal(path, identity)
    assert reopened.claim_anonymous(_id(1)) == claim
    with pytest.raises(ContentCleanupUnavailable):
        reopened.confirm_bound_delete(claim)
    assert reopened.get(_id(1)) == claim
    assert not reopened.bind(_id(1), _OWNER, reservation_etag=_ETAG)
    assert reopened.finish_reap(claim, remote_terminal=True)


def test_allocation_schema_rejects_partial_binding_and_busy_writer_defers(tmp_path: Path) -> None:
    path = tmp_path / "allocation.sqlite"
    identity = SQLiteAllocationJournal.provision(path, database_id="primary", namespace="bucket")
    journal = SQLiteAllocationJournal(path, identity, busy_timeout_ms=0)
    deadline = _now_ms() + 10_000
    assert journal.reserve(_id(1), pin_until_ms=deadline)
    with closing(sqlite3.connect(path)) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE allocation SET state='bound',tenant_id='tenant'")
        connection.rollback()
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(ContentUnavailable):
            journal.bind(_id(1), _OWNER, reservation_etag=_ETAG)
        connection.rollback()
    assert journal.bind(_id(1), _OWNER, reservation_etag=_ETAG)


def test_existing_only_open_rejects_missing_wrong_or_altered_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "allocation.sqlite"
    identity = SQLiteAllocationJournal.provision(path, database_id="primary", namespace="bucket")
    with pytest.raises(FileExistsError):
        SQLiteAllocationJournal.provision(path, database_id="primary", namespace="bucket")

    missing = tmp_path / "missing.sqlite"
    with pytest.raises(ContentUnavailable):
        SQLiteAllocationJournal(missing, identity)
    assert not missing.exists()
    for field in ("journal_id", "database_id", "namespace"):
        with pytest.raises(ContentUnavailable, match="identity"):
            SQLiteAllocationJournal(path, replace(identity, **{field: "wrong"}))

    path.chmod(0o644)
    with pytest.raises(ContentUnavailable, match="owner-only"):
        SQLiteAllocationJournal(path, identity)
    path.chmod(0o600)
    opened = SQLiteAllocationJournal(path, identity)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DROP INDEX allocation_admission")
    with pytest.raises(ContentUnavailable, match="schema"):
        SQLiteAllocationJournal(path, identity)
    with pytest.raises(ContentUnavailable, match="schema"):
        opened.scoped_allocations("tenant", ("parent",))

    corrupt = tmp_path / "corrupt.sqlite"
    corrupt.write_bytes(os.urandom(512))
    corrupt.chmod(0o600)
    with pytest.raises(ContentUnavailable):
        SQLiteAllocationJournal(corrupt, identity)
