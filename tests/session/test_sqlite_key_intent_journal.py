from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from vv_agent.kernel import _allocation_journal
from vv_agent.kernel._allocation_journal import AllocationOwner, SQLiteAllocationJournal
from vv_agent.kernel._content import (
    ContentCleanupUnavailable,
    KeyDestroyOutcome,
    KeyDestroyStatus,
    KeyUnavailable,
    _ExactKeyIntent,
)
from vv_agent.session.content_store import LocalKeyProvider

_PIN = 2_000_000_000_000


def _intent(provider_id: str, token: str = "a") -> _ExactKeyIntent:
    return _ExactKeyIntent("s1-exact-key-" + token * 64, provider_id, "tenant", "child", "parent", "child-admission")


@pytest.fixture
def journal(tmp_path: Path) -> SQLiteAllocationJournal:
    path = tmp_path / "allocation.sqlite"
    identity = SQLiteAllocationJournal.provision(path, database_id="primary", namespace="bucket")
    return SQLiteAllocationJournal(path, identity)


def test_unknown_provider_response_reconciles_exact_key_after_primary_rollback(
    journal: SQLiteAllocationJournal, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_path = tmp_path / "provider.sqlite"
    with (
        closing(LocalKeyProvider(b"k" * 32, state_path=key_path)) as provider,
        closing(sqlite3.connect(tmp_path / "primary.sqlite")) as primary,
    ):
        intent = _intent(provider._provision_exact_key_lifecycle())
        primary.execute("CREATE TABLE keys (key_id TEXT PRIMARY KEY)")
        primary.execute("BEGIN IMMEDIATE")
        # Lose the first journal reply after its independent transaction commits.
        assert journal.reserve_key_intent(intent, pin_until_ms=_PIN) is not None
        reopened = SQLiteAllocationJournal(journal._path, journal._identity)
        reserved = reopened.get_key_intent(intent.provider_instance_id, intent.key_id)
        assert reserved is not None and reserved.state == "reserved" and reserved.intent == intent
        assert (
            reopened.reserve_key_intent(
                replace(intent, key_id=_intent(intent.provider_instance_id, "b").key_id), pin_until_ms=_PIN
            )
            is None
        )
        assert reopened.key_intents_for_admission("tenant", "child", "parent", "child-admission") == (reserved,)
        allocation = reopened.claim_key_allocation(reserved)
        assert allocation is not None and allocation.state == "allocating"
        assert journal.claim_key_allocation(reserved) is None
        assert journal.claim_key_allocation(allocation) is None
        # The provider committed, but the publisher receives no usable acknowledgement.
        lost_reply = provider._allocate_exact_key(intent)
        primary.rollback()
        assert primary.execute("SELECT key_id FROM keys").fetchall() == []

    with closing(LocalKeyProvider(b"k" * 32, state_path=key_path)) as eraser:
        current = journal.get_key_intent(intent.provider_instance_id, intent.key_id)
        assert current == allocation
        assert journal.claim_key_intent(allocation) is None
        monkeypatch.setattr(_allocation_journal, "_NOW_MS", str(_PIN))
        claim = journal.claim_key_intent(allocation)
        assert claim is not None and claim.state == "reaping"
        with pytest.raises(ContentCleanupUnavailable):
            journal.record_key_retirement(claim, KeyDestroyOutcome(True, KeyDestroyStatus.DESTROYING, None))
        lost_retire_reply = eraser._retire_exact_key(claim.intent)
        repeated = eraser._retire_exact_key(claim.intent)
        assert repeated.status is KeyDestroyStatus.ALREADY_DESTROYED
        assert repeated.evidence_id == lost_retire_reply.evidence_id
        retired = journal.record_key_retirement(claim, repeated)
        assert retired is not None and retired.state == "retired"
        assert journal.record_key_retirement(claim, repeated) == retired
        assert journal.record_key_retirement(claim, replace(repeated, evidence_id="other-receipt")) is None
        assert journal.finish_key_intent(retired)
        assert journal.get_key_intent(intent.provider_instance_id, intent.key_id) is None
        with pytest.raises(KeyUnavailable):
            eraser.unwrap(intent.tenant_id, intent.session_id, lost_reply)
        with pytest.raises(KeyUnavailable):
            eraser._allocate_exact_key(intent)


def test_key_reaping_fences_existing_object_reservations_and_preserves_replacement(
    journal: SQLiteAllocationJournal, tmp_path: Path
) -> None:
    with closing(LocalKeyProvider(b"k" * 32, state_path=tmp_path / "provider.sqlite")) as provider:
        provider_id = provider._provision_exact_key_lifecycle()
        intent = _intent(provider_id)
        replacement = replace(_intent(provider_id, "b"), admission_root_id="child", admission_id="replacement")
        reserved = journal.reserve_key_intent(intent, pin_until_ms=_PIN)
        replacement_reserved = journal.reserve_key_intent(replacement, pin_until_ms=_PIN)
        assert reserved is not None and replacement_reserved is not None

        owner = AllocationOwner("tenant", "child", "content", "parent", "child-admission", provider_id, intent.key_id)
        existing_id, late_id = bytes(32), b"\x01" * 32
        assert journal.reserve(existing_id, pin_until_ms=_PIN)
        assert journal.reserve(late_id, pin_until_ms=_PIN)
        assert not journal.bind(existing_id, owner, reservation_etag='"marker"')

        allocated = journal.claim_key_allocation(reserved)
        assert allocated is not None
        provider._allocate_exact_key(intent)
        replacement_active = journal.claim_key_allocation(replacement_reserved)
        assert replacement_active is not None
        current_key = provider._allocate_exact_key(replacement)
        assert not journal.bind(existing_id, replace(owner, tenant_id="foreign"), reservation_etag='"marker"')
        assert journal.bind(existing_id, owner, reservation_etag='"marker"')

        (old_parent_key,) = journal.scoped_key_intents("tenant", ("parent",))
        assert old_parent_key == allocated
        assert journal.scoped_key_intents("foreign", ("parent", "child")) == ()
        mismatched = replace(allocated, intent=replace(intent, admission_id=replacement.admission_id))
        assert journal.claim_key_intent(mismatched, require_expired_pin=False) is None
        claim = journal.claim_key_intent(allocated, require_expired_pin=False)
        assert claim is not None
        assert not journal.bind(late_id, owner, reservation_etag='"marker"')
        retired = journal.record_key_retirement(claim, provider._retire_exact_key(claim.intent))
        assert retired is not None and not journal.finish_key_intent(retired)
        (bound,) = journal.scoped_allocations("tenant", ("parent",))
        reaping = journal.claim_bound(bound)
        assert reaping is not None and not journal.finish_key_intent(retired)
        assert journal.finish_reap(reaping, remote_terminal=True)
        assert journal.finish_key_intent(retired)
        assert not journal.bind(late_id, owner, reservation_etag='"marker"')
        replacement_owner = replace(owner, key_id=replacement.key_id, admission_session_id="child", admission_id="next")
        assert journal.bind(late_id, replacement_owner, reservation_etag='"marker"')
        assert journal.scoped_allocations("tenant", ("parent",)) == ()
        assert journal.get_key_intent(provider_id, replacement.key_id) == replacement_active
        assert len(provider.unwrap("tenant", "child", current_key)) == 32


def test_expired_unstarted_dispatch_can_be_terminally_retired(
    journal: SQLiteAllocationJournal, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with closing(LocalKeyProvider(b"k" * 32, state_path=tmp_path / "provider.sqlite")) as provider:
        intent = _intent(provider._provision_exact_key_lifecycle())
        assert journal.reserve_key_intent(intent, pin_until_ms=time.time_ns() // 1_000_000 - 1) is None
        reservation = journal.reserve_key_intent(intent, pin_until_ms=_PIN)
        assert reservation is not None
        monkeypatch.setattr(_allocation_journal, "_NOW_MS", str(_PIN))
        assert journal.claim_key_allocation(reservation) is None
        claim = journal.claim_key_intent(reservation)
        assert claim is not None
        retired = journal.record_key_retirement(claim, provider._retire_exact_key(claim.intent))
        assert retired is not None and journal.finish_key_intent(retired)
        with pytest.raises(KeyUnavailable):
            provider._allocate_exact_key(intent)
