from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from hashlib import sha256
from threading import Event
from uuid import uuid4

import pytest
from test_postgres_object_root_erasure import erasure_storage as erasure_storage
from test_s1_postgres_rls import TENANT_A
from test_s1_postgres_rls import disposable_pg as disposable_pg
from test_s1_postgres_rls import rls_database as rls_database

from vv_agent.kernel._allocation_journal import AllocationOwner
from vv_agent.kernel._content import ContentCleanupUnavailable, KeyUnavailable, _ExactKeyIntent
from vv_agent.kernel._content_runtime import bind_content_runtime
from vv_agent.kernel._external_content import _ErasureCurrentnessChallenge, _PostgresRootErasure
from vv_agent.kernel._store import Conflict
from vv_agent.session._postgres_object_maintenance import (
    maintain_postgres_expired_bound_objects,
    reconcile_postgres_failed_root_create_key,
)
from vv_agent.session.content_store import AESGCMCipher
from vv_agent.session.postgres import PostgresStore
from vv_agent.session.surfaces import SessionDriver

_EXPIRED_TTL_MS = 1_000
_EXPIRY_WAIT_SECONDS = 1.05


def _provision_orphan(
    objects,
    provider,
    session_id: str,
    content_id: str,
    *,
    pin_ttl_ms: int,
) -> tuple[AllocationOwner, bytes]:
    journal = objects.journal
    existing = journal.key_intents_for_admission(TENANT_A, session_id, session_id, "session/create")
    if existing:
        (key,) = existing
        intent = key.intent
    else:
        intent = _ExactKeyIntent(
            objects._key_prefix() + uuid4().hex,
            objects.provider_instance_id,
            TENANT_A,
            session_id,
            session_id,
            "session/create",
        )
        reserved = journal.reserve_key_intent(intent, pin_until_ms=journal._pin_until_ms(10_000))
        assert reserved is not None
        assert journal.claim_key_allocation(reserved) is not None
        assert provider._allocate_exact_key(intent).key_id == intent.key_id
    owner = objects._owner(intent, content_id)
    locator = objects.objects.put(
        owner,
        b"expired-bound-orphan",
        pin_until_ms=journal._pin_until_ms(pin_ttl_ms),
        check_fence=lambda: None,
    )
    return owner, locator


def _authority(objects, cleanup, *, current: bool = True) -> _PostgresRootErasure:
    def prove_currentness(challenge: _ErasureCurrentnessChallenge) -> _ErasureCurrentnessChallenge | None:
        return challenge if current else None

    return _PostgresRootErasure(objects, cleanup, prove_currentness, terminal_bound_delete=True)


def _insert_locator(store: PostgresStore, owner: AllocationOwner, locator: bytes) -> None:
    store._rows(
        "INSERT INTO sk_content(tenant_id,session_id,content_id,key_id,mac,size,labels,nonce,"
        "created_ms,state,pin_until_ms,ciphertext,object_locator) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'staged',0,NULL,%s)",
        (
            owner.tenant_id,
            owner.session_id,
            owner.content_id,
            owner.key_id,
            sha256(owner.content_id.encode()).hexdigest(),
            1,
            b"",
            b"n" * 12,
            0,
            locator,
        ),
    )


def _open_sql_session(store: PostgresStore, provider, objects, session_id: str) -> None:
    store.configure_input_admission(allow_unsigned=True)
    bind_content_runtime(store, key_provider=provider, cipher=AESGCMCipher(), _object_content=objects)
    with closing(SessionDriver(store=store)) as driver:
        driver.create(session_id, "/maintenance")


def _bind_root_create(store: PostgresStore, provider, objects) -> None:
    store.configure_input_admission(allow_unsigned=True)
    bind_content_runtime(store, key_provider=provider, cipher=AESGCMCipher(), _object_content=objects)


def _lose_failed_create_key_reply(store: PostgresStore, provider, objects, root_id: str, monkeypatch) -> _ExactKeyIntent:
    journal = objects.journal
    allocate = provider._allocate_exact_key
    intents: list[_ExactKeyIntent] = []

    with monkeypatch.context() as patch:
        patch.setattr(type(objects), "pin", lambda self: self.journal._pin_until_ms(500))

        def allocate_and_raise(intent):
            allocate(intent)
            intents.append(intent)
            raise TimeoutError("fixture lost exact key allocation reply")

        patch.setattr(provider, "_allocate_exact_key", allocate_and_raise)
        with closing(SessionDriver(store=store)) as driver, pytest.raises(TimeoutError, match="lost exact key"):
            driver.create(root_id, "/failed")
    (intent,) = intents
    time.sleep(0.51)
    (key,) = journal.key_intents_for_admission(TENANT_A, root_id, root_id, "session/create")
    assert key.intent == intent and key.state == "allocating"
    return intent


def test_pg_bound_maintenance_pages_expired_candidates_and_isolates_tenant(erasure_storage):
    dsn, provider, objects, cleanup, foreign, client = erasure_storage
    expired = [
        _provision_orphan(
            objects,
            provider,
            f"expired-{index}",
            f"content-{index}",
            pin_ttl_ms=_EXPIRED_TTL_MS,
        )
        for index in range(3)
    ]
    live_owner, live_locator = _provision_orphan(objects, provider, "live-pin", "live-content", pin_ttl_ms=60_000)
    time.sleep(_EXPIRY_WAIT_SECONDS)
    authority = _authority(objects, cleanup)

    assert foreign.expired_bound_candidates(limit=8) == ()
    candidates = cleanup.expired_bound_candidates(limit=8)
    assert {candidate.allocation_id for candidate in candidates} == {locator[1:] for _, locator in expired}

    with PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as store:
        with pytest.raises(ValueError, match="between 1 and 8"):
            maintain_postgres_expired_bound_objects(store, authority, deadline_ns=time.monotonic_ns() + 1_000_000_000, limit=9)
        with pytest.raises(TimeoutError, match="deadline expired"):
            maintain_postgres_expired_bound_objects(store, authority, deadline_ns=time.monotonic_ns() - 1)
        with pytest.raises(ContentCleanupUnavailable, match="authenticated tenant cleanup role"):
            maintain_postgres_expired_bound_objects(
                store,
                _authority(objects, foreign),
                deadline_ns=time.monotonic_ns() + 5_000_000_000,
            )

        first = maintain_postgres_expired_bound_objects(
            store, authority, deadline_ns=time.monotonic_ns() + 5_000_000_000, limit=2
        )
        assert first.examined == 2
        assert first.reaped == tuple(candidate.allocation_id for candidate in candidates[:2])
        assert first.preserved == ()
        assert first.next_after == (candidates[1].pin_until_ms, candidates[1].allocation_id)
        assert first.next_after is not None
        second = maintain_postgres_expired_bound_objects(
            store,
            authority,
            deadline_ns=time.monotonic_ns() + 5_000_000_000,
            limit=2,
            after=first.next_after,
        )
        assert second.examined == 1
        assert second.reaped == (candidates[2].allocation_id,)
        assert second.next_after == (candidates[2].pin_until_ms, candidates[2].allocation_id)

    assert cleanup.expired_bound_candidates(limit=8) == ()
    assert not any(locator[1:].hex() in key for _, locator in expired for key in client.objects)
    assert any(live_locator[1:].hex() in key for key in client.objects)
    (live_claim,) = cleanup.scoped_allocations(TENANT_A, (live_owner.session_id,))
    assert live_claim.state == "bound"


@pytest.mark.parametrize("phase", ["delete", "confirmation"])
def test_pg_bound_maintenance_resumes_exact_lost_ack(erasure_storage, monkeypatch, phase):
    dsn, provider, objects, cleanup, _, client = erasure_storage
    owner, locator = _provision_orphan(objects, provider, "unknown-delete", "content", pin_ttl_ms=_EXPIRED_TTL_MS)
    time.sleep(_EXPIRY_WAIT_SECONDS)
    authority = _authority(objects, cleanup)

    with PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as store:
        delete_object = client.delete_object

        def lost_ack(**request):
            delete_object(**request)
            raise TimeoutError("fixture lost DELETE acknowledgement")

        confirm_bound_delete = type(cleanup).confirm_bound_delete

        def lost_confirmation(journal, claim):
            confirm_bound_delete(journal, claim)
            raise ContentCleanupUnavailable("fixture lost confirmation acknowledgement")

        with monkeypatch.context() as patch, pytest.raises(ContentCleanupUnavailable, match=r"did not confirm|lost confirmation"):
            if phase == "delete":
                patch.setattr(client, "delete_object", lost_ack)
            else:
                patch.setattr(type(cleanup), "confirm_bound_delete", lost_confirmation)
            maintain_postgres_expired_bound_objects(store, authority, deadline_ns=time.monotonic_ns() + 5_000_000_000)

    (interrupted,) = cleanup.scoped_allocations(TENANT_A, (owner.session_id,))
    assert (interrupted.allocation_id, interrupted.owner) == (locator[1:], owner)
    assert interrupted.state == "reaping" and interrupted.delete_confirmed is (phase == "confirmation")
    assert not any(locator[1:].hex() in key for key in client.objects)

    def unexpected_delete(**_request):
        pytest.fail("a confirmed claim must resume without contacting the object store")

    if phase == "confirmation":
        monkeypatch.setattr(client, "delete_object", unexpected_delete)
    with PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as restored:
        resumed = maintain_postgres_expired_bound_objects(restored, authority, deadline_ns=time.monotonic_ns() + 5_000_000_000)
    assert resumed.reaped == (locator[1:],)
    assert cleanup.scoped_allocations(TENANT_A, (owner.session_id,)) == ()
    assert not any(locator[1:].hex() in key for key in client.objects)
    key = cleanup.get_key_intent(owner.provider_instance_id, owner.key_id)
    assert key is not None and key.intent.tenant_id == TENANT_A


def test_pg_bound_maintenance_honors_primary_fence_commit_and_rollback(erasure_storage):
    dsn, provider, objects, cleanup, _, client = erasure_storage
    session_id = "primary-fence"
    authority = _authority(objects, cleanup)
    with PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as store:
        _open_sql_session(store, provider, objects, session_id)
        committed_owner, committed_locator = _provision_orphan(
            objects, provider, session_id, "committed-unlinked", pin_ttl_ms=_EXPIRED_TTL_MS
        )
        time.sleep(_EXPIRY_WAIT_SECONDS)

        class Rollback(Exception):
            pass

        def publish(owner: AllocationOwner, locator: bytes, *, commit: bool) -> None:
            inserted = Event()
            finish = Event()

            def transaction() -> None:
                with PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as publisher:
                    try:
                        with publisher._object_publication_atomic(
                            objects.journal,
                            check_deadline=lambda: None,
                            content_digest=objects._binding_digest(),
                        ):
                            _insert_locator(publisher, owner, locator)
                            inserted.set()
                            assert finish.wait(5)
                            if not commit:
                                raise Rollback
                    except Rollback:
                        return

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(transaction)
                if not inserted.wait(5):
                    future.result(timeout=0)
                    pytest.fail("publisher did not reach the commit or rollback cut")
                with pytest.raises(Conflict, match="publication is busy"):
                    maintain_postgres_expired_bound_objects(store, authority, deadline_ns=time.monotonic_ns() + 5_000_000_000)
                finish.set()
                future.result(timeout=10)

        publish(committed_owner, committed_locator, commit=True)
        preserved = maintain_postgres_expired_bound_objects(store, authority, deadline_ns=time.monotonic_ns() + 5_000_000_000)
        assert committed_locator[1:] in preserved.preserved
        assert store._one(
            "SELECT state,pin_until_ms,object_locator FROM sk_content WHERE content_id=%s",
            (committed_owner.content_id,),
        ) == ("staged", 0, committed_locator)
        assert any(committed_locator[1:].hex() in key for key in client.objects)
        assert len(cleanup.scoped_allocations(TENANT_A, (session_id,))) >= 1

        rolled_back_owner, rolled_back_locator = _provision_orphan(
            objects, provider, session_id, "rolled-back-orphan", pin_ttl_ms=_EXPIRED_TTL_MS
        )
        time.sleep(_EXPIRY_WAIT_SECONDS)
        publish(rolled_back_owner, rolled_back_locator, commit=False)
        assert not store._rows("SELECT 1 FROM sk_content WHERE content_id=%s", (rolled_back_owner.content_id,))
        reaped = maintain_postgres_expired_bound_objects(store, authority, deadline_ns=time.monotonic_ns() + 5_000_000_000)
        assert rolled_back_locator[1:] in reaped.reaped
        assert not any(rolled_back_locator[1:].hex() in key for key in client.objects)


def test_pg_bound_maintenance_requires_currentness_and_exact_locator_identity(erasure_storage):
    dsn, provider, objects, cleanup, _, client = erasure_storage
    session_id = "currentness"
    with PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as store:
        _open_sql_session(store, provider, objects, session_id)
        owner, locator = _provision_orphan(objects, provider, session_id, "exact-content", pin_ttl_ms=_EXPIRED_TTL_MS)
        time.sleep(_EXPIRY_WAIT_SECONDS)

        with pytest.raises(ContentCleanupUnavailable, match="current storage evidence"):
            maintain_postgres_expired_bound_objects(
                store,
                _authority(objects, cleanup, current=False),
                deadline_ns=time.monotonic_ns() + 5_000_000_000,
            )
        (unchanged,) = [row for row in cleanup.scoped_allocations(TENANT_A, (session_id,)) if row.owner == owner]
        assert unchanged.state == "bound" and not unchanged.delete_confirmed
        assert any(locator[1:].hex() in key for key in client.objects)

        with store._object_publication_atomic(
            objects.journal,
            check_deadline=lambda: None,
            content_digest=objects._binding_digest(),
        ):
            _insert_locator(store, replace(owner, content_id="wrong-identity"), locator)
        with pytest.raises(ContentCleanupUnavailable, match="conflicting primary identity"):
            maintain_postgres_expired_bound_objects(
                store, _authority(objects, cleanup), deadline_ns=time.monotonic_ns() + 5_000_000_000
            )
        (still_bound,) = [row for row in cleanup.scoped_allocations(TENANT_A, (session_id,)) if row.owner == owner]
        assert still_bound.state == "bound" and not still_bound.delete_confirmed
        assert any(locator[1:].hex() in key for key in client.objects)


def test_pg_failed_root_create_key_reconciles_and_keeps_committed_root_protected(erasure_storage, monkeypatch):
    dsn, provider, objects, cleanup, _, _ = erasure_storage
    root_id = "failed-root-create"
    authority = _authority(objects, cleanup)

    with PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as store:
        _bind_root_create(store, provider, objects)
        stale = _lose_failed_create_key_reply(store, provider, objects, root_id, monkeypatch)

        assert not store._rows("SELECT 1 FROM sk_session WHERE session_id=%s", (root_id,))
        assert not store._rows("SELECT 1 FROM sk_session_key WHERE session_id=%s", (root_id,))
        assert not store._rows("SELECT 1 FROM sk_content WHERE session_id=%s", (root_id,))

        with closing(SessionDriver(store=store)) as driver, pytest.raises(KeyUnavailable, match="requires reconciliation"):
            driver.create(root_id, "/retry")

        reconciled = reconcile_postgres_failed_root_create_key(
            store, authority, root_id, deadline_ns=time.monotonic_ns() + 5_000_000_000
        )
        assert reconciled.status == "completed" and reconciled.key_id == stale.key_id
        assert reconciled.evidence_id
        assert cleanup.get_key_intent(stale.provider_instance_id, stale.key_id) is None
        with pytest.raises(KeyUnavailable):
            provider._allocate_exact_key(stale)

        with closing(SessionDriver(store=store)) as driver:
            driver.create(root_id, "/fresh")
        (fresh_key,) = store._one("SELECT key_id FROM sk_session_key WHERE session_id=%s", (root_id,))
        assert fresh_key != stale.key_id
        (fresh,) = cleanup.scoped_key_intents(TENANT_A, (root_id,))
        assert fresh.intent.key_id == fresh_key and fresh.state == "allocating"

        protected = reconcile_postgres_failed_root_create_key(store, authority, root_id)
        assert protected.status == "preserved" and protected.key_id == fresh_key
        assert cleanup.get_key_intent(fresh.intent.provider_instance_id, fresh_key) == fresh
        assert provider._verify_exact_retirement(fresh.intent) is None


def test_pg_failed_root_create_key_defers_allocations_and_missing_currentness(erasure_storage, monkeypatch):
    dsn, provider, objects, cleanup, _, _ = erasure_storage
    root_id = "failed-root-allocation"
    authority = _authority(objects, cleanup)

    with PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as store:
        _bind_root_create(store, provider, objects)
        intent = _lose_failed_create_key_reply(store, provider, objects, root_id, monkeypatch)

        with pytest.raises(ContentCleanupUnavailable, match="current storage evidence"):
            reconcile_postgres_failed_root_create_key(store, _authority(objects, cleanup, current=False), root_id)
        current = cleanup.get_key_intent(intent.provider_instance_id, intent.key_id)
        assert current is not None and current.state == "allocating"
        assert provider._verify_exact_retirement(intent) is None

        owner = objects._owner(intent, "orphan-content")
        locator = objects.objects.put(
            owner,
            b"orphan-content",
            pin_until_ms=objects.journal._pin_until_ms(60_000),
            check_fence=lambda: None,
        )
        assert any(locator[1:].hex() in key for key in erasure_storage[-1].objects)

        deferred = reconcile_postgres_failed_root_create_key(store, authority, root_id)
        assert deferred.status == "deferred" and deferred.key_id == intent.key_id
        assert cleanup.get_key_intent(intent.provider_instance_id, intent.key_id) == current
        assert provider._verify_exact_retirement(intent) is None


def test_pg_failed_root_create_key_preserves_tombstone_owned_state(erasure_storage, monkeypatch):
    dsn, provider, objects, cleanup, _, _ = erasure_storage
    root_id = "failed-root-tombstone"

    with PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as store:
        _bind_root_create(store, provider, objects)
        intent = _lose_failed_create_key_reply(store, provider, objects, root_id, monkeypatch)
        store._rows(
            "INSERT INTO sk_tombstone(tenant_id,session_id,final_seq,chain_digest,deleted_ms,actor,key_id,"
            "signature,state,scope_root_session_id) VALUES (%s,%s,0,%s,0,%s,%s,%s,'pending',%s)",
            (TENANT_A, root_id, sha256(b"tombstone").hexdigest(), b"{}", intent.key_id, b"signature", root_id),
        )

        result = reconcile_postgres_failed_root_create_key(store, _authority(objects, cleanup), root_id)
        assert result.status == "preserved" and result.key_id == intent.key_id
        current = cleanup.get_key_intent(intent.provider_instance_id, intent.key_id)
        assert current is not None and current.state == "allocating"
        assert provider._verify_exact_retirement(intent) is None


@pytest.mark.parametrize("phase", ["provider", "journal"])
def test_pg_failed_root_create_key_retries_lost_terminal_ack(erasure_storage, monkeypatch, phase):
    dsn, provider, objects, cleanup, _, _ = erasure_storage
    root_id = "failed-root-lost-retire"
    authority = _authority(objects, cleanup)

    with PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as store:
        _bind_root_create(store, provider, objects)
        intent = _lose_failed_create_key_reply(store, provider, objects, root_id, monkeypatch)
        retire = provider._retire_exact_key
        evidence: list[str] = []

        def retire_then_lose_reply(expected):
            outcome = retire(expected)
            assert outcome.evidence_id
            evidence.append(outcome.evidence_id)
            raise TimeoutError("fixture lost terminal key acknowledgement")

        commit_retirement = type(cleanup).record_key_retirement

        def record_then_lose_reply(journal, claim, outcome):
            assert commit_retirement(journal, claim, outcome) is not None
            assert outcome.evidence_id
            evidence.append(outcome.evidence_id)
            raise TimeoutError("fixture lost terminal key journal acknowledgement")

        with monkeypatch.context() as patch, pytest.raises(TimeoutError, match="terminal key"):
            if phase == "provider":
                patch.setattr(provider, "_retire_exact_key", retire_then_lose_reply)
            else:
                patch.setattr(type(cleanup), "record_key_retirement", record_then_lose_reply)
            reconcile_postgres_failed_root_create_key(store, authority, root_id)

        interrupted = cleanup.get_key_intent(intent.provider_instance_id, intent.key_id)
        assert interrupted is not None and interrupted.state == ("reaping" if phase == "provider" else "retired")
        assert provider._verify_exact_retirement(intent) == evidence[0]

        if phase == "journal":
            with monkeypatch.context() as patch, pytest.raises(ContentCleanupUnavailable, match="retirement evidence changed"):
                patch.setattr(provider, "_verify_exact_retirement", lambda _intent: None)
                reconcile_postgres_failed_root_create_key(store, authority, root_id)
            assert cleanup.get_key_intent(intent.provider_instance_id, intent.key_id) == interrupted

        def unexpected_retirement(_intent):
            pytest.fail("reconciliation should reuse the provider's terminal evidence")

        with monkeypatch.context() as patch:
            patch.setattr(provider, "_retire_exact_key", unexpected_retirement)
            resumed = reconcile_postgres_failed_root_create_key(store, authority, root_id)
        assert resumed.status == "completed" and resumed.evidence_id == evidence[0]
        assert cleanup.get_key_intent(intent.provider_instance_id, intent.key_id) is None
        assert reconcile_postgres_failed_root_create_key(store, authority, root_id).status == "completed"
