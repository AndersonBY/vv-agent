"""W4 direct session acquisition uses the same drive readiness predicate as claims."""

import pytest

from vv_agent.kernel._requirements import SessionRequirements
from vv_agent.session.records import SessionSpec

from .helpers import content_ref, item, principal


def _create_session(store, session_id: str) -> None:
    with store.atomic() as tx:
        tx.create(
            SessionSpec(
                session_id,
                principal(),
                content_ref("workspace"),
                attributes=content_ref(f"attributes/{session_id}"),
            ),
            consumers=(),
        )


def _push(store, session_id: str, **kwargs) -> None:
    with store.atomic() as tx:
        tx.push(session_id, item(**kwargs))


def _create_routed_session(store, session_id: str, requirements: SessionRequirements) -> None:
    store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
    with store.atomic() as tx:
        tx.create(
            SessionSpec(
                session_id,
                principal(),
                content_ref("workspace"),
                attributes=content_ref(f"attributes/{session_id}"),
                feature_level=2,
                bootstrap=requirements,
            ),
            consumers=(),
        )
        tx.push(
            session_id,
            item(
                input_id="ready",
                feature_level=2,
                requirements=requirements,
            ),
        )


@pytest.mark.persistent_store
def test_direct_acquire_honors_future_next_drive(store) -> None:
    session_id = "future-next-drive"
    _create_session(store, session_id)
    future = store._now() + 60_000
    store._rows(
        "UPDATE sk_session SET next_drive_ms=%s WHERE tenant_id=%s AND session_id=%s",
        (future, store.tenant_id, session_id),
    )

    assert store.claim_runnable(owner="claim", ttl_ms=10_000, limit=1) == ()
    assert store.acquire(session_id, owner="direct", ttl_ms=10_000) is None


@pytest.mark.persistent_store
def test_direct_acquire_uses_ready_inbox_even_when_next_drive_is_future(store) -> None:
    session_id = "ready-inbox"
    _create_session(store, session_id)
    _push(store, session_id, input_id="ready")
    store._rows(
        "UPDATE sk_session SET next_drive_ms=%s WHERE tenant_id=%s AND session_id=%s",
        (store._now() + 60_000, store.tenant_id, session_id),
    )

    lease = store.acquire(session_id, owner="direct", ttl_ms=10_000)
    assert lease is not None
    assert store.release(lease)

    claimed = store.claim_runnable(owner="claim", ttl_ms=10_000, limit=1)
    assert len(claimed) == 1 and claimed[0].lease is not None
    assert store.release(claimed[0].lease)


@pytest.mark.persistent_store
def test_direct_acquire_honors_future_inbox_availability(store) -> None:
    session_id = "future-inbox"
    _create_session(store, session_id)
    future = store._now() + 60_000
    _push(store, session_id, input_id="future", available_ms=future)

    assert store.claim_runnable(owner="claim", ttl_ms=10_000, limit=1) == ()
    assert store.acquire(session_id, owner="direct", ttl_ms=10_000) is None


@pytest.mark.persistent_store
def test_direct_acquire_honors_retry_gate(store) -> None:
    session_id = "retry-gate"
    _create_session(store, session_id)
    _push(store, session_id, input_id="ready")
    store._rows(
        "UPDATE sk_session SET drive_retry_at_ms=%s WHERE tenant_id=%s AND session_id=%s",
        (store._now() + 60_000, store.tenant_id, session_id),
    )

    assert store.claim_runnable(owner="claim", ttl_ms=10_000, limit=1) == ()
    assert store.acquire(session_id, owner="direct", ttl_ms=10_000) is None


@pytest.mark.persistent_store
def test_direct_acquire_preserves_worker_compatibility(store) -> None:
    session_id = "worker-compatibility"
    requirements = SessionRequirements.normalized(
        level=2,
        handler_version="handler/required",
        capabilities=("tool:required",),
    )
    _create_routed_session(store, session_id, requirements)
    store.configure_worker(level=2, handler_versions=("handler/required",), capabilities=())

    assert store.claim_runnable(owner="claim", ttl_ms=10_000, limit=1) == ()
    assert store.acquire(session_id, owner="direct", ttl_ms=10_000) is None

    store.configure_worker(
        level=2,
        handler_versions=("handler/required",),
        capabilities=("tool:required",),
    )
    lease = store.acquire(session_id, owner="direct", ttl_ms=10_000)
    assert lease is not None and lease.tenant_id == store.tenant_id and lease.session_id == session_id
    assert lease.epoch == 1
    assert lease.expires_at_ms > store._now()
    assert store.release(lease)


@pytest.mark.persistent_store
def test_direct_acquire_preserves_an_active_lease(store) -> None:
    session_id = "active-lease"
    _create_session(store, session_id)
    _push(store, session_id, input_id="ready")
    lease = store.acquire(session_id, owner="holder", ttl_ms=10_000)
    assert lease is not None

    assert store.acquire(session_id, owner="competitor", ttl_ms=10_000) is None
    assert store._one(
        "SELECT lease_epoch,lease_owner,lease_until_ms FROM sk_session WHERE tenant_id=%s AND session_id=%s",
        (store.tenant_id, session_id),
    ) == (lease.epoch, lease.owner, lease.expires_at_ms)
    assert store.release(lease)


@pytest.mark.persistent_store
def test_direct_acquire_advances_only_expired_lease_epoch(store) -> None:
    session_id = "expired-lease"
    _create_session(store, session_id)
    _push(store, session_id, input_id="ready")
    store._rows(
        "UPDATE sk_session SET lease_epoch=4,lease_owner='expired',lease_until_ms=%s WHERE tenant_id=%s AND session_id=%s",
        (store._now() - 1, store.tenant_id, session_id),
    )

    lease = store.acquire(session_id, owner="replacement", ttl_ms=10_000)
    assert lease is not None
    assert (lease.epoch, lease.owner, lease.tenant_id, lease.session_id) == (
        5,
        "replacement",
        store.tenant_id,
        session_id,
    )
    assert store.release(lease)
