"""Host maintenance for expired PostgreSQL object allocations."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from vv_agent.kernel._allocation_journal import Allocation, AllocationOwner
from vv_agent.kernel._content import ContentCleanupUnavailable
from vv_agent.kernel._external_content import _PostgresObjectContent, _PostgresRootErasure
from vv_agent.kernel._postgres_allocation_journal import _BOUND_PAGE_LIMIT

from .postgres import PostgresStore
from .s3_content import _LOCATOR_VERSION


@dataclass(frozen=True, slots=True)
class PostgresBoundMaintenanceBatch:
    examined: int
    reaped: tuple[bytes, ...]
    preserved: tuple[bytes, ...]
    next_after: tuple[int, bytes] | None


def _validate_owner(objects: _PostgresObjectContent, candidate: Allocation) -> AllocationOwner:
    owner = candidate.owner
    if (
        candidate.journal_id != objects.journal.identity.journal_id
        or candidate.state not in {"bound", "reaping"}
        or owner is None
        or owner.tenant_id != objects.journal.tenant_id
        or owner.admission_session_id != owner.session_id
        or owner.admission_id != "session/create"
        or owner.provider_instance_id != objects.provider_instance_id
        or not owner.key_id.startswith(objects._key_prefix())
    ):
        raise ContentCleanupUnavailable("PostgreSQL bound maintenance candidate has the wrong immutable identity")
    return owner


def _primary_owns_content(store: PostgresStore, owner: AllocationOwner, locator: bytes) -> bool:
    rows = store._rows(
        "SELECT key_id,object_locator FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
        (owner.tenant_id, owner.session_id, owner.content_id),
    )
    if rows == [(owner.key_id, locator)]:
        return True
    if rows or store._rows(
        "SELECT 1 FROM sk_content WHERE tenant_id=%s AND object_locator=%s LIMIT 1",
        (owner.tenant_id, locator),
    ):
        raise ContentCleanupUnavailable("PostgreSQL bound maintenance locator has a conflicting primary identity")
    return False


def _maintain_candidate(
    store: PostgresStore,
    authority: _PostgresRootErasure,
    candidate: Allocation,
    *,
    deadline_ns: int,
    clock_ns: Callable[[], int],
    check_deadline: Callable[[], None],
) -> bool:
    objects = authority.content
    owner = _validate_owner(objects, candidate)
    locator = _LOCATOR_VERSION + candidate.allocation_id
    with store._object_publication_atomic(
        objects.journal, check_deadline=check_deadline, content_digest=objects._binding_digest()
    ) as (_, check_fence):
        remaining_ns = deadline_ns - clock_ns()
        check_fence()
        statement_ms = min(2**31 - 1, max(1, remaining_ns // 1_000_000))
        store._rows("SELECT set_config('statement_timeout',%s,true)", (f"{statement_ms}ms",))
        if _primary_owns_content(store, owner, locator):
            return False
        if store._rows(
            "SELECT 1 FROM sk_session WHERE tenant_id=%s AND session_id=%s AND deletion_state<>'live'",
            (owner.tenant_id, owner.session_id),
        ):
            return False

        def check_current() -> None:
            check_fence()
            authority.require_current(owner.session_id)
            check_fence()

        check_current()
        key = objects.journal.get_key_intent(owner.provider_instance_id, owner.key_id)
        if key is None or key.journal_id != candidate.journal_id or objects._owner(key.intent, owner.content_id) != owner:
            raise ContentCleanupUnavailable("PostgreSQL bound maintenance candidate lacks its exact key intent")
        check_fence()
        claim = authority.cleanup.claim_bound(candidate, require_expired_pin=True)
        if claim is None:
            raise ContentCleanupUnavailable("PostgreSQL bound maintenance claim changed")

        if not claim.delete_confirmed:
            check_current()
            objects.objects.delete_claimed(claim, check_fence=check_fence, deadline_ns=deadline_ns, clock_ns=clock_ns)
            check_current()
            claim = authority.cleanup.confirm_bound_delete(claim)
        check_current()
        if not authority.cleanup.finish_reap(claim, remote_terminal=True):
            raise ContentCleanupUnavailable("PostgreSQL bound maintenance completion changed")
    return True


def maintain_postgres_expired_bound_objects(
    store: PostgresStore,
    authority: _PostgresRootErasure,
    *,
    deadline_ns: int,
    limit: int = _BOUND_PAGE_LIMIT,
    after: tuple[int, bytes] = (0, b""),
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> PostgresBoundMaintenanceBatch:
    """Process one page as host maintenance with a matching tenant cleanup authority.

    Pass ``next_after`` to continue the scan. An interrupted deletion keeps its
    journal claim for the next attempt. The absolute monotonic deadline bounds
    provider work and supplies a primary statement timeout.
    """
    if type(deadline_ns) is not int or deadline_ns <= 0:
        raise ValueError("PostgreSQL bound maintenance deadline must be positive monotonic nanoseconds")

    def check_deadline() -> None:
        if clock_ns() >= deadline_ns:
            raise TimeoutError("PostgreSQL bound maintenance deadline expired")

    check_deadline()
    authority.open()
    journal = authority.content.journal
    if (
        store.runtime_mode != "tenant"
        or store.credential_tenant_id != journal.tenant_id
        or not authority.content.objects._uses_journal(journal)
    ):
        raise ContentCleanupUnavailable("PostgreSQL bound maintenance requires matching tenant and object authority")
    candidates = authority.cleanup.expired_bound_candidates(limit=limit, after=after)
    reaped: list[bytes] = []
    preserved: list[bytes] = []
    cursor = None
    for candidate in candidates:
        if cursor is not None and clock_ns() >= deadline_ns:
            break
        if _maintain_candidate(
            store, authority, candidate, deadline_ns=deadline_ns, clock_ns=clock_ns, check_deadline=check_deadline
        ):
            reaped.append(candidate.allocation_id)
        else:
            preserved.append(candidate.allocation_id)
        cursor = (candidate.pin_until_ms, candidate.allocation_id)
    return PostgresBoundMaintenanceBatch(len(reaped) + len(preserved), tuple(reaped), tuple(preserved), cursor)
