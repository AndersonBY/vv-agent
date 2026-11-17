"""Claim runnable work and dispatch each claim with its drive lease."""

from collections.abc import Callable
from uuid import uuid4

from vv_agent.kernel._store import ProjectionConsumers, WorkKinds

from ._drive_phase import drive_with_runtime
from .runtime import Runtime
from .store import Lease, SessionStore


def tick(
    store: SessionStore,
    *,
    project: Callable[[str, str], None],
    runtime: Callable[[str], Runtime] | None = None,
    dispatch: Callable[[str], None] | None = None,
    dispatch_claimed: Callable[[str, Lease], None] | None = None,
    project_claimed: Callable[[str, str, str], None] | None = None,
    page_size: int = 100,
    failure_backoff_ms: int = 1000,
    claim_owner: str | None = None,
    claim_ttl_ms: int = 15_000,
    work_kinds: WorkKinds | None = None,
    projection_consumers: ProjectionConsumers | None = None,
    stop_after_claimed_drive: bool = False,
) -> int:
    """Dispatch runnable work once.

    A claimed drive lease transfers to ``drive_with_runtime`` or
    ``dispatch_claimed`` at invocation. The receiver owns its defer/release
    lifecycle, including when it raises.
    """

    if dispatch is None and dispatch_claimed is None and runtime is None:
        raise ValueError("tick requires runtime, dispatch or dispatch_claimed")
    if type(failure_backoff_ms) is not int or failure_backoff_ms <= 0:
        raise ValueError("failure_backoff_ms must be a positive integer")
    if type(page_size) is not int or page_size <= 0:
        raise ValueError("page_size must be a positive integer")
    if type(claim_ttl_ms) is not int or claim_ttl_ms <= 0:
        raise ValueError("claim_ttl_ms must be a positive integer")
    owner = claim_owner or uuid4().hex
    if not isinstance(owner, str) or not owner or "\x00" in owner:
        raise ValueError("claim_owner must be non-empty text without NUL")
    cursor, count = None, 0
    failures = []
    if dispatch_claimed is None and dispatch is not None:
        while page := store.list_runnable(
            limit=page_size,
            after=cursor,
            work_kinds=work_kinds,
            projection_consumers=projection_consumers,
        ):
            for work in page:
                try:
                    if work.kind == "drive":
                        dispatch(work.session_id)
                    else:
                        project(work.session_id, work.consumer)
                except Exception as exc:
                    exc.add_note(f"tick {work.kind}: session={work.session_id}, consumer={work.consumer}")
                    failures.append(exc)
                    if work.kind == "project":
                        try:
                            store.defer_projection(work.session_id, work.consumer, retry_after_ms=failure_backoff_ms)
                        except Exception as backoff_error:
                            backoff_error.add_note(f"tick backoff: {work.cursor}")
                            failures.append(backoff_error)
                count += 1
                cursor = work.cursor
    else:
        while page := store.claim_runnable(
            owner=owner,
            ttl_ms=claim_ttl_ms,
            limit=1,
            after=cursor,
            work_kinds=work_kinds,
            projection_consumers=projection_consumers,
        ):
            claimed = page[0]
            work, lease = claimed.work, claimed.lease
            try:
                if work.kind == "drive":
                    assert lease is not None
                    if dispatch_claimed is not None:
                        dispatch_claimed(work.session_id, lease)
                    else:
                        assert runtime is not None
                        drive_store = store.for_tenant(work.tenant_id)
                        drive_with_runtime(
                            drive_store,
                            work.session_id,
                            runtime=runtime,
                            failure_backoff_ms=failure_backoff_ms,
                            lease=lease,
                        )
                else:
                    if project_claimed is not None:
                        project_claimed(work.tenant_id, work.session_id, work.consumer)
                    else:
                        project(work.session_id, work.consumer)
            except Exception as exc:
                exc.add_note(f"tick {work.kind}: session={work.session_id}, consumer={work.consumer}")
                failures.append(exc)
                try:
                    if work.kind == "project":
                        project_store = store.for_tenant(work.tenant_id)
                        project_store.defer_projection(
                            work.session_id,
                            work.consumer,
                            retry_after_ms=failure_backoff_ms,
                        )
                except Exception as backoff_error:
                    backoff_error.add_note(f"tick backoff: {work.cursor}")
                    failures.append(backoff_error)
            count += 1
            cursor = work.cursor
            if stop_after_claimed_drive and work.kind == "drive":
                break
    if failures:
        raise ExceptionGroup("Session tick failures", failures)
    return count
