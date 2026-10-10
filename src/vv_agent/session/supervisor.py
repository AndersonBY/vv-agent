"""One paginated scan; claims belong exclusively to drive/store primitives."""

from collections.abc import Callable
from contextlib import suppress
from uuid import uuid4

from .kernel import Runtime, drive
from .store import LeaseLost, SessionStore


def tick(
    store: SessionStore,
    *,
    project: Callable[[str, str], None],
    runtime: Callable[[str], Runtime] | None = None,
    dispatch: Callable[[str], None] | None = None,
    page_size: int = 100,
    failure_backoff_ms: int = 1000,
) -> int:
    if dispatch is None and runtime is None:
        raise ValueError("tick requires runtime or dispatch")
    if type(failure_backoff_ms) is not int or failure_backoff_ms <= 0:
        raise ValueError("failure_backoff_ms must be a positive integer")
    cursor, count = None, 0
    failures = []
    while page := store.list_runnable(limit=page_size, after=cursor):
        for work in page:
            try:
                if work.kind == "drive":
                    if dispatch is not None:
                        dispatch(work.session_id)
                    else:
                        assert runtime is not None
                        drive(store, work.session_id, runtime=runtime, failure_backoff_ms=failure_backoff_ms)
                else:
                    project(work.session_id, work.consumer)
            except Exception as exc:
                exc.add_note(f"tick {work.kind}: session={work.session_id}, consumer={work.consumer}")
                failures.append(exc)
                try:
                    if work.kind == "project":
                        store.defer_projection(work.session_id, work.consumer, retry_after_ms=failure_backoff_ms)
                    else:
                        lease = store.acquire(work.session_id, owner=uuid4().hex, ttl_ms=15000)
                        if lease is not None:
                            try:
                                with suppress(LeaseLost):
                                    store.defer_drive(lease, retry_after_ms=failure_backoff_ms)
                            finally:
                                store.release(lease)
                except Exception as backoff_error:
                    backoff_error.add_note(f"tick backoff: {work.cursor}")
                    failures.append(backoff_error)
            count += 1
        cursor = page[-1].cursor
    if failures:
        raise ExceptionGroup("Session tick failures", failures)
    return count
