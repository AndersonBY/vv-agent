"""One paginated scan; claims belong exclusively to drive/store primitives."""

from collections.abc import Callable

from .kernel import Runtime, drive
from .store import SessionStore


def tick(
    store: SessionStore, *, runtime: Callable[[str], Runtime], project: Callable[[str, str], None], page_size: int = 100
) -> int:
    cursor, count = None, 0
    while page := store.list_runnable(limit=page_size, after=cursor):
        for work in page:
            if work.kind == "drive":
                drive(store, work.session_id, runtime=runtime(work.session_id))
            else:
                project(work.session_id, work.consumer)
            count += 1
        cursor = page[-1].cursor
    return count
