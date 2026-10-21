"""Bounded durable event pages over existing record positions."""

from hashlib import sha256
from typing import Any

from .projection import project_records
from .store import SessionStore


class InvalidCursor(ValueError):
    pass


def list_events(store: SessionStore, sid: str, *, limit: int, after=None, through=None) -> dict[str, Any]:
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError("limit must be in 1..256")
    first = store.read(sid, limit=1)
    if not first.records or "app_server" not in first.records[0].record._payload.get("attributes", {}):
        raise KeyError(sid)
    cache = {}

    def projected(row):
        if row.seq not in cache:
            cache[row.seq] = [
                e for e in project_records(store.projection_context(sid, (row,))) if e.metadata["session_seq"] == row.seq
            ]
        return cache[row.seq]

    def cursor(row, offset):
        return {"threadId": sid, "seq": row.seq, "offset": offset, "recordDigest": sha256(row.record.encode()).hexdigest()}

    def validate(value):
        if not isinstance(value, dict) or set(value) != {"threadId", "seq", "offset", "recordDigest"}:
            raise InvalidCursor()
        if (
            value["threadId"] != sid
            or type(value["seq"]) is not int
            or not 1 <= value["seq"] <= first.head_seq
            or type(value["offset"]) is not int
            or value["offset"] < 0
            or not isinstance(value["recordDigest"], str)
        ):
            raise InvalidCursor()
        rows = store.read(sid, after_seq=value["seq"] - 1, through_seq=value["seq"], limit=1).records
        if not rows or sha256(rows[0].record.encode()).hexdigest() != value["recordDigest"]:
            raise InvalidCursor()
        events = projected(rows[0])
        if value["offset"] > len(events):
            raise InvalidCursor()
        return rows[0], events

    if after is None:
        after = cursor(first.records[0], 0)
    start_row, start_events = validate(after)
    if through is None:
        head = store.read(sid, after_seq=first.head_seq - 1, through_seq=first.head_seq, limit=1).records[0]
        through = cursor(head, len(projected(head)))
    validate(through)

    def position(c):
        return c["seq"], c["offset"]

    if position(after) > position(through):
        raise InvalidCursor()
    next_cursor = after
    result = []
    after_seq = start_row.seq if after["offset"] == len(start_events) else start_row.seq - 1
    rows = store.read(sid, after_seq=after_seq, through_seq=through["seq"], limit=limit).records
    for row in rows:
        events = projected(row)
        start = after["offset"] if row.seq == after["seq"] else 0
        stop = through["offset"] if row.seq == through["seq"] else len(events)
        for index in range(start, stop):
            next_cursor = cursor(row, index + 1)
            result.append({"cursor": next_cursor, "event": events[index].to_dict()})
            if len(result) == limit:
                return {
                    "events": result,
                    "nextCursor": next_cursor,
                    "throughCursor": through,
                    "more": position(next_cursor) < position(through),
                }
        next_cursor = cursor(row, stop)
    return {
        "events": result,
        "nextCursor": next_cursor,
        "throughCursor": through,
        "more": position(next_cursor) < position(through),
    }
