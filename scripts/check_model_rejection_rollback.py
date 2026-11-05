"""Read-only 0.33.1 rollback preflight for a caller-supplied session store.

Stop every driver first. An empty result only means no rejection history was
found; the host's 0.33.2 session ownership exclusion still applies.
"""

from __future__ import annotations

from typing import Protocol

from vv_agent.session.store import ReadPage


class ReadableStore(Protocol):
    def list_sessions(self, *, limit: int = 100, after: str | None = None) -> tuple[str, ...]: ...

    def read(self, session_id: str, *, after_seq: int = 0, through_seq: int | None = None, limit: int = 1024) -> ReadPage: ...


def check_store(store: ReadableStore, *, page_size: int = 256) -> list[dict]:
    """Return affected session/turn identities and reasons; never acquires or writes.

    Conservatively excludes all rejection histories, including completed turns.
    Scan errors propagate: callers must not treat a failed scan as a clean store.
    """
    if type(page_size) is not int or page_size < 1:
        raise ValueError("page_size must be a positive integer")
    findings = []
    after = None
    while sessions := store.list_sessions(limit=page_size, after=after):
        for sid in sessions:
            receipts: dict[tuple[str | None, str | None], int] = {}
            retries: set[tuple[str | None, str | None]] = set()
            ended: set[str | None] = set()
            seq, head = 0, None
            while True:
                page = store.read(sid, after_seq=seq, through_seq=head, limit=page_size)
                head = page.head_seq if head is None else head
                for stored in page.records:
                    record = stored.record
                    key = record.turn_id, record.operation_id
                    result = record.payload.get("result")
                    if (
                        record.kind == "op_completed"
                        and isinstance(result, dict)
                        and result.get("error_code") == "model_request_rejected"
                    ):
                        receipts[key] = record.attempt or 1
                    elif record.kind == "op_planned" and key in receipts and (record.attempt or 1) > receipts[key]:
                        retries.add(key)
                    elif record.kind == "turn_ended":
                        ended.add(record.turn_id)
                if not page.records:
                    if seq < head:
                        raise ValueError("incomplete session history")
                    break
                seq = page.records[-1].seq
                if seq >= head:
                    break
            for turn_id in sorted({key[0] for key in receipts}, key=str):
                reasons = ["model_rejection_history"]
                if turn_id not in ended:
                    reasons.append("rejection_without_turn_ended")
                if any(key[0] == turn_id for key in retries):
                    reasons.append("rejection_retry_history")
                findings.append({"session_id": sid, "turn_id": turn_id, "reasons": reasons})
        after = sessions[-1]
    return findings
