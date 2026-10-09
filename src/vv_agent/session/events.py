"""Session event projection and acknowledgements over the store's consumer cursors."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from vv_agent.event_store import RunEventReplayQuery, _resolve_replay_query
from vv_agent.events import RunEvent

from .children import child_handles
from .projection import project_records
from .store import ConsumerBatch, SessionStore, SessionTx


class SessionRunEventStore:
    def __init__(self, store: SessionStore, session_id: str, consumer: str = "events"):
        self.store, self.session_id, self.consumer = store, session_id, consumer

    def append(self, event: RunEvent) -> None:
        # This bridge has no independent event ledger or append authority.
        persisted = next((e for e in self.replay(run_id=event.run_id) if e.event_id == event.event_id), None)
        if persisted is None or persisted.to_dict() != event.to_dict():
            raise ValueError("events must be projected from committed session records")

    def replay(self, query: RunEventReplayQuery | None = None, *, run_id: str | None = None) -> Iterator[RunEvent]:
        resolved = _resolve_replay_query(query, run_id=run_id)
        _, records, _ = self.store.read_state(self.session_id)
        for event in project_records(records):
            if event.run_id == resolved.run_id:
                yield event
        if resolved.include_children:
            for stored in records:
                r = stored.record
                if r.turn_id != resolved.run_id or r.kind != "op_parked" or r._payload["handle"]["kind"] != "child":
                    continue
                group = r._payload["handle"]
                for h in child_handles(group):
                    child = SessionRunEventStore(self.store, h["session_id"])
                    for event in child.replay(RunEventReplayQuery(run_id=h["turn_id"], include_children=True)):
                        payload = event.to_dict() | {"parent_run_id": resolved.run_id}
                        from vv_agent.events import event_from_dict

                        yield event_from_dict(payload)

    def batch(self, tx: SessionTx, *, limit: int = 256) -> tuple[ConsumerBatch, list[RunEvent]] | None:
        batch = tx.consumer_batch(self.session_id, self.consumer, limit=limit)
        if batch is None:
            return None
        _, records, _ = self.store.read_state(self.session_id)
        events = project_records(r for r in records if r.seq <= batch.through_seq)
        return batch, [e for e in events if batch.from_seq <= e.metadata["session_seq"] <= batch.through_seq]

    def consume(self, sink: Callable[[RunEvent], None], *, limit: int = 256) -> int:
        with self.store.atomic() as tx:
            result = self.batch(tx, limit=limit)
            if result is None:
                return 0
            batch, events = result
            for event in events:
                sink(event)
            tx.ack(batch)
            return len(batch.records)
