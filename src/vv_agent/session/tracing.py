"""Stable run/agent/tool spans projected from records, with durable delivery cursors."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace

from vv_agent.kernel._sql import SQLStore
from vv_agent.tracing import Span, TraceProcessor

from .projection import (
    SessionHydration,
    hydration_for_session_store,
    projection_records_frame,
)
from .records import copy_json, digest
from .store import SessionStore, StoredRecord

_SPAN_CONTENT_KINDS = frozenset({"turn_started", "op_planned", "op_prepared", "op_completed", "turn_ended"})


def project_spans(
    records: tuple[StoredRecord, ...],
    *,
    hydration: SessionHydration | None = None,
) -> list[tuple[int, str, Span]]:
    with projection_records_frame(records, hydration, hydrate_kinds=_SPAN_CONTENT_KINDS) as hydrated:
        return _project_hydrated_spans(hydrated)


def _project_hydrated_spans(records: tuple[StoredRecord, ...]) -> list[tuple[int, str, Span]]:
    spans, deliveries, plans = {}, [], {}
    for stored in records:
        r = stored.record
        p = copy_json(r._payload)
        tid = r.turn_id
        if tid is None:
            continue
        if r.kind == "turn_started":
            agent = p["definition"].get("agent_name", p["definition"]["task"].get("metadata", {}).get("agent_name"))
            run = Span(
                "run",
                tid,
                span_id=f"sk/span/{digest([r.session_id, tid, 'run'])}",
                started_at=stored.created_ms / 1000,
                metadata={"agent_name": agent, "workflow_name": None},
            )
            agent_span = Span(
                "agent",
                tid,
                span_id=f"sk/span/{digest([r.session_id, tid, 'agent'])}",
                parent_id=run.span_id,
                started_at=stored.created_ms / 1000,
                metadata={"agent_name": agent},
            )
            spans[(tid, "run")], spans[(tid, "agent")] = run, agent_span
            deliveries.extend([(stored.seq, "on_span_start", run), (stored.seq, "on_span_start", agent_span)])
        elif r.kind == "op_planned":
            plans[(r.operation_id, r.attempt)] = p
        elif r.kind == "op_prepared":
            plans[(r.operation_id, r.attempt)] = plans[(r.operation_id, r.attempt)] | {"request": p["request"]}
        elif r.kind == "op_started" and plans[(r.operation_id, r.attempt)]["op_kind"] != "model":
            plan = plans[(r.operation_id, r.attempt)]
            parent = spans[(tid, "agent")]
            span = Span(
                "tool",
                tid,
                span_id=f"sk/span/{digest([r.session_id, r.operation_id, r.attempt])}",
                parent_id=parent.span_id,
                started_at=stored.created_ms / 1000,
                metadata={"tool_name": plan["request"]["name"], **parent.metadata},
            )
            spans[(tid, r.operation_id, r.attempt)] = span
            deliveries.append((stored.seq, "on_span_start", span))
        elif r.kind == "op_completed":
            span = spans.pop((tid, r.operation_id, r.attempt), None)
            if span:
                deliveries.append(
                    (
                        stored.seq,
                        "on_span_end",
                        replace(
                            span,
                            ended_at=stored.created_ms / 1000,
                            metadata=span.metadata | {"status": p["result"]["status_code"].lower()},
                        ),
                    )
                )
        elif r.kind == "turn_ended":
            for key in list(spans):
                if key[0] == tid and len(key) == 3:
                    span = spans.pop(key)
                    deliveries.append(
                        (
                            stored.seq,
                            "on_span_end",
                            replace(span, ended_at=stored.created_ms / 1000, metadata=span.metadata | {"status": "abandoned"}),
                        )
                    )
            for name in ("agent", "run"):
                span = spans.pop((tid, name))
                deliveries.append(
                    (
                        stored.seq,
                        "on_span_end",
                        replace(
                            span,
                            ended_at=stored.created_ms / 1000,
                            metadata=span.metadata | {"status": p["status"], "final_output": copy_json(p["result"])},
                        ),
                    )
                )
    return deliveries


def deliver_spans(store: SessionStore, session_id: str, processors: list[TraceProcessor], *, consumer: str = "traces") -> int:
    # Telemetry is at-most-once: the cursor must commit before nontransactional processors.
    transaction_store = store
    seen: set[int] = set()
    while id(transaction_store) not in seen:
        seen.add(id(transaction_store))
        parent = getattr(transaction_store, "_root_store", None) or getattr(transaction_store, "_root", None)
        if parent is None or parent is transaction_store:
            break
        transaction_store = parent
    if isinstance(transaction_store, SQLStore):
        try:
            transaction_store._transaction_id()
        except RuntimeError:
            pass
        else:
            raise ValueError("span delivery requires its own top-level transaction")
    with store.atomic() as tx:
        batch = tx.consumer_batch(session_id, consumer)
        if batch is None:
            return 0
        _, records, _ = store.read_state(session_id)
        hydration = hydration_for_session_store(store, session_id)
        deliveries = [
            row
            for row in project_spans(
                tuple(r for r in records if r.seq <= batch.through_seq),
                hydration=hydration,
            )
            if batch.from_seq <= row[0] <= batch.through_seq
        ]
        tx.ack(batch)
    for _, method, span in deliveries:
        for processor in processors:
            try:
                getattr(processor, method)(span)
            except Exception:
                continue
    for processor in processors:
        flush = getattr(processor, "flush", None)
        if flush:
            with suppress(Exception):
                flush()
    return len(deliveries)
