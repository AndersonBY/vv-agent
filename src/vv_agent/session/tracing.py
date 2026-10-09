"""Stable run/agent/tool spans projected from records, with durable delivery cursors."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace

from vv_agent.tracing import Span, TraceProcessor

from .records import copy_json, digest
from .sql import SQLStore
from .store import SessionStore, StoredRecord


def project_spans(records: tuple[StoredRecord, ...]) -> list[tuple[int, str, Span]]:
    spans, deliveries, plans = {}, [], {}
    for stored in records:
        r, p = stored.record, stored.record._payload
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
    if isinstance(store, SQLStore):
        try:
            store._transaction_id()
        except RuntimeError:
            pass
        else:
            raise ValueError("span delivery requires its own top-level transaction")
    with store.atomic() as tx:
        batch = tx.consumer_batch(session_id, consumer)
        if batch is None:
            return 0
        _, records, _ = store.read_state(session_id)
        deliveries = [
            row
            for row in project_spans(tuple(r for r in records if r.seq <= batch.through_seq))
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
