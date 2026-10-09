from typing import Any

from vv_agent.session.records import InboxItem, Record, digest, make_record


def item(input_id: str = "i", kind: str = "user", **kw: Any) -> InboxItem:
    payload = {"content": "hello"}
    if kind == "control":
        payload = {"action": kw.pop("action", "cancel")}
    return InboxItem(input_id=input_id, kind=kind, payload=payload, **kw)


def record(kind: str, *, sid: str = "s", tid: str = "t", oid: str = "o", attempt: int = 1, **kw: Any) -> Record:
    request = {"messages": []}
    payloads: dict[str, dict[str, Any]] = {
        "session_created": {
            "principal": "p",
            "workspace": "w",
            "parent_session_id": None,
            "parent_operation_id": None,
            "attributes": {},
        },
        "turn_started": {
            "input_ids": [],
            "definition": {},
            "definition_digest": digest({}),
            "handler_version": "1",
            "budget": {},
            "binding": "task",
            "generation": 1,
        },
        "input_applied": {
            "input": item().to_dict(),
            "input_digest": digest(item().to_dict()),
            "disposition": "applied",
            "reason": None,
            "target_operation_id": None,
            "target_wait_id": None,
            "position": "before_dispatch",
        },
        "op_planned": {
            "op_kind": "model",
            "purpose": "primary",
            "request": request,
            "request_digest": digest(request),
            "context_version": "v1",
            "dependencies": [],
            "tool": None,
            "idempotency_key": "key",
            "budget_admission": {},
            "not_before_ms": None,
            "provider_binding": "provider",
            "consumed_unknowns": [],
        },
        "op_started": {"dispatch_id": "d", "authorization_version": "v1", "epoch": 1, "mode": "sync"},
        "op_parked": {
            "phase": "after_dispatch",
            "handle": {
                "kind": "provider",
                "provider": "provider",
                "job_id": "job",
                "operation_id": oid,
                "attempt": attempt,
                "request_digest": digest(request),
                "evidence": "accepted",
                "query_ref": "query",
                "cancel_ref": None,
            },
            "poll_at_ms": None,
            "deadline_ms": None,
        },
        "op_completed": {
            "result": {"ok": True},
            "result_digest": digest({"ok": True}),
            "usage": {},
            "shared_state": None,
            "evidence": ["accepted"],
            "execution_started": True,
            "context": "normal",
            "request_digest": digest(request),
            "provider_binding": "provider",
        },
        "op_unknown": {
            "reason": "lost",
            "dispatch_evidence": ["d"],
            "observation": {},
            "retry": "retry",
            "retry_at_ms": 0,
            "duplicate_cost_risk": True,
            "measurement_missing": True,
        },
        "context_compacted": {
            "source_digest": digest({}),
            "prefix_ids": [],
            "tail_ids": [],
            "mode": "micro",
            "summary_operation_id": None,
            "replacement": [],
            "evidence_manifest": {},
        },
        "usage_observed": {"meter_id": "m", "observation": 1, "mode": "cumulative", "usage": {}, "source": "host"},
        "turn_ended": {
            "status": "completed",
            "reason": None,
            "result": {},
            "adopted_results": [],
            "budget": {},
            "unconfirmed_operations": [],
        },
    }
    payload = payloads[kind] | kw
    return make_record(
        kind,
        session_id=sid,
        turn_id=None if kind == "session_created" else tid,
        operation_id=oid if kind.startswith("op_") else None,
        attempt=attempt if kind.startswith("op_") else None,
        payload=payload,
    )


def base() -> list[Record]:
    return [record("session_created"), record("turn_started"), record("op_planned"), record("op_started")]
