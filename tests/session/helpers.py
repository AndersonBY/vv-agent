from typing import Any

try:
    from support.s1_fixture_values import (
        content_ref,
        create_fixture_session,
        fixture_child_spec,
        fixture_content_reader,
        fixture_key_provider,
        principal,
        push_fixture_input,
        read_fixture_state,
        wrap_fixture_store,
    )
except ModuleNotFoundError:
    from scripts.s1_fixture_values import (
        content_ref,
        create_fixture_session,
        fixture_child_spec,
        fixture_content_reader,
        fixture_key_provider,
        principal,
        push_fixture_input,
        read_fixture_state,
        wrap_fixture_store,
    )
from vv_agent.session.records import InboxItem, Record, digest, make_record

__all__ = [
    "base",
    "content_ref",
    "create_fixture_session",
    "fixture_child_spec",
    "fixture_content_reader",
    "fixture_key_provider",
    "item",
    "principal",
    "push_fixture_input",
    "read_fixture_state",
    "record",
    "runtime_turn_record",
    "wrap_fixture_store",
]


def item(input_id: str = "i", kind: str = "user", **kw: Any) -> InboxItem:
    payload = {"content": content_ref("hello")}
    if kind == "control":
        payload = {"action": kw.pop("action", "cancel")}
    actor = kw.pop("actor", principal(input_id, "service" if kind == "provider_evidence" else "user"))
    admission_proof = kw.pop("admission_proof", None)
    return InboxItem(
        input_id=input_id,
        kind=kind,
        payload=payload,
        actor=actor,
        admission_proof=admission_proof,
        **kw,
    )


def record(kind: str, *, sid: str = "s", tid: str = "t", oid: str = "o", attempt: int = 1, **kw: Any) -> Record:
    request = content_ref("request")
    result = content_ref("result")
    payloads: dict[str, dict[str, Any]] = {
        "session_created": {
            "principal": principal(),
            "workspace": content_ref("workspace"),
            "parent_session_id": None,
            "parent_operation_id": None,
            "attributes": content_ref("attributes"),
        },
        "turn_started": {
            "input_ids": [],
            "definition": content_ref("definition"),
            "definition_digest": digest(content_ref("definition")),
            "handler_version": "1",
            "budget": content_ref("budget"),
            "binding": "task",
            "generation": 1,
            "policy_digest": None,
            "requirements": {"level": 1, "handler_version": "1", "capabilities": []},
            "tokenizer_binding": None,
            "environment": None,
        },
        "input_applied": {
            "input": item().to_dict(),
            "input_digest": digest(item().to_dict()),
            "input_proof": None,
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
            "budget_admission": content_ref("budget-admission"),
            "not_before_ms": None,
            "provider_binding": "provider",
            "consumed_unknowns": [],
            "policy_decision": None,
        },
        "op_started": {"dispatch_id": "d", "authorization_version": "v1", "epoch": 1, "mode": "sync"},
        "op_parked": {
            "phase": "after_dispatch",
            "handle": {
                "kind": "provider",
                "provider": "provider",
                "job_id": content_ref("job"),
                "operation_id": oid,
                "attempt": attempt,
                "request_digest": digest(request),
                "evidence": content_ref("evidence"),
                "query_ref": content_ref("query"),
                "cancel_ref": None,
            },
            "poll_at_ms": None,
            "deadline_ms": None,
        },
        "op_completed": {
            "result": result,
            "result_digest": digest(result),
            "usage": content_ref("usage"),
            "shared_state": None,
            "evidence": content_ref("evidence"),
            "execution_started": True,
            "context": "normal",
            "request_digest": digest(request),
            "provider_binding": "provider",
            "policy_decision": None,
        },
        "op_unknown": {
            "reason": content_ref("lost"),
            "dispatch_evidence": content_ref("dispatch"),
            "observation": content_ref("observation"),
            "retry": "retry",
            "retry_at_ms": 0,
            "duplicate_cost_risk": True,
            "measurement_missing": True,
        },
        "context_compacted": {
            "source_digest": digest(content_ref("source")),
            "prefix_ids": [],
            "tail_ids": [],
            "mode": "micro",
            "summary_operation_id": None,
            "replacement": [],
            "evidence_manifest": content_ref("manifest"),
        },
        "usage_observed": {
            "meter_id": "m",
            "observation": 1,
            "mode": "cumulative",
            "usage": content_ref("usage"),
            "source": content_ref("host"),
        },
        "turn_ended": {
            "status": "completed",
            "reason": None,
            "result": None,
            "adopted_results": [],
            "budget": content_ref("budget"),
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


def runtime_turn_record(tid: str = "t") -> Record:
    """Keep transaction probes at the logical boundary until hydration lands."""
    return record("turn_started", tid=tid)
