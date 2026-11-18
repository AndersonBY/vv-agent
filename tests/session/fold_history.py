"""Growing full-message snapshots for logical folds and S1 durable appends."""

from contextlib import contextmanager
from unittest.mock import patch

from support.s1_fixture_values import append_fixture_records

from vv_agent.kernel._budget_values import RunBudgetLimits
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._record_codec import _load
from vv_agent.session.records import InboxItem, Record, copy_json, digest, record_identity
from vv_agent.session.reducer import Fold
from vv_agent.types import AgentTask, Message, ToolCall, ToolExecutionResult

from .helpers import create_fixture_session, principal, record
from .test_recovery_matrix import runtime


@contextmanager
def logical_record_codec():
    """Use the pre-storage canonical JSON contract for reducer-only fixtures."""

    def encode(value):
        return canonical_json_bytes(value.to_dict())

    with patch.object(Record, "encode", encode), patch.object(InboxItem, "encode", encode):
        yield


def extend_logical(fold: Fold, records, *, consumed_inputs=()):
    records = tuple(records)
    with logical_record_codec():
        return fold.extend(
            records,
            consumed_inputs=consumed_inputs,
            bodies=(canonical_json_bytes(record.to_dict()) for record in records),
        )


def logical_input_digest(item: InboxItem) -> str:
    return digest(item.to_dict())


def parse_logical_record(source: bytes | str, *, cache=None) -> Record:
    """Parse reducer-only logical fixtures without entering the S1 record codec."""
    body = source.encode() if isinstance(source, str) else source
    value = _load(body)
    payload = value["payload"]
    if cache is None:
        from vv_agent.session.records import _MessageCache

        cache = _MessageCache()
    logical_messages = []
    if value["kind"] == "op_planned" and isinstance(payload.get("request"), dict):
        logical_messages = payload["request"].get("messages", [])
    elif value["kind"] == "boundary_recorded" and isinstance(payload.get("data"), dict):
        logical_messages = payload["data"].get("messages", [])
    for message in logical_messages:
        cache.parse(cache.share(message))
    request = payload.get("request")
    if isinstance(request, dict) and digest(request) != payload.get("request_digest"):
        raise ValueError("request digest mismatch")
    return _retain_logical_payload(Record(**value), body=bytes(body))


def attach_logical_task(record: Record) -> Record:
    if record.kind == "turn_started":
        definition = record.payload["definition"]
        object.__setattr__(record, "_task_value", AgentTask.from_dict(definition["task"]))
    return record


def _retain_logical_payload(record: Record, *, body: bytes | None = None) -> Record:
    value = copy_json(record.to_dict())
    canonical = canonical_json_bytes(value)
    retained = body if body is not None and canonical_json_bytes(_load(body)) == body else canonical
    object.__setattr__(record, "_value", value)
    object.__setattr__(record, "_body", retained)
    object.__setattr__(record, "_digest", digest(value))
    object.__setattr__(record, "payload", {})
    return attach_logical_task(record)


def logical_input(
    input_id: str,
    kind: str,
    payload: dict,
    target_turn_id: str | None = None,
    generation: int | None = None,
) -> InboxItem:
    subject_kind = "service" if kind in {"provider_result", "provider_evidence", "child_result"} else "user"
    return InboxItem(
        input_id,
        kind,
        payload,
        target_turn_id=target_turn_id,
        generation=generation,
        actor=principal(f"fixture/{input_id}", subject_kind),
    )


def logical_record(kind: str, *, sid: str = "s", tid: str = "t", oid: str = "o", attempt: int = 1, **kw):
    """Build an encoder-free Record for reducer-only tests."""

    try:
        base = record(kind, sid=sid, tid=tid, oid=oid, attempt=attempt)
    except KeyError:
        result = Record(
            record_identity(
                kind,
                sid,
                tid,
                oid if kind.startswith("op_") else None,
                attempt if kind.startswith("op_") else None,
                kw,
            ),
            kind,
            sid,
            tid if kind != "session_created" else None,
            oid if kind.startswith("op_") else None,
            attempt if kind.startswith("op_") else None,
            kw,
        )
    else:
        payload = base.payload | kw
        result = Record(
            record_identity(
                kind,
                sid,
                tid if kind != "session_created" else None,
                oid if kind.startswith("op_") else None,
                attempt if kind.startswith("op_") else None,
                payload,
            ),
            kind,
            sid,
            tid if kind != "session_created" else None,
            oid if kind.startswith("op_") else None,
            attempt if kind.startswith("op_") else None,
            payload,
        )
    return _retain_logical_payload(result)


def logical_session_record(
    sid: str = "s",
    *,
    actor=None,
    workspace="/tmp",
    attributes: dict | None = None,
    preferred_payload: dict | None = None,
) -> Record:
    """Build a logical creation record while the S1 producer keeps refs on disk."""

    payload = preferred_payload or {
        "principal": actor if actor is not None else principal("test"),
        "workspace": workspace,
        "parent_session_id": None,
        "parent_operation_id": None,
        "attributes": attributes
        if attributes is not None
        else {
            "seed": {
                "messages": [],
                "shared_state": {},
                "unseen_input_protection": True,
                "response_rejection": True,
            }
        },
    }
    return _retain_logical_payload(
        Record(
            record_identity("session_created", sid, None, None, None, payload),
            "session_created",
            sid,
            None,
            None,
            None,
            payload,
        )
    )


def growing_history(steps: int, *, initial_bytes: int = 128 * 1024, steer: bool = False, usage: int = 0):
    rt = runtime(None, [])
    task = rt.compile("x" * initial_bytes, "t")
    definition = rt.definition(task)
    rows = [
        logical_record(
            "turn_started",
            definition=definition,
            definition_digest=digest(definition),
            budget=RunBudgetLimits().to_dict(),
        )
    ]
    messages = [Message("system", task.prompt_bundle.flatten(), metadata=task.metadata), Message("user", task.user_prompt)]
    for cycle in range(1, steps + 1):
        oid = f"t/model/{cycle}"
        if steer:
            item = InboxItem(
                f"steer-{cycle}",
                "steer",
                {"content": f"UNSEEN-{cycle}"},
                actor=principal(),
            )
            rows.append(logical_record("input_applied", input=item.to_dict(), input_digest=logical_input_digest(item)))
            messages.append(Message("user", item.payload["content"]))
        for _ in range(usage):
            rows.append(record("usage_observed", meter_id="padding", observation=len(rows)))
        snapshot = [m.to_dict() for m in messages]
        rows.append(
            logical_record(
                "boundary_recorded",
                stage="before_memory",
                boundary_id=str(cycle),
                source_operation_id=None,
                source_digest=digest(snapshot),
                data={"messages": snapshot, "shared_state": {}},
            )
        )
        request = {"messages": snapshot, "model": "scripted", "metadata": {}}
        checksum = digest(request)
        tool_id = f"{oid}/attempt/1/tool/0"
        call = ToolCall(f"call-{cycle}", "read", {})
        response = {"content": "", "tool_calls": [call.to_dict()]}
        result = ToolExecutionResult(tool_call_id=call.id, content=f"{cycle}:" + "y" * 128)
        tool_request = call.to_dict()
        rows.extend(
            [
                logical_record(
                    "op_planned",
                    oid=oid,
                    request=request,
                    request_digest=checksum,
                    context_version=str(len(rows) + 1),
                    budget_admission={},
                ),
                logical_record("op_started", oid=oid),
                logical_record(
                    "op_completed",
                    oid=oid,
                    request_digest=checksum,
                    result=response,
                    result_digest=digest(response),
                    usage={},
                    evidence="proof",
                ),
                logical_record(
                    "op_planned",
                    oid=tool_id,
                    op_kind="tool",
                    purpose=None,
                    request=tool_request,
                    request_digest=digest(tool_request),
                    dependencies=[oid],
                    budget_admission={},
                ),
                logical_record("op_started", oid=tool_id),
                logical_record(
                    "op_completed",
                    oid=tool_id,
                    request_digest=digest(tool_request),
                    result=result.to_dict(),
                    result_digest=digest(result.to_dict()),
                    usage={},
                    evidence="proof",
                ),
            ]
        )
        messages.append(
            Message(
                "assistant",
                "",
                tool_calls=[
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            )
        )
        messages.append(result.to_tool_message())
    return tuple(rows), messages


def persist_history(store, steps: int, *, initial_bytes: int = 128 * 1024, enabled: bool):
    rows, messages = growing_history(steps, initial_bytes=initial_bytes)
    create_fixture_session(
        store,
        "s",
        attributes={"seed": {"messages": [], "shared_state": {}, "unseen_input_protection": enabled}},
    )
    append_fixture_records(store, "s", rows, commit_id="history")
    return messages
