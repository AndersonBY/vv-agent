"""Growing full-message snapshots through the normal record and append validators."""

from vv_agent.session.records import SessionSpec, digest, make_record
from vv_agent.types import Message, ToolCall, ToolExecutionResult

from .helpers import record
from .test_recovery_matrix import runtime


def growing_history(steps: int, *, initial_bytes: int = 128 * 1024):
    rt = runtime(None, [])
    task = rt.compile("x" * initial_bytes, "t")
    definition = rt.definition(task)
    rows = [record("turn_started", definition=definition, definition_digest=digest(definition))]
    messages = [Message("system", task.prompt_bundle.flatten(), metadata=task.metadata), Message("user", task.user_prompt)]
    for cycle in range(1, steps + 1):
        oid = f"t/model/{cycle}"
        snapshot = [m.to_dict() for m in messages]
        rows.append(
            make_record(
                "boundary_recorded",
                session_id="s",
                turn_id="t",
                payload={
                    "stage": "before_memory",
                    "boundary_id": str(cycle),
                    "source_operation_id": None,
                    "source_digest": digest(snapshot),
                    "data": {"messages": snapshot, "shared_state": {}},
                },
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
                record("op_planned", oid=oid, request=request, request_digest=checksum),
                record("op_started", oid=oid),
                record("op_completed", oid=oid, request_digest=checksum, result=response, result_digest=digest(response)),
                record(
                    "op_planned",
                    oid=tool_id,
                    op_kind="tool",
                    purpose=None,
                    request=tool_request,
                    request_digest=digest(tool_request),
                    dependencies=[oid],
                ),
                record("op_started", oid=tool_id),
                record(
                    "op_completed",
                    oid=tool_id,
                    request_digest=digest(tool_request),
                    result=result.to_dict(),
                    result_digest=digest(result.to_dict()),
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


def persist_history(store, steps: int, *, initial_bytes: int = 128 * 1024):
    rows, messages = growing_history(steps, initial_bytes=initial_bytes)
    with store.atomic() as tx:
        tx.create(SessionSpec("s", "test", "/tmp"), consumers=())
    lease = store.acquire("s", owner="profile", ttl_ms=3_600_000)
    assert lease is not None
    with store.atomic() as tx:
        tx.append("s", lease=lease, expected_seq=1, commit_id="history", records=rows)
    store.release(lease)
    return messages
