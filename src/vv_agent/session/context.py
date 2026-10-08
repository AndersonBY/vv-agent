"""Model context is a projection; accepted replacements never rewrite the raw log."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from vv_agent.types import AgentTask, Message, ToolExecutionResult

from .records import digest
from .store import StoredRecord

if TYPE_CHECKING:
    from .reducer import ExecutionState


def project_context(records: tuple[StoredRecord, ...], state: ExecutionState) -> list[Message]:
    messages: list[Message] = []
    for stored in records:
        r, p = stored.record, stored.record.payload
        if r.kind == "context_compacted":
            messages = [Message.from_dict(m) for m in p["replacement"]]
        elif r.kind == "turn_started":
            task = AgentTask.from_dict(p["definition"]["task"])
            messages.extend(task.initial_messages)
            messages.append(Message("user", task.user_prompt))
        elif r.kind == "input_applied" and p["disposition"] == "applied" and p["input"]["kind"] == "steer":
            messages.append(Message("user", str(p["input"]["payload"]["content"])))
        elif (
            r.kind == "input_applied"
            and p["disposition"] == "applied"
            and p["input"]["kind"] == "child_result"
            and p["reason"] == "background child notification"
        ):
            messages.append(Message("user", f"Child completed: {json.dumps(p['input']['payload'], ensure_ascii=False)}"))
        elif r.kind == "op_completed" and r.operation_id:
            assert r.attempt is not None
            op = state.operations[r.operation_id]
            if op.kind != "model" and p["context"] == "correction":
                messages.append(Message("user", f"Correction for {r.operation_id}: {p['result']['content']}"))
            if (
                op.kind != "model"
                or p["context"] != "normal"
                or op.selected_attempt != r.attempt
                or op.attempts[r.attempt].plan.payload["purpose"] != "primary"
                or p["result"].get("error_code")
            ):
                continue
            result = p["result"]
            calls = result.get("tool_calls", [])
            messages.append(
                Message(
                    "assistant",
                    result["content"],
                    reasoning_content=result.get("reasoning_content"),
                    tool_calls=[
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
                        }
                        for c in calls
                    ]
                    or None,
                )
            )
            for i, call in enumerate(calls):
                tool = state.operations.get(f"{r.operation_id}/attempt/{r.attempt}/tool/{i}")
                if tool is None:
                    continue
                a = tool.attempts[tool.selected_attempt or max(tool.attempts)]
                if a.result and a.context == "normal":
                    result = ToolExecutionResult.from_dict(a.result.payload["result"])
                    message = result.to_tool_message()
                    message.name = call["name"]
                    messages.append(message)
                else:
                    content = json.dumps({"error": "tool_outcome_unknown", "retryable": False})
                    messages.append(Message("tool", content, tool_call_id=call["id"], name=call["name"]))
    return messages


def message_ids(messages: list[Message]) -> list[str]:
    return [f"{i}/{digest(m.to_dict())}" for i, m in enumerate(messages)]
