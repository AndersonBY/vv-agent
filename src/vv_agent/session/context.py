"""Model context is a projection; accepted replacements never rewrite the raw log."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING

from vv_agent.runtime.tool_call_runner import ToolCallRunner
from vv_agent.types import Message, ToolExecutionResult

from .records import copy_json, digest
from .store import StoredRecord

if TYPE_CHECKING:
    from .reducer import ExecutionState


def project_context(records: tuple[StoredRecord, ...], state: ExecutionState) -> list[Message]:
    seed = records[0].record._payload["attributes"].get("seed", {}) if records else {}
    messages: list[Message] = [Message.from_dict(copy_json(m)) for m in seed.get("messages", [])]
    for stored in records:
        r = stored.record
        if r.kind not in {"context_compacted", "boundary_recorded", "turn_started", "input_applied", "op_completed"}:
            continue
        p = r._payload
        if r.kind == "boundary_recorded":
            if p["stage"] == "before_memory":
                messages = [Message.from_dict(copy_json(m)) for m in p["data"]["messages"]]
            elif p["stage"] == "after_cycle" and not p["data"]["error"]:
                messages.extend(Message("user", text) for text in p["data"]["steering_messages"])
        elif r.kind == "context_compacted":
            messages = [Message.from_dict(copy_json(m)) for m in p["replacement"]]
        elif r.kind == "turn_started":
            task = r._task()
            history = deepcopy(task.initial_messages) if task.initial_messages else messages
            messages = [
                Message("system", task.prompt_bundle.flatten(), metadata=copy_json(task.metadata)),
                *[m for m in history if m.role != "system"],
            ]
            initial_input = task.metadata.get("vv_session", {}).get("input_messages")
            messages.extend(
                [Message.from_dict(m) for m in initial_input] if initial_input else [Message("user", task.user_prompt)]
            )
        elif r.kind == "input_applied" and p["disposition"] == "applied" and p["input"]["kind"] == "steer":
            content = p["input"]["payload"]["content"]
            messages.extend(
                [Message.from_dict(m) for m in content["messages"]]
                if isinstance(content, dict) and "messages" in content
                else [Message("user", str(content))]
            )
        elif (
            r.kind == "input_applied"
            and p["disposition"] == "applied"
            and p["input"]["kind"] == "child_result"
            and p["reason"] == "background child notification"
        ):
            messages.append(Message("user", f"Child completed: {json.dumps(p['input']['payload'], ensure_ascii=False)}"))
        elif (
            r.kind == "input_applied"
            and p["disposition"] == "applied"
            and p["input"]["kind"] == "user"
            and p["target_wait_id"]
            and p["target_operation_id"] is None
        ):
            messages.append(Message("user", str(p["input"]["payload"]["content"]["text"])))
        elif r.kind == "op_completed" and r.operation_id:
            assert r.attempt is not None
            op = state.operations[r.operation_id]
            if op.kind != "model" and p["context"] == "correction":
                messages.append(Message("user", f"Correction for {r.operation_id}: {p['result']['content']}"))
            if (
                op.kind != "model"
                or p["context"] != "normal"
                or op.selected_attempt != r.attempt
                or op.attempts[r.attempt].plan._payload["purpose"] != "primary"
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
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(c["arguments"], ensure_ascii=False, separators=(",", ":")),
                            },
                            **({"extra_content": copy_json(c["extra_content"])} if "extra_content" in c else {}),
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
                    result = ToolExecutionResult.from_dict(copy_json(a.result._payload["result"]))
                    message = result.to_tool_message()
                    messages.append(message)
                    image = ToolCallRunner._build_image_notification(result=result, include_image=task.native_multimodal)
                    if image is not None:
                        messages.append(image)
                elif a.wait and "interaction_result" in a.wait:
                    messages.append(ToolExecutionResult.from_dict(copy_json(a.wait["interaction_result"])).to_tool_message())
                else:
                    content = json.dumps({"error": "tool_outcome_unknown", "retryable": False})
                    messages.append(Message("tool", content, tool_call_id=call["id"], name=call["name"]))
    return messages


def message_ids(messages: list[Message]) -> list[str]:
    return [f"{i}/{digest(m.to_dict())}" for i, m in enumerate(messages)]
