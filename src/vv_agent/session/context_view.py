"""Read-only App Server context projection from a captured log and inbox cut."""

from __future__ import annotations

import json
from typing import Any

from vv_agent.memory.manager import MemoryManager
from vv_agent.memory.token_utils import _estimate_tokens
from vv_agent.types import Message

from .compaction import memory_settings
from .context import project_context
from .records import InboxItem, digest
from .reducer import fold
from .store import Conflict
from .summary_requests import compaction_request


class ContextSnapshotBusy(RuntimeError):
    pass


def context_view(store, session_id: str) -> dict[str, Any]:
    # Capture records and pending inbox together without locking the session row.
    for _ in range(8):
        try:
            first = store.read(session_id)
        except Conflict as exc:
            if str(exc) == "session or cursor not found":
                raise KeyError(session_id) from exc
            raise
        records = first.records
        while records[-1].seq < first.head_seq:
            records += store.read(session_id, after_seq=records[-1].seq, through_seq=first.head_seq).records
        inbox = store.peek_inbox(session_id, through_input_seq=first.inbox_seq, limit=max(1, first.inbox_seq), ready_only=False)
        check = store.read(session_id, limit=1)
        if (check.head_seq, check.inbox_seq) == (first.head_seq, first.inbox_seq):
            break
    else:
        raise ContextSnapshotBusy("context_snapshot_busy")
    return project_view(session_id, records, inbox)


def context_messages(records, state, definition):
    messages = project_context(records, state)
    attributes = records[0].record._payload["attributes"]
    if not state.turns and definition and "app_server_fork" not in attributes:
        from vv_agent.types import AgentTask

        task = AgentTask.from_dict(definition["task"])
        messages = [Message("system", task.prompt_bundle.flatten()), *[m for m in messages if m.role != "system"]]
    return messages


def project_view(session_id, records, inbox=()):
    state = fold(
        (r.record for r in records),
        consumed_inputs=(InboxItem(**r.record._payload["input"]) for r in records if r.record.kind == "input_applied"),
    )
    attributes = records[0].record._payload["attributes"]
    if "app_server" not in attributes:
        raise KeyError(session_id)
    pending = [
        InboxItem(**r._payload["input"])
        for key, r in state.applied_inputs.items()
        if r._payload["disposition"] == "queued" and key not in state.admitted_inputs
    ]
    # A consumed input can overlap the captured prefix and the later inbox read.
    pending.extend(x.item for x in inbox if x.item.input_id not in state.applied_inputs)
    pending = [x for x in pending if x.kind in {"user", "steer", "follow_up", "model_selection"} and not compaction_request(x)]
    start = state.turns[next(reversed(state.turns))].start if state.turns else None
    definition = start._payload["definition"] if start else attributes.get("app_server_context")
    selection = state.model_selection
    if selection is not None:
        effective = selection._payload["input"]["payload"]["selection"] | {"binding": selection._payload["model_binding"]}
    elif attributes.get("app_server_fork"):
        effective = attributes["app_server_fork"]["effective_model"]
    elif definition:
        task = definition["task"]
        effective = {"model": task["model"], "model_settings": task.get("model_settings"), "binding": definition["model_binding"]}
    else:
        effective = None
    tools = definition["tools"] if definition else None
    baseline = attributes.get("app_server_context") if "app_server_fork" in attributes else None
    settings = memory_settings(baseline or definition, selection) if definition else None
    manager = MemoryManager(**settings) if settings else None
    binding = effective["binding"] if effective else {}
    messages = context_messages(records, state, definition)
    summary = [m for m in messages if m.name == "memory_summary"]
    instructions = [m for m in messages if m.role == "system" and m.name != "memory_summary"]
    history = [m for m in messages if m.role != "system" and m.name != "memory_summary"]

    def estimate(value):
        return _estimate_tokens(json.dumps(value, ensure_ascii=False, separators=(",", ":"))) if value else 0

    def message_tokens(part):
        return estimate([m.to_openai_message() for m in part])

    pending_messages = []
    for item in pending:
        if item.kind == "model_selection":
            continue
        content = item.payload["content"]
        if isinstance(content, dict) and "messages" in content:
            pending_messages.extend(Message.from_dict(m) for m in content["messages"])
        else:
            pending_messages.append(
                Message("user", content["text"] if isinstance(content, dict) and "text" in content else str(content))
            )
    parts = {
        "instructions": message_tokens(instructions) if definition or instructions else None,
        "tools": estimate(tools) if tools is not None else None,
        "history": message_tokens(history),
        "summary": message_tokens(summary),
        "pendingInput": message_tokens(pending_messages),
    }
    compact = next(
        (r for r in reversed(records) if r.record.kind == "context_compacted" and r.record._payload["mode"] != "micro"), None
    )
    last = attributes.get("app_server_fork", {}).get("last_compaction")
    if compact:
        payload = compact.record._payload
        last = {
            "seq": compact.seq,
            "turnId": compact.record.turn_id,
            "createdAt": compact.created_ms,
            "replacedMessageCount": len(payload["prefix_ids"]),
            "summary": "\n".join(m.get("content", "") for m in payload["replacement"] if m.get("name") == "memory_summary"),
        }
    raw = project_context(records, state, compacted=False)
    return {
        "threadId": session_id,
        **({"lineage": attributes["app_server_fork"]["lineage"]} if "app_server_fork" in attributes else {}),
        "contextVersion": digest(
            {
                "messages": [m.to_openai_message() for m in messages],
                "tools": tools,
                "effectiveModel": effective,
                "memorySettings": settings,
                "pending": [x.to_dict() for x in pending],
            }
        ),
        "projectionStage": "durable_pre_hooks",
        "estimates": {
            "estimated": True,
            "estimator": "local_cjk_character_estimate_v1",
            "parts": parts,
            "total": sum(v for v in parts.values() if v is not None) if all(v is not None for v in parts.values()) else None,
        },
        "limits": {
            "contextWindow": binding.get("context_length"),
            "maxOutputTokens": binding.get("max_output_tokens"),
            "runtimeContextWindow": manager.model_context_window if manager else None,
            "outputReserve": manager.reserved_output_tokens if manager else None,
            "outputReserveSource": manager.reserved_output_source if manager else None,
            "compactionThreshold": manager.autocompact_threshold if manager else None,
        },
        "lastCompaction": last,
        "counts": {
            "rawRecordCount": len(records),
            "rawHistoryMessages": sum(m.role != "system" for m in raw),
            "modelVisibleMessages": len(messages),
            "modelVisibleHistoryMessages": len(history),
            "pendingInputs": sum(x.kind != "model_selection" for x in pending),
        },
    }
