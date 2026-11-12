"""Slow record-zero oracle, independent of the incremental context implementation.

Only lifecycle validation is reused, with both seed policies disabled. Projection,
source matching, tool rendering and coverage are computed here from public bytes.
"""

import difflib
import json
from dataclasses import replace
from unittest.mock import patch

from vv_agent.kernel import _reducer as reducer
from vv_agent.session.records import InboxItem, digest
from vv_agent.session.reducer import Fold
from vv_agent.types import Message, ToolExecutionResult


def plain_creation(row):
    if row.kind != "session_created":
        return row
    payload = row.payload
    seed = payload["attributes"].get("seed", {})
    seed.pop("unseen_input_protection", None)
    seed.pop("response_rejection", None)
    if not seed:
        payload["attributes"].pop("seed", None)
    return replace(row, payload=payload)


def nonempty(messages):
    return [m for m in messages if m.role != "assistant" or m.content or m.tool_calls or m.reasoning_content]


def key(message):
    return json.dumps(message.to_dict(), sort_keys=True, ensure_ascii=False)


def group(row, state, task):
    result = row.payload["result"]
    calls = result.get("tool_calls", [])
    messages = nonempty(
        [
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
                        **({"extra_content": c["extra_content"]} if "extra_content" in c else {}),
                    }
                    for c in calls
                ]
                or None,
            )
        ]
    )
    images = []
    for i, call in enumerate(calls):
        tool = state.operations.get(f"{row.operation_id}/attempt/{row.attempt}/tool/{i}")
        if tool is None:
            continue
        attempt = tool.attempts[tool.selected_attempt or max(tool.attempts)]
        if attempt.result and attempt.context == "normal":
            result = ToolExecutionResult.from_dict(attempt.result.payload["result"])
            messages.append(result.to_tool_message())
            if task.native_multimodal and (result.image_url or result.image_path):
                images.append(
                    Message(
                        "user", f"[Image loaded] {result.image_path}" if result.image_path else "", image_url=result.image_url
                    )
                )
        elif attempt.wait and "interaction_result" in attempt.wait:
            messages.append(ToolExecutionResult.from_dict(attempt.wait["interaction_result"]).to_tool_message())
        else:
            messages.append(
                Message(
                    "tool",
                    json.dumps({"error": "tool_outcome_unknown", "retryable": False}),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )
    messages.extend(images)
    if not calls and task.no_tool_policy == "continue" and int(row.operation_id.rsplit("/", 1)[1]) < task.max_cycles:
        boundary = state.boundaries.get((row.turn_id, "after_cycle", row.operation_id))
        if boundary is None or boundary.payload["data"]["action"] != "steer":
            messages.append(Message("user", "Continue working on the task."))
    return messages


def reference(records):
    """Reconstruct a single prefix from record zero; no incremental expected state."""
    fold = Fold()
    messages, origins, groups = [], [], {}
    covered, seen_turn, task = 0, False, None

    def atomic(position):
        for i, m in enumerate(messages):
            if m.tool_calls and i < position <= i + len(m.tool_calls):
                return i
        return position

    def remap(updated):
        nonlocal messages, origins, groups
        old, new = list(map(key, messages)), list(map(key, updated))
        mapping = {
            a + k: b + k
            for a, b, n in difflib.SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks()
            for k in range(n)
        }
        active = [p for s, p in origins if s > covered]
        if active:
            begin = atomic(min(active))
            suffix = old[begin:]
            offset = next((i for i in range(len(new) - len(suffix) + 1) if new[i : i + len(suffix)] == suffix), None)
            assert offset is not None, "replacement removed protected suffix"
            mapping.update({begin + k: offset + k for k in range(len(suffix))})
        origins = [(s, mapping[p]) for s, p in origins if p in mapping]
        groups = {
            oid: (row, mapping[a], mapping[b - 1] + 1)
            for oid, (row, a, b) in groups.items()
            if a < b and all(i in mapping and mapping[i] == mapping[a] + i - a for i in range(a, b))
        }
        messages = updated

    for seq, row in enumerate(records, 1):
        p, kind = row.payload, row.kind
        if kind == "context_compacted" or (kind == "boundary_recorded" and p["stage"] == "before_memory"):
            assert digest([m.to_dict() for m in messages]) == p["source_digest"]
        items = [InboxItem(**p["input"])] if kind == "input_applied" else []
        # Already-produced compactions are accepted as replacements, not replanned
        # under the legacy policy used solely to reconstruct operation lifecycle.
        require = reducer.require

        def lifecycle_require(condition, reason, require=require):
            # Legacy snapshot projection cannot validate the new callback source.
            # The oracle checks replacement bytes itself, independently below.
            if reason != "boundary source mismatch":
                require(condition, reason)

        with (
            patch.object(reducer, "_compacted", lambda state, record, history, **_ports: state.compactions.append(record)),
            patch.object(reducer, "require", lifecycle_require),
        ):
            fold.extend([plain_creation(row)], consumed_inputs=items)
        state = fold.state
        if kind == "session_created":
            messages = nonempty([Message.from_dict(m) for m in p["attributes"].get("seed", {}).get("messages", [])])
            first = next((i for i, m in enumerate(messages) if m.role != "system"), None)
            if first is not None:
                origins.append((seq, first))
        elif kind == "turn_started":
            task = row._task()
            control = task.metadata.get("_vv_agent_app_server_compaction_only")
            if control and seen_turn:
                continue
            history = nonempty(task.initial_messages) if task.initial_messages else messages
            updated = [
                Message("system", task.prompt_bundle.flatten(), metadata=task.metadata),
                *[m for m in history if m.role != "system"],
            ]
            initial = task.metadata.get("vv_session", {}).get("input_messages")
            additions = nonempty([Message.from_dict(m) for m in initial]) if initial else [Message("user", task.user_prompt)]
            if not control:
                updated += additions
            remap(updated)
            if not control:
                start = len(messages) - len(additions)
                if task.initial_messages or (not seen_turn and not covered):
                    start = next((i for i, m in enumerate(messages) if m.role != "system"), len(messages))
                if start < len(messages):
                    origins.append((seq, atomic(start)))
            seen_turn = True
        elif kind == "context_compacted" or (kind == "boundary_recorded" and p["stage"] == "before_memory"):
            remap(
                nonempty(
                    [Message.from_dict(m) for m in (p["replacement"] if kind == "context_compacted" else p["data"]["messages"])]
                )
            )
        elif kind == "input_applied" and p["disposition"] == "applied":
            ikind, target = p["input"]["kind"], p["target_operation_id"]
            content = p["input"]["payload"].get("content")
            added = []
            if ikind == "steer":
                added = (
                    nonempty([Message.from_dict(m) for m in content["messages"]])
                    if isinstance(content, dict) and "messages" in content
                    else [Message("user", str(content))]
                )
            elif ikind == "child_result" and p["reason"] == "background child notification":
                added = [Message("user", f"Child completed: {json.dumps(p['input']['payload'], ensure_ascii=False)}")]
            elif ikind == "user" and p["target_wait_id"] and target is None:
                added = [Message("user", str(content["text"]))]
            if added:
                begin = len(messages)
                messages += added
                origins.append((seq, atomic(begin)))
            elif target and (
                ikind in {"user", "child_result", "provider_result"}
                or (ikind == "approval_answer" and p["input"]["payload"]["decision"] in {"deny", "timeout"})
            ):
                op = state.operations[target]
                turn = state.turns[op.turn_id]
                normal = not (turn.cancelled or turn.ended or op.turn_id != state.active_turn_id)
                if ikind in {"child_result", "provider_result"}:
                    n = p["input"]["payload"]["attempt"]
                    normal = (
                        normal
                        and not op.attempts[n].unknown_consumed
                        and not any(a.started for k, a in op.attempts.items() if k > n)
                    )
                deps = op.attempts[1].plan.payload["dependencies"]
                if normal and op.kind != "model" and deps and deps[0] in groups:
                    origins.append((seq, groups[deps[0]][1]))
        elif kind == "boundary_recorded" and p["stage"] == "after_cycle" and not p["data"]["error"]:
            oid = p["source_operation_id"]
            if p["data"]["action"] == "steer" and oid in groups:
                row0, a, b = groups[oid]
                if messages[b - 1].content == "Continue working on the task.":
                    remap(messages[: b - 1] + messages[b:])
                    groups[oid] = (row0, a, b - 1)
            if p["data"]["steering_messages"]:
                begin = len(messages)
                messages += [Message("user", s) for s in p["data"]["steering_messages"]]
                origins.append((seq, atomic(begin)))
        if kind in {"op_planned", "op_parked", "op_completed"} and row.operation_id in state.operations:
            op = state.operations[row.operation_id]
            if op.kind != "model":
                deps = op.attempts[1].plan.payload["dependencies"]
                if deps and deps[0] in groups:
                    parent, a, b = groups[deps[0]]
                    rendered = group(parent, state, task)
                    shift = len(rendered) - (b - a)
                    origins = [(s, pos + shift if pos >= b else pos) for s, pos in origins]
                    groups = {
                        oid: (r0, x + shift if x >= b else x, y + shift if y > b else y) for oid, (r0, x, y) in groups.items()
                    }
                    messages[a:b] = rendered
                    groups[deps[0]] = (parent, a, a + len(rendered))
            if kind == "op_completed":
                if (
                    op.kind == "model"
                    and op.selected_attempt == row.attempt
                    and p["context"] == "normal"
                    and not p["result"].get("error_code")
                    and op.attempts[row.attempt].plan.payload["purpose"] == "primary"
                ):
                    covered = max(covered, int(op.attempts[row.attempt].plan.payload["context_version"]))
                    begin = len(messages)
                    messages += group(row, state, task)
                    groups[row.operation_id] = (row, begin, len(messages))
                elif op.kind != "model" and p["context"] == "correction":
                    begin = len(messages)
                    messages.append(Message("user", f"Correction for {row.operation_id}: {p['result']['content']}"))
                    origins.append((seq, atomic(begin)))
        origins = [(s, p0) for s, p0 in origins if s > covered]
    return min((p for _, p in origins), default=None), [m.to_dict() for m in messages]
