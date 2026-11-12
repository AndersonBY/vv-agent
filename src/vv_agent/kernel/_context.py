"""Model context is a projection; accepted replacements never rewrite the raw log."""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from heapq import heapify, heappop, heappush
from typing import TYPE_CHECKING

from vv_agent.kernel._context_messages import atomic_start, filter_empty_assistant_messages, retained_start
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._message_values import Message
from vv_agent.kernel._records import _MessageCache, copy_json, digest
from vv_agent.kernel._store import StoredRecord
from vv_agent.kernel._tool_behavior import build_image_notification
from vv_agent.kernel._tool_values import ToolExecutionResult

if TYPE_CHECKING:
    from vv_agent.kernel._state import ExecutionState


def project_context(
    records: Sequence[StoredRecord],
    state: ExecutionState,
    *,
    compacted: bool = True,
    _message_cache: _MessageCache | None = None,
    _initial: list[Message] | None = None,
    _task: AgentTask | None = None,
    _seen_turn: bool = False,
) -> list[Message]:
    if compacted and _initial is None and state._protection is not None:
        if not records:
            return []
        assert records[-1].seq == state._protection.prefix_seq, "context prefix is not aligned"
        return deepcopy(state._protection.messages)

    def decode(value):
        return _message_cache.parse(value) if _message_cache is not None else Message.from_dict(copy_json(value))

    start = 0
    task = _task
    if compacted and _initial is None:
        for index in range(len(records) - 1, -1, -1):
            r = records[index].record
            if r.kind == "context_compacted" or (r.kind == "boundary_recorded" and r._payload["stage"] == "before_memory"):
                start = index
                assert r.turn_id is not None
                task = state.turns[r.turn_id].start._task()
                break
    seed = (
        records[0].record._payload["attributes"].get("seed", {})
        if records and not start and _initial is None and records[0].record.kind == "session_created"
        else {}
    )
    messages: list[Message] = (
        filter_empty_assistant_messages([Message.from_dict(copy_json(m)) for m in seed["messages"]])
        if seed.get("messages")
        else []
    )
    if _initial is not None:
        messages = _initial
    seen_turn = bool(start) or _seen_turn
    for index in range(start, len(records)):
        r = records[index].record
        if r.kind not in {"context_compacted", "boundary_recorded", "turn_started", "input_applied", "op_completed"}:
            continue
        p = r._payload
        if r.kind == "boundary_recorded":
            if p["stage"] == "before_memory" and compacted:
                messages = filter_empty_assistant_messages([decode(m) for m in p["data"]["messages"]])
            elif p["stage"] == "after_cycle" and not p["data"]["error"]:
                messages.extend(Message("user", text) for text in p["data"]["steering_messages"])
        elif r.kind == "context_compacted" and compacted:
            messages = filter_empty_assistant_messages([decode(m) for m in p["replacement"]])
        elif r.kind == "turn_started":
            task = r._task()
            if task.metadata.get("_vv_agent_app_server_compaction_only") and seen_turn:
                continue
            seen_turn = True
            history = filter_empty_assistant_messages(deepcopy(task.initial_messages)) if task.initial_messages else messages
            messages = [
                Message("system", task.prompt_bundle.flatten(), metadata=copy_json(task.metadata)),
                *[m for m in history if m.role != "system"],
            ]
            initial_input = task.metadata.get("vv_session", {}).get("input_messages")
            messages.extend(
                []
                if task.metadata.get("_vv_agent_app_server_compaction_only")
                else filter_empty_assistant_messages([Message.from_dict(m) for m in initial_input])
                if initial_input
                else [Message("user", task.user_prompt)]
            )
        elif r.kind == "input_applied" and p["disposition"] == "applied" and p["input"]["kind"] == "steer":
            content = p["input"]["payload"]["content"]
            messages.extend(
                filter_empty_assistant_messages([Message.from_dict(m) for m in content["messages"]])
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
            assert task is not None
            assert r.attempt is not None and r.turn_id is not None
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
            assistant = Message(
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
            if isinstance(assistant.content, str) and assistant.content.strip():
                messages.append(assistant)
            else:
                messages.extend(filter_empty_assistant_messages([assistant]))
            images = []
            for i, call in enumerate(calls):
                tool = state.operations.get(f"{r.operation_id}/attempt/{r.attempt}/tool/{i}")
                if tool is None:
                    continue
                a = tool.attempts[tool.selected_attempt or max(tool.attempts)]
                if a.result and a.context == "normal":
                    result = ToolExecutionResult.from_dict(copy_json(a.result._payload["result"]))
                    message = result.to_tool_message()
                    messages.append(message)
                    image = build_image_notification(result=result, include_image=task.native_multimodal)
                    if image is not None:
                        images.append(image)
                elif a.wait and "interaction_result" in a.wait:
                    messages.append(ToolExecutionResult.from_dict(copy_json(a.wait["interaction_result"])).to_tool_message())
                else:
                    content = json.dumps({"error": "tool_outcome_unknown", "retryable": False})
                    messages.append(Message("tool", content, tool_call_id=call["id"], name=call["name"]))
            messages.extend(images)
            if (
                not calls
                and task.no_tool_policy == "continue"
                and int(r.operation_id.rsplit("/", 1)[1]) < task.max_cycles
                and (
                    (boundary := state.boundaries.get((r.turn_id, "after_cycle", r.operation_id))) is None
                    or boundary._payload["data"]["action"] != "steer"
                )
            ):
                messages.append(Message("user", "Continue working on the task."))
    return messages


def message_ids(messages: list[Message]) -> list[str]:
    return [f"{i}/{digest(m.to_dict())}" for i, m in enumerate(messages)]


@dataclass
class InputProtection:
    """Disposable projection and source positions, rebuilt with the validated prefix."""

    covered: int = 0
    prefix_seq: int = 0
    # A position heap, with a lazy common shift for suffix-preserving replacements.
    inputs: list[tuple[int, int]] = field(default_factory=list)
    input_offset: int = 0
    messages: list[Message] = field(default_factory=list)
    groups: dict[str, tuple[StoredRecord, int, int]] = field(default_factory=dict)
    task: AgentTask | None = None
    seen_turn: bool = False

    def fork(self):
        from dataclasses import replace

        return replace(self, inputs=self.inputs.copy(), messages=self.messages.copy(), groups=self.groups.copy())

    @property
    def start(self) -> int | None:
        while self.inputs and self.inputs[0][1] <= self.covered:
            heappop(self.inputs)
        return self.inputs[0][0] + self.input_offset if self.inputs else None

    def protect(self, seq: int, position: int) -> None:
        heappush(self.inputs, (position - self.input_offset, seq))

    def shift_after(self, end: int, shift: int) -> None:
        if self.inputs and shift:
            self.inputs = [(pos + shift if pos + self.input_offset >= end else pos, seq) for pos, seq in self.inputs]
            heapify(self.inputs)

    def replace(self, replacement: list[Message]) -> None:
        if self.messages == replacement:
            self.messages = replacement
            return
        start = self.start
        boundary = atomic_start(self.messages, start) if start is not None else None
        offset = retained_start(self.messages, replacement, start)
        if start is not None:
            assert offset is not None
            assert boundary is not None
            shift = offset - boundary
            self.input_offset += shift
        # Map by preserved source positions, never reused provider tool-call IDs.
        # Match whole retained blocks, including groups outside the protected tail.
        old = [digest(m.to_dict()) for m in self.messages]
        new = [digest(m.to_dict()) for m in replacement]
        mapping = {
            a + k: b + k for a, b, n in SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks() for k in range(n)
        }
        if boundary is not None:
            assert offset is not None
            mapping.update({i: offset + i - boundary for i in range(boundary, len(old))})
        groups = {}
        for oid, (row, begin, end) in self.groups.items():
            if begin < end and all(i in mapping and mapping[i] == mapping[begin] + i - begin for i in range(begin, end)):
                groups[oid] = (row, mapping[begin], mapping[end - 1] + 1)
        self.groups, self.messages = groups, replacement

    def advance(self, row: StoredRecord, state: ExecutionState, cache: _MessageCache) -> None:
        r, seq = row.record, row.seq
        self.prefix_seq = seq
        p, kind = r._payload, r.kind
        if kind == "session_created":
            self.messages = filter_empty_assistant_messages(
                [cache.parse(m) for m in p["attributes"].get("seed", {}).get("messages", [])]
            )
            first = next((i for i, m in enumerate(self.messages) if m.role != "system"), None)
            if first is not None:
                self.protect(seq, first)
            return
        if kind == "op_completed":
            assert r.operation_id is not None and r.attempt is not None
            op = state.operations[r.operation_id]
            if (
                op.kind == "model"
                and op.selected_attempt == r.attempt
                and p["context"] == "normal"
                and op.attempts[r.attempt].plan._payload["purpose"] == "primary"
                and not p["result"].get("error_code")
            ):
                self.covered = max(self.covered, int(op.attempts[r.attempt].plan._payload["context_version"]))
                begin = len(self.messages)
                self.messages = project_context((row,), state, _initial=self.messages, _task=self.task, _message_cache=cache)
                self.groups[r.operation_id] = (row, begin, len(self.messages))
                return
        # Tool receipts are rendered at their original assistant call, not appended.
        if kind in {"op_planned", "op_completed", "op_parked"} and r.operation_id in state.operations:
            op = state.operations[r.operation_id]
            if op.kind != "model":
                dependencies = op.attempts[1].plan._payload["dependencies"]
                if dependencies and dependencies[0] in self.groups:
                    parent = dependencies[0]
                    model, begin, end = self.groups[parent]
                    group = project_context((model,), state, _initial=[], _task=self.task, _message_cache=cache)
                    shift = len(group) - (end - begin)
                    self.messages[begin:end] = group
                    if shift:
                        self.shift_after(end, shift)
                        self.groups = {
                            oid: (record, a + shift if a >= end else a, b + shift if b >= end else b)
                            for oid, (record, a, b) in self.groups.items()
                        }
                    self.groups[parent] = (model, begin, begin + len(group))
        if (
            kind == "input_applied"
            and p["disposition"] == "applied"
            and p["target_operation_id"]
            and (
                p["input"]["kind"] in {"user", "child_result", "provider_result"}
                or (p["input"]["kind"] == "approval_answer" and p["input"]["payload"]["decision"] in {"deny", "timeout"})
            )
            and p["reason"] != "background child notification"
        ):
            target = state.operations[p["target_operation_id"]]
            if target.kind != "model":
                turn = state.turns[target.turn_id]
                normal = not (turn.cancelled or turn.ended or target.turn_id != state.active_turn_id)
                if normal and p["input"]["kind"] in {"provider_result", "child_result"}:
                    number = p["input"]["payload"]["attempt"]
                    attempt = target.attempts[number]
                    normal = not (attempt.unknown_consumed or any(a.started for n, a in target.attempts.items() if n > number))
                if normal:
                    dependencies = target.attempts[1].plan._payload["dependencies"]
                    if dependencies and dependencies[0] in self.groups:
                        self.protect(seq, self.groups[dependencies[0]][1])
        if kind not in {"turn_started", "input_applied", "boundary_recorded", "op_completed"}:
            return
        if kind == "op_completed":
            assert r.operation_id is not None
            if state.operations[r.operation_id].kind == "model" or p["context"] != "correction":
                return
        if kind == "boundary_recorded" and p["stage"] != "after_cycle":
            return
        before = len(self.messages)
        if kind == "boundary_recorded" and p["stage"] == "after_cycle" and p["data"]["action"] == "steer":
            oid = p["source_operation_id"]
            if oid in self.groups:
                _, begin, end = self.groups[oid]
                if (
                    end > begin
                    and self.messages[end - 1].role == "user"
                    and self.messages[end - 1].content == "Continue working on the task."
                ):
                    self.messages = self.messages[: end - 1] + self.messages[end:]
                    self.shift_after(end, -1)
                    before -= 1
                    self.groups[oid] = (self.groups[oid][0], begin, end - 1)
        after = project_context(
            (row,), state, _initial=self.messages, _task=self.task, _seen_turn=self.seen_turn, _message_cache=cache
        )
        if kind == "turn_started":
            task = r._task()
            self.replace(after)
            if not task.metadata.get("_vv_agent_app_server_compaction_only"):
                initial = task.metadata.get("vv_session", {}).get("input_messages")
                count = len(filter_empty_assistant_messages([cache.parse(m) for m in initial])) if initial else 1
                position = len(after) - count
                if task.initial_messages or (not self.seen_turn and self.covered == 0):
                    position = 1 if after and after[0].role == "system" else 0
                if position < len(after):
                    self.protect(seq, atomic_start(after, position))
            self.task, self.seen_turn = task, True
        else:
            self.messages = after
            if len(after) > before:
                # New appended content can only split the final assistant batch.
                position = before
                for i in range(before - 1, -1, -1):
                    calls = after[i].tool_calls
                    if calls:
                        if i < before <= i + len(calls):
                            position = i
                        break
                    if after[i].role != "tool":
                        break
                self.protect(seq, position)


def protected_start(records: Sequence[StoredRecord], state: ExecutionState) -> int | None:
    """Read the same incremental committed-prefix state used by reducer validation."""
    return state._protection.start if state._protection is not None else None
