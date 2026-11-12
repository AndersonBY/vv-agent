"""Pure summary planning, acceptance, prompts and retained recovery evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, cast

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._compaction_text import has_recovery_envelope
from vv_agent.kernel._context_messages import atomic_start, filter_empty_assistant_messages, retained_start
from vv_agent.kernel._message_values import COMPACTION_METADATA_KEY, Message, validate_compaction_metadata
from vv_agent.kernel._token_count import TokenCount
from vv_agent.kernel._tool_values import ToolExecutionResult

_MEMORY_SUMMARY_NAME = "memory_summary"
_ORIGINAL_USER_REQUEST_PATTERN = re.compile(r"<Original User Request>\s*(.*?)\s*</Original User Request>", re.DOTALL)
_COMPRESSED_AGENT_MEMORY_PATTERN = re.compile(
    r"<Compressed Agent Memory>\s*([\s\S]*?)\s*</Compressed Agent Memory>",
    re.IGNORECASE,
)
_ANALYSIS_BLOCK_PATTERN = re.compile(r"<analysis>[\s\S]*?</analysis>", re.IGNORECASE)
_SUMMARY_BLOCK_PATTERN = re.compile(r"<summary>\s*([\s\S]*?)\s*</summary>", re.IGNORECASE)


_COMPRESS_MEMORY_PROMPTS = {
    "zh-CN": """你正在总结一段用户与 AI 编程助手的对话。
请先在 <analysis> 标签中进行思考 (该部分后续会被剥离), 然后输出结构化 JSON 摘要。

<analysis>
请逐步思考: 哪些信息必须保留, 哪些用户原话不能丢, 哪些文件/错误/当前状态会影响后续继续执行。
</analysis>

<Previous Summary>
{previous_summary_jcs}
</Previous Summary>

<Conversation Prefix>
{conversation_prefix_jcs}
</Conversation Prefix>

请将以上对话压缩为结构化 JSON「Task Status Summary」, 让 Agent 能快速恢复任务, 并保留用户约束、关键决策、文件操作与当前工作状态。

要求:
- 只输出 JSON, 不要 Markdown。
- 字段内容简洁、可检索, 短句表达。
- 没有信息的字段使用 [] 或 ""。
- `original_user_messages` 字段至关重要: 尽量保留用户原话, 不要做概括式改写。

JSON Schema:
{{
  "summary_version": "2.0",
  "original_user_messages": ["..."],
  "user_constraints": ["..."],
  "decisions": ["..."],
  "files_examined_or_modified": [
    {{"path": "...", "action": "read|created|modified|deleted", "summary": "..."}}
  ],
  "errors_and_fixes": [
    {{"error": "...", "fix": "...", "file": "..."}}
  ],
  "progress": ["最多保留 {event_limit} 条关键进展"],
  "key_facts": ["..."],
  "open_issues": ["..."],
  "current_work_state": "...",
  "next_steps": ["..."]
}}
""",
    "en-US": """You are summarizing a conversation between a user and an AI coding assistant.
Provide your analysis in <analysis> tags first (this section will be stripped), then output a structured JSON summary.

<analysis>
Think step by step about what information is critical to preserve, especially the user's exact wording,
the current work state, file operations, and any errors that were resolved.
</analysis>

<Previous Summary>
{previous_summary_jcs}
</Previous Summary>

<Conversation Prefix>
{conversation_prefix_jcs}
</Conversation Prefix>

Please compress the conversation into a structured JSON "Task Status Summary".
This summary should allow the Agent to quickly resume the task
while preserving user constraints, key decisions, file operations, and critical context.

Requirements:
- Output JSON only, no Markdown.
- Keep fields concise and searchable; use short sentences.
- If a field has no data, use [] or "" as appropriate.
- The "original_user_messages" field is critical. Preserve user messages verbatim or near-verbatim.

JSON Schema:
{{
  "summary_version": "2.0",
  "original_user_messages": ["..."],
  "user_constraints": ["..."],
  "decisions": ["..."],
  "files_examined_or_modified": [
    {{"path": "...", "action": "read|created|modified|deleted", "summary": "..."}}
  ],
  "errors_and_fixes": [
    {{"error": "...", "fix": "...", "file": "..."}}
  ],
  "progress": ["Preserve up to {event_limit} critical events"],
  "key_facts": ["..."],
  "open_issues": ["..."],
  "current_work_state": "...",
  "next_steps": ["..."]
}}
""",
}


@dataclass(frozen=True, slots=True)
class SummaryPlan:
    """Exact history and prompt for a separately persisted summary model operation."""

    messages: list[Message]
    systems: list[Message]
    previous: list[Message]
    prefix: list[Message]
    tail: list[Message]
    prompt: str
    keep_recent: int


@dataclass(frozen=True, slots=True)
class _SummaryAcceptance:
    messages: list[Message]
    accepted: bool
    current_tokens: int | None


def _summary_parts(
    messages: list[Message], keep_recent: int, *, protected_start: int | None
) -> tuple[list[Message], list[Message], list[Message], list[Message]] | None:
    # Validate each original contiguous block; never repair or match ids globally.
    blocks: list[tuple[int, int]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "tool":
            return None
        end = index + 1
        if message.tool_calls:
            if message.role != "assistant":
                return None
            ids = [call.get("id") for call in message.tool_calls]
            if any(not isinstance(call_id, str) or not call_id for call_id in ids) or len(set(ids)) != len(ids):
                return None
            end += len(ids)
            results = messages[index + 1 : end]
            # A short slice is only possible at EOF: retain that unfinished
            # block in the tail. A non-tool interruption inside a batch or
            # a mismatched result is still invalid.
            if any(
                result.role != "tool" or result.tool_call_id != call_id for result, call_id in zip(results, ids, strict=False)
            ):
                return None
        blocks.append((index, end))
        index = end
    retained_ids = {id(message) for message in filter_empty_assistant_messages(messages)}
    protected_from = atomic_start(messages, protected_start) if protected_start is not None else len(messages)
    raw_indices = [
        i
        for i, m in enumerate(messages)
        if i >= protected_from or (id(m) in retained_ids and m.role != "system" and m.name != _MEMORY_SUMMARY_NAME)
    ]
    tail_count = max(keep_recent, 1)
    cut = raw_indices[-tail_count] if len(raw_indices) > tail_count else (raw_indices[0] if raw_indices else 0)
    if protected_start is not None:
        cut = min(cut, atomic_start(messages, protected_start))
    for start, end in blocks:
        if start < cut < end:
            cut = start
    systems = [m for i, m in enumerate(messages) if i < protected_from and m.role == "system" and m.name != _MEMORY_SUMMARY_NAME]
    previous = [m for i, m in enumerate(messages) if i < protected_from and m.name == _MEMORY_SUMMARY_NAME]
    prefix = [messages[i] for i in raw_indices if i < cut]
    tail = [messages[i] for i in raw_indices if i >= cut]
    return systems, previous, prefix, tail


def plan_summary(
    messages: list[Message],
    *,
    keep_recent: int,
    drop_ratio: float,
    protected_start: int | None,
    language: str,
    event_limit: int,
) -> SummaryPlan | None:
    """Plan without inference, pruning, or callbacks; preserve complete block boundaries."""
    keep = keep_recent
    keep = max(1, int(keep * (1 - min(max(drop_ratio, 0.0), 0.95))))
    parts = _summary_parts(messages, keep, protected_start=protected_start)
    if parts is None or not parts[2]:
        return None
    systems, previous, prefix, tail = parts
    return SummaryPlan(
        list(messages),
        systems,
        previous,
        prefix,
        tail,
        _build_compress_memory_prompt(prefix, previous=previous, language=language, event_limit=event_limit),
        keep,
    )


def plan_full_summary(
    messages: list[Message],
    *,
    protected_start: int | None,
    language: str,
    event_limit: int,
) -> SummaryPlan | None:
    """Summarize the full history without changing the caller's context."""
    parts = _summary_parts(messages, 1, protected_start=protected_start)
    if parts is None:
        return None
    systems, previous, prefix, tail = parts
    prefix = [*prefix, *tail]
    if not prefix and not previous:
        return None
    return SummaryPlan(
        list(messages),
        systems,
        previous,
        prefix,
        [],
        _build_compress_memory_prompt(prefix, previous=previous, language=language, event_limit=event_limit),
        0,
    )


def accept_summary(
    plan: SummaryPlan,
    text: str,
    *,
    protected_start: int | None,
    recovery_tool_available: bool,
    effective_context_window: int,
    require_reduction: bool,
    model: str = "",
) -> Generator[TokenCount, int, _SummaryAcceptance]:
    """Validate a retained model result. Log owners defer notification until commit."""
    messages, systems, previous, prefix, tail = plan.messages, plan.systems, plan.previous, plan.prefix, plan.tail
    summary = _normalize_summary_payload(_parse_summary_payload(_normalize_summary_output(text)))
    if not any(value for key, value in summary.items() if key not in {"summary_version", "user_constraints"}):
        return _SummaryAcceptance(list(messages), False, None)
    try:
        evidence = _collect_evidence(previous, prefix)
    except (TypeError, ValueError):
        return _SummaryAcceptance(list(messages), False, None)
    if (evidence["artifacts"] or evidence["cursors"]) and not recovery_tool_available:
        return _SummaryAcceptance(list(messages), False, None)
    paths = summary["files_examined_or_modified"]
    seen = {item["path"] for item in paths}
    prior_paths = []
    for message in previous:
        match = _COMPRESSED_AGENT_MEMORY_PATTERN.search(message.content)
        if match:
            prior_paths.extend(_normalize_summary_payload(_parse_summary_payload(match.group(1)))["files_examined_or_modified"])
    for item in [*prior_paths, *_collect_file_actions(prefix, first_path_wins=True)]:
        if item["path"] not in seen:
            paths.append(item)
            seen.add(item["path"])
    originals = "\n\n".join(summary["original_user_messages"])
    content = (
        f"<Original User Request>\n{originals}\n</Original User Request>\n\n"
        f"<Compressed Agent Memory>\n{canonical_json_bytes(summary).decode()}\n</Compressed Agent Memory>\n\n"
        + _render_evidence(evidence)
    )
    candidate = [
        *systems,
        Message(role="user", name=_MEMORY_SUMMARY_NAME, content=content, metadata={COMPACTION_METADATA_KEY: evidence}),
        *tail,
    ]
    retained_start(messages, candidate, protected_start)
    tokens = yield TokenCount(candidate, model)
    if (require_reduction and tokens >= (yield TokenCount(messages, model))) or tokens > effective_context_window:
        return _SummaryAcceptance(list(messages), False, None)
    return _SummaryAcceptance(candidate, True, tokens)


def _normalize_summary_payload(payload: dict[str, object]) -> dict[str, Any]:
    normalized: dict[str, Any] = {"summary_version": "2.0"}
    for key in (
        "original_user_messages",
        "user_constraints",
        "decisions",
        "progress",
        "key_facts",
        "open_issues",
        "next_steps",
    ):
        value = payload.get(key)
        normalized[key] = list(value) if isinstance(value, list) and all(isinstance(item, str) for item in value) else []
    state = payload.get("current_work_state")
    normalized["current_work_state"] = state if isinstance(state, str) else ""
    for key, fields in (
        ("files_examined_or_modified", ("path", "action", "summary")),
        ("errors_and_fixes", ("error", "fix", "file")),
    ):
        records = payload.get(key)
        normalized[key] = []
        for item in records if isinstance(records, list) else []:
            if not isinstance(item, dict):
                continue
            record = cast(dict[str, object], item)
            if key == "files_examined_or_modified":
                path = record.get("path")
                if (
                    not isinstance(path, str)
                    or not path.strip()
                    or record.get("action") not in ("read", "created", "modified", "deleted")
                ):
                    continue
            elif not isinstance(record.get("error"), str):
                continue
            normalized[key].append({field: value if isinstance(value := record.get(field), str) else "" for field in fields})
    return normalized


def compaction_evidence(messages: list[Message], *, protected_start: int | None) -> dict[str, list[dict[str, Any]]]:
    """Collect validated recovery references without reading files or dropping evidence."""
    parts = _summary_parts(messages, 1, protected_start=protected_start)
    if parts is None:
        raise ValueError("invalid compaction blocks")
    return _collect_evidence(parts[1], [*parts[2], *parts[3]])


def _collect_evidence(previous: list[Message], prefix: list[Message]) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = {"artifacts": [], "cursors": []}
    seen: dict[str, set[bytes]] = {"artifacts": set(), "cursors": set()}

    def merge(manifest: dict[str, Any]) -> None:
        validate_compaction_metadata({COMPACTION_METADATA_KEY: manifest})
        for key in evidence:
            for record in manifest[key]:
                identity = canonical_json_bytes(record)
                if identity not in seen[key]:
                    seen[key].add(identity)
                    evidence[key].append(record)

    for message in previous:
        if COMPACTION_METADATA_KEY in message.metadata:
            merge(message.metadata[COMPACTION_METADATA_KEY])
    calls: dict[str, Any] = {}
    for message in prefix:
        if message.role == "assistant":
            calls = {call["id"]: call["function"] for call in message.tool_calls or []}
        if message.role != "tool":
            continue
        assert message.tool_call_id is not None  # Validated by _summary_parts.
        function = calls[message.tool_call_id]
        arguments = json.loads(function["arguments"])
        common = {
            "tool_call_id": message.tool_call_id,
            "tool_name": function["name"],
            "arguments": canonical_json_bytes(arguments).decode(),
        }
        refs: dict[str, Any] = {"artifacts": [], "cursors": []}
        if message.artifact_ref is not None:
            refs["artifacts"].append({**common, "artifact_ref": message.artifact_ref.to_dict()})
        if has_recovery_envelope(message.content):
            body, _, raw = message.content.rpartition("\n")
            envelope = json.loads(raw)
            if set(envelope) != {"vv_agent_recovery"}:
                raise ValueError("invalid recovery envelope")
            recovery = envelope["vv_agent_recovery"]
            if not isinstance(recovery, dict) or set(recovery) - {
                "truncated",
                "truncation_reason",
                "original_bytes",
                "visible_bytes",
                "artifact",
                "cursor",
            }:
                raise ValueError("invalid recovery fields")
            result = ToolExecutionResult.from_dict(
                {
                    "tool_call_id": message.tool_call_id,
                    "content": body,
                    "status_code": "SUCCESS",
                    "directive": "continue",
                    **recovery,
                }
            )
            if result.artifact:
                refs["artifacts"].append({**common, "artifact_ref": result.artifact.to_dict()})
            if result.cursor:
                refs["cursors"].append({**common, "cursor": result.cursor.to_dict()})
        merge(refs)
    return evidence


def _render_evidence(evidence: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["<Persisted Artifacts>"]
    for key, pointer in (("artifacts", "artifact_ref"), ("cursors", "cursor")):
        for record in evidence[key]:
            visible = {field: record[field] for field in ("tool_call_id", "tool_name", "arguments")}
            if key == "artifacts":
                visible.update(artifact_path=record[pointer]["path"], retrieval_hint="use read_file on artifact_path if needed")
            else:
                visible.update(
                    path=record[pointer]["path"],
                    offset_chars=record[pointer]["offset_chars"],
                    retrieval_hint="use read_file on path if needed",
                )
            lines.append("- " + canonical_json_bytes(visible).decode())
    return "\n".join([*lines, "</Persisted Artifacts>"])


def _build_compress_memory_prompt(
    messages: list[Message],
    *,
    previous: list[Message] | None,
    language: str,
    event_limit: int,
) -> str:
    template = _COMPRESS_MEMORY_PROMPTS.get(language, _COMPRESS_MEMORY_PROMPTS["en-US"])

    def project(message: Message) -> dict[str, Any]:
        value = message.to_dict()
        value.pop("artifact_ref", None)
        if "image_url" in value:
            del value["image_url"]
            value["content"] = f"[image omitted from summary input: {message.content or 'image'}]"
        metadata = value.get("metadata", {})
        metadata.pop(COMPACTION_METADATA_KEY, None)
        if not metadata:
            value.pop("metadata", None)
        return value

    return template.format(
        previous_summary_jcs=canonical_json_bytes([project(m) for m in previous or []]).decode(),
        conversation_prefix_jcs=canonical_json_bytes([project(m) for m in messages]).decode(),
        event_limit=max(event_limit, 1),
    )


def _build_local_summary(messages: list[Message], artifacts: list[dict[str, str]], *, event_limit: int) -> str:
    events = _build_summary_events(messages[2:], event_limit=event_limit)
    artifact_facts = [f"{item.get('path', '')} (tool={item.get('tool', 'unknown')})" for item in artifacts if item.get("path")]
    payload = {
        "summary_version": "2.0",
        "original_user_messages": _collect_original_user_messages(messages),
        "user_constraints": [],
        "decisions": [],
        "files_examined_or_modified": _collect_file_actions(messages, first_path_wins=False),
        "errors_and_fixes": _collect_errors_and_fixes(messages),
        "progress": events,
        "key_facts": artifact_facts,
        "open_issues": [],
        "current_work_state": _build_current_work_state(messages),
        "next_steps": [],
    }
    return json.dumps(payload, ensure_ascii=False)


def _normalize_summary_output(text: str) -> str:
    cleaned = _strip_markdown_code_fence(text)
    cleaned = _ANALYSIS_BLOCK_PATTERN.sub("", cleaned).strip()
    summary_match = _SUMMARY_BLOCK_PATTERN.search(cleaned)
    if summary_match:
        cleaned = summary_match.group(1).strip()
    return cleaned


def _strip_markdown_code_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if len(lines) < 2:
        return cleaned
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_summary_payload(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if not cleaned:
        return {}

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _collect_original_user_messages(messages: list[Message]) -> list[str]:
    user_messages: list[str] = []
    seen: set[str] = set()

    def append_unique(value: object) -> None:
        if not isinstance(value, str):
            return
        normalized = value.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        user_messages.append(normalized)

    for message in messages[1:]:
        if message.role != "user":
            continue
        content = message.content.strip()
        if not content:
            continue

        compressed_match = _COMPRESSED_AGENT_MEMORY_PATTERN.search(content)
        if compressed_match:
            summary_data = _parse_summary_payload(compressed_match.group(1))
            previous_originals = summary_data.get("original_user_messages")
            if isinstance(previous_originals, list):
                for previous_original in previous_originals:
                    append_unique(previous_original)

        original_match = _ORIGINAL_USER_REQUEST_PATTERN.search(content)
        if original_match and not compressed_match:
            append_unique(original_match.group(1))

        if compressed_match or original_match or "<Compressed Agent Memory>" in content:
            continue
        append_unique(content)
    return user_messages


def _collect_file_actions(messages: list[Message], *, first_path_wins: bool) -> list[dict[str, str]]:
    action_priority = {"modified": 0, "created": 1, "deleted": 2, "read": 3}
    tool_action_map = {
        "read_file": "read",
        "file_info": "read",
        "write_file": "modified",
        "edit_file": "modified",
    }
    actions_by_path: dict[str, dict[str, str]] = {}
    ordered_paths: list[str] = []

    for message in messages:
        if message.role != "assistant" or not message.tool_calls:
            continue
        for tool_call in message.tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function_payload = tool_call.get("function")
            if not isinstance(function_payload, dict):
                continue
            tool_name = function_payload.get("name")
            if not isinstance(tool_name, str):
                continue
            action = tool_action_map.get(tool_name)
            if action is None:
                continue
            arguments = _parse_tool_arguments(function_payload.get("arguments"))
            path = arguments.get("path") if first_path_wins else _extract_file_path_from_arguments(arguments)
            if not isinstance(path, str) or not path.strip():
                continue

            summary = _summarize_file_action(tool_name, path)
            existing = actions_by_path.get(path)
            if existing is None:
                actions_by_path[path] = {"path": path, "action": action, "summary": summary}
                ordered_paths.append(path)
                continue

            if first_path_wins:
                continue
            if action_priority[action] < action_priority.get(existing["action"], 99):
                existing["action"] = action
            existing["summary"] = summary

    return [actions_by_path[path] for path in ordered_paths]


def _parse_tool_arguments(raw_arguments: object) -> dict[str, object]:
    if isinstance(raw_arguments, dict):
        normalized: dict[str, object] = {}
        for key, value in raw_arguments.items():
            normalized[str(key)] = value
        return normalized
    if not isinstance(raw_arguments, str) or not raw_arguments.strip():
        return {}
    try:
        parsed: object = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    normalized: dict[str, object] = {}
    for key, value in parsed.items():
        normalized[str(key)] = value
    return normalized


def _extract_file_path_from_arguments(arguments: dict[str, object]) -> str | None:
    for key in ("path", "file_path", "filepath", "target_file"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _summarize_file_action(tool_name: str, path: str) -> str:
    if tool_name == "read_file":
        return f"Read {path}"
    if tool_name == "file_info":
        return f"Inspected {path}"
    if tool_name == "write_file":
        return f"Updated {path}"
    if tool_name == "edit_file":
        return f"Modified {path}"
    return f"Touched {path}"


def _collect_errors_and_fixes(messages: list[Message]) -> list[dict[str, str]]:
    error_entries: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if message.role != "tool":
            continue
        lowered = message.content.lower()
        if not any(token in lowered for token in ("error", "exception", "traceback", "failed")):
            continue
        fix = ""
        for follow_message in messages[index + 1 :]:
            if follow_message.role == "assistant" and follow_message.content.strip():
                fix = _summarize_message_content(follow_message.content)
                break
        error_entries.append(
            {
                "error": _summarize_message_content(message.content),
                "fix": fix,
                "file": "",
            }
        )
        if len(error_entries) >= 5:
            break
    return error_entries


def _build_current_work_state(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role not in {"assistant", "user"}:
            continue
        content = message.content.strip()
        if content:
            return _summarize_message_content(content)
    return ""


def _summarize_message_content(content: str, *, limit: int = 240) -> str:
    normalized = " ".join(content.split()).strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def _build_summary_events(middle: list[Message], *, event_limit: int) -> list[str]:
    events: list[str] = []
    limit = max(event_limit, 1)
    for idx, message in enumerate(middle[:limit], start=1):
        content = message.content.replace("\n", " ").strip()
        if len(content) > 160:
            content = f"{content[:157]}..."
        note = f"{idx:02d}. {message.role}: {content}"
        if message.tool_call_id:
            note += f" (tool_call_id={message.tool_call_id})"
        if message.tool_calls:
            tool_names: list[str] = []
            for tool_call in message.tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function_payload = tool_call.get("function")
                if isinstance(function_payload, dict):
                    tool_name = function_payload.get("name")
                    if isinstance(tool_name, str) and tool_name:
                        tool_names.append(tool_name)
            if tool_names:
                note += f" (tool_calls={','.join(tool_names)})"
        events.append(note)
    if len(middle) > limit:
        events.append(f"... {len(middle) - limit} more messages omitted ...")
    return events
