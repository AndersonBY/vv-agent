from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from vv_agent.memory.token_utils import count_tokens

_ACTION_PRIORITY = {
    "modified": 0,
    "created": 1,
    "deleted": 2,
    "read": 3,
}


@dataclass(slots=True)
class PostCompactRestoreConfig:
    total_budget_tokens: int = 30_000
    max_tokens_per_file: int = 5_000
    max_files: int = 10
    token_model: str = ""


def restore_key_files(
    summary_data: dict[str, Any],
    workspace: Path | None,
    config: PostCompactRestoreConfig | None = None,
) -> str:
    if config is None:
        config = PostCompactRestoreConfig()
    if workspace is None:
        return ""

    raw_files = summary_data.get("files_examined_or_modified", [])
    if not isinstance(raw_files, list):
        return ""

    workspace_root = workspace.resolve()
    indexed_files: list[tuple[int, dict[str, Any]]] = [
        (index, cast(dict[str, Any], item))
        for index, item in enumerate(raw_files)
        if isinstance(item, dict)
    ]
    indexed_files.sort(
        key=lambda item: (
            _ACTION_PRIORITY.get(str(item[1].get("action", "read")).strip().lower(), 99),
            item[0],
        )
    )

    restored_parts: list[str] = []
    total_tokens = 0

    for _, file_info in indexed_files[: max(config.max_files, 0)]:
        path_value = str(file_info.get("path", "")).strip()
        if not path_value:
            continue

        resolved_path = _resolve_workspace_file(workspace_root, path_value)
        if resolved_path is None or not resolved_path.exists() or not resolved_path.is_file():
            continue

        try:
            content = resolved_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        action = str(file_info.get("action", "read")).strip().lower() or "read"
        content = _truncate_to_token_budget(
            content,
            max(config.max_tokens_per_file, 1),
            model=config.token_model,
        )
        candidate = f"<file path=\"{path_value}\" action=\"{action}\">\n{content}\n</file>"
        candidate_tokens = count_tokens(candidate, model=config.token_model)
        if total_tokens + candidate_tokens > config.total_budget_tokens:
            break

        restored_parts.append(candidate)
        total_tokens += candidate_tokens

    if not restored_parts:
        return ""

    return (
        "<Post-Compaction File Context>\n"
        "The following files were relevant in the previous conversation context:\n\n"
        + "\n\n".join(restored_parts)
        + "\n</Post-Compaction File Context>"
    )


def _resolve_workspace_file(workspace_root: Path, relative_path: str) -> Path | None:
    candidate = (workspace_root / relative_path).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError:
        return None
    return candidate


def _truncate_to_token_budget(content: str, max_tokens: int, *, model: str) -> str:
    if count_tokens(content, model=model) <= max_tokens:
        return content

    notice = "\n... [truncated after compaction restore]"
    low = 0
    high = len(content)
    best = notice
    while low <= high:
        middle = (low + high) // 2
        candidate = content[:middle].rstrip() + notice
        candidate_tokens = count_tokens(candidate, model=model)
        if candidate_tokens <= max_tokens:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best
