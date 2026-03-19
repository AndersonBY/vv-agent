"""Unified skill list normalization used by both prompt rendering and tool activation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vv_agent.skills.errors import SkillError
from vv_agent.skills.parser import discover_skill_dirs, find_skill_md, read_properties, read_skill


@dataclass(slots=True)
class SkillEntry:
    """Normalized skill record used across prompt rendering and activation."""

    name: str
    description: str
    location: str | None = None
    instructions: str | None = None
    compatibility: str | None = None
    allowed_tools: str | None = None
    metadata: dict[str, str] | None = None
    load_error: str | None = None


def normalize_skill_list(
    raw_skills: list[dict[str, Any] | str] | None,
    *,
    workspace: Path | None = None,
    load_instructions: bool = False,
) -> list[SkillEntry]:
    """Normalize a mixed skill list into deduplicated SkillEntry records.

    Accepts two input formats:
    - ``str``: a filesystem path to a skill directory (or parent containing skills).
    - ``dict``: must contain ``name`` and ``description``; optionally ``location``.

    When *load_instructions* is True the full SKILL.md body is read (needed for
    activation).  When False only frontmatter metadata is loaded (sufficient for
    prompt rendering).

    Returns a list of :class:`SkillEntry` deduplicated by name (first wins).
    """
    if not isinstance(raw_skills, list):
        return []

    entries: list[SkillEntry] = []
    for item in raw_skills:
        if isinstance(item, str):
            entries.extend(_entries_from_path(item.strip(), workspace=workspace, load_instructions=load_instructions))
        elif isinstance(item, dict):
            entries.extend(_entries_from_dict(item, workspace=workspace, load_instructions=load_instructions))

    seen: set[str] = set()
    deduped: list[SkillEntry] = []
    for entry in entries:
        if entry.name in seen:
            continue
        seen.add(entry.name)
        deduped.append(entry)
    return deduped


def _resolve_skill_path(raw_path: str, *, workspace: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and workspace is not None:
        path = (workspace / path).resolve()
    if path.is_file() and path.name.lower() == "skill.md":
        return path.parent
    return path


def _path_exists(raw_path: str, *, workspace: Path | None) -> bool:
    path = Path(raw_path).expanduser()
    if path.exists():
        return True
    if workspace is not None and not path.is_absolute():
        return (workspace / path).exists()
    return False


def _load_entry(skill_dir: Path, *, workspace: Path | None, load_instructions: bool) -> SkillEntry | None:
    try:
        if load_instructions:
            loaded = read_skill(skill_dir)
            skill_md = loaded.skill_md_path
            location = _relative_location(skill_md, workspace)
            return SkillEntry(
                name=loaded.properties.name,
                description=loaded.properties.description,
                location=location,
                instructions=loaded.instructions,
                compatibility=loaded.properties.compatibility,
                allowed_tools=loaded.properties.allowed_tools,
                metadata=loaded.properties.metadata or None,
            )
        else:
            props = read_properties(skill_dir)
            skill_md = find_skill_md(skill_dir)
            location = _relative_location(skill_md, workspace) if skill_md else None
            return SkillEntry(
                name=props.name,
                description=props.description,
                location=location,
                compatibility=props.compatibility,
                allowed_tools=props.allowed_tools,
                metadata=props.metadata or None,
            )
    except SkillError as exc:
        fallback_name = skill_dir.name
        return SkillEntry(
            name=fallback_name,
            description="",
            location=str(skill_dir),
            load_error=str(exc),
        )


def _relative_location(skill_md: Path, workspace: Path | None) -> str:
    if workspace is not None:
        try:
            return skill_md.relative_to(workspace).as_posix()
        except ValueError:
            pass
    return skill_md.as_posix()


def _entries_from_path(
    raw_path: str,
    *,
    workspace: Path | None,
    load_instructions: bool,
) -> list[SkillEntry]:
    if not raw_path or not _path_exists(raw_path, workspace=workspace):
        return []
    resolved = _resolve_skill_path(raw_path, workspace=workspace)
    if resolved.is_dir() and find_skill_md(resolved) is None:
        entries: list[SkillEntry] = []
        for skill_dir in discover_skill_dirs(resolved):
            entry = _load_entry(skill_dir, workspace=workspace, load_instructions=load_instructions)
            if entry is not None:
                entries.append(entry)
        return entries
    entry = _load_entry(resolved, workspace=workspace, load_instructions=load_instructions)
    return [entry] if entry is not None else []


def _entries_from_dict(
    item: dict[str, Any],
    *,
    workspace: Path | None,
    load_instructions: bool,
) -> list[SkillEntry]:
    name = str(item.get("name") or "").strip()
    description = str(item.get("description") or "").strip()
    location = str(item.get("location") or "").strip() or None

    if (not name or not description) and location and _path_exists(location, workspace=workspace):
        return _entries_from_path(location, workspace=workspace, load_instructions=load_instructions)

    if not name or not description:
        return []

    instructions = str(item.get("instructions") or "").strip() or None
    if not instructions and load_instructions and location and _path_exists(location, workspace=workspace):
        loaded_entries = _entries_from_path(location, workspace=workspace, load_instructions=True)
        if loaded_entries:
            return loaded_entries

    compatibility = str(item.get("compatibility") or "").strip() or None
    allowed_tools_raw = item.get("allowed-tools") or item.get("allowed_tools")
    allowed_tools = str(allowed_tools_raw).strip() if isinstance(allowed_tools_raw, str) and allowed_tools_raw else None
    metadata_raw = item.get("metadata")
    metadata = {str(k): str(v) for k, v in metadata_raw.items()} if isinstance(metadata_raw, dict) else None

    return [SkillEntry(
        name=name,
        description=description,
        location=location,
        instructions=instructions,
        compatibility=compatibility,
        allowed_tools=allowed_tools,
        metadata=metadata,
    )]
