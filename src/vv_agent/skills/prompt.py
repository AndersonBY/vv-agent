"""Skill prompt rendering with budget management."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from vv_agent.skills.normalize import SkillEntry, normalize_skill_list
from vv_agent.skills.parser import find_skill_md, read_properties

MAX_SKILLS_PROMPT_CHARS = 8000


def skill_entry_to_xml(entry: SkillEntry, *, include_location: bool = True) -> str:
    """Render a single SkillEntry as an XML ``<skill>`` block."""
    lines = [
        "<skill>",
        "<name>",
        html.escape(entry.name),
        "</name>",
        "<description>",
        html.escape(entry.description),
        "</description>",
    ]
    if include_location and entry.location:
        lines.extend(["<location>", html.escape(entry.location), "</location>"])
    lines.append("</skill>")
    return "\n".join(lines)


def render_skills_xml(
    entries: list[SkillEntry],
    *,
    budget: int = MAX_SKILLS_PROMPT_CHARS,
) -> str:
    """Render ``<available_skills>`` XML with progressive degradation.

    Tier 1 - Full: name + description + location for all skills.
    Tier 2 - Compact: name + description only (drop location).
    Tier 3 - Truncated: as many compact entries as fit, then a summary comment.
    """
    if not entries:
        return "<available_skills>\n</available_skills>"

    full_xml = _render_all(entries, include_location=True)
    if len(full_xml) <= budget:
        return full_xml

    compact_xml = _render_all(entries, include_location=False)
    if len(compact_xml) <= budget:
        return compact_xml

    wrapper_overhead = len("<available_skills>\n</available_skills>") + 80
    remaining = budget - wrapper_overhead
    lines = ["<available_skills>"]
    included = 0
    for entry in entries:
        xml = skill_entry_to_xml(entry, include_location=False)
        if len(xml) + 1 > remaining:
            break
        lines.append(xml)
        remaining -= len(xml) + 1
        included += 1

    omitted = len(entries) - included
    if omitted > 0:
        lines.append(f"<!-- {omitted} more skills available; use activate_skill to discover -->")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _render_all(entries: list[SkillEntry], *, include_location: bool) -> str:
    lines = ["<available_skills>"]
    for entry in entries:
        lines.append(skill_entry_to_xml(entry, include_location=include_location))
    lines.append("</available_skills>")
    return "\n".join(lines)


def to_available_skills_xml(skill_dirs: list[Path]) -> str:
    """Convenience: render ``<available_skills>`` directly from skill directory paths."""
    if not skill_dirs:
        return "<available_skills>\n</available_skills>"

    lines = ["<available_skills>"]
    for skill_dir in skill_dirs:
        normalized_dir = Path(skill_dir).resolve()
        properties = read_properties(normalized_dir)
        skill_md_path = find_skill_md(normalized_dir)
        entry = SkillEntry(
            name=properties.name,
            description=properties.description,
            location=str(skill_md_path) if skill_md_path else None,
        )
        lines.append(skill_entry_to_xml(entry))
    lines.append("</available_skills>")
    return "\n".join(lines)


def metadata_to_prompt_entries(
    available_skills: list[dict[str, Any] | str],
    *,
    workspace: Path | None = None,
) -> list[SkillEntry]:
    """Normalize runtime skill metadata into prompt-friendly entries.

    Thin wrapper around :func:`normalize_skill_list` for backward compatibility.
    """
    return normalize_skill_list(available_skills, workspace=workspace)
