from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v_agent.skills.errors import SkillError
from v_agent.skills.models import SkillProperties
from v_agent.skills.parser import find_skill_md, read_properties


@dataclass(slots=True)
class PromptSkillEntry:
    name: str
    description: str
    location: str | None = None


def skill_to_prompt_entry(*, properties: SkillProperties, location: str | None = None) -> str:
    lines = [
        "<skill>",
        "<name>",
        html.escape(properties.name),
        "</name>",
        "<description>",
        html.escape(properties.description),
        "</description>",
    ]
    if location:
        lines.extend(["<location>", html.escape(location), "</location>"])
    lines.append("</skill>")
    return "\n".join(lines)


def to_available_skills_xml(skill_dirs: list[Path]) -> str:
    if not skill_dirs:
        return "<available_skills>\n</available_skills>"

    lines = ["<available_skills>"]
    for skill_dir in skill_dirs:
        normalized_dir = Path(skill_dir).resolve()
        properties = read_properties(normalized_dir)
        skill_md_path = find_skill_md(normalized_dir)
        location = str(skill_md_path) if skill_md_path else None
        lines.append(skill_to_prompt_entry(properties=properties, location=location))

    lines.append("</available_skills>")
    return "\n".join(lines)


def _resolve_skill_location(location: str, *, workspace: Path | None = None) -> Path:
    path = Path(location).expanduser()
    if not path.is_absolute() and workspace is not None:
        path = (workspace / path).resolve()
    if path.is_file() and path.name.lower() == "skill.md":
        return path.parent
    return path


def metadata_to_prompt_entries(
    available_skills: list[dict[str, Any] | str],
    *,
    workspace: Path | None = None,
) -> list[PromptSkillEntry]:
    """Normalize runtime skill metadata into prompt-friendly entries."""
    entries: list[PromptSkillEntry] = []

    for item in available_skills:
        if isinstance(item, str):
            raw_location = item.strip()
            if not raw_location:
                continue
            try:
                skill_dir = _resolve_skill_location(raw_location, workspace=workspace)
                properties = read_properties(skill_dir)
                skill_md_path = find_skill_md(skill_dir)
            except SkillError:
                continue
            entries.append(
                PromptSkillEntry(
                    name=properties.name,
                    description=properties.description,
                    location=str(skill_md_path) if skill_md_path else raw_location,
                )
            )
            continue

        if not isinstance(item, dict):
            continue

        name = str(item.get("name") or "").strip()
        description = str(item.get("description") or "").strip()
        location_value = (
            item.get("location")
            or item.get("skill_md_path")
            or item.get("skill_directory")
            or item.get("directory")
            or item.get("path")
        )
        location = str(location_value).strip() if isinstance(location_value, str) and location_value.strip() else None

        if (not name or not description) and location:
            try:
                skill_dir = _resolve_skill_location(location, workspace=workspace)
                properties = read_properties(skill_dir)
                skill_md_path = find_skill_md(skill_dir)
            except SkillError:
                continue
            name = properties.name
            description = properties.description
            location = str(skill_md_path) if skill_md_path else location

        if not name or not description:
            continue

        entries.append(PromptSkillEntry(name=name, description=description, location=location))

    return entries
