from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

from v_agent.skills.errors import SkillParseError

MAX_SKILL_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_COMPATIBILITY_LENGTH = 500

ALLOWED_FIELDS = {
    "name",
    "description",
    "license",
    "compatibility",
    "allowed-tools",
    "metadata",
}


def _validate_name(name: Any, *, skill_dir: Path | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(name, str) or not name.strip():
        return ["Field 'name' must be a non-empty string"]

    normalized = unicodedata.normalize("NFKC", name.strip())
    if len(normalized) > MAX_SKILL_NAME_LENGTH:
        errors.append(
            f"Skill name '{normalized}' exceeds {MAX_SKILL_NAME_LENGTH} character limit ({len(normalized)} chars)"
        )
    if normalized != normalized.lower():
        errors.append(f"Skill name '{normalized}' must be lowercase")
    if normalized.startswith("-") or normalized.endswith("-"):
        errors.append("Skill name cannot start or end with a hyphen")
    if "--" in normalized:
        errors.append("Skill name cannot contain consecutive hyphens")
    if not all(ch.isalnum() or ch == "-" for ch in normalized):
        errors.append(
            f"Skill name '{normalized}' contains invalid characters. Only letters, digits, and hyphens are allowed."
        )

    if skill_dir is not None:
        dir_name = unicodedata.normalize("NFKC", skill_dir.name)
        if dir_name != normalized:
            errors.append(f"Directory name '{skill_dir.name}' must match skill name '{normalized}'")
    return errors


def _validate_description(description: Any) -> list[str]:
    if not isinstance(description, str) or not description.strip():
        return ["Field 'description' must be a non-empty string"]
    if len(description) > MAX_DESCRIPTION_LENGTH:
        return [f"Description exceeds {MAX_DESCRIPTION_LENGTH} character limit ({len(description)} chars)"]
    return []


def _validate_compatibility(compatibility: Any) -> list[str]:
    if compatibility is None:
        return []
    if not isinstance(compatibility, str):
        return ["Field 'compatibility' must be a string"]
    if len(compatibility) > MAX_COMPATIBILITY_LENGTH:
        return [f"Compatibility exceeds {MAX_COMPATIBILITY_LENGTH} character limit ({len(compatibility)} chars)"]
    return []


def validate_metadata(metadata: dict[str, Any], *, skill_dir: Path | None = None) -> list[str]:
    errors: list[str] = []

    extra_fields = set(metadata.keys()) - ALLOWED_FIELDS
    if extra_fields:
        errors.append(
            f"Unexpected fields in frontmatter: {', '.join(sorted(extra_fields))}. "
            f"Only {sorted(ALLOWED_FIELDS)} are allowed."
        )

    if "name" not in metadata:
        errors.append("Missing required field in frontmatter: name")
    else:
        errors.extend(_validate_name(metadata["name"], skill_dir=skill_dir))

    if "description" not in metadata:
        errors.append("Missing required field in frontmatter: description")
    else:
        errors.extend(_validate_description(metadata["description"]))

    errors.extend(_validate_compatibility(metadata.get("compatibility")))
    return errors


def validate(skill_dir: Path) -> list[str]:
    """Validate a skill directory path and return all validation errors."""
    from v_agent.skills.parser import find_skill_md, parse_frontmatter

    skill_dir = Path(skill_dir)

    if not skill_dir.exists():
        return [f"Path does not exist: {skill_dir}"]
    if not skill_dir.is_dir():
        return [f"Not a directory: {skill_dir}"]

    skill_md = find_skill_md(skill_dir)
    if skill_md is None:
        return ["Missing required file: SKILL.md"]

    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
        metadata, _ = parse_frontmatter(content)
    except SkillParseError as exc:
        return [str(exc)]

    return validate_metadata(metadata, skill_dir=skill_dir)
