from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from vv_agent.skills.errors import SkillParseError, SkillValidationError
from vv_agent.skills.models import LoadedSkill, SkillProperties


def find_skill_md(skill_dir: Path) -> Path | None:
    """Find SKILL.md in a skill directory.

    The spec prefers uppercase `SKILL.md`; lowercase `skill.md` is accepted.
    """
    for name in ("SKILL.md", "skill.md"):
        path = skill_dir / name
        if path.exists() and path.is_file():
            return path
    return None


def discover_skill_dirs(root: Path) -> list[Path]:
    """Discover all skill directories under a root path.

    A directory is considered a skill directory when it directly contains SKILL.md/skill.md.
    """
    root = Path(root)
    if not root.exists() or not root.is_dir():
        return []

    discovered: list[Path] = []
    seen: set[Path] = set()

    def add_if_skill(dir_path: Path) -> None:
        normalized = dir_path.resolve()
        if normalized in seen:
            return
        if find_skill_md(normalized) is None:
            return
        seen.add(normalized)
        discovered.append(normalized)

    add_if_skill(root)
    for candidate in sorted(root.rglob("*")):
        if candidate.is_dir():
            add_if_skill(candidate)

    return discovered


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter and markdown body from SKILL.md content."""
    if not content.startswith("---"):
        raise SkillParseError("SKILL.md must start with YAML frontmatter (---)")

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise SkillParseError("SKILL.md frontmatter not properly closed with ---")

    frontmatter_str = parts[1]
    body = parts[2].strip()
    try:
        parsed = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as exc:
        raise SkillParseError(f"Invalid YAML in frontmatter: {exc}") from exc

    if not isinstance(parsed, dict):
        raise SkillParseError("SKILL.md frontmatter must be a YAML mapping")

    metadata = dict(parsed)
    raw_metadata = metadata.get("metadata")
    if isinstance(raw_metadata, dict):
        metadata["metadata"] = {str(key): str(value) for key, value in raw_metadata.items()}

    return metadata, body


def _build_properties(metadata: dict[str, Any]) -> SkillProperties:
    name = metadata.get("name")
    description = metadata.get("description")

    if not isinstance(name, str) or not name.strip():
        raise SkillValidationError("Field 'name' must be a non-empty string")
    if not isinstance(description, str) or not description.strip():
        raise SkillValidationError("Field 'description' must be a non-empty string")

    license_text = metadata.get("license")
    compatibility = metadata.get("compatibility")
    allowed_tools = metadata.get("allowed-tools")
    raw_meta = metadata.get("metadata")

    return SkillProperties(
        name=name.strip(),
        description=description.strip(),
        license=str(license_text).strip() if isinstance(license_text, str) and license_text.strip() else None,
        compatibility=str(compatibility).strip() if isinstance(compatibility, str) and compatibility.strip() else None,
        allowed_tools=str(allowed_tools).strip() if isinstance(allowed_tools, str) and allowed_tools.strip() else None,
        metadata=raw_meta if isinstance(raw_meta, dict) else {},
    )


def read_properties(skill_dir: Path) -> SkillProperties:
    """Read frontmatter metadata from a skill directory.

    This function loads only metadata and does not validate directory/name constraints.
    """
    skill_dir = Path(skill_dir)
    skill_md = find_skill_md(skill_dir)
    if skill_md is None:
        raise SkillParseError(f"SKILL.md not found in {skill_dir}")

    content = skill_md.read_text(encoding="utf-8", errors="replace")
    metadata, _ = parse_frontmatter(content)

    if "name" not in metadata:
        raise SkillValidationError("Missing required field in frontmatter: name")
    if "description" not in metadata:
        raise SkillValidationError("Missing required field in frontmatter: description")

    return _build_properties(metadata)


def read_skill(skill_dir: Path) -> LoadedSkill:
    """Read full skill content and validate against Agent Skills constraints."""
    from vv_agent.skills.validator import validate_metadata

    skill_dir = Path(skill_dir)
    skill_md = find_skill_md(skill_dir)
    if skill_md is None:
        raise SkillParseError(f"SKILL.md not found in {skill_dir}")

    content = skill_md.read_text(encoding="utf-8", errors="replace")
    metadata, body = parse_frontmatter(content)

    errors = validate_metadata(metadata, skill_dir=skill_dir)
    if errors:
        raise SkillValidationError("; ".join(errors))

    properties = _build_properties(metadata)
    return LoadedSkill(properties=properties, skill_md_path=skill_md.resolve(), instructions=body)
