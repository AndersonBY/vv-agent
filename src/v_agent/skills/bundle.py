"""Utility for preparing a validated skill bundle from discovered skill directories."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from v_agent.skills.parser import discover_skill_dirs, read_properties
from v_agent.skills.validator import validate


@dataclass(slots=True)
class PreparedSkill:
    name: str
    source: Path
    runtime: Path


def prepare_skill_bundle(
    source_root: Path,
    workspace: Path,
    *,
    cache_subdir: str = ".v_agent_skill_cache/bundle",
    clean: bool = True,
) -> list[PreparedSkill]:
    """Discover, copy, validate and deduplicate skills into a runtime bundle.

    Args:
        source_root: Root directory to scan for SKILL.md files.
        workspace: Agent workspace root (used to resolve the cache directory).
        cache_subdir: Relative path under *workspace* for the runtime bundle.
        clean: If ``True``, remove any previous bundle before copying.

    Returns:
        List of :class:`PreparedSkill` entries ready for agent configuration.

    Raises:
        FileNotFoundError: If *source_root* does not exist.
        ValueError: If no valid skills are found, or a skill fails validation.
    """
    source_root = Path(source_root).resolve()
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"Skills directory not found: {source_root}")

    discovered = discover_skill_dirs(source_root)
    if not discovered:
        raise ValueError(f"No SKILL.md discovered under: {source_root}")

    runtime_root = (workspace / cache_subdir).resolve()
    if clean and runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    prepared: list[PreparedSkill] = []

    for skill_dir in discovered:
        props = read_properties(skill_dir)
        if props.name in used_names:
            continue

        target_dir = (runtime_root / props.name).resolve()
        shutil.copytree(skill_dir, target_dir, dirs_exist_ok=True)

        errors = validate(target_dir)
        if errors:
            joined = "\n".join(f"- {e}" for e in errors)
            raise ValueError(f"Skill '{props.name}' invalid after copy:\n{joined}")

        used_names.add(props.name)
        prepared.append(PreparedSkill(name=props.name, source=skill_dir, runtime=target_dir))

    if not prepared:
        raise ValueError("No valid skills available after preparation.")

    return prepared
