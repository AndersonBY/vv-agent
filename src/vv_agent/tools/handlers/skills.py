from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vv_agent.skills import SkillParseError, SkillValidationError, discover_skill_dirs, find_skill_md, read_skill
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import to_json
from vv_agent.types import ToolExecutionResult, ToolResultStatus


@dataclass(slots=True)
class _SkillBinding:
    name: str
    display_name: str
    description: str = ""
    instructions: str = ""
    location: str | None = None
    compatibility: str | None = None
    allowed_tools: str | None = None
    metadata: dict[str, str] | None = None
    load_error: str | None = None


def _error(*, error_code: str, message: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        status_code=ToolResultStatus.ERROR,
        error_code=error_code,
        content=to_json({"error": message, "error_code": error_code}),
    )


def _resolve_skill_path(raw_path: str, *, workspace: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (workspace / path).resolve()

    if path.is_file() and path.name.lower() == "skill.md":
        return path.parent
    return path


def _path_exists(raw_path: str, *, workspace: Path) -> bool:
    path = Path(raw_path).expanduser()
    if path.exists():
        return True
    if not path.is_absolute():
        return (workspace / path).exists()
    return False


def _load_from_standard_skill(
    raw_path: str,
    *,
    workspace: Path,
    configured_name: str | None = None,
) -> _SkillBinding:
    skill_dir = _resolve_skill_path(raw_path, workspace=workspace)
    try:
        loaded = read_skill(skill_dir)
    except (SkillParseError, SkillValidationError) as exc:
        fallback_name = configured_name or Path(raw_path).name
        return _SkillBinding(
            name=fallback_name,
            display_name=fallback_name,
            location=str(skill_dir),
            load_error=str(exc),
        )

    binding = _SkillBinding(
        name=loaded.properties.name,
        display_name=loaded.properties.name,
        description=loaded.properties.description,
        instructions=loaded.instructions,
        location=loaded.skill_md_path.as_posix(),
        compatibility=loaded.properties.compatibility,
        allowed_tools=loaded.properties.allowed_tools,
        metadata=loaded.properties.metadata,
    )

    if configured_name and configured_name != binding.name:
        binding.load_error = (
            f"Configured skill name '{configured_name}' does not match SKILL.md name '{binding.name}'"
        )
    return binding


def _bindings_from_path(
    raw_path: str,
    *,
    workspace: Path,
    configured_name: str | None = None,
) -> list[_SkillBinding]:
    resolved = _resolve_skill_path(raw_path, workspace=workspace)

    if resolved.is_dir() and find_skill_md(resolved) is None:
        discovered_dirs = discover_skill_dirs(resolved)
        if not discovered_dirs:
            fallback_name = configured_name or Path(raw_path).name
            return [
                _SkillBinding(
                    name=fallback_name,
                    display_name=fallback_name,
                    location=str(resolved),
                    load_error=f"No SKILL.md found under {resolved}",
                )
            ]

        bindings = [
            _load_from_standard_skill(skill_dir.as_posix(), workspace=workspace)
            for skill_dir in discovered_dirs
        ]
        if configured_name:
            for binding in bindings:
                if binding.name == configured_name:
                    return [binding]
            return [
                _SkillBinding(
                    name=configured_name,
                    display_name=configured_name,
                    location=str(resolved),
                    load_error=(
                        f"Configured skill name '{configured_name}' not found in skill collection '{resolved}'"
                    ),
                )
            ]
        return bindings

    return [
        _load_from_standard_skill(
            raw_path,
            workspace=workspace,
            configured_name=configured_name,
        )
    ]


def _looks_like_path(text: str) -> bool:
    return "/" in text or "\\" in text or text.endswith(".md") or text.startswith(".")


def _normalize_skills(raw_skills: Any, *, workspace: Path) -> dict[str, _SkillBinding]:
    skill_map: dict[str, _SkillBinding] = {}
    if not isinstance(raw_skills, list):
        return skill_map

    for item in raw_skills:
        if isinstance(item, str):
            value = item.strip()
            if not value:
                continue

            if _looks_like_path(value) or _path_exists(value, workspace=workspace):
                for binding in _bindings_from_path(value, workspace=workspace):
                    skill_map[binding.name] = binding
            else:
                skill_map[value] = _SkillBinding(name=value, display_name=value)
            continue

        if not isinstance(item, dict):
            continue

        configured_name_raw = item.get("name") or item.get("skill_name")
        configured_name = (
            configured_name_raw.strip()
            if isinstance(configured_name_raw, str) and configured_name_raw.strip()
            else None
        )

        location = (
            item.get("location")
            or item.get("skill_md_path")
            or item.get("skill_directory")
            or item.get("directory")
            or item.get("path")
        )
        if isinstance(location, str) and location.strip():
            bindings = _bindings_from_path(
                location.strip(),
                workspace=workspace,
                configured_name=configured_name,
            )
            for binding in bindings:
                key = configured_name or binding.name
                skill_map[key] = binding
                configured_name = None
            continue

        if not configured_name:
            continue

        display_name_raw = item.get("display_name")
        display_name = (
            display_name_raw.strip()
            if isinstance(display_name_raw, str) and display_name_raw.strip()
            else configured_name
        )
        metadata_value = item.get("metadata")
        metadata: dict[str, str] | None = None
        if isinstance(metadata_value, dict):
            metadata = {str(key): str(value) for key, value in metadata_value.items()}

        allowed_tools_raw = item.get("allowed-tools")
        if not isinstance(allowed_tools_raw, str) or not allowed_tools_raw.strip():
            allowed_tools_raw = item.get("allowed_tools")

        skill_map[configured_name] = _SkillBinding(
            name=configured_name,
            display_name=display_name,
            description=str(item.get("description", "") or "").strip(),
            instructions=str(item.get("instructions", "") or ""),
            compatibility=str(item.get("compatibility") or "").strip() or None,
            allowed_tools=str(allowed_tools_raw).strip() if isinstance(allowed_tools_raw, str) else None,
            metadata=metadata,
        )

    return skill_map


def _resolve_instruction(binding: _SkillBinding) -> str:
    instructions = binding.instructions.strip()
    if instructions:
        return instructions
    return (
        f"Skill '{binding.display_name}' is activated, but no instruction text is available. "
        "Please inspect the skill files or provide explicit instructions."
    )


def activate_skill(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    skill_name = str(arguments.get("skill_name", "")).strip()
    reason = str(arguments.get("reason", "")).strip()
    if not skill_name:
        return _error(error_code="skill_name_required", message="`skill_name` is required")

    raw_skills = context.shared_state.get("available_skills")
    if raw_skills is None:
        raw_skills = context.shared_state.get("bound_skills")

    skill_map = _normalize_skills(raw_skills, workspace=context.workspace)
    if not skill_map:
        return _error(
            error_code="no_bound_skills_configured",
            message="No bound skills are configured for this task",
        )

    binding = skill_map.get(skill_name)
    if binding is None:
        return _error(
            error_code="skill_not_allowed",
            message=f"Skill '{skill_name}' is not allowed for this task",
        )
    if binding.load_error:
        return _error(
            error_code="skill_invalid",
            message=f"Skill '{skill_name}' is invalid: {binding.load_error}",
        )

    instructions = _resolve_instruction(binding)
    active_skills = context.shared_state.setdefault("active_skills", [])
    if isinstance(active_skills, list) and binding.name not in active_skills:
        active_skills.append(binding.name)

    activation_log = context.shared_state.setdefault("skill_activation_log", [])
    if isinstance(activation_log, list):
        activation_log.append(
            {
                "skill_name": binding.name,
                "reason": reason,
                "cycle_index": context.cycle_index,
            }
        )

    response_data: dict[str, Any] = {
        "status": "activated",
        "skill_name": binding.name,
        "skill_display_name": binding.display_name,
        "message": f"Skill '{binding.display_name}' has been activated. Follow the instructions below.",
        "instructions": instructions,
    }
    if binding.description:
        response_data["description"] = binding.description
    if binding.location:
        response_data["location"] = binding.location
    if binding.compatibility:
        response_data["compatibility"] = binding.compatibility
    if binding.allowed_tools:
        response_data["allowed_tools"] = binding.allowed_tools
    if binding.metadata:
        response_data["metadata"] = binding.metadata
    if reason:
        response_data["reason"] = reason

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        status_code=ToolResultStatus.SUCCESS,
        content=to_json(response_data),
        metadata=response_data,
    )
