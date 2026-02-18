from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v_agent.tools.base import ToolContext
from v_agent.tools.handlers.common import to_json
from v_agent.types import ToolExecutionResult, ToolResultStatus


@dataclass(slots=True)
class _SkillConfig:
    name: str
    display_name: str
    description: str = ""
    instructions: str = ""
    skill_directory: str | None = None
    instruction_file: str | None = None


def _error(*, error_code: str, message: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        status_code=ToolResultStatus.ERROR,
        error_code=error_code,
        content=to_json(
            {
                "error": message,
                "error_code": error_code,
            }
        ),
    )


def _normalize_skills(raw_skills: Any) -> dict[str, _SkillConfig]:
    skills: dict[str, _SkillConfig] = {}
    if not isinstance(raw_skills, list):
        return skills

    for item in raw_skills:
        if isinstance(item, str):
            name = item.strip()
            if not name:
                continue
            skills[name] = _SkillConfig(name=name, display_name=name)
            continue

        if not isinstance(item, dict):
            continue

        raw_name = item.get("name") or item.get("skill_name")
        if not isinstance(raw_name, str):
            continue
        name = raw_name.strip()
        if not name:
            continue

        display_name_raw = item.get("display_name")
        display_name = display_name_raw.strip() if isinstance(display_name_raw, str) and display_name_raw.strip() else name
        description = str(item.get("description", "") or "")
        instructions = str(item.get("instructions", "") or "")

        skill_directory = item.get("skill_directory") or item.get("directory") or item.get("path")
        if isinstance(skill_directory, str) and skill_directory.strip():
            skill_directory_str: str | None = str(skill_directory).strip()
        else:
            skill_directory_str = None

        instruction_file = item.get("instruction_file") or item.get("skill_md_path")
        instruction_file_str = (
            str(instruction_file).strip() if isinstance(instruction_file, str) and instruction_file.strip() else None
        )

        skills[name] = _SkillConfig(
            name=name,
            display_name=display_name,
            description=description,
            instructions=instructions,
            skill_directory=skill_directory_str,
            instruction_file=instruction_file_str,
        )

    return skills


def _resolve_text_from_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _resolve_instruction(skill: _SkillConfig, *, workspace: Path) -> tuple[str, str | None]:
    direct = skill.instructions.strip()
    if direct:
        return direct, "metadata.instructions"

    directory: Path | None = None
    if skill.skill_directory:
        maybe_dir = Path(skill.skill_directory).expanduser()
        if not maybe_dir.is_absolute():
            maybe_dir = (workspace / maybe_dir).resolve()
        directory = maybe_dir

    if skill.instruction_file:
        instruction_file = Path(skill.instruction_file).expanduser()
        if not instruction_file.is_absolute():
            if directory is not None:
                instruction_file = (directory / instruction_file).resolve()
            else:
                instruction_file = (workspace / instruction_file).resolve()
        text = _resolve_text_from_file(instruction_file)
        if text is not None:
            return text, instruction_file.as_posix()

    if directory is not None:
        default_skill_md = (directory / "SKILL.md").resolve()
        text = _resolve_text_from_file(default_skill_md)
        if text is not None:
            return text, default_skill_md.as_posix()

    fallback = (
        f"Skill '{skill.display_name}' is activated, but no instruction text was found. "
        "Please check skill metadata or provide instruction_file/SKILL.md."
    )
    return fallback, None


def activate_skill(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    skill_name = str(arguments.get("skill_name", "")).strip()
    reason = str(arguments.get("reason", "")).strip()
    if not skill_name:
        return _error(error_code="skill_name_required", message="`skill_name` is required")

    raw_skills = context.shared_state.get("available_skills")
    if raw_skills is None:
        raw_skills = context.shared_state.get("bound_skills")

    skill_map = _normalize_skills(raw_skills)
    if not skill_map:
        return _error(
            error_code="no_bound_skills_configured",
            message="No bound skills are configured for this task",
        )

    skill = skill_map.get(skill_name)
    if skill is None:
        return _error(
            error_code="skill_not_allowed",
            message=f"Skill '{skill_name}' is not allowed for this task",
        )

    instructions, instruction_source = _resolve_instruction(skill, workspace=context.workspace)

    active_skills = context.shared_state.setdefault("active_skills", [])
    if isinstance(active_skills, list) and skill.name not in active_skills:
        active_skills.append(skill.name)

    activation_log = context.shared_state.setdefault("skill_activation_log", [])
    if isinstance(activation_log, list):
        activation_log.append(
            {
                "skill_name": skill.name,
                "reason": reason,
                "cycle_index": context.cycle_index,
            }
        )

    response_data: dict[str, Any] = {
        "status": "activated",
        "skill_name": skill.name,
        "skill_display_name": skill.display_name,
        "message": f"Skill '{skill.display_name}' has been activated. Follow the instructions below.",
        "instructions": instructions,
    }
    if skill.description:
        response_data["description"] = skill.description
    if reason:
        response_data["reason"] = reason
    if skill.skill_directory:
        response_data["skill_directory"] = skill.skill_directory
        response_data["resources_note"] = (
            f"Additional skill docs/resources may exist under: {skill.skill_directory}/ "
            "Use file tools to read them when needed."
        )
    if instruction_source:
        response_data["instruction_source"] = instruction_source

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        status_code=ToolResultStatus.SUCCESS,
        content=to_json(response_data),
        metadata=response_data,
    )
