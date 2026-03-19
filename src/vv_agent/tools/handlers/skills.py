from __future__ import annotations

from typing import Any

from vv_agent.skills.normalize import normalize_skill_list
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import to_json
from vv_agent.types import ToolExecutionResult, ToolResultStatus


def _error(*, error_code: str, message: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        status_code=ToolResultStatus.ERROR,
        error_code=error_code,
        content=to_json({"error": message, "error_code": error_code}),
    )


def activate_skill(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    skill_name = str(arguments.get("skill_name", "")).strip()
    reason = str(arguments.get("reason", "")).strip()
    if not skill_name:
        return _error(error_code="skill_name_required", message="`skill_name` is required")

    raw_skills = context.shared_state.get("available_skills")
    entries = normalize_skill_list(raw_skills, workspace=context.workspace, load_instructions=True)
    entry_map = {e.name: e for e in entries}

    if not entry_map:
        return _error(
            error_code="no_skills_configured",
            message="No skills are configured for this task",
        )

    entry = entry_map.get(skill_name)
    if entry is None:
        return _error(
            error_code="skill_not_allowed",
            message=f"Skill '{skill_name}' is not allowed for this task",
        )
    if entry.load_error:
        return _error(
            error_code="skill_invalid",
            message=f"Skill '{skill_name}' is invalid: {entry.load_error}",
        )

    instructions = (entry.instructions or "").strip()
    if not instructions:
        instructions = (
            f"Skill '{skill_name}' is activated, but no instruction text is available. "
            "Please inspect the skill files or provide explicit instructions."
        )

    active_skills = context.shared_state.setdefault("active_skills", [])
    if isinstance(active_skills, list) and entry.name not in active_skills:
        active_skills.append(entry.name)

    activation_log = context.shared_state.setdefault("skill_activation_log", [])
    if isinstance(activation_log, list):
        activation_log.append(
            {
                "skill_name": entry.name,
                "reason": reason,
                "cycle_index": context.cycle_index,
            }
        )

    response_data: dict[str, Any] = {
        "status": "activated",
        "skill_name": entry.name,
        "message": f"Skill '{entry.name}' has been activated. Follow the instructions below.",
        "instructions": instructions,
    }
    if entry.description:
        response_data["description"] = entry.description
    if entry.location:
        response_data["location"] = entry.location
    if entry.compatibility:
        response_data["compatibility"] = entry.compatibility
    if entry.allowed_tools:
        response_data["allowed_tools"] = entry.allowed_tools
    if entry.metadata:
        response_data["metadata"] = entry.metadata
    if reason:
        response_data["reason"] = reason

    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        status_code=ToolResultStatus.SUCCESS,
        content=to_json(response_data),
        metadata=response_data,
    )
