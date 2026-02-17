from __future__ import annotations

from typing import Any

from v_agent.tools.base import ToolContext
from v_agent.tools.handlers.common import to_json
from v_agent.types import ToolExecutionResult, ToolResultStatus


def activate_skill(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context, arguments
    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        status_code=ToolResultStatus.ERROR,
        error_code="skill_activation_not_enabled",
        content=to_json(
            {
                "error": "Skill activation is not enabled in this runtime",
                "error_code": "skill_activation_not_enabled",
            }
        ),
    )
