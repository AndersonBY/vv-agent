from __future__ import annotations

from typing import Any

from v_agent.tools.base import ToolContext
from v_agent.tools.handlers.common import resolve_workspace_path, to_json
from v_agent.types import ToolExecutionResult, ToolResultStatus

_ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_image(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    raw_path = str(arguments.get("path", "")).strip()
    if not raw_path:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="path_required",
            content=to_json({"error": "`path` is required"}),
        )

    lowered = raw_path.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        payload = {
            "status": "loaded",
            "source": "url",
            "image_url": raw_path,
        }
        return ToolExecutionResult(
            tool_call_id="",
            status="success",
            status_code=ToolResultStatus.SUCCESS,
            content=to_json(payload),
            image_url=raw_path,
            metadata=payload,
        )

    target = resolve_workspace_path(context, raw_path)
    if not target.exists() or not target.is_file():
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="image_not_found",
            content=to_json({"error": f"image file not found: {raw_path}"}),
        )

    if target.suffix.lower() not in _ALLOWED_IMAGE_EXTENSIONS:
        return ToolExecutionResult(
            tool_call_id="",
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="unsupported_image_format",
            content=to_json({"error": f"unsupported image format: {target.suffix}"}),
        )

    relative_path = target.relative_to(context.workspace).as_posix()
    payload = {
        "status": "loaded",
        "source": "workspace",
        "image_path": relative_path,
    }
    return ToolExecutionResult(
        tool_call_id="",
        status="success",
        status_code=ToolResultStatus.SUCCESS,
        content=to_json(payload),
        image_path=relative_path,
        metadata=payload,
    )
