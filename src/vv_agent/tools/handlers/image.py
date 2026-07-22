from __future__ import annotations

import base64
from pathlib import PurePosixPath
from typing import Any

from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.common import builtin_error, to_json
from vv_agent.types import ToolExecutionResult, ToolResultStatus

_ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_EXTENSION_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
_MAX_INLINE_IMAGE_BYTES = 5 * 1024 * 1024


def read_image(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    raw_path = str(arguments.get("path", "")).strip()
    if not raw_path:
        return builtin_error("`path` is required", "path_required")

    lowered = raw_path.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        payload = {
            "status": "loaded",
            "source": "url",
            "image_url": raw_path,
        }
        return ToolExecutionResult(
            tool_call_id="",
            status_code=ToolResultStatus.SUCCESS,
            content=to_json(payload),
            image_url=raw_path,
            metadata={"source": "url"},
        )

    try:
        context.resolve_workspace_path(raw_path)
    except ValueError as exc:
        return builtin_error(str(exc), "path_escapes_workspace")

    backend = context.workspace_backend
    if not backend.exists(raw_path) or not backend.is_file(raw_path):
        return builtin_error(f"image file not found: {raw_path}", "image_not_found")

    suffix = PurePosixPath(raw_path).suffix.lower()
    if suffix not in _ALLOWED_IMAGE_EXTENSIONS:
        return builtin_error(
            f"unsupported image format: {suffix}",
            "unsupported_image_format",
        )

    try:
        image_bytes = backend.read_bytes(raw_path)
    except OSError as exc:
        return builtin_error(str(exc), "image_not_found")
    if len(image_bytes) > _MAX_INLINE_IMAGE_BYTES:
        details = {
            "max_bytes": _MAX_INLINE_IMAGE_BYTES,
            "actual_bytes": len(image_bytes),
        }
        return builtin_error(
            "image is too large for inline message transport",
            "image_too_large",
            details=details,
            metadata=details,
        )

    relative_path = raw_path
    suffix = PurePosixPath(raw_path).suffix.lower()
    mime_type = _EXTENSION_TO_MIME.get(suffix, "application/octet-stream")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    image_url = f"data:{mime_type};base64,{encoded}"
    payload = {
        "status": "loaded",
        "source": "workspace",
        "image_path": relative_path,
        "mime_type": mime_type,
        "inline_transport": True,
    }
    return ToolExecutionResult(
        tool_call_id="",
        status_code=ToolResultStatus.SUCCESS,
        content=to_json(payload),
        image_url=image_url,
        image_path=relative_path,
        metadata={
            "source": "workspace",
            "mime_type": mime_type,
            "inline_transport": True,
        },
    )
