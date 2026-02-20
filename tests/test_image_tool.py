from __future__ import annotations

import json
from pathlib import Path

from v_agent.constants import READ_IMAGE_TOOL_NAME
from v_agent.tools import ToolContext, build_default_registry
from v_agent.types import ToolCall, ToolResultStatus
from v_agent.workspace import LocalWorkspaceBackend

_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a"
    "0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360000000020001e221bc330000000049454e44ae426082"
)


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace=tmp_path, shared_state={"todo_list": []},
        cycle_index=1, workspace_backend=LocalWorkspaceBackend(tmp_path),
    )


def test_read_image_from_workspace_file(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)

    image_path = tmp_path / "img.png"
    image_path.write_bytes(_PNG_1X1)

    result = registry.execute(
        ToolCall(id="c1", name=READ_IMAGE_TOOL_NAME, arguments={"path": "img.png"}),
        context,
    )

    payload = json.loads(result.content)
    assert result.status_code == ToolResultStatus.SUCCESS
    assert result.image_path == "img.png"
    assert isinstance(result.image_url, str)
    assert result.image_url.startswith("data:image/png;base64,")
    assert payload["source"] == "workspace"
    assert payload["inline_transport"] is True


def test_read_image_from_url(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)

    result = registry.execute(
        ToolCall(id="c2", name=READ_IMAGE_TOOL_NAME, arguments={"path": "https://example.com/a.png"}),
        context,
    )

    assert result.status_code == ToolResultStatus.SUCCESS
    assert result.image_url == "https://example.com/a.png"


def test_read_image_rejects_unsupported_extension(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)

    text_file = tmp_path / "x.txt"
    text_file.write_text("not image", encoding="utf-8")

    result = registry.execute(
        ToolCall(id="c3", name=READ_IMAGE_TOOL_NAME, arguments={"path": "x.txt"}),
        context,
    )

    assert result.status_code == ToolResultStatus.ERROR
    assert result.error_code == "unsupported_image_format"
