from __future__ import annotations

import json
import time
from pathlib import Path

from v_agent.constants import BASH_TOOL_NAME, CHECK_BACKGROUND_COMMAND_TOOL_NAME
from v_agent.tools import ToolContext, build_default_registry
from v_agent.types import ToolCall, ToolResultStatus
from v_agent.workspace import LocalWorkspaceBackend


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace=tmp_path, shared_state={"todo_list": []},
        cycle_index=1, workspace_backend=LocalWorkspaceBackend(tmp_path),
    )


def test_bash_tool_executes_command(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)

    result = registry.execute(
        ToolCall(
            id="c1",
            name=BASH_TOOL_NAME,
            arguments={"command": "echo hello"},
        ),
        context,
    )

    payload = json.loads(result.content)
    assert result.status_code == ToolResultStatus.SUCCESS
    assert payload["exit_code"] == 0
    assert "hello" in payload["output"]


def test_bash_tool_blocks_dangerous_command(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)

    result = registry.execute(
        ToolCall(
            id="c2",
            name=BASH_TOOL_NAME,
            arguments={"command": "rm -rf /"},
        ),
        context,
    )

    assert result.status_code == ToolResultStatus.ERROR
    assert result.error_code == "dangerous_command"


def test_background_command_lifecycle(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)

    start = registry.execute(
        ToolCall(
            id="c3",
            name=BASH_TOOL_NAME,
            arguments={
                "command": "python -c \"import time; time.sleep(0.2); print('done')\"",
                "run_in_background": True,
                "timeout": 5,
            },
        ),
        context,
    )
    start_payload = json.loads(start.content)
    assert start.status_code == ToolResultStatus.RUNNING
    session_id = start_payload["session_id"]

    final_payload: dict[str, object] | None = None
    for _ in range(20):
        probe = registry.execute(
            ToolCall(
                id="c4",
                name=CHECK_BACKGROUND_COMMAND_TOOL_NAME,
                arguments={"session_id": session_id},
            ),
            context,
        )
        probe_payload = json.loads(probe.content)
        if probe.status_code == ToolResultStatus.RUNNING:
            time.sleep(0.05)
            continue
        final_payload = probe_payload
        break

    assert final_payload is not None
    assert final_payload["status"] == "completed"
    assert "done" in str(final_payload.get("output", ""))
