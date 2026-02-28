from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from vv_agent.constants import BASH_TOOL_NAME, CHECK_BACKGROUND_COMMAND_TOOL_NAME
from vv_agent.tools import ToolContext, build_default_registry
from vv_agent.tools.handlers import bash as bash_handler
from vv_agent.types import ToolCall, ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend


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
                "command": (
                    f'"{sys.executable}" -c "import time; time.sleep(0.2); print(\'done\')"'
                ),
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


def test_bash_tool_uses_context_shell_defaults(tmp_path: Path, monkeypatch) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)
    context.task_metadata = {
        "bash_shell": "powershell",
        "windows_shell_priority": ["git-bash", "powershell", "cmd"],
    }

    captured: dict[str, object] = {}

    def fake_prepare(
        command: str,
        *,
        auto_confirm: bool,
        stdin: str | None,
        shell: str | None = None,
        windows_shell_priority: list[str] | None = None,
    ) -> tuple[list[str], str | None]:
        captured["command"] = command
        captured["auto_confirm"] = auto_confirm
        captured["stdin"] = stdin
        captured["shell"] = shell
        captured["windows_shell_priority"] = windows_shell_priority
        return ["powershell", "-Command", command], stdin

    monkeypatch.setattr(bash_handler, "prepare_shell_execution", fake_prepare)
    monkeypatch.setattr(
        bash_handler.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="ok\n", stderr="", returncode=0),
    )

    result = registry.execute(
        ToolCall(
            id="c5",
            name=BASH_TOOL_NAME,
            arguments={
                "command": "echo ok",
                "shell": "cmd",  # ignored: runtime shell config must come from caller metadata.
            },
        ),
        context,
    )

    payload = json.loads(result.content)
    assert result.status_code == ToolResultStatus.SUCCESS
    assert payload["exit_code"] == 0
    assert captured["shell"] == "powershell"
    assert captured["windows_shell_priority"] == ["git-bash", "powershell", "cmd"]


def test_bash_tool_applies_context_bash_env(tmp_path: Path, monkeypatch) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)
    context.task_metadata = {
        "bash_env": {
            "VV_AGENT_CUSTOM_ENV": "custom-value",
            "VV_AGENT_SHARED_ENV": "from-task",
        }
    }

    captured: dict[str, object] = {}

    def fake_prepare(
        command: str,
        *,
        auto_confirm: bool,
        stdin: str | None,
        shell: str | None = None,
        windows_shell_priority: list[str] | None = None,
    ) -> tuple[list[str], str | None]:
        del auto_confirm, shell, windows_shell_priority
        return ["bash", "-lc", command], stdin

    def fake_run(*args, **kwargs):
        del args
        captured["env"] = kwargs.get("env")
        return SimpleNamespace(stdout="ok\n", stderr="", returncode=0)

    monkeypatch.setenv("VV_AGENT_SHARED_ENV", "from-process")
    monkeypatch.setenv("VV_AGENT_BASE_ENV", "base-value")
    monkeypatch.setattr(bash_handler, "prepare_shell_execution", fake_prepare)
    monkeypatch.setattr(bash_handler.subprocess, "run", fake_run)

    result = registry.execute(
        ToolCall(
            id="c6",
            name=BASH_TOOL_NAME,
            arguments={"command": "echo ok"},
        ),
        context,
    )

    payload = json.loads(result.content)
    assert result.status_code == ToolResultStatus.SUCCESS
    assert payload["exit_code"] == 0
    process_env = captured.get("env")
    assert isinstance(process_env, dict)
    assert process_env["VV_AGENT_CUSTOM_ENV"] == "custom-value"
    assert process_env["VV_AGENT_SHARED_ENV"] == "from-task"
    assert process_env["VV_AGENT_BASE_ENV"] == "base-value"


def test_bash_tool_rejects_invalid_bash_env_metadata(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)
    context.task_metadata = {"bash_env": ["invalid"]}

    result = registry.execute(
        ToolCall(
            id="c7",
            name=BASH_TOOL_NAME,
            arguments={"command": "echo ok"},
        ),
        context,
    )

    payload = json.loads(result.content)
    assert result.status_code == ToolResultStatus.ERROR
    assert result.error_code == "invalid_shell_config"
    assert "bash_env" in payload["error"]
