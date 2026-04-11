from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from vv_agent.constants import BASH_TOOL_NAME, CHECK_BACKGROUND_COMMAND_TOOL_NAME
from vv_agent.runtime import background_sessions as background_runtime
from vv_agent.runtime import processes as process_runtime
from vv_agent.tools import ToolContext, build_default_registry
from vv_agent.tools.handlers import bash as bash_handler
from vv_agent.types import ToolCall, ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace=tmp_path,
        shared_state={"todo_list": []},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
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
    assert "command" not in payload


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


def test_bash_tool_allows_absolute_exec_dir_when_enabled(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)
    outside_dir = (tmp_path.parent / f"{tmp_path.name}_outside_exec").resolve()
    outside_dir.mkdir(parents=True, exist_ok=True)
    context.task_metadata = {"allow_outside_workspace_paths": True}
    context.workspace_backend = LocalWorkspaceBackend(
        tmp_path,
        allow_outside_root=True,
    )

    result = registry.execute(
        ToolCall(
            id="c2_abs",
            name=BASH_TOOL_NAME,
            arguments={"command": "echo outside", "exec_dir": str(outside_dir)},
        ),
        context,
    )

    payload = json.loads(result.content)
    assert result.status_code == ToolResultStatus.SUCCESS
    assert payload["cwd"] == str(outside_dir)


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
    assert "command" not in start_payload
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
    assert final_payload["command"] == (
        f'"{sys.executable}" -c "import time; time.sleep(0.2); print(\'done\')"'
    )
    assert "done" in str(final_payload.get("output", ""))


def test_background_command_listener_receives_terminal_event(tmp_path: Path) -> None:
    manager = background_runtime.BackgroundSessionManager()
    notified = threading.Event()
    received: dict[str, object] = {}

    command = f'"{sys.executable}" -c "import time; time.sleep(0.2); print(\'done\')"'
    session_id = manager.start(command=command, cwd=tmp_path, timeout_seconds=5)

    manager.subscribe(
        session_id,
        lambda payload: (received.update(payload), notified.set()),
    )

    assert notified.wait(timeout=5.0) is True
    assert received["session_id"] == session_id
    assert received["status"] == "completed"
    assert received["command"] == command
    assert "done" in str(received.get("output", ""))


def test_bash_tool_uses_context_shell_defaults(tmp_path: Path, monkeypatch) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)
    context.task_metadata = {
        "bash_shell": "powershell",
        "windows_shell_priority": ["git-bash", "powershell", "cmd"],
    }

    captured: dict[str, object] = {}
    output_file = tmp_path / "shell-defaults.log"
    output_file.write_text("ok\n", encoding="utf-8")

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

    class _FakeProcess:
        returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            captured["wait_timeout"] = timeout
            return 0

        def poll(self) -> int:
            return 0

    def fake_start(command: list[str], *, cwd: Path, stdin_text: str | None, env=None):
        del command, cwd, stdin_text, env
        return SimpleNamespace(process=_FakeProcess(), output_path=output_file)

    monkeypatch.setattr(bash_handler, "prepare_shell_execution", fake_prepare)
    monkeypatch.setattr(bash_handler, "start_captured_process", fake_start)
    monkeypatch.setattr(
        bash_handler,
        "read_captured_output",
        lambda path, *, limit_chars: Path(path).read_text(encoding="utf-8")[:limit_chars],
    )
    monkeypatch.setattr(bash_handler, "remove_captured_output", lambda path: None)

    result = registry.execute(
        ToolCall(
            id="c5",
            name=BASH_TOOL_NAME,
            arguments={
                "command": "echo ok",
                "shell": "cmd",
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
    output_file = tmp_path / "env.log"
    output_file.write_text("ok\n", encoding="utf-8")

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

    class _FakeProcess:
        returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def poll(self) -> int:
            return 0

    def fake_start(command: list[str], *, cwd: Path, stdin_text: str | None, env=None):
        del command, cwd, stdin_text
        captured["env"] = env
        return SimpleNamespace(process=_FakeProcess(), output_path=output_file)

    monkeypatch.setenv("VV_AGENT_SHARED_ENV", "from-process")
    monkeypatch.setenv("VV_AGENT_BASE_ENV", "base-value")
    monkeypatch.setattr(bash_handler, "prepare_shell_execution", fake_prepare)
    monkeypatch.setattr(bash_handler, "start_captured_process", fake_start)
    monkeypatch.setattr(
        bash_handler,
        "read_captured_output",
        lambda path, *, limit_chars: Path(path).read_text(encoding="utf-8")[:limit_chars],
    )
    monkeypatch.setattr(bash_handler, "remove_captured_output", lambda path: None)

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
    raw_env = captured.get("env")
    assert isinstance(raw_env, dict)
    process_env = cast("dict[str, str]", raw_env)
    assert process_env["VV_AGENT_CUSTOM_ENV"] == "custom-value"
    assert process_env["VV_AGENT_SHARED_ENV"] == "from-task"
    assert process_env["VV_AGENT_BASE_ENV"] == "base-value"


def test_build_process_env_injects_windows_python_encoding_defaults(monkeypatch) -> None:
    monkeypatch.setattr(bash_handler.os, "name", "nt", raising=False)
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)

    process_env = bash_handler._build_process_env(None)

    assert process_env is not None
    assert process_env["PYTHONUTF8"] == "1"
    assert process_env["PYTHONIOENCODING"] == "utf-8"


def test_build_process_env_preserves_explicit_windows_python_encoding_overrides(monkeypatch) -> None:
    monkeypatch.setattr(bash_handler.os, "name", "nt", raising=False)
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setenv("PYTHONIOENCODING", "gbk")

    process_env = bash_handler._build_process_env({"PYTHONIOENCODING": "utf-8:replace"})

    assert process_env is not None
    assert process_env["PYTHONUTF8"] == "0"
    assert process_env["PYTHONIOENCODING"] == "utf-8:replace"


def test_start_captured_process_uses_replace_error_handler_for_decoding(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeProcess:
        stdin = None

    def fake_popen(*args, **kwargs):
        del args
        captured["text"] = kwargs.get("text")
        captured["errors"] = kwargs.get("errors")
        captured["stdin"] = kwargs.get("stdin")
        return _FakeProcess()

    monkeypatch.setattr(process_runtime.subprocess, "Popen", fake_popen)

    started = process_runtime.start_captured_process(["bash", "-lc", "echo ok"], cwd=tmp_path)

    assert captured["text"] is True
    assert captured["errors"] == "replace"
    assert captured["stdin"] == process_runtime.subprocess.DEVNULL
    process_runtime.remove_captured_output(started.output_path)


def test_start_captured_process_hides_console_window_on_windows(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeProcess:
        stdin = None

    class _StartupInfo:
        def __init__(self) -> None:
            self.dwFlags = 0
            self.wShowWindow = None

    def fake_popen(*args, **kwargs):
        del args
        captured["creationflags"] = kwargs.get("creationflags")
        captured["startupinfo"] = kwargs.get("startupinfo")
        captured["start_new_session"] = kwargs.get("start_new_session")
        return _FakeProcess()

    monkeypatch.setattr(process_runtime.os, "name", "nt", raising=False)
    monkeypatch.setattr(process_runtime.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    monkeypatch.setattr(process_runtime.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(process_runtime.subprocess, "STARTF_USESHOWWINDOW", 0x001, raising=False)
    monkeypatch.setattr(process_runtime.subprocess, "SW_HIDE", 0, raising=False)
    monkeypatch.setattr(process_runtime.subprocess, "STARTUPINFO", _StartupInfo, raising=False)
    monkeypatch.setattr(process_runtime.subprocess, "Popen", fake_popen)

    started = process_runtime.start_captured_process(["bash", "-lc", "echo ok"], cwd=tmp_path)

    assert captured["creationflags"] == (0x200 | 0x08000000)
    startupinfo = captured["startupinfo"]
    assert isinstance(startupinfo, _StartupInfo)
    assert startupinfo.dwFlags == 0x001
    assert startupinfo.wShowWindow == 0
    assert captured["start_new_session"] is None
    process_runtime.remove_captured_output(started.output_path)


def test_background_session_uses_replace_error_handler_for_decoding(tmp_path: Path, monkeypatch) -> None:
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

    class _FakeStdin:
        def write(self, value: str) -> None:
            del value

        def close(self) -> None:
            pass

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdin: _FakeStdin | None = _FakeStdin()

        def poll(self):
            return None

    def fake_popen(*args, **kwargs):
        del args
        captured["text"] = kwargs.get("text")
        captured["errors"] = kwargs.get("errors")
        return _FakeProcess()

    monkeypatch.setattr(background_runtime, "prepare_shell_execution", fake_prepare)
    monkeypatch.setattr(process_runtime.subprocess, "Popen", fake_popen)

    manager = background_runtime.BackgroundSessionManager()
    session_id = manager.start(command="echo ok", cwd=tmp_path, timeout_seconds=5)

    assert session_id.startswith("bg_")
    assert captured["text"] is True
    assert captured["errors"] == "replace"


def test_background_session_manager_can_adopt_running_process(tmp_path: Path) -> None:
    output_file = tmp_path / "adopt.log"
    output_file.write_text("still running\n", encoding="utf-8")

    class _FakeProcess:
        returncode = None

        def poll(self):
            return None

    manager = background_runtime.BackgroundSessionManager()
    session_id = manager.adopt_running_process(
        command="sleep 10",
        cwd=tmp_path,
        timeout_seconds=5,
        process=cast(subprocess.Popen[str], _FakeProcess()),
        output_path=output_file,
        shell="bash",
    )

    payload = manager.check(session_id)

    assert payload["status"] == "running"
    assert payload["session_id"] == session_id
    assert payload["command"] == "sleep 10"
    assert payload["shell"] == "bash"


def test_bash_tool_timeout_moves_process_to_background_and_returns_session(tmp_path: Path, monkeypatch) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)
    captured: dict[str, object] = {}
    output_file = tmp_path / "timeout.log"
    output_file.write_text("partial output\n", encoding="utf-8")

    def fake_prepare(
        command: str,
        *,
        auto_confirm: bool,
        stdin: str | None,
        shell: str | None = None,
        windows_shell_priority: list[str] | None = None,
    ) -> tuple[list[str], str | None]:
        del auto_confirm, stdin, shell, windows_shell_priority
        return ["bash", "-lc", command], None

    class _FakeProcess:
        returncode = None

        def wait(self, timeout: float | None = None) -> int:
            raise subprocess.TimeoutExpired(cmd=["bash", "-lc", "sleep 10"], timeout=timeout or 0)

        def poll(self):
            return None

    def fake_start(command: list[str], *, cwd: Path, stdin_text: str | None, env=None):
        del command, cwd, stdin_text, env
        return SimpleNamespace(process=_FakeProcess(), output_path=output_file)

    def fake_adopt_running_process(
        *,
        command: str,
        cwd: Path,
        timeout_seconds: int,
        process,
        output_path: Path,
        shell: str | None = None,
        started_at: float | None = None,
    ) -> str:
        captured["command"] = command
        captured["cwd"] = cwd
        captured["timeout_seconds"] = timeout_seconds
        captured["process"] = process
        captured["output_path"] = output_path
        captured["shell"] = shell
        captured["started_at"] = started_at
        return "bg_timeout_123"

    monkeypatch.setattr(bash_handler, "prepare_shell_execution", fake_prepare)
    monkeypatch.setattr(bash_handler, "start_captured_process", fake_start)
    monkeypatch.setattr(
        bash_handler.background_session_manager,
        "adopt_running_process",
        fake_adopt_running_process,
    )
    monkeypatch.setattr(
        bash_handler,
        "read_captured_output",
        lambda path, *, limit_chars: Path(path).read_text(encoding="utf-8")[:limit_chars],
    )
    monkeypatch.setattr(bash_handler, "remove_captured_output", lambda path: None)

    result = registry.execute(
        ToolCall(
            id="c7_timeout",
            name=BASH_TOOL_NAME,
            arguments={"command": "sleep 10", "timeout": 1},
        ),
        context,
    )

    payload = json.loads(result.content)
    assert result.status_code == ToolResultStatus.RUNNING
    assert result.error_code is None
    assert payload["status"] == "running"
    assert payload["session_id"] == "bg_timeout_123"
    assert "command" not in payload
    assert payload["transitioned_to_background"] is True
    assert "check_background_command" in payload["message"]
    assert payload["output"] == "partial output\n"
    assert captured["command"] == "sleep 10"
    assert captured["cwd"] == tmp_path
    assert captured["timeout_seconds"] == 1
    assert captured["output_path"] == output_file


def test_background_session_timeout_kills_process_tree_and_reads_output(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    output_file = tmp_path / "background-timeout.log"
    output_file.write_text("background partial\n", encoding="utf-8")

    def fake_prepare(
        command: str,
        *,
        auto_confirm: bool,
        stdin: str | None,
        shell: str | None = None,
        windows_shell_priority: list[str] | None = None,
    ) -> tuple[list[str], str | None]:
        del auto_confirm, stdin, shell, windows_shell_priority
        return ["bash", "-lc", command], None

    class _FakeProcess:
        returncode = None

        def poll(self):
            return None

    def fake_start(command: list[str], *, cwd: Path, stdin_text: str | None, env=None):
        del command, cwd, stdin_text, env
        return SimpleNamespace(process=_FakeProcess(), output_path=output_file)

    def fake_kill(process):
        captured["killed"] = True
        process.returncode = -9

    monkeypatch.setattr(background_runtime, "prepare_shell_execution", fake_prepare)
    monkeypatch.setattr(background_runtime, "start_captured_process", fake_start)
    monkeypatch.setattr(background_runtime, "kill_process_tree", fake_kill)
    monkeypatch.setattr(
        background_runtime,
        "read_captured_output",
        lambda path, *, limit_chars: Path(path).read_text(encoding="utf-8")[:limit_chars],
    )

    manager = background_runtime.BackgroundSessionManager()
    session_id = manager.start(command="sleep 10", cwd=tmp_path, timeout_seconds=1)
    manager._sessions[session_id].started_at -= 5

    payload = manager.check(session_id)

    assert payload["status"] == "timeout"
    assert payload["output"] == "background partial\n"
    assert payload["exit_code"] == -9
    assert captured["killed"] is True


def test_bash_tool_rejects_invalid_bash_env_metadata(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(tmp_path)
    context.task_metadata = {"bash_env": ["invalid"]}

    result = registry.execute(
        ToolCall(
            id="c8",
            name=BASH_TOOL_NAME,
            arguments={"command": "echo ok"},
        ),
        context,
    )

    payload = json.loads(result.content)
    assert result.status_code == ToolResultStatus.ERROR
    assert result.error_code == "invalid_shell_config"
    assert "bash_env" in payload["error"]
