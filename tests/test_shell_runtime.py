from __future__ import annotations

from pathlib import Path

import pytest

from vv_agent.runtime import shell as shell_runtime


def test_prepare_shell_execution_posix_auto_confirm(monkeypatch) -> None:
    monkeypatch.setattr(shell_runtime.os, "name", "posix", raising=False)

    invocation, stdin = shell_runtime.prepare_shell_execution(
        "echo hello",
        auto_confirm=True,
        stdin="payload",
    )

    assert invocation == ["bash", "-lc", "yes | (echo hello)"]
    assert stdin == "payload"


def test_prepare_shell_execution_windows_auto_confirm(monkeypatch) -> None:
    monkeypatch.setattr(shell_runtime.os, "name", "nt", raising=False)
    monkeypatch.setitem(shell_runtime.os.environ, "COMSPEC", "cmd.exe")

    invocation, stdin = shell_runtime.prepare_shell_execution(
        "echo hello",
        auto_confirm=True,
        stdin=None,
    )

    assert invocation == ["cmd.exe", "/c", "echo hello"]
    assert stdin is not None
    assert stdin.startswith("y\ny\n")


def test_build_shell_invocation_windows_fallback(monkeypatch) -> None:
    monkeypatch.setattr(shell_runtime.os, "name", "nt", raising=False)
    monkeypatch.delenv("COMSPEC", raising=False)

    invocation = shell_runtime.build_shell_invocation("ver")
    assert Path(invocation[0]).name.lower() == "cmd.exe"
    assert invocation[1:] == ["/c", "ver"]


def test_build_shell_invocation_windows_priority_prefers_git_bash(monkeypatch) -> None:
    monkeypatch.setattr(shell_runtime.os, "name", "nt", raising=False)
    monkeypatch.setattr(shell_runtime, "_resolve_windows_git_bash", lambda: r"C:\\Program Files\\Git\\bin\\bash.exe")
    monkeypatch.setattr(
        shell_runtime,
        "_resolve_windows_powershell",
        lambda: r"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
    )

    invocation = shell_runtime.build_shell_invocation(
        "echo hello",
        windows_shell_priority=["git-bash", "powershell", "cmd"],
    )

    assert invocation == [r"C:\\Program Files\\Git\\bin\\bash.exe", "-lc", "echo hello"]


def test_build_shell_invocation_windows_priority_falls_back_to_powershell(monkeypatch) -> None:
    monkeypatch.setattr(shell_runtime.os, "name", "nt", raising=False)
    monkeypatch.setattr(shell_runtime, "_resolve_windows_git_bash", lambda: None)
    monkeypatch.setattr(
        shell_runtime,
        "_resolve_windows_powershell",
        lambda: r"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
    )

    invocation = shell_runtime.build_shell_invocation(
        "Write-Host hello",
        windows_shell_priority=["git-bash", "powershell", "cmd"],
    )

    assert invocation == [
        r"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-Command",
        "Write-Host hello",
    ]


def test_build_shell_invocation_windows_explicit_unavailable_shell_raises(monkeypatch) -> None:
    monkeypatch.setattr(shell_runtime.os, "name", "nt", raising=False)
    monkeypatch.setattr(shell_runtime, "_resolve_windows_git_bash", lambda: None)

    with pytest.raises(ValueError, match="Configured shell is unavailable"):
        shell_runtime.build_shell_invocation("echo hello", shell="git-bash")
