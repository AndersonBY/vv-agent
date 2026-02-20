from __future__ import annotations

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

    assert shell_runtime.build_shell_invocation("ver") == ["cmd.exe", "/c", "ver"]
