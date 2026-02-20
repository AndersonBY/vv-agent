from __future__ import annotations

import os

_WINDOWS_AUTO_CONFIRM_LINES = 512


def build_shell_invocation(command: str) -> list[str]:
    if os.name == "nt":
        return [os.environ.get("COMSPEC", "cmd.exe"), "/c", command]
    return ["bash", "-lc", command]


def prepare_shell_execution(
    command: str,
    *,
    auto_confirm: bool,
    stdin: str | None,
) -> tuple[list[str], str | None]:
    if not auto_confirm:
        return build_shell_invocation(command), stdin

    if os.name == "nt":
        # cmd.exe has no built-in equivalent to `yes`, so prefill stdin with confirmations.
        auto_confirm_stdin = "y\n" * _WINDOWS_AUTO_CONFIRM_LINES
        return build_shell_invocation(command), f"{auto_confirm_stdin}{stdin or ''}"

    return build_shell_invocation(f"yes | ({command})"), stdin
