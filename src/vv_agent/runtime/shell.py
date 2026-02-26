from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

_WINDOWS_AUTO_CONFIRM_LINES = 512
_WINDOWS_DEFAULT_SHELL_PRIORITY = ("cmd",)
_WINDOWS_GIT_BASH_RELATIVE_PATHS = (
    ("Git", "bin", "bash.exe"),
    ("Git", "usr", "bin", "bash.exe"),
)
_WINDOWS_POWERSHELL_RELATIVE_PATH = ("System32", "WindowsPowerShell", "v1.0", "powershell.exe")


@dataclass(slots=True, frozen=True)
class ShellInvocation:
    kind: str
    prefix: list[str]


def _normalize_shell_name(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _looks_like_path(value: str) -> bool:
    return any(token in value for token in ("/", "\\", ":"))


def _windows_program_roots() -> list[str]:
    roots: list[str] = []
    for env_key in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)", "LocalAppData"):
        raw = os.environ.get(env_key, "").strip()
        if raw and raw not in roots:
            roots.append(raw)
    return roots


def _resolve_windows_powershell() -> str | None:
    for exe_name in ("powershell.exe", "powershell"):
        resolved = shutil.which(exe_name)
        if resolved:
            return resolved

    system_root = os.environ.get("SYSTEMROOT", "").strip()
    if system_root:
        candidate = Path(system_root, *_WINDOWS_POWERSHELL_RELATIVE_PATH)
        if candidate.is_file():
            return str(candidate)
    return None


def _resolve_windows_git_bash() -> str | None:
    candidates: list[Path] = []
    for root in _windows_program_roots():
        root_path = Path(root)
        for rel_path in _WINDOWS_GIT_BASH_RELATIVE_PATHS:
            candidate = root_path.joinpath(*rel_path)
            if candidate not in candidates:
                candidates.append(candidate)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    for exe_name in ("bash.exe", "bash"):
        resolved = shutil.which(exe_name)
        if resolved:
            return resolved
    return None


def _normalize_windows_priority(raw: list[str] | None) -> list[str]:
    if not raw:
        return list(_WINDOWS_DEFAULT_SHELL_PRIORITY)
    seen: set[str] = set()
    normalized: list[str] = []
    for item in raw:
        name = _normalize_shell_name(str(item or ""))
        if not name or name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized or list(_WINDOWS_DEFAULT_SHELL_PRIORITY)


def _resolve_windows_shell_entry(entry: str) -> ShellInvocation | None:
    shell_name = _normalize_shell_name(entry)
    if shell_name in {"cmd", "cmd.exe"}:
        comspec = os.environ.get("COMSPEC", "").strip()
        if not comspec:
            comspec = shutil.which("cmd.exe") or "cmd.exe"
        return ShellInvocation(kind="cmd", prefix=[comspec, "/c"])

    if shell_name in {"powershell", "powershell.exe"}:
        powershell = _resolve_windows_powershell()
        if not powershell:
            return None
        return ShellInvocation(kind="powershell", prefix=[powershell, "-NoLogo", "-NoProfile", "-Command"])

    if shell_name in {"pwsh", "pwsh.exe"}:
        pwsh = shutil.which("pwsh.exe") or shutil.which("pwsh")
        if not pwsh:
            return None
        return ShellInvocation(kind="pwsh", prefix=[pwsh, "-NoLogo", "-NoProfile", "-Command"])

    if shell_name in {"git-bash", "gitbash"}:
        git_bash = _resolve_windows_git_bash()
        if not git_bash:
            return None
        return ShellInvocation(kind="bash", prefix=[git_bash, "-lc"])

    if shell_name in {"bash", "bash.exe"}:
        bash_path = shutil.which("bash.exe") or shutil.which("bash")
        if not bash_path:
            bash_path = _resolve_windows_git_bash()
        if not bash_path:
            return None
        return ShellInvocation(kind="bash", prefix=[bash_path, "-lc"])

    if not shell_name:
        return None

    if _looks_like_path(entry):
        candidate = Path(entry).expanduser()
        if candidate.is_file():
            inferred = _infer_shell_kind(candidate.name)
            if inferred == "bash":
                return ShellInvocation(kind="bash", prefix=[str(candidate), "-lc"])
            if inferred == "powershell":
                return ShellInvocation(
                    kind="powershell",
                    prefix=[str(candidate), "-NoLogo", "-NoProfile", "-Command"],
                )
            if inferred == "pwsh":
                return ShellInvocation(
                    kind="pwsh",
                    prefix=[str(candidate), "-NoLogo", "-NoProfile", "-Command"],
                )
            if inferred == "cmd":
                return ShellInvocation(kind="cmd", prefix=[str(candidate), "/c"])
        return None

    found = shutil.which(entry)
    if not found:
        return None
    inferred = _infer_shell_kind(Path(found).name)
    if inferred in {"powershell", "pwsh"}:
        return ShellInvocation(
            kind=inferred,
            prefix=[found, "-NoLogo", "-NoProfile", "-Command"],
        )
    if inferred == "cmd":
        return ShellInvocation(kind="cmd", prefix=[found, "/c"])
    return ShellInvocation(kind="bash", prefix=[found, "-lc"])


def _infer_shell_kind(executable_name: str) -> str:
    lowered = executable_name.strip().lower()
    if lowered in {"cmd", "cmd.exe"}:
        return "cmd"
    if lowered.startswith("pwsh"):
        return "pwsh"
    if "powershell" in lowered:
        return "powershell"
    if lowered in {"bash", "bash.exe", "sh", "zsh", "dash", "ksh", "fish"}:
        return "bash"
    return "bash"


def _resolve_posix_shell(shell: str | None) -> ShellInvocation:
    if not shell:
        return ShellInvocation(kind="bash", prefix=["bash", "-lc"])

    normalized = _normalize_shell_name(shell)
    if normalized == "auto":
        return ShellInvocation(kind="bash", prefix=["bash", "-lc"])

    if normalized in {"bash", "bash.exe"}:
        bash_path = shutil.which("bash") or "bash"
        return ShellInvocation(kind="bash", prefix=[bash_path, "-lc"])

    executable = shell
    if _looks_like_path(shell):
        executable = str(Path(shell).expanduser())
    else:
        resolved = shutil.which(shell)
        if resolved:
            executable = resolved

    inferred = _infer_shell_kind(Path(executable).name)
    if inferred in {"powershell", "pwsh"}:
        return ShellInvocation(
            kind=inferred,
            prefix=[executable, "-NoLogo", "-NoProfile", "-Command"],
        )
    if inferred == "cmd":
        return ShellInvocation(kind="cmd", prefix=[executable, "/c"])
    return ShellInvocation(kind="bash", prefix=[executable, "-lc"])


def resolve_shell_invocation(
    *,
    shell: str | None = None,
    windows_shell_priority: list[str] | None = None,
) -> ShellInvocation:
    selected_shell = str(shell or "").strip()
    if os.name != "nt":
        return _resolve_posix_shell(selected_shell or None)

    if selected_shell and _normalize_shell_name(selected_shell) != "auto":
        resolved = _resolve_windows_shell_entry(selected_shell)
        if resolved is None:
            raise ValueError(f"Configured shell is unavailable on Windows: {selected_shell}")
        return resolved

    for entry in _normalize_windows_priority(windows_shell_priority):
        resolved = _resolve_windows_shell_entry(entry)
        if resolved is not None:
            return resolved

    fallback = _resolve_windows_shell_entry("cmd")
    if fallback is None:
        raise ValueError("Unable to resolve Windows command shell.")
    return fallback


def build_shell_invocation(
    command: str,
    *,
    shell: str | None = None,
    windows_shell_priority: list[str] | None = None,
) -> list[str]:
    resolved = resolve_shell_invocation(
        shell=shell,
        windows_shell_priority=windows_shell_priority,
    )
    return [*resolved.prefix, command]


def prepare_shell_execution(
    command: str,
    *,
    auto_confirm: bool,
    stdin: str | None,
    shell: str | None = None,
    windows_shell_priority: list[str] | None = None,
) -> tuple[list[str], str | None]:
    resolved = resolve_shell_invocation(
        shell=shell,
        windows_shell_priority=windows_shell_priority,
    )

    if not auto_confirm:
        return [*resolved.prefix, command], stdin

    if resolved.kind != "bash":
        # cmd/powershell shells have no built-in equivalent to `yes`, so prefill stdin with confirmations.
        auto_confirm_stdin = "y\n" * _WINDOWS_AUTO_CONFIRM_LINES
        return [*resolved.prefix, command], f"{auto_confirm_stdin}{stdin or ''}"

    return [*resolved.prefix, f"yes | ({command})"], stdin
