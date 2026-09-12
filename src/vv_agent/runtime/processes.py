from __future__ import annotations

import codecs
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_SUPERVISOR_START_TIMEOUT_SECONDS = 5.0


@dataclass(slots=True, frozen=True)
class CapturedProcess:
    process: subprocess.Popen[str]
    output_path: Path
    started_at: float


class _LinuxProcessTree:
    """Private control and completion proof for one isolated supervisor."""

    def __init__(self, control: socket.socket, *, pending_start: bytes | None = None) -> None:
        self.control = control
        self.complete = False
        self.pending_start = None if pending_start is None else bytearray(pending_start)

    def confirmed(self) -> bool:
        if self.complete:
            return True
        if self.pending_start is not None:
            # A bounded startup wait can end in the middle of the ready packet.
            # Preserve those bytes so its suffix cannot masquerade as proof.
            while len(self.pending_start) < 9:
                try:
                    chunk = self.control.recv(9 - len(self.pending_start))
                except BlockingIOError:
                    return False
                if not chunk:
                    raise OSError("command supervisor exited before completing its startup handshake")
                self.pending_start.extend(chunk)
                if self.pending_start[:1] != b"R":
                    raise OSError("command supervisor did not confirm command startup")
            self.pending_start = None
        try:
            proof = self.control.recv(1)
        except BlockingIOError:
            return False
        if proof != b"D":
            raise OSError("command supervisor exited without confirming all descendants")
        self.complete = True
        self.control.close()
        return True

    def request_stop(self, *, force: bool) -> None:
        if not self.complete:
            self.control.send(b"K" if force else b"T")

    def __del__(self) -> None:
        self.control.close()


def _start_linux_supervisor(command: list[str], kwargs: dict[str, Any]) -> tuple[subprocess.Popen[str], float]:
    import fcntl

    def above_stdio(value: socket.socket) -> socket.socket:
        if value.fileno() > 2:
            return value
        fd = fcntl.fcntl(value.fileno(), fcntl.F_DUPFD_CLOEXEC, 3)
        replacement = socket.socket(fileno=fd)
        value.close()
        return replacement

    control, child_control = socket.socketpair()
    try:
        # Command's stdio setup may replace 0/1/2 in an embedding application
        # that closed its standard descriptors. Keep control outside that range.
        control = above_stdio(control)
        child_control = above_stdio(child_control)
        started_at = time.monotonic()
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(Path(__file__).with_name("processes_supervisor.py")),
                str(child_control.fileno()),
                *command,
            ],
            pass_fds=(child_control.fileno(),),
            **kwargs,
        )
        child_control.close()
        deadline = time.monotonic() + _SUPERVISOR_START_TIMEOUT_SECONDS
        packet = bytearray()
        try:
            expected_size = 1
            while len(packet) < expected_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("command supervisor startup handshake timed out")
                control.settimeout(remaining)
                chunk = control.recv(expected_size - len(packet))
                if not chunk:
                    raise OSError("command supervisor failed to start")
                packet.extend(chunk)
                if packet[:1] not in (b"R", b"E"):
                    raise OSError("invalid command supervisor startup handshake")
                expected_size = 9 if packet[:1] == b"R" else 5
        except OSError:
            # The command may already exist. Keep an owner-queryable handle and
            # its capture, request cleanup, and require the usual tree proof.
            control.setblocking(False)
            tree = _LinuxProcessTree(control, pending_start=bytes(packet))
            cast(Any, process)._vv_process_tree = tree
            with suppress(OSError):
                tree.request_stop(force=True)
            return process, started_at
        if packet[:1] == b"E":
            # E is sent only when no original command was started. This helper
            # can therefore be reaped directly without orphaning command code.
            code = struct.unpack("!i", packet[1:])[0]
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
            raise OSError(code, os.strerror(code))
        control.setblocking(False)
        cast(Any, process)._vv_process_tree = _LinuxProcessTree(control)
        return process, struct.unpack("!d", packet[1:])[0]
    except BaseException:
        control.close()
        raise
    finally:
        child_control.close()


def _build_windows_hidden_startupinfo() -> Any | None:
    startupinfo_factory = getattr(subprocess, "STARTUPINFO", None)
    if startupinfo_factory is None:
        return None
    startupinfo = startupinfo_factory()
    startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
    startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    return startupinfo


def _build_captured_process_platform_kwargs() -> dict[str, Any]:
    if os.name != "nt":
        return {"start_new_session": True}

    kwargs: dict[str, Any] = {}
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if creationflags:
        kwargs["creationflags"] = creationflags

    startupinfo = _build_windows_hidden_startupinfo()
    if startupinfo is not None:
        kwargs["startupinfo"] = startupinfo
    return kwargs


def start_captured_process(
    command: list[str],
    *,
    cwd: Path,
    stdin_text: str | None = None,
    env: Mapping[str, str] | None = None,
) -> CapturedProcess:
    output_path: Path | None = None
    try:
        with ExitStack() as stack:
            output_handle = stack.enter_context(
                tempfile.NamedTemporaryFile(
                    mode="w+",
                    encoding="utf-8",
                    errors="replace",
                    delete=False,
                    prefix="vv_agent_process_",
                    suffix=".log",
                )
            )
            output_path = Path(output_handle.name)
            stdin_target: Any = subprocess.DEVNULL
            if stdin_text is not None:
                # A child that does not read stdin must not block yield or its deadline.
                stdin_target = stack.enter_context(tempfile.TemporaryFile(mode="w+", encoding="utf-8"))
                stdin_target.write(stdin_text)
                stdin_target.seek(0)
            started_at = time.monotonic()
            kwargs: dict[str, Any] = {
                "cwd": str(cwd),
                "stdin": stdin_target,
                "stdout": output_handle,
                "stderr": subprocess.STDOUT,
                "text": True,
                "errors": "replace",
                "env": dict(env) if env is not None else None,
                **_build_captured_process_platform_kwargs(),
            }
            if sys.platform == "linux" and os.name == "posix":
                process, started_at = _start_linux_supervisor(command, kwargs)
            else:
                process = subprocess.Popen(command, **kwargs)
    except Exception:
        if output_path is not None:
            with suppress(Exception):
                output_path.unlink(missing_ok=True)
        raise

    return CapturedProcess(process=process, output_path=output_path, started_at=started_at)


@contextmanager
def snapshot_captured_output(path: Path) -> Iterator[Path]:
    """Freeze a growing capture's current UTF-8 prefix.

    An incomplete final code point stays in the original capture for the next
    read. Read at most the size observed at entry even if the child keeps writing.
    """
    snapshot_path: Path | None = None
    try:
        with (
            path.open("rb") as source,
            tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", delete=False, prefix="vv_agent_snapshot_", suffix=".log"
            ) as output,
        ):
            snapshot_path = Path(output.name)
            remaining = os.fstat(source.fileno()).st_size
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            while remaining:
                chunk = source.read(min(remaining, 65536))
                if not chunk:
                    break
                remaining -= len(chunk)
                text = decoder.decode(chunk, final=False)
                output.write(text)
        yield snapshot_path
    finally:
        if snapshot_path is not None:
            remove_captured_output(snapshot_path)


def read_captured_output(path: Path, *, limit_chars: int) -> str:
    if limit_chars <= 0 or not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit_chars)
    except Exception:
        return ""


def remove_captured_output(path: Path) -> None:
    with suppress(Exception):
        path.unlink(missing_ok=True)


def process_tree_is_running(process: subprocess.Popen[str]) -> bool | None:
    """Return None when the OS cannot establish whether the managed tree runs."""
    tree = getattr(process, "_vv_process_tree", None)
    if isinstance(tree, _LinuxProcessTree):
        if process.poll() is None:
            return None if tree.pending_start is not None else True
        try:
            return not tree.confirmed()
        except OSError:
            return None
    # A parent/group snapshot cannot exclude descendants that already detached
    # and lost an intermediate parent. Retain the handle and capture.
    return True if process.poll() is None else None


def _windows_tree_pids(root_pid: int) -> set[int]:
    if sys.platform != "win32":
        raise OSError("Windows process enumeration is unavailable on this platform")
    import ctypes
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("size", wintypes.DWORD),
            ("usage", wintypes.DWORD),
            ("pid", wintypes.DWORD),
            ("heap", ctypes.c_size_t),
            ("module", wintypes.DWORD),
            ("threads", wintypes.DWORD),
            ("parent", wintypes.DWORD),
            ("priority", wintypes.LONG),
            ("flags", wintypes.DWORD),
            ("exe", wintypes.WCHAR * 260),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    parents: dict[int, int] = {}
    try:
        entry = ProcessEntry()
        entry.size = ctypes.sizeof(entry)
        found = kernel.Process32FirstW(handle, ctypes.byref(entry))
        while found:
            parents[int(entry.pid)] = int(entry.parent)
            found = kernel.Process32NextW(handle, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)
    family = {root_pid}
    while True:
        descendants = {pid for pid, parent in parents.items() if parent in family}
        if descendants.issubset(family):
            return family.intersection(parents)
        family.update(descendants)


def kill_process_tree(process: subprocess.Popen[str], *, wait_seconds: float = 0.5) -> bool:
    """Attempt bounded termination, confirming the whole tree before success."""
    normalized_wait = max(float(wait_seconds), 0.1)

    def stopped() -> bool:
        return process.poll() is not None and process_tree_is_running(process) is False

    def wait_stopped() -> bool:
        deadline = time.monotonic() + normalized_wait
        while True:
            if stopped():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    tree = getattr(process, "_vv_process_tree", None)
    if isinstance(tree, _LinuxProcessTree):
        for force in (False, True):
            if stopped():
                return True
            try:
                tree.request_stop(force=force)
            except OSError:
                return wait_stopped()
            if wait_stopped():
                return True
        return False

    # An untracked, reaped PID may already name another process. The original
    # group/snapshot is not a stable identity for descendants after that point.
    if process.poll() is not None:
        return False

    if os.name == "nt":
        try:
            pids = _windows_tree_pids(process.pid)
            if pids:
                command = ["taskkill"]
                for pid in sorted(pids):
                    command.extend(["/PID", str(pid)])
                subprocess.run(
                    [*command, "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=max(normalized_wait, 1.0),
                    check=False,
                )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return wait_stopped()

    for sig in (signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return False
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return wait_stopped()
        except OSError:
            return False
        if wait_stopped():
            return True
    return False
