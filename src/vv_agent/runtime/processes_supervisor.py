"""Linux per-command subreaper, executed in a fresh stdlib-only interpreter.

The application never becomes a subreaper and never waits for unrelated children.
The control descriptor is closed in the command. EOF therefore also lets this
supervisor clean up its own tree when its owning application disappears.
"""

from __future__ import annotations

import ctypes
import errno
import os
import select
import signal
import socket
import struct
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import NoReturn


def _signal_children(root_pid: int, root_reaped: bool, sig: int) -> None:
    # Do not reap between enumeration and signaling: a direct child's PID cannot
    # be reused until this sole parent reaps it. An exited intermediate parent
    # reparents even setsid/double-fork descendants to this supervisor.
    children = Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()
    if not root_reaped:
        with suppress(ProcessLookupError):
            os.killpg(root_pid, sig)
    for child in children:
        with suppress(ProcessLookupError):
            os.kill(int(child), sig)


def _exit_like(status: int) -> NoReturn:
    code = os.waitstatus_to_exitcode(status)
    if code < 0:
        sig = -code
        if sig not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(sig, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {sig})
        os.kill(os.getpid(), sig)
        # A signal exit must never be converted into a successful exit.
        os._exit(255)
    os._exit(code)


def main() -> None:
    control = socket.socket(fileno=int(sys.argv[1]))
    control.set_inheritable(False)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "cannot establish command subreaper")
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        started_at = time.monotonic()
        command = subprocess.Popen(sys.argv[2:], start_new_session=True)
    except OSError as exc:
        control.sendall(b"E" + struct.pack("!i", exc.errno or errno.EIO))
        os._exit(127)
    connected = True
    stop_signal: int | None = None
    try:
        control.sendall(b"R" + struct.pack("!d", started_at))
    except OSError:
        # The owner can disappear after the command fork but before readiness.
        # Still enter the reaper loop; exiting here would orphan its command.
        connected = False
        stop_signal = signal.SIGKILL
    # The command inherited the original stdin and capture. The supervisor does
    # not retain them, consume stdin, or add text to the command's output.
    for fd in (0, 1, 2):
        os.close(fd)

    root_status: int | None = None
    while True:
        if stop_signal is not None:
            # Lack of signaling evidence does not establish a stopped tree.
            with suppress(OSError):
                _signal_children(command.pid, root_status is not None, stop_signal)
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                if root_status is None:
                    os._exit(255)  # No completion proof if the parent status was lost.
                with suppress(OSError):
                    control.sendall(b"D")
                _exit_like(root_status)
            if pid == 0:
                break
            if pid == command.pid:
                root_status = status
        if connected:
            ready, _, _ = select.select([control], [], [], 0.01)
            if ready:
                try:
                    requests = control.recv(1024)
                except OSError:
                    requests = b""
                if not requests:
                    connected = False
                    stop_signal = signal.SIGKILL
                elif b"K" in requests:
                    stop_signal = signal.SIGKILL
                elif b"T" in requests and stop_signal != signal.SIGKILL:
                    stop_signal = signal.SIGTERM
        else:
            time.sleep(0.01)


if __name__ == "__main__":
    main()
