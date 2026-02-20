from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

_OUTPUT_LIMIT = 50_000


@dataclass(slots=True)
class _SessionState:
    session_id: str
    command: str
    cwd: str
    started_at: float
    timeout_seconds: int
    process: subprocess.Popen[str]
    status: str = "running"
    output: str = ""
    exit_code: int | None = None


class BackgroundSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, _SessionState] = {}
        self._lock = Lock()

    def start(
        self,
        *,
        command: str,
        cwd: Path,
        timeout_seconds: int,
        stdin: str | None = None,
        auto_confirm: bool = False,
    ) -> str:
        wrapped_command = command
        if auto_confirm:
            wrapped_command = f"yes | ({command})"

        process = subprocess.Popen(
            ["bash", "-lc", wrapped_command],
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        if stdin and process.stdin is not None:
            process.stdin.write(stdin)
            process.stdin.close()
        elif process.stdin is not None:
            process.stdin.close()
        process.stdin = None

        session_id = f"bg_{uuid.uuid4().hex[:12]}"
        session = _SessionState(
            session_id=session_id,
            command=command,
            cwd=str(cwd),
            started_at=time.time(),
            timeout_seconds=timeout_seconds,
            process=process,
        )

        with self._lock:
            self._sessions[session_id] = session
        return session_id

    def check(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)

        if session is None:
            return {
                "status": "missing",
                "session_id": session_id,
                "error": "Background session not found",
            }

        if session.status in {"completed", "failed", "timeout"}:
            return self._snapshot(session)

        if session.process.poll() is None:
            elapsed = time.time() - session.started_at
            if elapsed > session.timeout_seconds:
                session.process.kill()
                session.status = "timeout"
                session.exit_code = -9
                session.output = "Command timed out in background session"
                return self._snapshot(session)

            return {
                "status": "running",
                "session_id": session.session_id,
                "command": session.command,
                "elapsed_seconds": round(elapsed, 2),
            }

        output, _ = session.process.communicate(timeout=1)
        session.output = (output or "")[:_OUTPUT_LIMIT]
        session.exit_code = session.process.returncode
        session.status = "completed" if session.exit_code == 0 else "failed"
        return self._snapshot(session)

    @staticmethod
    def _snapshot(session: _SessionState) -> dict[str, Any]:
        return {
            "status": session.status,
            "session_id": session.session_id,
            "command": session.command,
            "exit_code": session.exit_code,
            "output": session.output,
        }


background_session_manager = BackgroundSessionManager()
