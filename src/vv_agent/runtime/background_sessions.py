from __future__ import annotations

import subprocess
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any

from vv_agent.runtime.processes import (
    kill_process_tree,
    remove_captured_output,
    start_captured_process,
)
from vv_agent.runtime.shell import prepare_shell_execution
from vv_agent.types import ToolArtifactRef
from vv_agent.workspace.artifacts import (
    ArtifactPathInvalidError,
    BoundedTextPreview,
    bounded_captured_text_preview,
    bounded_text_preview,
    persist_captured_text_artifact,
)

if TYPE_CHECKING:
    from vv_agent.workspace.base import WorkspaceBackend

_WATCH_POLL_INTERVAL_SECONDS = 0.2

BackgroundSessionListener = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class _SessionState:
    session_id: str
    command: str
    shell: str | None
    cwd: str
    started_at: float
    timeout_seconds: int
    process: subprocess.Popen[str]
    output_path: Path
    status: str = "running"
    preview: BoundedTextPreview | None = None
    exit_code: int | None = None
    listeners: list[BackgroundSessionListener] | None = None
    artifact: ToolArtifactRef | None = None
    artifact_error: str | None = None
    artifact_error_code: str | None = None
    artifact_backend: WorkspaceBackend | None = None
    artifact_task_id: str = ""
    artifact_tool_call_id: str = ""

    def __post_init__(self) -> None:
        if self.listeners is None:
            self.listeners = []


class BackgroundSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, _SessionState] = {}
        self._lock = Lock()

    def _register_session(
        self,
        *,
        command: str,
        cwd: Path,
        timeout_seconds: int,
        process: subprocess.Popen[str],
        output_path: Path,
        shell: str | None = None,
        started_at: float | None = None,
        artifact_backend: WorkspaceBackend | None = None,
        artifact_task_id: str = "",
        artifact_tool_call_id: str = "",
    ) -> str:
        session_id = f"bg_{uuid.uuid4().hex[:12]}"
        session = _SessionState(
            session_id=session_id,
            command=command,
            shell=shell,
            cwd=str(cwd),
            started_at=time.time() if started_at is None else float(started_at),
            timeout_seconds=max(1, int(timeout_seconds)),
            process=process,
            output_path=output_path,
            artifact_backend=artifact_backend,
            artifact_task_id=artifact_task_id,
            artifact_tool_call_id=artifact_tool_call_id,
        )

        with self._lock:
            self._sessions[session_id] = session
        self._start_watch_thread(session_id)
        return session_id

    def start(
        self,
        *,
        command: str,
        cwd: Path,
        timeout_seconds: int,
        stdin: str | None = None,
        auto_confirm: bool = False,
        shell: str | None = None,
        windows_shell_priority: list[str] | None = None,
        env: Mapping[str, str] | None = None,
        artifact_backend: WorkspaceBackend | None = None,
        artifact_task_id: str = "",
        artifact_tool_call_id: str = "",
    ) -> str:
        shell_command, prepared_stdin = prepare_shell_execution(
            command,
            auto_confirm=auto_confirm,
            stdin=stdin,
            shell=shell,
            windows_shell_priority=windows_shell_priority,
        )

        started_process = start_captured_process(
            shell_command,
            cwd=cwd,
            stdin_text=prepared_stdin,
            env=env,
        )

        return self._register_session(
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            process=started_process.process,
            output_path=started_process.output_path,
            shell=shell,
            artifact_backend=artifact_backend,
            artifact_task_id=artifact_task_id,
            artifact_tool_call_id=artifact_tool_call_id,
        )

    def adopt_running_process(
        self,
        *,
        command: str,
        cwd: Path,
        timeout_seconds: int,
        process: subprocess.Popen[str],
        output_path: Path,
        shell: str | None = None,
        started_at: float | None = None,
        artifact_backend: WorkspaceBackend | None = None,
        artifact_task_id: str = "",
        artifact_tool_call_id: str = "",
    ) -> str:
        return self._register_session(
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            process=process,
            output_path=output_path,
            shell=shell,
            started_at=started_at,
            artifact_backend=artifact_backend,
            artifact_task_id=artifact_task_id,
            artifact_tool_call_id=artifact_tool_call_id,
        )

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
                payload, listeners = self._finalize_timeout(session_id)
                self._notify_listeners(listeners, payload)
                return payload

            return {
                "status": "running",
                "session_id": session.session_id,
                "command": session.command,
                "elapsed_seconds": round(elapsed, 2),
                "shell": session.shell,
            }

        payload, listeners = self._finalize_completed(session_id)
        self._notify_listeners(listeners, payload)
        return payload

    def check_for_tool(
        self,
        session_id: str,
        workspace_backend: WorkspaceBackend,
        task_id: str,
        tool_call_id: str,
    ) -> dict[str, Any]:
        payload = self.check(session_id)
        if payload.get("status") not in {"completed", "failed", "timeout"}:
            return payload

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return payload
            self._ensure_artifact(
                session,
                fallback_backend=workspace_backend,
                fallback_task_id=task_id,
                fallback_tool_call_id=tool_call_id,
            )
            return self._snapshot(session)

    def subscribe(
        self,
        session_id: str,
        listener: BackgroundSessionListener,
    ) -> Callable[[], None]:
        snapshot: dict[str, Any] | None = None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return lambda: None
            if session.status in {"completed", "failed", "timeout"}:
                snapshot = self._snapshot(session)
            else:
                assert session.listeners is not None
                session.listeners.append(listener)

        if snapshot is not None:
            listener(snapshot)
            return lambda: None

        def _unsubscribe() -> None:
            with self._lock:
                current = self._sessions.get(session_id)
                if current is None or not current.listeners:
                    return
                if listener in current.listeners:
                    current.listeners.remove(listener)

        return _unsubscribe

    def _start_watch_thread(self, session_id: str) -> None:
        thread = Thread(
            target=self._watch_session,
            args=(session_id,),
            daemon=True,
            name=f"vv-agent-bg-{session_id}",
        )
        thread.start()

    def _watch_session(self, session_id: str) -> None:
        while True:
            with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    return
                if session.status in {"completed", "failed", "timeout"}:
                    return
                process = session.process
                elapsed = time.time() - session.started_at
                timeout_seconds = float(session.timeout_seconds)

            if process.poll() is None:
                if elapsed > timeout_seconds:
                    payload, listeners = self._finalize_timeout(session_id)
                    self._notify_listeners(listeners, payload)
                    return
                if not hasattr(process, "wait"):
                    time.sleep(_WATCH_POLL_INTERVAL_SECONDS)
                    continue
                try:
                    process.wait(timeout=_WATCH_POLL_INTERVAL_SECONDS)
                except subprocess.TimeoutExpired:
                    continue

            payload, listeners = self._finalize_completed(session_id)
            self._notify_listeners(listeners, payload)
            return

    def _finalize_completed(self, session_id: str) -> tuple[dict[str, Any], list[Any]]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                payload = {
                    "status": "missing",
                    "session_id": session_id,
                    "error": "Background session not found",
                }
                return payload, []
            if session.status in {"completed", "failed", "timeout"}:
                return self._snapshot(session), []

            process_returncode = getattr(session.process, "returncode", None)
            session.exit_code = process_returncode if process_returncode is not None else 0
            session.status = "completed" if session.exit_code == 0 else "failed"
            self._capture_terminal_preview(session, fallback="")
            self._ensure_artifact(
                session,
                fallback_backend=None,
                fallback_task_id="",
                fallback_tool_call_id="",
            )
            payload = self._snapshot(session)
            listeners = list(session.listeners or [])
            session.listeners = []

        return payload, listeners

    def _finalize_timeout(self, session_id: str) -> tuple[dict[str, Any], list[Any]]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                payload = {
                    "status": "missing",
                    "session_id": session_id,
                    "error": "Background session not found",
                }
                return payload, []
            if session.status in {"completed", "failed", "timeout"}:
                return self._snapshot(session), []
            process = session.process

        kill_process_tree(process)

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                payload = {
                    "status": "missing",
                    "session_id": session_id,
                    "error": "Background session not found",
                }
                return payload, []
            if session.status in {"completed", "failed", "timeout"}:
                return self._snapshot(session), []

            session.status = "timeout"
            process_returncode = getattr(session.process, "returncode", None)
            session.exit_code = process_returncode if process_returncode is not None else -9
            self._capture_terminal_preview(session, fallback="Command timed out in background session")
            self._ensure_artifact(
                session,
                fallback_backend=None,
                fallback_task_id="",
                fallback_tool_call_id="",
            )
            payload = self._snapshot(session)
            listeners = list(session.listeners or [])
            session.listeners = []

        return payload, listeners

    @staticmethod
    def _capture_terminal_preview(session: _SessionState, *, fallback: str) -> None:
        try:
            preview = bounded_captured_text_preview(session.output_path)
        except OSError:
            preview = bounded_text_preview("")
        if not preview.content and fallback:
            preview = bounded_text_preview(fallback)
        session.preview = preview
        if not preview.truncated:
            remove_captured_output(session.output_path)

    @staticmethod
    def _ensure_artifact(
        session: _SessionState,
        *,
        fallback_backend: WorkspaceBackend | None,
        fallback_task_id: str,
        fallback_tool_call_id: str,
    ) -> None:
        preview = session.preview or bounded_text_preview("")
        if session.status not in {"completed", "failed", "timeout"} or not preview.truncated:
            return
        if session.artifact is not None:
            return

        backend = session.artifact_backend or fallback_backend
        if backend is None:
            return
        task_id = session.artifact_task_id.strip() or fallback_task_id
        tool_call_id = session.artifact_tool_call_id.strip() or fallback_tool_call_id
        try:
            session.artifact = persist_captured_text_artifact(
                backend,
                task_id,
                tool_call_id,
                session.output_path,
            )
        except Exception as exc:
            session.artifact_error = str(exc)
            session.artifact_error_code = (
                "artifact_path_invalid" if isinstance(exc, ArtifactPathInvalidError) else "artifact_persist_failed"
            )
            return
        session.artifact_error = None
        session.artifact_error_code = None
        remove_captured_output(session.output_path)

    @staticmethod
    def _notify_listeners(
        listeners: list[BackgroundSessionListener],
        payload: dict[str, Any],
    ) -> None:
        for listener in listeners:
            try:
                listener(payload)
            except Exception:
                continue

    @staticmethod
    def _snapshot(session: _SessionState) -> dict[str, Any]:
        preview = session.preview or bounded_text_preview("")
        payload: dict[str, Any] = {
            "status": session.status,
            "session_id": session.session_id,
            "command": session.command,
            "shell": session.shell,
            "exit_code": session.exit_code,
            "output": preview.content,
            "output_truncated": preview.truncated,
        }
        if preview.truncated:
            payload["output_original_bytes"] = preview.original_bytes
            payload["output_visible_bytes"] = preview.visible_bytes
        if session.artifact is not None:
            payload["artifact"] = session.artifact.to_dict()
        if session.artifact_error is not None:
            payload["artifact_error"] = session.artifact_error
            payload["artifact_error_code"] = session.artifact_error_code or "artifact_persist_failed"
        return payload


background_session_manager = BackgroundSessionManager()
