from __future__ import annotations

import subprocess
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

from vv_agent.runtime.processes import (
    kill_process_tree,
    process_tree_is_running,
    remove_captured_output,
    snapshot_captured_output,
    start_captured_process,
)
from vv_agent.runtime.shell import prepare_shell_execution
from vv_agent.types import ToolArtifactRef
from vv_agent.workspace.artifacts import (
    ArtifactPathInvalidError,
    BoundedTextPreview,
    bounded_captured_text_preview,
    persist_captured_text_artifact,
)

if TYPE_CHECKING:
    from vv_agent.workspace.base import WorkspaceBackend

_WATCH_POLL_INTERVAL_SECONDS = 0.05
_TERMINAL_STATUSES = frozenset({"completed", "failed", "timeout", "stopped"})
BackgroundSessionListener = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class _SessionState:
    session_id: str
    command: str
    shell: str | None
    cwd: str
    started_at: float
    timeout_seconds: int | None
    process: subprocess.Popen[str]
    output_path: Path
    owner_task_id: str
    owner_workspace: str
    status: str = "running"
    stop_reason: str | None = None
    preview: BoundedTextPreview | None = None
    exit_code: int | None = None
    observation_error: str | None = None
    output_error: str | None = None
    listeners: list[BackgroundSessionListener] = field(default_factory=list)
    artifact: ToolArtifactRef | None = None
    artifact_error: str | None = None
    artifact_error_code: str | None = None
    artifact_backend: WorkspaceBackend | None = None
    artifact_task_id: str = ""
    artifact_tool_call_id: str = ""
    lock: Any = field(default_factory=Lock)
    done: Event = field(default_factory=Event)


class BackgroundSessionManager:
    """Process-local handles. Tool access is checked before touching a session."""

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionState] = {}
        self._lock = Lock()

    def _register_session(
        self,
        *,
        command: str,
        cwd: Path,
        timeout_seconds: int | None,
        process: subprocess.Popen[str],
        output_path: Path,
        shell: str | None = None,
        started_at: float | None = None,
        owner_task_id: str = "",
        owner_workspace: Path | None = None,
        artifact_backend: WorkspaceBackend | None = None,
        artifact_task_id: str = "",
        artifact_tool_call_id: str = "",
    ) -> str:
        if timeout_seconds is not None and (type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 86400):
            raise ValueError("timeout_seconds must be an integer from 1 through 86400")
        session_id = f"bg_{uuid.uuid4().hex[:12]}"
        session = _SessionState(
            session_id=session_id,
            command=command,
            shell=shell,
            cwd=str(cwd),
            started_at=time.monotonic() if started_at is None else started_at,
            timeout_seconds=timeout_seconds,
            process=process,
            output_path=output_path,
            owner_task_id=owner_task_id,
            owner_workspace=str((owner_workspace or cwd).resolve()),
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
        timeout_seconds: int | None = None,
        stdin: str | None = None,
        auto_confirm: bool = False,
        shell: str | None = None,
        windows_shell_priority: list[str] | None = None,
        env: Mapping[str, str] | None = None,
        owner_task_id: str = "",
        owner_workspace: Path | None = None,
        artifact_backend: WorkspaceBackend | None = None,
        artifact_task_id: str = "",
        artifact_tool_call_id: str = "",
    ) -> str:
        if timeout_seconds is not None and (type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 86400):
            raise ValueError("timeout_seconds must be an integer from 1 through 86400")
        shell_command, prepared_stdin = prepare_shell_execution(
            command,
            auto_confirm=auto_confirm,
            stdin=stdin,
            shell=shell,
            windows_shell_priority=windows_shell_priority,
        )
        started = start_captured_process(shell_command, cwd=cwd, stdin_text=prepared_stdin, env=env)
        return self.adopt_running_process(
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            process=started.process,
            output_path=started.output_path,
            shell=shell,
            started_at=started.started_at,
            owner_task_id=owner_task_id,
            owner_workspace=owner_workspace,
            artifact_backend=artifact_backend,
            artifact_task_id=artifact_task_id,
            artifact_tool_call_id=artifact_tool_call_id,
        )

    def adopt_running_process(
        self,
        *,
        command: str,
        cwd: Path,
        timeout_seconds: int | None,
        process: subprocess.Popen[str],
        output_path: Path,
        shell: str | None = None,
        started_at: float | None = None,
        owner_task_id: str = "",
        owner_workspace: Path | None = None,
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
            owner_task_id=owner_task_id,
            owner_workspace=owner_workspace,
            artifact_backend=artifact_backend,
            artifact_task_id=artifact_task_id,
            artifact_tool_call_id=artifact_tool_call_id,
        )

    def _get(self, session_id: str) -> _SessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    @staticmethod
    def _missing(session_id: str) -> dict[str, Any]:
        return {"status": "missing", "session_id": session_id, "error": "Background session not found"}

    def wait(self, session_id: str, yield_time_ms: int) -> None:
        """Wait only for the initial yield; the watchdog owns the execution deadline."""
        session = self._get(session_id)
        if session is not None and yield_time_ms:
            deadline = session.started_at + yield_time_ms / 1000
            while not session.done.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                session.done.wait(timeout=remaining)

    def check(self, session_id: str) -> dict[str, Any]:
        """Trusted local observation; model tools must use check_for_tool."""
        session = self._get(session_id)
        if session is None:
            return self._missing(session_id)
        with session.lock:
            listeners = self._advance(session)
            observation = replace(session)
        # Freeze state under its lock, but do not hold up the watchdog while a
        # live prefix is copied or uploaded through a potentially remote backend.
        payload = self._snapshot(observation, live=True)
        self._notify_listeners(listeners, payload)
        return payload

    def check_for_tool(
        self,
        session_id: str,
        workspace_backend: WorkspaceBackend,
        task_id: str,
        tool_call_id: str,
        *,
        workspace: Path,
    ) -> dict[str, Any]:
        return self._access_for_tool(session_id, workspace_backend, task_id, tool_call_id, workspace, stop=False)

    def stop_for_tool(
        self,
        session_id: str,
        workspace_backend: WorkspaceBackend,
        task_id: str,
        tool_call_id: str,
        *,
        workspace: Path,
    ) -> dict[str, Any]:
        return self._access_for_tool(session_id, workspace_backend, task_id, tool_call_id, workspace, stop=True)

    def _access_for_tool(
        self,
        session_id: str,
        backend: WorkspaceBackend,
        task_id: str,
        tool_call_id: str,
        workspace: Path,
        *,
        stop: bool,
    ) -> dict[str, Any]:
        session = self._get(session_id)
        if session is None:
            return self._missing(session_id)
        # Artifact labels are deliberately not used for ownership.
        if session.owner_task_id != task_id or session.owner_workspace != str(workspace.resolve()):
            return {
                "status": "forbidden",
                "session_id": session_id,
                "error": "Background session belongs to another task or workspace",
                "error_code": "background_session_forbidden",
            }
        with session.lock:
            if session.artifact_backend is None:
                session.artifact_backend = backend
                session.artifact_task_id = task_id
                session.artifact_tool_call_id = tool_call_id
            listeners = self._advance(session)
            if stop and session.status not in _TERMINAL_STATUSES:
                listeners.extend(self._stop(session, "stopped"))
            observation = replace(session)
        payload = self._snapshot(observation, live=True)
        self._notify_listeners(listeners, payload)
        return payload

    def subscribe(self, session_id: str, listener: BackgroundSessionListener) -> Callable[[], None]:
        session = self._get(session_id)
        if session is None:
            return lambda: None
        with session.lock:
            snapshot = self._snapshot(session) if session.status in _TERMINAL_STATUSES else None
            if snapshot is None:
                session.listeners.append(listener)
        if snapshot is not None:
            listener(snapshot)

        def unsubscribe() -> None:
            with session.lock:
                if listener in session.listeners:
                    session.listeners.remove(listener)

        return unsubscribe

    def _start_watch_thread(self, session_id: str) -> None:
        Thread(target=self._watch_session, args=(session_id,), daemon=True, name=f"vv-agent-bg-{session_id}").start()

    def _watch_session(self, session_id: str) -> None:
        session = self._get(session_id)
        if session is None:
            return
        while True:
            with session.lock:
                listeners = self._advance(session)
                terminal = session.status in _TERMINAL_STATUSES
                payload = self._snapshot(session) if terminal else {}
            self._notify_listeners(listeners, payload)
            if terminal:
                return
            session.done.wait(_WATCH_POLL_INTERVAL_SECONDS)

    def _advance(self, session: _SessionState) -> list[BackgroundSessionListener]:
        if session.status in _TERMINAL_STATUSES:
            self._ensure_terminal_output(session)
            return []
        try:
            exit_code = session.process.poll()
            tree_running = process_tree_is_running(session.process)
            if exit_code is not None and tree_running is False:
                return self._finish(session, exit_code)
            session.status = "stopping" if session.stop_reason else ("unknown" if tree_running is None else "running")
            session.observation_error = None
        except OSError as exc:
            session.status = "unknown"
            session.observation_error = str(exc)
        if session.timeout_seconds is not None and time.monotonic() >= session.started_at + session.timeout_seconds:
            return self._stop(session, "timeout")
        return []

    def _stop(self, session: _SessionState, reason: str) -> list[BackgroundSessionListener]:
        if session.status in _TERMINAL_STATUSES:
            return []
        session.stop_reason = session.stop_reason or reason
        session.status = "stopping"
        try:
            confirmed = kill_process_tree(session.process)
            exit_code = session.process.poll()
            if confirmed and exit_code is not None:
                return self._finish(session, exit_code)
        except OSError as exc:
            session.status = "unknown"
            session.observation_error = str(exc)
        return []

    def _finish(self, session: _SessionState, exit_code: int) -> list[BackgroundSessionListener]:
        session.exit_code = exit_code
        session.status = session.stop_reason or ("completed" if exit_code == 0 else "failed")
        self._ensure_terminal_output(session)
        listeners, session.listeners = session.listeners, []
        session.done.set()
        return listeners

    @staticmethod
    def _ensure_terminal_output(session: _SessionState) -> None:
        if session.status not in _TERMINAL_STATUSES:
            return
        if session.preview is None:
            try:
                session.preview = bounded_captured_text_preview(session.output_path)
                session.output_error = None
            except OSError as exc:
                session.output_error = str(exc)
                return
            if not session.preview.truncated:
                remove_captured_output(session.output_path)
        if not session.preview.truncated:
            return
        if session.artifact is not None or session.artifact_backend is None:
            return
        try:
            session.artifact = persist_captured_text_artifact(
                session.artifact_backend,
                session.artifact_task_id,
                session.artifact_tool_call_id,
                session.output_path,
            )
            session.artifact_error = session.artifact_error_code = None
            remove_captured_output(session.output_path)
        except Exception as exc:
            session.artifact_error = str(exc)
            session.artifact_error_code = (
                "artifact_path_invalid" if isinstance(exc, ArtifactPathInvalidError) else "artifact_persist_failed"
            )

    @staticmethod
    def _notify_listeners(listeners: list[BackgroundSessionListener], payload: dict[str, Any]) -> None:
        for listener in listeners:
            try:
                listener(dict(payload))
            except Exception:
                continue

    def _snapshot(self, session: _SessionState, *, live: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": session.status,
            "session_id": session.session_id,
            "command": session.command,
        }
        if session.shell is not None:
            payload["shell"] = session.shell
        if session.exit_code is not None:
            payload["exit_code"] = session.exit_code
        else:
            payload["elapsed_seconds"] = round(time.monotonic() - session.started_at, 2)
        if session.observation_error:
            payload["observation_error"] = session.observation_error
        if session.output_error:
            payload["output_error"] = session.output_error
        if session.status in _TERMINAL_STATUSES:
            self._add_preview(payload, session.preview)
            if session.artifact is not None:
                payload["artifact"] = session.artifact.to_dict()
            if session.artifact_error is not None:
                payload["artifact_error"] = session.artifact_error
                payload["artifact_error_code"] = session.artifact_error_code
        elif live:
            try:
                with snapshot_captured_output(session.output_path) as snapshot:
                    preview = bounded_captured_text_preview(snapshot)
                    self._add_preview(payload, preview)
                    if preview.truncated and session.artifact_backend is not None:
                        try:
                            payload["artifact"] = persist_captured_text_artifact(
                                session.artifact_backend,
                                session.artifact_task_id,
                                session.artifact_tool_call_id,
                                snapshot,
                            ).to_dict()
                        except Exception as exc:
                            payload.update(
                                artifact_error=str(exc),
                                artifact_error_code="artifact_path_invalid"
                                if isinstance(exc, ArtifactPathInvalidError)
                                else "artifact_persist_failed",
                            )
            except Exception as exc:
                payload.update(output_error=str(exc))
        return payload

    @staticmethod
    def _add_preview(payload: dict[str, Any], preview: BoundedTextPreview | None) -> None:
        if preview is None:
            payload["output"] = ""
            return
        payload.update(output=preview.content, output_truncated=preview.truncated, output_json_bytes=preview.json_bytes)
        if preview.truncated:
            payload.update(output_original_bytes=preview.original_bytes, output_visible_bytes=preview.visible_bytes)


background_session_manager = BackgroundSessionManager()
