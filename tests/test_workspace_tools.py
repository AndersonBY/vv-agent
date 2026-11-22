from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from vv_agent.runtime.local_environment import (
    LOCAL_ENVIRONMENT_CAPABILITIES_METADATA_KEY,
    LOCAL_ENVIRONMENT_METADATA_KEY,
    LocalEnvironment,
)
from vv_agent.runtime.processes import CapturedProcess, start_captured_process
from vv_agent.tools import ToolContext
from vv_agent.tools.handlers import bash as bash_handler
from vv_agent.tools.handlers import search as search_handler
from vv_agent.tools.handlers import workspace_io
from vv_agent.types import ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend


class _SearchHandle:
    def __init__(self, stdout: Any) -> None:
        self.stdout = stdout
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


@pytest.mark.parametrize("mode", ["default", "explicit", "frozen"])
def test_find_files_and_search_files_use_requested_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    def process_output(text: bool) -> _SearchHandle:
        if text:
            events = [
                {"type": "begin", "data": {"path": {"text": "hit.txt"}}},
                {
                    "type": "match",
                    "data": {
                        "path": {"text": "hit.txt"},
                        "line_number": 1,
                        "lines": {"text": "needle\n"},
                        "submatches": [{"match": {"text": "needle"}, "start": 0, "end": 6}],
                    },
                },
            ]
            return _SearchHandle(io.StringIO("".join(f"{json.dumps(event)}\n" for event in events)))
        return _SearchHandle(io.BytesIO(b"hit.txt\x00"))

    calls = []

    class TrackingEnvironment(LocalEnvironment):
        def start_search(self, command: list[str], *, cwd: Path, text: bool, capture_stderr: bool, **_: Any):
            assert cwd == tmp_path
            calls.append(("environment", text, capture_stderr))
            return process_output(text)

    def popen(command: list[str], *, cwd: Path, stdout: int, stderr: int, text: bool = False, **_: Any) -> _SearchHandle:
        assert cwd == tmp_path and stdout == subprocess.PIPE
        calls.append(("subprocess", text, stderr == subprocess.PIPE))
        return process_output(text)

    environment = TrackingEnvironment()
    metadata: dict[str, Any] = {}
    if mode != "default":
        metadata[LOCAL_ENVIRONMENT_METADATA_KEY] = environment
    if mode == "frozen":
        metadata[LOCAL_ENVIRONMENT_CAPABILITIES_METADATA_KEY] = environment.capabilities
    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
        metadata=metadata,
    )
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(workspace_io, "_resolve_rg_executable", lambda: "rg")
    monkeypatch.setattr(search_handler, "_resolve_rg_executable", lambda: "rg")

    found = workspace_io.find_files(context, {"path": "."})
    searched = search_handler.search_files(context, {"pattern": "needle", "output_mode": "files_with_matches"})

    assert json.loads(found.content)["files"] == ["hit.txt"]
    assert searched.metadata["files"] == ["hit.txt"]
    path = "subprocess" if mode == "default" else "environment"
    assert calls == [(path, False, True), (path, True, False)]


def test_bash_uses_explicit_process_starter_without_frozen_capabilities(monkeypatch, tmp_path):
    calls = []

    def start(command, **kwargs) -> CapturedProcess:
        calls.append(kwargs["cwd"])
        return start_captured_process(command, **kwargs)

    def direct_start(*_args, **_kwargs):
        pytest.fail("Bash bypassed the requested environment")

    monkeypatch.setattr(bash_handler, "start_captured_process", direct_start)
    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
        metadata={LOCAL_ENVIRONMENT_METADATA_KEY: LocalEnvironment(process_starter=start)},
    )
    result = bash_handler.run_bash_command(context, {"command": "echo adapter"})

    assert result.status_code is ToolResultStatus.SUCCESS
    assert result.content == "adapter\n"
    assert calls == [tmp_path]


def test_bound_search_and_discovery_fail_without_rg_instead_of_falling_back(monkeypatch, tmp_path):
    (tmp_path / "hit.txt").write_text("needle\n")
    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
        metadata={LOCAL_ENVIRONMENT_METADATA_KEY: LocalEnvironment()},
    )
    monkeypatch.setattr(workspace_io, "_resolve_rg_executable", lambda: None)
    monkeypatch.setattr(search_handler, "_resolve_rg_executable", lambda: None)

    found = workspace_io.find_files(context, {"path": "."})
    searched = search_handler.search_files(context, {"pattern": "needle"})

    assert found.error_code == searched.error_code == "local_environment_rejected"
