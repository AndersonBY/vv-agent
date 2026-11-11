"""Pipe ownership must not add an unbounded wait to bounded reap failures."""

import io
import subprocess
from types import SimpleNamespace

import pytest

from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers import search, workspace_io
from vv_agent.workspace import LocalWorkspaceBackend


def invoke(handler, tmp_path):
    context = ToolContext(tmp_path, {}, 0, LocalWorkspaceBackend(tmp_path))
    if handler == "search":
        return search._search_files_local_rg(
            context,
            path=".",
            glob_pattern="**/*",
            pattern="needle",
            output_mode="content",
            file_type=None,
            case_insensitive=False,
            multiline_mode=False,
            context_before=0,
            context_after=0,
            include_hidden=False,
            include_ignored=True,
            include_sensitive=False,
            literal=False,
        )
    return workspace_io._find_files_local_rg(
        context=context,
        base_path=tmp_path,
        base_is_workspace_root=True,
        glob_pattern="**/*",
        include_hidden=False,
        include_ignored=True,
        ignored_root_names=[],
        max_results=10,
        scan_limit=100,
        glob_regex=SimpleNamespace(match=lambda path: True),
    )


class ReapTimeout:
    # Use the real stdlib context protocol so the before test detects wait(None).
    __enter__ = subprocess.Popen.__enter__
    __exit__ = subprocess.Popen.__exit__

    def __init__(self, handler):
        self.stdout = io.StringIO("") if handler == "search" else io.BytesIO(b"")
        self.stderr = io.BytesIO(b"") if handler == "workspace" else None
        self.stdin = None
        self.returncode = None
        self.calls = []

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if timeout is not None:
            raise subprocess.TimeoutExpired("controlled-reap", timeout)
        self.returncode = -9
        return self.returncode

    def kill(self):
        self.calls.append(("kill",))


@pytest.mark.parametrize("handler", ["search", "workspace"])
def test_second_reap_timeout_closes_pipes_without_another_wait(handler, tmp_path, monkeypatch):
    process = ReapTimeout(handler)
    module = search if handler == "search" else workspace_io
    monkeypatch.setattr(module, "_resolve_rg_executable", lambda: "controlled-reap")
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        invoke(handler, tmp_path)
    assert caught.value.timeout == 1
    print("PIPE_REAP_CALLS", handler, process.calls, flush=True)
    assert process.calls == [("wait", 1), ("kill",), ("wait", 1)]
    assert process.stdout.closed
    assert process.stderr is None or process.stderr.closed


@pytest.mark.parametrize("handler", ["search", "workspace"])
def test_real_rg_pipes_are_closed_by_the_handler(handler, tmp_path, monkeypatch):
    (tmp_path / "one.txt").write_text("needle\n")
    processes = []
    original = subprocess.Popen

    def record(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", record)
    result = invoke(handler, tmp_path)
    assert result is not None and len(processes) == 1
    process = processes[0]
    assert process.returncode == 0 and process.stdout.closed
    assert process.stderr is None or process.stderr.closed
