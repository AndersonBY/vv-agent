"""Regression cases from the artifact reread review."""

import hashlib
import json
import subprocess
import sys
import time

import pytest

from vv_agent import is_task_artifact_path
from vv_agent.memory import MemoryManager
from vv_agent.tools.base import ToolContext
from vv_agent.tools.builtins import build_default_registry
from vv_agent.tools.handlers.search import search_files
from vv_agent.tools.handlers.workspace_io import find_files, read_file
from vv_agent.types import Message, ToolCall, ToolExecutionResult, ToolResultStatus
from vv_agent.workspace import DiscoveryFilteredWorkspaceBackend, LocalWorkspaceBackend, MemoryWorkspaceBackend
from vv_agent.workspace.artifacts import _session_artifact_scope, persist_text_artifact


def test_character_page_projection_preserves_bytes_and_accepts_summary(tmp_path):
    backend = MemoryWorkspaceBackend()
    backend.write_text("source.txt", "x" * 40_000)
    context = ToolContext(tmp_path, {}, 0, backend)
    page = read_file(context, {"path": "source.txt", "start_char": 1})
    page.tool_call_id = "read"
    projected = page.to_tool_message()
    assert projected.content.rpartition("\n")[0] == page.content
    assert page.visible_bytes == len(page.content.encode())
    assert page.original_bytes is not None and page.visible_bytes is not None
    assert page.original_bytes >= page.visible_bytes
    assert page.cursor is not None
    history = [
        Message("user", "Read source"),
        Message(
            "assistant",
            "",
            tool_calls=[
                {"id": "read", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"source.txt"}'}}
            ],
        ),
        projected,
        Message("assistant", "recent"),
    ]
    manager = MemoryManager(keep_recent_messages=1, recovery_tool_available=True)
    evidence = manager.compaction_evidence(history)
    assert evidence["cursors"][0]["cursor"] == page.cursor.to_dict()
    plan = manager.plan_summary(history, keep_recent=1)
    assert plan is not None
    summary, accepted = manager.accept_summary(plan, '{"progress":["Read the first page"]}')
    assert accepted
    assert summary[0].metadata["_vv_agent_compaction"]["cursors"] == evidence["cursors"]


def test_cursor_continuations_always_return_coordinates(tmp_path):
    backend = MemoryWorkspaceBackend()
    backend.write_text("source.txt", "a" * 50_000)
    context = ToolContext(tmp_path, {}, 0, backend)
    first = read_file(context, {"path": "source.txt", "start_char": 1})
    assert first.cursor is not None
    second = read_file(context, {"path": "source.txt", "cursor": first.cursor.to_dict()})
    assert second.cursor is not None
    third = read_file(context, {"path": "source.txt", "cursor": second.cursor.to_dict()})
    assert [p.metadata["read_page"] for p in (first, second, third)] == [
        {"next_line": 1, "next_start_char": offset} for offset in (12_001, 24_001, 36_001)
    ]


def test_read_schema_has_only_fixed_character_budget():
    from vv_agent.constants.workspace import get_default_tool_schemas

    schema = get_default_tool_schemas()["read_file"]["function"]["parameters"]
    assert "start_char" in schema["properties"]
    assert "max_chars" not in schema["properties"]
    assert schema["additionalProperties"] is False


def test_unavailable_marker_in_source_is_bounded_and_command_runs_once(tmp_path):
    """A subprocess deadline also bounds the pre-fix growing-string loop."""
    program = r"""
import json, resource, shlex, sys, time
from pathlib import Path
from vv_agent.tools.base import ToolContext
from vv_agent.tools.builtins import build_default_registry
from vv_agent.tools.handlers.background import _UNAVAILABLE_MARKER, background_command_result
from vv_agent.runtime.background_sessions import background_session_manager
from vv_agent.types import ToolCall, ToolResultStatus
from vv_agent.workspace import MemoryWorkspaceBackend
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
class FailedWriter(MemoryWorkspaceBackend):
    def write_text_chunks_exclusive(self, path, chunks):
        raise OSError("writer failed")
root = Path(sys.argv[1])
counter = root / "effects"
source = ("from pathlib import Path; Path(" + repr(str(counter))
          + ").open('a').write('effect\\n'); print("
          + repr(_UNAVAILABLE_MARKER + "x" * (12001 - len(_UNAVAILABLE_MARKER))) + ",end='')")
command = shlex.quote(sys.executable) + " -c " + shlex.quote(source)
registry = build_default_registry()
context = ToolContext(root, {}, 0, FailedWriter(), task_id="task")
result = registry.execute(ToolCall("start", "bash", {"command":command, "yield_time_ms":0}), context)
sid = json.loads(result.content)["session_id"]
deadline = time.monotonic() + 8
while True:
    assert time.monotonic() < deadline
    payload = background_session_manager.check_for_tool(sid, context.workspace_backend, context.task_id, "check", workspace=root)
    if payload["status"] == "completed":
        result = background_command_result(payload, force_handle=True)
        break
    time.sleep(.01)
assert result.status_code is ToolResultStatus.SUCCESS
assert result.metadata["exit_code"] == 0 and result.directive.value == "continue"
assert result.metadata["retention"] == {"status":"unavailable","reason":"artifact_persist_failed"}
assert result.artifact is None and result.cursor is None and result.truncated is None
assert len(json.loads(result.content)["output"]) <= 12000
again = registry.execute(ToolCall("again", "check_background_command", {"session_id":sid}), context)
assert again.status_code is ToolResultStatus.SUCCESS
assert counter.read_text() == "effect\n"
"""
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_first_capture_read_error_preserves_confirmed_process_result(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from vv_agent.runtime import background_sessions
    from vv_agent.runtime.background_sessions import BackgroundSessionManager
    from vv_agent.tools.handlers import bash

    class CompletedProcess:
        def poll(self):
            return 0

    calls = []

    def start(*args, **kwargs):
        calls.append("effect")
        return SimpleNamespace(process=CompletedProcess(), output_path=tmp_path / "missing", started_at=time.monotonic())

    monkeypatch.setattr(bash, "start_captured_process", start)
    monkeypatch.setattr(bash, "background_session_manager", BackgroundSessionManager())
    monkeypatch.setattr(background_sessions, "process_tree_is_running", lambda _process: False)
    result = build_default_registry().execute(
        ToolCall("run", "bash", {"command": "ignored", "yield_time_ms": 0}),
        ToolContext(tmp_path, {}, 0, MemoryWorkspaceBackend(), task_id="task"),
    )
    assert isinstance(result, ToolExecutionResult)
    assert result.status_code is ToolResultStatus.SUCCESS and result.error_code is None
    assert result.directive.value == "continue"
    assert result.metadata["status"] == "completed" and result.metadata["exit_code"] == 0
    assert result.metadata["retention"] == {"status": "unavailable", "reason": "artifact_persist_failed"}
    assert result.artifact is None and calls == ["effect"]


def test_task_string_cannot_impersonate_compaction_scope():
    backend = MemoryWorkspaceBackend()
    ref = persist_text_artifact(backend, _session_artifact_scope("victim", "task"), "call", "secret", reuse_existing=True)
    assert is_task_artifact_path(ref.path, task_id="task", session_id="victim")
    assert not is_task_artifact_path(ref.path, task_id='session/["victim","task"]')
    assert not is_task_artifact_path(ref.path, task_id='["session","victim","task"]')


def test_task_and_session_hashes_use_typed_json_domains():
    from vv_agent.workspace.artifacts import _artifact_task_segment

    for value, identity in [("task/a", ["task", "task/a"]), (_session_artifact_scope("s", "t"), ["session", "s", "t"])]:
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        assert _artifact_task_segment(value) == "s-" + digest[:62]


def test_032_uuid_task_artifact_path_remains_byte_identical_and_authorized():
    task = "0c8624dcc611467f828ad5994f351575"
    text = "retained"
    digest = hashlib.sha256(text.encode()).hexdigest()
    expected = f".vv-agent/artifacts/{task}/call-{digest}.txt"
    ref = persist_text_artifact(MemoryWorkspaceBackend(), task, "call", text, reuse_existing=True)
    assert ref.path == expected
    assert is_task_artifact_path(expected, task_id=task)
    old_compaction = f".vv-agent/artifacts/session-sid-{task}/call-{digest}.txt"
    assert not is_task_artifact_path(old_compaction, task_id=task, session_id="sid")


@pytest.mark.parametrize("artifact", [False, True])
def test_reverse_matching_is_rejected_before_window_scanning(tmp_path, artifact):
    backend = MemoryWorkspaceBackend()
    text = "needle" + "x" * 28_994 + "needle" + "x" * 30_000
    path = "source.txt"
    if artifact:
        path = persist_text_artifact(backend, "task", "call", text, reuse_existing=True).path
    else:
        backend.write_text(path, text)
    result = search_files(ToolContext(tmp_path, {}, 0, backend, task_id="task"), {"path": path, "pattern": "(?r)needle"})
    assert result.status_code is ToolResultStatus.ERROR
    assert "REVERSE" in result.content


def test_reverse_flag_is_rejected():
    import regex

    from vv_agent.workspace.matching import compile_pattern

    with pytest.raises(ValueError, match="REVERSE"):
        compile_pattern("needle", regex.REVERSE)


@pytest.mark.parametrize("artifact", [False, True])
@pytest.mark.parametrize("pattern,columns", [(r"\w+", [0]), (r"\b", [0, 3]), (r"\W", []), ("(?i)i", [])])
def test_unicode_matching_uses_current_engine_rules(tmp_path, artifact, pattern, columns):
    backend = MemoryWorkspaceBackend()
    text = "\u0131" if pattern == "(?i)i" else "a\u0301b"
    path = "source.txt"
    if artifact:
        path = persist_text_artifact(backend, "task", "call", text, reuse_existing=True).path
    else:
        backend.write_text(path, text)
    result = search_files(
        ToolContext(tmp_path, {}, 0, backend, task_id="task"),
        {"path": path, "pattern": pattern, "case_sensitive": True, "output_mode": "content"},
    )
    if artifact:
        assert [m["column"] for m in result.metadata["artifact_search"]["matches"]] == columns
    else:
        assert result.metadata["summary"]["total_matches"] == len(columns)


@pytest.mark.parametrize("backend_name", ["memory", "local", "s3"])
def test_glob_enumeration_interrupts_pathological_matching(tmp_path, backend_name):
    if backend_name == "local":
        backend = LocalWorkspaceBackend(tmp_path)
    elif backend_name == "s3":
        from test_workspace_backends import _FakeS3Client

        from vv_agent.workspace.s3 import S3WorkspaceBackend

        backend = object.__new__(S3WorkspaceBackend)
        backend._bucket, backend._prefix, backend._client = "test", "", _FakeS3Client()
    else:
        backend = MemoryWorkspaceBackend()
    for i in range(40):
        backend.write_text("a" * 210 + str(i), "")
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        backend.list_files(".", "*a" * 20 + "Z")
    assert time.monotonic() - started < 2


@pytest.mark.parametrize("handler", [find_files, search_files])
def test_discovery_timeout_is_an_explicit_tool_error(tmp_path, handler):
    backend = MemoryWorkspaceBackend()
    backend.write_text("a" * 30_000, "")
    started = time.monotonic()
    arguments = {"glob": "*a" * 20 + "Z"}
    if handler is search_files:
        arguments["pattern"] = "needle"
    result = handler(ToolContext(tmp_path, {}, 0, backend), arguments)
    assert time.monotonic() - started < 2
    assert result.status_code is ToolResultStatus.ERROR
    assert "too slow; aborted" in result.content


def test_sub_task_exclude_regex_has_interruptible_discovery(tmp_path):
    backend = MemoryWorkspaceBackend()
    backend.write_text("a" * 30_000 + "!", "")
    child_backend = DiscoveryFilteredWorkspaceBackend(backend, "(a+)+$")
    started = time.monotonic()
    result = search_files(ToolContext(tmp_path, {}, 0, child_backend), {"pattern": "needle"})
    assert time.monotonic() - started < 2
    assert result.status_code is ToolResultStatus.ERROR and "too slow; aborted" in result.content


def test_local_find_glob_fallback_interrupts_pathological_matching(tmp_path, monkeypatch):
    from vv_agent.tools.handlers import workspace_io

    monkeypatch.setattr(workspace_io, "_resolve_rg_executable", lambda: None)
    backend = LocalWorkspaceBackend(tmp_path)
    for i in range(40):
        backend.write_text("a" * 210 + str(i), "")
    started = time.monotonic()
    result = find_files(ToolContext(tmp_path, {}, 0, backend), {"glob": "*a" * 20 + "Z"})
    assert time.monotonic() - started < 2
    assert result.status_code is ToolResultStatus.ERROR and result.error_code == "regex_timeout"


def test_search_enumeration_cpu_is_charged_before_content_matching(tmp_path, monkeypatch):
    from vv_agent.workspace import matching

    clock = [0.0]
    monkeypatch.setattr(matching, "thread_time", lambda: clock[0])

    class Enumeration(MemoryWorkspaceBackend):
        def list_files(self, base, glob):
            clock[0] += 0.3
            return []

    result = search_files(ToolContext(tmp_path, {}, 0, Enumeration()), {"pattern": "needle"})
    assert result.status_code is ToolResultStatus.ERROR and "too slow; aborted" in result.content


def test_unavailable_handle_does_not_regain_artifact_pointer_in_kernel(monkeypatch):
    from vv_agent.tools.handlers import bash
    from vv_agent.tools.handlers.background import background_command_result

    from .test_tool_result_artifacts import ArchiveWorkspace, produce

    def handle(payload, *, foreground_cwd=None, force_handle=False):
        return background_command_result(payload, foreground_cwd=foreground_cwd, force_handle=True)

    monkeypatch.setattr(bash, "background_command_result", handle)
    backend = ArchiveWorkspace()
    backend.fail_write = True
    _plan, receipt, _backend = produce("H" * 12_001, backend=backend, terminal=True)
    raw = receipt.payload["result"]
    assert raw["status_code"] == "SUCCESS" and raw["directive"] == "continue"
    assert raw["metadata"]["exit_code"] == 0
    assert raw["metadata"]["retention"] == {"status": "unavailable", "reason": "artifact_persist_failed"}
    assert not {"artifact", "cursor", "truncated"} & raw.keys()
    assert not {"vv_artifact", "app_server_resources"} & raw["metadata"].keys()
    assert len(raw["content"]) <= 12_000
    assert json.loads(raw["content"])["retention"]["status"] == "unavailable"
