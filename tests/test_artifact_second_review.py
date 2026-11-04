"""Enumeration budgets, platform glob semantics and task upgrade boundaries."""

import hashlib
import json
import ntpath
import os
import posixpath

import pytest
from test_workspace_backends import _FakeS3Client

from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.search import search_files
from vv_agent.types import ToolResultStatus
from vv_agent.workspace import DiscoveryFilteredWorkspaceBackend, LocalWorkspaceBackend, MemoryWorkspaceBackend
from vv_agent.workspace.s3 import S3WorkspaceBackend


@pytest.mark.parametrize("empty_page", [False, True])
def test_s3_budget_stops_before_requesting_next_page(monkeypatch, empty_page):
    from vv_agent.workspace import matching

    clock, pages = [0.0], []
    monkeypatch.setattr(matching, "thread_time", lambda: clock[0])

    class Paginator:
        def paginate(self, **kwargs):
            pages.append(1)
            clock[0] += 0.3
            yield {"Contents": [] if empty_page else [{"Key": "a.txt"}]}
            pages.append(2)
            yield {"Contents": [{"Key": "b.txt"}]}

    class Client:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return Paginator()

    backend = object.__new__(S3WorkspaceBackend)
    backend._bucket, backend._prefix, backend._client = "test", "", Client()
    with pytest.raises(TimeoutError, match="too slow; aborted"):
        backend.list_files(".", "**/*")
    assert pages == [1]


def test_s3_budget_stops_between_objects(monkeypatch):
    from vv_agent.workspace import matching

    clock, visited = [0.0], []
    monkeypatch.setattr(matching, "thread_time", lambda: clock[0])

    class Objects:
        def __iter__(self):
            visited.append("a")
            clock[0] += 0.3
            yield {"Key": "a.txt"}
            visited.append("b")
            yield {"Key": "b.txt"}

    class Paginator:
        def paginate(self, **kwargs):
            yield {"Contents": Objects()}

    class Client:
        def get_paginator(self, name):
            return Paginator()

    backend = object.__new__(S3WorkspaceBackend)
    backend._bucket, backend._prefix, backend._client = "test", "", Client()
    with pytest.raises(TimeoutError, match="too slow; aborted"):
        backend.list_files(".", "**/*")
    assert visited == ["a"]


def test_local_budget_stops_before_entering_next_empty_directory(tmp_path, monkeypatch):
    from vv_agent.workspace import matching

    clock, entered = [0.0], []
    monkeypatch.setattr(matching, "thread_time", lambda: clock[0])

    def walk(root, **kwargs):
        entered.append("first")
        clock[0] += 0.3
        yield str(root), ["second"], []
        entered.append("second")
        yield str(root / "second"), [], []

    monkeypatch.setattr(os, "walk", walk)
    with pytest.raises(TimeoutError, match="too slow; aborted"):
        LocalWorkspaceBackend(tmp_path).list_files(".", "**/*")
    assert entered == ["first"]


def test_nested_filter_backend_and_content_share_request_budget(tmp_path, monkeypatch):
    from vv_agent.workspace import matching

    clock, stages, deadlines = [0.0], [], []
    monkeypatch.setattr(matching, "thread_time", lambda: clock[0])
    original_init = matching.RegexBudget.__init__
    original_match, original_search = matching.RegexBudget.match, matching.RegexBudget.search

    def initialize(self, pattern):
        original_init(self, pattern)
        deadlines.append(self.deadline)
        clock[0] += 0.02

    def match(self, text):
        stages.append("builtin glob")
        clock[0] += 0.07
        return original_match(self, text)

    def search(self, text):
        stages.append("exclude")
        clock[0] += 0.07
        return original_search(self, text)

    class Backend(MemoryWorkspaceBackend):
        def read_text(self, path):
            stages.append("content")
            clock[0] += 0.07
            return super().read_text(path)

    monkeypatch.setattr(matching.RegexBudget, "__init__", initialize)
    monkeypatch.setattr(matching.RegexBudget, "match", match)
    monkeypatch.setattr(matching.RegexBudget, "search", search)
    backend = Backend()
    backend.write_text("a.txt", "needle")
    filtered = DiscoveryFilteredWorkspaceBackend(backend, "excluded")
    result = search_files(ToolContext(tmp_path, {}, 0, filtered), {"pattern": "needle", "include_sensitive": True})
    assert stages == ["builtin glob", "exclude", "content"]
    assert deadlines == [matching.MATCH_CPU_SECONDS] * 3
    assert "too slow; aborted" in result.content
    assert result.metadata["regex_timeout"] == {"path": "a.txt", "line": 1, "column": 0}
    assert result.metadata["summary"]["total_matches"] == 0


@pytest.mark.parametrize("backend_name", ["memory", "local", "s3"])
@pytest.mark.parametrize("platform", ["posix", "windows"])
@pytest.mark.parametrize("pattern", ["[ab].txt", "*.txt", "**/*.txt", "*.TXT"])
def test_plain_glob_matches_032_baseline_matrix(tmp_path, monkeypatch, backend_name, platform, pattern):
    backend = _backend(backend_name, tmp_path)
    paths = ["a.txt", "b.txt", "A.TXT", "sub/a.txt", "sub/deep/b.txt", "sub/A.TXT", "[ab].txt", "notes.md"]
    for path in paths:
        backend.write_text(path, "")
    # Only fnmatch's branch normalizes case/separators on Windows.
    with monkeypatch.context() as patch:
        patch.setattr(os.path, "normcase", ntpath.normcase if platform == "windows" else posixpath.normcase)
        actual = backend.list_files(".", pattern)
    lower_root = ["[ab].txt", "a.txt", "b.txt"]
    lower_all = [*lower_root, "sub/a.txt", "sub/deep/b.txt"]
    expected = {
        "[ab].txt": ["[ab].txt"],
        "*.txt": lower_root,
        "**/*.txt": lower_all,
        "*.TXT": ["A.TXT"],
    }[pattern]
    if backend_name == "s3":
        if pattern == "[ab].txt":
            expected = lower_root + (["A.TXT"] if platform == "windows" else [])
        elif platform == "windows":
            expected = lower_all + ["sub/A.TXT"] + ([] if pattern == "**/*.txt" else ["A.TXT"])
        elif pattern == "*.txt":
            expected = lower_all
        elif pattern == "*.TXT":
            expected = ["A.TXT", "sub/A.TXT"]
    assert actual == sorted(expected)


def _backend(name, root):
    if name == "local":
        return LocalWorkspaceBackend(root)
    if name == "memory":
        return MemoryWorkspaceBackend()
    backend = object.__new__(S3WorkspaceBackend)
    backend._bucket, backend._prefix, backend._client = "test", "", _FakeS3Client()
    return backend


@pytest.mark.parametrize(
    "task,old_segment,unchanged",
    [
        ("task", "task", True),
        ("0c8624dcc611467f828ad5994f351575", "0c8624dcc611467f828ad5994f351575", True),
        ("A" + "._-9" * 15 + "xyz", "A" + "._-9" * 15 + "xyz", True),
        ("s-task", "s-task", False),
        ("S-task", "S-task", True),
        ("task/a", "task-a", False),
        ("task\\a", "task-a", False),
        ("t" * 65, "t" * 64, False),
        ("任务", "task", False),
        ("t任务", "t-", False),
        ("-task", "task", False),
    ],
)
def test_task_path_upgrade_matrix_preserves_only_short_literal_ids(tmp_path, task, old_segment, unchanged):
    from vv_agent.tools.handlers.workspace_io import read_file
    from vv_agent.workspace.artifacts import is_task_artifact_path, persist_text_artifact

    backend = MemoryWorkspaceBackend()
    text = "retained"
    suffix = "call-" + hashlib.sha256(text.encode()).hexdigest() + ".txt"
    old_path = f".vv-agent/artifacts/{old_segment}/{suffix}"
    encoded = json.dumps(["task", task], ensure_ascii=False, separators=(",", ":")).encode()
    segment = task if unchanged else "s-" + hashlib.sha256(encoded).hexdigest()[:62]
    new_path = f".vv-agent/artifacts/{segment}/{suffix}"
    # Old objects remain physically present, even when their scope is no longer authorized.
    backend.write_text_exclusive(old_path, text)
    ref = persist_text_artifact(backend, task, "call", text, reuse_existing=True)
    assert ref.path == new_path
    assert backend.read_text(old_path) == backend.read_text(new_path) == text
    assert is_task_artifact_path(new_path, task_id=task)
    assert is_task_artifact_path(old_path, task_id=task) is unchanged
    context = ToolContext(tmp_path, {}, 0, backend, task_id=task)
    assert read_file(context, {"path": new_path}).status_code is ToolResultStatus.SUCCESS
    assert read_file(context, {"path": old_path}).status_code is (
        ToolResultStatus.SUCCESS if unchanged else ToolResultStatus.ERROR
    )
