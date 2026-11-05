"""Per-engine ReDoS limits and request CPU limits on real search producers."""

import time
from typing import cast

import pytest
import regex

from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.search import search_files
from vv_agent.tools.handlers.workspace_io import find_files
from vv_agent.types import ToolResultStatus
from vv_agent.workspace import LocalWorkspaceBackend, MemoryWorkspaceBackend, matching
from vv_agent.workspace.artifacts import persist_text_artifact


@pytest.fixture(params=["memory", "local", "s3"])
def backend(request, tmp_path):
    if request.param == "local":
        return LocalWorkspaceBackend(tmp_path)
    if request.param == "s3":
        from test_workspace_backends import _make_s3_backend

        return _make_s3_backend()
    return MemoryWorkspaceBackend()


@pytest.mark.parametrize("controlled_cpu", [False, True], ids=["real-clock", "controlled-clock"])
@pytest.mark.parametrize("artifact", [False, True], ids=["workspace", "artifact"])
@pytest.mark.parametrize("literal", [False, True], ids=["regex", "literal"])
def test_large_linear_search_completes(backend, tmp_path, monkeypatch, artifact, literal, controlled_cpu):
    if controlled_cpu:
        clock = [0.0]
        monkeypatch.setattr(matching, "thread_time", lambda: clock[0])
        original = matching.RegexBudget.finditer

        def per_line(self, text, pos=0):
            clock[0] += 1 / 131_072  # Exactly 250 ms at line 32768, before the target.
            yield from original(self, text, pos)

        monkeypatch.setattr(matching.RegexBudget, "finditer", per_line)
    lines = ["ordinary content " + "x" * 32 + "\n"] * 40_000
    lines[33_332] = "NEEDLE-33333 " + "x" * 36 + "\n"
    text = "".join(lines)
    assert len(text.encode()) == 2_000_000
    if artifact:
        path = persist_text_artifact(backend, "task", "result", text, reuse_existing=True).path
    else:
        path = "source.txt"
        backend.write_text(path, text)
    started_cpu, started = time.thread_time(), time.monotonic()
    result = search_files(
        ToolContext(tmp_path, {}, 0, backend, task_id="task"),
        {"path": path, "pattern": "NEEDLE" if literal else r"NEEDLE-\d+", "literal": literal, "output_mode": "content"},
    )
    elapsed, cpu = time.monotonic() - started, time.thread_time() - started_cpu
    print(
        f"LINEAR controlled_cpu={controlled_cpu} {type(backend).__name__} artifact={artifact} literal={literal} "
        f"bytes={len(text)} wall={elapsed:.6f}s cpu={cpu:.6f}s result={result.content!r}"
    )
    assert result.status_code is ToolResultStatus.SUCCESS
    assert not result.metadata.get("regex_timeout"), result.content
    if artifact:
        details = result.metadata["artifact_search"]
        assert not details["timed_out"] and not details["truncated"], result.content
        hits = details["matches"]
        assert details["matched_lines"] == 1
    else:
        assert not result.metadata["truncated"], result.content
        hits = result.metadata["matches"]
        assert result.metadata["summary"]["total_matches"] == 1
    assert [hit["line"] for hit in hits] == [33_333], result.content
    assert "aborted" not in result.content and "incomplete" not in result.content


@pytest.mark.parametrize("operation", ["finditer", "match", "search"])
def test_each_engine_call_gets_250ms_and_shares_10s(monkeypatch, operation):
    clock = [0.0]
    calls = []
    monkeypatch.setattr(matching, "thread_time", lambda: clock[0])

    class Pattern:
        flags = 0

        def invoke(self, text, *args, timeout):
            calls.append(timeout)
            clock[0] += 0.125
            return iter(()) if operation == "finditer" else None

        finditer = match = search = invoke

    with matching.matching_request():
        first = matching.RegexBudget(cast(regex.Pattern, Pattern()))
        for index in range(79):
            # New matchers and nested scopes must not reset the request deadline.
            with matching.matching_request():
                matcher = first if index < 40 else matching.RegexBudget(cast(regex.Pattern, Pattern()))
                value = getattr(matcher, operation)("line")
                if operation == "finditer":
                    list(value)
        with pytest.raises(TimeoutError, match="total CPU limit"):
            value = getattr(first, operation)("line")
            if operation == "finditer":
                list(value)
    assert clock[0] == 10.0
    assert calls == [0.25] * 79 + [0.125]
    print(f"BOUNDARY {operation}: {len(calls)} calls, total_cpu={clock[0]}s, last_timeout={calls[-1]}s")
    # A later request starts with a fresh budget.
    with matching.matching_request():
        matching.check_matching_budget()


@pytest.mark.parametrize("operation", ["finditer", "match", "search"])
def test_engine_timeout_is_distinct_from_total_limit(monkeypatch, operation):
    monkeypatch.setattr(matching, "thread_time", lambda: 0.0)

    class Pattern:
        flags = 0

        def invoke(self, text, *args, timeout):
            assert timeout == 0.25
            raise TimeoutError("regex timed out")

        finditer = match = search = invoke

    with matching.matching_request(), pytest.raises(TimeoutError, match="Regular expression too slow"):
        value = getattr(matching.RegexBudget(cast(regex.Pattern, Pattern())), operation)("line")
        if operation == "finditer":
            list(value)


@pytest.mark.parametrize("artifact", [False, True])
def test_many_slow_lines_stop_at_total_limit_with_coordinates(tmp_path, monkeypatch, artifact):
    clock = [0.0]
    monkeypatch.setattr(matching, "thread_time", lambda: clock[0])
    original = matching.RegexBudget.finditer

    def slow_line(self, text, pos=0):
        yield from original(self, text, pos)
        clock[0] += 0.125  # Each line is below 250 ms, but the request reaches 10 s.

    monkeypatch.setattr(matching.RegexBudget, "finditer", slow_line)
    b = MemoryWorkspaceBackend()
    text = "needle\n" + "ordinary\n" * 99
    if artifact:
        path = persist_text_artifact(b, "task", "result", text, reuse_existing=True).path
    else:
        path = "source.txt"
        b.write_text(path, text)
    result = search_files(
        ToolContext(tmp_path, {}, 0, b, task_id="task"), {"path": path, "pattern": "needle", "output_mode": "content"}
    )
    assert "total CPU limit" in result.content and "Regular expression too slow" not in result.content
    if artifact:
        details = result.metadata["artifact_search"]
        assert details["timed_out"] and details["truncated"]
        assert (details["stopped_line"], details["stopped_column"]) == (81, 0)
        assert [hit["line"] for hit in details["matches"]] == [1]
    else:
        assert result.metadata["regex_timeout"] == {"path": path, "line": 81, "column": 0}
        assert [hit["line"] for hit in result.metadata["matches"]] == [1]
    print(f"TOTAL artifact={artifact} cpu={clock[0]} result={result.content!r}")


@pytest.mark.parametrize("handler", [search_files, find_files])
def test_enumeration_limit_message_is_distinct(tmp_path, monkeypatch, handler):
    clock = [0.0]
    monkeypatch.setattr(matching, "thread_time", lambda: clock[0])

    class Enumeration(MemoryWorkspaceBackend):
        def list_files(self, base, glob):
            clock[0] = 10.0
            return []

    result = handler(
        ToolContext(tmp_path, {}, 0, Enumeration()),
        {"glob": "**/*", "pattern": "needle"} if handler is search_files else {"glob": "**/*"},
    )
    assert result.status_code is ToolResultStatus.ERROR
    assert "total CPU limit" in result.content and "file enumeration" in result.content
