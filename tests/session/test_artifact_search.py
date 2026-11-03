"""Task-scoped streaming search and its real kernel completion projection."""

import json
import runpy
import tracemalloc
from contextlib import nullcontext
from pathlib import Path

import pytest

from vv_agent import Agent, RunConfig
from vv_agent.config import ResolvedModelConfig
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.runtime import Runtime
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.tools.base import ToolContext
from vv_agent.tools.builtins import build_default_registry
from vv_agent.tools.handlers.search import _MAX_RESULT_CHARS, _MAX_RESULT_LINES, search_files
from vv_agent.tools.handlers.workspace_io import read_file
from vv_agent.types import LLMResponse, ToolCall
from vv_agent.workspace import LocalWorkspaceBackend, MemoryWorkspaceBackend
from vv_agent.workspace.artifacts import _artifact_segment, persist_text_artifact
from vv_agent.workspace.base import DiscoveryFilteredWorkspaceBackend

PRODUCER = runpy.run_path(str(Path(__file__).parents[2] / "scripts/artifact_search_fixtures.py"))
TASK = PRODUCER["TASK"]


@pytest.mark.parametrize("name", list(PRODUCER["CASES"]))
def test_real_handler_cases(name):
    result, _backend, arguments = PRODUCER["produce"](PRODUCER["CASES"][name])
    if name in {"denied", "invalid_path"}:
        assert result.error_code == "artifact_search_denied"
    elif name == "missing":
        assert result.error_code == "artifact_unavailable"
    elif name == "invalid_regex":
        assert result.error_code == "invalid_arguments"
    else:
        assert result.status_code.value == "SUCCESS"
        assert result.artifact is None and result.truncated is None
        assert len(result.content) <= _MAX_RESULT_CHARS
        assert len(result.content.splitlines()) <= _MAX_RESULT_LINES
        assert result.content.endswith("Artifact line numbers can be used directly with read_file start_line/end_line.")
        details = result.metadata["artifact_search"]
        if name == "no_hit":
            assert details["matched_lines"] == 0 and not details["truncated"]
        elif name == "long_line":
            assert details["truncated"] and details["oversized_line"] == 1
            assert "was not matched" in result.content and "Search truncated" in result.content
        elif name.startswith("budget"):
            assert details["truncated"] and "narrow pattern" in result.content
        else:
            expected_line = 3001 if name == "line_numbers" else 2
            assert f"\n{expected_line}: " in result.content
            context = ToolContext(Path("/tmp"), {}, 0, _backend, task_id=TASK)
            recovered = read_file(context, {"path": arguments["path"], "start_line": expected_line, "end_line": expected_line})
            assert recovered.content.removesuffix("\r") in result.content


@pytest.mark.parametrize("backend_kind", ["memory", "local", "filtered"])
def test_direct_read_and_discovery_exclusion_coexist(tmp_path, backend_kind):
    backend = LocalWorkspaceBackend(tmp_path / "ws") if backend_kind == "local" else MemoryWorkspaceBackend()
    ref = persist_text_artifact(backend, TASK, "call", "first\nneedle\n", reuse_existing=True)
    if backend_kind == "filtered":
        backend = DiscoveryFilteredWorkspaceBackend(backend, r"\.vv-agent/")
    assert backend.list_files(".", "**/*") == []
    assert backend.list_files(".vv-agent/artifacts", "**/*") == []
    assert backend.read_text(ref.path) == "first\nneedle\n"
    result = search_files(
        ToolContext(tmp_path, {}, 0, backend, task_id=TASK),
        {"path": ref.path, "pattern": "needle", "output_mode": "content", "type": "invalid", "glob": "no"},
    )
    assert "\n2: needle\n" in result.content


def test_host_task_identity_uses_the_same_rule_as_completion_storage(tmp_path):
    class TaskWorkspace(MemoryWorkspaceBackend):
        task_id = "host-task / 漢🙂"

    backend = TaskWorkspace()
    own = persist_text_artifact(backend, backend.task_id, "call", "needle\n", reuse_existing=True)
    other = persist_text_artifact(backend, TASK, "call", "needle\n", reuse_existing=True)
    context = ToolContext(tmp_path, {}, 0, backend, task_id=TASK)
    assert search_files(context, {"path": own.path, "pattern": "needle"}).status_code.value == "SUCCESS"
    assert search_files(context, {"path": other.path, "pattern": "needle"}).error_code == "artifact_search_denied"


@pytest.mark.parametrize(
    "options,expected",
    [
        ({"output_mode": "count"}, "2 matches on 2 lines"),
        ({"output_mode": "files_with_matches"}, ".vv-agent/artifacts/"),
        ({"output_mode": "content", "offset": 1, "head_limit": 1}, "\n3: needle SECOND\n"),
        ({"output_mode": "content", "case_sensitive": True}, "\n3: needle SECOND\n"),
        ({"output_mode": "content", "literal": True, "pattern": "a.b"}, "\n4: a.b\n"),
    ],
)
def test_modes_case_literal_and_pagination(tmp_path, options, expected):
    backend = MemoryWorkspaceBackend()
    ref = persist_text_artifact(backend, TASK, "call", "first\nNEEDLE FIRST\nneedle SECOND\na.b\n", reuse_existing=True)
    result = search_files(
        ToolContext(tmp_path, {}, 0, backend, task_id=TASK),
        {"path": ref.path, "pattern": "needle", **options},
    )
    assert result.status_code.value == "SUCCESS"
    assert expected in result.content
    if options.get("case_sensitive"):
        assert "NEEDLE FIRST" not in result.content


def test_oversized_line_is_not_falsely_matched_and_later_lines_stay_searchable(tmp_path):
    backend = MemoryWorkspaceBackend()
    ref = persist_text_artifact(backend, TASK, "call", "prefix\n" + "needle" * 10000 + "\nneedle last", reuse_existing=True)
    result = search_files(
        ToolContext(tmp_path, {}, 0, backend, task_id=TASK),
        {"path": ref.path, "pattern": "^needle", "output_mode": "content"},
    )
    assert "\n3: needle last\n" in result.content
    assert result.metadata["artifact_search"]["oversized_line"] == 2
    assert result.metadata["artifact_search"]["matched_lines"] == 1
    assert "Search truncated" in result.content


@pytest.mark.parametrize(
    "path",
    [
        ".vv-agent/artifacts",
        ".vv-agent/artifacts/",
        ".vv-agent/artifacts/{task}",
        ".vv-agent/artifacts/{task}/",
        ".vv-agent/artifacts/{task}/*.txt",
        ".vv-agent/artifacts/{task}/../call-" + "a" * 64 + ".txt",
        ".vv-agent/artifacts/{task}/call-" + "a" * 64 + ".json",
        ".vv-agent/artifacts/{task}/call.txt",
        ".vv-agent/artifacts/{task}/" + "a" * 60 + "-" + "b" * 64 + ".txt",
        ".vv-agent/artifacts/{task}/sub/call-" + "a" * 64 + ".txt",
        "./.vv-agent/artifacts/{task}/call-" + "a" * 64 + ".txt",
        "normal/../.vv-agent/artifacts/{task}/call-" + "a" * 64 + ".txt",
        "/.vv-agent/artifacts/{task}/call-" + "a" * 64 + ".txt",
        ".vv-agent\\artifacts\\{task}\\call-" + "a" * 64 + ".txt",
    ],
)
def test_invalid_paths_never_read_or_list(tmp_path, path):
    class NoAccessBackend(MemoryWorkspaceBackend):
        def is_file(self, path):
            raise AssertionError("denied path must not touch storage")

        def list_files(self, base, glob):
            raise AssertionError("artifact route must not traverse")

    result = search_files(
        ToolContext(tmp_path, {}, 0, NoAccessBackend(), task_id=TASK),
        {"path": path.format(task=_artifact_segment(TASK, "task")), "pattern": "needle"},
    )
    assert result.error_code == "artifact_search_denied"


@pytest.mark.parametrize("fault", ["unreadable", "invalid_utf8", "directory"])
def test_unavailable(tmp_path, fault):
    class BrokenBackend(MemoryWorkspaceBackend):
        def _read_bytes_chunks(self, path):
            if fault == "unreadable":
                raise PermissionError("denied")
            yield b"\xff"

    backend = BrokenBackend()
    ref = persist_text_artifact(backend, TASK, "call", "needle\n", reuse_existing=True)
    if fault == "directory":
        del backend._files[ref.path]
        backend.mkdir(ref.path)
    result = search_files(ToolContext(tmp_path, {}, 0, backend, task_id=TASK), {"path": ref.path, "pattern": "needle"})
    assert result.error_code == "artifact_unavailable"


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 7, 31])
def test_regex_utf8_crlf_and_lines_across_chunk_boundaries(tmp_path, chunk_size):
    class TinyChunks(MemoryWorkspaceBackend):
        writes = reads = 0

        def _read_bytes_chunks(self, path):
            self.reads += 1
            data = self._files[path]
            for start in range(0, len(data), chunk_size):
                yield data[start : start + chunk_size]

        def write_text_chunks_exclusive(self, path, chunks):
            self.writes += 1
            return super().write_text_chunks_exclusive(path, chunks)

        def read_bytes(self, path):
            raise AssertionError("search must stream")

        def read_text(self, path):
            raise AssertionError("search must stream")

    backend = TinyChunks()
    text = "first\r\nprefix漢🙂跨块suffix\r\nlast"
    ref = persist_text_artifact(backend, TASK, "call", text, reuse_existing=True)
    result = search_files(
        ToolContext(tmp_path, {}, 0, backend, task_id=TASK),
        {"path": ref.path, "pattern": "^prefix漢.跨块suffix$", "output_mode": "content"},
    )
    assert "\n2: prefix漢🙂跨块suffix\n" in result.content
    assert backend.reads == 1 and backend.writes == 1
    assert result.artifact is None


@pytest.mark.parametrize("long_line", [False, True])
def test_streaming_search_memory_is_bounded(tmp_path, long_line, record_property):
    backend = LocalWorkspaceBackend(tmp_path / "ws")
    text = ("no match🙂" if long_line else "no match🙂\n") * 1_000_000
    ref = persist_text_artifact(backend, TASK, "call", text, reuse_existing=True)
    context = ToolContext(tmp_path, {}, 0, backend, task_id=TASK)
    del text
    tracemalloc.start()
    result = search_files(context, {"path": ref.path, "pattern": "needle", "output_mode": "content"})
    _used, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    record_property("peak_bytes", peak)
    assert peak < 4_000_000
    assert result.metadata["artifact_search"]["truncated"] is long_line


@pytest.mark.parametrize("host_task", [False, True])
def test_real_kernel_search_completion_uses_unified_projection_without_repeating_effect(tmp_path, host_task):
    class TaskWorkspace(MemoryWorkspaceBackend):
        task_id = "host-task / 漢🙂"

    backend = TaskWorkspace() if host_task else MemoryWorkspaceBackend()
    text = ("needle" + "x" * 1000 + "\n") * 100
    ref = persist_text_artifact(backend, getattr(backend, "task_id", TASK), "call", text, reuse_existing=True)
    arguments = {"path": ref.path, "pattern": "needle", "output_mode": "content", "head_limit": 0}
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("artifact-search", "test", str(tmp_path)), consumers=())
            tx.push("artifact-search", InboxItem("initial", "user", {"content": "search"}))
        runtime = Runtime(
            agent=Agent(name="search", instructions="Search."),
            config=RunConfig(
                workspace_backend=backend,
                session_memory_enabled=False,
                tool_registry_factory=build_default_registry,
            ),
            resolved=ResolvedModelConfig("scripted", "scripted", "scripted", "scripted", []),
            llm=ScriptedLLM([LLMResponse("", [ToolCall("search", "search_files", arguments)]), LLMResponse("done")]),
            heartbeat_store=lambda: nullcontext(store),
        )
        drive(store, "artifact-search", runtime=runtime)
        records = [row.record for row in store.read("artifact-search", limit=100).records]
        result = next(
            r.payload["result"] for r in records if r.kind == "op_completed" and r.payload["result"].get("tool_call_id")
        )
        assert result["status_code"] == "SUCCESS", result
        assert result["truncated"] and len(result["content"]) == 12000
        assert result["metadata"]["vv_artifact"]["source_complete"] is True
        archived = backend.read_text(result["artifact"]["path"])
        assert len(archived) > 12000 and archived.endswith("read_file start_line/end_line.")
        before = [r.encode() for r in records]
        drive(store, "artifact-search", runtime=runtime)
        assert before == [r.record.encode() for r in store.read("artifact-search", limit=100).records]


def test_vectors_are_generated_by_the_real_handler():
    expected = json.loads((Path(__file__).parents[1] / "fixtures/parity/artifact_search.json").read_text())
    assert PRODUCER["generate"]() == expected
