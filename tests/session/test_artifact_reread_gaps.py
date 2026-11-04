"""Real producers for retained-text recovery regressions."""

import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from vv_agent.tools.base import ToolContext
from vv_agent.tools.builtins import build_default_registry
from vv_agent.tools.handlers.search import search_files
from vv_agent.tools.handlers.workspace_io import file_info, read_file
from vv_agent.types import ToolCall, ToolExecutionResult, ToolResultStatus
from vv_agent.workspace import MemoryWorkspaceBackend
from vv_agent.workspace.artifacts import _session_artifact_scope, persist_text_artifact


def test_d10_current_compaction_scope_can_be_searched(tmp_path):
    backend = MemoryWorkspaceBackend()
    ref = persist_text_artifact(backend, _session_artifact_scope("sid", "task"), "call", "needle", reuse_existing=True)
    context = ToolContext(tmp_path, {}, 0, backend, task_id="task", metadata={"session_id": "sid"})
    result = search_files(context, {"path": ref.path, "pattern": "needle", "output_mode": "content"})
    assert result.status_code is ToolResultStatus.SUCCESS, result.to_dict()
    assert "needle" in result.content


def test_d10_other_compaction_scope_denied_before_read(tmp_path):
    backend = MemoryWorkspaceBackend()
    ref = persist_text_artifact(backend, _session_artifact_scope("other", "task"), "call", "needle", reuse_existing=True)
    context = ToolContext(tmp_path, {}, 0, backend, task_id="task", metadata={"session_id": "sid"})
    assert read_file(context, {"path": ref.path}).error_code == "artifact_read_denied"
    assert file_info(context, {"path": ref.path}).error_code == "artifact_read_denied"


def test_d9_search_column_round_trips_to_streaming_read(tmp_path):
    backend = MemoryWorkspaceBackend()
    ref = persist_text_artifact(backend, "task", "call", "漢" * 40_000 + "needle終", reuse_existing=True)
    context = ToolContext(tmp_path, {}, 0, backend, task_id="task")
    result = search_files(context, {"path": ref.path, "pattern": "needle", "output_mode": "content"})
    details = result.metadata["artifact_search"]
    assert details["matched_lines"] == 1, result.content
    hit = details["matches"][0]
    assert hit["line"] == 1 and hit["column"] == 40_000
    recovered = read_file(
        context,
        {"path": ref.path, "start_line": hit["line"], "end_line": hit["line"], "start_char": hit["column"]},
    )
    assert recovered.content.startswith("needle")
    assert recovered.cursor is None


def test_d8_pathological_regex_returns_within_bound(tmp_path):
    source = """
from pathlib import Path
from vv_agent.tools.base import ToolContext
from vv_agent.tools.handlers.search import search_files
from vv_agent.workspace import MemoryWorkspaceBackend
from vv_agent.workspace.artifacts import _session_artifact_scope, persist_text_artifact
b = MemoryWorkspaceBackend()
r = persist_text_artifact(b, 'task', 'call', 'a'*32+'!', reuse_existing=True)
r = search_files(ToolContext(Path('/tmp'), {}, 0, b, task_id='task'), {'path':r.path, 'pattern':'(a+)+$'})
print(r.to_dict())
"""
    started = time.monotonic()
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=2)
    assert completed.returncode == 0, completed.stderr
    assert time.monotonic() - started < 2


def test_d6_writer_failure_preserves_success_and_directive(tmp_path, monkeypatch):
    from vv_agent.runtime import background_sessions as background_runtime
    from vv_agent.tools.handlers import bash as bash_handler

    class FailingBackend(MemoryWorkspaceBackend):
        def write_text_chunks_exclusive(self, path, chunks):
            raise OSError("disposable writer failure")

    class Process:
        returncode = 0

        def poll(self):
            return 0

    capture = tmp_path / "capture.txt"
    capture.write_text("x" * 12001)
    monkeypatch.setattr(
        bash_handler,
        "start_captured_process",
        lambda *a, **k: SimpleNamespace(process=Process(), output_path=capture, started_at=time.monotonic()),
    )
    monkeypatch.setattr(background_runtime, "process_tree_is_running", lambda p: False)
    result = build_default_registry().execute(
        ToolCall("call", "bash", {"command": "printf ignored"}),
        ToolContext(tmp_path, {}, 0, FailingBackend(), task_id="task"),
    )
    assert isinstance(result, ToolExecutionResult)
    assert result.status_code is ToolResultStatus.SUCCESS, result.to_dict()
    assert result.error_code is None and result.directive.value == "continue"
    assert result.metadata["exit_code"] == 0
    assert result.metadata["retention"]["status"] == "unavailable"
    assert "complete text could not be saved" in result.content
    assert result.truncated is None and result.artifact is None
    assert capture.read_text() == "x" * 12001


def test_d10_session_scope_preserves_full_identity(tmp_path):
    from vv_agent.workspace.artifacts import _artifact_task_segment

    sid = "s" * 40
    task = "t" * 40
    assert _artifact_task_segment(_session_artifact_scope(sid, task)) != _artifact_task_segment(
        _session_artifact_scope(sid, task + "x")
    )
    assert _artifact_task_segment(_session_artifact_scope("a-b", "task")) != _artifact_task_segment(
        _session_artifact_scope("a", "b-task")
    )


def _search_text(tmp_path, text, pattern, **options):
    backend = MemoryWorkspaceBackend()
    ref = persist_text_artifact(backend, "task", "call", text, reuse_existing=True)
    context = ToolContext(tmp_path, {}, 0, backend, task_id="task")
    result = search_files(context, {"path": ref.path, "pattern": pattern, "output_mode": "content", **options})
    return result, context, ref


@pytest.mark.parametrize("column", [0, 25_902, 25_903, 25_904, 29_999, 40_000, 59_999])
def test_window_edges_return_exact_columns_without_duplicate_hits(tmp_path, column):
    result, _, _ = _search_text(tmp_path, "x" * column + "needle" + "x" * 40_000, "needle")
    assert [m["column"] for m in result.metadata["artifact_search"]["matches"]] == [column]
    assert all(len(m["text"]) <= 202 for m in result.metadata["artifact_search"]["matches"])
    assert "4096-character overlap" in result.content


@pytest.mark.parametrize("pattern,expected", [("aa", [25_903]), ("(?=aa)", [25_903, 25_904])])
def test_nonoverlapping_and_zero_width_window_semantics(tmp_path, pattern, expected):
    result, _, _ = _search_text(tmp_path, "x" * 25_903 + "aaa" + "x" * 40_000, pattern)
    assert [m["column"] for m in result.metadata["artifact_search"]["matches"]] == expected


@pytest.mark.parametrize("pattern,columns", [("^x", [0]), ("x$", [59_999]), ("^x+$", [])])
def test_internal_window_edges_do_not_create_line_anchors(tmp_path, pattern, columns):
    result, _, _ = _search_text(tmp_path, "x" * 60_000, pattern)
    assert [m["column"] for m in result.metadata["artifact_search"]["matches"]] == columns
    assert "search is incomplete" in result.content


@pytest.mark.parametrize("prefix", ["", "\ufeff", "first\r\n", "\ufefffirst\r\n"])
def test_unicode_bom_and_line_local_offsets_round_trip(tmp_path, prefix):
    result, context, ref = _search_text(tmp_path, prefix + "漢🙂" * 20_000 + "needle", "needle")
    hit = result.metadata["artifact_search"]["matches"][0]
    assert hit["column"] == 40_000
    assert hit["line"] == (2 if "\n" in prefix else 1)
    page = read_file(
        context,
        {"path": ref.path, "start_line": hit["line"], "end_line": hit["line"], "start_char": hit["column"]},
    )
    assert page.content == "needle"
    assert "read_page" not in page.metadata


def test_real_regex_timeout_retains_earlier_hits_and_stop_position(tmp_path):
    started = time.monotonic()
    result, _, _ = _search_text(tmp_path, "needle\nneedle " + "a" * 29_000 + "!", "needle|(a+)+$")
    assert time.monotonic() - started < 2
    details = result.metadata["artifact_search"]
    assert details["timed_out"] and details["truncated"]
    assert details["stopped_line"] == 2 and details["stopped_column"] == 0
    assert details["matched_lines"] == 2
    assert [m["line"] for m in details["matches"]] == [1, 2]
    assert "Regular expression too slow; aborted" in result.content
    assert "No matches found" not in result.content


@pytest.mark.parametrize("multiline", [False, True])
def test_workspace_regex_fallback_also_interrupts_and_keeps_matches(tmp_path, multiline):
    backend = MemoryWorkspaceBackend()
    backend.write_text("normal.txt", "needle " + "a" * 29_000 + "!")
    result = search_files(
        ToolContext(tmp_path, {}, 0, backend),
        {"path": "normal.txt", "pattern": "needle|(a+)+$", "output_mode": "content", "multiline": multiline},
    )
    assert result.metadata["regex_timeout"] == {"path": "normal.txt", "line": 1, "column": 0}
    assert "Regular expression too slow; aborted" in result.content
    assert "needle" in result.content and result.metadata["summary"]["total_matches"] == 1


@pytest.mark.parametrize(
    "pattern", ["(?i)NEEDLE", "(?i:NEEDLE)", "(?<=prefix)needle", r"(needle)\1", "^prefixneedle$", r"\bneedle\b"]
)
def test_normal_regex_syntax_matches_re_v0(tmp_path, pattern):
    import re

    text = "prefixneedle\nneedleneedle\nneedle"
    result, _, _ = _search_text(tmp_path, text, pattern, case_sensitive=True)
    expected = [(number, m.start()) for number, line in enumerate(text.splitlines(), 1) for m in re.finditer(pattern, line)]
    assert [(m["line"], m["column"]) for m in result.metadata["artifact_search"]["matches"]] == expected


@pytest.mark.parametrize(
    "field,value",
    [("start_char", -1), ("start_char", True), ("start_char", "1")],
)
def test_character_page_argument_validation(tmp_path, field, value):
    _, context, ref = _search_text(tmp_path, "needle", "needle")
    result = read_file(context, {"path": ref.path, field: value})
    assert result.error_code == "invalid_arguments"


def test_character_pages_return_cursor_next_line_and_model_notice(tmp_path):
    _, context, ref = _search_text(tmp_path, "\nabc\n" + "d" * 40_000, "abc")
    page = read_file(context, {"path": ref.path, "start_line": 2, "start_char": 1})
    assert page.content.startswith("bc\n" + "d" * 100)
    assert page.metadata["read_page"] == {"next_line": 3, "next_start_char": 11_997}
    assert "start_line=3, start_char=11997" in page.to_tool_message().content
    assert page.cursor is not None
    rest = read_file(context, {"path": ref.path, "cursor": page.cursor.to_dict()})
    assert rest.metadata["read_page"] == {"next_line": 3, "next_start_char": 23_997}
    assert rest.cursor is not None
    assert (
        read_file(context, {"path": ref.path, "cursor": page.cursor.to_dict(), "start_char": 1}).error_code == "invalid_arguments"
    )


def test_offset_read_never_grants_full_edit_baseline(tmp_path):
    from vv_agent.tools.handlers.workspace_io import write_file

    backend = MemoryWorkspaceBackend()
    backend.write_text("normal.txt", "prefixneedle")
    context = ToolContext(tmp_path, {}, 0, backend)
    page = read_file(context, {"path": "normal.txt", "start_char": 6})
    assert page.content == "needle"
    assert write_file(context, {"path": "normal.txt", "content": "replacement"}).error_code == "file_not_read"


@pytest.mark.parametrize("session,task,allowed", [("sid", "task", True), ("other", "task", False), ("sid", "other", False)])
def test_host_task_and_sdk_compaction_identity_remain_distinct(tmp_path, session, task, allowed):
    from vv_agent import is_task_artifact_path

    class HostBackend(MemoryWorkspaceBackend):
        task_id = "host-task"

    backend = HostBackend()
    ref = persist_text_artifact(backend, _session_artifact_scope(session, task), "call", "needle", reuse_existing=True)
    context = ToolContext(tmp_path, {}, 0, backend, task_id="task", metadata={"session_id": "sid"})
    assert is_task_artifact_path(ref.path, task_id="host-task", session_id="sid", session_task_id="task") is allowed
    assert (read_file(context, {"path": ref.path}).error_code is None) is allowed
    assert (file_info(context, {"path": ref.path}).error_code is None) is allowed
    assert (search_files(context, {"path": ref.path, "pattern": "needle"}).error_code is None) is allowed


def test_task_scope_encoding_cannot_alias_raw_or_transformed_id(tmp_path):
    from vv_agent import is_task_artifact_path
    from vv_agent.workspace.artifacts import _artifact_task_segment

    for original, other in [("task/a", "task-a"), ("a" * 64 + "x", "a" * 64 + "y"), ("", "task")]:
        ref = persist_text_artifact(MemoryWorkspaceBackend(), original, "call", "needle", reuse_existing=True)
        assert is_task_artifact_path(ref.path, task_id=original)
        assert not is_task_artifact_path(ref.path, task_id=other)
        assert not is_task_artifact_path(ref.path, task_id=_artifact_task_segment(original))


def test_character_offset_read_memory_is_independent_of_line_length(tmp_path, record_property):
    import tracemalloc

    from vv_agent.workspace import LocalWorkspaceBackend

    backend = LocalWorkspaceBackend(tmp_path / "ws")
    ref = persist_text_artifact(backend, "task", "call", "漢🙂" * 500_000 + "needle", reuse_existing=True)
    tracemalloc.start()
    result = read_file(
        ToolContext(tmp_path, {}, 0, backend, task_id="task"),
        {"path": ref.path, "start_line": 1, "end_line": 1, "start_char": 1_000_000},
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    record_property("read_peak_bytes", peak)
    assert peak < 2_000_000
    assert result.content == "needle"


def test_kernel_retention_failure_commits_success_without_replaying_command(tmp_path):
    import shlex
    from contextlib import nullcontext

    from vv_agent import Agent, RunConfig
    from vv_agent.config import ResolvedModelConfig
    from vv_agent.llm.scripted import ScriptedLLM
    from vv_agent.session.kernel import drive
    from vv_agent.session.records import InboxItem, SessionSpec
    from vv_agent.session.runtime import Runtime
    from vv_agent.session.sqlite import SQLiteStore
    from vv_agent.types import LLMResponse

    class FailingBackend(MemoryWorkspaceBackend):
        task_id = "host-task"

        def write_text_chunks_exclusive(self, path, chunks):
            raise OSError("disposable writer failure")

    from vv_agent.tools import ToolRegistry

    def registry():
        selected = ToolRegistry()
        selected.register_executor(build_default_registry().get_executor("bash"))
        return selected

    backend = FailingBackend()
    counter = tmp_path / "effect-counter"
    program = f"from pathlib import Path; Path({str(counter)!r}).open('a').write('effect\\n'); print('x'*12001, end='')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("retention", "test", str(tmp_path)), consumers=())
            tx.push("retention", InboxItem("initial", "user", {"content": "run once"}))
        runtime = Runtime(
            agent=Agent("capture", "Run the command once."),
            config=RunConfig(
                workspace=tmp_path, workspace_backend=backend, session_memory_enabled=False, tool_registry_factory=registry
            ),
            resolved=ResolvedModelConfig("scripted", "scripted", "scripted", "scripted", []),
            llm=ScriptedLLM([LLMResponse("", [ToolCall("capture", "bash", {"command": command})]), LLMResponse("done")]),
            heartbeat_store=lambda: nullcontext(store),
        )
        drive(store, "retention", runtime=runtime)
        before = [row.record for row in store.read("retention", limit=100).records]
        result = next(r.payload["result"] for r in before if r.kind == "op_completed" and r.payload["result"].get("tool_call_id"))
        assert result["status_code"] == "SUCCESS" and result["directive"] == "continue", result
        assert "error_code" not in result and result["metadata"]["exit_code"] == 0
        assert result["metadata"]["retention"]["status"] == "unavailable"
        assert "app_server_resources" not in result["metadata"]
        assert "vv_artifact" not in result["metadata"]
        assert "artifact" not in result
        assert "complete text could not be saved" in result["content"]
        drive(store, "retention", runtime=runtime)
        assert [r.encode() for r in before] == [row.record.encode() for row in store.read("retention", limit=100).records]
        assert counter.read_text() == "effect\n"


@pytest.mark.parametrize("reader", [read_file, file_info])
def test_foreign_scope_and_path_aliases_never_touch_storage(tmp_path, reader):
    class NoAccess(MemoryWorkspaceBackend):
        def is_file(self, path):
            raise AssertionError("denied read touched storage")

        def file_info(self, path):
            raise AssertionError("denied stat touched storage")

    backend = MemoryWorkspaceBackend()
    ref = persist_text_artifact(backend, _session_artifact_scope("other", "task"), "call", "needle", reuse_existing=True)
    context = ToolContext(tmp_path, {}, 0, NoAccess(), task_id="task", metadata={"session_id": "sid"})
    for path in (ref.path, "./" + ref.path, "/" + ref.path, ref.path.replace("/", "\\"), "x/../" + ref.path):
        assert reader(context, {"path": path}).error_code == "artifact_read_denied"


def test_real_microcompaction_producer_can_read_its_own_evidence(tmp_path):
    from vv_agent.memory import MemoryManager
    from vv_agent.microcompaction import MicrocompactionPolicy
    from vv_agent.types import Message

    backend = MemoryWorkspaceBackend()
    manager = MemoryManager(
        compact_threshold=1000,
        model="unknown-provider-model",
        model_context_window=1000,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        workspace_backend=backend,
        recovery_tool_available=True,
        artifact_scope=_session_artifact_scope("sid", "task"),
        microcompaction_policy=MicrocompactionPolicy(keep_recent_cycles=1, min_result_chars=500),
    )
    messages = [
        Message("user", "read evidence"),
        Message(
            "assistant", "", tool_calls=[{"id": "old", "type": "function", "function": {"name": "source", "arguments": "{}"}}]
        ),
        Message("tool", "needle " * 2000, tool_call_id="old"),
        Message("assistant", "recent"),
    ]
    plan = manager.plan_microcompaction(messages, cycle_index=4, current_tokens=1000)
    assert plan is not None
    compacted = manager.apply_microcompaction(messages, plan=plan)
    assert compacted.archived_count == 1
    ref = compacted.messages[2].artifact_ref
    assert ref is not None
    context = ToolContext(tmp_path, {}, 0, backend, task_id="task", metadata={"session_id": "sid"})
    found = search_files(context, {"path": ref.path, "pattern": "needle", "output_mode": "content", "head_limit": 1})
    assert found.metadata["artifact_search"]["matches"][0]["column"] == 0
    assert read_file(context, {"path": ref.path, "start_line": 1, "start_char": 0}).content.startswith("needle")


def test_live_capture_retention_failure_preserves_running_result(tmp_path):
    import json
    import shlex

    from vv_agent.runtime.background_sessions import background_session_manager

    class FailingBackend(MemoryWorkspaceBackend):
        def write_text_chunks_exclusive(self, path, chunks):
            raise OSError("disposable writer failure")

    context = ToolContext(tmp_path, {}, 0, FailingBackend(), task_id="task")
    registry = build_default_registry()
    program = "import time; print('x'*12001, end='', flush=True); time.sleep(5)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"
    started = registry.execute(ToolCall("start", "bash", {"command": command, "yield_time_ms": 0}), context)
    assert isinstance(started, ToolExecutionResult)
    sid = json.loads(started.content)["session_id"]
    try:
        deadline = time.monotonic() + 4
        while True:
            result = registry.execute(ToolCall("check", "check_background_command", {"session_id": sid}), context)
            assert isinstance(result, ToolExecutionResult)
            assert result.status_code is ToolResultStatus.SUCCESS, result.to_dict()
            if "retention" in result.metadata:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert result.directive.value == "continue" and result.error_code is None
        assert result.metadata["status"] == "running"
        assert result.metadata["retention"]["status"] == "unavailable"
        assert "complete text could not be saved" in json.loads(result.content)["output"]
    finally:
        background_session_manager.stop_for_tool(
            sid, context.workspace_backend, context.task_id, "stop", workspace=context.workspace
        )


def test_fixed_character_page_keeps_line_numbers_and_advances(tmp_path):
    _, context, ref = _search_text(tmp_path, "a" * 40_000, "a", head_limit=1)
    first = read_file(context, {"path": ref.path, "show_line_numbers": True})
    assert first.cursor is not None
    assert first.content.startswith("1: aaa") and first.cursor.offset_chars == 11_997
    second = read_file(context, {"path": ref.path, "show_line_numbers": True, "cursor": first.cursor.to_dict()})
    assert second.cursor is not None
    assert second.content.startswith("1: aaa") and second.cursor.offset_chars == 23_994


@pytest.mark.parametrize("offset,column", [(0, 0), (1, 7), (2, 14)])
def test_search_paginates_matches_on_the_same_line(tmp_path, offset, column):
    result, _, _ = _search_text(tmp_path, "needle needle needle", "needle", offset=offset, head_limit=1)
    assert [m["column"] for m in result.metadata["artifact_search"]["matches"]] == [column]


@pytest.mark.parametrize("pattern,columns", [("needle$", []), (r"\r$", [6]), (r"^needle\r$", [0])])
def test_eof_carriage_return_is_preserved(tmp_path, pattern, columns):
    result, _, _ = _search_text(tmp_path, "needle\r", pattern)
    assert [m["column"] for m in result.metadata["artifact_search"]["matches"]] == columns


def test_workspace_timeout_notice_survives_bounded_model_projection(tmp_path):
    backend = MemoryWorkspaceBackend()
    backend.write_text("normal.txt", "needle " + "a" * 60_000 + "!")
    result = search_files(
        ToolContext(tmp_path, {}, 0, backend), {"path": "normal.txt", "pattern": "needle|(a+)+$", "output_mode": "content"}
    )
    assert result.metadata["regex_timeout"]
    assert result.content.startswith("Regular expression too slow; aborted")
    assert "Regular expression too slow; aborted" in result.to_tool_message().content


def test_open_read_page_metadata_does_not_crash_projection():
    from vv_agent.types import ToolResultCursor

    result = ToolExecutionResult(
        "read",
        "a",
        truncated=True,
        truncation_reason="read_limit",
        original_bytes=3,
        visible_bytes=1,
        cursor=ToolResultCursor("read_file", "a.txt", 1, "a" * 64),
        metadata={"read_page": {}},
    )
    assert result.to_tool_message().content.startswith("a\n")


def test_partial_capture_object_does_not_block_retention_retry(tmp_path):
    from vv_agent.workspace.artifacts import persist_captured_text_artifact

    class PartialWriter(MemoryWorkspaceBackend):
        fail = True

        def write_text_chunks_exclusive(self, path, chunks):
            if self.fail:
                super().write_text_chunks_exclusive(path, [next(iter(chunks))[:100]])
                raise OSError("disposable partial write")
            return super().write_text_chunks_exclusive(path, chunks)

    backend = PartialWriter()
    body = "head" * 4000 + "tail"
    capture = tmp_path / "capture.txt"
    capture.write_text(body)
    with pytest.raises(OSError, match="disposable partial write"):
        persist_captured_text_artifact(backend, "task", "call", capture)
    damaged = next(iter(backend._files))
    before = backend.read_text(damaged)
    backend.fail = False
    recovered = persist_captured_text_artifact(backend, "task", "call", capture)
    assert recovered.path != damaged and backend.read_text(recovered.path) == body
    assert backend.read_text(damaged) == before


def test_capture_read_failure_after_retention_failure_preserves_success(tmp_path, monkeypatch):
    from vv_agent.workspace import artifacts

    from .test_tool_result_artifacts import PRODUCER

    original = artifacts._iter_captured_text_chunks
    reads = []

    def controlled(path):
        reads.append(path)
        if len(reads) > 1:
            raise OSError("disposable capture read failure")
        yield from original(path)

    class FailingWriter(MemoryWorkspaceBackend):
        def write_text_chunks_exclusive(self, path, chunks):
            raise OSError("disposable writer failure")

    monkeypatch.setattr(artifacts, "_iter_captured_text_chunks", controlled)
    result = PRODUCER["terminal_capture_result"](FailingWriter(), "H" * 6000 + "middle" * 1000 + "TAIL" * 1500 + "last line")
    assert result.status_code is ToolResultStatus.SUCCESS and result.error_code is None
    assert result.directive.value == "continue" and result.metadata["exit_code"] == 0
    assert result.metadata["retention"]["status"] == "unavailable"
    assert len(reads) == 1
    assert result.content.endswith("TAIL" * 10 + "last line")


def test_write_then_error_reports_unavailable_without_expected_pointer(tmp_path):
    from .test_tool_result_artifacts import PRODUCER

    class LateErrorWriter(MemoryWorkspaceBackend):
        def write_text_chunks_exclusive(self, path, chunks):
            super().write_text_chunks_exclusive(path, chunks)
            raise OSError("disposable error after completed write")

    backend = LateErrorWriter()
    result = PRODUCER["terminal_capture_result"](backend, "x" * 12001)
    assert result.status_code is ToolResultStatus.SUCCESS and result.error_code is None
    assert result.metadata["retention"]["status"] == "unavailable"
    assert result.artifact is None
    assert all(backend.read_text(path) == "x" * 12001 for path in backend._files)


def test_session_and_task_separators_cannot_alias_each_other(tmp_path):
    from vv_agent import is_task_artifact_path

    left = _session_artifact_scope("a", "b/c")
    right = _session_artifact_scope("a/b", "c")
    assert left != right
    ref = persist_text_artifact(MemoryWorkspaceBackend(), left, "call", "needle", reuse_existing=True)
    assert is_task_artifact_path(ref.path, task_id="b/c", session_id="a")
    assert not is_task_artifact_path(ref.path, task_id="c", session_id="a/b")


def test_writer_failure_does_not_prescan_or_reread_capture(tmp_path, monkeypatch):
    from vv_agent.workspace import artifacts

    capture = tmp_path / "capture.txt"
    capture.write_text("x" * 12001)
    reads = []
    original = artifacts._iter_captured_text_chunks

    def tracked(path):
        reads.append(path)
        yield from original(path)

    class FailedWriter(MemoryWorkspaceBackend):
        def write_text_chunks_exclusive(self, path, chunks):
            raise OSError("disposable writer failure")

    monkeypatch.setattr(artifacts, "_iter_captured_text_chunks", tracked)
    with pytest.raises(OSError, match="disposable writer failure"):
        artifacts.persist_captured_text_artifact(FailedWriter(), "task", "call", capture)
    assert reads == []
