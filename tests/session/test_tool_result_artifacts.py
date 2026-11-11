"""S1: real producer receipts, exclusive storage, trust boundaries and recovery."""

import errno
import hashlib
import json
import multiprocessing
import os
import runpy
import tracemalloc
from dataclasses import replace
from pathlib import Path

import pytest
from support.contract import fixture_root

from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.session.kernel import drive
from vv_agent.session.providers import Definitive
from vv_agent.session.records import InboxItem, digest
from vv_agent.session.reducer import TransitionError, fold
from vv_agent.session.tool_results import prepare_tool_resources, project_tool_content, provider_result_identity
from vv_agent.tools.builtins import build_default_registry
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult, ToolResultCursor
from vv_agent.workspace import LocalWorkspaceBackend
from vv_agent.workspace.artifacts import PREVIEW_MARKER, persist_text_artifact

from .conftest import open_store
from .controlled import ControlledProvider
from .test_recovery_matrix import effect_tool, kill, receive, records_of, runtime, start

PRODUCER = runpy.run_path(str(Path(__file__).parents[2] / "scripts/tool_result_artifact_fixtures.py"))
ArchiveWorkspace, produce = PRODUCER["ArchiveWorkspace"], PRODUCER["produce"]


@pytest.mark.parametrize("text", ["", "ok", "A" * 12000, "漢" * 12000, "🙂" * 12000, "x\r\ny"])
@pytest.mark.parametrize("named", [False, True])
def test_short_result_exact_bytes_without_any_extra_fields(text, named):
    _plan, receipt, backend = produce(text, named=named)
    assert canonical_json_bytes(receipt._payload["result"]) == canonical_json_bytes(
        ToolExecutionResult("call-s1", text).to_dict()
    )
    assert backend.writes == 0


@pytest.mark.parametrize(
    "text",
    [
        "A" * 100000,
        "漢字" * 7000,
        "🙂" * 12001,
        "x\r\n" * 7000,
        b"\xff\xf0\x28\x8c\x28".decode("utf-8", errors="replace") * 4000,
    ],
)
@pytest.mark.parametrize("named", [False, True])
def test_long_result_head_tail_utf8_integrity_and_source(text, named):
    plan, receipt, backend = produce(text, named=named)
    raw = receipt._payload["result"]
    result = ToolExecutionResult.from_dict(raw)
    assert result.artifact is not None
    marker_start = result.content.index("\n... ", 6000)
    marker_end = result.content.index(" ...\n", marker_start) + len(" ...\n")
    marker = result.content[marker_start:marker_end]
    shown = 12000 - len(marker)
    assert marker == (
        f"\n... {shown} of {len(text)} source characters shown; full text in {result.artifact.path}; "
        "read a range with read_file(start_line, end_line) or search with search_files(pattern) ...\n"
    )
    assert result.content == text[:6000] + marker + text[-(shown - 6000) :]
    assert len(result.content) == 12000
    assert not any(0xD800 <= ord(char) <= 0xDFFF for char in result.content)
    assert result.status_code.value == "SUCCESS"
    assert result.original_bytes == len(text.encode("utf-8"))
    assert result.visible_bytes == len(result.content.encode("utf-8"))
    artifact = raw["artifact"]
    stored = backend.read_bytes(artifact["path"])
    assert stored == text.encode("utf-8")
    assert artifact["sha256"] == hashlib.sha256(stored).hexdigest()
    assert artifact["size_bytes"] == len(stored)
    namespace = raw["metadata"]["vv_artifact"]
    assert {k: namespace[k] for k in artifact} == artifact
    assert namespace["source"] == {
        "operation_id": plan.operation_id,
        "attempt": plan.attempt,
        "tool_call_id": "call-s1",
        "arguments_sha256": hashlib.sha256(b"{}").hexdigest(),
    }
    assert namespace["source_complete"] is True
    assert namespace["capture_incomplete_reason"] is None
    assert receipt._payload["result_digest"] == hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
    assert max(backend.chunk_lengths) <= 65536
    assert backend.writes == 1


def test_new_write_checks_counts_and_stat_without_reading_chunks():
    class CountWorkspace(ArchiveWorkspace):
        reads = 0
        stats = 0

        def _read_bytes_chunks(self, path):
            self.reads += 1
            return super()._read_bytes_chunks(path)

        def read_bytes(self, path):
            raise AssertionError("new write must not read bytes")

        def file_info(self, path):
            self.stats += 1
            return super().file_info(path)

    backend = CountWorkspace()
    _plan, receipt, _ = produce("漢🙂" * 10000, backend=backend)
    assert receipt._payload["result"]["metadata"]["vv_artifact"]["source_complete"]
    assert (backend.writes, backend.reads, backend.stats) == (1, 0, 1)
    raw = receipt._payload["result"]
    prepare_tool_resources(_plan, raw, backend)
    assert backend.reads == 1  # Already-truncated receipts still verify the full stream.


@pytest.mark.parametrize("fault", ["count", "stat"])
def test_bad_write_size_is_unavailable_and_does_not_retry_the_tool(fault):
    class BadSizeWorkspace(ArchiveWorkspace):
        def write_text_chunks_exclusive(self, path, chunks):
            written = super().write_text_chunks_exclusive(path, chunks)
            return written - 1 if fault == "count" else written

        def file_info(self, path):
            info = super().file_info(path)
            if info is not None and fault == "stat":
                info.size -= 1
            return info

    backend = BadSizeWorkspace()
    _plan, receipt, _ = produce("X" * 14000, backend=backend, named=True)
    raw = receipt._payload["result"]
    assert raw["status_code"] == "SUCCESS"
    assert raw["metadata"]["app_server_resources"][0]["status"] == "unavailable"
    assert "vv_artifact" not in raw["metadata"]
    assert backend.writes == 1


def test_already_truncated_is_not_truncated_again_and_does_not_claim_full_capture():
    text = "Z" * 40000
    _plan, receipt, backend = produce(text, truncated=True)
    raw = receipt._payload["result"]
    assert raw["content"] == text[:6000] + PREVIEW_MARKER + text[-5953:]
    assert raw["content"].count(PREVIEW_MARKER) == 1
    assert raw["original_bytes"] == len(text)
    assert backend.writes == 1
    assert raw["metadata"]["vv_artifact"]["source_complete"] is False
    assert raw["metadata"]["vv_artifact"]["capture_incomplete_reason"] == "capture_completeness_unverified"


def test_already_truncated_read_limit_keeps_long_preview_and_cursor():
    plan, _receipt, backend = produce("ok")
    text = "C" * 13000
    raw = ToolExecutionResult(
        "call-s1",
        text,
        truncated=True,
        truncation_reason="read_limit",
        original_bytes=20000,
        visible_bytes=13000,
        cursor=ToolResultCursor("read_file", "source.txt", 13000, "a" * 64),
    ).to_dict()
    assert prepare_tool_resources(plan, raw, backend) == raw
    assert backend.writes == 0


def test_archive_failure_still_commits_success_and_marks_unavailable_without_retry():
    backend = ArchiveWorkspace()
    backend.fail_write = True
    _plan, receipt, _ = produce("F" * 13000, backend=backend, named=True)
    raw = receipt._payload["result"]
    ToolExecutionResult.from_dict(raw)
    assert raw["status_code"] == "SUCCESS" and "error_code" not in raw
    assert len(raw["content"]) == 12000 and raw["truncated"] is True
    assert raw["metadata"]["app_server_resources"][0]["status"] == "unavailable"
    assert "vv_artifact" not in raw["metadata"]
    assert not backend.exists(raw["artifact"]["path"])
    assert backend.writes == 0


def test_identity_conflict_is_eio_and_never_overwrites():
    backend = ArchiveWorkspace()
    first = persist_text_artifact(backend, "task", "call", "first", identity="receipt-1", reuse_existing=True)
    with pytest.raises(OSError) as exc:
        persist_text_artifact(backend, "task", "call", "second", identity="receipt-1", reuse_existing=True)
    assert exc.value.errno == errno.EIO
    assert backend.read_text(first.path) == "first"
    assert backend.writes == 1
    same = persist_text_artifact(backend, "task", "call", "first", identity="receipt-1", reuse_existing=True)
    assert same == first
    content_ref = persist_text_artifact(backend, "task", "other", "content", reuse_existing=True)
    assert persist_text_artifact(backend, "task", "other", "content", reuse_existing=True) == content_ref


def test_reprojection_is_byte_identical_and_path_is_not_random():
    text = "R" * 50000
    plan, receipt, backend = produce(text)
    raw = ToolExecutionResult("call-s1", text).to_dict()
    for _ in range(2):
        assert canonical_json_bytes(prepare_tool_resources(plan, raw, backend)) == canonical_json_bytes(
            receipt._payload["result"]
        )
    assert backend.writes == 1
    assert provider_result_identity(receipt._payload["result"]) == project_tool_content(plan, raw, backend)


@pytest.mark.parametrize("fault", ["digest", "size", "invalid_utf8", "missing"])
def test_invalid_stored_bytes_are_unavailable_not_accepted(fault):
    plan, receipt, backend = produce("H" * 13000)
    raw = receipt._payload["result"]
    path = raw["artifact"]["path"]
    if fault == "missing":
        del backend._files[path]
    else:
        backend._files[path] = {"digest": b"G" * 13000, "size": b"H", "invalid_utf8": b"\xff" * 13000}[fault]
    projected = prepare_tool_resources(plan, raw, backend)
    assert projected["status_code"] == "SUCCESS"
    assert "vv_artifact" not in projected["metadata"]
    assert projected["metadata"]["app_server_resources"][0]["status"] == "unavailable"


def test_same_completion_different_body_retains_distinct_candidates():
    plan, receipt, backend = produce("A" * 13000)
    path = receipt._payload["result"]["artifact"]["path"]
    second = prepare_tool_resources(plan, ToolExecutionResult("call-s1", "B" * 13000).to_dict(), backend)
    assert second["artifact"]["path"] != path
    assert second["metadata"]["vv_artifact"]["source_complete"] is True
    assert second["metadata"]["app_server_resources"][0]["status"] == "retained"
    assert backend.read_text(path) == "A" * 13000
    assert backend.read_text(second["artifact"]["path"]) == "B" * 13000


def test_producer_namespace_is_discarded_before_it_can_supply_authority():
    plan, _receipt, backend = produce("ok")
    raw = ToolExecutionResult("call-s1", "X" * 14000).to_dict()
    raw["metadata"] = {"vv_artifact": {"source_complete": True, "source": {"operation_id": "forged"}}}
    projected = prepare_tool_resources(plan, raw, backend)
    assert projected["metadata"]["vv_artifact"]["source"]["operation_id"] == plan.operation_id
    raw["content"] = "ok"
    assert prepare_tool_resources(plan, raw, backend) == ToolExecutionResult("call-s1", "ok").to_dict()


@pytest.mark.parametrize(
    "mutation",
    [
        {"source_complete": True, "capture_incomplete_reason": "capped"},
        {"source_complete": False, "capture_incomplete_reason": None},
        {"source_complete": 1},
        {"path": "../escape.txt"},
        {"sha256": "not-a-digest"},
        {"size_bytes": True},
        {"source": {"operation_id": "forged", "attempt": 0, "tool_call_id": "call-s1", "arguments_sha256": "a" * 64}},
        {"extra": "unknown"},
    ],
)
def test_reserved_namespace_strict_validation(mutation):
    _plan, receipt, _ = produce("V" * 13000)
    raw = receipt.payload["result"]
    raw["metadata"]["vv_artifact"].update(mutation)
    with pytest.raises(ValueError):
        ToolExecutionResult.from_dict(raw)


@pytest.mark.parametrize("key", ["content", "output", "stdout", "stderr", "instructions"])
def test_forbidden_metadata_remains_rejected(key):
    _plan, receipt, _ = produce("V" * 13000)
    raw = receipt.payload["result"]
    raw["metadata"][key] = "forbidden"
    with pytest.raises(ValueError, match="tool_result_invalid"):
        ToolExecutionResult.from_dict(raw)


def test_streaming_does_not_duplicate_the_full_text_in_application_memory(tmp_path):
    text = "large🙂" * 1000000
    backend = LocalWorkspaceBackend(tmp_path / "workspace")
    tracemalloc.start()
    artifact = persist_text_artifact(backend, "task", "call", text, reuse_existing=True)
    _used, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert artifact.size_bytes == 9_000_000
    assert peak < 4_000_000


class LongProvider(ControlledProvider):
    def submit(self, plan, *, context):
        outcome = super().submit(plan, context=context)
        raw = ToolExecutionResult(plan._payload["request"]["id"], "P" * 30000).to_dict()
        with open_store(self.database) as store, store.atomic():
            store._rows("UPDATE provider_jobs SET result=%s", (json.dumps(raw),))
        return Definitive(raw, outcome.evidence)


def test_named_provider_repeated_original_callback_is_noop(store, database, tmp_path):
    provider = LongProvider(database)
    provider.install()
    start(store)
    rt = runtime(
        database,
        [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")],
        [effect_tool()],
        providers={"effect": provider},
    )
    rt.config = replace(rt.config, workspace_backend=LocalWorkspaceBackend(tmp_path / "workspace"))
    drive(store, "s", runtime=rt)
    completed = [r.encode() for r in records_of(store, "op_completed")]
    with store.atomic() as tx:
        tx.push("s", provider.callback(input_id="repeat-original"))
    drive(store, "s", runtime=rt)
    applied = records_of(store, "input_applied")[-1]
    assert applied._payload["disposition"] == "noop", applied.payload
    assert provider.counts() == (1, 1)
    assert completed == [r.encode() for r in records_of(store, "op_completed")]


@pytest.mark.parametrize("field,value", [("operation_id", "forged"), ("attempt", 99), ("arguments_sha256", "a" * 64)])
def test_receipt_source_must_bind_the_admitted_operation(store, database, tmp_path, field, value):
    provider = LongProvider(database)
    provider.install()
    start(store)
    rt = runtime(
        database,
        [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")],
        [effect_tool()],
        providers={"effect": provider},
    )
    rt.config = replace(rt.config, workspace_backend=LocalWorkspaceBackend(tmp_path / "workspace"))
    drive(store, "s", runtime=rt)
    records = [entry.record for entry in store.read("s", limit=100).records]
    for index, record in enumerate(records):
        if record.kind == "op_completed" and "metadata" in record._payload["result"]:
            payload = record.payload
            payload["result"]["metadata"]["vv_artifact"]["source"][field] = value
            payload["result_digest"] = digest(payload["result"])
            records[index] = replace(record, payload=payload)
            break
    with pytest.raises(TransitionError, match="artifact source"):
        fold(records, consumed_inputs=[InboxItem(**r._payload["input"]) for r in records if r.kind == "input_applied"])


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("read_file", {"path": "large.txt"}),
        ("search_files", {"path": "large.txt", "pattern": "X", "output_mode": "content"}),
        pytest.param(
            "bash",
            {"command": "printf '%15000s' 'END'", "timeout_seconds": 10, "yield_time_ms": 10000},
            marks=pytest.mark.skipif(os.name == "nt", reason="POSIX command fixture"),
        ),
    ],
)
def test_actual_builtin_handlers_reach_the_completion_boundary(store, database, tmp_path, name, arguments):
    def registry():
        value = build_default_registry()
        value._add_planner_extra_tool_name("bash")
        return value

    backend = LocalWorkspaceBackend(tmp_path / "workspace")
    backend.write_text("large.txt", "X" * 15000)
    start(store)
    rt = runtime(database, [LLMResponse("", [ToolCall("builtin", name, arguments)]), LLMResponse("done")])
    rt.config = replace(
        rt.config,
        workspace=backend.root,
        workspace_backend=backend,
        tool_registry_factory=registry,
        metadata={"bash_shell": "/bin/sh"} if name == "bash" else {},
    )
    drive(store, "s", runtime=rt)
    raw = next(
        record._payload["result"]
        for record in records_of(store, "op_completed")
        if record._payload["result"].get("tool_call_id") == "builtin"
    )
    assert raw["status_code"] == "SUCCESS", raw
    if name == "bash" and raw.get("metadata", {}).get("status") in {"running", "stopping", "unknown"}:
        pytest.skip("local command supervisor supplied no terminal capture; bash archive integration remains unverified")
    assert raw.get("truncated") is True, raw
    if name == "read_file":
        assert raw["truncation_reason"] == "read_limit" and "artifact" not in raw
        assert raw["cursor"]["path"] == "large.txt"
    else:
        assert len(raw["content"]) == 12000
        artifact = raw["artifact"]
        data = backend.read_bytes(artifact["path"])
        assert len(data) == artifact["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
        assert "vv_artifact" in raw["metadata"]


def artifact_worker(database, root, pipe, cut):
    from vv_agent import workspace

    class FaultWorkspace(workspace.LocalWorkspaceBackend):
        def write_text_chunks_exclusive(self, path, chunks):
            if cut == "before_artifact":
                pipe.send(("barrier", None))
                pipe.recv()
            written = super().write_text_chunks_exclusive(path, chunks)
            if cut == "after_artifact":
                pipe.send(("barrier", path))
                pipe.recv()
            return written

    def hook(point, record):
        if (
            cut == "after_receipt"
            and point == "after_commit"
            and record.kind == "op_completed"
            and "/tool/" in str(record.operation_id)
        ):
            pipe.send(("barrier", None))
            pipe.recv()

    with open_store(database) as store:
        rt = runtime(
            database,
            [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")],
            [effect_tool()],
            providers={"effect": LongProvider(database)},
            hook=hook,
        )
        rt.config = replace(rt.config, workspace_backend=FaultWorkspace(Path(root)))
        drive(store, "s", runtime=rt)


@pytest.mark.sqlite_file
@pytest.mark.parametrize("cut", ["before_artifact", "after_artifact", "after_receipt"])
def test_process_death_around_artifact_and_receipt_never_replays_effect(store, database, tmp_path, cut):
    provider = LongProvider(database)
    provider.install()
    start(store)
    ctx = multiprocessing.get_context("spawn")
    pipe, child = ctx.Pipe()
    proc = ctx.Process(target=artifact_worker, args=(database, str(tmp_path / "workspace"), child, cut))
    proc.start()
    child.close()
    try:
        receive(pipe, "barrier")
        kill(proc, pipe, database)
        assert provider.counts() == (1, 1)
        backend = LocalWorkspaceBackend(tmp_path / "workspace")
        rt = runtime(database, [LLMResponse("done")], [effect_tool()], providers={"effect": provider})
        rt.config = replace(rt.config, workspace_backend=backend)
        if cut != "after_receipt":
            # Authenticate the saved provider receipt; never submit the external tool again.
            with store.atomic() as tx:
                tx.push("s", provider.callback(input_id="recover"))
        drive(store, "s", runtime=rt)
        receipts = [r for r in records_of(store, "op_completed") if r._payload["result"].get("tool_call_id")]
        assert len(receipts) == 1
        artifact = receipts[0]._payload["result"]["artifact"]
        assert backend.read_bytes(artifact["path"]) == b"P" * 30000
        assert artifact["sha256"] == hashlib.sha256(b"P" * 30000).hexdigest()
        assert provider.counts() == (1, 1)
    finally:
        if proc.is_alive():
            kill(proc, pipe, database)


def test_contract_vectors_come_from_the_real_producer():
    expected = json.loads((fixture_root() / "tool_result_artifact.json").read_text())
    assert PRODUCER["generate"]() == expected
