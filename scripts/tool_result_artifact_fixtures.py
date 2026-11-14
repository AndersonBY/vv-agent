"""Deterministic S1 vectors from real session completions and workspace writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from vv_agent import Agent, RunConfig, function_tool
from vv_agent.config import ResolvedModelConfig
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.session.kernel import drive
from vv_agent.session.providers import Definitive
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.runtime import Runtime
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.tool_results import prepare_tool_resources
from vv_agent.tools.base import ToolContext
from vv_agent.tools.builtins import build_default_registry
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult
from vv_agent.workspace import MemoryWorkspaceBackend
from vv_agent.workspace.artifacts import bounded_text_preview, persist_text_artifact


class ArchiveWorkspace(MemoryWorkspaceBackend):
    fail_write = False

    def __init__(self):
        super().__init__()
        self.writes = 0
        self.chunk_lengths = []

    @property
    def retained_workspace_id(self):
        return hashlib.sha256(b"s1-fixture-workspace").hexdigest()

    def write_text_chunks_exclusive(self, path, chunks):
        if self.fail_write:
            raise OSError("disposable artifact storage unavailable")

        def measured():
            for chunk in chunks:
                self.chunk_lengths.append(len(chunk))
                yield chunk

        written = super().write_text_chunks_exclusive(path, measured())
        self.writes += 1
        return written


def terminal_capture_result(backend, text):
    from vv_agent.runtime.background_sessions import BackgroundSessionManager

    class CompletedProcess:
        returncode = 0

        def poll(self):
            return 0

    with TemporaryDirectory(prefix="capture-fixture-") as directory:
        root = Path(directory)
        capture = root / "capture.txt"
        capture.write_text(text)
        started = SimpleNamespace(process=CompletedProcess(), output_path=capture, started_at=time.monotonic())
        with (
            patch("vv_agent.tools.handlers.bash.start_captured_process", return_value=started),
            patch("vv_agent.tools.handlers.bash.background_session_manager", BackgroundSessionManager()),
            patch("vv_agent.runtime.background_sessions.process_tree_is_running", return_value=False),
        ):
            return build_default_registry().execute(
                ToolCall("call-s1", "bash", {"command": "printf fixture"}),
                ToolContext(root, {}, 0, backend, task_id="s1", tool_call_id="call-s1"),
            )


def produce(text, *, backend=None, truncated=False, named=False, forged=False, terminal=False):
    backend = backend if backend is not None else ArchiveWorkspace()
    effects = []

    def result():
        effects.append("effect")
        if terminal:
            value = terminal_capture_result(backend, text)
            value.tool_call_id = "call-s1"
            return value
        if truncated:
            artifact = persist_text_artifact(backend, "s1", "existing", text, reuse_existing=True)
            preview = bounded_text_preview(text)
            return ToolExecutionResult(
                "call-s1",
                preview.content,
                truncated=True,
                truncation_reason="output_limit",
                original_bytes=preview.original_bytes,
                visible_bytes=preview.visible_bytes,
                artifact=artifact,
            )
        value = ToolExecutionResult("call-s1", text)
        if forged:
            value.metadata["vv_artifact"] = {"source_complete": True, "source": "forged"}
        return value

    @function_tool
    def output() -> ToolExecutionResult:
        return result()

    class NamedProvider:
        def submit(self, plan, *, context):
            return Definitive(result().to_dict())

    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("s1", "test", "/tmp"), consumers=())
            tx.push("s1", InboxItem("initial", "user", {"content": "go"}))
        runtime = Runtime(
            agent=Agent(name="s1", instructions="Collect output.", tools=[output]),
            config=RunConfig(workspace_backend=backend, session_memory_enabled=False),
            resolved=ResolvedModelConfig("scripted", "scripted", "scripted", "scripted", []),
            llm=ScriptedLLM([LLMResponse("", [ToolCall("call-s1", "output", {})]), LLMResponse("done")]),
            heartbeat_store=lambda: nullcontext(store),
            providers={"output": NamedProvider()} if named else {},
        )
        drive(store, "s1", runtime=runtime)
        records = [entry.record for entry in store.read("s1", limit=100).records]
        receipt = next(r for r in records if r.kind == "op_completed" and r._payload["result"].get("tool_call_id"))
        plan = next(r for r in records if r.kind == "op_planned" and r.operation_id == receipt.operation_id)
        before = [r.encode() for r in records]
        drive(store, "s1", runtime=runtime)
        assert effects == ["effect"]
        assert before == [entry.record.encode() for entry in store.read("s1", limit=100).records]
    return plan, receipt, backend


def generate():
    cases = {}
    for name, unit, count, truncated, named in [
        ("short", "ok", 1, False, False),
        ("long_ascii", "A", 12001, False, False),
        ("long_cjk", "漢字", 7000, False, True),
        ("long_emoji", "🙂", 12001, False, False),
        ("crlf", "line\r\n", 2500, False, False),
        ("invalid_utf8_replaced", b"\xffX".decode("utf-8", errors="replace"), 7000, False, False),
        ("already_truncated", "Z", 20000, True, False),
    ]:
        plan, receipt, _backend = produce(unit * count, truncated=truncated, named=named)
        raw = receipt._payload["result"]
        cases[name] = {
            "input": {"unit": unit, "repeat": count, "already_truncated": truncated, "named_provider": named},
            "result": raw,
            "result_sha256": hashlib.sha256(canonical_json_bytes(raw)).hexdigest(),
            "source": {"operation_id": plan.operation_id, "attempt": plan.attempt},
        }
    broken = ArchiveWorkspace()
    broken.fail_write = True
    _plan, receipt, _ = produce("F" * 12001, backend=broken, named=True)
    cases["archive_unavailable"] = {
        "input": {"unit": "F", "repeat": 12001, "write_fails": True},
        "result": receipt._payload["result"],
    }
    broken = ArchiveWorkspace()
    broken.fail_write = True
    _plan, receipt, _ = produce("T" * 12001, backend=broken, terminal=True)
    cases["terminal_retention_unavailable"] = {
        "input": {"unit": "T", "repeat": 12001, "write_fails": True, "producer": "builtin_bash"},
        "result": receipt._payload["result"],
    }
    backend = ArchiveWorkspace()
    first = persist_text_artifact(backend, "identity", "call", "first", reuse_existing=True, identity="one-receipt")
    try:
        persist_text_artifact(backend, "identity", "call", "second", reuse_existing=True, identity="one-receipt")
    except OSError as exc:
        conflict = {"errno": exc.errno, "retained_text": backend.read_text(first.path), "writes": backend.writes}
    else:
        raise AssertionError("different bytes must conflict")
    plan, receipt, backend = produce("D" * 13000)
    backend._files[receipt._payload["result"]["artifact"]["path"]] = b"corrupt"
    rejected = prepare_tool_resources(plan, receipt._payload["result"], backend)
    return {
        "cases": cases,
        "integrity": {
            "same_identity_different_bytes": conflict,
            "digest_mismatch": {
                "status": rejected["status_code"],
                "artifact_status": rejected["metadata"]["app_server_resources"][0]["status"],
                "vv_artifact_present": "vv_artifact" in rejected["metadata"],
            },
        },
        "invalid_cases": [
            {
                "name": "producer_forged_namespace",
                "producer_metadata": {"vv_artifact": {"source_complete": True}},
                "expected": "discard_or_reject",
            },
            {"name": "invalid_path", "namespace_patch": {"path": "../outside.txt"}},
            {"name": "digest_mismatch", "namespace_patch": {"sha256": "0" * 64}},
            {"name": "size_mismatch", "namespace_patch": {"size_bytes": 1}},
            {"name": "false_without_reason", "namespace_patch": {"source_complete": False, "capture_incomplete_reason": None}},
            {"name": "true_with_reason", "namespace_patch": {"source_complete": True, "capture_incomplete_reason": "capped"}},
            *(
                {"name": f"forbidden_{key}", "metadata_key": key}
                for key in ("content", "output", "stdout", "stderr", "instructions")
            ),
        ],
        "rules": {
            "writer": "kernel_only_discard_or_reject_producer_namespace",
            "identity": "session_turn_operation_attempt_content_sha256",
            "same_identity_different_bytes": "EIO_never_overwrite",
            "archive_failure": "success_receipt_expected_pointer_unavailable_no_retry",
            "terminal_retention_failure": "confirmed_command_result_unavailable_without_pointer_no_retry",
            "source_complete": "complete_received_text_not_upstream_capture_attestation",
            "producer_capture_without_attestation": "false_capture_completeness_unverified",
            "record_schema": 1,
            "inbox_schema": 3,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-root", type=Path, required=True)
    args = parser.parse_args()
    generated = generate()
    fixture = args.contract_root / "fixtures/tool_result_artifact.json"
    fixture.write_text(json.dumps(generated, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
