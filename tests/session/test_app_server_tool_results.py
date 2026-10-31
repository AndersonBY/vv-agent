"""Result reads use real App Server producers on SQLite memory/file and PostgreSQL."""

import base64
import hashlib
import json
import runpy
import tracemalloc
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent import Agent, ScriptedModelProvider, ToolResultResourceRef
from vv_agent.app_server import AppServerClient, ToolResultReadParams, ToolResultRef
from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.types import LLMResponse, ToolCall
from vv_agent.workspace import LocalWorkspaceBackend, MemoryWorkspaceBackend

from .conftest import open_store

PRODUCER = runpy.run_path(str(Path(__file__).parents[2] / "scripts/app_server_tool_result_fixtures.py"))
ResultHost, ResultServer, VersionWorkspace = (PRODUCER[k] for k in ("ResultHost", "ResultServer", "VersionWorkspace"))
KINDS, THREAD, UNICODE, BINARY = (PRODUCER[k] for k in ("KINDS", "THREAD", "UNICODE", "BINARY"))


def events(server):
    collected, after = [], None
    while True:
        params = {"threadId": THREAD, "limit": 11}
        if after is not None:
            params["afterCursor"] = after
        response = server.rpc("thread/events/list", params)
        assert "result" in response, response
        page = response["result"]
        collected.extend(e["event"] for e in page["events"])
        if not page["more"]:
            return collected
        after = page["nextCursor"]


def error(response, reason, code=-32602):
    assert response["error"]["code"] == code, response
    assert response["error"].get("data", {}).get("reason", response["error"]["message"]) == reason, response


def test_five_kinds_artifact_cursor_unicode_all_projections_and_recreated_pages(store, database):
    backend, bodies, pages_before = VersionWorkspace(), {}, {}
    host = ResultHost(backend)
    server = ResultServer(store, host)
    try:
        server.start()
        summaries = server.summaries()
        assert [s["kind"] for s in summaries] == ["text", "json", "image", "file", "error", "text", "text", "text", "image"]
        assert backend.storage_calls == ["put_version"]  # Only the tool's own text artifact write.
        assert backend.retains == 3 and backend.copies == 0 and backend.opens == 0
        completed = [e for e in events(server) if e["type"] == "tool_call_completed"]
        diagnostics = [e for e in events(server) if e["type"] == "diagnostic" and e["code"] == "tool_result"]
        assert [e["metadata"]["tool_result"] for e in completed] == summaries
        assert [e["details"] for e in diagnostics] == summaries
        live = [
            m["params"]["payload"]["result"]
            for m in server.messages
            if m.get("method") == "item/completed" and m["params"].get("type") == "toolCall"
        ]
        assert live == summaries
        exposed = canonical_json_bytes({"events": events(server), "summaries": summaries})
        for private in (b"version:", b"private-metadata", b"private.invalid", b"credential", b".vv-agent/artifacts/"):
            assert private not in exposed
        assert UNICODE.encode() not in exposed
        for kind, summary in zip(KINDS, summaries, strict=True):
            assert len(summary["preview"].encode()) <= 2048
            assert len(canonical_json_bytes(summary)) <= 16_384
            if "value" in summary:
                assert len(canonical_json_bytes(summary["value"])) <= 8192
            body, pages = server.all_bytes(summary["ref"], limit=4093)
            bodies[kind], pages_before[(kind, "content")] = body, pages
            assert pages[0]["kind"] == summary["kind"] and pages[0]["context"] == "normal"
            for resource in summary["resources"]:
                if resource["status"] == "retained":
                    data, pages = server.all_bytes(summary["ref"], part=resource["id"], limit=65_536)
                    bodies[kind + "-resource"] = data
                    pages_before[(kind, resource["id"])] = pages
        assert bodies["unicode"] == UNICODE.encode()
        unicode_ref = summaries[7]["ref"]
        raw = store.read(THREAD, after_seq=unicode_ref["seq"] - 1, limit=1).records[0].record.payload["result"]
        assert "metadata" not in raw
        assert bodies["file-resource"] == BINARY
        assert bodies["artifact-resource"] == UNICODE.encode()
        assert bodies["cursor"] == "第一页".encode()
        assert summaries[6]["cursor"] == {
            "kind": "read_file",
            "offsetChars": 3,
            "sha256": hashlib.sha256("第一页\n第二页".encode()).hexdigest(),
        }
        assert summaries[5]["truncated"] is True and summaries[5]["resources"][0]["fullText"] is True
        error(server.page(summaries[-1]["ref"], part="resource:0"), "resource_not_retained")
        before = store.read(THREAD, limit=1).head_seq
        executions = list(host.executions)
        server.close()
        if database is not store:
            store.connection.close()
        with open_store(database) as reopened:
            server = ResultServer(reopened, ResultHost(backend))
            assert server.summaries() == summaries
            for kind, summary in zip(KINDS, summaries, strict=True):
                _, pages = server.all_bytes(summary["ref"], limit=4093)
                assert pages == pages_before[(kind, "content")]
                for resource in summary["resources"]:
                    if resource["status"] == "retained":
                        _, pages = server.all_bytes(summary["ref"], part=resource["id"], limit=65_536)
                        assert pages == pages_before[(kind, resource["id"])]
            assert reopened.read(THREAD, limit=1).head_seq == before
            assert server.host.executions == [] and host.executions == executions
            assert backend.peak_chunk <= 65_536
            server.close()
    finally:
        server.close()


@pytest.mark.parametrize("workspace", ["local", "memory", "s3"])
def test_retained_file_survives_source_replace_and_delete_and_detects_resource_damage(store, tmp_path, workspace):
    if workspace == "s3":
        from test_workspace_backends import _make_s3_backend

        backend = _make_s3_backend()
    else:
        backend = LocalWorkspaceBackend(tmp_path / "workspace") if workspace == "local" else MemoryWorkspaceBackend()
    backend = cast(Any, backend)
    if workspace == "local":
        backend.root.mkdir()
        (backend.root / "output.bin").write_bytes(BINARY)
    elif workspace == "memory":
        backend._files["output.bin"] = BINARY
    else:
        backend._client.objects[backend._key("output.bin")] = BINARY
    server = ResultServer(store, ResultHost(backend, kinds=("file",)))
    try:
        server.start()
        summary = server.summaries()[0]
        assert summary["resources"][0]["status"] == "retained"
        backend.write_text("output.bin", "replacement")
        assert server.all_bytes(summary["ref"], part="resource:0")[0] == BINARY
        if workspace == "local":
            (backend.root / "output.bin").unlink()
        elif workspace == "memory":
            del backend._files["output.bin"]
        else:
            del backend._client.objects[backend._key("output.bin")]
        assert server.all_bytes(summary["ref"], part="resource:0")[0] == BINARY
        row = store.read(THREAD, after_seq=summary["ref"]["seq"] - 1, limit=1).records[0]
        retained = row.record.payload["result"]["metadata"]["app_server_resources"][0]["ref"]
        path = retained["locator"]
        if workspace == "local":
            target = backend._artifact_root / path
            target.write_bytes(b"x" * len(BINARY))
        elif workspace == "memory":
            backend._files[path] = b"x" * len(BINARY)
        else:
            backend._client.objects[backend._key(path)] = b"x" * len(BINARY)
        error(server.page(summary["ref"], part="resource:0"), "stale_reference")
        if workspace == "local":
            target.unlink()
        elif workspace == "memory":
            del backend._files[path]
        else:
            del backend._client.objects[backend._key(path)]
        error(server.page(summary["ref"], part="resource:0"), "resource_missing")
    finally:
        server.close()


def test_every_page_authorized_before_open_and_cross_workspace_denied(store):
    backend, host = VersionWorkspace(), None
    host = ResultHost(backend, kinds=("file",))
    server = ResultServer(store, host)
    try:
        server.start()
        summary = server.summaries()[0]
        host.denied.add(THREAD)
        error(server.page(summary["ref"], part="resource:0"), "permission_denied", -32027)
        assert backend.opens == 0
        host.denied.clear()
        host.deny_resource = True
        error(server.page(summary["ref"], part="resource:0"), "permission_denied", -32027)
        assert backend.opens == 0
        host.deny_resource = False
        for offset in (0, 1, 7):
            assert "result" in server.page(summary["ref"], part="resource:0", offset=offset, limit=1)
        assert len(host.resource_authorizations) == 4 and backend.opens == 3
        host._run_config = replace(host._run_config, workspace_backend=MemoryWorkspaceBackend())
        error(server.page(summary["ref"], part="resource:0"), "permission_denied", -32027)
        assert backend.opens == 3
    finally:
        server.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("sessionId", "another-thread"),
        ("turnId", "wrong"),
        ("operationId", "wrong"),
        ("attempt", 2),
        ("toolCallId", "wrong"),
        ("recordDigest", "0" * 64),
        ("seq", True),
    ],
)
def test_forged_result_ref_rejected_without_open(store, field, value):
    backend = VersionWorkspace()
    server = ResultServer(store, ResultHost(backend, kinds=("file",)))
    try:
        server.start()
        ref = server.summaries()[0]["ref"] | {field: value}
        error(server.page(ref, part="resource:0"), "invalid_result_ref")
        assert backend.opens == 0
    finally:
        server.close()


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"offset": -1}, "invalid_range"),
        ({"offset": True}, "invalid_range"),
        ({"offset": 2**53}, "invalid_range"),
        ({"limit": 0}, "invalid_range"),
        ({"limit": 65_537}, "invalid_range"),
        ({"limit": 1.5}, "invalid_range"),
        ({"part": "resource:8"}, "invalid_result_ref"),
        ({"part": "../../secret"}, "invalid_result_ref"),
    ],
)
def test_ranges_and_resource_ids_are_closed(store, changes, reason):
    server = ResultServer(store, ResultHost(VersionWorkspace(), kinds=("file",)))
    try:
        server.start()
        error(server.page(server.summaries()[0]["ref"], **changes), reason)
    finally:
        server.close()


def test_missing_completion_and_eof_and_large_file_page_buffer_bound(store, tmp_path, monkeypatch):
    from vv_agent.session import tool_result_read

    sizes = []

    class PageBuffer(bytearray):
        def extend(self, data):
            super().extend(data)
            sizes.append(len(self))
            assert len(self) <= requested[0]

    requested = [7]
    monkeypatch.setattr(tool_result_read, "bytearray", PageBuffer, raising=False)
    backend = LocalWorkspaceBackend(tmp_path)
    # 16 MiB distinguishes streaming from read_bytes(), even when requesting just seven bytes.
    data = bytes(range(256)) * 65_536
    (tmp_path / "output.bin").write_bytes(data)
    server = ResultServer(store, ResultHost(backend, kinds=("file",)))
    try:
        server.start()
        summary = server.summaries()[0]
        before = store.read(THREAD, limit=1).head_seq
        tracemalloc.start()
        try:
            response = server.page(summary["ref"], part="resource:0", offset=3, limit=7)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert base64.b64decode(response["result"]["data"]) == data[3:10]
        assert sizes == [7]
        assert peak < 2_000_000
        assert store.read(THREAD, limit=1).head_seq == before
        requested[0] = 1
        eof = server.page(summary["ref"], part="resource:0", offset=len(data), limit=1)["result"]
        assert eof["eof"] is True and eof["data"] == ""
        error(server.page(summary["ref"], part="resource:0", offset=len(data) + 1), "invalid_range")
        error(server.page(summary["ref"] | {"seq": before + 100}), "result_not_found")
        client = AppServerClient(server=server.server, transport=server.transport)
        client._next_request_id = server.identity + 1
        page = client.read_tool_result(ToolResultReadParams(THREAD, ToolResultRef.from_dict(summary["ref"]), limit=1))
        assert page["ref"] == summary["ref"]
    finally:
        server.close()


def test_missing_capability_or_explicit_opt_out_never_copies(store):
    class NoRetention:
        def __init__(self):
            self.copy_attempts = 0
            self.storage = MemoryWorkspaceBackend()
            self.storage._files["output.bin"] = BINARY

        def __getattr__(self, name):
            if name in {"retain_output", "open_retained", "retained_workspace_id"}:
                raise AttributeError(name)
            return getattr(self.storage, name)

        def read_bytes(self, path):
            self.copy_attempts += 1
            raise AssertionError("kernel must not copy")

    backend = NoRetention()
    server = ResultServer(store, ResultHost(backend, kinds=("file",)))
    try:
        server.start()
        summary = server.summaries()[0]
        assert summary["resources"][0]["status"] == "not_retained"
        error(server.page(summary["ref"], part="resource:0"), "resource_not_retained")
        assert backend.copy_attempts == 0
    finally:
        server.close()


def test_local_retention_rejects_escape_and_source_symlink(store, tmp_path):
    backend = LocalWorkspaceBackend(tmp_path / "workspace", allow_outside_root=True)
    backend.root.mkdir()
    secret = tmp_path / "secret"
    secret.write_bytes(b"private")
    (backend.root / "output.bin").symlink_to(secret)
    server = ResultServer(store, ResultHost(backend, kinds=("file",)))
    try:
        server.start()
        summary = server.summaries()[0]
        assert summary["resources"][0]["status"] == "unavailable"
        error(server.page(summary["ref"], part="resource:0"), "resource_not_retained")
    finally:
        server.close()


def test_child_results_require_real_admission_and_child_permission(store):
    backend = VersionWorkspace()
    host = ResultHost(backend, kinds=("file",))
    child = Agent("child", "Produce a file.", tools=[host.output_tool])
    parent = Agent("parent", "Delegate.", tools=[child.as_tool(name="delegate")])

    def model(request):
        if any(tool["function"]["name"] == "delegate" for tool in request.tools):
            if any(m.role == "tool" for m in request.messages):
                return LLMResponse("parent done")
            return LLMResponse(
                "",
                [
                    ToolCall(
                        "delegate",
                        "delegate",
                        {"task_description": "file"},
                    )
                ],
            )
        if any(m.role == "tool" for m in request.messages):
            return LLMResponse("child done")
        return LLMResponse("", [ToolCall("child-file", "output", {"kind": "file"})])

    host._agent = parent
    host._run_config = replace(host._run_config, model_provider=ScriptedModelProvider.from_callback("test", "m", model))
    server = ResultServer(store, host)
    try:
        server.start()
        children = [
            sid
            for sid in store.list_sessions()
            if store.read(sid, limit=1).records[0].record.payload["parent_session_id"] == THREAD
        ]
        assert len(children) == 1
        sid = children[0]
        from vv_agent.session.projection import project_records

        summaries = [
            e.metadata["tool_result"] for e in project_records(store.read_state(sid)[1]) if e.type == "tool_call_completed"
        ]
        assert len(summaries) == 1
        ref = summaries[0]["ref"]
        assert server.all_bytes(ref, part="resource:0")[0] == BINARY
        error(server.page(ref, thread=sid), "invalid_result_ref")
        host.denied.add(sid)
        error(server.page(ref, part="resource:0"), "permission_denied", -32027)
        assert (server.transport.connection_id, "thread/toolResult/read", sid) in host.authorizations
    finally:
        server.close()


def test_canonical_fixture_is_generated_by_real_app_server():
    canonical = Path(__file__).parents[1] / "fixtures/parity/app_server_tool_results.json"
    assert PRODUCER["produce"]() == json.loads(canonical.read_text())


@pytest.mark.parametrize("mode", ["correction", "audit", "old_attempt"])
def test_immutable_attempt_contexts_remain_readable_after_recreate(store, database, mode):
    from vv_agent import Definitive, ToolExecutionResult, Unknown
    from vv_agent.session.records import InboxItem

    class HistoryProvider:
        def __init__(self):
            self.plans, self.submits, self.pushed = {}, 0, False

        def result(self, plan):
            return ToolExecutionResult(plan.payload["request"]["id"], f"attempt-{plan.attempt}").to_dict()

        def submit(self, plan, *, context):
            self.plans[plan.attempt] = plan
            self.submits += 1
            if plan.attempt == 1:
                return Unknown("external acceptance awaiting receipt")
            return Definitive(self.result(plan), ("terminal-receipt",))

        def query(self, handle):
            raise AssertionError("no queryable handle")

        def cancel(self, handle):
            return Unknown("no cancellation evidence")

        def callback(self):
            plan = self.plans[1]
            return InboxItem(
                "late-result",
                "provider_result",
                {
                    "operation_id": plan.operation_id,
                    "attempt": plan.attempt,
                    "request_digest": plan.payload["request_digest"],
                    "provider_binding": "output",
                    "result": self.result(plan),
                    "evidence": ["terminal-receipt"],
                    "usage": {},
                },
                target_turn_id=plan.turn_id,
                generation=0,
            )

        def authenticate(self, item, plan):
            return item.encode() == self.callback().encode()

        def push(self):
            self.pushed = True
            with store.atomic() as tx:
                tx.push(THREAD, self.callback())

    provider, backend = HistoryProvider(), VersionWorkspace()

    def model(request):
        if any(message.role == "tool" for message in request.messages):
            if mode == "correction" and not provider.pushed:
                provider.push()
            return LLMResponse("done")
        return LLMResponse("", [ToolCall("same-call", "output", {"kind": "file"})])

    host = ResultHost(backend, kinds=("file",), model=model)
    if mode == "old_attempt":
        from vv_agent.tools.metadata import ToolIdempotency, ToolMetadata

        host.output_tool.tool_metadata = ToolMetadata(idempotency=ToolIdempotency.SUPPORTED)
    host._run_config = replace(host._run_config, tool_providers={"output": provider})
    server = ResultServer(store, host)
    try:
        server.start()
        if mode != "correction":
            provider.push()
            server.server.wake(THREAD)
            server.server.run_adapter.join()
        summaries = server.summaries()
        assert len(summaries) == (2 if mode == "old_attempt" else 1)
        contexts = {}
        for summary in summaries:
            body, pages = server.all_bytes(summary["ref"], limit=3)
            assert body == f"attempt-{summary['ref']['attempt']}".encode()
            contexts[summary["ref"]["attempt"]] = pages[0]["context"]
            assert summary["context"] == pages[0]["context"]
        assert contexts[1] == ("correction" if mode == "correction" else "audit")
        if mode == "old_attempt":
            assert contexts[2] == "normal"
        submits = provider.submits
        server.close()
        if database is not store:
            store.connection.close()
        with open_store(database) as reopened:
            server = ResultServer(reopened, host)
            assert server.summaries() == summaries
            for summary in summaries:
                _, pages = server.all_bytes(summary["ref"], limit=3)
                assert pages[0]["context"] == contexts[summary["ref"]["attempt"]]
            assert provider.submits == submits
            server.close()
    finally:
        server.close()


def test_builtin_open_rejects_cross_workspace_and_bad_private_locator(tmp_path):
    from vv_agent.workspace.resources import ToolResultResourceExpectation

    first, second = LocalWorkspaceBackend(tmp_path / "one"), LocalWorkspaceBackend(tmp_path / "two")
    first.root.mkdir()
    (first.root / "file").write_bytes(b"file")
    ref = first.retain_output("file", expected=ToolResultResourceExpectation("a" * 64, "text/plain", "file"))
    with pytest.raises(PermissionError):
        list(second.open_retained(ref))
    for path in ("/etc/passwd", "../secret", ".vv-agent/resources/../secret", "public.txt"):
        with pytest.raises(ValueError):
            list(first.open_retained(replace(ref, locator=path)))
    permissive = LocalWorkspaceBackend(first.root, allow_outside_root=True)
    with pytest.raises(PermissionError):
        permissive.write_text(str(first._artifact_root / ref.locator), "overwrite")
    forged = ToolResultResourceRef.from_dict(ref.to_dict() | {"owner": "b" * 64})
    with pytest.raises(ValueError):
        list(first.open_retained(forged))


@pytest.mark.parametrize("response", ["none", "wrong_owner", "error"])
def test_capability_opt_out_or_bad_reference_does_not_retry_the_tool(store, response):
    class Backend(VersionWorkspace):
        def retain_output(self, path, *, expected):
            if response == "none":
                return None
            if response == "error":
                raise OSError("object storage unavailable")
            return replace(super().retain_output(path, expected=expected), owner="f" * 64)

    backend = Backend()
    server = ResultServer(store, ResultHost(backend, kinds=("file",)))
    try:
        server.start()
        summary = server.summaries()[0]
        assert summary["resources"][0]["status"] == ("not_retained" if response == "none" else "unavailable")
        error(server.page(summary["ref"], part="resource:0"), "resource_not_retained")
        assert server.host.executions == ["file"] and backend.storage_calls == []
    finally:
        server.close()


def test_other_existing_thread_ref_cannot_be_read_through_this_thread(store):
    backend = VersionWorkspace()
    server = ResultServer(store, ResultHost(backend, kinds=("file",)))
    try:
        server.start()
        server.start(thread="other-existing-thread")
        other = server.summaries(thread="other-existing-thread")[0]
        error(server.page(other["ref"], part="resource:0"), "invalid_result_ref")
        assert backend.opens == 0
    finally:
        server.close()


def test_duplicate_authenticated_file_receipt_does_not_retain_or_submit_again(store):
    from vv_agent import Definitive, ToolExecutionResult
    from vv_agent.session.records import InboxItem

    class FileProvider:
        def __init__(self):
            self.submits = 0

        def result(self):
            return ToolExecutionResult(
                "call-file",
                '{"path":"output.bin","mime_type":"application/octet-stream"}',
                metadata={"output_type": "file"},
            ).to_dict()

        def submit(self, plan, *, context):
            self.plan = plan
            self.submits += 1
            return Definitive(self.result(), ("file-receipt",), usage={})

        def callback(self):
            plan = self.plan
            return InboxItem(
                "duplicate-file",
                "provider_result",
                {
                    "operation_id": plan.operation_id,
                    "attempt": plan.attempt,
                    "request_digest": plan.payload["request_digest"],
                    "provider_binding": "output",
                    "result": self.result(),
                    "evidence": ["file-receipt"],
                    "usage": {},
                },
                target_turn_id=plan.turn_id,
                generation=0,
            )

        def authenticate(self, item, plan):
            return item.encode() == self.callback().encode()

        def query(self, handle):
            raise AssertionError("no query")

        def cancel(self, handle):
            raise AssertionError("no cancel")

    backend, provider = VersionWorkspace(), FileProvider()
    host = ResultHost(backend, kinds=("file",))
    host._run_config = replace(host._run_config, tool_providers={"output": provider})
    server = ResultServer(store, host)
    try:
        server.start()
        before = server.summaries()
        assert backend.retains == provider.submits == 1
        with store.atomic() as tx:
            tx.push(THREAD, provider.callback())
        server.server.wake(THREAD)
        server.server.run_adapter.join()
        state = store.read_state(THREAD)[0]
        assert state.applied_inputs["duplicate-file"].payload["disposition"] == "noop"
        assert server.summaries() == before
        assert backend.retains == provider.submits == 1
    finally:
        server.close()


@pytest.mark.parametrize(
    "kind,body",
    [
        ("json", "not json"),
        ("json", "NaN"),
        ("file", '{"path":42,"mime_type":null}'),
        ("image", '{"path":"image.png","mime_type":null,"secret":"private"}'),
        ("error", '{"ok":true,"error":"bad","error_code":null,"retryable":false}'),
    ],
)
def test_invalid_declared_output_body_is_not_recorded_as_a_completed_result(store, kind, body):
    from vv_agent import Definitive, ToolExecutionResult, Unknown

    class InvalidProvider:
        submits = 0

        def submit(self, plan, *, context):
            self.submits += 1
            return Definitive(
                ToolExecutionResult(
                    plan.payload["request"]["id"],
                    body,
                    metadata={"output_type": kind},
                ).to_dict()
            )

        def query(self, handle):
            raise AssertionError("no handle")

        def cancel(self, handle):
            return Unknown("cannot confirm")

        def authenticate(self, item, plan):
            return False

    backend, provider = VersionWorkspace(), InvalidProvider()
    host = ResultHost(backend, kinds=("file",))
    host._run_config = replace(host._run_config, tool_providers={"output": provider})
    server = ResultServer(store, host)
    try:
        server.start()
        assert server.summaries() == []
        assert provider.submits == 1 and backend.retains == 0
        assert any(event.get("code") == "provider_result_invalid" for event in events(server))
    finally:
        server.close()


def test_local_snapshot_reopens_on_new_backend_after_source_deletion(store, database, tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "output.bin").write_bytes(BINARY)
    server = ResultServer(store, ResultHost(LocalWorkspaceBackend(root), kinds=("file",)))
    try:
        server.start()
        summary = server.summaries()[0]
        expected = server.all_bytes(summary["ref"], part="resource:0", limit=65535)
        (root / "output.bin").unlink()
        server.close()
        if database is not store:
            store.connection.close()
        with open_store(database) as reopened:
            server = ResultServer(reopened, ResultHost(LocalWorkspaceBackend(root), kinds=("file",)))
            assert server.all_bytes(summary["ref"], part="resource:0", limit=65535) == expected
            assert server.host.executions == []
            server.close()
    finally:
        server.close()


def test_result_wire_schemas_are_closed_and_validate_real_producer_pages(store):
    from jsonschema import Draft202012Validator

    from vv_agent.app_server.schema import json_schema_bundle

    server = ResultServer(store, ResultHost(VersionWorkspace(), kinds=("json", "file")))
    try:
        server.start()
        schemas = json_schema_bundle()
        for summary in server.summaries():
            Draft202012Validator(json.loads(schemas["ToolResultSummary"])).validate(summary)
            params = ToolResultReadParams(THREAD, ToolResultRef.from_dict(summary["ref"])).to_dict()
            request_schema = Draft202012Validator(json.loads(schemas["ToolResultReadParams"]))
            request_schema.validate(params)
            assert list(request_schema.iter_errors(params | {"path": "/etc/passwd"}))
            assert list(request_schema.iter_errors(params | {"limit": 65537}))
            response = server.page(summary["ref"])["result"]
            Draft202012Validator(json.loads(schemas["ToolResultReadResponse"])).validate(response)
        ref = server.summaries()[0]["ref"]
        server.identity = 2**63 - 2
        assert "result" in server.page(ref)
    finally:
        server.close()


def test_old_turn_resource_rebuild_uses_original_workspace_metadata(store):
    class Workspace(VersionWorkspace):
        def __init__(self, identity):
            self.identity = identity
            super().__init__()

        @property
        def retained_workspace_id(self):
            return hashlib.sha256(self.identity.encode()).hexdigest()

    first, second = Workspace("first-workspace"), Workspace("second-workspace")

    def model(request):
        last_user = max(i for i, message in enumerate(request.messages) if message.role == "user")
        if any(message.role == "tool" for message in request.messages[last_user + 1 :]):
            return LLMResponse("done")
        return LLMResponse("", [ToolCall("reused-call", "output", {"kind": "file"})])

    class Host(ResultHost):
        def build_run_config(self, request):
            config = super().build_run_config(request)
            backend = first if (request.metadata or {}).get("workspace_choice") == "first" else second
            return replace(config, workspace_backend=backend)

    host = Host(first, kinds=("file",), model=model)
    server = ResultServer(store, host)
    try:
        assert "result" in server.rpc("thread/start", {"inputId": THREAD})
        prior = None
        for choice in ("first", "second"):
            response = server.rpc(
                "turn/start",
                {
                    "threadId": THREAD,
                    "inputId": choice,
                    "expectedTurnId": prior,
                    "input": [{"type": "text", "text": "produce a file"}],
                    "metadata": {"workspace_choice": choice},
                },
            )
            assert "result" in response, response
            prior = response["result"]["turnId"]
            server.server.run_adapter.join()
        summaries = server.summaries()
        assert len(summaries) == 2
        for summary in summaries:
            assert server.all_bytes(summary["ref"], part="resource:0")[0] == BINARY
        assert first.retains == second.retains == 1
    finally:
        server.close()


@pytest.mark.parametrize("kind", ["json", "error"])
def test_truncated_structured_preview_reads_full_original_artifact(store, kind):
    from vv_agent import ToolExecutionResult
    from vv_agent.types import ToolResultStatus
    from vv_agent.workspace.artifacts import persist_text_artifact

    backend = VersionWorkspace()
    original = json.dumps(
        {"message": UNICODE} if kind == "json" else {"ok": False, "error": UNICODE, "error_code": None, "retryable": False},
        ensure_ascii=False,
    )
    artifact = persist_text_artifact(backend, "structured", "output", original, reuse_existing=True)
    preview = '{"partial":true}'
    host = ResultHost(backend, kinds=(kind,))
    host.output_tool.on_invoke = lambda context, arguments: ToolExecutionResult(
        "",
        preview,
        metadata={"output_type": kind},
        status_code=ToolResultStatus.ERROR if kind == "error" else ToolResultStatus.SUCCESS,
        truncated=True,
        truncation_reason="output_limit",
        original_bytes=len(original.encode()),
        visible_bytes=len(preview.encode()),
        artifact=artifact,
    )
    server = ResultServer(store, host)
    try:
        server.start()
        summary = server.summaries()[0]
        assert summary["kind"] == kind and summary["truncated"] is True
        assert "value" not in summary and summary["resources"][0]["fullText"] is True
        assert server.all_bytes(summary["ref"])[0] == preview.encode()
        assert server.all_bytes(summary["ref"], part="resource:0")[0] == original.encode()
    finally:
        server.close()


def test_json_large_integer_stays_exact_in_body_without_unsafe_inline_value(store):
    from vv_agent.tools.outputs import ToolOutputJson

    value = {"native_id": 10**30}
    host = ResultHost(VersionWorkspace(), kinds=("json",))
    host.output_tool.on_invoke = lambda context, arguments: ToolOutputJson(value)
    server = ResultServer(store, host)
    try:
        server.start()
        summary = server.summaries()[0]
        assert summary["kind"] == "json" and "value" not in summary
        body, _ = server.all_bytes(summary["ref"], limit=5)
        assert body == json.dumps(value, ensure_ascii=False).encode()
        assert json.loads(body) == value
    finally:
        server.close()
