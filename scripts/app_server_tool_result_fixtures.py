"""Tool results from the real JSON-RPC App Server and immutable-version workspace."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from vv_agent import (
    Agent,
    RunConfig,
    ScriptedModelProvider,
    ToolExecutionResult,
    ToolResultCursor,
    ToolResultResourceRef,
    function_tool,
)
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.tools.outputs import ToolOutputError, ToolOutputFile, ToolOutputImage, ToolOutputJson, ToolOutputText
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import LLMResponse, ToolCall
from vv_agent.workspace import MemoryWorkspaceBackend
from vv_agent.workspace.artifacts import persist_text_artifact

THREAD = "result-thread"
UNICODE = "开头🙂漢字\n" * 20_000
BINARY = bytes(range(256)) * 4097
IMAGE = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==")
KINDS = ("text", "json", "image", "file", "error", "artifact", "cursor", "unicode", "url")


class VersionWorkspace(MemoryWorkspaceBackend):
    """Existing immutable versions; retention must perform no storage reads or writes."""

    def __init__(self):
        super().__init__()
        self.versions = {}
        self.retains = self.opens = self.copies = self.peak_chunk = 0
        self._files["output.bin"] = BINARY
        self._files["image.png"] = IMAGE
        self._files["cursor.txt"] = "第一页\n第二页".encode()
        self.storage_calls = []
        # Version metadata already exists before any tool returns.
        for path, data in self._files.items():
            self.versions[path] = (data, hashlib.sha256(data).hexdigest())

    @property
    def retained_workspace_id(self):
        return hashlib.sha256(b"fixture-version-workspace").hexdigest()

    def read_bytes(self, path):
        self.storage_calls.append("get_current")
        return super().read_bytes(path)

    def _read_bytes_chunks(self, path):
        self.storage_calls.append("get_current_stream")
        return super()._read_bytes_chunks(path)

    def write_bytes_chunks_exclusive(self, path, chunks):
        self.storage_calls.append("put_version")
        size = super().write_bytes_chunks_exclusive(path, chunks)
        data = self._files[path]
        self.versions[path] = (data, hashlib.sha256(data).hexdigest())
        return size

    def retain_output(self, path, *, expected):
        self.retains += 1
        data, digest = self.versions[path]
        return ToolResultResourceRef(
            self.retained_workspace_id,
            expected.owner,
            "version:" + digest,
            len(data),
            digest,
            expected.media_type,
            expected.name,
        )

    def open_retained(self, ref):
        if ref.workspace_id != self.retained_workspace_id:
            raise PermissionError("wrong workspace")
        self.opens += 1
        self.storage_calls.append("get_version")
        data = next((data for data, digest in self.versions.values() if "version:" + digest == ref.locator), None)
        if data is None:
            raise FileNotFoundError("missing version")
        for offset in range(0, len(data), 65_536):
            chunk = data[offset : offset + 65_536]
            self.peak_chunk = max(self.peak_chunk, len(chunk))
            yield chunk


class ResultHost(DefaultAppServerHost):
    def __init__(self, backend, *, kinds=KINDS, agent=None, model=None):
        self.backend, self.executions, self.builds = backend, [], 0
        self.authorizations, self.resource_authorizations = [], []
        self.denied, self.deny_resource = set(), False

        @function_tool
        def output(kind: str):
            self.executions.append(kind)
            if kind == "text":
                return ToolOutputText("plain text", metadata={"output_type": "image", "secret": "private-metadata"})
            if kind == "json":
                return ToolOutputJson({"answer": 42, "汉字": "🙂"}, metadata={"output_type": "file"})
            if kind == "image":
                return ToolOutputImage(path="image.png", mime_type="image/png")
            if kind == "file":
                return ToolOutputFile("output.bin", "application/octet-stream")
            if kind == "error":
                return ToolOutputError("expected error", "fixture_error")
            if kind == "artifact":
                artifact = persist_text_artifact(backend, "result", "artifact", UNICODE, reuse_existing=True)
                return ToolExecutionResult(
                    "",
                    "artifact preview",
                    truncated=True,
                    truncation_reason="output_limit",
                    original_bytes=len(UNICODE.encode()),
                    visible_bytes=len(b"artifact preview"),
                    artifact=artifact,
                )
            if kind == "cursor":
                return ToolExecutionResult(
                    "",
                    "第一页",
                    truncated=True,
                    truncation_reason="read_limit",
                    original_bytes=len("第一页\n第二页".encode()),
                    visible_bytes=len("第一页".encode()),
                    cursor=ToolResultCursor("read_file", "cursor.txt", 3, hashlib.sha256("第一页\n第二页".encode()).hexdigest()),
                )
            if kind == "url":
                return ToolOutputImage(url="https://private.invalid/image?secret=credential", mime_type="image/png")
            return ToolOutputText(UNICODE)

        self.output_tool = output

        def script(request):
            if any(m.role == "tool" for m in request.messages):
                return LLMResponse("done")
            return LLMResponse("outputs", [ToolCall("call-" + k, "output", {"kind": k}) for k in kinds])

        super().__init__(
            agent=agent or Agent("results", "Produce tool outputs.", tools=[output]),
            run_config=RunConfig(
                workspace="/fixture-results",
                workspace_backend=backend,
                model_provider=ScriptedModelProvider.from_callback("test", "m", model or script),
                tool_registry_factory=ToolRegistry,
            ),
        )

    def build_run_config(self, request):
        self.builds += 1
        return super().build_run_config(request)

    def authorize(self, connection_id, method, params):
        self.authorizations.append((connection_id, method, params.get("threadId")))
        return params.get("threadId") not in self.denied

    def authorize_tool_result_resource(self, request):
        self.resource_authorizations.append(request)
        return not self.deny_resource and super().authorize_tool_result_resource(request)


class ResultServer:
    def __init__(self, store, host):
        self.host = host
        self.transport = ChannelTransport(connection_id="result-owner", outbound_capacity=10000)
        self.server = AppServer(transport=self.transport, host=host, store=store)
        self.messages, self.identity = [], 0
        self.rpc("initialize", {"clientInfo": {"name": "results"}})
        self.server.processor.process_message(self.transport.connection_id, {"jsonrpc": "2.0", "method": "initialized"})

    def rpc(self, method, params):
        self.identity += 1
        self.server.processor.process_message(
            self.transport.connection_id,
            {"jsonrpc": "2.0", "id": self.identity, "method": method, "params": params},
        )
        while True:
            wire = self.transport.receive_outbound(timeout=20)
            self.messages.append(wire)
            if wire.get("id") == self.identity:
                return wire

    def start(self, thread=THREAD):
        assert "result" in self.rpc("thread/start", {"inputId": thread})
        response = self.rpc(
            "turn/start",
            {
                "threadId": thread,
                "inputId": "outputs",
                "expectedTurnId": None,
                "input": [{"type": "text", "text": "go"}],
            },
        )
        assert "result" in response, response
        self.server.run_adapter.join()
        return response["result"]

    def summaries(self, thread=THREAD):
        response = self.rpc("thread/read", {"threadId": thread})
        assert "result" in response, response
        return [
            item["payload"]["result"]
            for item in response["result"]["items"]
            if item["type"] == "toolCall" and "result" in item["payload"]
        ]

    def page(self, ref, *, thread=THREAD, part="content", offset=0, limit=65_536):
        return self.rpc(
            "thread/toolResult/read",
            {
                "threadId": thread,
                "ref": ref,
                "part": part,
                "offset": offset,
                "limit": limit,
            },
        )

    def all_bytes(self, ref, *, thread=THREAD, part="content", limit=65_536):
        pages, data, offset = [], bytearray(), 0
        while True:
            response = self.page(ref, thread=thread, part=part, offset=offset, limit=limit)
            assert "result" in response, response
            page = response["result"]
            chunk = base64.b64decode(page["data"], validate=True)
            assert len(chunk) <= limit
            assert len(canonical_json_bytes(response)) <= 98_304
            assert page["offset"] == offset and page["nextOffset"] == offset + len(chunk)
            data.extend(chunk)
            pages.append(page)
            if page["eof"]:
                assert len(data) == page["totalBytes"]
                assert hashlib.sha256(data).hexdigest() == page["sha256"]
                return bytes(data), pages
            assert chunk
            offset = page["nextOffset"]

    def close(self):
        self.server.run_adapter.join()
        self.server.run_adapter.close()
        self.server.kernel.close()


def produce():
    backend = VersionWorkspace()
    host = ResultHost(backend)
    with SQLiteStore.standalone(":memory:") as store:
        server = ResultServer(store, host)
        try:
            server.start()
            summaries = server.summaries()
            completion_storage_calls = list(backend.storage_calls)
            entries = []
            for kind, summary in zip(KINDS, summaries, strict=True):
                body, pages = server.all_bytes(summary["ref"])
                resources = []
                for resource in summary["resources"]:
                    if resource["status"] == "retained":
                        data, resource_pages = server.all_bytes(summary["ref"], part=resource["id"])
                        resources.append(
                            {
                                "id": resource["id"],
                                "bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "firstPage": resource_pages[0],
                                "lastPage": resource_pages[-1],
                            }
                        )
                entries.append(
                    {
                        "name": kind,
                        "summary": summary,
                        "contentSha256": hashlib.sha256(body).hexdigest(),
                        "firstPage": pages[0],
                        "lastPage": pages[-1],
                        "resources": resources,
                    }
                )
            ref = summaries[3]["ref"]
            failures = {
                "invalid_range": server.page(ref, limit=65_537)["error"],
                "invalid_result_ref": server.page(ref | {"attempt": 2})["error"],
                "forged_resource": server.page(ref, part="resource:7")["error"],
                "result_not_found": server.page(ref | {"seq": store.read(THREAD, limit=1).head_seq + 1})["error"],
                "resource_not_retained": server.page(summaries[-1]["ref"], part="resource:0")["error"],
            }
            host.deny_resource = True
            failures["permission_denied"] = server.page(ref, part="resource:0")["error"]
            host.deny_resource = False
            original = backend.versions["output.bin"]
            backend.versions["output.bin"] = (b"changed", original[1])
            failures["stale_reference"] = server.page(ref, part="resource:0")["error"]
            del backend.versions["output.bin"]
            failures["resource_missing"] = server.page(ref, part="resource:0")["error"]
            backend.versions["output.bin"] = original
            return {
                "contract": "app_server_tool_results",
                "protocolVersion": "v5",
                "limits": {
                    "previewBytes": 2048,
                    "inlineBytes": 8192,
                    "descriptorBytes": 16384,
                    "resources": 8,
                    "labelBytes": 256,
                    "pageBytes": 65536,
                    "responseBytes": 98304,
                    "chunkBytes": 65536,
                },
                "cases": entries,
                "negativeCases": failures,
                "storageCallsAtCompletion": completion_storage_calls,
                "effects": {"toolExecutions": len(host.executions), "copies": backend.copies},
            }
        finally:
            server.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "app_server_tool_results.json").write_text(
        json.dumps(produce(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
