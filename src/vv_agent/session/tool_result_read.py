"""Authorized reads of immutable receipts; never drives a session."""

from __future__ import annotations

import base64
import hashlib

from vv_agent.app_server.host import ToolResultResourceAuthorizationRequest
from vv_agent.app_server.protocol.tool_result import ToolResultReadResponse, ToolResultRef
from vv_agent.types import ToolExecutionResult
from vv_agent.workspace.resources import ToolResultResourceRef

from .children import child_handles
from .store import Conflict
from .tool_results import PAGE_BYTES, project_tool_result, public_content, resource_entries, result_owner


class ToolResultReadError(ValueError):
    def __init__(self, reason, *, denied=False):
        super().__init__(reason)
        self.reason, self.denied = reason, denied


def validate_child_graph(store, root, sid, *, active=False):
    """Follow immutable admission handles, not child-supplied parent strings alone."""
    visited, chain, root_turn = set(), [], None
    while sid != root:
        if sid in visited:
            raise ToolResultReadError("invalid_result_ref")
        visited.add(sid)
        chain.append(sid)
        try:
            created = store.read(sid, limit=1).records[0].record
            parent = created._payload["parent_session_id"]
            oid = created._payload["parent_operation_id"]
            handle = created._payload["attributes"]["child_handle"]
            if not parent:
                raise KeyError("parent")
            state, _, _ = store.read_state(parent)
            op = state.operations[oid]
            admitted = [
                h
                for attempt in op.attempts.values()
                if attempt.child_handle is not None
                for h in child_handles(attempt.child_handle)
            ]
            if active:
                child_state, _, _ = store.read_state(sid)
                if (
                    state.active_turn_id != op.turn_id
                    or state.closed
                    or state.archived
                    or child_state.active_turn_id != handle["turn_id"]
                ):
                    raise KeyError("inactive admission")
            if (
                handle not in admitted
                or handle["session_id"] != sid
                or handle["delivery_target"]["session_id"] != parent
                or handle["delivery_target"]["operation_id"] != oid
                or handle["delivery_target"]["turn_id"] != op.turn_id
            ):
                raise KeyError("admission")
        except (KeyError, IndexError, Conflict):
            raise ToolResultReadError("invalid_result_ref") from None
        sid = parent
        root_turn = op.turn_id
    return list(reversed(chain)), root_turn


def read_tool_result(adapter, host, connection_id, params):
    store = adapter.kernel.store
    try:
        ref = ToolResultRef.from_dict(params["ref"])
    except (KeyError, ValueError, TypeError):
        raise ToolResultReadError("invalid_result_ref") from None
    sid = params["threadId"]
    if (
        type(params.get("offset")) is not int
        or not 0 <= params["offset"] <= 2**53 - 1
        or type(params.get("limit")) is not int
        or not 1 <= params["limit"] <= PAGE_BYTES
    ):
        raise ToolResultReadError("invalid_range")
    part = params.get("part")
    if not isinstance(part, str):
        raise ToolResultReadError("invalid_result_ref")
    try:
        root = store.read(sid, limit=1).records[0].record
        if "app_server" not in root._payload["attributes"]:
            raise ToolResultReadError("invalid_result_ref")
    except (KeyError, IndexError, Conflict):
        raise ToolResultReadError("invalid_result_ref") from None
    child_chain = []
    root_turn = ref.turn_id
    if ref.session_id != sid:
        if not host.authorize(connection_id, "thread/toolResult/read", params | {"threadId": ref.session_id}):
            raise ToolResultReadError("permission_denied", denied=True)
        child_chain, root_turn = validate_child_graph(store, sid, ref.session_id)
    try:
        rows = store.read(ref.session_id, after_seq=ref.seq - 1, through_seq=ref.seq, limit=1).records
    except (KeyError, Conflict):
        raise ToolResultReadError("result_not_found") from None
    if not rows:
        raise ToolResultReadError("result_not_found")
    stored = rows[0]
    record = stored.record
    if record.kind != "op_completed":
        raise ToolResultReadError("invalid_result_ref")
    # Model receipts have a different result shape and are never tool results.
    try:
        descriptor = project_tool_result(stored)
    except (KeyError, ValueError, TypeError):
        raise ToolResultReadError("invalid_result_ref") from None
    if descriptor["ref"] != ref.to_dict():
        raise ToolResultReadError("invalid_result_ref")
    result = ToolExecutionResult.from_dict(record._payload["result"])
    offset, limit = params["offset"], params["limit"]
    if part == "content":
        content = public_content(result)
        total, digest = len(content), hashlib.sha256(content).hexdigest()
        data = content[offset : offset + limit]
    else:
        matches = [i for i, r in enumerate(descriptor["resources"]) if r["id"] == part]
        if not matches:
            raise ToolResultReadError("invalid_result_ref")
        entry = resource_entries(result)[matches[0]]
        if entry["status"] != "retained":
            raise ToolResultReadError("resource_not_retained")
        # Rebuild dependencies without resume, dispatch, wake, tools or provider submits.
        _, agent, config = adapter._binding(sid, turn_id=root_turn)
        runtime = adapter.kernel.runtime(agent, config)
        if ref.session_id != sid:
            from .delegation import reconstruct

            for child_id in child_chain:
                runtime = reconstruct(runtime, store, child_id)
        backend = runtime.workspace_backend
        if not callable(getattr(backend, "open_retained", None)):
            raise ToolResultReadError("resource_not_retained")
        try:
            if "original_artifact" in entry:
                artifact = entry["original_artifact"]
                retained = ToolResultResourceRef(
                    backend.retained_workspace_id,
                    result_owner(record),
                    artifact.path,
                    artifact.size_bytes,
                    artifact.sha256,
                    artifact.media_type,
                    "full-text.txt",
                )
            else:
                retained = ToolResultResourceRef.from_dict(entry["ref"])
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ToolResultReadError("invalid_result_ref") from None
        if retained.owner != result_owner(record):
            raise ToolResultReadError("invalid_result_ref")
        request = ToolResultResourceAuthorizationRequest(connection_id, sid, ref, backend, retained)
        if not host.authorize_tool_result_resource(request):
            raise ToolResultReadError("permission_denied", denied=True)
        if retained.workspace_id != getattr(backend, "retained_workspace_id", None):
            raise ToolResultReadError("permission_denied", denied=True)
        total, digest = retained.size_bytes, retained.sha256
        if offset > total:
            raise ToolResultReadError("invalid_range")
        page = bytearray()
        try:
            for chunk in backend.open_retained(retained, offset=offset, length=limit):
                if not isinstance(chunk, bytes) or len(chunk) > PAGE_BYTES or len(page) + len(chunk) > limit:
                    raise ToolResultReadError("stale_reference")
                page.extend(chunk)
        except FileNotFoundError:
            raise ToolResultReadError("resource_missing") from None
        except PermissionError:
            raise ToolResultReadError("permission_denied", denied=True) from None
        except (OSError, ValueError) as exc:
            if isinstance(exc, ToolResultReadError):
                raise
            raise ToolResultReadError("stale_reference") from None
        if len(page) != min(limit, total - offset):
            raise ToolResultReadError("stale_reference")
        data = page
    if offset > total:
        raise ToolResultReadError("invalid_range")
    return ToolResultReadResponse(
        ref,
        descriptor["kind"],
        record._payload["context"],
        part,
        base64.b64encode(data).decode("ascii"),
        offset,
        offset + len(data),
        total,
        digest,
        offset + len(data) == total,
    ).to_dict()
