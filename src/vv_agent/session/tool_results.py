"""One pure, bounded tool-result projection and workspace-owned preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath

from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.types import ToolExecutionResult
from vv_agent.workspace.resources import ToolResultResourceExpectation, ToolResultResourceRef

PREVIEW_BYTES = 2048
INLINE_BYTES = 8192
DESCRIPTOR_BYTES = 16_384
PAGE_BYTES = 65_536
RESPONSE_BYTES = 98_304
MAX_RESOURCES = 8


def bounded_label(value, limit=256):
    return str(value or "").encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def result_owner(record):
    return hashlib.sha256(
        canonical_json_bytes([record.session_id, record.turn_id, record.operation_id, record.attempt])
    ).hexdigest()


def result_kind(result):
    kind = result.metadata.get("output_type", "text")
    return kind if isinstance(kind, str) and kind in {"text", "json", "image", "file", "error"} else "text"


def _reject_json_constant(value):
    raise ValueError("non-finite JSON constant")


def validate_output_body(result):
    kind = result_kind(result)
    if kind == "text" or result.truncated:
        return
    body = json.loads(result.content, parse_constant=_reject_json_constant)
    if kind == "json":
        return
    if not isinstance(body, dict):
        raise ValueError("structured tool body must be an object")
    if kind in {"file", "image"}:
        fields = {"path", "mime_type"} | ({"url"} if kind == "image" else set())
        if set(body) != fields or any(value is not None and not isinstance(value, str) for value in body.values()):
            raise ValueError("invalid file/image body")
        if kind == "file" and not body["path"]:
            raise ValueError("file output needs a path")
    elif (
        set(body) != {"ok", "error", "error_code", "retryable"}
        or body["ok"] is not False
        or not isinstance(body["error"], str)
        or (body["error_code"] is not None and not isinstance(body["error_code"], str))
        or type(body["retryable"]) is not bool
    ):
        raise ValueError("invalid error body")


def output_location(result):
    if result_kind(result) not in {"file", "image"}:
        return None
    try:
        value = json.loads(result.content)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def prepare_tool_resources(record, raw, backend):
    result = ToolExecutionResult.from_dict(raw)
    metadata = dict(result.metadata)
    # Only trusted preparation may write retained locators; provider metadata is not authority.
    metadata.pop("app_server_resources", None)
    candidates = []
    location = output_location(result)
    if location is not None:
        path = location.get("path")
        candidates.append((path, location.get("mime_type") or "application/octet-stream", None, None, False))
    if result.artifact is not None:
        artifact = result.artifact
        candidates.append((artifact.path, artifact.media_type, artifact.size_bytes, artifact.sha256, True))
    if not candidates:
        return provider_result_identity(raw)
    resources = []
    retain = getattr(backend, "retain_output", None)
    opener = getattr(backend, "open_retained", None)
    for path, media_type, size, sha, artifact in candidates[:MAX_RESOURCES]:
        entry = {
            "status": "not_retained",
            "name": bounded_label(PurePosixPath(path.replace("\\", "/")).name if isinstance(path, str) else "image"),
            "media_type": bounded_label(media_type),
            "artifact": artifact,
        }
        if isinstance(path, str) and callable(retain) and callable(opener):
            try:
                expected = ToolResultResourceExpectation(result_owner(record), entry["media_type"], entry["name"], size, sha)
                retained = retain(path, expected=expected)
                if retained is not None:
                    retained = ToolResultResourceRef.from_dict(retained.to_dict())
                    if (
                        retained.owner != expected.owner
                        or retained.workspace_id != getattr(backend, "retained_workspace_id", None)
                        or retained.media_type != expected.media_type
                        or retained.name != expected.name
                        or (size is not None and retained.size_bytes != size)
                        or (sha is not None and retained.sha256 != sha)
                    ):
                        raise ValueError("invalid retained output identity")
                    entry.update(status="retained", ref=retained.to_dict())
            except Exception:
                # Retention cannot turn a completed external effect into a tool retry.
                entry["status"] = "unavailable"
        resources.append(entry)
    metadata["app_server_resources"] = resources
    return raw | {"metadata": metadata}


def provider_result_identity(raw):
    """Retention is kernel-owned enrichment, not part of the provider's original receipt."""
    if not isinstance(raw, dict) or not isinstance(raw.get("metadata"), dict):
        return raw
    result = dict(raw)
    metadata = {k: v for k, v in raw["metadata"].items() if k != "app_server_resources"}
    if metadata:
        result["metadata"] = metadata
    else:
        result.pop("metadata", None)
    return result


def resource_entries(result):
    entries = result.metadata.get("app_server_resources")
    if isinstance(entries, list):
        return entries[:MAX_RESOURCES]
    entries = []
    location = output_location(result)
    if location is not None:
        path = location.get("path")
        entries.append(
            {
                "status": "not_retained",
                "name": bounded_label(PurePosixPath(path).name if isinstance(path, str) else "image"),
                "media_type": bounded_label(location.get("mime_type") or "application/octet-stream"),
                "artifact": False,
            }
        )
    if result.artifact is not None:
        entries.append(
            {
                "status": "retained",
                "name": "full-text.txt",
                "media_type": result.artifact.media_type,
                "artifact": True,
                "original_artifact": result.artifact,
            }
        )
    return entries


def public_resources(result):
    resources = []
    for index, entry in enumerate(resource_entries(result)):
        resource = {
            "id": f"resource:{index}",
            "status": entry["status"],
            "name": bounded_label(entry["name"]),
            "mediaType": bounded_label(entry["media_type"]),
            "fullText": bool(entry["artifact"]),
        }
        if entry["status"] == "retained":
            retained = entry.get("ref") or {
                "size_bytes": result.artifact.size_bytes,
                "sha256": result.artifact.sha256,
            }
            resource.update(byteLength=retained["size_bytes"], sha256=retained["sha256"])
        resources.append(resource)
    return resources


def public_content(result):
    if result_kind(result) in {"file", "image"}:
        return canonical_json_bytes({"resources": public_resources(result)})
    return result.content.encode("utf-8")


def project_tool_result(stored):
    from vv_agent.app_server.protocol.tool_result import ToolResultRef, ToolResultSummary

    record = stored.record
    result = ToolExecutionResult.from_dict(record._payload["result"])
    content = public_content(result)
    preview = content[:PREVIEW_BYTES].decode("utf-8", errors="ignore")
    descriptor = ToolResultSummary(
        kind=result_kind(result),
        preview=preview,
        byte_length=len(content),
        truncated=bool(result.truncated or len(preview.encode("utf-8")) < len(content)),
        ref=ToolResultRef(
            record.session_id,
            record.turn_id,
            record.operation_id,
            record.attempt,
            result.tool_call_id,
            stored.seq,
            hashlib.sha256(record.encode()).hexdigest(),
        ),
        resources=public_resources(result),
        context=record._payload["context"],
    ).to_dict()
    if result.cursor is not None:
        cursor = {"kind": result.cursor.kind, "offsetChars": result.cursor.offset_chars, "sha256": result.cursor.sha256}
        if len(canonical_json_bytes(cursor)) <= INLINE_BYTES:
            # The cursor is a read-file position, not a path or download authority.
            descriptor["cursor"] = cursor
    if descriptor["kind"] in {"json", "error"} and not result.truncated and len(content) <= INLINE_BYTES:
        try:
            value = json.loads(content)
            if len(canonical_json_bytes(value)) <= INLINE_BYTES:
                descriptor["value"] = value
        except (ValueError, TypeError):
            pass
    for optional in ("cursor", "value", "preview"):
        if len(canonical_json_bytes(descriptor)) <= DESCRIPTOR_BYTES:
            break
        if optional == "preview":
            descriptor["preview"] = ""
            descriptor["truncated"] = True
        else:
            descriptor.pop(optional, None)
    if len(canonical_json_bytes(descriptor)) > DESCRIPTOR_BYTES:
        raise ValueError("result identity exceeds descriptor bound")
    return descriptor
