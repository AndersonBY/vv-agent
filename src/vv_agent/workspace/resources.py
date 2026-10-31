"""Optional workspace-owned immutable output retention; no kernel copy fallback."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Protocol

from vv_agent.workspace.base import _exclusive_workspace_path_segments

RESOURCE_CHUNK_BYTES = 65_536
MAX_SAFE_INTEGER = 2**53 - 1


@dataclass(frozen=True, slots=True)
class ToolResultResourceExpectation:
    owner: str
    media_type: str
    name: str
    size_bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ToolResultResourceRef:
    workspace_id: str
    owner: str
    locator: str
    size_bytes: int
    sha256: str
    media_type: str
    name: str

    def __post_init__(self):
        for value in (self.workspace_id, self.owner, self.sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid retained resource identity")
        if type(self.size_bytes) is not int or not 0 <= self.size_bytes <= MAX_SAFE_INTEGER:
            raise ValueError("invalid retained resource size")
        if not isinstance(self.locator, str) or not self.locator or len(self.locator.encode()) > 4096:
            raise ValueError("invalid retained locator")
        for value in (self.media_type, self.name):
            if not isinstance(value, str) or len(value.encode()) > 256:
                raise ValueError("invalid retained resource label")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("invalid retained resource")
        return cls(**value)


class OutputRetentionBackend(Protocol):
    @property
    def retained_workspace_id(self) -> str: ...
    def retain_output(self, path: str, *, expected: ToolResultResourceExpectation) -> ToolResultResourceRef | None: ...
    def open_retained(self, ref: ToolResultResourceRef, *, offset: int = 0, length: int | None = None) -> Iterable[bytes]: ...


def private_resource_path(path: str) -> str:
    segments = _exclusive_workspace_path_segments(path)
    if len(segments) < 3 or segments[:2] not in {(".vv-agent", "resources"), (".vv-agent", "artifacts")}:
        raise ValueError("invalid private resource path")
    return path


def validate_retained_workspace(ref: ToolResultResourceRef, workspace_id: str) -> str:
    if ref.workspace_id != workspace_id:
        raise PermissionError("resource belongs to another workspace")
    path = private_resource_path(ref.locator)
    if path.startswith(".vv-agent/resources/") and path.split("/")[2] != ref.owner:
        raise ValueError("resource belongs to another owner")
    return path


def copy_retained_output(backend, path: str, *, expected: ToolResultResourceExpectation) -> ToolResultResourceRef:
    """Called by copying backends only; immutable hosts implement their own capability."""
    import uuid

    _exclusive_workspace_path_segments(path)
    if path.startswith(".vv-agent/artifacts/") and expected.sha256 is not None and expected.size_bytes is not None:
        return ToolResultResourceRef(
            backend.retained_workspace_id,
            expected.owner,
            path,
            expected.size_bytes,
            expected.sha256,
            expected.media_type,
            expected.name,
        )
    locator = f".vv-agent/resources/{expected.owner}/{uuid.uuid4().hex}"
    size = 0
    digest = hashlib.sha256()

    def chunks():
        nonlocal size
        for chunk in backend._read_output_chunks(path):
            if not isinstance(chunk, bytes) or len(chunk) > RESOURCE_CHUNK_BYTES:
                raise ValueError("invalid resource chunk")
            size += len(chunk)
            digest.update(chunk)
            yield chunk

    written = backend.write_bytes_chunks_exclusive(locator, chunks())
    if type(written) is not int or written != size:
        raise OSError("incomplete retained output")
    if (expected.size_bytes is not None and size != expected.size_bytes) or (
        expected.sha256 is not None and digest.hexdigest() != expected.sha256
    ):
        raise ValueError("output changed before retention")
    return ToolResultResourceRef(
        backend.retained_workspace_id,
        expected.owner,
        locator,
        size,
        digest.hexdigest(),
        expected.media_type,
        expected.name,
    )


_VERIFIED: OrderedDict[ToolResultResourceRef, object] = OrderedDict()
_VERIFY_LOCK = RLock()


def verify_retained(ref, version, chunks):
    """Verify a digest once per immutable reference; every range must attest its version.

    The bounded process cache survives host configuration rebuilds. Eviction or process
    restart requires one fresh digest pass. Version changes never refresh a live entry.
    """
    with _VERIFY_LOCK:
        if ref in _VERIFIED:
            if _VERIFIED[ref] != version:
                raise ValueError("stale_reference")
            _VERIFIED.move_to_end(ref)
            return
        size, digest = 0, hashlib.sha256()
        for chunk in chunks():
            size += len(chunk)
            digest.update(chunk)
            if size > ref.size_bytes:
                raise ValueError("stale_reference")
        if size != ref.size_bytes or digest.hexdigest() != ref.sha256:
            raise ValueError("stale_reference")
        _VERIFIED[ref] = version
        # Bound both entry count and memory held by immutable memory-backend versions.
        while len(_VERIFIED) > 1024 or (
            len(_VERIFIED) > 1
            and sum(len(v[1]) for v in _VERIFIED.values() if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], bytes))
            > 268_435_456
        ):
            _VERIFIED.popitem(last=False)


def retained_range(ref, offset, length):
    if type(offset) is not int or not 0 <= offset <= ref.size_bytes:
        raise ValueError("invalid_range")
    if length is not None and (type(length) is not int or length < 0):
        raise ValueError("invalid_range")
    return min(ref.size_bytes - offset, length) if length is not None else ref.size_bytes - offset
