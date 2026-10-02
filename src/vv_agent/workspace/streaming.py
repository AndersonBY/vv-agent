from __future__ import annotations

import codecs
import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vv_agent.workspace.base import WorkspaceBackend


def read_chunks(backend: WorkspaceBackend, path: str) -> Iterable[bytes]:
    reader = getattr(backend, "_read_bytes_chunks", None)
    if reader is not None and "_read_bytes_chunks" in type(backend).__dict__:
        return reader(path)
    # Custom backends retain their existing read contract.
    return (backend.read_bytes(path),)


@dataclass(frozen=True, slots=True)
class TextScan:
    size_bytes: int
    sha256: str
    valid_utf8: bool


def scan_text(backend: WorkspaceBackend, path: str, consume: Callable[[str], None]) -> TextScan:
    """Hash and decode the same byte stream; publish only after full validation."""
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")()
    size = 0
    valid = True
    for chunk in read_chunks(backend, path):
        digest.update(chunk)
        size += len(chunk)
        if valid:
            try:
                text = decoder.decode(chunk)
            except UnicodeDecodeError:
                valid = False
            else:
                consume(text)
    if valid:
        try:
            consume(decoder.decode(b"", final=True))
        except UnicodeDecodeError:
            valid = False
    return TextScan(size, digest.hexdigest(), valid)
