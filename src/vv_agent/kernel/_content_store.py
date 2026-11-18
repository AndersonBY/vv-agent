"""Standard-library contracts for S1 encrypted content storage.

The kernel owns backend-neutral content protocols, the narrow cipher injection
point, and persisted state shapes. Concrete key providers, encryption and
process locks live in the session layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ._content import ContentBindingMismatch, ContentLabels, ContentRef, KeyUnavailable, WrappedDataKey


@runtime_checkable
class ContentStore(Protocol):
    """Backend-neutral encrypted content operations for one session binding."""

    def put(
        self,
        tenant_id: str,
        session_id: str,
        plaintext: bytes,
        labels: ContentLabels,
    ) -> ContentRef: ...

    def get(self, tenant_id: str, session_id: str, ref: ContentRef) -> bytes: ...

    def forget_session(self, tenant_id: str, session_id: str) -> None: ...


class ContentCipher(Protocol):
    """The sole cipher and randomness port used by S1 content storage."""

    def encrypt(self, key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes: ...

    def decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes: ...

    def random_bytes(self, n: int) -> bytes: ...


class ContentState(StrEnum):
    STAGED = "staged"
    LIVE = "live"
    GC_PENDING = "gc_pending"


class SessionKeyState(StrEnum):
    ACTIVE = "active"
    DESTROYING = "destroying"


@dataclass(frozen=True, slots=True)
class SessionKeyRecord:
    """The complete persisted session-key envelope and lifecycle state."""

    tenant_id: str
    session_id: str
    key_id: str
    wrapped_key: bytes = field(repr=False)
    provider_id: str
    state: SessionKeyState

    def __post_init__(self) -> None:
        _text(self.tenant_id, "tenant_id")
        _text(self.session_id, "session_id")
        _text(self.key_id, "key_id")
        _text(self.provider_id, "provider_id")
        if not isinstance(self.wrapped_key, bytes) or not self.wrapped_key:
            raise ContentBindingMismatch("wrapped_key must be non-empty bytes")
        try:
            object.__setattr__(self, "state", SessionKeyState(self.state))
        except ValueError as exc:
            raise KeyUnavailable("session key state is not recognized") from exc


@dataclass(frozen=True, slots=True)
class ContentLink:
    """A verified owner-to-content association; it carries no second ledger."""

    tenant_id: str
    session_id: str
    record_seq: int | None
    input_id: str | None
    slot: str
    ref: ContentRef


@runtime_checkable
class DurableContentStore(ContentStore, Protocol):
    """Encrypted content, key envelopes, and owner links backed by caller SQL."""

    def put(
        self,
        tenant_id: str,
        session_id: str,
        plaintext: bytes,
        labels: ContentLabels,
        *,
        state: ContentState = ContentState.STAGED,
    ) -> ContentRef: ...

    def create_session_key(self, tenant_id: str, session_id: str) -> WrappedDataKey: ...

    def put_session_key(self, key: SessionKeyRecord) -> None: ...

    def link_record(self, tenant_id: str, session_id: str, record_seq: int, ref: ContentRef, slot: str) -> None: ...

    def link_input(self, tenant_id: str, session_id: str, input_id: str, ref: ContentRef, slot: str) -> None: ...

    def read_links(
        self,
        tenant_id: str,
        session_id: str,
        *,
        record_seq: int | None = None,
        input_id: str | None = None,
    ) -> tuple[ContentLink, ...]: ...

    def assert_link(
        self,
        tenant_id: str,
        session_id: str,
        ref: ContentRef,
        slot: str,
        *,
        record_seq: int | None = None,
        input_id: str | None = None,
    ) -> ContentRef: ...

    def collect_garbage(self, tenant_id: str, session_id: str) -> int: ...


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContentBindingMismatch(f"{field_name} must be a non-empty string")
    return value


__all__ = [
    "ContentCipher",
    "ContentLink",
    "ContentState",
    "ContentStore",
    "DurableContentStore",
    "SessionKeyRecord",
    "SessionKeyState",
]
