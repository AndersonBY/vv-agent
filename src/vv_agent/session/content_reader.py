"""Owner-bound readers for the durable S1 content adapters."""

from __future__ import annotations

from typing import Any

from vv_agent.kernel._content import (
    ContentBindingMismatch,
    ContentRef,
    ContentStoreError,
    ContentUnavailable,
)
from vv_agent.kernel._content import (
    ContentIntegrityError as BackendContentIntegrityError,
)
from vv_agent.kernel._content import (
    KeyUnavailable as BackendKeyUnavailable,
)
from vv_agent.kernel._content_store import DurableContentStore
from vv_agent.kernel._hydration import (
    HydrationError,
    IntegrityFailure,
    KeyUnavailable,
    MissingContent,
    MissingContentLink,
)

_RECORD_PREFIX = "record:"
_INPUT_PREFIX = "input:"


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def record_owner(record_seq: int) -> str:
    """Name a content owner by its committed record sequence."""

    if type(record_seq) is not int or record_seq < 1:
        raise ValueError("record_seq must be a positive integer")
    return f"{_RECORD_PREFIX}{record_seq}"


def input_owner(input_id: str) -> str:
    """Name a content owner by its stable inbox input id."""

    return f"{_INPUT_PREFIX}{_non_empty(input_id, 'input_id')}"


def owner_scope(owner: str) -> tuple[str, Any]:
    """Return the store-owned record or input identity encoded by ``owner``."""

    owner = _non_empty(owner, "owner")
    if owner.startswith(_RECORD_PREFIX):
        value = owner.removeprefix(_RECORD_PREFIX)
        if not value.isascii() or not value.isdecimal():
            raise MissingContentLink("content owner is not a valid record sequence")
        record_seq = int(value)
        if record_seq < 1 or str(record_seq) != value:
            raise MissingContentLink("content owner is not a canonical record sequence")
        return "record_seq", record_seq
    if owner.startswith(_INPUT_PREFIX):
        input_id = owner.removeprefix(_INPUT_PREFIX)
        if not input_id:
            raise MissingContentLink("content owner has no input id")
        return "input_id", input_id
    raise MissingContentLink("content owner must be a record sequence or input id")


class DurableContentReader:
    """Bind one durable content adapter to one tenant and session.

    The adapter must use a caller-owned store or transaction. This reader only
    verifies the owner link and reads the encrypted object; it never opens a
    connection, starts a transaction, or retains plaintext.
    """

    def __init__(self, store: DurableContentStore, *, tenant_id: str, session_id: str) -> None:
        if not callable(getattr(store, "assert_link", None)) or not callable(getattr(store, "get", None)):
            raise TypeError("store must implement DurableContentStore")
        self._store = store
        self.tenant_id = _non_empty(tenant_id, "tenant_id")
        self.session_id = _non_empty(session_id, "session_id")

    @property
    def binding(self) -> tuple[str, str]:
        return self.tenant_id, self.session_id

    def read(
        self,
        tenant_id: str,
        session_id: str,
        owner: str,
        slot: str,
        ref: ContentRef,
    ) -> bytes:
        if (tenant_id, session_id) != self.binding:
            raise MissingContentLink("content reader session binding does not match")
        owner_kind, owner_value = owner_scope(owner)
        owner_args = {owner_kind: owner_value}

        try:
            self._store.assert_link(tenant_id, session_id, ref, slot, **owner_args)
        except (ContentUnavailable, ContentBindingMismatch) as exc:
            raise MissingContentLink(f"content owner link is unavailable: {owner}/{slot}") from exc
        except BackendKeyUnavailable as exc:
            raise KeyUnavailable("session content key is unavailable") from exc
        except BackendContentIntegrityError as exc:
            raise IntegrityFailure("content failed integrity validation") from exc
        except ContentStoreError as exc:
            raise HydrationError("content link validation failed") from exc

        try:
            return self._store.get(tenant_id, session_id, ref)
        except ContentUnavailable as exc:
            raise MissingContent(f"content is unavailable: {owner}/{slot}") from exc
        except ContentBindingMismatch as exc:
            raise MissingContentLink("content binding does not match the session") from exc
        except BackendKeyUnavailable as exc:
            raise KeyUnavailable("session content key is unavailable") from exc
        except BackendContentIntegrityError as exc:
            raise IntegrityFailure("content failed integrity validation") from exc
        except ContentStoreError as exc:
            raise HydrationError("content read failed") from exc


__all__ = ["DurableContentReader", "input_owner", "owner_scope", "record_owner"]
