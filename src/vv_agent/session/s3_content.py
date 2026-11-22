"""S1 ciphertext placement over a verified, unversioned S3 endpoint."""

from __future__ import annotations

import re
import secrets
import time
from collections.abc import Callable
from hashlib import sha256
from typing import Protocol

from vv_agent.kernel._allocation_journal import Allocation, AllocationOwner, JournalIdentity
from vv_agent.kernel._content import (
    ContentBindingMismatch,
    ContentCleanupUnavailable,
    ContentIntegrityError,
    ContentUnavailable,
    KeyUnavailable,
)

from ._s3_transport import _disable_region_redirect as _disable_region_redirect
from ._s3_transport import _S3Reply, _S3Request, _S3Transport, _S3TransportError, _S3TransportTimeout

_OPERATION_NS = 5_000_000_000
_LOCATOR_VERSION = b"\x01"
_MARKER_BYTES = 8
_STRONG_ETAG = re.compile(r'"[\x21\x23-\x2b\x2d-\x7e]+"')


class CiphertextTemporarilyUnavailable(KeyUnavailable):
    """Object I/O shares the retry gate for unavailable content dependencies."""

    def __init__(self, message: str, *, may_have_dispatched: bool | None = None) -> None:
        super().__init__(message)
        self.may_have_dispatched = may_have_dispatched


class _ObjectJournal(Protocol):
    @property
    def identity(self) -> JournalIdentity: ...

    def reserve(self, allocation_id: bytes, *, pin_until_ms: int) -> bool: ...

    def bind(self, allocation_id: bytes, owner: AllocationOwner, *, reservation_etag: str) -> bool: ...

    def get_anonymous(self, allocation_id: bytes) -> Allocation | None: ...

    def get_owned(self, allocation_id: bytes, owner: AllocationOwner) -> Allocation | None: ...

    def _live_allocation_ms(self, expected: Allocation) -> int | None: ...


class S3CiphertextStore:
    """Use a host-owned deadline-bound transport and a verified journal."""

    def __init__(
        self,
        transport: _S3Transport,
        *,
        bucket: str,
        journal: _ObjectJournal,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not bucket:
            raise ValueError("S3 ciphertext bucket is required")
        self._transport = transport
        self._bucket = bucket
        namespace = sha256(journal.identity.namespace.encode()).hexdigest()
        self._prefix = f"vv-agent-s1/{namespace}/alloc/"
        self._journal = journal
        self._monotonic_ns = monotonic_ns
        self._require_unversioned(time.monotonic_ns() + _OPERATION_NS)

    @property
    def _storage_identity(self) -> tuple[str, str, str]:
        return self._transport.endpoint_url, self._bucket, self._prefix

    def _uses_journal(self, journal: object) -> bool:
        return self._journal is journal

    @staticmethod
    def _transport_deadline(deadline_ns: int | None, clock_ns: Callable[[], int]) -> int:
        started_ns = time.monotonic_ns()
        remaining_ns = _OPERATION_NS if deadline_ns is None else min(_OPERATION_NS, deadline_ns - clock_ns())
        return started_ns + remaining_ns

    def _require_unversioned(self, deadline_ns: int) -> None:
        try:
            reply = self._transport.execute(_S3Request("get_bucket_versioning", self._bucket), deadline_ns=deadline_ns)
        except _S3TransportError as exc:
            raise ContentCleanupUnavailable("external ciphertext requires a verified unversioned bucket") from exc
        if reply.status != 200 or reply.versioning is not None:
            raise ContentCleanupUnavailable("external ciphertext requires a verified unversioned bucket")

    def _put_object(self, request: _S3Request, *, deadline_ns: int, authorize: Callable[[], int]) -> _S3Reply:
        try:
            return self._transport.execute(request, deadline_ns=deadline_ns, authorize=authorize)
        except _S3TransportTimeout as exc:
            raise CiphertextTemporarilyUnavailable(
                "external ciphertext write deadline expired", may_have_dispatched=exc.may_have_dispatched
            ) from None
        except _S3TransportError as exc:
            raise CiphertextTemporarilyUnavailable(
                "external ciphertext transport is unavailable", may_have_dispatched=exc.may_have_dispatched
            ) from None

    def _key(self, allocation_id: bytes) -> str:
        return self._prefix + allocation_id.hex()

    def _lookup(self, allocation_id: bytes, *, owner: AllocationOwner | None = None) -> Allocation | None:
        try:
            if owner is None:
                return self._journal.get_anonymous(allocation_id)
            return self._journal.get_owned(allocation_id, owner)
        except ContentUnavailable:
            raise CiphertextTemporarilyUnavailable("external ciphertext journal is unavailable") from None

    def _require_live_pin(self, deadline_ns: int) -> None:
        if self._monotonic_ns() >= deadline_ns:
            raise CiphertextTemporarilyUnavailable("external ciphertext allocation pin expired")

    def _confirm_live_pin(self, expected: Allocation, check_fence: Callable[[], None], deadline_ns: int | None = None) -> int:
        check_fence()
        if deadline_ns is not None:
            self._require_live_pin(deadline_ns)
        started_ns = self._monotonic_ns()
        try:
            remaining_ms = self._journal._live_allocation_ms(expected)
        except ContentUnavailable:
            raise CiphertextTemporarilyUnavailable("external ciphertext journal is unavailable") from None
        if remaining_ms is None:
            raise CiphertextTemporarilyUnavailable("external ciphertext allocation could not be confirmed with a live pin")
        # Account for the whole journal round trip and the clock's millisecond resolution.
        observed_deadline_ns = started_ns + (remaining_ms - 1) * 1_000_000
        deadline_ns = min(observed_deadline_ns, deadline_ns) if deadline_ns is not None else observed_deadline_ns
        check_fence()
        self._require_live_pin(deadline_ns)
        return deadline_ns

    def put(
        self,
        owner: AllocationOwner,
        ciphertext: bytes,
        *,
        pin_until_ms: int,
        check_fence: Callable[[], None],
        deadline_ns: int | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> bytes:
        """Reserve one fresh anonymous key, then publish over its acknowledged marker."""
        if not isinstance(ciphertext, bytes) or len(ciphertext) <= _MARKER_BYTES:
            raise ValueError("ciphertext must be longer than the reservation marker")
        transport_deadline_ns = self._transport_deadline(deadline_ns, clock_ns)
        self._require_before_transport_deadline(transport_deadline_ns)
        check_fence()
        allocation_id = secrets.token_bytes(32)
        marker = secrets.token_bytes(_MARKER_BYTES)
        key = self._key(allocation_id)
        try:
            if not self._journal.reserve(allocation_id, pin_until_ms=pin_until_ms):
                raise CiphertextTemporarilyUnavailable("external ciphertext reservation is unavailable")
        except ContentUnavailable:
            # The exact live read below determines whether the reservation committed.
            pass

        expected = Allocation(self._journal.identity.journal_id, allocation_id, "unbound", pin_until_ms)
        pin_deadline_ns: int | None = None

        def authorize() -> int:
            nonlocal pin_deadline_ns, transport_deadline_ns
            pin_deadline_ns = self._confirm_live_pin(expected, check_fence, pin_deadline_ns)
            transport_deadline_ns = min(
                transport_deadline_ns,
                self._transport_deadline(pin_deadline_ns, self._monotonic_ns),
                self._transport_deadline(deadline_ns, clock_ns),
            )
            return transport_deadline_ns

        reply = self._put_object(
            _S3Request("put_object", self._bucket, key, body=marker, if_none_match="*"),
            deadline_ns=transport_deadline_ns,
            authorize=authorize,
        )
        etag = reply.etag
        if (
            reply.status != 200
            or reply.version_id is not None
            or etag is None
            or etag == '"*"'
            or not _STRONG_ETAG.fullmatch(etag)
        ):
            raise CiphertextTemporarilyUnavailable("external ciphertext marker did not confirm success", may_have_dispatched=True)
        check_fence()
        assert pin_deadline_ns is not None
        self._require_live_pin(pin_deadline_ns)
        self._require_before_transport_deadline(transport_deadline_ns)
        try:
            if not self._journal.bind(allocation_id, owner, reservation_etag=etag):
                raise CiphertextTemporarilyUnavailable("external ciphertext binding is unavailable")
        except ContentUnavailable:
            # Binding may have committed even if its acknowledgement was lost.
            pass

        expected = Allocation(self._journal.identity.journal_id, allocation_id, "bound", pin_until_ms, owner, etag)
        reply = self._put_object(
            _S3Request("put_object", self._bucket, key, body=ciphertext, if_match=etag),
            deadline_ns=transport_deadline_ns,
            authorize=authorize,
        )
        if reply.status != 200 or reply.version_id is not None:
            raise CiphertextTemporarilyUnavailable("external ciphertext upload did not confirm success", may_have_dispatched=True)
        self._confirm_live_pin(expected, check_fence, pin_deadline_ns)
        transport_deadline_ns = min(transport_deadline_ns, self._transport_deadline(deadline_ns, clock_ns))
        self._require_before_transport_deadline(transport_deadline_ns)
        return _LOCATOR_VERSION + allocation_id

    @staticmethod
    def _require_before_transport_deadline(deadline_ns: int) -> None:
        if time.monotonic_ns() >= deadline_ns:
            raise CiphertextTemporarilyUnavailable("external ciphertext write deadline expired")

    def get(self, owner: AllocationOwner, locator: bytes, *, ciphertext_size: int) -> bytes:
        if not isinstance(locator, bytes) or len(locator) != 33 or not locator.startswith(_LOCATOR_VERSION):
            raise ContentBindingMismatch("external ciphertext locator is invalid")
        allocation_id = locator[1:]
        current = self._lookup(allocation_id, owner=owner)
        if current is None or current.state != "bound":
            raise ContentUnavailable("external ciphertext allocation is unavailable")
        if current.owner != owner:
            raise ContentBindingMismatch("external ciphertext locator does not match its exact owner")
        if type(ciphertext_size) is not int or ciphertext_size <= _MARKER_BYTES:
            raise ContentIntegrityError("external ciphertext size is invalid")
        try:
            reply = self._transport.execute(
                _S3Request("get_object", self._bucket, self._key(allocation_id), ciphertext_size=ciphertext_size),
                deadline_ns=time.monotonic_ns() + _OPERATION_NS,
            )
        except _S3TransportError:
            raise CiphertextTemporarilyUnavailable("external ciphertext read did not confirm success") from None
        if reply.error == "missing":
            raise ContentUnavailable("external ciphertext is unavailable")
        if reply.error == "integrity":
            raise ContentIntegrityError("external ciphertext reply does not match its reference")
        if reply.status != 200 or reply.body is None:
            raise CiphertextTemporarilyUnavailable("external ciphertext read did not confirm success")
        if reply.version_id is not None or len(reply.body) != ciphertext_size:
            raise ContentIntegrityError("external ciphertext reply does not match its reference")
        return reply.body

    def delete_claimed(
        self,
        claim: Allocation,
        *,
        check_fence: Callable[[], None],
        deadline_ns: int | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        """Delete an exact committed claim without completing the journal entry.

        A DELETE acknowledgement alone cannot exclude a late anonymous marker PUT.
        """
        transport_deadline_ns = self._transport_deadline(deadline_ns, clock_ns)
        check_fence()
        try:
            current = self._lookup(claim.allocation_id, owner=claim.owner)
        except (ContentBindingMismatch, CiphertextTemporarilyUnavailable, ValueError):
            raise ContentCleanupUnavailable("external ciphertext journal claim is unavailable") from None
        if claim.state != "reaping" or current != claim:
            raise ContentCleanupUnavailable("external ciphertext deletion requires an exact cleanup claim")
        self._require_unversioned(transport_deadline_ns)

        def authorize() -> int:
            check_fence()
            return min(transport_deadline_ns, self._transport_deadline(deadline_ns, clock_ns))

        try:
            reply = self._transport.execute(
                _S3Request("delete_object", self._bucket, self._key(claim.allocation_id)),
                deadline_ns=transport_deadline_ns,
                authorize=authorize,
            )
        except _S3TransportError as exc:
            raise ContentCleanupUnavailable("external ciphertext deletion did not confirm success") from exc
        if reply.status != 204 or reply.version_id is not None or reply.delete_marker:
            raise ContentCleanupUnavailable("external ciphertext deletion did not confirm success")
        check_fence()
        if time.monotonic_ns() >= transport_deadline_ns or (deadline_ns is not None and clock_ns() >= deadline_ns):
            raise ContentCleanupUnavailable("external ciphertext deletion deadline expired")
