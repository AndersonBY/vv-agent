"""S1 ciphertext placement over a verified, unversioned S3 endpoint."""

from __future__ import annotations

import re
import secrets
import time
from collections.abc import Callable
from hashlib import sha256
from typing import Any, Protocol, cast

from vv_agent.kernel._allocation_journal import Allocation, AllocationOwner, JournalIdentity
from vv_agent.kernel._content import (
    ContentBindingMismatch,
    ContentCleanupUnavailable,
    ContentIntegrityError,
    ContentUnavailable,
    KeyUnavailable,
)

_LOCATOR_VERSION = b"\x01"
_MARKER_BYTES = 8
_STRONG_ETAG = re.compile(r'"[\x21\x23-\x2b\x2d-\x7e]+"')


class CiphertextTemporarilyUnavailable(KeyUnavailable):
    """Object I/O shares the retry gate for unavailable content dependencies."""


class _ObjectJournal(Protocol):
    @property
    def identity(self) -> JournalIdentity: ...

    def reserve(self, allocation_id: bytes, *, pin_until_ms: int) -> bool: ...

    def bind(self, allocation_id: bytes, owner: AllocationOwner, *, reservation_etag: str) -> bool: ...

    def get_anonymous(self, allocation_id: bytes) -> Allocation | None: ...

    def get_owned(self, allocation_id: bytes, owner: AllocationOwner) -> Allocation | None: ...


def _disable_region_redirect(request_dict: dict[str, Any], **_: Any) -> None:
    # Botocore may perform region discovery before an attempted resend.
    context = request_dict["context"]
    context["s3_redirect"]["redirected"] = True


def _require_first_request(request: Any, **_: Any) -> None:
    if request.context.get("retries", {}).get("attempt") != 1:
        raise CiphertextTemporarilyUnavailable("external ciphertext request retry was blocked")


def _status(reply: dict[str, Any]) -> int | None:
    return reply.get("ResponseMetadata", {}).get("HTTPStatusCode")


def _require_live_pin(pin_until_ms: int) -> None:
    if time.time_ns() // 1_000_000 >= pin_until_ms:
        raise CiphertextTemporarilyUnavailable("external ciphertext allocation pin expired")


class S3CiphertextStore:
    """Use a dedicated S3 client and a journal already verified by the host.

    The caller holds the publication fence for writes and bound cleanup. Its
    fence callback must also check the remaining monotonic operation budget.
    Socket timeouts alone do not enforce that budget while a request is running.
    """

    def __init__(self, client: Any, *, bucket: str, journal: _ObjectJournal) -> None:
        if not bucket:
            raise ValueError("S3 ciphertext bucket is required")
        retries = client.meta.config.retries
        if retries is None or retries.get("total_max_attempts") != 1:
            raise ValueError("S3 ciphertext client requires total_max_attempts=1")
        client.meta.events.register_first("needs-retry.s3", _disable_region_redirect, unique_id="vv-agent-s1-fixed-region")
        client.meta.events.register_first(
            "request-created.s3", _require_first_request, unique_id="vv-agent-s1-first-request-only"
        )
        self._client = client
        self._bucket = bucket
        namespace = sha256(journal.identity.namespace.encode()).hexdigest()
        self._prefix = f"vv-agent-s1/{namespace}/alloc/"
        self._journal = journal
        self._require_unversioned()

    @property
    def _storage_identity(self) -> tuple[str, str, str]:
        return self._client.meta.endpoint_url, self._bucket, self._prefix

    def _uses_journal(self, journal: object) -> bool:
        return self._journal is journal

    def _require_unversioned(self) -> None:
        try:
            reply = self._client.get_bucket_versioning(Bucket=self._bucket)
            if _status(reply) == 200 and reply.get("Status") is None:
                return
        except Exception:
            pass
        raise ContentCleanupUnavailable("external ciphertext requires a verified unversioned bucket")

    def _key(self, allocation_id: bytes) -> str:
        return self._prefix + allocation_id.hex()

    def _lookup(self, allocation_id: bytes, *, owner: AllocationOwner | None = None) -> Allocation | None:
        try:
            if owner is None:
                return self._journal.get_anonymous(allocation_id)
            return self._journal.get_owned(allocation_id, owner)
        except ContentUnavailable:
            raise CiphertextTemporarilyUnavailable("external ciphertext journal is unavailable") from None

    def put(
        self,
        owner: AllocationOwner,
        ciphertext: bytes,
        *,
        pin_until_ms: int,
        check_fence: Callable[[], None],
    ) -> bytes:
        """Reserve one fresh anonymous key, then publish over its acknowledged marker."""
        if not isinstance(ciphertext, bytes) or len(ciphertext) <= _MARKER_BYTES:
            raise ValueError("ciphertext must be longer than the reservation marker")
        check_fence()
        allocation_id = secrets.token_bytes(32)
        marker = secrets.token_bytes(_MARKER_BYTES)
        key = self._key(allocation_id)
        try:
            reserved = self._journal.reserve(allocation_id, pin_until_ms=pin_until_ms)
        except ContentUnavailable:
            check_fence()
            _require_live_pin(pin_until_ms)
            reserved = self._lookup(allocation_id) == Allocation(
                self._journal.identity.journal_id, allocation_id, "unbound", pin_until_ms
            )
        if not reserved:
            raise CiphertextTemporarilyUnavailable("external ciphertext reservation is unavailable")

        check_fence()
        _require_live_pin(pin_until_ms)
        try:
            reply = self._client.put_object(Bucket=self._bucket, Key=key, Body=marker, ContentLength=len(marker), IfNoneMatch="*")
            etag = reply.get("ETag")
            acknowledged = (
                _status(reply) == 200
                and reply.get("VersionId") is None
                and isinstance(etag, str)
                and etag != '"*"'
                and _STRONG_ETAG.fullmatch(etag) is not None
            )
        except Exception:
            acknowledged = False
        if not acknowledged:
            raise CiphertextTemporarilyUnavailable("external ciphertext marker did not confirm success")
        etag = cast(str, etag)

        check_fence()
        try:
            bound = self._journal.bind(allocation_id, owner, reservation_etag=etag)
        except ContentUnavailable:
            check_fence()
            _require_live_pin(pin_until_ms)
            try:
                bound = self._lookup(allocation_id, owner=owner) == Allocation(
                    self._journal.identity.journal_id, allocation_id, "bound", pin_until_ms, owner, etag
                )
            except ContentBindingMismatch:
                bound = False
        if not bound:
            raise CiphertextTemporarilyUnavailable("external ciphertext binding is unavailable")

        check_fence()
        _require_live_pin(pin_until_ms)
        try:
            reply = self._client.put_object(
                Bucket=self._bucket, Key=key, Body=ciphertext, ContentLength=len(ciphertext), IfMatch=etag
            )
            acknowledged = _status(reply) == 200 and reply.get("VersionId") is None
        except Exception:
            acknowledged = False
        if not acknowledged:
            raise CiphertextTemporarilyUnavailable("external ciphertext upload did not confirm success")
        check_fence()
        _require_live_pin(pin_until_ms)
        return _LOCATOR_VERSION + allocation_id

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
            reply = self._client.get_object(Bucket=self._bucket, Key=self._key(allocation_id))
            stream = reply["Body"]
            try:
                if _status(reply) != 200 or reply.get("VersionId") is not None:
                    raise ContentIntegrityError("external ciphertext reply is invalid")
                if reply.get("ContentLength") != ciphertext_size:
                    raise ContentIntegrityError("external ciphertext size does not match its reference")
                ciphertext = stream.read(ciphertext_size + 1)
                if not isinstance(ciphertext, bytes) or len(ciphertext) != ciphertext_size:
                    raise ContentIntegrityError("external ciphertext body length does not match its reference")
                return ciphertext
            finally:
                stream.close()
        except ContentIntegrityError:
            raise
        except Exception as exc:
            response = getattr(exc, "response", {})
            if isinstance(response, dict) and response.get("Error", {}).get("Code") in {"NoSuchKey", "NotFound", "404"}:
                raise ContentUnavailable("external ciphertext is unavailable") from None
            raise CiphertextTemporarilyUnavailable("external ciphertext read did not confirm success") from None

    def delete_claimed(self, claim: Allocation, *, check_fence: Callable[[], None]) -> None:
        """Delete an exact committed claim without completing the journal entry.

        A DELETE acknowledgement alone cannot exclude a late anonymous marker PUT.
        """
        check_fence()
        try:
            current = self._lookup(claim.allocation_id, owner=claim.owner)
        except (ContentBindingMismatch, CiphertextTemporarilyUnavailable, ValueError):
            raise ContentCleanupUnavailable("external ciphertext journal claim is unavailable") from None
        if claim.state != "reaping" or current != claim:
            raise ContentCleanupUnavailable("external ciphertext deletion requires an exact cleanup claim")
        self._require_unversioned()
        check_fence()
        try:
            reply = self._client.delete_object(Bucket=self._bucket, Key=self._key(claim.allocation_id))
            acknowledged = _status(reply) == 204 and reply.get("VersionId") is None and not reply.get("DeleteMarker")
        except Exception:
            acknowledged = False
        if not acknowledged:
            raise ContentCleanupUnavailable("external ciphertext deletion did not confirm success")
        check_fence()
