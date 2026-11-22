"""Deadline-bound local transport for the S1 S3 adapter."""

from __future__ import annotations

import json
import multiprocessing
import socket
import struct
import time
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Literal, Protocol

_META_LIMIT = 64 * 1024
_BODY_LIMIT = 64 * 1024 * 1024
_NS = 1_000_000_000
_STARTUP_NS = 5 * _NS
_HEADER = struct.Struct("!IIQ")


@dataclass(frozen=True, slots=True)
class _S3Request:
    operation: Literal["get_bucket_versioning", "put_object", "get_object", "delete_object"]
    bucket: str
    key: str | None = None
    body: bytes | None = None
    if_match: str | None = None
    if_none_match: str | None = None
    ciphertext_size: int | None = None


@dataclass(frozen=True, slots=True)
class _S3Reply:
    status: int | None = None
    version_id: str | None = None
    etag: str | None = None
    delete_marker: bool = False
    versioning: str | None = None
    body: bytes | None = None
    error: Literal["unavailable", "missing", "integrity"] | None = None


class _S3TransportError(RuntimeError):
    pass


class _S3TransportTimeout(_S3TransportError):
    def __init__(self, *, may_have_dispatched: bool) -> None:
        super().__init__("external ciphertext transport deadline expired")
        self.may_have_dispatched = may_have_dispatched


class _S3Transport(Protocol):
    @property
    def endpoint_url(self) -> str: ...

    def execute(self, request: _S3Request, *, deadline_ns: int, authorize: Callable[[], int] | None = None) -> _S3Reply: ...


def _disable_region_redirect(request_dict: dict[str, Any], **_: Any) -> None:
    request_dict["context"].setdefault("s3_redirect", {})["redirected"] = True


def _require_first_request(request: Any, **_: Any) -> None:
    if request.context.get("retries", {}).get("attempt") != 1:
        raise _S3TransportError("external ciphertext request retry was blocked")


class _LimitedResponse:
    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[bytes]:
        size = 0
        try:
            for chunk in self._raw.stream(*args, **kwargs):
                size += len(chunk)
                if size > _META_LIMIT:
                    raise _S3TransportError("external ciphertext response metadata exceeds its limit")
                yield chunk
        finally:
            self._raw.close()


class _LimitedSession:
    def __init__(self, session: Any) -> None:
        self._session = session

    def send(self, request: Any) -> Any:
        streaming = request.stream_output
        # Let Botocore parse its normal response using our bounded metadata stream.
        request.stream_output = True
        response = self._session.send(request)
        if not streaming or response.status_code >= 300:
            response.raw = _LimitedResponse(response.raw)
        return response

    def close(self) -> None:
        self._session.close()


def _configure_client(client: Any) -> None:
    retries = client.meta.config.retries
    if retries is None or retries.get("total_max_attempts") != 1:
        raise ValueError("S3 ciphertext client requires total_max_attempts=1")
    client.meta.events.register_first("needs-retry.s3", _disable_region_redirect, unique_id="vv-agent-s1-fixed-region")
    client.meta.events.register_first("request-created.s3", _require_first_request, unique_id="vv-agent-s1-first-request-only")


def _sdk_request(client: Any, request: _S3Request) -> _S3Reply:
    """Consume and close SDK streaming bodies inside the transport owner."""
    try:
        args: dict[str, Any] = {"Bucket": request.bucket}
        if request.key is not None:
            args["Key"] = request.key
        if request.operation == "put_object":
            assert request.body is not None
            args.update(Body=request.body, ContentLength=len(request.body))
            if request.if_match is not None:
                args["IfMatch"] = request.if_match
            if request.if_none_match is not None:
                args["IfNoneMatch"] = request.if_none_match
        reply = getattr(client, request.operation)(**args)
        status = reply.get("ResponseMetadata", {}).get("HTTPStatusCode")
        version = reply.get("VersionId")
        body = None
        if request.operation == "get_object":
            stream = reply["Body"]
            try:
                size = request.ciphertext_size
                assert size is not None
                if status != 200 or version is not None or reply.get("ContentLength") != size:
                    return _S3Reply(error="integrity")
                body = stream.read(size + 1)
                if not isinstance(body, bytes) or len(body) != size:
                    return _S3Reply(error="integrity")
            finally:
                stream.close()
        return _S3Reply(status, version, reply.get("ETag"), bool(reply.get("DeleteMarker")), reply.get("Status"), body)
    except Exception as exc:
        response = getattr(exc, "response", None)
        missing = isinstance(response, dict) and response.get("Error", {}).get("Code") in {"NoSuchKey", "NotFound", "404"}
        return _S3Reply(error="missing" if missing else "unavailable")


def _remaining(deadline_ns: int | None) -> float | None:
    if deadline_ns is None:
        return None
    remaining = deadline_ns - time.monotonic_ns()
    if remaining <= 0:
        raise TimeoutError
    return remaining / _NS


def _send(channel: socket.socket, data: bytes, deadline_ns: int | None) -> None:
    channel.settimeout(_remaining(deadline_ns))
    channel.sendall(data)


def _read(channel: socket.socket, size: int, deadline_ns: int | None) -> bytes:
    buffer = bytearray(size)
    view = memoryview(buffer)
    while view:
        channel.settimeout(_remaining(deadline_ns))
        received = channel.recv_into(view)
        if not received:
            raise EOFError
        view = view[received:]
    return bytes(buffer)


def _send_frame(channel: socket.socket, fields: dict[str, Any], deadline_ns: int) -> None:
    body = fields.pop("body") or b""
    metadata = json.dumps(fields, separators=(",", ":")).encode()
    _check_sizes(len(metadata), len(body))
    _send(channel, _HEADER.pack(len(metadata), len(body), deadline_ns), deadline_ns)
    _send(channel, metadata, deadline_ns)
    if body:
        _send(channel, body, deadline_ns)


def _check_sizes(metadata_size: int, body_size: int) -> None:
    if not 0 < metadata_size <= _META_LIMIT or not 0 <= body_size <= _BODY_LIMIT:
        raise _S3TransportError("external ciphertext transport frame exceeds its size limit")


def _read_frame(channel: socket.socket, deadline_ns: int | None) -> tuple[dict[str, Any], int]:
    metadata_size, body_size, announced_deadline = _HEADER.unpack(_read(channel, _HEADER.size, deadline_ns))
    _check_sizes(metadata_size, body_size)
    deadline_ns = announced_deadline if deadline_ns is None else min(deadline_ns, announced_deadline)
    fields = json.loads(_read(channel, metadata_size, deadline_ns))
    fields["body"] = _read(channel, body_size, deadline_ns) if body_size else None
    return fields, deadline_ns


def _s3_worker(channel: socket.socket, factory: Callable[[], Any], endpoint_url: str, startup_deadline_ns: int) -> None:
    with channel:
        try:
            client = factory()
            _configure_client(client)
            if client.meta.endpoint_url != endpoint_url:
                raise ValueError("S3 ciphertext client endpoint does not match the configured binding")
            client._endpoint.http_session = _LimitedSession(client._endpoint.http_session)
        except Exception:
            _send(channel, b"E", startup_deadline_ns)
            return
        try:
            _send(channel, b"R", startup_deadline_ns)
            while True:
                fields, deadline_ns = _read_frame(channel, None)
                request = _S3Request(**fields)
                _send(channel, b"R", deadline_ns)
                action = _read(channel, 1, deadline_ns)
                if action == b"C":
                    continue
                if action != b"G":
                    return
                authorized_ns = struct.unpack("!Q", _read(channel, 8, deadline_ns))[0]
                deadline_ns = min(deadline_ns, authorized_ns)
                _remaining(deadline_ns)
                reply = _sdk_request(client, request)
                _send_frame(channel, asdict(reply), deadline_ns)
        except (OSError, EOFError, _S3TransportError):
            return
        finally:
            client.close()


class _S3ProcessTransport:
    """Own one spawned SDK client. Start and close from the host lifecycle.

    Calls fail immediately when the slot is busy or unavailable. After a
    timeout, the host can start a new generation once the old one has exited.
    The factory must keep its credentials local and avoid unmanaged subprocesses.
    """

    def __init__(self, factory: Callable[[], Any], *, endpoint_url: str) -> None:
        self._factory = factory
        self._endpoint_url = endpoint_url
        self._lock = Lock()
        self._channel: socket.socket | None = None
        self._process: Any = None

    @property
    def endpoint_url(self) -> str:
        return self._endpoint_url

    def __enter__(self) -> _S3ProcessTransport:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start(self) -> None:
        with self._lock:
            if self._channel is not None:
                return
            self._reap()
            if self._process is not None:
                raise _S3TransportError("previous S3 transport worker has not exited")
            parent, child = socket.socketpair()
            deadline_ns = time.monotonic_ns() + _STARTUP_NS
            process = multiprocessing.get_context("spawn").Process(
                target=_s3_worker, args=(child, self._factory, self._endpoint_url, deadline_ns), daemon=True
            )
            with child:
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
            self._process = process
            self._channel = parent
            try:
                if _read(parent, 1, deadline_ns) != b"R":
                    raise _S3TransportError("S3 transport worker could not initialize")
            except BaseException:
                self._invalidate()
                raise

    def _invalidate(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        if self._process is not None:
            with suppress(ProcessLookupError):
                self._process.kill()

    def _reap(self, timeout: float = 0) -> None:
        if self._process is not None:
            self._process.join(timeout)
            if self._process.exitcode is not None:
                self._process.close()
                self._process = None

    def close(self) -> None:
        with self._lock:
            self._invalidate()
            self._reap(timeout=1)

    def execute(self, request: _S3Request, *, deadline_ns: int, authorize: Callable[[], int] | None = None) -> _S3Reply:
        if request.ciphertext_size is not None and request.ciphertext_size > _BODY_LIMIT:
            raise _S3TransportError("external ciphertext read exceeds its size limit")
        if request.body is not None and len(request.body) > _BODY_LIMIT:
            raise _S3TransportError("external ciphertext upload exceeds its size limit")
        if not self._lock.acquire(blocking=False):
            raise _S3TransportError("S3 transport slot is busy")
        may_have_dispatched = False
        try:
            channel = self._channel
            if channel is None:
                raise _S3TransportError("S3 transport slot is unavailable")
            if time.monotonic_ns() >= deadline_ns:
                raise _S3TransportTimeout(may_have_dispatched=False)
            try:
                _send_frame(channel, asdict(request), deadline_ns)
                if _read(channel, 1, deadline_ns) != b"R":
                    raise _S3TransportError("S3 transport worker did not prepare the request")
            except TimeoutError:
                self._invalidate()
                raise _S3TransportTimeout(may_have_dispatched=False) from None
            except (OSError, EOFError, _S3TransportError):
                self._invalidate()
                raise _S3TransportError("S3 transport failed before dispatch") from None
            try:
                if authorize is not None:
                    deadline_ns = min(deadline_ns, authorize())
                _remaining(deadline_ns)
            except TimeoutError:
                self._invalidate()
                raise _S3TransportTimeout(may_have_dispatched=False) from None
            except BaseException:
                try:
                    _send(channel, b"C", deadline_ns)
                except OSError:
                    self._invalidate()
                raise
            try:
                # The worker cannot dispatch before receiving the complete grant.
                may_have_dispatched = True
                _send(channel, b"G" + struct.pack("!Q", deadline_ns), deadline_ns)
                fields, _ = _read_frame(channel, deadline_ns)
                _remaining(deadline_ns)
                return _S3Reply(**fields)
            except TimeoutError:
                self._invalidate()
                raise _S3TransportTimeout(may_have_dispatched=may_have_dispatched) from None
            except (OSError, EOFError, _S3TransportError, ValueError, TypeError):
                self._invalidate()
                if time.monotonic_ns() >= deadline_ns:
                    raise _S3TransportTimeout(may_have_dispatched=may_have_dispatched) from None
                raise _S3TransportError("S3 transport failed after dispatch") from None
            except BaseException:
                self._invalidate()
                raise
        finally:
            self._lock.release()
