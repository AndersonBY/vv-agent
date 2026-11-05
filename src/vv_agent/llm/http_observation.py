"""Ephemeral HTTP evidence for one model invocation; never logs or persists payloads."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

import httpx2
from anthropic import DEFAULT_CONNECTION_LIMITS as ANTHROPIC_CONNECTION_LIMITS
from httpx2._exceptions import request_context
from openai import DefaultHttpxClient as OpenAIDefaultHttpxClient

if TYPE_CHECKING:
    # The public alias is annotated as bare Client, hiding its cooperative initializer.
    from anthropic._base_client import _DefaultHttpxClient as AnthropicDefaultHttpxClient
else:
    from anthropic import DefaultHttpxClient as AnthropicDefaultHttpxClient

_PREFETCH_SECONDS = 10.0
_PREFETCH_BYTES = 1024 * 1024


class _PrefetchedStream(httpx2.SyncByteStream):
    """Replay raw bytes or the original error after a bounded decoding probe.

    The hook reads synchronously with a shortened native recv timeout. After the
    wall deadline or size bound, consumption resumes the same raw iterator.
    """

    def __init__(self, response: httpx2.Response, seconds: float) -> None:
        assert isinstance(response.stream, httpx2.SyncByteStream)
        self.stream = response.stream
        self.iterator = iter(self.stream)
        self.prefix: list[bytes] = []
        self.error: BaseException | None = None
        self.content: bytes | None = None
        self.raw_size = 0
        self.deadline = monotonic() + seconds
        self._read(response)

    def _read(self, response: httpx2.Response) -> None:
        decoded: list[bytes] = []
        decoded_size = 0
        try:
            # Use HTTPX's real decoder, but leave the response unconsumed for SDKs.
            probe = httpx2.Response(response.status_code, headers=response.headers, stream=httpx2.ByteStream(b""))
            decoder = probe._get_content_decoder()
            with request_context(request=response.request):
                while monotonic() < self.deadline:
                    try:
                        chunk = next(self.iterator)
                    except StopIteration:
                        if monotonic() >= self.deadline:
                            return
                        for tail in decoder.flush():
                            decoded_size += len(tail)
                            if decoded_size > _PREFETCH_BYTES or monotonic() >= self.deadline:
                                return
                            decoded.append(tail)
                        if monotonic() >= self.deadline:
                            return
                        self.content = b"".join(decoded)
                        return
                    self.prefix.append(chunk)
                    self.raw_size += len(chunk)
                    if self.raw_size > _PREFETCH_BYTES or monotonic() >= self.deadline:
                        return
                    for part in decoder.decode(chunk):
                        decoded_size += len(part)
                        if decoded_size > _PREFETCH_BYTES or monotonic() >= self.deadline:
                            return
                        decoded.append(part)
        except BaseException as error:
            self.error = error

    def __iter__(self):
        yield from self.prefix
        self.prefix.clear()
        if self.error is not None:
            raise self.error
        yield from self.iterator

    def close(self) -> None:
        self.stream.close()


@dataclass
class HttpObservation:
    outcomes: list[int | str | None] = field(default_factory=list)
    uncertain: bool = False
    pending: set[int] = field(default_factory=set)

    @property
    def all_rejected(self) -> bool:
        return (
            not self.uncertain
            and not self.pending
            and bool(self.outcomes)
            and all(type(status) is int and 400 <= status < 500 and status != 408 for status in self.outcomes)
        )


def _failure(error: BaseException) -> str:
    if isinstance(error, httpx2.TimeoutException):
        return "timeout"
    return "transport_error" if isinstance(error, httpx2.TransportError) else "incomplete"


class _ObservedStream(httpx2.SyncByteStream):
    def __init__(self, stream: httpx2.SyncByteStream, observation: HttpObservation, index: int) -> None:
        self.stream, self.observation, self.index = stream, observation, index
        self.exhausted = False

    def failed(self, error: BaseException) -> None:
        self.observation.outcomes[self.index] = _failure(error)
        self.observation.pending.discard(self.index)

    def __iter__(self):
        try:
            yield from self.stream
        except BaseException as error:
            self.failed(error)
            raise
        else:
            self.exhausted = True

    def decoded(self, iterator: Callable[[int | None], Iterator[bytes]], chunk_size: int | None = None) -> Iterator[bytes]:
        try:
            yield from iterator(chunk_size)
        except BaseException as error:
            self.failed(error)
            raise
        else:
            if self.exhausted:
                self.observation.pending.discard(self.index)

    def close(self) -> None:
        if not self.exhausted and self.index in self.observation.pending:
            self.observation.outcomes[self.index] = "incomplete"
            self.observation.pending.discard(self.index)
        try:
            self.stream.close()
        except BaseException as error:
            self.failed(error)
            raise


class _ObservationHooks(httpx2.Client):
    """Hooks see each send, including SDK retries and HTTP redirects."""

    def __init__(self, observation: HttpObservation, **kwargs: Any) -> None:
        self.observation = observation
        self._indices: ContextVar[list[int] | None] = ContextVar("http_observation_indices", default=None)
        self._index_lock = Lock()
        self._responses = ExitStack()
        self._configure_network(kwargs)
        super().__init__(event_hooks={"request": [self._started], "response": [self._responded]}, **kwargs)

    def _configure_network(self, kwargs: dict[str, Any]) -> None:
        pass

    def _started(self, request: httpx2.Request) -> None:
        with self._index_lock:
            index = len(self.observation.outcomes)
            self.observation.outcomes.append(None)
        request.extensions["vv_agent_observation_index"] = index
        indices = self._indices.get()
        if indices is not None:
            indices.append(index)

    def _responded(self, response: httpx2.Response) -> None:
        index = response.request.extensions["vv_agent_observation_index"]
        self.observation.outcomes[index] = response.status_code
        self.observation.pending.add(index)
        response.extensions["vv_agent_http_observation"] = self.observation
        self._responses.callback(response.close)
        # SDKs may read a streamed error body after send() has already returned.
        assert isinstance(response.stream, httpx2.SyncByteStream)
        stream = _ObservedStream(response.stream, self.observation, index)
        # In-memory transports may have fully read and decoded the body already.
        stream.exhausted = response.is_stream_consumed and hasattr(response, "_content")
        response.stream = stream
        iterator = response.iter_bytes

        def decoded(chunk_size: int | None = None) -> Iterator[bytes]:
            replay = response.stream
            if isinstance(replay, _PrefetchedStream) and replay.error is not None:
                stream.failed(replay.error)
                raise replay.error
            yield from stream.decoded(iterator, chunk_size)

        # Observe the response's decoder as well as the underlying raw stream.
        cast(Any, response).iter_bytes = decoded
        if response.is_closed and not stream.exhausted:
            self.observation.outcomes[index] = "incomplete"
            self.observation.pending.discard(index)
        elif stream.exhausted:
            self.observation.pending.discard(index)
            if 400 <= response.status_code < 500 and response.status_code != 408 and len(response.content) > _PREFETCH_BYTES:
                self.observation.outcomes[index] = "incomplete"
        if 400 <= response.status_code < 500 and response.status_code != 408 and not response.is_closed:
            self._prefetch(response, index)

    def _prefetch(self, response: httpx2.Response, index: int) -> None:
        read_timeout = response.request.extensions.get("timeout", {}).get("read")
        seconds = _PREFETCH_SECONDS if read_timeout is None else min(read_timeout, _PREFETCH_SECONDS)
        # HTTP/1.1 captures this timeout on first iteration; keep the copy for replay.
        response.request.extensions["timeout"] = {**response.request.extensions.get("timeout", {}), "read": seconds}
        replay = _PrefetchedStream(response, seconds)
        response.stream = replay
        if replay.error is not None and not isinstance(replay.error, Exception):
            raise replay.error
        if replay.content is not None:
            try:
                response.close()
            except Exception as error:
                # Even closing failures must reach the SDK through body consumption.
                replay.error = error
                response.is_closed = False
            else:
                response._content = replay.content
                replay.content = None
                replay.prefix.clear()
                response._num_bytes_downloaded = replay.raw_size
                response.is_stream_consumed = True
                self.observation.pending.discard(index)
                return
        self.observation.outcomes[index] = _failure(replay.error) if replay.error is not None else "incomplete"
        self.observation.pending.discard(index)

    def send(self, request: httpx2.Request, **kwargs: Any) -> httpx2.Response:
        indices: list[int] = []
        token = self._indices.set(indices)
        try:
            return super().send(request, **kwargs)
        except BaseException as error:
            # Also veto failures while reading a response body after the response hook.
            if indices:
                self.observation.outcomes[indices[-1]] = _failure(error)
            else:
                self.observation.uncertain = True
            raise
        finally:
            self._indices.reset(token)

    def close(self) -> None:
        try:
            self._responses.close()
        finally:
            super().close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        try:
            self._responses.close()
        finally:
            super().__exit__(exc_type, exc_value, traceback)


class ObservedHttpClient(_ObservationHooks, OpenAIDefaultHttpxClient):
    """OpenAI's exported defaults, with per-invocation observation."""


class ObservedAnthropicHttpClient(AnthropicDefaultHttpxClient, _ObservationHooks):
    """Anthropic's exported defaults include platform-specific TCP keepalive."""

    def __init__(self, observation: HttpObservation, **kwargs: Any) -> None:
        self._requested_mounts = kwargs.get("mounts", {})
        super().__init__(observation=observation, **kwargs)

    def _configure_network(self, kwargs: dict[str, Any]) -> None:
        transport = kwargs.get("transport")
        # The exported constructor is the source of platform-specific socket options.
        pool = getattr(transport, "_pool", None)
        self._socket_options = getattr(pool, "_socket_options", None)
        if kwargs.get("proxy") is not None or not kwargs.get("trust_env", True):
            # Vendor-generated environment mounts must not override host proxy policy.
            for mount in kwargs.get("mounts", {}).values():
                if mount is not None and mount not in self._requested_mounts.values():
                    mount.close()
            kwargs["mounts"] = self._requested_mounts

    def _init_proxy_transport(
        self,
        proxy: Any,
        verify: Any = True,
        cert: Any = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: httpx2.Limits = ANTHROPIC_CONNECTION_LIMITS,
    ) -> httpx2.BaseTransport:
        return httpx2.HTTPTransport(
            proxy=proxy,
            verify=verify,
            cert=cert,
            trust_env=trust_env,
            http1=http1,
            http2=http2,
            limits=limits,
            socket_options=self._socket_options,
        )
