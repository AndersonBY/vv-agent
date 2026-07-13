from __future__ import annotations

import json
import queue
import sys
from collections.abc import Iterator
from contextlib import suppress
from typing import Any, Protocol, TextIO, cast

from vv_agent.app_server.protocol import AppServerError, JsonRpcError, RequestId

OVERLOADED_MESSAGE = "Server overloaded; retry later."


class AppServerOverloadedError(RuntimeError):
    pass


class OutboundAppServerTransport(Protocol):
    connection_id: str

    def write_outbound(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError


class AppServerTransport(OutboundAppServerTransport, Protocol):
    def read_messages(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError


_CHANNEL_CLOSED = object()


class ChannelTransport:
    def __init__(self, *, connection_id: str = "channel", inbound_capacity: int = 0, outbound_capacity: int = 100) -> None:
        self.connection_id = connection_id
        self._inbound: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=inbound_capacity)
        self._outbound: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=outbound_capacity)
        self._closed = False

    def send_inbound(self, payload: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Transport is closed")
        self._inbound.put_nowait(payload)

    def receive_inbound(self, *, timeout: float | None = None) -> dict[str, Any]:
        payload = self._inbound.get(timeout=timeout)
        if payload is _CHANNEL_CLOSED:
            raise EOFError("Transport is closed")
        assert isinstance(payload, dict)
        return cast(dict[str, Any], payload)

    def read_messages(self) -> Iterator[dict[str, Any]]:
        while True:
            if self._closed and self._inbound.empty():
                return
            payload = self._inbound.get()
            if payload is _CHANNEL_CLOSED:
                return
            assert isinstance(payload, dict)
            yield cast(dict[str, Any], payload)

    def write_outbound(self, payload: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Transport is closed")
        try:
            self._outbound.put_nowait(payload)
        except queue.Full as exc:
            raise AppServerOverloadedError(OVERLOADED_MESSAGE) from exc

    def receive_outbound(self, *, timeout: float | None = None) -> dict[str, Any]:
        return self._outbound.get(timeout=timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(queue.Full):
            self._inbound.put_nowait(_CHANNEL_CLOSED)


class StdioJsonlTransport:
    def __init__(
        self,
        *,
        connection_id: str = "stdio",
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        self.connection_id = connection_id
        self._input = input_stream or sys.stdin
        self._output = output_stream or sys.stdout

    def read_messages(self) -> Iterator[dict[str, Any]]:
        for line in self._input:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                self.write_outbound(JsonRpcError(id=RequestId(None), error=AppServerError.parse_error()).to_dict())
                continue
            if not isinstance(payload, dict):
                self.write_outbound(JsonRpcError(id=RequestId(None), error=AppServerError.invalid_request()).to_dict())
                continue
            yield payload

    def write_outbound(self, payload: dict[str, Any]) -> None:
        self._output.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._output.flush()
