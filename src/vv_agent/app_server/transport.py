from __future__ import annotations

import json
import queue
import sys
from collections.abc import Iterator
from typing import Any, Protocol, TextIO

OVERLOADED_MESSAGE = "Server overloaded; retry later."


class AppServerTransport(Protocol):
    connection_id: str

    def write_outbound(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError


class ChannelTransport:
    def __init__(self, *, connection_id: str = "channel", inbound_capacity: int = 0, outbound_capacity: int = 100) -> None:
        self.connection_id = connection_id
        self._inbound: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=inbound_capacity)
        self._outbound: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=outbound_capacity)
        self._closed = False

    def send_inbound(self, payload: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Transport is closed")
        self._inbound.put_nowait(payload)

    def receive_inbound(self, *, timeout: float | None = None) -> dict[str, Any]:
        return self._inbound.get(timeout=timeout)

    def write_outbound(self, payload: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Transport is closed")
        try:
            self._outbound.put_nowait(payload)
        except queue.Full as exc:
            raise RuntimeError(OVERLOADED_MESSAGE) from exc

    def receive_outbound(self, *, timeout: float | None = None) -> dict[str, Any]:
        return self._outbound.get(timeout=timeout)

    def close(self) -> None:
        self._closed = True


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
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object per line")
            yield payload

    def write_outbound(self, payload: dict[str, Any]) -> None:
        self._output.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._output.flush()
