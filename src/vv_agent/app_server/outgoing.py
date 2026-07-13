from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vv_agent.app_server.protocol import (
    AppServerError,
    AppServerErrorCode,
    JsonRpcError,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    RequestId,
)
from vv_agent.app_server.transport import OutboundAppServerTransport


@dataclass(slots=True)
class PendingServerRequest:
    connection_id: str
    request_id: RequestId
    method: str
    params: dict[str, Any] | None = None
    _event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _result: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _error: AppServerError | None = field(default=None, init=False, repr=False)

    def resolve_result(self, result: dict[str, Any] | None) -> None:
        self._result = result
        self._event.set()

    def resolve_error(self, error: AppServerError) -> None:
        self._error = error
        self._event.set()

    def result(self, timeout: float | None = None) -> dict[str, Any] | None:
        if not self._event.wait(timeout):
            raise TimeoutError(f"Timed out waiting for server request {self.request_id.to_wire()}")
        if self._error is not None:
            raise RuntimeError(self._error.message)
        return self._result


class OutgoingRouter:
    def __init__(self, *, should_send_notification: Callable[[str, str], bool] | None = None) -> None:
        self._transports: dict[str, OutboundAppServerTransport] = {}
        self._pending: dict[str | int, PendingServerRequest] = {}
        self._server_request_ids = itertools.count(1)
        self._lock = threading.Lock()
        self._should_send_notification = should_send_notification or (lambda _connection_id, _method: True)
        self._disconnect_handler: Callable[[str], None] | None = None

    def register_transport(self, transport: OutboundAppServerTransport) -> None:
        self._transports[transport.connection_id] = transport

    def unregister_transport(self, connection_id: str) -> None:
        self._transports.pop(connection_id, None)
        self._resolve_connection_error(connection_id, AppServerError(AppServerErrorCode.INTERNAL_ERROR, "client_disconnected"))
        if self._disconnect_handler is not None:
            self._disconnect_handler(connection_id)

    def set_notification_filter(self, callback: Callable[[str, str], bool]) -> None:
        self._should_send_notification = callback

    def set_disconnect_handler(self, callback: Callable[[str], None]) -> None:
        self._disconnect_handler = callback

    def is_registered(self, connection_id: str) -> bool:
        return connection_id in self._transports

    def send_response(
        self,
        connection_id: str,
        request_id: RequestId,
        result: dict[str, Any] | list[Any] | str | int | bool | None,
    ) -> None:
        self._write_or_disconnect(connection_id, JsonRpcResponse(id=request_id, result=result).to_dict())

    def send_error(self, connection_id: str, request_id: RequestId, error: AppServerError) -> None:
        self._write_or_disconnect(connection_id, JsonRpcError(id=request_id, error=error).to_dict())

    def send_notification(self, connection_id: str, method: str, params: dict[str, Any] | None = None) -> None:
        if not self._should_send_notification(connection_id, method):
            return
        self._write_or_disconnect(connection_id, JsonRpcNotification(method=method, params=params).to_dict())

    def broadcast_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        for connection_id, _transport in list(self._transports.items()):
            if self._should_send_notification(connection_id, method):
                self._write_or_disconnect(connection_id, JsonRpcNotification(method=method, params=params).to_dict())

    def send_server_request(
        self,
        connection_id: str,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        request_id: RequestId | None = None,
    ) -> PendingServerRequest:
        request_id = request_id or RequestId(f"srv_{next(self._server_request_ids)}")
        pending = PendingServerRequest(connection_id=connection_id, request_id=request_id, method=method, params=params)
        with self._lock:
            wire_id = request_id.to_wire()
            if wire_id is None:
                raise ValueError("Server request id cannot be null")
            if wire_id in self._pending:
                raise ValueError(f"Duplicate server request id: {wire_id}")
            self._pending[wire_id] = pending
        self._write_or_disconnect(connection_id, JsonRpcRequest(id=request_id, method=method, params=params).to_dict())
        return pending

    def cancel_server_request(self, request_id: RequestId, error: AppServerError | None = None) -> bool:
        with self._lock:
            pending = self._pending.pop(request_id.require_wire(), None)
        if pending is None:
            return False
        pending.resolve_error(error or AppServerError(AppServerErrorCode.INTERNAL_ERROR, "cancelled"))
        return True

    def cancel_matching_server_requests(
        self,
        *,
        method: str | None = None,
        thread_id: str | None = None,
        turn_id: str | None = None,
        error: AppServerError | None = None,
    ) -> list[PendingServerRequest]:
        with self._lock:
            request_ids = [
                request_id
                for request_id, pending in self._pending.items()
                if (method is None or pending.method == method)
                and (thread_id is None or (pending.params or {}).get("threadId") == thread_id)
                and (turn_id is None or (pending.params or {}).get("turnId") == turn_id)
            ]
            pending_requests = [self._pending.pop(request_id) for request_id in request_ids]
        resolution = error or AppServerError(AppServerErrorCode.INTERNAL_ERROR, "cancelled")
        for pending in pending_requests:
            pending.resolve_error(resolution)
        return pending_requests

    def pending_server_request_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def resolve_response(self, connection_id: str, response: JsonRpcResponse) -> bool:
        with self._lock:
            wire_id = response.id.require_wire()
            candidate = self._pending.get(wire_id)
            pending = (
                self._pending.pop(wire_id, None) if candidate is not None and candidate.connection_id == connection_id else None
            )
        if pending is None:
            return False
        result = response.result if isinstance(response.result, dict) else {"value": response.result}
        pending.resolve_result(result)
        return True

    def resolve_server_request(
        self,
        connection_id: str,
        request_id: RequestId,
        result: dict[str, Any],
        *,
        method: str | None = None,
        thread_id: str | None = None,
        turn_id: str | None = None,
    ) -> bool:
        wire_id = request_id.require_wire()
        with self._lock:
            pending = self._pending.get(wire_id)
            pending_thread_id = pending.params.get("threadId") if pending is not None and pending.params is not None else None
            pending_turn_id = pending.params.get("turnId") if pending is not None and pending.params is not None else None
            if (
                pending is None
                or pending.connection_id != connection_id
                or (method is not None and pending.method != method)
                or (thread_id is not None and pending_thread_id != thread_id)
                or (turn_id is not None and pending_turn_id != turn_id)
            ):
                return False
            self._pending.pop(wire_id, None)
        pending.resolve_result(result)
        return True

    def resolve_error(self, connection_id: str, error: JsonRpcError) -> bool:
        with self._lock:
            wire_id = error.id.require_wire()
            candidate = self._pending.get(wire_id)
            pending = (
                self._pending.pop(wire_id, None) if candidate is not None and candidate.connection_id == connection_id else None
            )
        if pending is None:
            return False
        pending.resolve_error(error.error)
        return True

    def _transport(self, connection_id: str) -> OutboundAppServerTransport:
        try:
            return self._transports[connection_id]
        except KeyError as exc:
            raise KeyError(f"Unknown App Server connection: {connection_id}") from exc

    def _transport_or_none(self, connection_id: str) -> OutboundAppServerTransport | None:
        return self._transports.get(connection_id)

    def _write_or_disconnect(self, connection_id: str, payload: dict[str, Any]) -> None:
        transport = self._transport_or_none(connection_id)
        if transport is None:
            self.unregister_transport(connection_id)
            return
        try:
            transport.write_outbound(payload)
        except Exception:
            self.unregister_transport(connection_id)

    def _resolve_connection_error(self, connection_id: str, error: AppServerError) -> None:
        with self._lock:
            request_ids = [request_id for request_id, pending in self._pending.items() if pending.connection_id == connection_id]
            pending_requests = [self._pending.pop(request_id) for request_id in request_ids]
        for pending in pending_requests:
            pending.resolve_error(error)
