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
from vv_agent.app_server.transport import AppServerOverloadedError, AppServerTransport


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
        self._transports: dict[str, AppServerTransport] = {}
        self._pending: dict[str | int, PendingServerRequest] = {}
        self._server_request_ids = itertools.count(1)
        self._lock = threading.Lock()
        self._should_send_notification = should_send_notification or (lambda _connection_id, _method: True)

    def register_transport(self, transport: AppServerTransport) -> None:
        self._transports[transport.connection_id] = transport

    def unregister_transport(self, connection_id: str) -> None:
        self._transports.pop(connection_id, None)
        self._resolve_connection_error(connection_id, AppServerError(AppServerErrorCode.INTERNAL_ERROR, "client_disconnected"))

    def set_notification_filter(self, callback: Callable[[str, str], bool]) -> None:
        self._should_send_notification = callback

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

    def send_server_request(self, connection_id: str, method: str, params: dict[str, Any] | None = None) -> PendingServerRequest:
        request_id = RequestId(f"srv_{next(self._server_request_ids)}")
        pending = PendingServerRequest(connection_id=connection_id, request_id=request_id, method=method, params=params)
        with self._lock:
            self._pending[request_id.to_wire()] = pending
        self._write_or_disconnect(connection_id, JsonRpcRequest(id=request_id, method=method, params=params).to_dict())
        return pending

    def cancel_server_request(self, request_id: RequestId, error: AppServerError | None = None) -> bool:
        with self._lock:
            pending = self._pending.pop(request_id.to_wire(), None)
        if pending is None:
            return False
        pending.resolve_error(error or AppServerError(AppServerErrorCode.INTERNAL_ERROR, "cancelled"))
        return True

    def resolve_response(self, response: JsonRpcResponse) -> bool:
        with self._lock:
            pending = self._pending.pop(response.id.to_wire(), None)
        if pending is None:
            return False
        result = response.result if isinstance(response.result, dict) else {"value": response.result}
        pending.resolve_result(result)
        return True

    def resolve_error(self, error: JsonRpcError) -> bool:
        with self._lock:
            pending = self._pending.pop(error.id.to_wire(), None)
        if pending is None:
            return False
        pending.resolve_error(error.error)
        return True

    def _transport(self, connection_id: str) -> AppServerTransport:
        try:
            return self._transports[connection_id]
        except KeyError as exc:
            raise KeyError(f"Unknown App Server connection: {connection_id}") from exc

    def _transport_or_none(self, connection_id: str) -> AppServerTransport | None:
        return self._transports.get(connection_id)

    def _write_or_disconnect(self, connection_id: str, payload: dict[str, Any]) -> None:
        transport = self._transport_or_none(connection_id)
        if transport is None:
            return
        try:
            transport.write_outbound(payload)
        except AppServerOverloadedError:
            self.unregister_transport(connection_id)

    def _resolve_connection_error(self, connection_id: str, error: AppServerError) -> None:
        with self._lock:
            request_ids = [
                request_id
                for request_id, pending in self._pending.items()
                if pending.connection_id == connection_id
            ]
            pending_requests = [self._pending.pop(request_id) for request_id in request_ids]
        for pending in pending_requests:
            pending.resolve_error(error)
