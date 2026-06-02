from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vv_agent.app_server.outgoing import OutgoingRouter
from vv_agent.app_server.protocol import (
    AppServerError,
    InitializeResponse,
    JsonRpcError,
    JsonRpcMessage,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    ModelListResponse,
)
from vv_agent.app_server.transport import ChannelTransport


@dataclass(slots=True)
class ConnectionState:
    initialized: bool = False


class MessageProcessor:
    def __init__(self, *, router: OutgoingRouter) -> None:
        self._router = router
        self._connections: dict[str, ConnectionState] = {}

    def process_next(self, transport: ChannelTransport) -> None:
        self.process_message(transport.connection_id, transport.receive_inbound())

    def process_message(self, connection_id: str, payload: dict[str, Any]) -> None:
        message = JsonRpcMessage.from_dict(payload).message
        if isinstance(message, JsonRpcResponse):
            self._router.resolve_response(message)
            return
        if isinstance(message, JsonRpcError):
            self._router.resolve_error(message)
            return
        if isinstance(message, JsonRpcNotification):
            self._handle_notification(connection_id, message)
            return
        self._handle_request(connection_id, message)

    def _handle_notification(self, connection_id: str, notification: JsonRpcNotification) -> None:
        if notification.method == "initialized":
            self._state(connection_id).initialized = True

    def _handle_request(self, connection_id: str, request: JsonRpcRequest) -> None:
        state = self._state(connection_id)
        if request.method == "initialize":
            self._handle_initialize(connection_id, request, state)
            return
        if not state.initialized:
            self._router.send_error(connection_id, request.id, AppServerError.not_initialized())
            return
        if request.method == "model/list":
            self._router.send_response(connection_id, request.id, ModelListResponse().to_dict())
            return
        self._router.send_error(connection_id, request.id, AppServerError.method_not_found(request.method))

    def _handle_initialize(self, connection_id: str, request: JsonRpcRequest, state: ConnectionState) -> None:
        if state.initialized:
            self._router.send_error(connection_id, request.id, AppServerError.already_initialized())
            return
        state.initialized = True
        response = InitializeResponse(
            user_agent="vv-agent-app-server",
            protocol_version="v1",
            capabilities={"modelList": True},
        )
        self._router.send_response(connection_id, request.id, response.to_dict())

    def _state(self, connection_id: str) -> ConnectionState:
        state = self._connections.get(connection_id)
        if state is None:
            state = ConnectionState()
            self._connections[connection_id] = state
        return state
