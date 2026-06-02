from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vv_agent.app_server.host import AppServerHost, DefaultAppServerHost
from vv_agent.app_server.outgoing import OutgoingRouter
from vv_agent.app_server.protocol import (
    AppServerError,
    InitializeResponse,
    JsonRpcError,
    JsonRpcMessage,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    ModelListRequest,
)
from vv_agent.app_server.run_adapter import RunAdapter
from vv_agent.app_server.thread_state import ThreadStateManager
from vv_agent.app_server.thread_store import ThreadStore
from vv_agent.app_server.transport import ChannelTransport


@dataclass(slots=True)
class ConnectionState:
    initialized: bool = False


class MessageProcessor:
    def __init__(
        self,
        *,
        router: OutgoingRouter,
        host: AppServerHost | None = None,
        store: ThreadStore | None = None,
        state_manager: ThreadStateManager | None = None,
        run_adapter: RunAdapter | None = None,
    ) -> None:
        self._router = router
        self._host = host or DefaultAppServerHost()
        self._store = store or ThreadStore()
        self._state_manager = state_manager or ThreadStateManager()
        self._run_adapter = run_adapter or RunAdapter(
            host=self._host,
            store=self._store,
            state_manager=self._state_manager,
            router=self._router,
        )
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
            self._router.send_response(connection_id, request.id, self._host.list_models(ModelListRequest()).to_dict())
            return
        if request.method == "thread/start":
            self._handle_thread_start(connection_id, request)
            return
        if request.method == "turn/start":
            self._handle_turn_start(connection_id, request)
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

    def _handle_thread_start(self, connection_id: str, request: JsonRpcRequest) -> None:
        params = request.params if isinstance(request.params, dict) else {}
        agent_key = str(params.get("agentKey") or "default")
        cwd = params.get("cwd")
        metadata = params.get("metadata")
        thread = self._store.create_thread(
            agent_key=agent_key,
            cwd=str(cwd) if cwd is not None else None,
            metadata=metadata if isinstance(metadata, dict) else None,
        )
        self._state_manager.subscribe(thread.thread_id, connection_id)
        payload = {"threadId": thread.thread_id, "agentKey": thread.agent_key, "cwd": thread.cwd, "status": "idle"}
        self._router.send_notification(connection_id, "thread/started", payload)
        self._router.send_response(connection_id, request.id, payload)

    def _handle_turn_start(self, connection_id: str, request: JsonRpcRequest) -> None:
        params = request.params if isinstance(request.params, dict) else {}
        thread_id = str(params.get("threadId") or "")
        raw_input = params.get("input")
        input_items = [dict(item) for item in raw_input] if isinstance(raw_input, list) else []
        if not thread_id:
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("Missing threadId"))
            return
        self._run_adapter.start_turn(connection_id=connection_id, thread_id=thread_id, input=input_items, request_id=request.id)
