from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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
from vv_agent.app_server.request_serialization import (
    RequestAccess,
    RequestQueueOverloaded,
    RequestScope,
    RequestSerializationQueues,
)
from vv_agent.app_server.run_adapter import RunAdapter
from vv_agent.app_server.thread_state import ThreadStateManager
from vv_agent.app_server.thread_store import ThreadRecord, ThreadStore
from vv_agent.app_server.transport import ChannelTransport

CLIENT_METHODS: tuple[str, ...] = (
    "initialize",
    "model/list",
    "thread/start",
    "thread/resume",
    "thread/read",
    "thread/list",
    "thread/archive",
    "thread/unsubscribe",
    "turn/start",
    "turn/steer",
    "turn/followUp",
    "turn/interrupt",
)


@dataclass(slots=True)
class ConnectionState:
    initialized: bool = False
    client_name: str | None = None
    client_title: str | None = None
    client_version: str | None = None
    experimental_api: bool = False
    opt_out_notification_methods: set[str] = field(default_factory=set)


class MessageProcessor:
    def __init__(
        self,
        *,
        router: OutgoingRouter,
        host: AppServerHost | None = None,
        store: ThreadStore | None = None,
        state_manager: ThreadStateManager | None = None,
        run_adapter: RunAdapter | None = None,
        serialization_queues: RequestSerializationQueues | None = None,
    ) -> None:
        self._router = router
        self._host = host or DefaultAppServerHost()
        self._store = store or ThreadStore()
        self._state_manager = state_manager or ThreadStateManager()
        self._serialization_queues = serialization_queues or RequestSerializationQueues()
        self._run_adapter = run_adapter or RunAdapter(
            host=self._host,
            store=self._store,
            state_manager=self._state_manager,
            router=self._router,
        )
        self._connections: dict[str, ConnectionState] = {}
        self._router.set_notification_filter(self._should_send_notification)

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
            self._serialize_or_run(
                connection_id,
                request,
                lambda: self._router.send_response(
                    connection_id,
                    request.id,
                    self._host.list_models(ModelListRequest()).to_dict(),
                ),
            )
            return
        if request.method == "thread/start":
            self._serialize_or_run(connection_id, request, lambda: self._handle_thread_start(connection_id, request))
            return
        if request.method == "thread/read":
            self._serialize_or_run(connection_id, request, lambda: self._handle_thread_read(connection_id, request))
            return
        if request.method == "thread/resume":
            self._serialize_or_run(connection_id, request, lambda: self._handle_thread_resume(connection_id, request))
            return
        if request.method == "thread/list":
            self._serialize_or_run(connection_id, request, lambda: self._handle_thread_list(connection_id, request))
            return
        if request.method == "thread/archive":
            self._serialize_or_run(connection_id, request, lambda: self._handle_thread_archive(connection_id, request))
            return
        if request.method == "thread/unsubscribe":
            self._serialize_or_run(connection_id, request, lambda: self._handle_thread_unsubscribe(connection_id, request))
            return
        if request.method == "turn/start":
            self._serialize_or_run(connection_id, request, lambda: self._handle_turn_start(connection_id, request))
            return
        if request.method == "turn/steer":
            self._serialize_or_run(connection_id, request, lambda: self._handle_turn_steer(connection_id, request))
            return
        if request.method == "turn/followUp":
            self._serialize_or_run(connection_id, request, lambda: self._handle_turn_follow_up(connection_id, request))
            return
        if request.method == "turn/interrupt":
            self._serialize_or_run(connection_id, request, lambda: self._handle_turn_interrupt(connection_id, request))
            return
        self._router.send_error(connection_id, request.id, AppServerError.method_not_found(request.method))

    def _serialize_or_run(self, connection_id: str, request: JsonRpcRequest, handler: Callable[[], None]) -> None:
        try:
            future = self._serialization_queues.enqueue(
                key=self._request_scope(request),
                access=self._request_access(request.method),
                fn=handler,
            )
        except RequestQueueOverloaded:
            self._router.send_error(connection_id, request.id, AppServerError.server_overloaded())
            return
        try:
            future.result()
        except Exception as exc:
            self._router.send_error(connection_id, request.id, AppServerError.internal_error(str(exc)))

    def _request_scope(self, request: JsonRpcRequest) -> RequestScope:
        if request.method == "model/list":
            return RequestScope.global_read("model")
        if request.method == "thread/start":
            return RequestScope.global_scope("thread-create")
        if request.method == "thread/list":
            return RequestScope.global_read("thread-list")
        thread_id = self._thread_id_from_request(request)
        return RequestScope.thread(thread_id or "missing-thread")

    def _request_access(self, method: str) -> RequestAccess:
        if method in {"model/list", "thread/list", "thread/read"}:
            return "shared_read"
        return "exclusive"

    def _handle_initialize(self, connection_id: str, request: JsonRpcRequest, state: ConnectionState) -> None:
        if state.initialized:
            self._router.send_error(connection_id, request.id, AppServerError.already_initialized())
            return
        params = request.params if isinstance(request.params, dict) else {}
        raw_client_info = params.get("clientInfo")
        client_info = raw_client_info if isinstance(raw_client_info, dict) else {}
        raw_capabilities = params.get("capabilities")
        capabilities = raw_capabilities if isinstance(raw_capabilities, dict) else {}
        raw_opt_out = capabilities.get("optOutNotificationMethods", [])
        if not isinstance(raw_opt_out, list):
            self._router.send_error(
                connection_id,
                request.id,
                AppServerError.invalid_params("optOutNotificationMethods must be a list of strings"),
            )
            return
        opt_out_methods: list[str] = []
        for method in raw_opt_out:
            if not isinstance(method, str):
                self._router.send_error(
                    connection_id,
                    request.id,
                    AppServerError.invalid_params("optOutNotificationMethods must be a list of strings"),
                )
                return
            opt_out_methods.append(method)
        state.initialized = True
        state.client_name = str(client_info.get("name") or "")
        state.client_title = str(client_info["title"]) if client_info.get("title") is not None else None
        state.client_version = str(client_info["version"]) if client_info.get("version") is not None else None
        state.experimental_api = bool(capabilities.get("experimentalApi", False))
        state.opt_out_notification_methods = set(opt_out_methods)
        response = InitializeResponse(
            user_agent="vv-agent-app-server",
            protocol_version="v1",
            capabilities={"modelList": True, "threadLifecycle": True, "notificationOptOut": True},
        )
        self._router.send_response(connection_id, request.id, response.to_dict())

    def connection_state(self, connection_id: str) -> ConnectionState:
        return self._state(connection_id)

    def _should_send_notification(self, connection_id: str, method: str) -> bool:
        state = self._connections.get(connection_id)
        if state is None:
            return True
        return method not in state.opt_out_notification_methods

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

    def _handle_thread_read(self, connection_id: str, request: JsonRpcRequest) -> None:
        thread_id = self._thread_id_from_request(request)
        if not thread_id:
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("Missing threadId"))
            return
        try:
            snapshot = self._snapshot_to_dict(thread_id)
        except KeyError:
            self._router.send_error(connection_id, request.id, AppServerError.thread_not_found())
            return
        self._router.send_response(connection_id, request.id, snapshot)

    def _handle_thread_resume(self, connection_id: str, request: JsonRpcRequest) -> None:
        thread_id = self._thread_id_from_request(request)
        if not thread_id:
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("Missing threadId"))
            return
        try:
            snapshot = self._state_manager.subscribe_and_snapshot(
                thread_id,
                connection_id,
                lambda: self._snapshot_to_dict(thread_id),
            )
        except KeyError:
            self._state_manager.unsubscribe(thread_id, connection_id)
            self._router.send_error(connection_id, request.id, AppServerError.thread_not_found())
            return
        self._router.send_response(connection_id, request.id, snapshot)

    def _handle_thread_list(self, connection_id: str, request: JsonRpcRequest) -> None:
        params = request.params if isinstance(request.params, dict) else {}
        include_archived = bool(params.get("includeArchived", False))
        records = self._store.list_threads(include_archived=include_archived)
        self._router.send_response(
            connection_id,
            request.id,
            {"threads": [self._thread_record_to_dict(record) for record in records]},
        )

    def _handle_thread_archive(self, connection_id: str, request: JsonRpcRequest) -> None:
        thread_id = self._thread_id_from_request(request)
        if not thread_id:
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("Missing threadId"))
            return
        try:
            self._store.archive_thread(thread_id)
        except KeyError:
            self._router.send_error(connection_id, request.id, AppServerError.thread_not_found())
            return
        self._state_manager.set_status(thread_id, "archived")
        payload = {"threadId": thread_id, "archived": True}
        self._router.send_response(connection_id, request.id, payload)
        for subscriber in self._state_manager.subscribers(thread_id) | {connection_id}:
            self._router.send_notification(subscriber, "thread/archived", payload)
            self._router.send_notification(
                subscriber,
                "thread/status/changed",
                {"threadId": thread_id, "status": "archived"},
            )

    def _handle_thread_unsubscribe(self, connection_id: str, request: JsonRpcRequest) -> None:
        thread_id = self._thread_id_from_request(request)
        if not thread_id:
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("Missing threadId"))
            return
        try:
            self._store.read_thread(thread_id)
        except KeyError:
            self._router.send_error(connection_id, request.id, AppServerError.thread_not_found())
            return
        self._state_manager.unsubscribe(thread_id, connection_id)
        closed = self._state_manager.close_if_idle(thread_id)
        self._router.send_response(connection_id, request.id, {"threadId": thread_id, "subscribed": False, "closed": closed})
        if closed:
            self._router.send_notification(connection_id, "thread/closed", {"threadId": thread_id})
            self._router.send_notification(connection_id, "thread/status/changed", {"threadId": thread_id, "status": "closed"})

    def _handle_turn_start(self, connection_id: str, request: JsonRpcRequest) -> None:
        params = request.params if isinstance(request.params, dict) else {}
        thread_id = str(params.get("threadId") or "")
        raw_input = params.get("input")
        input_items = [dict(item) for item in raw_input] if isinstance(raw_input, list) else []
        if not thread_id:
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("Missing threadId"))
            return
        try:
            snapshot = self._store.read_thread(thread_id)
        except KeyError:
            self._router.send_error(connection_id, request.id, AppServerError.thread_not_found())
            return
        if snapshot.thread.archived_at is not None:
            self._router.send_error(connection_id, request.id, AppServerError.thread_archived())
            return
        self._run_adapter.start_turn(connection_id=connection_id, thread_id=thread_id, input=input_items, request_id=request.id)

    def _handle_turn_steer(self, connection_id: str, request: JsonRpcRequest) -> None:
        control = self._validated_active_turn(connection_id, request)
        if control is None:
            return
        thread_id, turn_id, _reason, input_items = control
        self._state_manager.queue_steering(thread_id, input_items)
        self._router.send_response(connection_id, request.id, {"threadId": thread_id, "turnId": turn_id, "queued": True})

    def _handle_turn_follow_up(self, connection_id: str, request: JsonRpcRequest) -> None:
        control = self._validated_active_turn(connection_id, request)
        if control is None:
            return
        thread_id, turn_id, _reason, input_items = control
        self._state_manager.queue_follow_up(thread_id, input_items)
        self._router.send_response(connection_id, request.id, {"threadId": thread_id, "turnId": turn_id, "queued": True})

    def _handle_turn_interrupt(self, connection_id: str, request: JsonRpcRequest) -> None:
        control = self._validated_active_turn(connection_id, request)
        if control is None:
            return
        thread_id, turn_id, reason, _input_items = control
        active = self._state_manager.active_turn(thread_id)
        cancelled = bool(active is not None and active.handle.cancel(reason or "Interrupted by App Server client."))
        self._router.send_response(connection_id, request.id, {"threadId": thread_id, "turnId": turn_id, "cancelled": cancelled})

    def _validated_active_turn(
        self,
        connection_id: str,
        request: JsonRpcRequest,
    ) -> tuple[str, str, str, list[dict[str, Any]]] | None:
        params = request.params if isinstance(request.params, dict) else {}
        thread_id = str(params.get("threadId") or "")
        expected_turn_id = str(params.get("expectedTurnId") or "")
        reason = str(params.get("reason") or "")
        raw_input = params.get("input")
        input_items = [dict(item) for item in raw_input] if isinstance(raw_input, list) else []
        if not thread_id:
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("Missing threadId"))
            return None
        active = self._state_manager.active_turn(thread_id)
        if active is None:
            self._router.send_error(connection_id, request.id, AppServerError.active_turn_not_found())
            return None
        if expected_turn_id and active.turn_id != expected_turn_id:
            self._router.send_error(connection_id, request.id, AppServerError.turn_id_mismatch())
            return None
        return thread_id, active.turn_id, reason, input_items

    def _thread_id_from_request(self, request: JsonRpcRequest) -> str:
        params = request.params if isinstance(request.params, dict) else {}
        return str(params.get("threadId") or "")

    def _snapshot_to_dict(self, thread_id: str) -> dict[str, Any]:
        snapshot = self._store.read_thread(thread_id)
        return {
            "thread": self._thread_record_to_dict(snapshot.thread),
            "turns": [
                {
                    "turnId": turn.turn_id,
                    "threadId": turn.thread_id,
                    "runId": turn.run_id,
                    "status": turn.status,
                    "startedAt": turn.started_at,
                    "completedAt": turn.completed_at,
                    "input": [dict(item) for item in turn.input],
                    "result": dict(turn.result),
                }
                for turn in snapshot.turns
            ],
            "items": [item.to_dict() for item in snapshot.items],
        }

    def _thread_record_to_dict(self, record: ThreadRecord) -> dict[str, Any]:
        archived = record.archived_at is not None
        return {
            "threadId": record.thread_id,
            "agentKey": record.agent_key,
            "cwd": record.cwd,
            "createdAt": record.created_at,
            "updatedAt": record.updated_at,
            "archivedAt": record.archived_at,
            "status": self._state_manager.status(record.thread_id, archived=archived),
            "metadata": dict(record.metadata),
        }
