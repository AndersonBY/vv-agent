from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vv_agent.app_server.host import AppServerHost, DefaultAppServerHost
from vv_agent.app_server.outgoing import OutgoingRouter
from vv_agent.app_server.protocol import (
    ApprovalDecision,
    ApprovalResolveParams,
    AppServerError,
    AppServerErrorCode,
    InitializeResponse,
    JsonRpcError,
    JsonRpcMessage,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    ModelListRequest,
    RequestId,
)
from vv_agent.app_server.request_serialization import (
    RequestAccess,
    RequestQueueOverloaded,
    RequestScope,
    RequestSerializationQueues,
)
from vv_agent.app_server.run_adapter import RunAdapter, TurnResumeError
from vv_agent.app_server.thread_state import ThreadStateManager
from vv_agent.app_server.thread_store import ThreadRecord, ThreadStore
from vv_agent.app_server.transport import ChannelTransport
from vv_agent.checkpoint import CheckpointError

CLIENT_METHODS: tuple[str, ...] = (
    "initialize",
    "initialized",
    "model/list",
    "thread/start",
    "thread/resume",
    "thread/read",
    "thread/list",
    "thread/archive",
    "thread/unsubscribe",
    "turn/start",
    "turn/resume",
    "turn/steer",
    "turn/followUp",
    "turn/interrupt",
    "approval/resolve",
    "schema/export",
)

SERVER_NOTIFICATION_METHODS: tuple[str, ...] = (
    "thread/started",
    "thread/status/changed",
    "thread/archived",
    "thread/closed",
    "turn/started",
    "item/started",
    "item/agentMessage/delta",
    "item/toolCall/delta",
    "item/completed",
    "approval/requested",
    "approval/resolved",
    "error/warning",
    "turn/completed",
)

SERVER_REQUEST_METHODS: tuple[str, ...] = ("approval/request",)

SERVER_CAPABILITIES: dict[str, object] = {
    "modelList": True,
    "threadLifecycle": True,
    "notificationOptOut": True,
    "schemaExport": True,
    "approvalResolve": True,
}


@dataclass(slots=True)
class ConnectionState:
    initialized: bool = False
    ready_for_notifications: bool = False
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
        self._state_manager.set_active_turn_persister(self._store.set_active_turn)
        self._serialization_queues = serialization_queues or RequestSerializationQueues()
        self._run_adapter = run_adapter or RunAdapter(
            host=self._host,
            store=self._store,
            state_manager=self._state_manager,
            router=self._router,
        )
        self._connections: dict[str, ConnectionState] = {}
        self._router.set_notification_filter(self._should_send_notification)
        self._router.set_disconnect_handler(self._handle_disconnect)

    def process_next(self, transport: ChannelTransport) -> None:
        self.process_message(transport.connection_id, transport.receive_inbound())

    def process_message(self, connection_id: str, payload: dict[str, Any]) -> None:
        try:
            message = JsonRpcMessage.from_dict(payload).message
        except (KeyError, TypeError, ValueError):
            raw_id = payload.get("id")
            request_id = RequestId(raw_id if isinstance(raw_id, (str, int)) and not isinstance(raw_id, bool) else None)
            self._router.send_error(connection_id, request_id, AppServerError.invalid_request())
            return
        if isinstance(message, JsonRpcResponse):
            self._router.resolve_response(connection_id, message)
            return
        if isinstance(message, JsonRpcError):
            self._router.resolve_error(connection_id, message)
            return
        if isinstance(message, JsonRpcNotification):
            self._handle_notification(connection_id, message)
            return
        self._handle_request(connection_id, message)

    def _handle_notification(self, connection_id: str, notification: JsonRpcNotification) -> None:
        if notification.method == "initialized":
            state = self._state(connection_id)
            if state.initialized:
                state.ready_for_notifications = True

    def _handle_request(self, connection_id: str, request: JsonRpcRequest) -> None:
        state = self._state(connection_id)
        if request.method == "initialize":
            self._handle_initialize(connection_id, request, state)
            return
        if not state.initialized:
            self._router.send_error(connection_id, request.id, AppServerError.not_initialized())
            return
        if request.method == "model/list":
            self._serialize_or_run(connection_id, request, lambda: self._handle_model_list(connection_id, request))
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
        if request.method == "turn/resume":
            self._serialize_or_run(connection_id, request, lambda: self._handle_turn_resume(connection_id, request))
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
        if request.method == "approval/resolve":
            self._serialize_or_run(connection_id, request, lambda: self._handle_approval_resolve(connection_id, request))
            return
        if request.method == "schema/export":
            self._serialize_or_run(connection_id, request, lambda: self._handle_schema_export(connection_id, request))
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
        if request.method == "schema/export":
            return RequestScope.global_read("schema")
        if request.method == "thread/start":
            return RequestScope.global_scope("thread-create")
        if request.method == "thread/list":
            return RequestScope.global_read("thread-list")
        thread_id = self._thread_id_from_request(request)
        return RequestScope.thread(thread_id or "missing-thread")

    def _request_access(self, method: str) -> RequestAccess:
        if method in {"model/list", "schema/export", "thread/list", "thread/read"}:
            return "shared_read"
        return "exclusive"

    def _handle_initialize(self, connection_id: str, request: JsonRpcRequest, state: ConnectionState) -> None:
        if state.initialized:
            self._router.send_error(connection_id, request.id, AppServerError.already_initialized())
            return
        if not isinstance(request.params, dict):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("params must be an object"))
            return
        params = request.params
        raw_client_info = params.get("clientInfo")
        if not isinstance(raw_client_info, dict):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("clientInfo is required"))
            return
        client_info = raw_client_info
        client_name = client_info.get("name")
        if not isinstance(client_name, str) or not client_name.strip():
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("clientInfo.name is required"))
            return
        for field_name in ("title", "version"):
            value = client_info.get(field_name)
            if value is not None and not isinstance(value, str):
                self._router.send_error(
                    connection_id,
                    request.id,
                    AppServerError.invalid_params(f"clientInfo.{field_name} must be a string"),
                )
                return
        raw_capabilities = params.get("capabilities")
        if raw_capabilities is not None and not isinstance(raw_capabilities, dict):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("capabilities must be an object"))
            return
        capabilities = raw_capabilities if isinstance(raw_capabilities, dict) else {}
        experimental_api = capabilities.get("experimentalApi", False)
        if not isinstance(experimental_api, bool):
            self._router.send_error(
                connection_id,
                request.id,
                AppServerError.invalid_params("experimentalApi must be a boolean"),
            )
            return
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
        state.client_name = client_name
        state.client_title = client_info.get("title")
        state.client_version = client_info.get("version")
        state.experimental_api = experimental_api
        state.opt_out_notification_methods = set(opt_out_methods)
        state.ready_for_notifications = True
        response = InitializeResponse(
            user_agent="vv-agent-app-server",
            protocol_version="v1",
            capabilities=SERVER_CAPABILITIES,
        )
        self._router.send_response(connection_id, request.id, response.to_dict())

    def connection_state(self, connection_id: str) -> ConnectionState:
        return self._state(connection_id)

    def disconnect_connection(self, connection_id: str) -> None:
        self._router.unregister_transport(connection_id)

    def _should_send_notification(self, connection_id: str, method: str) -> bool:
        state = self._connections.get(connection_id)
        if state is None or not state.initialized:
            return False
        return method not in state.opt_out_notification_methods

    def _state(self, connection_id: str) -> ConnectionState:
        state = self._connections.get(connection_id)
        if state is None:
            state = ConnectionState()
            self._connections[connection_id] = state
        return state

    def _handle_disconnect(self, connection_id: str) -> None:
        self._connections.pop(connection_id, None)
        self._state_manager.unsubscribe_connection(connection_id)

    def _handle_model_list(self, connection_id: str, request: JsonRpcRequest) -> None:
        params = self._params_object(connection_id, request, allow_missing=True)
        if params is None:
            return
        agent_key = self._optional_string_param(connection_id, request, params, "agentKey")
        if agent_key is _INVALID_PARAM:
            return
        assert agent_key is None or isinstance(agent_key, str)
        provider = self._optional_string_param(connection_id, request, params, "provider")
        if provider is _INVALID_PARAM:
            return
        assert provider is None or isinstance(provider, str)
        model_request = ModelListRequest(agent_key=agent_key, provider=provider)
        self._router.send_response(connection_id, request.id, self._host.list_models(model_request).to_dict())

    def _handle_thread_start(self, connection_id: str, request: JsonRpcRequest) -> None:
        params = self._params_object(connection_id, request, allow_missing=True)
        if params is None:
            return
        agent_key = params.get("agentKey", "default")
        if not isinstance(agent_key, str):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("agentKey must be a string"))
            return
        cwd = params.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("cwd must be a string"))
            return
        metadata = params.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("metadata must be an object"))
            return
        thread = self._store.create_thread(
            agent_key=agent_key,
            cwd=cwd,
            metadata=metadata if isinstance(metadata, dict) else None,
        )
        self._state_manager.subscribe(thread.thread_id, connection_id)
        payload = {"threadId": thread.thread_id, "agentKey": thread.agent_key, "cwd": thread.cwd, "status": "idle"}
        self._router.send_response(connection_id, request.id, payload)
        self._router.send_notification(connection_id, "thread/started", payload)

    def _handle_thread_read(self, connection_id: str, request: JsonRpcRequest) -> None:
        params = self._params_object(connection_id, request)
        if params is None:
            return
        thread_id = self._required_string_param(connection_id, request, params, "threadId")
        if not thread_id:
            return
        after_item_id = self._optional_string_param(connection_id, request, params, "afterItemId")
        if after_item_id is _INVALID_PARAM:
            return
        assert after_item_id is None or isinstance(after_item_id, str)
        try:
            snapshot = self._snapshot_to_dict(thread_id, after_item_id=after_item_id)
        except KeyError:
            self._router.send_error(connection_id, request.id, AppServerError.thread_not_found())
            return
        self._router.send_response(connection_id, request.id, snapshot)

    def _handle_thread_resume(self, connection_id: str, request: JsonRpcRequest) -> None:
        params = self._params_object(connection_id, request)
        if params is None:
            return
        thread_id = self._required_string_param(connection_id, request, params, "threadId")
        if not thread_id:
            return
        subscribe = params.get("subscribe", True)
        if not isinstance(subscribe, bool):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("subscribe must be a boolean"))
            return
        try:
            if subscribe:
                snapshot = self._state_manager.subscribe_and_snapshot(
                    thread_id,
                    connection_id,
                    lambda: self._snapshot_to_dict(thread_id, reopen_closed=True),
                )
            else:
                snapshot = self._snapshot_to_dict(thread_id, reopen_closed=True)
                self._state_manager.reopen(thread_id)
                thread = snapshot.get("thread")
                if isinstance(thread, dict):
                    thread["status"] = self._state_manager.status(
                        thread_id,
                        archived=thread.get("archivedAt") is not None,
                    )
        except KeyError:
            self._router.send_error(connection_id, request.id, AppServerError.thread_not_found())
            return
        self._router.send_response(connection_id, request.id, snapshot)

    def _handle_thread_list(self, connection_id: str, request: JsonRpcRequest) -> None:
        params = self._params_object(connection_id, request, allow_missing=True)
        if params is None:
            return
        include_archived = params.get("includeArchived", False)
        archived = params.get("archived")
        if not isinstance(include_archived, bool):
            self._router.send_error(
                connection_id,
                request.id,
                AppServerError.invalid_params("includeArchived must be a boolean"),
            )
            return
        if archived is not None and not isinstance(archived, bool):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("archived must be a boolean"))
            return
        offset = self._optional_non_negative_int(connection_id, request, params, "offset")
        if offset is _INVALID_PARAM:
            return
        assert offset is None or isinstance(offset, int)
        limit = self._optional_non_negative_int(connection_id, request, params, "limit")
        if limit is _INVALID_PARAM:
            return
        assert limit is None or isinstance(limit, int)
        include_archived = include_archived or archived is True
        records = self._store.list_threads(include_archived=include_archived)
        if archived is not None:
            records = [record for record in records if (record.archived_at is not None) is archived]
        start = offset or 0
        end = None if limit is None else start + limit
        records = records[start:end]
        self._router.send_response(
            connection_id,
            request.id,
            {"threads": [self._thread_record_to_dict(record) for record in records]},
        )

    def _handle_thread_archive(self, connection_id: str, request: JsonRpcRequest) -> None:
        params = self._params_object(connection_id, request)
        if params is None:
            return
        thread_id = self._required_string_param(connection_id, request, params, "threadId")
        if not thread_id:
            return
        try:
            self._store.archive_thread(thread_id)
        except KeyError:
            self._router.send_error(connection_id, request.id, AppServerError.thread_not_found())
            return
        self._state_manager.set_status(thread_id, "archived")
        payload = {"threadId": thread_id, "archived": True}
        self._router.send_response(connection_id, request.id, payload)
        self._router.send_notification(connection_id, "thread/archived", payload)
        self._router.send_notification(
            connection_id,
            "thread/status/changed",
            {"threadId": thread_id, "status": "archived"},
        )

    def _handle_thread_unsubscribe(self, connection_id: str, request: JsonRpcRequest) -> None:
        params = self._params_object(connection_id, request)
        if params is None:
            return
        thread_id = self._required_string_param(connection_id, request, params, "threadId")
        if not thread_id:
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
        params = self._params_object(connection_id, request)
        if params is None:
            return
        thread_id = self._required_string_param(connection_id, request, params, "threadId")
        if not thread_id:
            return
        raw_input = params.get("input")
        if raw_input is None:
            raw_input = []
        if not isinstance(raw_input, list) or any(not isinstance(item, dict) for item in raw_input):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("input must be a list of objects"))
            return
        input_items = [dict(item) for item in raw_input]
        metadata = params.get("metadata", {})
        if not isinstance(metadata, dict):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("metadata must be an object"))
            return
        try:
            snapshot = self._store.read_thread(thread_id)
        except KeyError:
            self._router.send_error(connection_id, request.id, AppServerError.thread_not_found())
            return
        if snapshot.thread.archived_at is not None:
            self._router.send_error(connection_id, request.id, AppServerError.thread_archived())
            return
        if self._state_manager.active_turn(thread_id) is not None or snapshot.thread.active_turn_id is not None:
            self._router.send_error(
                connection_id,
                request.id,
                AppServerError.invalid_params("Thread already has an active turn"),
            )
            return
        self._run_adapter.start_turn(
            connection_id=connection_id,
            thread_id=thread_id,
            input=input_items,
            metadata=dict(metadata),
            request_id=request.id,
        )

    def _handle_turn_resume(self, connection_id: str, request: JsonRpcRequest) -> None:
        params = self._params_object(connection_id, request)
        if params is None:
            return
        expected_fields = {"threadId", "turnId", "checkpointKey"}
        if set(params) != expected_fields:
            self._router.send_error(
                connection_id,
                request.id,
                AppServerError.invalid_params(
                    "turn/resume requires exactly threadId, turnId, and checkpointKey"
                ),
            )
            return
        thread_id = self._required_string_param(connection_id, request, params, "threadId")
        if not thread_id:
            return
        turn_id = self._required_string_param(connection_id, request, params, "turnId")
        if not turn_id:
            return
        checkpoint_key = self._required_string_param(connection_id, request, params, "checkpointKey")
        if not checkpoint_key:
            return
        try:
            snapshot = self._store.read_thread(thread_id)
        except KeyError:
            self._router.send_error(connection_id, request.id, AppServerError.thread_not_found())
            return
        if snapshot.thread.archived_at is not None:
            self._router.send_error(connection_id, request.id, AppServerError.thread_archived())
            return
        try:
            self._run_adapter.resume_turn(
                connection_id=connection_id,
                thread_id=thread_id,
                turn_id=turn_id,
                checkpoint_key=checkpoint_key,
                request_id=request.id,
            )
        except TurnResumeError as exc:
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params(str(exc)))
        except CheckpointError as exc:
            self._router.send_error(
                connection_id,
                request.id,
                AppServerError.invalid_params(
                    "Checkpoint resume rejected",
                    data={"checkpointErrorCode": exc.code},
                ),
            )

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
        pending_approvals = self._router.cancel_matching_server_requests(
            method="approval/request",
            thread_id=thread_id,
            turn_id=turn_id,
            error=AppServerError(AppServerErrorCode.INTERNAL_ERROR, reason or "turn interrupted"),
        )
        for pending in pending_approvals:
            params = ApprovalResolveParams(
                thread_id=thread_id,
                turn_id=turn_id,
                request_id=str(pending.request_id.require_wire()),
                decision=ApprovalDecision.TIMEOUT,
            ).to_dict()
            for subscriber in self._state_manager.subscribers(thread_id):
                self._router.send_notification(subscriber, "approval/resolved", params)
        self._router.send_response(connection_id, request.id, {"threadId": thread_id, "turnId": turn_id, "cancelled": cancelled})

    def _handle_approval_resolve(self, connection_id: str, request: JsonRpcRequest) -> None:
        params = self._params_object(connection_id, request)
        if params is None:
            return
        thread_id = self._required_string_param(connection_id, request, params, "threadId")
        if not thread_id:
            return
        turn_id = self._required_string_param(connection_id, request, params, "turnId")
        if not turn_id:
            return
        approval_request_id = self._required_string_param(connection_id, request, params, "requestId")
        if not approval_request_id:
            return
        raw_decision = params.get("decision")
        if not isinstance(raw_decision, str):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("decision must be a string"))
            return
        try:
            decision = ApprovalDecision.from_wire(raw_decision)
        except ValueError:
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("Invalid approval decision"))
            return
        reason = params.get("reason", "")
        if not isinstance(reason, str):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("reason must be a string"))
            return
        decision_metadata = params.get("metadata", {})
        if not isinstance(decision_metadata, dict):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("metadata must be an object"))
            return
        active = self._state_manager.active_turn(thread_id)
        if active is None:
            self._router.send_error(connection_id, request.id, AppServerError.active_turn_not_found())
            return
        if active.turn_id != turn_id:
            self._router.send_error(connection_id, request.id, AppServerError.turn_id_mismatch())
            return
        resolved = self._router.resolve_server_request(
            connection_id,
            RequestId(approval_request_id),
            {"decision": decision.value, "reason": reason, "metadata": dict(decision_metadata)},
            method="approval/request",
            thread_id=thread_id,
            turn_id=turn_id,
        )
        if not resolved:
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("Unknown approval requestId"))
            return
        self._router.send_response(connection_id, request.id, {})

    def _handle_schema_export(self, connection_id: str, request: JsonRpcRequest) -> None:
        from vv_agent.app_server.schema import export_schema_bundles

        params = self._params_object(connection_id, request, allow_missing=True)
        if params is None:
            return
        if params:
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("params must be empty"))
            return
        self._router.send_response(connection_id, request.id, export_schema_bundles())

    def _validated_active_turn(
        self,
        connection_id: str,
        request: JsonRpcRequest,
    ) -> tuple[str, str, str, list[dict[str, Any]]] | None:
        params = self._params_object(connection_id, request)
        if params is None:
            return None
        thread_id = self._required_string_param(connection_id, request, params, "threadId")
        if not thread_id:
            return None
        expected_turn_id = self._optional_string_param(connection_id, request, params, "expectedTurnId")
        if expected_turn_id is _INVALID_PARAM:
            return None
        assert expected_turn_id is None or isinstance(expected_turn_id, str)
        reason = self._optional_string_param(connection_id, request, params, "reason")
        if reason is _INVALID_PARAM:
            return None
        assert reason is None or isinstance(reason, str)
        raw_input = params.get("input", [])
        if not isinstance(raw_input, list) or any(not isinstance(item, dict) for item in raw_input):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("input must be a list of objects"))
            return None
        input_items: list[dict[str, Any]] = [dict(item) for item in raw_input]
        active = self._state_manager.active_turn(thread_id)
        if active is None:
            self._router.send_error(connection_id, request.id, AppServerError.active_turn_not_found())
            return None
        if expected_turn_id and active.turn_id != expected_turn_id:
            self._router.send_error(connection_id, request.id, AppServerError.turn_id_mismatch())
            return None
        return thread_id, active.turn_id, reason or "", input_items

    def _thread_id_from_request(self, request: JsonRpcRequest) -> str:
        params = request.params if isinstance(request.params, dict) else {}
        return str(params.get("threadId") or "")

    def _snapshot_to_dict(
        self,
        thread_id: str,
        *,
        after_item_id: str | None = None,
        reopen_closed: bool = False,
    ) -> dict[str, Any]:
        snapshot = self._store.read_thread(thread_id)
        reopened_status: str | None = None
        active_turn_id: str | None = None
        if reopen_closed and snapshot.thread.status == "closed":
            active = self._state_manager.active_turn(thread_id)
            active_turn_id = active.turn_id if active is not None else None
            reopened_status = "running" if active is not None else "idle"
        items = snapshot.items
        if after_item_id is not None:
            marker = next((index for index, item in enumerate(items) if item.item_id == after_item_id), None)
            if marker is not None:
                items = items[marker + 1 :]
        result: dict[str, Any] = {
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
            "items": [item.to_dict() for item in items],
        }
        if reopened_status is not None:
            self._store.set_active_turn(thread_id, active_turn_id, reopened_status)
            result["thread"]["status"] = reopened_status
        return result

    def _params_object(
        self,
        connection_id: str,
        request: JsonRpcRequest,
        *,
        allow_missing: bool = False,
    ) -> dict[str, Any] | None:
        if request.params is None and allow_missing:
            return {}
        if not isinstance(request.params, dict):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params("params must be an object"))
            return None
        return request.params

    def _required_string_param(
        self,
        connection_id: str,
        request: JsonRpcRequest,
        params: dict[str, Any],
        name: str,
    ) -> str | None:
        value = params.get(name)
        if not isinstance(value, str) or not value:
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params(f"Missing {name}"))
            return None
        return value

    def _optional_string_param(
        self,
        connection_id: str,
        request: JsonRpcRequest,
        params: dict[str, Any],
        name: str,
    ) -> str | None | object:
        value = params.get(name)
        if value is not None and not isinstance(value, str):
            self._router.send_error(connection_id, request.id, AppServerError.invalid_params(f"{name} must be a string"))
            return _INVALID_PARAM
        return value

    def _optional_non_negative_int(
        self,
        connection_id: str,
        request: JsonRpcRequest,
        params: dict[str, Any],
        name: str,
    ) -> int | None | object:
        value = params.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            self._router.send_error(
                connection_id,
                request.id,
                AppServerError.invalid_params(f"{name} must be a non-negative integer"),
            )
            return _INVALID_PARAM
        return value

    def _thread_record_to_dict(self, record: ThreadRecord) -> dict[str, Any]:
        archived = record.archived_at is not None
        return {
            "threadId": record.thread_id,
            "agentKey": record.agent_key,
            "cwd": record.cwd,
            "createdAt": record.created_at,
            "updatedAt": record.updated_at,
            "archivedAt": record.archived_at,
            "status": self._state_manager.status(
                record.thread_id,
                archived=archived,
                persisted_status=record.status,
            ),
            "metadata": dict(record.metadata),
        }


_INVALID_PARAM = object()
