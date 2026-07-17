from __future__ import annotations

from collections import deque
from typing import Any

from vv_agent.agent import Agent
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.app_server.protocol import (
    ApprovalResolveParams,
    ClientInfo,
    InitializeParams,
    ModelListRequest,
    ThreadArchiveParams,
    ThreadListParams,
    ThreadReadParams,
    ThreadResumeParams,
    ThreadStartParams,
    ThreadUnsubscribeParams,
    TurnFollowUpParams,
    TurnInterruptParams,
    TurnResumeParams,
    TurnStartParams,
    TurnSteerParams,
)
from vv_agent.app_server.server import AppServer
from vv_agent.app_server.transport import ChannelTransport
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.run_config import RunConfig
from vv_agent.types import LLMResponse, ToolCall


class AppServerClientError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class AppServerClient:
    def __init__(self, *, server: AppServer, transport: ChannelTransport) -> None:
        if server.transport is not transport:
            raise ValueError("AppServerClient transport must be the server transport")
        self._server = server
        self._transport = transport
        self._next_request_id = 1
        self._backlog: deque[dict[str, Any]] = deque()
        self._closed = False

    @classmethod
    def for_host(
        cls,
        host: DefaultAppServerHost,
        *,
        connection_id: str = "client",
        capacity: int = 128,
    ) -> AppServerClient:
        transport = ChannelTransport(
            connection_id=connection_id,
            inbound_capacity=capacity,
            outbound_capacity=capacity,
        )
        return cls(server=AppServer(transport=transport, host=host), transport=transport)

    def initialize(self, info: ClientInfo) -> dict[str, Any]:
        response = self._send_request("initialize", InitializeParams(client_info=info).to_dict())
        self._send_notification("initialized")
        return response

    def start_thread(self, params: ThreadStartParams) -> dict[str, Any]:
        return self._send_request("thread/start", params.to_dict())

    def resume_thread(self, params: ThreadResumeParams) -> dict[str, Any]:
        return self._send_request("thread/resume", params.to_dict())

    def read_thread(self, params: ThreadReadParams) -> dict[str, Any]:
        return self._send_request("thread/read", params.to_dict())

    def list_threads(self, params: ThreadListParams | None = None) -> dict[str, Any]:
        return self._send_request("thread/list", (params or ThreadListParams()).to_dict())

    def archive_thread(self, params: ThreadArchiveParams) -> dict[str, Any]:
        return self._send_request("thread/archive", params.to_dict())

    def unsubscribe_thread(self, params: ThreadUnsubscribeParams) -> dict[str, Any]:
        return self._send_request("thread/unsubscribe", params.to_dict())

    def start_turn(self, params: TurnStartParams) -> dict[str, Any]:
        return self._send_request("turn/start", params.to_dict())

    def resume_turn(self, params: TurnResumeParams) -> dict[str, Any]:
        return self._send_request("turn/resume", params.to_dict())

    def interrupt_turn(self, params: TurnInterruptParams) -> dict[str, Any]:
        return self._send_request("turn/interrupt", params.to_dict())

    def steer_turn(self, params: TurnSteerParams) -> dict[str, Any]:
        return self._send_request("turn/steer", params.to_dict())

    def follow_up_turn(self, params: TurnFollowUpParams) -> dict[str, Any]:
        return self._send_request("turn/followUp", params.to_dict())

    def resolve_approval(self, params: ApprovalResolveParams) -> None:
        self.send_response(
            params.request_id,
            {
                "decision": params.decision.value,
                "reason": params.reason,
                "metadata": dict(params.metadata),
            },
        )

    def resolve_approval_request(self, params: ApprovalResolveParams) -> dict[str, Any]:
        return self._send_request("approval/resolve", params.to_dict())

    def send_response(self, request_id: str | int, result: dict[str, Any]) -> None:
        self._process({"jsonrpc": "2.0", "id": request_id, "result": result})

    def list_models(self, params: ModelListRequest | None = None) -> dict[str, Any]:
        return self._send_request("model/list", (params or ModelListRequest()).to_dict())

    def export_schema(self) -> dict[str, Any]:
        return self._send_request("schema/export", {})

    def next_message(self, *, timeout: float = 3.0) -> dict[str, Any]:
        self._ensure_open()
        if self._backlog:
            return self._backlog.popleft()
        try:
            return self._transport.receive_outbound(timeout=timeout)
        except Exception as exc:
            raise AppServerClientError(f"failed waiting for App Server message: {exc}") from exc

    def close(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        self._backlog.clear()
        try:
            self._server.processor.disconnect_connection(self._transport.connection_id)
        finally:
            self._transport.close()
        return True

    def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_request_id
        self._next_request_id += 1
        self._process({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        return self._wait_response(request_id, method)

    def _wait_response(self, request_id: int, method: str) -> dict[str, Any]:
        for _ in range(len(self._backlog)):
            message = self._backlog.popleft()
            if message.get("id") == request_id:
                return self._response_result(message, method)
            self._backlog.append(message)
        while True:
            try:
                message = self._transport.receive_outbound(timeout=3.0)
            except Exception as exc:
                raise AppServerClientError(f"failed waiting for App Server {method} response: {exc}") from exc
            if message.get("id") == request_id:
                return self._response_result(message, method)
            self._backlog.append(message)

    @staticmethod
    def _response_result(message: dict[str, Any], method: str) -> dict[str, Any]:
        error = message.get("error")
        if isinstance(error, dict):
            raw_code = error.get("code")
            code = raw_code if isinstance(raw_code, int) and not isinstance(raw_code, bool) else None
            raise AppServerClientError(
                str(error.get("message") or "App Server request failed"),
                code=code,
                data=error.get("data"),
            )
        result = message.get("result")
        if not isinstance(result, dict):
            raise AppServerClientError(f"App Server {method} returned a non-object result")
        return result

    def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._process(payload)

    def _process(self, payload: dict[str, Any]) -> None:
        self._ensure_open()
        self._server.processor.process_message(self._transport.connection_id, payload)

    def _ensure_open(self) -> None:
        if self._closed:
            raise AppServerClientError("App Server client is closed")


def run_debug_message(message: str) -> list[dict[str, object]]:
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            )
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
        return (
            llm,
            ResolvedModelConfig(
                backend="debug",
                requested_model="debug-model",
                selected_model="debug-model",
                model_id="debug-model",
                endpoint_options=[EndpointOption(endpoint=endpoint, model_id="debug-model")],
            ),
        )

    client = AppServerClient.for_host(
        DefaultAppServerHost(
            agent=Agent(name="assistant", instructions="Answer.", model="debug-model"),
            run_config=RunConfig(model_provider=model_provider, max_cycles=1),
        ),
        connection_id="debug",
    )
    client.initialize(ClientInfo(name="debug"))
    thread = client.start_thread(ThreadStartParams(agent_key="default"))
    client.start_turn(
        TurnStartParams(
            thread_id=str(thread["threadId"]),
            input=[{"type": "text", "text": message}],
        )
    )

    outbound: list[dict[str, object]] = []
    try:
        while True:
            item = client.next_message(timeout=10)
            outbound.append(item)
            if item.get("method") == "approval/request":
                client.send_response(
                    item["id"],
                    {"decision": "allow", "reason": "Approved by the debug client."},
                )
            if item.get("method") == "turn/completed":
                return outbound
    finally:
        client.close()
