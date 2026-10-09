from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from vv_agent import (
    Agent,
    RunBudgetLimits,
    RunConfig,
    ScriptedModelProvider,
    SubAgentConfig,
)
from vv_agent.app_server import (
    ApprovalDecision,
    AppServer,
    AppServerErrorCode,
    ChannelTransport,
    JsonRpcError,
    JsonRpcMessage,
    JsonRpcRequest,
    MessageProcessor,
    ModelListRequest,
    ModelListResponse,
    ModelSummary,
    OutgoingRouter,
    RequestId,
    TurnStartParams,
)
from vv_agent.app_server.host import AgentResolutionRequest, AppServerHost, DefaultAppServerHost, RunConfigResolutionRequest
from vv_agent.app_server.item_mapper import map_run_event
from vv_agent.app_server.schema import _schema_bundle, typescript_schema_bundle
from vv_agent.app_server.thread_state import ThreadStateManager
from vv_agent.app_server.thread_store import ThreadStore
from vv_agent.events import (
    ToolCallPlannedEvent,
)
from vv_agent.runtime.cancellation import CancellationToken
from vv_agent.session.app_server import _KernelThreadStore
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import SessionStore
from vv_agent.types import (
    LLMResponse,
    ToolCall,
)


def _observable_contract() -> dict[str, Any]:
    fixture = Path(__file__).parent / "fixtures" / "parity" / "app_server_observable.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


def _mapped_notifications(
    event: Any,
    *,
    thread_id: str = "thread-tool",
    turn_id: str = "turn-tool",
) -> list[dict[str, Any]]:
    projection = map_run_event(event, thread_id=thread_id, turn_id=turn_id)
    messages: list[dict[str, Any]] = []
    if projection.notification_method is not None:
        messages.append(
            {
                "jsonrpc": "2.0",
                "method": projection.notification_method,
                "params": projection.notification_params,
            }
        )
    messages.extend(
        {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        for method, params in projection.additional_notifications
    )
    return messages


def test_tool_lifecycle_app_server_projection_matches_shared_fixture():
    contract = _observable_contract()["toolLifecycle"]
    planned = ToolCallPlannedEvent(
        run_id="run_tool",
        trace_id="trace_tool",
        event_id="evt_tool_planned",
        created_at=100,
        tool_name="inspect",
        tool_call_id="call_tool",
        arguments={"path": "README.md"},
    )
    assert _mapped_notifications(planned) == contract["plannedHasNoNotification"]["notifications"]
    agent = Agent("fixture", "Delegate.", sub_agents={"child": SubAgentConfig(model="m", description="Answer.")})
    with _real_turn(
        [
            LLMResponse(
                "",
                [
                    ToolCall(
                        "child", "create_sub_task", {"agent_id": "child", "task_description": "go", "wait_for_completion": True}
                    )
                ],
            ),
            LLMResponse("child done"),
            LLMResponse("done"),
        ],
        agent=agent,
        thread_number=2,
    ) as (_server, messages):
        started = [m for m in messages if m.get("method") == "item/started" and m["params"]["type"] == "toolCall"]
        completed = [m for m in messages if m.get("method") == "item/completed" and m["params"]["type"] == "toolCall"]
        assert started == contract["executed"]["startedNotifications"]
        assert completed == contract["executed"]["completedNotifications"][:1]
        assert all(m["params"]["payload"]["executionStarted"] for m in completed)


def test_model_lifecycle_app_server_projection_matches_shared_fixture():
    contract = _observable_contract()["modelLifecycle"]
    with _real_turn([_usage_response()]) as (_server, messages):
        started = [m for m in messages if m.get("method") == "item/started" and m["params"]["type"] == "modelCall"]
        completed = [m for m in messages if m.get("method") == "item/completed" and m["params"]["type"] == "modelCall"]
        assert started == contract["startedNotifications"][:1]
        assert completed == contract["completedNotifications"][:1]
        for notification in [*started, *completed]:
            assert not set(notification["params"]["payload"]).intersection(contract["forbiddenPayloadFields"])


def test_terminal_token_usage_projection_matches_shared_fixture_and_store():
    expected = _observable_contract()["terminal"]["tokenUsageProjection"]["value"]
    with _real_turn([_usage_response()]) as (server, messages):
        payload = messages[-1]["params"]
        assert payload["tokenUsage"] == expected
        assert server.store.read_thread("thread_1").turns[0].result["tokenUsage"] == expected


class _ContractHost:
    def __init__(self) -> None:
        self.model_requests: list[ModelListRequest] = []
        self.agent_requests: list[AgentResolutionRequest] = []
        self.config_requests: list[RunConfigResolutionRequest] = []
        self.base_config = RunConfig(
            model_provider=ScriptedModelProvider.new("scripted", "m", [LLMResponse("done"), LLMResponse("done")]),
            metadata={"host": "base", "shared": "host"},
        )

    def resolve_agent(self, request: AgentResolutionRequest) -> Agent:
        self.agent_requests.append(request)
        return Agent(name=request.agent_key, instructions="Test agent.")

    def build_run_config(self, request: RunConfigResolutionRequest) -> RunConfig:
        self.config_requests.append(request)
        return self.base_config

    def list_models(self, request: ModelListRequest) -> ModelListResponse:
        self.model_requests.append(request)
        return ModelListResponse(
            models=[
                ModelSummary(
                    id="minimal",
                    provider="minimal-provider",
                    display_name="Minimal",
                    metadata={"tier": "standard"},
                ),
                ModelSummary(
                    id="modern",
                    context_length=128_000,
                    supports_tools=True,
                    metadata={"tier": "new"},
                ),
            ]
        )


def _initialized_processor(
    *,
    host: AppServerHost | None = None,
    store: SessionStore | _KernelThreadStore | None = None,
    state_manager: ThreadStateManager | None = None,
) -> tuple[MessageProcessor, ChannelTransport, _KernelThreadStore, ThreadStateManager]:
    transport = ChannelTransport(connection_id="conn_1")
    router = OutgoingRouter()
    router.register_transport(transport)
    resolved_state = state_manager or ThreadStateManager()
    resolved_store = store if isinstance(store, _KernelThreadStore) else ThreadStore(store)
    processor = MessageProcessor(router=router, host=host, store=resolved_store, state_manager=resolved_state)
    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"clientInfo": {"name": "contract-test"}}},
    )
    assert transport.receive_outbound(timeout=1)["id"] == 0
    processor.process_message("conn_1", {"jsonrpc": "2.0", "method": "initialized"})
    return processor, transport, resolved_store, resolved_state


def _response(transport: ChannelTransport, request_id: int) -> dict[str, Any]:
    while True:
        message = transport.receive_outbound(timeout=2)
        if message.get("id") == request_id:
            return message


def test_shared_fixture_enforces_json_rpc_version_and_request_ids() -> None:
    contract = _observable_contract()["jsonRpc"]
    version = contract["version"]

    for request_id in contract["validRequestIds"]:
        message = JsonRpcMessage.from_dict({"jsonrpc": version, "id": request_id, "method": "model/list"}).message
        assert isinstance(message, JsonRpcRequest)
        assert message.id.to_wire() == request_id

    for request_id in contract["invalidRequestIds"]:
        with pytest.raises(ValueError, match="Invalid JSON-RPC message"):
            JsonRpcMessage.from_dict({"jsonrpc": version, "id": request_id, "method": "model/list"})

    for payload in [
        {"id": 1, "method": "model/list"},
        {"jsonrpc": "1.0", "id": 1, "method": "model/list"},
    ]:
        with pytest.raises(ValueError, match="Invalid JSON-RPC message"):
            JsonRpcMessage.from_dict(payload)

    assert contract["errorResponseAllowsNullId"] is True
    error = JsonRpcMessage.from_dict(
        {
            "jsonrpc": version,
            "id": None,
            "error": {"code": AppServerErrorCode.PARSE_ERROR, "message": "Parse error"},
        }
    ).message
    assert isinstance(error, JsonRpcError)
    assert error.id.to_wire() is None


def test_shared_fixture_requires_object_input_items(surface) -> None:
    contract = _observable_contract()["input"]
    valid = contract["valid"]
    assert TurnStartParams(thread_id="thread_contract", input=valid).to_dict()["input"] == valid

    processor, transport, store, _state = _initialized_processor(store=surface.store)
    thread = store.create_thread(agent_key="default")
    for request_id, invalid_item in enumerate(contract["invalid"], start=1):
        processor.process_message(
            "conn_1",
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "turn/start",
                "params": {"threadId": thread.thread_id, "input": [invalid_item]},
            },
        )
        assert _response(transport, request_id)["error"]["code"] == AppServerErrorCode.INVALID_PARAMS


def test_shared_fixture_live_and_replay_item_payloads_match_in_epoch_seconds():
    contract = _observable_contract()
    with _real_turn([_usage_response()]) as (server, messages):
        live = [m["params"] for m in messages if m.get("method") == "item/completed"]
        replay = [item.to_dict() for item in server.store.read_thread("thread_1").items]
        assert contract["liveReplay"]["payloadMustMatch"]
        assert all(item in replay for item in live)
        assert all(item["createdAt"] == 1041.102 for item in live)
        assert contract["timestamps"]["eventMillis"] / 1000 == contract["timestamps"]["eventSeconds"]


def test_shared_fixture_thread_start_order_and_nullability(surface) -> None:
    contract = _observable_contract()
    processor, transport, _store, _state = _initialized_processor(store=surface.store)
    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "params": {}},
    )
    response = transport.receive_outbound(timeout=1)
    notification = transport.receive_outbound(timeout=1)

    observed = ["response", notification["method"]]
    assert observed == contract["ordering"]["threadStart"]
    assert response["result"]["cwd"] == contract["nullability"]["threadStartResponse"]["cwd"]


def test_shared_fixture_turn_start_and_terminal_order(surface) -> None:
    contract = _observable_contract()
    provider = ScriptedModelProvider.new("test", "m", [LLMResponse("done")])
    host = DefaultAppServerHost(agent=Agent("default", "Work.", model="m"), run_config=RunConfig(model_provider=provider))
    processor, transport, store, state = _initialized_processor(host=host, store=surface.store)
    thread = store.create_thread(agent_key="default")
    state.subscribe(thread.thread_id, "conn_1")
    processor.process_message(
        "conn_1",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "turn/start",
            "params": {"threadId": thread.thread_id, "input": contract["input"]["valid"]},
        },
    )
    started_messages = [transport.receive_outbound(timeout=2) for _ in range(3)]
    started_order = ["response" if "result" in message else str(message["method"]) for message in started_messages]
    assert started_order == contract["ordering"]["turnStart"]
    terminal_messages = []
    while True:
        message = transport.receive_outbound(timeout=2)
        terminal_messages.append(message)
        if message.get("method") == "turn/completed":
            break
    terminal_order = [
        str(message["method"]) for message in terminal_messages if message.get("method") in contract["ordering"]["turnTerminal"]
    ]
    assert terminal_order == contract["ordering"]["turnTerminal"]
    assert terminal_messages[-1]["params"]["status"] == "completed"
    snapshot = store.read_thread(thread.thread_id)
    assert snapshot.turns[0].input == contract["input"]["valid"]
    assert snapshot.thread.status == contract["terminal"]["threadStatusAfterTurn"]


def test_wait_user_turn_projects_as_interrupted_without_failure_error():
    with _real_turn([LLMResponse("assistant draft", [ToolCall("ask", "ask_user", {"question": "Choose one"})])]) as (
        server,
        messages,
    ):
        payload = messages[-1]["params"]
        assert payload["status"] == "interrupted"
        assert payload["waitReason"] == "Choose one"
        assert "error" not in payload
        state, rows, _ = server.kernel.store.read_state("thread_1")
        assert state.active_turn_id == "thread_1/turn/turn_1"
        assert not any(row.record.kind == "turn_ended" for row in rows)
        assert server.store.read_thread("thread_1").thread.status == "interrupted"


def test_cancelled_turn_projects_as_failed_with_error():
    token = CancellationToken()
    token.cancel()
    with _real_turn([], config=RunConfig(cancellation_token=token)) as (server, messages):
        payload = messages[-1]["params"]
        assert payload["status"] == "failed"
        assert payload["completionReason"] == "cancelled"
        assert "cancel" in payload["error"]
        assert server.store.read_thread("thread_1").turns[0].result["completionReason"] == "cancelled"


def test_budget_exhaustion_projects_typed_usage_to_turn_and_store():
    with _real_turn([_usage_response()], config=RunConfig(budget_limits=RunBudgetLimits(max_total_tokens=10))) as (
        server,
        messages,
    ):
        payload = messages[-1]["params"]
        assert payload["status"] == "failed"
        assert payload["completionReason"] == "budget_exhausted"
        assert payload["budgetUsage"]["total_tokens"] == 15
        assert payload["budgetExhaustion"]["enforcement_boundary"] == "model_call_complete"
        retained = server.store.read_thread("thread_1").turns[0].result
        assert retained["budgetUsage"] == payload["budgetUsage"]
        assert retained["budgetExhaustion"] == payload["budgetExhaustion"]


def test_shared_fixture_snapshot_nullability_and_restart_recovery(tmp_path):
    with SQLiteStore.standalone(str(tmp_path / "threads.sqlite")) as store:
        store.install_schema()
        with _real_turn([LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Choose"})])], store=store) as (
            server,
            _messages,
        ):
            before = server.store.read_thread("thread_1")
            reopened = ThreadStore(store).read_thread("thread_1")
            assert reopened == before
            assert reopened.thread.status == "interrupted"
            assert reopened.thread.cwd is None
            assert reopened.thread.archived_at is None
            assert reopened.turns[0].completed_at is None
            assert reopened.turns[0].run_id == reopened.turns[0].turn_id


def test_shared_fixture_connection_can_reinitialize_after_disconnect() -> None:
    contract = _observable_contract()
    processor, _transport, _store, _state = _initialized_processor()
    processor._router.unregister_transport("conn_1")
    replacement = ChannelTransport(connection_id="conn_1")
    processor._router.register_transport(replacement)
    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": {"name": "restarted"}}},
    )

    assert contract["restart"]["connectionIdCanReinitialize"] is True
    assert replacement.receive_outbound(timeout=1)["id"] == 1


def test_shared_fixture_duplicate_id_disconnect_cleanup_and_case_sensitivity() -> None:
    contract = _observable_contract()
    transport = ChannelTransport(connection_id="conn_1")
    router = OutgoingRouter()
    router.register_transport(transport)
    request_id = RequestId("approval_contract")
    pending = router.send_server_request(
        "conn_1",
        "approval/request",
        {"threadId": "thread_1", "turnId": "turn_1"},
        request_id=request_id,
    )
    assert router.pending_server_request_count() == 1

    assert contract["approval"]["duplicateServerRequestId"] == "reject"
    with pytest.raises(ValueError, match="Duplicate server request id"):
        router.send_server_request(
            "conn_1",
            "approval/request",
            {"threadId": "thread_1", "turnId": "turn_1"},
            request_id=request_id,
        )
    assert router.pending_server_request_count() == 1

    router.unregister_transport("conn_1")
    assert router.pending_server_request_count() == 0
    with pytest.raises(RuntimeError, match="client_disconnected"):
        pending.result(timeout=0)
    assert contract["approval"]["disconnectDecision"] == "retained_owner_until_absolute_deadline"

    for decision in contract["approval"]["decisions"]:
        assert ApprovalDecision.from_wire(decision).value == decision
        with pytest.raises(ValueError):
            ApprovalDecision.from_wire(decision.upper())
    assert contract["approval"]["caseSensitive"] is True


def test_model_list_forwards_optional_filters_and_emits_canonical_superset(surface) -> None:
    from vv_agent.app_server.server import AppServer

    host = _ContractHost()
    transport = ChannelTransport(connection_id="conn_1")
    processor = AppServer(transport=transport, host=host, store=surface.store).processor
    processor.process_message(
        "conn_1", {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"clientInfo": {"name": "contract-test"}}}
    )
    assert transport.receive_outbound(timeout=1)["id"] == 0

    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 1, "method": "model/list", "params": {"agentKey": "writer", "provider": "openai"}},
    )
    response = _response(transport, 1)

    assert host.model_requests == [ModelListRequest(agent_key="writer", provider="openai")]
    assert response["result"] == {
        "models": [
            {
                "id": "minimal",
                "provider": "minimal-provider",
                "displayName": "Minimal",
                "supportsTools": False,
                "metadata": {"tier": "standard"},
            },
            {
                "id": "modern",
                "contextLength": 128_000,
                "supportsTools": True,
                "metadata": {"tier": "new"},
            },
        ]
    }


def test_thread_resume_read_and_list_options_are_applied() -> None:
    processor, transport, store, state = _initialized_processor()
    active_1 = store.create_thread(agent_key="a")
    archived_1 = store.create_thread(agent_key="b")
    active_2 = store.create_thread(agent_key="c")
    archived_2 = store.create_thread(agent_key="d")
    store.archive_thread(archived_1.thread_id)
    store.archive_thread(archived_2.thread_id)
    store.kernel.start(
        active_1.thread_id,
        Agent("fixture", "Answer."),
        RunConfig(model_provider=ScriptedModelProvider.new("scripted", "m", [LLMResponse("done")])),
        {"text": "go", "app_server": {"owner": "conn_1", "input": [], "metadata": {}}},
        input_id="initial",
    ).result()
    items = store.read_thread(active_1.thread_id).items
    assert len(items) >= 3

    processor.process_message(
        "conn_1",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "thread/read",
            "params": {"threadId": active_1.thread_id, "afterItemId": items[0].item_id},
        },
    )
    assert [item["itemId"] for item in _response(transport, 1)["result"]["items"]] == [item.item_id for item in items[1:]]

    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 2, "method": "thread/resume", "params": {"threadId": active_1.thread_id, "subscribe": False}},
    )
    assert _response(transport, 2)["result"]["thread"]["threadId"] == active_1.thread_id
    assert state.subscribers(active_1.thread_id) == set()

    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 3, "method": "thread/list", "params": {"archived": True, "offset": 1, "limit": 1}},
    )
    assert [thread["threadId"] for thread in _response(transport, 3)["result"]["threads"]] == [archived_2.thread_id]

    processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 4, "method": "thread/list", "params": {"includeArchived": True}})
    assert [thread["threadId"] for thread in _response(transport, 4)["result"]["threads"]] == [
        active_1.thread_id,
        archived_1.thread_id,
        active_2.thread_id,
        archived_2.thread_id,
    ]


def test_turn_metadata_is_per_turn_and_does_not_mutate_host_config():
    host = _ContractHost()
    processor, transport, store, state = _initialized_processor(host=host)
    thread = store.create_thread(agent_key="default", metadata={"thread": "base", "shared": "thread"})
    state.subscribe(thread.thread_id, "conn_1")
    try:
        for index, metadata in enumerate(({"shared": "turn", "turnOnly": 1}, {"second": 2}), 1):
            processor.process_message(
                "conn_1",
                {
                    "jsonrpc": "2.0",
                    "id": index,
                    "method": "turn/start",
                    "params": {"threadId": thread.thread_id, "input": [{"type": "text", "text": "go"}], "metadata": metadata},
                },
            )
            _response(transport, index)
            _completed(transport)
            processor._run_adapter.join()
        assert host.config_requests[0].metadata == {"thread": "base", "shared": "turn", "turnOnly": 1}
        assert host.config_requests[1].metadata == {"thread": "base", "shared": "thread", "second": 2}
        assert host.base_config.metadata == {"host": "base", "shared": "host"}
        rows = store.kernel.store.read_state(thread.thread_id)[1]
        tasks = [r.record.payload["definition"]["task"] for r in rows if r.record.kind == "turn_started"]
        assert tasks[0]["metadata"]["turnOnly"] == 1
        assert "turnOnly" not in tasks[1]["metadata"]
        assert tasks[1]["metadata"]["second"] == 2
    finally:
        store.kernel.close()


def test_active_turn_and_missing_required_params_are_rejected() -> None:
    processor, transport, store, state = _initialized_processor()
    thread = store.create_thread(agent_key="default")
    state.set_active_turn(thread_id=thread.thread_id, turn_id="turn_active", handle=object())
    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 1, "method": "turn/start", "params": {"threadId": thread.thread_id}},
    )
    assert _response(transport, 1)["error"]["code"] == AppServerErrorCode.INVALID_PARAMS

    for request_id, method in enumerate(
        [
            "thread/resume",
            "thread/read",
            "thread/archive",
            "thread/unsubscribe",
            "turn/start",
            "turn/resume",
            "turn/steer",
            "turn/followUp",
            "turn/interrupt",
            "approval/resolve",
        ],
        start=2,
    ):
        processor.process_message("conn_1", {"jsonrpc": "2.0", "id": request_id, "method": method})
        assert _response(transport, request_id)["error"]["code"] == AppServerErrorCode.INVALID_PARAMS


def test_schema_matches_optional_and_required_runtime_params() -> None:
    durable_resume = _observable_contract()["durableResume"]
    bundle = _schema_bundle()
    client_request = bundle["ClientRequest"]
    definitions = client_request["$defs"]
    variants = {variant["properties"]["method"]["const"]: variant for variant in client_request["oneOf"]}

    assert definitions["ModelListParams"]["properties"] == {
        "agentKey": {"type": "string"},
        "provider": {"type": "string"},
    }
    assert definitions["ThreadResumeParams"]["properties"]["subscribe"] == {"type": "boolean"}
    assert definitions["ThreadReadParams"]["properties"]["afterItemId"] == {"type": "string"}
    assert set(definitions["ThreadListParams"]["properties"]) == {
        "includeArchived",
        "archived",
        "offset",
        "limit",
    }
    assert definitions["TurnStartParams"]["required"] == ["threadId"]
    assert definitions["TurnResumeParams"]["required"] == durable_resume["requestFields"]
    assert set(definitions["TurnResumeParams"]["properties"]) == set(durable_resume["requestFields"])
    assert set(definitions["TurnResumeResponse"]["properties"]) == set(durable_resume["responseFields"])
    assert "CheckpointSummary" not in definitions
    assert "InterruptionSummary" not in definitions
    assert definitions["ModelSummary"]["required"] == ["id", "supportsTools"]
    assert "params" not in variants["model/list"]["required"]
    assert "params" not in variants["thread/start"]["required"]
    assert "params" not in variants["thread/list"]["required"]
    assert "params" not in variants["schema/export"]["required"]
    assert "params" in variants["thread/read"]["required"]
    assert "params" in variants["turn/resume"]["required"]

    typescript = typescript_schema_bundle()["ClientRequest.ts"]
    assert "provider?: string" in typescript
    assert "afterItemId?: string" in typescript
    assert "supportsTools: boolean" in typescript
    assert "export interface TurnResumeParams" in typescript
    assert "export interface TurnResumeResponse" in typescript
    assert "import " not in typescript


def _usage_response():
    return LLMResponse("done", raw={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})


def _completed(transport):
    messages = []
    while True:
        message = transport.receive_outbound(timeout=3)
        messages.append(message)
        if message.get("method") == "turn/completed":
            return messages


@contextmanager
def _real_turn(steps, *, agent=None, config=None, thread_number=1, store=None):
    transport = ChannelTransport(connection_id="conn_1")
    server = AppServer(
        transport=transport,
        store=store,
        host=DefaultAppServerHost(
            agent=agent or Agent("fixture", "Answer."),
            run_config=replace(config or RunConfig(), model_provider=ScriptedModelProvider.new("scripted", "m", steps)),
        ),
    )
    assert isinstance(server.kernel.store, SQLiteStore)
    server.kernel.store.connection.create_function("session_now_ms", 0, lambda: 1041102)
    try:
        server.processor.process_message(
            "conn_1", {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"clientInfo": {"name": "producer"}}}
        )
        _response(transport, 0)
        server.processor.process_message("conn_1", {"jsonrpc": "2.0", "method": "initialized"})
        for _ in range(thread_number - 1):
            server.store.create_thread(agent_key="default")
        server.processor.process_message("conn_1", {"jsonrpc": "2.0", "id": 1, "method": "thread/start"})
        thread = _response(transport, 1)["result"]["threadId"]
        transport.receive_outbound(timeout=1)
        server.processor.process_message(
            "conn_1",
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "turn/start",
                "params": {"threadId": thread, "input": [{"type": "text", "text": "go"}]},
            },
        )
        _response(transport, 2)
        messages = _completed(transport)
        server.run_adapter.join()
        yield server, messages
    finally:
        for handle in server.kernel.handles:
            handle.cancel("test teardown")
        server.run_adapter.join()
        server.kernel.close()
