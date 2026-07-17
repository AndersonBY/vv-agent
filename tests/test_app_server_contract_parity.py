from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent import (
    Agent,
    BudgetDimension,
    BudgetEnforcementBoundary,
    BudgetExhaustion,
    BudgetExhaustionReason,
    BudgetUsageSnapshot,
    RunConfig,
)
from vv_agent.app_server import (
    ApprovalDecision,
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
    ThreadItem,
    TurnStartParams,
)
from vv_agent.app_server.host import AgentResolutionRequest, AppServerHost, RunConfigResolutionRequest
from vv_agent.app_server.run_adapter import RunAdapter, StartedTurn
from vv_agent.app_server.schema import _schema_bundle, typescript_schema_bundle
from vv_agent.app_server.thread_state import ThreadStateManager
from vv_agent.app_server.thread_store import ThreadStore
from vv_agent.result import RunResult
from vv_agent.run_handle import RunHandle
from vv_agent.runner import Runner
from vv_agent.types import AgentResult, AgentStatus, CompletionReason


def _observable_contract() -> dict[str, Any]:
    fixture = Path(__file__).parent / "fixtures" / "parity" / "app_server_observable_v1.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


def _status_projection(name: str) -> dict[str, Any]:
    return next(case for case in _observable_contract()["terminal"]["agentStatusProjection"] if case["name"] == name)


class _ContractHost:
    def __init__(self) -> None:
        self.model_requests: list[ModelListRequest] = []
        self.agent_requests: list[AgentResolutionRequest] = []
        self.config_requests: list[RunConfigResolutionRequest] = []
        self.base_config = RunConfig(metadata={"host": "base", "shared": "host"})

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
                ModelSummary("legacy", "legacy-provider", "Legacy", {"tier": "old"}),
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
    store: ThreadStore | None = None,
    state_manager: ThreadStateManager | None = None,
) -> tuple[MessageProcessor, ChannelTransport, ThreadStore, ThreadStateManager]:
    transport = ChannelTransport(connection_id="conn_1")
    router = OutgoingRouter()
    router.register_transport(transport)
    resolved_store = store or ThreadStore()
    resolved_state = state_manager or ThreadStateManager()
    processor = MessageProcessor(
        router=router,
        host=host,
        store=resolved_store,
        state_manager=resolved_state,
    )
    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"clientInfo": {"name": "contract-test"}}},
    )
    assert transport.receive_outbound(timeout=1)["id"] == 0
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


def test_shared_fixture_requires_object_input_items() -> None:
    contract = _observable_contract()["input"]
    valid = contract["valid"]
    assert TurnStartParams(thread_id="thread_contract", input=valid).to_dict()["input"] == valid

    processor, transport, store, _state = _initialized_processor()
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


def test_shared_fixture_live_and_replay_item_payloads_match_in_epoch_seconds() -> None:
    contract = _observable_contract()
    expected = dict(contract["liveReplay"]["item"])
    timestamps = contract["timestamps"]
    assert expected["createdAt"] == timestamps["eventSeconds"]
    assert timestamps["eventMillis"] / 1000 == timestamps["eventSeconds"]

    store = ThreadStore()
    thread = store.create_thread(agent_key="default")
    turn = store.create_turn(thread_id=thread.thread_id, input=[])
    expected["threadId"] = thread.thread_id
    expected["turnId"] = turn.turn_id
    item = ThreadItem(
        item_id=expected["itemId"],
        thread_id=expected["threadId"],
        turn_id=expected["turnId"],
        item_type=expected["type"],
        status=expected["status"],
        payload=expected["payload"],
        created_at=expected["createdAt"],
        updated_at=expected["updatedAt"],
    )

    live_payload = item.to_dict()
    store.append_item(item, run_event_id="evt_contract")
    replay_payload = store.read_thread(thread.thread_id).items[0].to_dict()

    assert contract["liveReplay"]["payloadMustMatch"] is True
    assert live_payload == expected
    assert replay_payload == live_payload


def test_shared_fixture_thread_start_order_and_nullability() -> None:
    contract = _observable_contract()
    processor, transport, _store, _state = _initialized_processor()
    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "params": {}},
    )
    response = transport.receive_outbound(timeout=1)
    notification = transport.receive_outbound(timeout=1)

    observed = ["response", notification["method"]]
    assert observed == contract["ordering"]["threadStart"]
    assert response["result"]["cwd"] == contract["nullability"]["threadStartResponse"]["cwd"]


def test_shared_fixture_turn_start_and_terminal_order(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = _observable_contract()
    store = ThreadStore()
    state = ThreadStateManager()
    thread = store.create_thread(agent_key="default")

    monkeypatch.setattr(Runner, "start", classmethod(lambda cls, agent, input, run_config=None: object()))
    monkeypatch.setattr(RunAdapter, "_pump_events", lambda self, connection_id, started: None)
    processor, transport, _store, _state = _initialized_processor(store=store, state_manager=state)
    state.subscribe(thread.thread_id, "conn_1")
    processor.process_message(
        "conn_1",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "turn/start",
            "params": {"threadId": thread.thread_id, "input": []},
        },
    )
    started_messages = [transport.receive_outbound(timeout=1) for _ in range(3)]
    started_order = ["response" if "result" in message else str(message["method"]) for message in started_messages]
    assert started_order == contract["ordering"]["turnStart"]

    snapshot = store.read_thread(thread.thread_id)
    active = state.active_turn(thread.thread_id)
    assert active is not None
    adapter = RunAdapter(
        host=_ContractHost(),
        store=store,
        state_manager=state,
        router=processor._router,
    )
    adapter._complete_turn(
        "conn_1",
        StartedTurn(thread=snapshot.thread, turn=snapshot.turns[0], handle=active.handle),
        result=None,
        error=RuntimeError("contract failure"),
    )
    terminal_messages: list[dict[str, Any]] = []
    while True:
        message = transport.receive_outbound(timeout=1)
        terminal_messages.append(message)
        if message.get("method") == "turn/completed":
            break
    terminal_order = [
        str(message["method"]) for message in terminal_messages if message.get("method") in contract["ordering"]["turnTerminal"]
    ]
    assert terminal_order == contract["ordering"]["turnTerminal"]
    assert state.status(thread.thread_id) == contract["terminal"]["threadStatusAfterTurn"]


def test_wait_user_turn_projects_as_interrupted_without_failure_error() -> None:
    contract = _observable_contract()
    expected = _status_projection("wait_user_is_interrupted_without_error")
    processor, transport, store, state = _initialized_processor()
    thread = store.create_thread(agent_key="default")
    turn = store.create_turn(thread_id=thread.thread_id, input=[], status="running")
    state.subscribe(thread.thread_id, "conn_1")
    adapter = RunAdapter(
        host=_ContractHost(),
        store=store,
        state_manager=state,
        router=processor._router,
    )
    raw_result = AgentResult(
        status=AgentStatus.WAIT_USER,
        messages=[],
        cycles=[],
        wait_reason="Choose one",
        completion_reason=CompletionReason.WAIT_USER,
        partial_output="assistant draft",
    )
    result = RunResult(
        input="choose",
        new_items=[],
        final_output="Choose one",
        status=AgentStatus.WAIT_USER,
        raw_result=raw_result,
        run_id="run_wait_user",
        trace_id="trace_wait_user",
        agent_name="default",
    )

    adapter._complete_turn(
        "conn_1",
        StartedTurn(thread=thread, turn=turn, handle=cast(RunHandle, object())),
        result=result,
        error=None,
    )

    messages: list[dict[str, Any]] = []
    while True:
        message = transport.receive_outbound(timeout=1)
        messages.append(message)
        if message.get("method") == "turn/completed":
            break
    payload = next(message["params"] for message in messages if message.get("method") == "turn/completed")
    stored_turn = store.read_thread(thread.thread_id).turns[0]

    assert payload["status"] == expected["turnStatus"]
    assert payload["status"] in contract["terminal"]["turnStatuses"]
    assert payload["completionReason"] == expected["completionReason"]
    assert payload["partialOutput"] == "assistant draft"
    assert ("error" in payload) is (expected["errorField"] == "present")
    assert stored_turn.status == expected["turnStatus"]
    assert stored_turn.result["completionReason"] == expected["completionReason"]
    assert ("error" in stored_turn.result) is (expected["errorField"] == "present")


def test_cancelled_turn_projects_as_failed_with_error() -> None:
    expected = _status_projection("cancelled_failure_stays_failed")
    processor, transport, store, state = _initialized_processor()
    thread = store.create_thread(agent_key="default")
    turn = store.create_turn(thread_id=thread.thread_id, input=[], status="running")
    state.subscribe(thread.thread_id, "conn_1")
    adapter = RunAdapter(
        host=_ContractHost(),
        store=store,
        state_manager=state,
        router=processor._router,
    )
    raw_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=[],
        cycles=[],
        error="run cancelled",
        completion_reason=CompletionReason.CANCELLED,
    )
    result = RunResult(
        input="cancel",
        new_items=[],
        final_output="run cancelled",
        status=AgentStatus.FAILED,
        raw_result=raw_result,
        run_id="run_cancelled",
        trace_id="trace_cancelled",
        agent_name="default",
    )

    adapter._complete_turn(
        "conn_1",
        StartedTurn(thread=thread, turn=turn, handle=cast(RunHandle, object())),
        result=result,
        error=None,
    )

    messages: list[dict[str, Any]] = []
    while True:
        message = transport.receive_outbound(timeout=1)
        messages.append(message)
        if message.get("method") == "turn/completed":
            break
    payload = next(message["params"] for message in messages if message.get("method") == "turn/completed")
    stored_turn = store.read_thread(thread.thread_id).turns[0]

    assert payload["status"] == expected["turnStatus"]
    assert payload["completionReason"] == expected["completionReason"]
    assert ("error" in payload) is (expected["errorField"] == "present")
    assert stored_turn.status == expected["turnStatus"]
    assert stored_turn.result["completionReason"] == expected["completionReason"]
    assert ("error" in stored_turn.result) is (expected["errorField"] == "present")


def test_budget_exhaustion_projects_typed_usage_to_turn_and_store() -> None:
    expected = _status_projection("budget_exhaustion_is_failed_with_typed_observation")
    processor, transport, store, state = _initialized_processor()
    thread = store.create_thread(agent_key="default")
    turn = store.create_turn(thread_id=thread.thread_id, input=[], status="running")
    state.subscribe(thread.thread_id, "conn_1")
    adapter = RunAdapter(
        host=_ContractHost(),
        store=store,
        state_manager=state,
        router=processor._router,
    )
    usage = BudgetUsageSnapshot(cycles=1, total_tokens=12, uncached_input_tokens=12, elapsed_ms=7)
    exhaustion = BudgetExhaustion(
        dimension=BudgetDimension.TOTAL_TOKENS,
        reason=BudgetExhaustionReason.LIMIT_EXCEEDED,
        limit=10,
        observed=12,
        attempted_increment=None,
        overshoot=2,
        unit="tokens",
        enforcement_boundary=BudgetEnforcementBoundary.LLM_COMPLETE,
    )
    raw_result = AgentResult(
        status=AgentStatus.FAILED,
        messages=[],
        cycles=[],
        error="Run budget exhausted.",
        completion_reason=CompletionReason.BUDGET_EXHAUSTED,
        partial_output="draft",
        budget_usage=usage,
        budget_exhaustion=exhaustion,
    )
    result = RunResult(
        input="run",
        new_items=[],
        final_output="Run budget exhausted.",
        status=AgentStatus.FAILED,
        raw_result=raw_result,
        run_id="run_budget",
        trace_id="trace_budget",
        agent_name="default",
    )

    adapter._complete_turn(
        "conn_1",
        StartedTurn(thread=thread, turn=turn, handle=cast(RunHandle, object())),
        result=result,
        error=None,
    )

    messages: list[dict[str, Any]] = []
    while True:
        message = transport.receive_outbound(timeout=1)
        messages.append(message)
        if message.get("method") == "turn/completed":
            break
    payload = next(message["params"] for message in messages if message.get("method") == "turn/completed")
    stored_turn = store.read_thread(thread.thread_id).turns[0]

    assert payload["status"] == expected["turnStatus"]
    assert payload["completionReason"] == expected["completionReason"]
    assert payload["budgetUsage"] == usage.to_dict()
    assert payload["budgetExhaustion"] == exhaustion.to_dict()
    assert ("error" in payload) is (expected["errorField"] == "present")
    assert stored_turn.result["budgetUsage"] == usage.to_dict()
    assert stored_turn.result["budgetExhaustion"] == exhaustion.to_dict()


def test_shared_fixture_snapshot_nullability_and_restart_recovery(tmp_path: Path) -> None:
    contract = _observable_contract()
    database = tmp_path / "app-server.sqlite3"
    first_store = ThreadStore(database)
    thread = first_store.create_thread(agent_key="default")
    first_store.create_turn(thread_id=thread.thread_id, input=[])
    stale_state = ThreadStateManager()
    stale_state.set_status(thread.thread_id, "running")

    restarted_store = ThreadStore(database)
    restarted_state = ThreadStateManager()
    processor, transport, _store, _state = _initialized_processor(
        store=restarted_store,
        state_manager=restarted_state,
    )
    processor.process_message(
        "conn_1",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "thread/read",
            "params": {"threadId": thread.thread_id},
        },
    )
    result = _response(transport, 1)["result"]

    assert result["thread"]["status"] == contract["restart"]["staleRunningThreadStatus"]
    for field, value in contract["nullability"]["threadSnapshot"].items():
        assert result["thread"][field] == value
    for field, value in contract["nullability"]["turnSnapshot"].items():
        assert result["turns"][0][field] == value


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
    assert contract["approval"]["disconnectDecision"] == ApprovalDecision.TIMEOUT.value

    for decision in contract["approval"]["decisions"]:
        assert ApprovalDecision.from_wire(decision).value == decision
        with pytest.raises(ValueError):
            ApprovalDecision.from_wire(decision.upper())
    assert contract["approval"]["caseSensitive"] is True


def test_model_list_forwards_optional_filters_and_emits_canonical_superset() -> None:
    host = _ContractHost()
    processor, transport, _store, _state = _initialized_processor(host=host)

    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 1, "method": "model/list", "params": {"agentKey": "writer", "provider": "openai"}},
    )
    response = _response(transport, 1)

    assert host.model_requests == [ModelListRequest(agent_key="writer", provider="openai")]
    assert response["result"] == {
        "models": [
            {
                "id": "legacy",
                "provider": "legacy-provider",
                "displayName": "Legacy",
                "supportsTools": False,
                "metadata": {"tier": "old"},
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
    turn = store.create_turn(thread_id=active_1.thread_id, input=[])
    for index in range(3):
        store.append_item(
            ThreadItem(
                item_id=f"item_evt_{index}",
                thread_id=active_1.thread_id,
                turn_id=turn.turn_id,
                item_type="agentMessage",
                status="completed",
            ),
            run_event_id=f"evt_{index}",
        )

    processor.process_message(
        "conn_1",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "thread/read",
            "params": {"threadId": active_1.thread_id, "afterItemId": "item_evt_0"},
        },
    )
    assert [item["itemId"] for item in _response(transport, 1)["result"]["items"]] == ["item_evt_1", "item_evt_2"]

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


def test_turn_metadata_is_per_turn_and_does_not_mutate_host_config(monkeypatch: pytest.MonkeyPatch) -> None:
    host = _ContractHost()
    store = ThreadStore()
    state = ThreadStateManager()
    thread = store.create_thread(agent_key="default", metadata={"thread": "base", "shared": "thread"})
    captured_configs: list[RunConfig] = []

    def fake_start(
        cls: type[Runner],
        agent: Agent,
        input: str,
        *,
        run_config: RunConfig | None = None,
    ) -> object:
        del cls, agent, input
        assert run_config is not None
        captured_configs.append(run_config)
        return object()

    monkeypatch.setattr(Runner, "start", classmethod(fake_start))
    monkeypatch.setattr(RunAdapter, "_pump_events", lambda self, connection_id, started: None)
    processor, transport, _store, _state = _initialized_processor(host=host, store=store, state_manager=state)

    processor.process_message(
        "conn_1",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "turn/start",
            "params": {
                "threadId": thread.thread_id,
                "metadata": {
                    "shared": "turn",
                    "turnOnly": 1,
                    "thread_id": "spoofed-thread",
                    "turn_id": "spoofed-turn",
                    "session_id": "spoofed-session",
                },
            },
        },
    )
    first = _response(transport, 1)["result"]
    state.clear_active_turn(thread.thread_id, first["turnId"])
    processor.process_message(
        "conn_1",
        {"jsonrpc": "2.0", "id": 2, "method": "turn/start", "params": {"threadId": thread.thread_id, "metadata": {"second": 2}}},
    )
    _response(transport, 2)

    assert host.config_requests[0].metadata == {
        "thread": "base",
        "shared": "turn",
        "turnOnly": 1,
        "thread_id": "spoofed-thread",
        "turn_id": "spoofed-turn",
        "session_id": "spoofed-session",
    }
    assert host.config_requests[1].metadata == {"thread": "base", "shared": "thread", "second": 2}
    assert host.agent_requests[0].metadata == host.config_requests[0].metadata
    assert host.agent_requests[1].metadata == host.config_requests[1].metadata
    assert captured_configs[0].metadata == {
        "host": "base",
        "thread": "base",
        "shared": "turn",
        "turnOnly": 1,
        "thread_id": thread.thread_id,
        "turn_id": first["turnId"],
        "session_id": thread.thread_id,
    }
    assert "turnOnly" not in captured_configs[1].metadata
    assert host.base_config.metadata == {"host": "base", "shared": "host"}


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
    assert set(definitions["CheckpointSummary"]["properties"]) == set(
        durable_resume["checkpointSummary"]["fields"]
    )
    assert set(definitions["InterruptionSummary"]["properties"]) == set(
        durable_resume["interruptionSummary"]["fields"]
    )
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
