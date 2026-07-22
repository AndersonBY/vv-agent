from __future__ import annotations

import json
import queue
from pathlib import Path
from typing import Any, cast

import pytest
from support import FixedModelProvider

from vv_agent import Agent, CheckpointConfig, RunConfig, function_tool
from vv_agent.app_server import (
    AppServer,
    AppServerErrorCode,
    ChannelTransport,
    DefaultAppServerHost,
    TurnResumeParams,
)
from vv_agent.app_server.item_mapper import map_run_event
from vv_agent.app_server.run_adapter import StartedTurn
from vv_agent.checkpoint import ResumePolicy
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.events import ToolCallCompletedEvent
from vv_agent.llm import ScriptedLLM
from vv_agent.model import ModelProvider
from vv_agent.run_handle import RunHandle
from vv_agent.runtime.stores.memory import InMemoryCheckpointStore
from vv_agent.types import LLMResponse, ToolCall

CHECKPOINT_KEY = "tenant-7/run-42"
TURN_INPUT = [{"type": "text", "text": "hello"}]


def _contract() -> dict[str, Any]:
    fixture = Path(__file__).parent / "fixtures" / "parity" / "app_server_observable.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


def _resolved_model() -> ResolvedModelConfig:
    endpoint = EndpointConfig(
        endpoint_id="test",
        api_key="test-key",
        api_base="https://example.invalid/v1",
    )
    return ResolvedModelConfig(
        backend="test",
        requested_model="test-model",
        selected_model="test-model",
        model_id="test-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
        function_call_available=True,
    )


def _checkpoint_config(store: InMemoryCheckpointStore) -> CheckpointConfig:
    capability_names = (
        "approval_provider",
        "before_cycle_messages",
        "behavior_affecting_run_metadata",
    )
    return CheckpointConfig(
        store=store,
        key=CHECKPOINT_KEY,
        resume_policy=ResumePolicy.NEW,
        capability_refs={name: {"id": f"app-server.{name}", "version": "1"} for name in capability_names},
    )


def _server(
    *,
    store: InMemoryCheckpointStore,
    agent: Agent,
    model_provider: ModelProvider,
) -> tuple[AppServer, ChannelTransport]:
    transport = ChannelTransport(connection_id="conn_1")
    host = DefaultAppServerHost(
        agent=agent,
        run_config=RunConfig(
            model_provider=model_provider,
            max_cycles=1,
            no_tool_policy="finish",
            checkpoint_config=_checkpoint_config(store),
        ),
    )
    return AppServer(transport=transport, host=host), transport


def _send(server: AppServer, payload: dict[str, Any]) -> None:
    server.processor.process_message("conn_1", payload)


def _start_thread_and_turn(
    server: AppServer,
    transport: ChannelTransport,
) -> tuple[str, str, list[dict[str, Any]]]:
    _send(
        server,
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"clientInfo": {"name": "durable-resume-test"}},
        },
    )
    assert transport.receive_outbound(timeout=1)["id"] == 0
    _send(server, {"jsonrpc": "2.0", "id": 1, "method": "thread/start", "params": {}})
    thread_response = transport.receive_outbound(timeout=1)
    assert transport.receive_outbound(timeout=1)["method"] == "thread/started"
    thread_id = str(thread_response["result"]["threadId"])
    _send(
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "turn/start",
            "params": {"threadId": thread_id, "input": TURN_INPUT},
        },
    )
    messages = _drain_until_completed(transport)
    response = next(message for message in messages if message.get("id") == 2)
    return thread_id, str(response["result"]["turnId"]), messages


def _resume_request(*, request_id: int, thread_id: str, turn_id: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "turn/resume",
        "params": TurnResumeParams(
            thread_id=thread_id,
            turn_id=turn_id,
            checkpoint_key=CHECKPOINT_KEY,
        ).to_dict(),
    }


def _drain_until_completed(transport: ChannelTransport) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    while True:
        message = transport.receive_outbound(timeout=5)
        messages.append(message)
        if message.get("method") == "turn/completed":
            return messages


def _assert_no_outbound(transport: ChannelTransport) -> None:
    with pytest.raises(queue.Empty):
        transport.receive_outbound(timeout=0.05)


def _assert_safe_projection(payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    for field in _contract()["durableResume"]["sensitiveFieldsNeverProjected"]:
        assert field not in serialized
    for field in ("operationReceipt", "toolReceipt", "extension_state", "idempotency_key"):
        assert field not in serialized
    assert "test-key" not in serialized
    assert "secret-tool-argument" not in serialized


def test_turn_resume_rejects_new_input_and_foreign_turn() -> None:
    store = InMemoryCheckpointStore()
    llm = ScriptedLLM(steps=[LLMResponse(content="done")])

    server, transport = _server(
        store=store,
        agent=Agent(name="assistant", instructions="Answer.", model="test-model"),
        model_provider=FixedModelProvider(llm, _resolved_model()),
    )
    thread_id, turn_id, _messages = _start_thread_and_turn(server, transport)

    request = _resume_request(request_id=3, thread_id=thread_id, turn_id=turn_id)
    request["params"]["input"] = [{"type": "text", "text": "new input"}]
    _send(server, request)
    assert transport.receive_outbound(timeout=1)["error"]["code"] == AppServerErrorCode.INVALID_PARAMS

    foreign_turn = server.processor._store.create_turn(thread_id=thread_id, input=TURN_INPUT)
    _send(server, _resume_request(request_id=4, thread_id=thread_id, turn_id=foreign_turn.turn_id))
    response = transport.receive_outbound(timeout=1)
    assert response["error"]["code"] == AppServerErrorCode.INVALID_PARAMS
    assert response["error"]["message"] == "Checkpoint is not bound to the requested turn"


def test_terminal_checkpoint_replay_is_response_only_on_original_turn() -> None:
    store = InMemoryCheckpointStore()
    model_calls = 0

    def complete(_request: Any) -> LLMResponse:
        nonlocal model_calls
        model_calls += 1
        return LLMResponse(content="done")

    llm = ScriptedLLM(steps=[complete])

    server, transport = _server(
        store=store,
        agent=Agent(name="assistant", instructions="Answer.", model="test-model"),
        model_provider=FixedModelProvider(llm, _resolved_model()),
    )
    thread_id, turn_id, _messages = _start_thread_and_turn(server, transport)
    completed_at = server.processor._store.read_thread(thread_id).turns[0].completed_at
    _send(server, _resume_request(request_id=3, thread_id=thread_id, turn_id=turn_id))
    response = transport.receive_outbound(timeout=5)

    expected = _contract()["durableResume"]["protocolCases"][2]
    assert expected["name"] == "terminal_replay_is_response_only"
    assert response["id"] == 3
    assert response["result"]["threadId"] == thread_id
    assert response["result"]["turnId"] == turn_id
    assert response["result"]["status"] == expected["response"]["result"]["status"]
    assert response["result"]["finalOutput"] == expected["response"]["result"]["finalOutput"]
    assert response["result"]["completionReason"] == expected["response"]["result"]["completionReason"]
    assert set(response["result"]["checkpoint"]) == set(_contract()["durableResume"]["checkpointSummary"]["fields"])
    assert response["result"]["checkpoint"]["terminalAcknowledged"] is True
    assert model_calls == 1
    snapshot = server.processor._store.read_thread(thread_id)
    assert len(snapshot.turns) == 1
    assert snapshot.turns[0].turn_id == turn_id
    assert snapshot.turns[0].input == TURN_INPUT
    assert snapshot.turns[0].completed_at == completed_at
    _assert_safe_projection(response)
    _assert_no_outbound(transport)


def test_live_claim_returns_existing_owner_without_notifications_or_execution() -> None:
    server, transport, store, effects = _crashing_tool_server()
    thread_id, turn_id, _messages = _start_thread_and_turn(server, transport)
    checkpoint = store.load_checkpoint(CHECKPOINT_KEY)
    assert checkpoint is not None
    assert checkpoint.claim_token is not None
    assert checkpoint.lease_expires_at_ms is not None

    before_effects = len(effects)
    _send(server, _resume_request(request_id=3, thread_id=thread_id, turn_id=turn_id))
    response = transport.receive_outbound(timeout=1)

    expected = _contract()["durableResume"]["protocolCases"][1]
    assert expected["name"] == "live_claim_keeps_existing_owner"
    assert response["result"]["runId"] == checkpoint.root_run_id
    assert response["result"]["status"] == "running"
    assert response["result"]["checkpoint"] == {
        "key": CHECKPOINT_KEY,
        "resumeAttempt": checkpoint.resume_attempt,
        "cycleIndex": checkpoint.cycle_index,
        "status": "running",
        "terminalAcknowledged": False,
    }
    assert len(effects) == before_effects
    snapshot = server.processor._store.read_thread(thread_id)
    assert len(snapshot.turns) == 1
    assert snapshot.turns[0].turn_id == turn_id
    assert snapshot.turns[0].input == TURN_INPUT
    _assert_safe_projection(response)
    _assert_no_outbound(transport)


def test_active_owner_does_not_predict_unpersisted_checkpoint_progress() -> None:
    store = InMemoryCheckpointStore()
    model_calls = 0

    def complete(_request: Any) -> LLMResponse:
        nonlocal model_calls
        model_calls += 1
        return LLMResponse(content="done")

    llm = ScriptedLLM(steps=[complete])

    server, transport = _server(
        store=store,
        agent=Agent(name="assistant", instructions="Answer.", model="test-model"),
        model_provider=FixedModelProvider(llm, _resolved_model()),
    )
    thread_id, turn_id, _messages = _start_thread_and_turn(server, transport)
    checkpoint = store.load_checkpoint(CHECKPOINT_KEY)
    assert checkpoint is not None
    assert checkpoint.terminal_result is not None
    resume_attempt = checkpoint.resume_attempt
    server.processor._state_manager.set_active_turn(
        thread_id=thread_id,
        turn_id=turn_id,
        handle=object(),
        checkpoint_key=CHECKPOINT_KEY,
        run_id=checkpoint.root_run_id,
    )

    _send(server, _resume_request(request_id=3, thread_id=thread_id, turn_id=turn_id))
    response = transport.receive_outbound(timeout=1)

    assert response["result"] == {
        "threadId": thread_id,
        "turnId": turn_id,
        "runId": checkpoint.root_run_id,
        "status": "running",
    }
    retained = store.load_checkpoint(CHECKPOINT_KEY)
    assert retained is not None
    assert retained.resume_attempt == resume_attempt
    assert model_calls == 1
    _assert_safe_projection(response)
    _assert_no_outbound(transport)


def test_replayed_durable_item_is_not_rebroadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryCheckpointStore()
    llm = ScriptedLLM(steps=[LLMResponse(content="done")])

    server, transport = _server(
        store=store,
        agent=Agent(name="assistant", instructions="Answer.", model="test-model"),
        model_provider=FixedModelProvider(llm, _resolved_model()),
    )
    thread_id, turn_id, _messages = _start_thread_and_turn(server, transport)
    snapshot = server.processor._store.read_thread(thread_id)
    turn = next(turn for turn in snapshot.turns if turn.turn_id == turn_id)
    event = ToolCallCompletedEvent(
        run_id="run-durable-replay",
        trace_id="trace-durable-replay",
        tool_name="write_once",
        tool_call_id="call-1",
        status="success",
        directive="continue",
        error_code=None,
        execution_started=True,
        duration_ms=1,
        event_id="evt-durable-replay",
        created_at=1,
    )
    projection = map_run_event(event, thread_id=thread_id, turn_id=turn_id)
    assert projection.item is not None
    assert server.processor._store.append_item(
        projection.item,
        run_event_id=event.event_id,
    )

    class ReplayHandle:
        def events(self):
            yield event

        def result(self, timeout: float | None = None):
            del timeout
            return None

    adapter = server.processor._run_adapter
    monkeypatch.setattr(adapter, "_complete_turn", lambda *_args, **_kwargs: None)
    adapter._pump_events(
        "conn_1",
        StartedTurn(
            thread=snapshot.thread,
            turn=turn,
            handle=cast(RunHandle, ReplayHandle()),
            is_durable_resume=True,
        ),
    )

    retained = server.processor._store.read_thread(thread_id)
    assert [item.item_id for item in retained.items].count(projection.item.item_id) == 1
    _assert_no_outbound(transport)


def test_reconciliation_resume_emits_canonical_interrupted_sequence() -> None:
    server, transport, store, effects = _crashing_tool_server()
    thread_id, turn_id, _messages = _start_thread_and_turn(server, transport)
    with store._lock:
        store._store[CHECKPOINT_KEY].lease_expires_at_ms = 1

    _send(server, _resume_request(request_id=3, thread_id=thread_id, turn_id=turn_id))
    messages = _drain_until_completed(transport)
    response, *notifications = messages
    labels = [
        (
            f"{message['method']}:{message['params']['status']}"
            if message["method"] == "thread/status/changed"
            else (f"turn/completed:{message['params']['status']}" if message["method"] == "turn/completed" else message["method"])
        )
        for message in notifications
    ]
    expected = _contract()["durableResume"]["protocolCases"][0]

    assert expected["name"] == "resume_reaches_reconciliation_interruption"
    assert response["result"] == {
        "threadId": thread_id,
        "turnId": turn_id,
        "runId": response["result"]["runId"],
        "status": "running",
    }
    assert labels == expected["notificationOrder"]
    completed = notifications[-1]["params"]
    assert completed["status"] == "interrupted"
    assert "completionReason" not in completed
    assert "error" not in completed
    assert "tokenUsage" not in completed
    assert set(completed["checkpoint"]) == set(_contract()["durableResume"]["checkpointSummary"]["fields"])
    assert set(completed["interruption"]) == set(_contract()["durableResume"]["interruptionSummary"]["fields"])
    assert completed["checkpoint"]["status"] == "reconciliation_required"
    assert completed["interruption"]["reason"] == "resume_requires_reconciliation"
    assert completed["interruption"]["idempotencySupport"] == "unknown"
    assert len(effects) == 1
    snapshot = server.processor._store.read_thread(thread_id)
    assert len(snapshot.turns) == 1
    assert snapshot.turns[0].turn_id == turn_id
    assert snapshot.turns[0].input == TURN_INPUT
    _assert_safe_projection({"messages": messages})


def _crashing_tool_server() -> tuple[AppServer, ChannelTransport, InMemoryCheckpointStore, list[str]]:
    store = InMemoryCheckpointStore()
    effects: list[str] = []

    @function_tool(name="unsafe_write", tool_metadata={"idempotency": "unknown"})
    def unsafe_write(value: str) -> str:
        effects.append(value)
        raise SystemExit("simulated process crash")

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-unsafe-1",
                        name="unsafe_write",
                        arguments={"value": "secret-tool-argument"},
                    )
                ],
            )
        ]
    )

    server, transport = _server(
        store=store,
        agent=Agent(
            name="assistant",
            instructions="Write once.",
            model="test-model",
            tools=[unsafe_write],
        ),
        model_provider=FixedModelProvider(llm, _resolved_model()),
    )
    return server, transport, store, effects
