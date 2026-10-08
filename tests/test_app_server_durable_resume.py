from __future__ import annotations

import json
import queue
from pathlib import Path
from threading import Event
from typing import Any, cast

import pytest
from support import FixedModelProvider

from vv_agent import Agent, CheckpointConfig, RunConfig, ToolContext, function_tool
from vv_agent.app_server import (
    AppServer,
    AppServerErrorCode,
    ChannelTransport,
    DefaultAppServerHost,
    TurnResumeParams,
)
from vv_agent.app_server.item_mapper import map_run_event
from vv_agent.app_server.run_adapter import StartedTurn
from vv_agent.checkpoint import AmbiguousToolPolicy, ResumePolicy
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
        ambiguous_tool_policy=AmbiguousToolPolicy.REQUIRE_RECONCILIATION,
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

    expected = next(
        case for case in _contract()["durableResume"]["protocolCases"] if case["name"] == "terminal_replay_is_response_only"
    )
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

    expected = next(
        case for case in _contract()["durableResume"]["protocolCases"] if case["name"] == "live_claim_keeps_existing_owner"
    )
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


def test_deferred_pending_producer_projects_interrupted_without_completion_or_error() -> None:
    store = InMemoryCheckpointStore()

    @function_tool(name="defer_remote", tool_metadata={"idempotency": "supported"})
    def defer_remote(context: ToolContext):
        return context.defer()

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="call-deferred", name="defer_remote", arguments={})],
            )
        ]
    )
    server, transport = _server(
        store=store,
        agent=Agent(
            name="assistant",
            instructions="Wait for the remote operation.",
            model="test-model",
            tools=[defer_remote],
        ),
        model_provider=FixedModelProvider(llm, _resolved_model()),
    )

    thread_id, turn_id, messages = _start_thread_and_turn(server, transport)
    response = next(message for message in messages if message.get("id") == 2)
    completed = next(message for message in messages if message.get("method") == "turn/completed")

    assert response["result"]["status"] == "running"
    assert completed["params"]["status"] == "interrupted"
    assert completed["params"]["waitReason"] == "deferred_pending"
    assert "completionReason" not in completed["params"]
    assert "error" not in completed["params"]
    assert completed["params"]["threadId"] == thread_id
    assert completed["params"]["turnId"] == turn_id
    checkpoint = store.load_checkpoint(CHECKPOINT_KEY)
    assert checkpoint is not None
    assert checkpoint.status.value == "deferred"
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


class _KernelTestTransport(ChannelTransport):
    def __init__(self, *, connection_id: str):
        super().__init__(connection_id=connection_id)
        self.approval_requested = Event()
        self.turn_completed = Event()

    def write_outbound(self, payload: dict[str, Any]) -> None:
        super().write_outbound(payload)
        if payload.get("method") == "approval/request":
            self.approval_requested.set()
        elif payload.get("method") == "turn/completed":
            self.turn_completed.set()


def _kernel_server(path: Path, cut: str | None = None):
    import os

    from vv_agent.llm.base import LlmRequest
    from vv_agent.session.surfaces import _SessionKernel

    calls_path = path.with_suffix(".calls")

    def model(request: LlmRequest):
        calls = calls_path.read_text().splitlines() if calls_path.exists() else []
        with calls_path.open("a") as f:
            f.write("model\n")
        if calls.count("model") == 0:
            return LLMResponse("work", [ToolCall("work", "work", {})])
        assert any(m.image_url == "https://example.invalid/image.png" for m in request.messages)
        return LLMResponse("done")

    @function_tool(needs_approval=path.name.startswith("approval"))
    def work():
        with calls_path.open("a") as f:
            f.write("tool\n")
        return "worked"

    kernel = _SessionKernel(path)
    original_runtime = kernel.runtime

    def runtime(*args, **kwargs):
        value = original_runtime(*args, **kwargs)
        value.ttl_ms = 60_000

        def hook(point, record):
            if (
                point == "after_commit"
                and record is not None
                and (
                    (cut == "model" and record.kind == "op_planned" and record._payload["op_kind"] != "model")
                    or (cut == "tool" and record.kind == "op_completed" and record._payload["result"].get("content") == "worked")
                )
            ):
                snapshot = server.store.read_thread("thread_1")
                path.with_suffix(".cursor").write_text(snapshot.items[-1].item_id)
                os._exit(91)

        value.hook = hook
        return value

    cast(Any, kernel).runtime = runtime
    transport = _KernelTestTransport(connection_id="owner")
    server = AppServer(
        transport=transport,
        _kernel=kernel,
        host=DefaultAppServerHost(
            agent=Agent("assistant", "Work.", model="test-model", tools=[work]),
            run_config=RunConfig(model_provider=FixedModelProvider(ScriptedLLM([model, model]), _resolved_model()), max_cycles=2),
        ),
    )
    server.processor.process_message(
        "owner", {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"clientInfo": {"name": "restart"}}}
    )
    transport.receive_outbound()
    server.processor.process_message("owner", {"jsonrpc": "2.0", "method": "initialized"})
    return kernel, server, transport


def _kernel_process(path: str, cut: str):
    import os

    kernel, server, transport = _kernel_server(Path(path), cut)
    server.processor.process_message("owner", {"jsonrpc": "2.0", "id": 1, "method": "thread/start"})
    server.processor.process_message(
        "owner",
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "turn/start",
            "params": {
                "threadId": "thread_1",
                "input": _contract()["input"]["valid"],
            },
        },
    )
    if cut == "approval":
        transport.approval_requested.wait()
        while (message := transport.receive_outbound()).get("method") != "approval/request":
            pass
        Path(path).with_suffix(".approval").write_text(json.dumps(message))
        os._exit(91)
    kernel.handles[0].result()
    raise AssertionError(f"Process did not reach the {cut} crash boundary")


@pytest.mark.parametrize("cut", ["model", "tool", "approval"])
def test_kernel_process_restart_retains_calls_approval_image_and_client_cursor(tmp_path: Path, cut: str):
    import os
    import subprocess
    import sys

    path = tmp_path / f"{'approval' if cut == 'approval' else 'turn'}.sqlite"
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from test_app_server_durable_resume import _kernel_process; import sys; _kernel_process(*sys.argv[1:])",
            str(path),
            cut,
        ],
        env=os.environ | {"PYTHONPATH": f"{Path(__file__).parent}:{Path(__file__).parents[1] / 'src'}"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 91, child.stderr
    kernel, server, transport = _kernel_server(path)
    try:
        # The writer process is dead; expire its retained lease without racing the clock.
        with kernel.store.atomic():
            epoch, owner, expiry = kernel.store.connection.execute(
                "SELECT lease_epoch,lease_owner,lease_until_ms FROM sk_session WHERE session_id='thread_1'"
            ).fetchone()
            assert owner is not None and expiry > kernel.store._now()
            kernel.store.connection.execute("UPDATE sk_session SET lease_until_ms=0 WHERE session_id='thread_1'")
        before = server.store.read_thread("thread_1")
        assert len(before.turns) == 1
        assert before.turns[0].input == _contract()["input"]["valid"]
        cursor_path = path.with_suffix(".cursor")
        cursor = cursor_path.read_text() if cursor_path.exists() else None
        expected = [item.to_dict() for item in before.items]
        if cursor:
            marker = next(i for i, item in enumerate(before.items) if item.item_id == cursor)
            expected = expected[marker + 1 :]
        approval_request = None
        if cut == "approval":
            observer = _KernelTestTransport(connection_id="observer")
            server.router.register_transport(observer)
            server.processor.process_message(
                "observer", {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"clientInfo": {"name": "observer"}}}
            )
            assert observer.receive_outbound()["id"] == 0
            server.processor.process_message("observer", {"jsonrpc": "2.0", "method": "initialized"})
            server.processor.process_message(
                "observer", {"jsonrpc": "2.0", "id": 1, "method": "thread/resume", "params": {"threadId": "thread_1"}}
            )
            assert observer.receive_outbound()["id"] == 1
            assert transport.approval_requested.wait(timeout=120)
            while (approval_request := transport.receive_outbound()).get("method") != "approval/request":
                pass
            server.processor.process_message(
                "observer",
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "approval/resolve",
                    "params": {
                        "threadId": "thread_1",
                        "turnId": before.turns[0].turn_id,
                        "requestId": approval_request["params"]["requestId"],
                        "decision": "allow",
                    },
                },
            )
            observer_messages = []
            while (message := observer.receive_outbound()).get("id") != 2:
                observer_messages.append(message)
            assert message["error"]["code"] == AppServerErrorCode.INVALID_PARAMS
            assert not any(m.get("method") == "approval/request" for m in observer_messages)
            assert path.with_suffix(".calls").read_text().splitlines() == ["model"]
        server.processor.process_message(
            "owner",
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "thread/resume",
                "params": {
                    "threadId": "thread_1",
                    **({"afterItemId": cursor} if cursor else {}),
                },
            },
        )
        replay = transport.receive_outbound()
        assert replay["id"] == 3 and replay["result"]["items"] == expected
        if cut == "approval":
            original = json.loads(path.with_suffix(".approval").read_text())
            request = approval_request
            assert request is not None
            assert request["id"] == original["id"]
            params = request["params"]
            server.processor.process_message(
                "owner",
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "approval/resolve",
                    "params": {
                        "threadId": params["threadId"],
                        "turnId": params["turnId"],
                        "requestId": params["requestId"],
                        "decision": "allow",
                    },
                },
            )
        assert transport.turn_completed.wait(timeout=120)
        while (completed := transport.receive_outbound()).get("method") != "turn/completed":
            pass
        assert completed["params"]["status"] == "completed"
        assert completed["params"]["finalOutput"] == "done"
        server.run_adapter.join()
        assert path.with_suffix(".calls").read_text().splitlines() == ["model", "tool", "model"]
        after = server.store.read_thread("thread_1")
        assert len(after.turns) == 1 and after.turns[0].turn_id == before.turns[0].turn_id
        assert after.turns[0].input == _contract()["input"]["valid"]
        assert after.turns[0].result["tokenUsage"] == completed["params"]["tokenUsage"]
        state, records, _ = kernel.store.read_state("thread_1")
        assert records[-1].writer_epoch > epoch
        assert sum(r.record.kind == "turn_ended" for r in records) == 1
        cursor_seq = kernel.store.connection.execute(
            "SELECT last_seq FROM sk_consumer WHERE session_id='thread_1' AND consumer='app_server'"
        ).fetchone()[0]
        assert cursor_seq == records[-1].seq
        assert state.active_turn_id is None
        tables = {row[0] for row in kernel.store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == {"sk_session", "sk_record", "sk_inbox", "sk_consumer", "sk_commit"}
    finally:
        server.router.cancel_matching_server_requests()
        server.run_adapter.join()
        kernel.close()


def _inject_lease_loss(monkeypatch: pytest.MonkeyPatch, cut: str, loss: str, *, takeover: bool = False):
    from vv_agent.session.kernel import _Driver
    from vv_agent.session.records import InboxItem

    original_step = _Driver.step
    leases = []

    def step(driver):
        before = len(driver.records)
        advanced = original_step(driver)
        matches = any(
            (cut == "admission" and record.kind == "turn_started")
            or (cut == "model" and record.kind == "op_completed" and "tool_calls" in record._payload["result"])
            or (cut == "tool" and record.kind == "op_completed" and record._payload["result"].get("content") == "worked")
            or (cut == "terminal" and record.kind == "turn_ended")
            for record in (stored.record for stored in driver.records[before:])
        )
        if matches and not leases:
            # Stop renewal before expiring the lease; no scheduler delay is needed.
            driver.scope.stop.set()
            driver.scope.thread.join()
            leases.append(driver.scope.lease)
            with driver.store.atomic():
                driver.store._rows("UPDATE sk_session SET lease_until_ms=0 WHERE session_id=%s", (driver.sid,))
            if cut == "terminal":
                turn = driver.state.turns[driver.records[-1].record.turn_id]
                initial = driver.state.applied_inputs[turn.start._payload["input_ids"][0]].payload["input"]
                with driver.store.atomic() as tx:
                    tx.push(driver.sid, InboxItem("next-turn", "user", initial["payload"]))
            if takeover:
                leases.append(driver.store.acquire(driver.sid, owner="other-writer", ttl_ms=60_000))
            if loss == "heartbeat":
                driver.scope.lost = True
                driver.scope.token.cancel("lease_lost")
        return advanced

    monkeypatch.setattr(_Driver, "step", step)
    return leases


@pytest.mark.parametrize("surface_name", ["app_server", "approval", "interactive"])
@pytest.mark.parametrize("cut", ["admission", "model", "tool", "terminal"])
@pytest.mark.parametrize("loss", ["heartbeat", "write"])
def test_kernel_surface_reacquires_lost_lease_without_repeating_committed_work(monkeypatch, tmp_path, surface_name, cut, loss):
    from vv_agent import AgentSessionOptions, AgentStatus, InteractiveAgentClient
    from vv_agent.session.store import LeaseLost
    from vv_agent.session.surfaces import _SessionKernel

    leases = _inject_lease_loss(monkeypatch, cut, loss)
    calls = []

    def model(_request):
        calls.append("model")
        return LLMResponse("work", [ToolCall("work", "work", {})]) if calls == ["model"] else LLMResponse("done")

    @function_tool(needs_approval=surface_name == "approval")
    def work():
        calls.append("tool")
        return "worked"

    kernel = _SessionKernel(tmp_path / "lease.sqlite")
    agent = Agent("assistant", "Work.", model="test-model", tools=[work])
    provider = FixedModelProvider(ScriptedLLM([model, model]), _resolved_model())
    server = None
    try:
        if surface_name != "interactive":
            transport = _KernelTestTransport(connection_id="conn_1")
            server = AppServer(
                transport=transport,
                _kernel=kernel,
                host=DefaultAppServerHost(agent=agent, run_config=RunConfig(model_provider=provider, max_cycles=2)),
            )
            _send(server, {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"clientInfo": {"name": "lease"}}})
            _send(server, {"jsonrpc": "2.0", "id": 1, "method": "thread/start"})
            _send(
                server,
                {"jsonrpc": "2.0", "id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": TURN_INPUT}},
            )
            if surface_name == "approval":
                assert transport.approval_requested.wait(timeout=30)
                while (request := transport.receive_outbound()).get("method") != "approval/request":
                    pass
                params = request["params"]
                _send(
                    server,
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "approval/resolve",
                        "params": {
                            "threadId": params["threadId"],
                            "turnId": params["turnId"],
                            "requestId": params["requestId"],
                            "decision": "allow",
                        },
                    },
                )
            result = kernel.handles[0].result(timeout=30)
            cast(Any, server.run_adapter).join()
            assert transport.turn_completed.is_set()
            completed = _drain_until_completed(transport)[-1]
            assert completed["params"]["status"] == "completed"
            assert completed["params"]["finalOutput"] == "done"
        else:
            client = InteractiveAgentClient(
                options=AgentSessionOptions(model_provider=provider, workspace=tmp_path), _kernel=kernel
            )
            session = client.create_session(agent=agent, session_id="interactive-lease")
            result = session.prompt("go", auto_follow_up=False)
        assert result.status is AgentStatus.COMPLETED and result.final_output == "done"
        assert calls == ["model", "tool", "model"]
        assert len(leases) == 1
        sid = kernel.handles[0].session_id
        state, records, _ = kernel.store.read_state(sid)
        assert len(state.turns) == 1 and state.active_turn_id is None
        assert sum(r.record.kind == "turn_ended" for r in records) == 1
        assert sum(r.record.kind == "op_started" for r in records) == 3
        assert records[-1].writer_epoch > leases[0].epoch if cut != "terminal" else records[-1].writer_epoch == leases[0].epoch
        if cut == "terminal":
            assert [pending.item.input_id for pending in kernel.store.peek_inbox(sid)] == ["next-turn"]
        with pytest.raises(LeaseLost), kernel.store.atomic() as tx:
            tx.append(sid, lease=leases[0], expected_seq=records[-1].seq, commit_id="stale-writer", records=())
    finally:
        if server is not None:
            server.router.cancel_matching_server_requests()
            cast(Any, server.run_adapter).join()
        kernel.close()


@pytest.mark.parametrize("loss", ["heartbeat", "write"])
def test_kernel_surface_waits_for_other_writer_after_lease_loss(monkeypatch, tmp_path, loss):
    from vv_agent.session.store import LeaseLost

    leases = _inject_lease_loss(monkeypatch, "model", loss, takeover=True)
    kernel, server, transport = _kernel_server(tmp_path / "turn.sqlite")
    blocked = Event()
    original_acquire = kernel.store.acquire

    def acquire(*args, **kwargs):
        lease = original_acquire(*args, **kwargs)
        if lease is None:
            blocked.set()
        return lease

    monkeypatch.setattr(kernel.store, "acquire", acquire)
    try:
        server.processor.process_message("owner", {"jsonrpc": "2.0", "id": 1, "method": "thread/start"})
        server.processor.process_message(
            "owner",
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "turn/start",
                "params": {"threadId": "thread_1", "input": _contract()["input"]["valid"]},
            },
        )
        assert blocked.wait(timeout=30)
        assert len(leases) == 2 and leases[1].epoch > leases[0].epoch
        assert kernel.handles[0]._thread.is_alive()
        assert not transport.turn_completed.is_set()
        assert (tmp_path / "turn.calls").read_text().splitlines() == ["model"]
        with pytest.raises(LeaseLost):
            kernel.store.renew(leases[0], ttl_ms=60_000)
        kernel.store.release(leases[1])
        result = kernel.handles[0].result(timeout=30)
        server.run_adapter.join()
        assert result.final_output == "done" and transport.turn_completed.is_set()
        assert (tmp_path / "turn.calls").read_text().splitlines() == ["model", "tool", "model"]
        records = kernel.store.read_state("thread_1")[1]
        assert records[-1].writer_epoch > leases[1].epoch
    finally:
        if len(leases) == 2:
            kernel.store.release(leases[1])
        server.router.cancel_matching_server_requests()
        server.run_adapter.join()
        kernel.close()
