from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from support import FixedModelProvider

from vv_agent import Agent, RunConfig, ScriptedModelProvider, function_tool
from vv_agent.app_server import AppServer, ChannelTransport, MessageProcessor, OutgoingRouter, RequestId
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.approval import ApprovalDecision
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.events import ApprovalRequestedEvent
from vv_agent.llm import ScriptedLLM
from vv_agent.model import resolve_model_selection
from vv_agent.runtime.hooks import BaseRuntimeHook, BeforeToolCallPatch, RuntimeHook
from vv_agent.session.app_server import _JoinDriveState, _KernelRunAdapter
from vv_agent.session.reducer import ExecutionState
from vv_agent.types import LLMResponse, ToolCall

pytestmark = pytest.mark.usefixtures("surface")


@pytest.fixture(autouse=True)
def _level_two_app_server_storage(surface) -> None:
    # App Server static bootstrap creates feature-level-2 sessions, while the
    # shared disposable fixture starts its SQLite store at write level 1.
    surface._ensure_static_worker_support(level=2, capabilities=())


def _resolved_model(model: str = "test-model") -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id=model)],
    )


def test_app_server_approval_allow_runs_tool_and_completes_turn() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls)

    approval_request = _start_and_wait_for_approval(server, transport)
    approval_params = cast(dict[str, object], approval_request["params"])
    assert approval_request["method"] == "approval/request"
    assert approval_params["toolName"] == "dangerous_tool"
    assert calls == []

    _send(
        transport,
        server,
        {"jsonrpc": "2.0", "id": approval_request["id"], "result": {"decision": "allow", "reason": "ok"}},
    )
    outbound = _drain_until_turn_completed(transport)

    assert calls == ["ran"]
    assert _completed_status(outbound) == "completed"


def test_wake_keeps_committed_progress_in_forward_approval_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls)
    _initialize(server, transport, request_id=0, client_name="owner")
    thread_id = _start_thread(server, transport, request_id=1)
    position = server.run_adapter._notification_positions[thread_id]
    worker = server.kernel._worker
    original_post_observation = worker.post_observation
    monkeypatch.setattr(worker, "post_observation", lambda _callback: None)
    server.wake(thread_id)
    monkeypatch.setattr(worker, "post_observation", original_post_observation)

    assert position.replay_approvals is False

    _send(transport, server, {"jsonrpc": "2.0", "id": 2, "method": "turn/start", "params": _turn_params()})
    outbound = []
    while True:
        message = transport.receive_outbound(timeout=10)
        outbound.append(message)
        if message.get("method") == "approval/request":
            approval = message
            break
    causal_methods = [
        method
        for method in (
            message.get("method")
            for message in outbound
            if message.get("method") in {"turn/started", "item/started", "approval/requested", "approval/request"}
        )
        if method is not None
    ]
    assert causal_methods == ["turn/started", "item/started", "approval/requested", "approval/request"]

    _send(
        transport,
        server,
        {"jsonrpc": "2.0", "id": approval["id"], "result": {"decision": "deny"}},
    )
    _drain_until_turn_completed(transport)
    assert calls == []


def test_app_server_approval_request_includes_tool_arguments() -> None:
    calls: list[dict[str, object]] = []
    arguments: dict[str, object] = {"path": "notes/private.txt", "force": True}
    server, transport = _server_with_parameterized_approval_tool(calls, arguments)

    approval_request = _start_and_wait_for_approval(server, transport)
    approval_params = cast(dict[str, object], approval_request["params"])
    requested = cast(dict[str, object], approval_request["_requested"])

    assert approval_params["arguments"] == arguments
    assert cast(dict[str, object], requested["params"])["arguments"] == arguments
    assert calls == []

    _send(
        transport,
        server,
        {"jsonrpc": "2.0", "id": approval_request["id"], "result": {"decision": "allow"}},
    )
    outbound = _drain_until_turn_completed(transport)

    _send(transport, server, {"jsonrpc": "2.0", "id": 3, "method": "thread/read", "params": {"threadId": "thread_1"}})
    read = _drain_until(transport, lambda message: message.get("id") == 3)
    items = cast(dict[str, object], read["result"])["items"]
    approval_items = [item for item in cast(list[dict[str, object]], items) if item["type"] == "approval"]
    assert cast(dict[str, object], approval_items[0]["payload"])["arguments"] == arguments

    assert calls == [arguments]
    assert _completed_status(outbound) == "completed"


def test_app_server_approval_uses_prepared_arguments_and_bound_digest(surface) -> None:
    calls: list[dict[str, object]] = []
    original: dict[str, object] = {"path": "notes/original.txt", "force": False}
    prepared = {"path": "notes/safe.txt", "force": True}

    class RewriteArguments(BaseRuntimeHook):
        def before_tool_call(self, event):
            return BeforeToolCallPatch(call=ToolCall(event.call.id, event.call.name, prepared))

    server, transport = _server_with_parameterized_approval_tool(calls, original, hooks=[RewriteArguments()])
    approval_request = _start_and_wait_for_approval(server, transport)
    approval_params = cast(dict[str, object], approval_request["params"])
    requested = cast(dict[str, object], approval_request["_requested"])

    assert approval_params["arguments"] == prepared
    assert cast(dict[str, object], requested["params"])["arguments"] == prepared
    assert calls == []

    state, _, _ = surface.store.read_state("thread_1")
    (operation_id, attempt_number), wait = next(iter(state.waits.items()))
    operation = state.operations[operation_id]
    attempt = operation.attempts[attempt_number]
    request_digest = wait["handle"]["request_digest"]
    assert attempt.prepared is not None
    assert request_digest == attempt.prepared._payload["request_digest"]
    assert request_digest == attempt.execution_plan._payload["request_digest"]

    _send(
        transport,
        server,
        {"jsonrpc": "2.0", "id": approval_request["id"], "result": {"decision": "allow"}},
    )
    outbound = _drain_until_turn_completed(transport)
    completed_state, _, _ = surface.store.read_state("thread_1")

    assert calls == [prepared]
    answer = completed_state.operations[operation_id].attempts[attempt_number].approval_answer
    assert answer is not None
    assert answer._payload["input"]["payload"]["request_digest"] == request_digest
    assert _completed_status(outbound) == "completed"

    _send(transport, server, {"jsonrpc": "2.0", "id": 3, "method": "thread/read", "params": {"threadId": "thread_1"}})
    read = _drain_until(transport, lambda message: message.get("id") == 3)
    items = cast(dict[str, object], read["result"])["items"]
    approval = next(
        item for item in cast(list[dict[str, object]], items) if item["type"] == "approval" and item["status"] == "started"
    )
    assert cast(dict[str, object], approval["payload"])["arguments"] == prepared


def test_app_server_approval_fails_closed_without_trusted_execution_plan() -> None:
    adapter = object.__new__(_KernelRunAdapter)
    event = ApprovalRequestedEvent(
        run_id="thread_1/turn/turn_1",
        trace_id="thread_1/turn/turn_1",
        tool_name="dangerous_tool",
        tool_call_id="call_1",
        message="Approval required",
        request_id="approval_1",
    )

    with pytest.raises(ValueError, match="trusted approval execution plan is unavailable"):
        adapter._approval_arguments(
            "thread_1",
            object(),
            ExecutionState(),
            (),
            event,
            {"operation_id": "missing", "attempt": 1},
        )


def test_explicit_automatic_approval_provider_is_preserved() -> None:
    class AutoApprove:
        def should_request(self, request) -> bool:
            del request
            return True

        def decide(self, request):
            del request
            return ApprovalDecision.allow()

    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls, approval_provider=AutoApprove())
    _initialize(server, transport, request_id=0, client_name="test")
    thread_id = _start_thread(server, transport, request_id=1)
    _send(
        transport,
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "turn/start",
            "params": {
                "inputId": "turn_1",
                "expectedTurnId": None,
                "threadId": thread_id,
                "input": [{"type": "text", "text": "run tool"}],
            },
        },
    )
    outbound = _drain_until_turn_completed(transport)
    assert calls == ["ran"]
    assert not any(message.get("method") == "approval/request" for message in outbound)
    assert server.router.pending_server_request_count() == 0


def test_app_server_approval_allow_session_is_preserved_in_notification() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls)

    approval_request = _start_and_wait_for_approval(server, transport)
    _send(
        transport,
        server,
        {"jsonrpc": "2.0", "id": approval_request["id"], "result": {"decision": "allow_session", "reason": "for this session"}},
    )
    outbound = _drain_until_turn_completed(transport)

    assert calls == ["ran"]
    assert _approval_resolutions(outbound) == ["allow_session"]
    assert _completed_status(outbound) == "completed"


def test_app_server_approval_deny_skips_tool_and_completes_turn() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls)

    approval_request = _start_and_wait_for_approval(server, transport)
    _send(
        transport,
        server,
        {"jsonrpc": "2.0", "id": approval_request["id"], "result": {"decision": "deny", "reason": "no"}},
    )
    outbound = _drain_until_turn_completed(transport)

    assert calls == []
    assert _approval_resolutions(outbound) == ["deny"]
    assert _completed_status(outbound) == "completed"


@pytest.mark.parametrize(
    ("store_now_ms", "answered", "expected_notification"),
    [(999, False, True), (1000, False, False), (999, True, False)],
)
def test_pending_approval_replay_respects_database_clock_and_durable_answer(
    monkeypatch: pytest.MonkeyPatch,
    store_now_ms: int,
    answered: bool,
    expected_notification: bool,
) -> None:
    class Store:
        def read_state(self, session_id: str):
            assert session_id == "thread_1"
            return ExecutionState(), (), 0

        def _now(self) -> int:
            return store_now_ms

        def lease_status(self, session_id: str):
            assert session_id == "thread_1"
            return 1, 0, 0

    class Router:
        def __init__(self) -> None:
            self.sent: list[tuple[str, str, dict[str, object], RequestId]] = []

        def is_registered(self, connection_id: str) -> bool:
            return connection_id == "conn_1"

        def get_pending_server_request(self, request_id: RequestId):
            del request_id
            return None

        def send_server_request(self, connection_id, method, params, *, request_id):
            self.sent.append((connection_id, method, params, request_id))

    class Worker:
        def wake(self, session_id: str) -> None:
            assert session_id == "thread_1"

    class Kernel:
        def __init__(self) -> None:
            self._runtimes: dict[str, object] = {}
            self._worker = Worker()

        def waits(self, session_id: str, *, snapshot, store):
            del snapshot
            assert store._now() == store_now_ms
            return [
                (
                    session_id,
                    None,
                    {
                        "kind": "approval",
                        "turn_id": "turn_1",
                        "request_id": "approval_1",
                        "operation_id": "operation_1",
                        "attempt": 1,
                        "deadline_ms": 1000,
                    },
                )
            ]

    event = ApprovalRequestedEvent(
        run_id="turn_1",
        trace_id="trace_1",
        tool_name="dangerous_tool",
        tool_call_id="call_1",
        message="Review this tool call.",
        request_id="approval_1",
        metadata={"session_seq": 1},
    )
    monkeypatch.setattr("vv_agent.session.app_server.project_records", lambda *_args, **_kwargs: [event])

    store = Store()
    router = Router()
    adapter = object.__new__(_KernelRunAdapter)
    adapter.kernel = Kernel()
    adapter._router = router
    adapter._turn_connections = {("thread_1", "turn_1"): "conn_1"}
    adapter._hydration = lambda *_args, **_kwargs: None
    adapter._approval_answered = lambda *_args: answered
    adapter._approval_arguments = lambda *_args, **_kwargs: {}

    adapter._send_pending_approvals("thread_1", store, after_seq=0, through_seq=1)

    assert bool(router.sent) is expected_notification


def test_pending_approval_without_durable_wait_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    class Store:
        def read_state(self, session_id: str):
            assert session_id == "thread_1"
            return ExecutionState(), (), 0

        def _now(self) -> int:
            return 1

    class Router:
        def __init__(self) -> None:
            self.sent: list[object] = []

        def is_registered(self, connection_id: str) -> bool:
            return connection_id == "conn_1"

        def get_pending_server_request(self, request_id: RequestId):
            del request_id
            return None

        def send_server_request(self, *args, **kwargs):
            self.sent.append((args, kwargs))

    class Worker:
        def wake(self, session_id: str) -> None:
            raise AssertionError(f"unexpected wake for {session_id}")

    class Kernel:
        def __init__(self) -> None:
            self._runtimes: dict[str, object] = {}
            self._worker = Worker()

        def waits(self, session_id: str, *, snapshot, store):
            del snapshot, store
            assert session_id == "thread_1"
            return []

    event = ApprovalRequestedEvent(
        run_id="turn_1",
        trace_id="trace_1",
        tool_name="dangerous_tool",
        tool_call_id="call_1",
        message="Review this tool call.",
        request_id="approval_1",
        metadata={"session_seq": 1},
    )
    monkeypatch.setattr("vv_agent.session.app_server.project_records", lambda *_args, **_kwargs: [event])

    router = Router()
    adapter = object.__new__(_KernelRunAdapter)
    adapter.kernel = Kernel()
    adapter._router = router
    adapter._turn_connections = {("thread_1", "turn_1"): "conn_1"}
    adapter._hydration = lambda *_args, **_kwargs: None
    adapter._approval_arguments = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("approval arguments must not be hydrated without a durable wait")
    )

    adapter._send_pending_approvals("thread_1", Store(), after_seq=0, through_seq=1)

    assert router.sent == []


def test_approval_answered_uses_payload_request_id_for_custom_input_id() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls)
    approval = _start_and_wait_for_approval(server, transport)
    params = cast(dict[str, object], approval["params"])
    thread_id = cast(str, params["threadId"])
    request_id = cast(str, params["requestId"])
    runtime = server.store._runtime_view(thread_id)

    server.kernel.approve(
        thread_id,
        runtime,
        request_id,
        ApprovalDecision.allow(),
        "custom-approval-answer",
    )

    assert server.run_adapter._approval_answered(server.kernel.store, thread_id, request_id) is True
    assert server.run_adapter._approval_answered(server.kernel.store, thread_id, request_id + "-different") is False
    server.kernel._worker.wake(thread_id)
    outbound = _drain_until_turn_completed(transport)
    assert calls == ["ran"]
    assert not any(message.get("method") == "approval/request" for message in outbound)


@pytest.mark.parametrize(("record_kind", "expected_due"), [("approval", 10.05), ("turn", 10.05)])
def test_committed_record_progress_immediately_resets_observation_backoff(record_kind: str, expected_due: float) -> None:
    class Clock:
        def monotonic(self) -> float:
            return 10.0

    class Store:
        committed = False

        def read(self, session_id: str, *, after_seq: int, limit: int):
            assert session_id == "thread_1"
            assert after_seq == 0 and limit == 1
            return SimpleNamespace(records=[object()] if self.committed else [])

    delivered: list[str] = []
    position = SimpleNamespace(seq=0, due=20.0, delay=0.8, warning_at=0.0, replay_approvals=False)
    store = Store()
    adapter = object.__new__(_KernelRunAdapter)
    adapter.clock = Clock()
    adapter._notification_positions = {"thread_1": position}
    adapter._state_manager = SimpleNamespace(subscribers=lambda _sid: {"conn_1"})
    adapter._publish_operational_error = lambda _sid, _store: False
    adapter._next_approval_seq = lambda _sid, _store, *, after_seq: 1 if record_kind == "approval" and store.committed else None

    def send_pending_approvals(*_args, **_kwargs):
        delivered.append("approval")
        return None

    def publish_turns(*_args, **kwargs):
        if not store.committed or kwargs.get("through_seq") == 0:
            return False
        if record_kind == "turn":
            delivered.append("turn")
        position.seq = 1
        return True

    adapter._send_pending_approvals = send_pending_approvals
    adapter._publish_turns = publish_turns
    adapter._reconcile_local_turn = lambda _sid, _store: None

    adapter._observe(store)
    assert delivered == []
    assert position.due == 20.0

    store.committed = True
    adapter._observe(store)

    assert delivered == [record_kind]
    assert position.due == pytest.approx(expected_due)


@pytest.mark.parametrize(
    ("runnable", "delay_ms", "lease_ms", "wake_session", "inflight_session", "expected"),
    [
        (False, None, 0, None, None, _JoinDriveState.NONE),
        (False, 250, 0, None, None, _JoinDriveState.FUTURE),
        (False, 0, 0, None, None, _JoinDriveState.DUE),
        (True, None, 0, None, None, _JoinDriveState.DUE),
        (False, 0, 250, "thread_1", None, _JoinDriveState.LOCAL_LEASE),
        (False, 0, 250, None, None, _JoinDriveState.REMOTE_LEASE),
        (False, 0, 250, None, "thread_1", _JoinDriveState.LOCAL_LEASE),
    ],
)
def test_join_drive_plan_distinguishes_schedule_and_lease_states(
    runnable: bool,
    delay_ms: int | None,
    lease_ms: int,
    wake_session: str | None,
    inflight_session: str | None,
    expected: _JoinDriveState,
) -> None:
    class Store:
        def lease_status(self, session_id: str) -> tuple[int, int, int]:
            assert session_id == "thread_1"
            return 1, 1 if lease_ms else 0, lease_ms

        def is_runnable(self, session_id: str) -> bool:
            assert session_id == "thread_1"
            return runnable

        def next_drive_delay_ms(self, session_id: str) -> int | None:
            assert session_id == "thread_1"
            return delay_ms

    worker = SimpleNamespace(
        _inflight_lease=SimpleNamespace(session_id=inflight_session) if inflight_session is not None else None,
        active_wake_sessions=frozenset({wake_session}) if wake_session is not None else frozenset(),
    )
    adapter = object.__new__(_KernelRunAdapter)
    adapter.kernel = SimpleNamespace(store=Store(), _worker=worker)

    assert adapter._join_drive_plan("thread_1").state is expected


@pytest.mark.parametrize(
    ("kind", "background", "child_pending", "retained"),
    [
        ("child", False, True, True),
        ("child", False, False, False),
        ("child", True, True, False),
        ("user", None, True, False),
        ("approval", None, True, False),
        ("provider", None, True, False),
    ],
)
def test_reconcile_retains_only_unquiescent_foreground_child_handle(
    kind: str,
    background: bool | None,
    child_pending: bool,
    retained: bool,
) -> None:
    class Manager:
        def __init__(self, active):
            self.active = active
            self.cleared = False

        def active_turn(self, sid: str):
            assert sid == "thread_1"
            return self.active

        def clear_active_turn(self, sid: str, turn_id: str, *, handle):
            assert sid == "thread_1" and turn_id == "thread_1/turn/turn_1"
            if self.active is None or self.active.handle is not handle:
                return False
            self.active = None
            self.cleared = True
            return True

    class Kernel:
        def __init__(self) -> None:
            self.pending_calls = 0
            self.abandoned: list[object] = []

        def _pending_child_work(self, store, state):
            del store, state
            self.pending_calls += 1
            return child_pending

        def _abandon_handle(self, handle) -> None:
            self.abandoned.append(handle)

    handle = object()
    active = SimpleNamespace(turn_id="thread_1/turn/turn_1", handle=handle)
    manager = Manager(active)
    child_handle: dict[str, object] = {"kind": kind}
    if background is not None:
        child_handle["background"] = background
    state = SimpleNamespace(
        turns={},
        phase="parked",
        waits={("operation_1", 1): {"handle": child_handle}},
    )
    store = SimpleNamespace(
        read_state=lambda _sid: (state, (), 0),
        lease_status=lambda _sid: (0, 0, 0),
    )
    kernel = Kernel()
    adapter = object.__new__(_KernelRunAdapter)
    adapter.kernel = kernel
    adapter._state_manager = manager
    adapter._settle_turn_connections = lambda *_args: None
    adapter._router = SimpleNamespace(cancel_matching_server_requests=lambda **_kwargs: None)

    adapter._reconcile_local_turn("thread_1", store)

    assert manager.cleared is not retained
    assert (manager.active is active) is retained
    assert kernel.abandoned == ([] if retained else [handle])
    assert kernel.pending_calls == (1 if kind == "child" and background is False else 0)


def test_app_server_approval_timeout_denies_without_running_tool(surface) -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls, approval_timeout_seconds=1)

    approval_request = _start_and_wait_for_approval(server, transport)
    assert approval_request["method"] == "approval/request"
    state, _, _ = surface.store.read_state("thread_1")
    deadline_ms = next(iter(state.waits.values()))["deadline_ms"]
    time.sleep(max(0.0, (deadline_ms - time.time() * 1000) / 1000) + 0.02)
    cast(Any, server.run_adapter).join()
    outbound = _drain_until_turn_completed(transport)

    assert calls == []
    assert _approval_resolutions(outbound) == ["timeout"]
    assert _completed_status(outbound) == "completed"


def test_app_server_approval_disconnect_resolves_at_retained_deadline(surface) -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls, approval_timeout_seconds=1)

    approval_request = _start_and_wait_for_approval(server, transport)
    assert approval_request["method"] == "approval/request"
    state, _, _ = surface.store.read_state("thread_1")
    wait = next(iter(state.waits.values()))
    deadline_ms = wait["deadline_ms"]
    assert isinstance(deadline_ms, int)
    server.router.unregister_transport("conn_1")
    assert server.store.read_thread("thread_1").thread.status == "approval_wait"
    cast(Any, server.run_adapter).recover("observer", "thread_1")
    still_waiting, _, _ = surface.store.read_state("thread_1")
    assert still_waiting.waits == state.waits
    assert not any(turn.ended for turn in still_waiting.turns.values())

    time.sleep(max(0.0, (deadline_ms - time.time() * 1000) / 1000) + 0.02)
    cast(Any, server.run_adapter).join()
    completion_deadline = time.time() + 2
    while time.time() < completion_deadline:
        snapshot = server.store.read_thread("thread_1")
        if snapshot.turns and snapshot.turns[-1].status == "completed":
            break
        time.sleep(0.01)

    snapshot = server.store.read_thread("thread_1")
    assert calls == []
    assert snapshot.turns[-1].status == "completed"


def test_app_server_approval_deadline_is_retained_after_subscriber_disconnect(surface) -> None:
    calls: list[str] = []
    server, owner = _server_with_approval_tool(calls, approval_timeout_seconds=1)
    observer = ChannelTransport(connection_id="conn_2")
    server.router.register_transport(observer)
    _initialize(server, owner, request_id=0, client_name="owner")
    thread_id = _start_thread(server, owner, request_id=1)
    _initialize(server, observer, request_id=10, client_name="observer")
    _send(
        observer,
        server,
        {"jsonrpc": "2.0", "id": 11, "method": "thread/resume", "params": {"threadId": thread_id}},
    )
    _drain_until(observer, lambda message: message.get("id") == 11)

    _send(
        owner,
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "turn/start",
            "params": {
                "inputId": "turn_1",
                "expectedTurnId": None,
                "threadId": thread_id,
                "input": [{"type": "text", "text": "run tool"}],
            },
        },
    )
    approval_request = _drain_until(owner, lambda message: message.get("method") == "approval/request")
    started = _drain_until(observer, lambda message: message.get("method") == "turn/started")
    observer_before_disconnect = _drain_available(observer)
    assert cast(dict[str, object], started["params"])["turnId"] == thread_id + "/turn/turn_1"
    assert not any(message.get("method") == "approval/request" for message in observer_before_disconnect)
    state, _, _ = surface.store.read_state(thread_id)
    deadline_ms = next(iter(state.waits.values()))["deadline_ms"]
    now_ms = [deadline_ms - 1]
    with _controlled_session_clock(server.kernel, lambda: now_ms[0]):
        server.router.unregister_transport(owner.connection_id)
        join_started = time.monotonic()
        cast(Any, server.run_adapter).join()
        assert time.monotonic() - join_started < 3
        before_deadline = _drain_available(observer)
        assert not any(_is_terminal_turn_notification(message) for message in before_deadline)
        time.sleep(max(0.0, (deadline_ms - time.time() * 1000) / 1000) + 0.02)
        now_ms[0] = deadline_ms
        cast(Any, server.run_adapter).recover(observer.connection_id, thread_id)
        observer_after_disconnect = _drain_until_turn_completed(observer, timeout=2)

    assert approval_request["method"] == "approval/request"
    assert not any(message.get("method") == "approval/request" for message in observer_after_disconnect)
    assert _approval_resolutions(observer_after_disconnect) == ["timeout"]
    assert calls == []


def test_pure_observer_reconnect_replays_durable_approval_without_local_runtime(surface) -> None:
    calls: list[str] = []
    first, owner = _server_with_approval_tool(calls)
    approval = _start_and_wait_for_approval(first, owner)
    approval_id = approval["id"]
    assert isinstance(approval_id, str)
    thread_id = "thread_1"
    assert thread_id in first.kernel._runtimes

    first.router.unregister_transport(owner.connection_id)
    first.run_adapter.close()
    first.kernel.close()

    second, observer = _server_with_approval_tool(calls)
    second.kernel._worker.support = replace(
        second.kernel._worker.support,
        drive=None,
        project=None,
        projection_consumers=(),
    )
    assert thread_id not in second.kernel._runtimes

    _initialize(second, observer, request_id=10, client_name="observer")
    _send(
        observer,
        second,
        {"jsonrpc": "2.0", "id": 11, "method": "thread/resume", "params": {"threadId": thread_id}},
    )
    replayed = _drain_until(observer, lambda message: message.get("method") == "approval/request")
    assert replayed["id"] == approval_id
    assert thread_id not in second.kernel._runtimes
    assert not second.kernel.handles
    assert second.kernel._worker.support.drive is None

    _send(
        observer,
        second,
        {"jsonrpc": "2.0", "id": approval_id, "result": {"decision": "allow", "reason": "reconnected"}},
    )
    answer_id = f"approval/{approval_id}/answer"
    assert any(entry.item.input_id == answer_id for entry in second.kernel.store.peek_inbox(thread_id))
    assert thread_id not in second.kernel._runtimes
    assert not second.kernel.handles


def test_owner_reconnect_replays_request_without_broadcasting_requested() -> None:
    calls: list[str] = []
    server, owner = _server_with_approval_tool(calls)
    observer = ChannelTransport(connection_id="observer")
    server.router.register_transport(observer)
    _initialize(server, owner, request_id=0, client_name="owner")
    thread_id = _start_thread(server, owner, request_id=1)
    _initialize(server, observer, request_id=10, client_name="observer")
    _send(
        observer,
        server,
        {"jsonrpc": "2.0", "id": 11, "method": "thread/resume", "params": {"threadId": thread_id}},
    )
    _drain_until(observer, lambda message: message.get("id") == 11)
    _drain_available(observer)

    _send(owner, server, {"jsonrpc": "2.0", "id": 2, "method": "turn/start", "params": _turn_params(thread_id)})
    approval = _drain_until(owner, lambda message: message.get("method") == "approval/request")
    _drain_until(observer, lambda message: message.get("method") == "approval/requested")
    _drain_available(observer)

    server.router.unregister_transport(owner.connection_id)
    replacement = ChannelTransport(connection_id="replacement")
    server.router.register_transport(replacement)
    _initialize(server, replacement, request_id=20, client_name="replacement")
    _send(
        replacement,
        server,
        {"jsonrpc": "2.0", "id": 21, "method": "thread/resume", "params": {"threadId": thread_id}},
    )
    replayed = _drain_until(replacement, lambda message: message.get("method") == "approval/request")
    observer_after_reconnect = _drain_available(observer)

    assert replayed["id"] == approval["id"]
    assert not any(message.get("method") in {"approval/requested", "approval/request"} for message in observer_after_reconnect)


def test_second_turn_registers_config_before_worker_can_drive() -> None:
    first_calls: list[str] = []
    second_calls: list[str] = []
    registered_turns: list[object] = []
    drive_turns: list[object] = []
    second_config_entered = threading.Event()
    release_second_config = threading.Event()
    transport = ChannelTransport(connection_id="owner")

    first_provider = ScriptedModelProvider.from_callback(
        "first",
        "m",
        lambda _request: first_calls.append("first") or LLMResponse("first done"),
    )

    server: AppServer

    def second_model(_request):
        second_calls.append("second")
        registered_turn = server.kernel._runtime_registration("thread_1").config.metadata.get("turn_id")
        assert registered_turn == "thread_1/turn/turn_2"
        registered_turns.append(registered_turn)
        return LLMResponse("second done")

    second_provider = ScriptedModelProvider.from_callback("second", "m", second_model)

    class PerTurnHost(DefaultAppServerHost):
        def build_run_config(self, request):
            run = request.metadata.get("run")
            if run == "second":
                second_config_entered.set()
                assert release_second_config.wait(10)
            provider = second_provider if run == "second" else first_provider
            return RunConfig(model_provider=provider, metadata={"run": run})

        def resolve_model(self, request):
            provider = second_provider if second_config_entered.is_set() else first_provider
            return resolve_model_selection(provider, request)

    server = AppServer(transport=transport, host=PerTurnHost())
    _initialize(server, transport, request_id=0, client_name="owner")
    _send(
        transport,
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "thread/start",
            "params": {"inputId": "thread_1", "metadata": {"run": "first"}},
        },
    )
    _drain_until(transport, lambda message: message.get("id") == 1)
    _send(
        transport,
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "turn/start",
            "params": {
                "inputId": "turn_1",
                "expectedTurnId": None,
                "threadId": "thread_1",
                "metadata": {"run": "first"},
                "input": [{"type": "text", "text": "first"}],
            },
        },
    )
    first_outbound = _drain_until_turn_completed(transport)
    assert _completed_status(first_outbound) == "completed"
    assert first_calls == ["first"]

    worker = server.kernel._worker
    original_drive = worker.support.drive
    assert original_drive is not None

    def tracked_drive(store, session_id, *, lease):
        drive_turns.append(server.kernel._runtime_registration(session_id).config.metadata.get("turn_id"))
        original_drive(store, session_id, lease=lease)

    worker.support = replace(worker.support, drive=tracked_drive)

    second_result: dict[str, object] = {}

    def start_second() -> None:
        _send(
            transport,
            server,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "turn/start",
                "params": {
                    "inputId": "turn_2",
                    "expectedTurnId": "thread_1/turn/turn_1",
                    "threadId": "thread_1",
                    "metadata": {"run": "second"},
                    "input": [{"type": "text", "text": "second"}],
                },
            },
        )
        second_result["response"] = _drain_until(transport, lambda message: message.get("id") == 3)

    thread = threading.Thread(target=start_second, name="second-turn-start")
    try:
        thread.start()
        assert second_config_entered.wait(5)
        previous_pass = worker.coordinate_pass
        worker.wake("thread_1")
        assert worker.wait_for_coordinate_pass(previous_pass, 5)
        worker.wait_idle(5)
        assert first_calls == ["first"]
        assert second_calls == []
        assert not any(entry.item.input_id == "turn_2" for entry in server.kernel.store.peek_inbox("thread_1"))
    finally:
        release_second_config.set()
        thread.join(10)
    assert not thread.is_alive()
    assert "result" in cast(dict[str, object], second_result["response"])
    second_outbound = _drain_until_turn_completed(transport)
    assert _completed_status(second_outbound) == "completed"
    assert first_calls == ["first"]
    assert second_calls == ["second"]
    assert registered_turns == ["thread_1/turn/turn_2"]
    assert drive_turns == ["thread_1/turn/turn_2"]


def test_app_server_follow_up_routes_callback_to_driving_subscriber() -> None:
    calls: list[str] = []
    server, owner = _server_with_approval_tool(calls)
    observer = ChannelTransport(connection_id="conn_2")
    server.router.register_transport(observer)
    _initialize(server, owner, request_id=0, client_name="owner")
    thread_id = _start_thread(server, owner, request_id=1)
    _initialize(server, observer, request_id=10, client_name="observer")
    _send(
        observer,
        server,
        {"jsonrpc": "2.0", "id": 11, "method": "thread/resume", "params": {"threadId": thread_id}},
    )
    _drain_until(observer, lambda message: message.get("id") == 11)

    _send(
        owner,
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "turn/start",
            "params": {
                "inputId": "turn_1",
                "expectedTurnId": None,
                "threadId": thread_id,
                "input": [{"type": "text", "text": "first"}],
            },
        },
    )
    first_approval = _drain_until(owner, lambda message: message.get("method") == "approval/request")
    first_params = cast(dict[str, object], first_approval["params"])
    _send(
        observer,
        server,
        {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "turn/followUp",
            "params": {
                "inputId": "follow_1",
                "threadId": thread_id,
                "expectedTurnId": first_params["turnId"],
                "input": [{"type": "text", "text": "second"}],
            },
        },
    )
    _drain_until(observer, lambda message: message.get("id") == 12)
    _send(
        owner,
        server,
        {"jsonrpc": "2.0", "id": first_approval["id"], "result": {"decision": "allow"}},
    )

    second_approval = _drain_until(owner, lambda message: message.get("method") == "approval/request")
    second_params = cast(dict[str, object], second_approval["params"])
    assert second_params["turnId"] == f"{thread_id}/turn/follow_1"
    assert second_params["turnId"] != first_params["turnId"]
    assert second_params["requestId"] != first_params["requestId"]
    observer_messages = _drain_available(observer)
    assert not any(message.get("method") == "approval/request" for message in observer_messages)
    _send(
        owner,
        server,
        {"jsonrpc": "2.0", "id": second_approval["id"], "result": {"decision": "allow"}},
    )
    _drain_until_turn_completed(owner)

    assert calls == ["ran", "ran"]


def test_app_server_approval_interrupt_releases_pending_request_without_waiting_for_timeout() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls, approval_timeout_seconds=None)

    approval_request = _start_and_wait_for_approval(server, transport)
    approval_params = cast(dict[str, object], approval_request["params"])
    _send(
        transport,
        server,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "turn/interrupt",
            "params": {
                "actionId": "interrupt_1",
                "threadId": approval_params["threadId"],
                "expectedTurnId": approval_params["turnId"],
            },
        },
    )
    outbound = _drain_until_turn_status(transport, "failed", timeout=2)

    assert calls == []
    assert server.router.pending_server_request_count() == 0
    assert any(
        message.get("id") == 3
        and message.get("result")
        == {
            "threadId": approval_params["threadId"],
            "turnId": approval_params["turnId"],
            "actionId": "interrupt_1",
            "accepted": True,
            "status": "running",
        }
        for message in outbound
    )
    assert _approval_resolutions(outbound) == []
    assert _completed_status(outbound) == "failed"


def test_app_server_approval_allow_session_can_be_resolved_by_client_request() -> None:
    calls: list[str] = []
    server, transport = _server_with_approval_tool(calls)

    approval_request = _start_and_wait_for_approval(server, transport)
    approval_params = cast(dict[str, object], approval_request["params"])
    requested = cast(dict[str, object] | None, approval_request.get("_requested"))

    _send(
        transport,
        server,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "approval/resolve",
            "params": {
                "threadId": approval_params["threadId"],
                "turnId": approval_params["turnId"],
                "requestId": approval_params["requestId"],
                "decision": "allow_session",
                "reason": "approved by owner",
                "metadata": {"ticket": 7},
            },
        },
    )
    outbound = _drain_until_turn_completed(transport)

    assert calls == ["ran"]
    assert server.router.pending_server_request_count() == 0
    requested = requested or next((message for message in outbound if message.get("method") == "approval/requested"), None)
    assert requested is not None
    assert cast(dict[str, object], requested["params"])["requestId"] == approval_params["requestId"]
    assert any(message.get("id") == 3 and message.get("result") == {} for message in outbound)
    assert _approval_resolutions(outbound) == ["allow_session"]
    resolved = next(
        cast(dict[str, object], message["params"])
        for message in outbound
        if message.get("method") == "approval/resolved" and isinstance(message.get("params"), dict)
    )
    assert resolved["reason"] == "approved by owner"
    assert resolved["metadata"] == {"ticket": 7}


def test_server_request_response_is_bound_to_connection_thread_and_turn() -> None:
    owner = ChannelTransport(connection_id="owner")
    attacker = ChannelTransport(connection_id="attacker")
    router = OutgoingRouter()
    router.register_transport(owner)
    router.register_transport(attacker)
    processor = MessageProcessor(router=router)
    pending = router.send_server_request(
        "owner",
        "approval/request",
        {"threadId": "thread_1", "turnId": "turn_1"},
        request_id=RequestId("approval_1"),
    )

    processor.process_message(
        "attacker",
        {"jsonrpc": "2.0", "id": "approval_1", "result": {"decision": "allow"}},
    )
    with pytest.raises(TimeoutError):
        pending.result(timeout=0)

    assert not router.resolve_server_request(
        "owner",
        RequestId("approval_1"),
        {"decision": "allow"},
        method="approval/request",
        thread_id="thread_other",
        turn_id="turn_1",
    )
    assert router.resolve_server_request(
        "owner",
        RequestId("approval_1"),
        {"decision": "allow"},
        method="approval/request",
        thread_id="thread_1",
        turn_id="turn_1",
    )
    assert pending.result(timeout=0) == {"decision": "allow"}


def _server_with_approval_tool(
    calls: list[str],
    *,
    approval_timeout_seconds: float | None = None,
    approval_provider=None,
) -> tuple[AppServer, ChannelTransport]:
    @function_tool(needs_approval=True)
    def dangerous_tool() -> str:
        calls.append("ran")
        return "allowed"

    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="dangerous_tool", arguments={})]),
            LLMResponse(content="done"),
            LLMResponse(content="calling again", tool_calls=[ToolCall(id="call_2", name="dangerous_tool", arguments={})]),
            LLMResponse(content="done again"),
        ]
    )

    transport = ChannelTransport(connection_id="conn_1")
    host = DefaultAppServerHost(
        agent=Agent(name="assistant", instructions="Use the tool.", model="test-model", tools=[dangerous_tool]),
        run_config=RunConfig(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            max_cycles=3,
            approval_timeout_seconds=approval_timeout_seconds,
            approval_provider=approval_provider,
        ),
    )
    return AppServer(transport=transport, host=host), transport


def _server_with_parameterized_approval_tool(
    calls: list[dict[str, object]],
    arguments: dict[str, object],
    *,
    hooks: list[RuntimeHook] | None = None,
) -> tuple[AppServer, ChannelTransport]:
    @function_tool(needs_approval=True)
    def dangerous_tool(path: str, force: bool) -> str:
        calls.append({"path": path, "force": force})
        return "allowed"

    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="dangerous_tool", arguments=arguments)]),
            LLMResponse(content="done"),
        ]
    )
    transport = ChannelTransport(connection_id="conn_1")
    host = DefaultAppServerHost(
        agent=Agent(
            name="assistant",
            instructions="Use the tool.",
            model="test-model",
            tools=[dangerous_tool],
            hooks=hooks or [],
        ),
        run_config=RunConfig(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            max_cycles=3,
        ),
    )
    return AppServer(transport=transport, host=host), transport


@contextmanager
def _controlled_session_clock(surface, now_ms: Callable[[], int]):
    """Apply test database time to worker drive and heartbeat connections."""

    original_open = surface.open_store
    original_heartbeat_open = surface._heartbeat_store_factory

    def configure(store) -> None:
        raw = store
        seen = set()
        while id(raw) not in seen:
            seen.add(id(raw))
            nested = getattr(raw, "_store", None)
            if nested is None or nested is raw:
                break
            raw = nested
        connection = getattr(raw, "connection", None)
        if connection is None:
            raise AssertionError("drive store has no SQLite connection")
        connection.create_function("session_now_ms", 0, now_ms)

    @contextmanager
    def open_store():
        with surface._open_default_drive_store() as store:
            configure(store)
            yield store

    @contextmanager
    def heartbeat_store():
        with original_heartbeat_open() as store:
            configure(store)
            yield store

    surface.open_store = open_store
    surface._heartbeat_store_factory = heartbeat_store
    try:
        yield
    finally:
        surface.open_store = original_open
        surface._heartbeat_store_factory = original_heartbeat_open


def _start_and_wait_for_approval(server: AppServer, transport: ChannelTransport) -> dict[str, object]:
    _initialize(server, transport, request_id=0, client_name="test")
    thread_id = _start_thread(server, transport, request_id=1)
    _send(
        transport,
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "turn/start",
            "params": {
                "inputId": "turn_1",
                "expectedTurnId": None,
                "threadId": thread_id,
                "input": [{"type": "text", "text": "run tool"}],
            },
        },
    )
    requested = None
    outbound = []
    while True:
        try:
            message = transport.receive_outbound(timeout=10)
        except queue.Empty as exc:
            raise AssertionError(f"approval request did not arrive; outbound={outbound!r}") from exc
        outbound.append(message)
        if message.get("method") == "approval/requested":
            requested = message
        if message.get("method") == "approval/request":
            if requested is not None:
                message["_requested"] = requested
            return message


def _turn_params(thread_id: str = "thread_1") -> dict[str, object]:
    return {
        "inputId": "turn_1",
        "expectedTurnId": None,
        "threadId": thread_id,
        "input": [{"type": "text", "text": "run tool"}],
    }


def _initialize(
    server: AppServer,
    transport: ChannelTransport,
    *,
    request_id: int,
    client_name: str,
) -> None:
    _send(
        transport,
        server,
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {"clientInfo": {"name": client_name}},
        },
    )
    _drain_until(transport, lambda message: message.get("id") == request_id)
    _send(transport, server, {"jsonrpc": "2.0", "method": "initialized"})


def _start_thread(server: AppServer, transport: ChannelTransport, *, request_id: int) -> str:
    _send(
        transport,
        server,
        {"jsonrpc": "2.0", "id": request_id, "method": "thread/start", "params": {"inputId": "thread_1", "agentKey": "default"}},
    )
    response = _drain_until(transport, lambda message: message.get("id") == request_id)
    result = cast(dict[str, object], response["result"])
    return cast(str, result["threadId"])


def _send(transport: ChannelTransport, server: AppServer, payload: dict[str, object]) -> None:
    transport.send_inbound(payload)
    server.processor.process_next(transport)


def _drain_until(
    transport: ChannelTransport,
    predicate: Callable[[dict[str, object]], bool],
    *,
    timeout: float = 10,
) -> dict[str, object]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        message = transport.receive_outbound(timeout=timeout)
        if predicate(message):
            return message
    raise AssertionError("Timed out waiting for App Server message")


def _is_terminal_turn_notification(message: dict[str, object]) -> bool:
    params = message.get("params")
    return (
        message.get("method") == "turn/completed"
        and isinstance(params, dict)
        and cast(dict[str, object], params).get("status")
        in (
            "completed",
            "failed",
        )
    )


def _drain_until_turn_completed(transport: ChannelTransport, *, timeout: float = 10) -> list[dict[str, object]]:
    outbound: list[dict[str, object]] = []
    deadline = time.monotonic() + timeout
    while True:
        message = transport.receive_outbound(timeout=max(0.0, deadline - time.monotonic()))
        outbound.append(message)
        if _is_terminal_turn_notification(message):
            return outbound


def _drain_until_turn_status(
    transport: ChannelTransport,
    status: str,
    *,
    timeout: float = 10,
) -> list[dict[str, object]]:
    outbound: list[dict[str, object]] = []
    deadline = time.monotonic() + timeout
    while True:
        message = transport.receive_outbound(timeout=max(0.0, deadline - time.monotonic()))
        outbound.append(message)
        params = message.get("params")
        if message.get("method") == "turn/completed" and isinstance(params, dict) and params.get("status") == status:
            return outbound


def _drain_available(transport: ChannelTransport) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    while True:
        try:
            messages.append(transport.receive_outbound(timeout=0))
        except queue.Empty:
            return messages


def _completed_status(outbound: list[dict[str, object]]) -> object:
    completed = [message for message in outbound if message.get("method") == "turn/completed"][-1]
    params = completed.get("params")
    assert isinstance(params, dict)
    return cast(dict[str, object], params)["status"]


def _approval_resolutions(outbound: list[dict[str, object]]) -> list[object]:
    return [
        cast(dict[str, object], message["params"])["decision"]
        for message in outbound
        if message.get("method") == "approval/resolved" and isinstance(message.get("params"), dict)
    ]
