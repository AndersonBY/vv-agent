"""W4 App Server observation ownership and coordinator lifecycle regressions."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from session.app_server_s1_helpers import declared_content_ref
from session.test_app_server_providers import THREAD as PROVIDER_THREAD
from session.test_app_server_providers import TURN as PROVIDER_TURN
from session.test_app_server_providers import BusinessDB, JobsProvider
from session.test_app_server_providers import Server as ProviderServer
from support import FixedModelProvider
from test_app_server_distributed import node as _distributed_node
from test_app_server_distributed import request, submission

from vv_agent import Accepted, Agent, ScriptedModelProvider, function_tool
from vv_agent.app_server import AppServer, ChannelTransport
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._provider_outcomes import ProviderResultInvalid
from vv_agent.llm import ScriptedLLM
from vv_agent.session.runtime import static_session_requirements
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse, ToolCall


def node(stack, path, connection, provider, *, agent=None, content_key_provider=None):
    server = _distributed_node(
        stack,
        path,
        connection,
        provider,
        agent=agent,
        content_key_provider=content_key_provider,
    )
    # Static App Server bootstrap persists level-2 sessions. These shared
    # SQLite fixtures install the default level-1 schema directly.
    server.kernel._root_store._rows("UPDATE sk_settings SET write_level=? WHERE singleton=1", (2,))
    return server


def _eventually(predicate: Callable[[], bool], timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def _observer_node(stack: ExitStack, path: Path, connection: str, provider):
    server = node(stack, path, connection, provider)
    support = replace(
        server.kernel._worker.support,
        drive=None,
        project=None,
        projection_consumers=(),
    )
    server.kernel._worker.support = support
    assert server.kernel._worker.support.observe is not None
    return server


def _drain(transport: ChannelTransport, *, timeout: float = 0.0) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            messages.append(transport.receive_outbound(timeout=remaining))
        except queue.Empty:
            return messages
        if timeout == 0:
            deadline = time.monotonic()


def _until(transport: ChannelTransport, predicate: Callable[[dict[str, Any]], bool], timeout: float = 5):
    messages: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            message = transport.receive_outbound(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            break
        messages.append(message)
        if predicate(message):
            return message, messages
    raise AssertionError(f"Timed out waiting for outbound message; got {messages!r}")


def _send(server, transport: ChannelTransport, payload: dict[str, Any]) -> None:
    transport.send_inbound(payload)
    server.processor.process_next(transport)


def _request_with_messages(server, transport: ChannelTransport, request_id: Any, method: str, params: dict[str, Any]):
    _send(
        server,
        transport,
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
    )
    return _until(transport, lambda message: message.get("id") == request_id)


def _resolved_model() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="test-model",
        selected_model="test-model",
        model_id="test-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
    )


def _approval_node(stack: ExitStack, path: Path, connection: str, calls: list[str]):
    @function_tool(needs_approval=True)
    def protected() -> str:
        calls.append("ran")
        return "allowed"

    llm = ScriptedLLM(
        steps=[
            LLMResponse("", [ToolCall("protected", "protected", {})]),
            LLMResponse("done"),
        ]
    )
    provider = FixedModelProvider(llm, _resolved_model())
    return node(
        stack,
        path,
        connection,
        provider,
        agent=Agent("test", "Use protected.", tools=[protected]),
    )


def _lifecycle(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [
        (cast(str, message["method"]), cast(dict[str, Any], message["params"])["turnId"])
        for message in messages
        if message.get("method") in {"turn/started", "turn/completed"}
    ]


def _create_level2_provider_thread(provider_server) -> None:
    kernel = provider_server.server.kernel
    kernel._next_session_requirements = static_session_requirements(
        provider_server.host._agent,
        provider_server.host._run_config,
        handler_version="1",
    )
    kernel.create(
        PROVIDER_THREAD,
        "/fixture",
        attributes={"app_server": {"agent_key": "default", "cwd": None, "metadata": {}}},
        principal={
            "tenant_id": kernel.tenant_id,
            "subject_id": "app-server",
            "subject_kind": "service",
            "attributes": {"role": "app-server"},
        },
    )


def _start_invalid_provider_wait(provider_server, business):
    _create_level2_provider_thread(provider_server)
    provider_server.start()
    handle = provider_server.attempt().wait["handle"]
    business.finish()
    business.setting("invalid", "call")
    provider_server.due()
    provider_server.resume()
    if not any(
        message.get("method") == "error/warning" and message.get("params", {}).get("code") == "provider_result_invalid"
        for message in provider_server.messages
    ):
        _, warning_messages = _until(
            provider_server.transport,
            lambda message: (
                message.get("method") == "error/warning" and message.get("params", {}).get("code") == "provider_result_invalid"
            ),
        )
        provider_server.messages.extend(warning_messages)
    failure = provider_server.server.kernel._errors[PROVIDER_THREAD]
    assert isinstance(failure.error, ProviderResultInvalid) and failure.terminal
    assert failure.warning_sent is True
    assert business.count("submit") == business.count("effect") == business.count("query") == 1
    return handle, failure


def _provider_invalid_warning_count(messages: list[dict[str, Any]]) -> int:
    return sum(
        message.get("method") == "error/warning" and message.get("params", {}).get("code") == "provider_result_invalid"
        for message in messages
    )


def _callback_for_business_job(business, *, input_id: str):
    connection = business.connect()
    try:
        keys = [row[0] for row in connection.execute("SELECT key FROM jobs").fetchall()]
    finally:
        connection.close()
    assert len(keys) == 1
    provider = JobsProvider(business)
    plan, _ = business.job(keys[0])
    return provider.callback(provider.handle(plan, keys[0]), input_id=input_id)


def _tx_push_provider_result(provider_server, item) -> None:
    kernel = provider_server.server.kernel
    with provider_server.server.kernel.store.atomic() as tx:
        kernel._push_in_transaction(
            tx,
            PROVIDER_THREAD,
            item,
            actor_subject_id="provider",
        )
    stored = kernel.store.read_input(PROVIDER_THREAD, item.input_id).item
    assert stored.actor is not None
    assert declared_content_ref(stored.actor["attributes"]) is not None
    assert declared_content_ref(stored.payload["result"]) is not None
    assert declared_content_ref(stored.payload["usage"]) is not None
    provider_server.server.wake(PROVIDER_THREAD)


class _YieldOnSteerJobsProvider(JobsProvider):
    def submit(self, plan, *, context):
        outcome = super().submit(plan, context=context)
        if isinstance(outcome, Accepted):
            outcome = replace(outcome, yield_on_steer=True)
        return outcome


class _S1JobsProvider(JobsProvider):
    def authenticate(self, item, plan):
        # AppServer attaches S1 actor/content references after the provider
        # creates its logical payload. Authenticate the logical input while
        # retaining the raw item for the kernel's S1 validation.
        logical = item.item if hasattr(item, "item") else item
        key = plan.payload["idempotency_key"]
        expected = self.callback(self.handle(plan, key), input_id=logical.input_id)
        retained = logical.to_dict() | {"actor": None}
        return canonical_json_bytes(expected.to_dict()) == canonical_json_bytes(retained)


def test_default_observer_accepts_empty_thread_and_queued_turn_without_provider(tmp_path):
    server = AppServer(transport=ChannelTransport(connection_id="observer", outbound_capacity=10000))
    try:
        server.kernel._worker.support = replace(
            server.kernel._worker.support,
            drive=None,
            project=None,
            projection_consumers=(),
        )
        assert "result" in request(server, "init", "initialize", {"clientInfo": {"name": "observer"}})
        server.processor.process_message("observer", {"jsonrpc": "2.0", "method": "initialized"})
        assert "result" in request(server, 1, "thread/start", {"inputId": "thread"})
        assert "result" in request(server, 2, "thread/resume", {"threadId": "thread"})

        receipt = request(server, 3, "turn/start", submission())["result"]
        assert receipt["accepted"]
        state, _, _ = server.kernel.store.read_state("thread")
        assert state.active_turn_id is None and not state.turns
        assert [entry.item.input_id for entry in server.kernel.store.peek_inbox("thread")] == ["prompt"]

        assert "result" in request(server, 4, "thread/resume", {"threadId": "thread"})
        assert server.kernel._worker.support.drive is None
        assert server.kernel._worker._drive_thread_ident is None
        assert server.kernel._runtimes == {}
    finally:
        server.run_adapter.close()
        server.kernel.close()


def test_pure_observer_continues_after_events_and_traces_are_acked(tmp_path):
    with ExitStack() as stack:
        a = node(stack, tmp_path / "store", "A", FixedModelProvider(ScriptedLLM([LLMResponse("done")]), _resolved_model()))
        b = _observer_node(
            stack,
            tmp_path / "store",
            "B",
            FixedModelProvider(ScriptedLLM([LLMResponse("unused")]), _resolved_model()),
        )
        request(a, 1, "thread/start", {"inputId": "thread"})
        assert "result" in request(b, 2, "thread/resume", {"threadId": "thread"})
        assert b.kernel._worker.support.drive is None

        with b.kernel.store.atomic() as tx:
            for consumer in ("events", "traces"):
                batch = tx.consumer_batch("thread", consumer)
                if batch is not None:
                    tx.ack(batch)
        before = {
            cast(str, row[0]): cast(int, row[1])
            for row in b.kernel.store._rows(
                "SELECT consumer,last_seq FROM sk_consumer WHERE session_id='thread' AND consumer IN ('events','traces')"
            )
        }

        tid = request(a, 3, "turn/start", submission())["result"]["turnId"]
        _, messages = _until(b.transport, lambda message: message.get("method") == "turn/completed")
        assert _lifecycle(messages) == [("turn/started", tid), ("turn/completed", tid)]
        after = {
            cast(str, row[0]): cast(int, row[1])
            for row in b.kernel.store._rows(
                "SELECT consumer,last_seq FROM sk_consumer WHERE session_id='thread' AND consumer IN ('events','traces')"
            )
        }
        assert after == before


def test_observer_lifecycle_order_matches_committed_turn_records_once(tmp_path):
    with ExitStack() as stack:
        a = node(
            stack,
            tmp_path / "store",
            "A",
            FixedModelProvider(
                ScriptedLLM([LLMResponse("one"), LLMResponse("two"), LLMResponse("three")]),
                _resolved_model(),
            ),
        )
        b = _observer_node(
            stack,
            tmp_path / "store",
            "B",
            FixedModelProvider(ScriptedLLM([LLMResponse("unused")]), _resolved_model()),
        )
        request(a, 1, "thread/start", {"inputId": "thread"})
        request(b, 2, "thread/resume", {"threadId": "thread"})

        messages: list[dict[str, Any]] = []
        previous = None
        for index in range(3):
            previous = request(
                a,
                index + 3,
                "turn/start",
                submission(identity=f"turn{index}", expected=previous),
            )["result"]["turnId"]
            a.run_adapter.join()
            _, completed = _until(b.transport, lambda message: message.get("method") == "turn/completed")
            messages.extend(completed)

        records = a.kernel.store.read_state("thread")[1]
        durable = [
            ("turn/started" if row.record.kind == "turn_started" else "turn/completed", row.record.turn_id)
            for row in records
            if row.record.kind in {"turn_started", "turn_ended"}
        ]
        assert _lifecycle(messages) == durable


def test_blocked_local_model_does_not_block_observer_store_or_thread(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def model(_request):
        entered.set()
        assert release.wait(5)
        return LLMResponse("done")

    with ExitStack() as stack:
        a = node(
            stack,
            tmp_path / "store",
            "A",
            ScriptedModelProvider.from_callback("test", "test-model", model),
        )
        b = _observer_node(
            stack,
            tmp_path / "store",
            "B",
            FixedModelProvider(ScriptedLLM([LLMResponse("unused")]), _resolved_model()),
        )
        stack.callback(release.set)
        observed: list[tuple[int, int, int | None]] = []
        observe = b.run_adapter._observe

        def record_observation(store):
            observed.append((threading.get_ident(), id(store), id(getattr(store, "connection", None))))
            observe(store)

        monkeypatch.setattr(b.run_adapter, "_observe", record_observation)
        request(a, 1, "thread/start", {"inputId": "thread"})
        assert "result" in request(b, 2, "thread/resume", {"threadId": "thread"})
        monkeypatch.setattr(b.kernel.store, "read", lambda *_a, **_k: pytest.fail("observer used kernel.store.read"))
        monkeypatch.setattr(
            b.kernel.store,
            "read_state",
            lambda *_a, **_k: pytest.fail("observer used kernel.store.read_state"),
        )
        tid = request(a, 3, "turn/start", submission())["result"]["turnId"]
        assert entered.wait(5)
        _, messages = _until(b.transport, lambda message: message.get("method") == "turn/started")
        assert _lifecycle(messages) == [("turn/started", tid)]
        _eventually(lambda: bool(observed))
        assert all(thread_id == b.kernel._worker._thread.ident for thread_id, _, _ in observed)
        assert all(store_id != id(b.kernel.store) for _, store_id, _ in observed)
        assert all(connection_id != id(getattr(b.kernel.store, "connection", None)) for _, _, connection_id in observed)
        assert b.kernel._worker._drive_thread_ident is None
        release.set()
        _until(b.transport, lambda message: message.get("method") == "turn/completed")


def test_observer_resume_never_claims_live_approval_request_owner(tmp_path):
    calls: list[str] = []
    with ExitStack() as stack:
        a = _approval_node(stack, tmp_path / "store", "A", calls)
        b = _observer_node(
            stack,
            tmp_path / "store",
            "B",
            FixedModelProvider(ScriptedLLM([LLMResponse("unused")]), _resolved_model()),
        )
        request(a, 1, "thread/start", {"inputId": "thread"})
        request(b, 2, "thread/resume", {"threadId": "thread"})
        tid = request(a, 3, "turn/start", submission())["result"]["turnId"]
        approval, _ = _until(a.transport, lambda message: message.get("method") == "approval/request")
        assert cast(dict[str, Any], approval["params"])["turnId"] == tid
        _, observer_messages = _until(b.transport, lambda message: message.get("method") == "turn/started")
        resume, resumed_messages = _request_with_messages(
            b,
            b.transport,
            "resume-active",
            "turn/resume",
            {"threadId": "thread", "turnId": tid},
        )
        assert resume["result"]["status"] == "interrupted"
        observer_messages.extend(resumed_messages)
        b.run_adapter.wake("thread")
        b.kernel._worker.wait_idle()
        observer_messages.extend(_drain(b.transport, timeout=0.2))
        assert not any(message.get("method") == "approval/request" for message in observer_messages)
        assert ("thread", tid) not in b.run_adapter._turn_connections
        assert b.router.pending_server_request_count() == 0
        assert a.router.pending_server_request_count() == 1


def test_observer_turn_resume_of_remote_owned_queued_turn_only_observes(tmp_path):
    with ExitStack() as stack:
        a = node(
            stack,
            tmp_path / "store",
            "A",
            FixedModelProvider(ScriptedLLM([LLMResponse("done")]), _resolved_model()),
        )
        b = _observer_node(
            stack,
            tmp_path / "store",
            "B",
            FixedModelProvider(ScriptedLLM([LLMResponse("unused")]), _resolved_model()),
        )
        request(a, 1, "thread/start", {"inputId": "thread"})
        request(b, 2, "thread/resume", {"threadId": "thread"})
        holder = a.kernel.store.acquire("thread", owner="remote-worker", ttl_ms=60_000)
        tid = request(a, 3, "turn/start", submission())["result"]["turnId"]
        assert tid not in a.kernel.store.read_state("thread")[0].turns

        response, _ = _request_with_messages(
            b,
            b.transport,
            4,
            "turn/resume",
            {"threadId": "thread", "turnId": tid},
        )
        assert response["result"] == {"threadId": "thread", "turnId": tid, "runId": tid, "status": "running"}
        assert b.state_manager.active_turn("thread") is None
        assert not [handle for handle in b.kernel.handles if handle.consumer == "app_server"]
        assert b.kernel._worker.support.drive is None
        assert b.kernel._worker._drive_thread_ident is None

        a.kernel.store.release(holder)
        a.run_adapter.wake("thread")
        a.run_adapter.join()
        _, messages = _until(b.transport, lambda message: message.get("method") == "turn/completed")
        assert _lifecycle(messages) == [("turn/started", tid), ("turn/completed", tid)]


@pytest.mark.parametrize("release", ["disconnect", "unsubscribe"])
def test_reconnect_replays_same_request_and_standard_response_answers_it(tmp_path, release):
    calls: list[str] = []
    with ExitStack() as stack:
        server = _approval_node(stack, tmp_path / "store", "A", calls)
        request(server, 1, "thread/start", {"inputId": "thread"})
        keepalive = ChannelTransport(connection_id="keepalive")
        server.router.register_transport(keepalive)
        _send(
            server,
            keepalive,
            {"jsonrpc": "2.0", "id": 20, "method": "initialize", "params": {"clientInfo": {"name": "keepalive"}}},
        )
        _until(keepalive, lambda message: message.get("id") == 20)
        _request_with_messages(server, keepalive, 21, "thread/resume", {"threadId": "thread"})
        tid = request(server, 2, "turn/start", submission())["result"]["turnId"]
        first, _ = _until(server.transport, lambda message: message.get("method") == "approval/request")
        first_params = cast(dict[str, Any], first["params"])

        if release == "disconnect":
            server.processor.disconnect_connection("A")
        else:
            request(server, "unsubscribe", "thread/unsubscribe", {"threadId": "thread"})
        assert server.router.pending_server_request_count() == 0
        assert server.run_adapter._turn_connections[(tid.split("/turn/")[0], tid)] is None
        server.run_adapter.join()

        reconnected = ChannelTransport(connection_id="B")
        server.router.register_transport(reconnected)
        _send(
            server,
            reconnected,
            {"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {"clientInfo": {"name": "B"}}},
        )
        _until(reconnected, lambda message: message.get("id") == 3)
        response, messages = _request_with_messages(
            server,
            reconnected,
            4,
            "thread/resume",
            {"threadId": "thread"},
        )
        assert "result" in response
        _, replay_messages = _until(reconnected, lambda message: message.get("method") == "approval/request")
        messages.extend(replay_messages)
        replay = next(message for message in messages if message.get("method") == "approval/request")
        replay_params = cast(dict[str, Any], replay["params"])
        assert replay_params["requestId"] == first_params["requestId"]
        assert replay_params["turnId"] == tid

        _, duplicate_messages = _request_with_messages(
            server,
            reconnected,
            5,
            "thread/resume",
            {"threadId": "thread"},
        )
        assert not any(message.get("method") == "approval/request" for message in duplicate_messages)

        _send(
            server,
            reconnected,
            {
                "jsonrpc": "2.0",
                "id": replay["id"],
                "result": {"decision": "allow", "reason": "reconnected"},
            },
        )
        completed, _ = _until(reconnected, lambda message: message.get("method") == "turn/completed")
        assert cast(dict[str, Any], completed["params"])["turnId"] == tid
        assert calls == ["ran"]


def test_reconnect_while_model_runs_moves_live_owner_without_duplicate_approval(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls: list[str] = []

    @function_tool(needs_approval=True)
    def protected() -> str:
        calls.append("ran")
        return "allowed"

    def first_model(_request):
        entered.set()
        assert release.wait(5)
        return LLMResponse("", [ToolCall("protected", "protected", {})])

    provider = FixedModelProvider(
        ScriptedLLM(steps=[first_model, LLMResponse("done")]),
        _resolved_model(),
    )
    with ExitStack() as stack:
        server = node(
            stack,
            tmp_path / "store",
            "A",
            provider,
            agent=Agent("test", "Use protected.", tools=[protected]),
        )
        stack.callback(release.set)
        request(server, 1, "thread/start", {"inputId": "thread"})
        tid = request(server, 2, "turn/start", submission())["result"]["turnId"]
        assert entered.wait(5)
        server.processor.disconnect_connection("A")
        assert server.run_adapter._turn_connections[("thread", tid)] is None
        assert server.kernel.store.lease_status("thread")[2] > 0

        reconnected = ChannelTransport(connection_id="B")
        server.router.register_transport(reconnected)
        _send(
            server,
            reconnected,
            {"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {"clientInfo": {"name": "B"}}},
        )
        _until(reconnected, lambda message: message.get("id") == 3)
        _request_with_messages(server, reconnected, 4, "thread/resume", {"threadId": "thread"})
        assert server.run_adapter._turn_connections[("thread", tid)] == "B"

        release.set()
        approval, _ = _until(reconnected, lambda message: message.get("method") == "approval/request")
        assert "A" not in server.state_manager.subscribers("thread")
        assert server.router.pending_server_request_count() == 1
        _send(
            server,
            reconnected,
            {
                "jsonrpc": "2.0",
                "id": approval["id"],
                "result": {"decision": "allow", "reason": "reconnected owner"},
            },
        )
        _until(reconnected, lambda message: message.get("method") == "turn/completed")
        assert calls == ["ran"]


def test_completed_turns_release_local_handles_and_adapter_close_clears_wait_state(tmp_path):
    calls: list[str] = []
    with ExitStack() as stack:
        server = _approval_node(stack, tmp_path / "store", "A", calls)
        request(server, 1, "thread/start", {"inputId": "thread"})
        request(server, 2, "turn/start", submission())
        _until(server.transport, lambda message: message.get("method") == "approval/request")
        server.run_adapter.join()
        assert not [handle for handle in server.kernel.handles if handle.consumer == "app_server"]

        server.run_adapter.close()
        assert server.router.pending_server_request_count() == 0
        assert not server.state_manager.subscribers("thread")
        assert not server.run_adapter._turn_connections
        assert not server.run_adapter._notification_positions
        assert server.kernel.store.read_state("thread")[0].active_turn_id is not None


def test_wake_after_kernel_close_is_a_dropped_hint(tmp_path, monkeypatch):
    with ExitStack() as stack:
        server = node(
            stack,
            tmp_path / "store",
            "A",
            FixedModelProvider(ScriptedLLM([LLMResponse("done")]), _resolved_model()),
        )
        request(server, 1, "thread/start", {"inputId": "thread"})
        server.kernel.close()
        server.run_adapter.wake("thread")
        assert server.kernel._worker.started
        assert not server.kernel._worker.running

        def unavailable(_callback):
            raise RuntimeError("observation store unavailable")

        monkeypatch.setattr(server.kernel._worker, "post_observation", unavailable)
        with pytest.raises(RuntimeError, match="observation store unavailable"):
            server.run_adapter.wake("thread")
        assert not server.kernel._worker.running


def test_due_provider_query_survives_subscription_and_local_handle_release(tmp_path):
    business = BusinessDB(tmp_path / "provider-business.sqlite")
    with SQLiteStore.standalone(str(tmp_path / "provider-session.sqlite")) as store:
        store.configure_input_admission(allow_unsigned=True)
        store.install_schema()
        with store.atomic():
            store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        provider_server = ProviderServer(store, business)
        try:
            provider_server.start()
            handle = provider_server.attempt().wait["handle"]
            assert provider_server.attempt().state == "parked"
            assert provider_server.attempt().result is None
            assert not [item for item in provider_server.server.kernel.handles if item.consumer == "app_server"]

            provider_server.server.processor.disconnect_connection("owner")
            assert not provider_server.server.run_adapter._notification_positions
            assert all(owner is None for owner in provider_server.server.run_adapter._turn_connections.values())
            assert provider_server.server.kernel._worker.running
            assert provider_server.server.kernel._worker.support.drive is not None

            business.finish()
            provider_server.due()
            _eventually(lambda: provider_server.attempt().result is not None, timeout=5)
            provider_server.join()
            assert provider_server.attempt().result is not None
            assert handle
            assert business.count("submit") == 1
            assert business.count("query") >= 1
        finally:
            provider_server.close()


def test_provider_result_invalid_warns_once_and_stays_parked_across_backoff(tmp_path):
    business = BusinessDB(tmp_path / "invalid-provider-business.sqlite")
    with SQLiteStore.standalone(str(tmp_path / "invalid-provider-session.sqlite")) as store:
        store.configure_input_admission(allow_unsigned=True)
        store.install_schema()
        with store.atomic():
            store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        provider_server = ProviderServer(store, business)
        try:
            provider_server.start()
            handle = provider_server.attempt().wait["handle"]
            business.finish()
            business.setting("invalid", "call")
            provider_server.due()
            provider_server.resume()
            _eventually(lambda: PROVIDER_THREAD in provider_server.server.kernel._errors)
            failure = provider_server.server.kernel._errors[PROVIDER_THREAD]
            _eventually(lambda: failure.warning_sent)
            provider_server.messages.extend(_drain(provider_server.transport, timeout=0.2))
            warning = next(
                message
                for message in provider_server.messages
                if message.get("method") == "error/warning" and message.get("params", {}).get("code") == "provider_result_invalid"
            )
            assert warning["params"]["code"] == "provider_result_invalid"

            query_count = business.count("query")
            for _ in range(3):
                provider_server.due()
                provider_server.join()

            provider_server.server.processor.disconnect_connection("owner")
            reconnected = provider_server.connect("reconnected")
            reconnect_response, reconnect_messages = _request_with_messages(
                provider_server.server,
                reconnected,
                4,
                "thread/resume",
                {"threadId": PROVIDER_THREAD},
            )
            assert "result" in reconnect_response
            provider_server.messages.extend(reconnect_messages)
            provider_server.messages.extend(_drain(reconnected, timeout=0.2))
            provider_server.join()

            state, records, _ = store.read_state(PROVIDER_THREAD)
            attempt = provider_server.attempt()
            assert business.count("query") == query_count
            assert business.count("submit") == 1
            assert attempt.state == "parked"
            assert attempt.wait["handle"] == handle
            assert attempt.unknown is None
            assert next(iter(state.waits.values()))["handle"] == handle
            assert any(row.record.kind == "op_parked" for row in records)
            assert not any(row.record.kind == "turn_ended" for row in records)
            assert isinstance(failure.error, ProviderResultInvalid) and failure.terminal
            assert state.next_drive_ms is not None
            assert (
                len(
                    [
                        message
                        for message in provider_server.messages
                        if message.get("method") == "error/warning"
                        and message.get("params", {}).get("code") == "provider_result_invalid"
                    ]
                )
                == 1
            )
            status = provider_server.server.run_adapter.public_thread_status(PROVIDER_THREAD)
            assert status["status"] == "provider_running"
            assert status["waitKind"] == "provider"

            business.setting("invalid", "")
            final_resume, final_messages = _request_with_messages(
                provider_server.server,
                reconnected,
                5,
                "turn/resume",
                {"threadId": PROVIDER_THREAD, "turnId": PROVIDER_TURN},
            )
            assert "result" in final_resume
            provider_server.messages.extend(final_messages)
            provider_server.join()
            _eventually(lambda: provider_server.attempt().state == "completed")
            assert business.count("submit") == 1
        finally:
            provider_server.close()


def test_first_reconnect_warns_when_future_query_invalidates_without_subscriber(tmp_path):
    business = BusinessDB(tmp_path / "detached-invalid-business.sqlite")
    with SQLiteStore.standalone(str(tmp_path / "detached-invalid-session.sqlite")) as store:
        store.configure_input_admission(allow_unsigned=True)
        store.install_schema()
        with store.atomic():
            store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        provider_server = ProviderServer(store, business)
        try:
            provider_server.start()
            handle = provider_server.attempt().wait["handle"]
            assert provider_server.attempt().state == "parked"

            provider_server.server.processor.disconnect_connection("owner")
            assert not provider_server.server.state_manager.subscribers(PROVIDER_THREAD)
            assert not provider_server.server.run_adapter._notification_positions

            business.finish()
            business.setting("invalid", "call")
            provider_server.due()
            _eventually(lambda: PROVIDER_THREAD in provider_server.server.kernel._errors)

            failure = provider_server.server.kernel._errors[PROVIDER_THREAD]
            assert isinstance(failure.error, ProviderResultInvalid) and failure.terminal
            assert failure.warning_sent is False
            assert not [
                message
                for message in provider_server.messages
                if message.get("method") == "error/warning" and message.get("params", {}).get("code") == "provider_result_invalid"
            ]
            assert business.count("submit") == business.count("effect") == business.count("query") == 1
            state, records, _ = store.read_state(PROVIDER_THREAD)
            assert state.active_turn_id == PROVIDER_TURN
            assert provider_server.attempt().wait["handle"] == handle
            assert not any(row.record.kind in {"op_unknown", "turn_ended"} for row in records)

            reconnected = provider_server.connect("reconnected")
            response, messages = _request_with_messages(
                provider_server.server,
                reconnected,
                4,
                "thread/resume",
                {"threadId": PROVIDER_THREAD},
            )
            assert "result" in response
            warning, warning_messages = _until(
                reconnected,
                lambda message: (
                    message.get("method") == "error/warning"
                    and message.get("params", {}).get("code") == "provider_result_invalid"
                ),
            )
            messages.extend(warning_messages)
            assert warning["params"]["code"] == "provider_result_invalid"
            assert failure.warning_sent is True

            query_count = business.count("query")
            provider_server.due()
            provider_server.join()
            messages.extend(_drain(reconnected, timeout=0.2))
            provider_server.server.processor.disconnect_connection("reconnected")
            reconnected_again = provider_server.connect("reconnected-again")
            response, reconnect_messages = _request_with_messages(
                provider_server.server,
                reconnected_again,
                5,
                "thread/resume",
                {"threadId": PROVIDER_THREAD},
            )
            assert "result" in response
            messages.extend(reconnect_messages)
            messages.extend(_drain(reconnected_again, timeout=0.2))
            assert business.count("query") == query_count
            assert (
                len(
                    [
                        message
                        for message in messages
                        if message.get("method") == "error/warning"
                        and message.get("params", {}).get("code") == "provider_result_invalid"
                    ]
                )
                == 1
            )

            business.setting("invalid", "")
            resumed, resume_messages = _request_with_messages(
                provider_server.server,
                reconnected_again,
                6,
                "turn/resume",
                {"threadId": PROVIDER_THREAD, "turnId": PROVIDER_TURN},
            )
            assert "result" in resumed
            messages.extend(resume_messages)
            provider_server.join()
            _eventually(lambda: provider_server.attempt().state == "completed")
            assert business.count("submit") == business.count("effect") == 1
            assert business.count("query") == 2
        finally:
            provider_server.close()


def test_observer_reconciles_invalid_provider_failure_after_remote_completion(tmp_path, monkeypatch):
    business = BusinessDB(tmp_path / "remote-complete-business.sqlite")
    session_path = tmp_path / "remote-complete-session.sqlite"
    with ExitStack() as stack:
        first_store = stack.enter_context(SQLiteStore.standalone(str(session_path)))
        first_store.configure_input_admission(allow_unsigned=True)
        first_store.install_schema()
        with first_store.atomic():
            first_store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        second_store = stack.enter_context(SQLiteStore.standalone(str(session_path)))
        second_store.configure_input_admission(allow_unsigned=True)
        first = ProviderServer(first_store, business)
        second = ProviderServer(second_store, business)
        s1_jobs = _S1JobsProvider(business)
        original_runtime = second.server.kernel.runtime

        def runtime(*args, **kwargs):
            value = original_runtime(*args, **kwargs)
            value.providers["work"] = s1_jobs
            return value

        monkeypatch.setattr(second.server.kernel, "runtime", runtime)
        stack.callback(first.close)
        stack.callback(second.close)

        first.start()
        first.server.processor.disconnect_connection("owner")
        assert not first.server.state_manager.subscribers(PROVIDER_THREAD)
        assert not first.server.run_adapter._notification_positions

        business.finish()
        business.setting("invalid", "call")
        first.due()
        _eventually(lambda: PROVIDER_THREAD in first.server.kernel._errors)
        failure = first.server.kernel._errors[PROVIDER_THREAD]
        assert isinstance(failure.error, ProviderResultInvalid) and failure.terminal
        assert failure.warning_sent is False
        first.server.kernel._worker.support = replace(
            first.server.kernel._worker.support,
            drive=None,
            project=None,
            projection_consumers=(),
        )

        observer = second.connect("observer")
        response, _messages = _request_with_messages(
            second.server,
            observer,
            1,
            "thread/resume",
            {"threadId": PROVIDER_THREAD},
        )
        assert "result" in response
        business.setting("invalid", "")
        query_count_before_result = business.count("query")
        callback = _callback_for_business_job(business, input_id="remote-completed-result")
        _tx_push_provider_result(second, callback)
        second.due()
        second.resume(transport=observer)
        _eventually(lambda: second.state().turns[PROVIDER_TURN].ended)
        applied = second_store.read_state(PROVIDER_THREAD)[0].applied_inputs[callback.input_id].payload
        assert applied["disposition"] == "applied"
        assert business.count("query") == query_count_before_result
        assert second.attempt().state == "completed"
        assert second.terminal()["status"] == "completed"
        query_count = business.count("query")

        reconnected = first.connect("reconnected")
        resumed, messages = _request_with_messages(
            first.server,
            reconnected,
            2,
            "thread/resume",
            {"threadId": PROVIDER_THREAD},
        )
        assert resumed["result"]["thread"]["status"] == "terminal"
        first.join()
        messages.extend(_drain(reconnected, timeout=0.2))
        retained, retained_messages = _request_with_messages(
            first.server,
            reconnected,
            3,
            "turn/resume",
            {"threadId": PROVIDER_THREAD, "turnId": PROVIDER_TURN},
        )
        messages.extend(retained_messages)
        assert retained["result"]["status"] == "completed"
        assert business.count("query") == query_count
        assert first.server.kernel._errors.get(PROVIDER_THREAD) is None
        assert not [
            message
            for message in messages
            if message.get("method") == "error/warning" and message.get("params", {}).get("code") == "provider_result_invalid"
        ]


def test_recover_attaches_committed_pending_input_before_worker_claim(tmp_path, monkeypatch):
    seen = []
    provider = ScriptedModelProvider.from_callback("test", "m", lambda r: seen.append(r) or LLMResponse("done"))
    with ExitStack() as stack:
        a = node(stack, tmp_path / "attach-store", "A", provider)
        b = node(stack, tmp_path / "attach-store", "B", provider)
        request(a, "thread", "thread/start", {"inputId": "thread"})

        entered = {"A": threading.Event(), "B": threading.Event()}
        release = {"A": threading.Event(), "B": threading.Event()}
        stack.callback(release["A"].set)
        stack.callback(release["B"].set)
        original_scans = {
            "A": a.kernel._worker._periodic_scan,
            "B": b.kernel._worker._periodic_scan,
        }

        def gate_scan(connection, store):
            entered[connection].set()
            release[connection].wait()
            return original_scans[connection](store)

        monkeypatch.setattr(a.kernel._worker, "_periodic_scan", lambda store: gate_scan("A", store))
        monkeypatch.setattr(b.kernel._worker, "_periodic_scan", lambda store: gate_scan("B", store))
        receipt, _ = a.run_adapter.submit_input(
            connection_id="A",
            thread_id="thread",
            input_id="prompt",
            expected_turn_id=None,
            kind="start",
            input=[{"type": "text", "text": "preserve this exact input"}],
        )
        assert entered["A"].wait(1), "node A never reached the admission barrier"
        assert any(entry.item.input_id == "prompt" for entry in a.kernel.store.peek_inbox("thread"))
        assert seen == []
        before = b.kernel.store.read_input("thread", "prompt").item.to_dict()
        assert before["actor"] is not None
        a.processor.disconnect_connection("A")

        try:
            response, messages = _request_with_messages(
                b,
                b.transport,
                "replay",
                "turn/start",
                submission(identity="prompt", text="preserve this exact input"),
            )
            try:
                assert "result" in response
                assert response["result"] == receipt | {"replayed": True}
                assert entered["B"].wait(1), "node B never reached the admission barrier"
                assert not any(
                    message.get("method") == "error/warning" and message.get("params", {}).get("code") == "drive_start"
                    for message in messages
                ), messages
                after = b.kernel.store.read_input("thread", "prompt").item.to_dict()
                assert after == before
                assert any(entry.item.input_id == "prompt" for entry in b.kernel.store.peek_inbox("thread"))
                active = b.state_manager.active_turn("thread")
                assert active is not None and active.turn_id == response["result"]["turnId"]
                assert seen == []
            finally:
                release["B"].set()
            b.kernel._worker.wake("thread")
            b.run_adapter.join()
            _eventually(lambda: store_is_terminal(b.kernel.store))
            assert len(seen) == 1
        finally:
            release["A"].set()
            release["B"].set()


def store_is_terminal(store) -> bool:
    state = store.read_state("thread")[0]
    turn = state.turns.get("thread/turn/prompt")
    return state.active_turn_id is None and turn is not None and turn.ended


def test_standard_tx_valid_provider_result_advances_invalid_fence_after_wake(tmp_path, monkeypatch):
    business = BusinessDB(tmp_path / "standard-tx-business.sqlite")
    with SQLiteStore.standalone(str(tmp_path / "standard-tx-session.sqlite")) as store:
        store.configure_input_admission(allow_unsigned=True)
        store.install_schema()
        with store.atomic():
            store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        provider_server = ProviderServer(store, business)
        s1_jobs = _S1JobsProvider(business)
        original_runtime = provider_server.server.kernel.runtime

        def runtime(*args, **kwargs):
            value = original_runtime(*args, **kwargs)
            value.providers["work"] = s1_jobs
            return value

        monkeypatch.setattr(provider_server.server.kernel, "runtime", runtime)
        try:
            _handle, failure = _start_invalid_provider_wait(provider_server, business)
            business.setting("invalid", "")
            callback = _callback_for_business_job(business, input_id="trusted-standard-result")
            _tx_push_provider_result(provider_server, callback)
            provider_server.join()
            _eventually(lambda: store.read_state(PROVIDER_THREAD)[0].turns[PROVIDER_TURN].ended)

            state, _, _ = store.read_state(PROVIDER_THREAD)
            assert state.active_turn_id is None
            assert provider_server.terminal()["status"] == "completed"
            assert provider_server.attempt().state == "completed"
            assert provider_server.attempt().result is not None
            applied = state.applied_inputs[callback.input_id].payload
            assert applied["disposition"] == "applied"
            _eventually(lambda: provider_server.server.kernel._errors.get(PROVIDER_THREAD) is not failure)
            assert business.count("query") == 1
            assert business.count("submit") == business.count("effect") == 1
            assert _provider_invalid_warning_count(provider_server.messages) == 1
        finally:
            provider_server.close()


def test_declared_yield_provider_steer_continues_parent_without_query(tmp_path, monkeypatch):
    business = BusinessDB(tmp_path / "declared-yield-business.sqlite")
    with SQLiteStore.standalone(str(tmp_path / "declared-yield-session.sqlite")) as store:
        store.configure_input_admission(allow_unsigned=True)
        store.install_schema()
        with store.atomic():
            store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        provider_server = ProviderServer(store, business)
        declared = _YieldOnSteerJobsProvider(business)
        original_runtime = provider_server.server.kernel.runtime

        def runtime(*args, **kwargs):
            value = original_runtime(*args, **kwargs)
            value.providers["work"] = declared
            return value

        monkeypatch.setattr(provider_server.server.kernel, "runtime", runtime)
        try:
            _start_invalid_provider_wait(provider_server, business)
            result = provider_server.rpc(
                "turn/steer",
                {
                    "threadId": PROVIDER_THREAD,
                    "inputId": "steer",
                    "expectedTurnId": PROVIDER_TURN,
                    "input": [{"type": "text", "text": "use the yielded observation"}],
                },
            )
            assert "result" in result
            provider_server.join()
            _eventually(lambda: store.read_state(PROVIDER_THREAD)[0].turns[PROVIDER_TURN].ended)

            state, records, _ = store.read_state(PROVIDER_THREAD)
            assert state.active_turn_id is None
            assert provider_server.terminal()["status"] == "completed"
            assert any(row.record.kind == "op_completed" and row.record.payload.get("yielded_by") == "steer" for row in records)
            assert business.count("query") == 1
            assert business.count("submit") == business.count("effect") == 1
            _eventually(lambda: provider_server.server.kernel._errors == {})
            assert _provider_invalid_warning_count(provider_server.messages) == 1
        finally:
            provider_server.close()


def test_archive_and_forged_callback_keep_invalid_failure_without_query(tmp_path):
    business = BusinessDB(tmp_path / "archive-forged-business.sqlite")
    with SQLiteStore.standalone(str(tmp_path / "archive-forged-session.sqlite")) as store:
        store.configure_input_admission(allow_unsigned=True)
        store.install_schema()
        with store.atomic():
            store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        provider_server = ProviderServer(store, business)
        try:
            handle, failure = _start_invalid_provider_wait(provider_server, business)
            archived = provider_server.rpc("thread/archive", {"threadId": PROVIDER_THREAD})
            assert "result" in archived
            provider_server.join()
            _eventually(lambda: store.read_state(PROVIDER_THREAD)[0].archived)
            state, _, _ = store.read_state(PROVIDER_THREAD)
            assert provider_server.server.kernel._errors[PROVIDER_THREAD] is failure
            assert failure.warning_sent is True
            assert provider_server.attempt().wait["handle"] == handle
            assert business.count("query") == 1
            assert _provider_invalid_warning_count(provider_server.messages) == 1

            callback = _callback_for_business_job(business, input_id="forged-matching-result")
            forged = replace(
                callback,
                payload=callback.payload | {"result": callback.payload["result"] | {"content": "forged business result"}},
            )
            _tx_push_provider_result(provider_server, forged)
            provider_server.join()
            _eventually(lambda: forged.input_id in store.read_state(PROVIDER_THREAD)[0].applied_inputs)
            state, _, _ = store.read_state(PROVIDER_THREAD)
            assert state.applied_inputs[forged.input_id].payload["disposition"] == "rejected"
            assert provider_server.server.kernel._errors[PROVIDER_THREAD] is failure
            assert failure.warning_sent is True
            assert provider_server.attempt().wait["handle"] == handle
            assert business.count("query") == 1
            assert _provider_invalid_warning_count(provider_server.messages) == 1
        finally:
            provider_server.close()


@pytest.mark.parametrize("recovery", ["resume", "cancel"])
def test_provider_invalid_fence_survives_unrelated_rpc_inputs_before_recovery(tmp_path, recovery):
    business = BusinessDB(tmp_path / f"held-invalid-business-{recovery}.sqlite")
    if recovery == "cancel":
        business.setting("cancel", "confirmed")
    with SQLiteStore.standalone(str(tmp_path / f"held-invalid-session-{recovery}.sqlite")) as store:
        store.configure_input_admission(allow_unsigned=True)
        store.install_schema()
        with store.atomic():
            store._rows("UPDATE sk_settings SET write_level=2 WHERE singleton=1")
        provider_server = ProviderServer(store, business)
        try:
            kernel = provider_server.server.kernel
            kernel._next_session_requirements = static_session_requirements(
                provider_server.host._agent,
                provider_server.host._run_config,
                handler_version="1",
            )
            kernel.create(
                PROVIDER_THREAD,
                "/fixture",
                attributes={"app_server": {"agent_key": "default", "cwd": None, "metadata": {}}},
                principal={
                    "tenant_id": kernel.tenant_id,
                    "subject_id": "app-server",
                    "subject_kind": "service",
                    "attributes": {"role": "app-server"},
                },
            )
            provider_server.start()
            handle = provider_server.attempt().wait["handle"]
            business.finish()
            business.setting("invalid", "call")
            provider_server.due()
            provider_server.resume()
            if not any(
                message.get("method") == "error/warning" and message.get("params", {}).get("code") == "provider_result_invalid"
                for message in provider_server.messages
            ):
                _, warning_messages = _until(
                    provider_server.transport,
                    lambda message: (
                        message.get("method") == "error/warning"
                        and message.get("params", {}).get("code") == "provider_result_invalid"
                    ),
                )
                provider_server.messages.extend(warning_messages)

            failure = provider_server.server.kernel._errors[PROVIDER_THREAD]
            assert failure.warning_sent is True
            assert business.count("submit") == business.count("effect") == business.count("query") == 1
            selection = {"threadId": PROVIDER_THREAD, "inputId": "select", "selection": {"model": "m", "route": {}}}
            selected = provider_server.rpc("thread/model/select", selection)
            replayed = provider_server.rpc("thread/model/select", selection)
            assert "result" in selected and replayed["result"] == selected["result"]

            steered = provider_server.rpc(
                "turn/steer",
                {
                    "threadId": PROVIDER_THREAD,
                    "inputId": "steer",
                    "expectedTurnId": PROVIDER_TURN,
                    "input": [{"type": "text", "text": "keep the current provider wait"}],
                },
            )
            assert "result" in steered
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline and business.count("query") == 1:
                provider_server.due()
                provider_server.server.wake(PROVIDER_THREAD)
                provider_server.join()
                provider_server.messages.extend(_drain(provider_server.transport, timeout=0.05))
                time.sleep(0.05)

            assert provider_server.server.kernel._errors[PROVIDER_THREAD] is failure
            assert failure.warning_sent is True
            assert business.count("query") == 1
            assert (
                len(
                    [
                        message
                        for message in provider_server.messages
                        if message.get("method") == "error/warning"
                        and message.get("params", {}).get("code") == "provider_result_invalid"
                    ]
                )
                == 1
            )
            assert provider_server.attempt().wait["handle"] == handle
            assert not any(row.record.kind in {"op_unknown", "turn_ended"} for row in store.read_state(PROVIDER_THREAD)[1])

            business.setting("invalid", "")
            if recovery == "resume":
                resumed, _ = _request_with_messages(
                    provider_server.server,
                    provider_server.transport,
                    7,
                    "turn/resume",
                    {"threadId": PROVIDER_THREAD, "turnId": PROVIDER_TURN},
                )
                assert "result" in resumed
                provider_server.join()
                _eventually(
                    lambda: (
                        provider_server.attempt().state == "completed"
                        and (recovered_state := store.read_state(PROVIDER_THREAD)[0]).active_turn_id is None
                        and recovered_state.turns[PROVIDER_TURN].ended
                    )
                )
                assert provider_server.terminal()["status"] == "completed"
            else:
                interrupted = provider_server.rpc(
                    "turn/interrupt",
                    {"threadId": PROVIDER_THREAD, "expectedTurnId": PROVIDER_TURN, "actionId": "cancel"},
                )
                assert "result" in interrupted
                provider_server.join()
                _eventually(lambda: store.read_state(PROVIDER_THREAD)[0].turns[PROVIDER_TURN].ended)
                assert provider_server.terminal()["status"] == "cancelled"
                assert business.count("cancel") == 1
            assert business.count("submit") == business.count("effect") == 1
        finally:
            provider_server.close()
