from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from session.app_server_s1_helpers import bind_development_content, hydrate_stored_payload
from support import FixedModelProvider

from vv_agent import Agent, RunConfig, function_tool
from vv_agent.app_server import (
    AppServer,
    ChannelTransport,
    JsonRpcResponse,
    MessageProcessor,
    OutgoingRouter,
    RequestId,
)
from vv_agent.app_server.host import AppServerApprovalProvider, DefaultAppServerHost, TrustedPrincipal
from vv_agent.approval import ApprovalRequest
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.kernel._content import INPUT_ACTOR_ATTRIBUTES, ContentRef
from vv_agent.kernel._sql import stored_input_proof
from vv_agent.llm import ScriptedLLM
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.timing import SystemClock
from vv_agent.types import LLMResponse, ToolCall

PRINCIPAL: dict[str, Any] = {
    "tenant_id": "tenant-a",
    "subject_id": "trusted-approval-user",
    "subject_kind": "user",
    "attributes": {"role": "approver"},
}
FORGED_PRINCIPAL = {
    "tenant_id": "tenant-evil",
    "subject_id": "forged-client",
    "subject_kind": "user",
    "attributes": {"role": "admin"},
}
FOREIGN_PRINCIPAL = {
    "tenant_id": "tenant-b",
    "subject_id": "foreign-approver",
    "subject_kind": "user",
    "attributes": {"role": "approver"},
}


class _Signer:
    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.key_id = "approval-response-test-key"

    def bind_key(self):
        signer = self

        class Bound:
            key_id = signer.key_id

            @staticmethod
            def sign(message: bytes) -> bytes:
                return signer.key.sign(message)

        return Bound()

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        if key_id != self.key_id:
            return False
        try:
            self.key.public_key().verify(signature, message)
        except InvalidSignature:
            return False
        return True


class _MutableClock(SystemClock):
    def __init__(self) -> None:
        super().__init__()
        self.offset_ms = 0

    def now_ms(self) -> int:
        return super().now_ms() + self.offset_ms

    def jump_to(self, now_ms: int) -> None:
        self.offset_ms = now_ms - super().now_ms()


class _TrustedHost(DefaultAppServerHost):
    def __init__(
        self,
        *,
        allow_approval_resolve: bool = True,
        principal: dict[str, Any] = PRINCIPAL,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.allow_approval_resolve = allow_approval_resolve
        self.principal = dict(principal)
        self.authorized: list[tuple[str, str, dict[str, Any]]] = []
        self.resolved: list[tuple[str, str, dict[str, Any]]] = []

    def authorize(self, connection_id: str, method: str, params: dict[str, Any]) -> bool:
        self.authorized.append((connection_id, method, dict(params)))
        return method != "approval/resolve" or self.allow_approval_resolve

    def resolve_principal(
        self,
        connection_id: str,
        method: str,
        params: dict[str, Any],
    ) -> TrustedPrincipal | None:
        self.resolved.append((connection_id, method, dict(params)))
        return dict(self.principal)


def _resolved_model() -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="approval", api_key="test-key", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model="test-model",
        selected_model="test-model",
        model_id="test-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
    )


def _app_server(
    tmp_path: Path,
    calls: list[str],
    *,
    clock: _MutableClock | None = None,
    approval_timeout_seconds: float | None = None,
    allow_approval_resolve: bool = True,
    principal: dict[str, Any] = PRINCIPAL,
) -> tuple[AppServer, ChannelTransport, SQLiteStore, _TrustedHost, _Signer]:
    @function_tool(needs_approval=True)
    def dangerous_tool() -> str:
        calls.append("ran")
        return "allowed"

    llm = ScriptedLLM(
        steps=[
            LLMResponse(content="calling", tool_calls=[ToolCall(id="call_1", name="dangerous_tool", arguments={})]),
            LLMResponse(content="done"),
        ]
    )
    host = _TrustedHost(
        allow_approval_resolve=allow_approval_resolve,
        principal=principal,
        agent=Agent(name="assistant", instructions="Use the tool.", model="test-model", tools=[dangerous_tool]),
        run_config=RunConfig(
            model_provider=FixedModelProvider(llm, _resolved_model()),
            max_cycles=3,
            approval_timeout_seconds=approval_timeout_seconds,
        ),
    )
    root_store = SQLiteStore(str(tmp_path / "approval-response.sqlite"), clock=clock)
    root_store.install_schema()
    signer = _Signer()
    root_store.configure_input_admission(signer)
    bind_development_content(root_store)
    store = root_store.for_tenant(PRINCIPAL["tenant_id"])
    transport = ChannelTransport(connection_id="tenant-a")
    return AppServer(host=host, store=store, transport=transport), transport, root_store, host, signer


def _close(server: AppServer, root_store: SQLiteStore) -> None:
    for handle in list(server.kernel.handles):
        handle.cancel("test teardown")
    server.router.cancel_matching_server_requests()
    server.run_adapter.join()
    server.kernel.close()
    root_store.connection.close()


def _request(
    server: AppServer,
    transport: ChannelTransport,
    request_id: int,
    method: str,
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    transport.send_inbound(payload)
    server.processor.process_next(transport)
    while True:
        message = transport.receive_outbound(timeout=5)
        if message.get("id") == request_id:
            return message


def _initialize(server: AppServer, transport: ChannelTransport) -> None:
    response = _request(
        server,
        transport,
        0,
        "initialize",
        {"clientInfo": {"name": transport.connection_id}},
    )
    assert "result" in response
    transport.send_inbound({"jsonrpc": "2.0", "method": "initialized"})
    server.processor.process_next(transport)


def _start_approval(server: AppServer, transport: ChannelTransport) -> tuple[str, str, dict[str, Any]]:
    _initialize(server, transport)
    thread_id = _request(
        server,
        transport,
        1,
        "thread/start",
        {"inputId": "thread", "agentKey": "default", "cwd": "."},
    )["result"]["threadId"]
    turn_id = _request(
        server,
        transport,
        2,
        "turn/start",
        {
            "threadId": thread_id,
            "inputId": "prompt",
            "expectedTurnId": None,
            "input": [{"type": "text", "text": "run the tool"}],
        },
    )["result"]["turnId"]
    approval = _drain_until(transport, lambda message: message.get("method") == "approval/request")
    return cast(str, thread_id), cast(str, turn_id), approval


def _send(server: AppServer, transport: ChannelTransport, payload: dict[str, Any]) -> None:
    transport.send_inbound(payload)
    server.processor.process_next(transport)


def _drain_until(
    transport: ChannelTransport,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout: float = 10,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = transport.receive_outbound(timeout=max(0.0, deadline - time.monotonic()))
        if predicate(message):
            return message
    raise AssertionError("Timed out waiting for App Server message")


def _drain_until_turn_completed(transport: ChannelTransport) -> list[dict[str, Any]]:
    outbound: list[dict[str, Any]] = []
    deadline = time.monotonic() + 10
    while True:
        message = transport.receive_outbound(timeout=max(0.0, deadline - time.monotonic()))
        outbound.append(message)
        if message.get("method") == "turn/completed":
            return outbound


def _drain_available(transport: ChannelTransport) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    while True:
        try:
            messages.append(transport.receive_outbound(timeout=0))
        except queue.Empty:
            return messages


def _wait_for_pending(router: OutgoingRouter, request_id: RequestId):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        pending = router.get_pending_server_request(request_id)
        if pending is not None:
            return pending
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for server request {request_id.to_wire()}")


def _assert_trusted_input_writer(server: AppServer, thread_id: str, input_id: str, signer: _Signer) -> None:
    stored = server.kernel.store.read_input(thread_id, input_id)
    assert stored is not None
    actor = stored.item.actor
    assert actor is not None
    assert {key: actor[key] for key in ("tenant_id", "subject_id", "subject_kind")} == {
        key: PRINCIPAL[key] for key in ("tenant_id", "subject_id", "subject_kind")
    }
    assert stored.item.admission_proof is not None
    assert stored.item.admission_proof["admission_key_id"] == signer.key_id
    assert stored_input_proof(stored) is not None

    with server.kernel.store.atomic() as tx:
        writer = SessionContentWriter.bind(
            tx,
            tenant_id=PRINCIPAL["tenant_id"],
            session_id=thread_id,
        )
        ref = ContentRef.from_dict(actor["attributes"])
        assert writer.assert_input(input_id, ref, INPUT_ACTOR_ATTRIBUTES) == ref

    attributes = server.kernel.read_input_content(
        thread_id,
        input_id,
        actor["attributes"],
        slot=INPUT_ACTOR_ATTRIBUTES,
    )
    assert attributes == PRINCIPAL["attributes"]


def _approval_input_record(server: AppServer, thread_id: str, input_id: str) -> dict[str, Any]:
    _state, records, _ = server.kernel.store.read_state(thread_id)
    matches = [
        hydrate_stored_payload(server.kernel.store, thread_id, stored)
        for stored in records
        if stored.record.kind == "input_applied" and stored.record._payload["input"]["input_id"] == input_id
    ]
    assert len(matches) == 1
    return matches[0]


def test_standard_approval_response_writes_trusted_actor_once_and_replays_explicitly(tmp_path) -> None:
    calls: list[str] = []
    server, transport, root_store, host, signer = _app_server(tmp_path, calls)
    try:
        thread_id, turn_id, approval = _start_approval(server, transport)
        params = cast(dict[str, Any], approval["params"])
        assert params["turnId"] == turn_id
        request_id = params["requestId"]
        input_id = f"approval/{request_id}/answer"
        result = {
            "decision": "allow",
            "reason": "approved by connection principal",
            "metadata": {"ticket": 7, "actor": FORGED_PRINCIPAL, "approver": FORGED_PRINCIPAL},
        }

        _send(
            server,
            transport,
            {"jsonrpc": "2.0", "id": approval["id"], "result": result},
        )
        outbound = _drain_until_turn_completed(transport)

        assert calls == ["ran"]
        assert any(message.get("method") == "approval/resolved" for message in outbound)
        assert [method for _, method, _ in host.resolved].count("approval/resolve") == 1
        assert [method for _, method, _ in host.authorized].count("approval/resolve") == 1
        assert server.router.get_pending_server_request(RequestId(request_id)) is None
        _assert_trusted_input_writer(server, thread_id, input_id, signer)

        answer = _approval_input_record(server, thread_id, input_id)
        assert answer["input"]["payload"]["decision"] == "approve"
        assert answer["input"]["payload"]["reason"] == result["reason"]
        assert answer["input"]["payload"]["metadata"] == result["metadata"]
        first_input = server.kernel.store.read_input(thread_id, input_id)
        assert first_input is not None
        first_seq = first_input.input_seq

        _drain_available(transport)
        _send(
            server,
            transport,
            {"jsonrpc": "2.0", "id": approval["id"], "result": result},
        )
        duplicate_outbound = _drain_available(transport)
        duplicate_input = server.kernel.store.read_input(thread_id, input_id)
        assert duplicate_input is not None
        assert duplicate_input.input_seq == first_seq
        _approval_input_record(server, thread_id, input_id)
        assert calls == ["ran"]
        assert not any(message.get("id") == approval["id"] for message in duplicate_outbound)

        replay = _request(
            server,
            transport,
            3,
            "approval/resolve",
            {
                "threadId": thread_id,
                "turnId": turn_id,
                "requestId": request_id,
                **result,
            },
        )

        assert replay["result"] == {}
        second_input = server.kernel.store.read_input(thread_id, input_id)
        assert second_input is not None
        assert second_input.input_seq == first_seq
        assert second_input.item.actor == first_input.item.actor
        assert calls == ["ran"]
        to_dict = answer["input"]
        assert to_dict["input_id"] == input_id
    finally:
        _close(server, root_store)


@pytest.mark.parametrize("failure", ["authorize", "tenant"])
def test_approval_response_host_denial_keeps_same_pending_and_has_no_side_effect(tmp_path, failure: str) -> None:
    calls: list[str] = []
    server, transport, root_store, host, _ = _app_server(
        tmp_path,
        calls,
        allow_approval_resolve=failure != "authorize",
    )
    try:
        thread_id, _, approval = _start_approval(server, transport)
        params = cast(dict[str, Any], approval["params"])
        request_id = cast(str, params["requestId"])
        pending = server.router.get_pending_server_request(RequestId(request_id))
        assert pending is not None
        if failure == "tenant":
            host.principal = dict(FOREIGN_PRINCIPAL)

        _send(
            server,
            transport,
            {
                "jsonrpc": "2.0",
                "id": approval["id"],
                "result": {"decision": "allow", "reason": "must not apply"},
            },
        )

        assert server.router.get_pending_server_request(RequestId(request_id)) is pending
        assert server.kernel.store.read_input(thread_id, f"approval/{request_id}/answer") is None
        assert calls == []
        assert not any(message.get("id") == approval["id"] for message in _drain_available(transport))
    finally:
        _close(server, root_store)


def test_invalid_approval_response_keeps_pending_request_recoverable(tmp_path) -> None:
    calls: list[str] = []
    server, transport, root_store, _, _ = _app_server(tmp_path, calls)
    try:
        _, _, approval = _start_approval(server, transport)
        approval_id = approval["id"]
        request_id = cast(dict[str, Any], approval["params"])["requestId"]
        pending = server.router.get_pending_server_request(RequestId(request_id))
        assert pending is not None

        invalid_results: list[Any] = [
            True,
            {"decision": "approve"},
            {"decision": "allow", "reason": "ok", "metadata": {}, "actor": FORGED_PRINCIPAL},
            {"decision": "allow", "reason": "ok", "metadata": []},
            [],
        ]
        for invalid_result in invalid_results:
            _drain_available(transport)
            _send(
                server,
                transport,
                {"jsonrpc": "2.0", "id": approval_id, "result": invalid_result},
            )
            assert server.router.get_pending_server_request(RequestId(request_id)) is pending
            assert calls == []
            assert not any(message.get("id") == approval_id for message in _drain_available(transport))

        _send(
            server,
            transport,
            {
                "jsonrpc": "2.0",
                "id": approval_id,
                "result": {"decision": "allow", "reason": "recovered"},
            },
        )
        _drain_until_turn_completed(transport)
        assert calls == ["ran"]
    finally:
        _close(server, root_store)


def test_wrong_connection_unknown_and_nonapproval_responses_keep_future_semantics() -> None:
    owner = ChannelTransport(connection_id="owner")
    attacker = ChannelTransport(connection_id="attacker")
    router = OutgoingRouter()
    router.register_transport(owner)
    router.register_transport(attacker)
    processor = MessageProcessor(router=router)
    approval = router.send_server_request(
        "owner",
        "approval/request",
        {
            "threadId": "thread_1",
            "turnId": "thread_1/turn/prompt",
            "requestId": "approval_1",
        },
        request_id=RequestId("approval_1"),
    )
    assert router.get_pending_server_request(RequestId("approval_1")) is approval
    _drain_available(owner)

    processor.process_message(
        "attacker",
        {"jsonrpc": "2.0", "id": "approval_1", "result": {"decision": "allow"}},
    )
    assert router.get_pending_server_request(RequestId("approval_1")) is approval

    processor.process_message(
        "owner",
        {"jsonrpc": "2.0", "id": "unknown", "result": {"decision": "allow"}},
    )
    assert router.get_pending_server_request(RequestId("approval_1")) is approval

    ordinary = router.send_server_request(
        "owner",
        "custom/request",
        {"value": 1},
        request_id=RequestId("ordinary_1"),
    )
    processor.process_message(
        "owner",
        {"jsonrpc": "2.0", "id": "ordinary_1", "result": {"ok": True}},
    )
    assert ordinary.result(timeout=0) == {"ok": True}
    assert router.get_pending_server_request(RequestId("ordinary_1")) is None
    assert router.get_pending_server_request(RequestId("approval_1")) is approval


def test_invalid_response_wire_shape_gets_no_protocol_response() -> None:
    owner = ChannelTransport(connection_id="owner")
    router = OutgoingRouter()
    router.register_transport(owner)
    processor = MessageProcessor(router=router)
    pending = router.send_server_request(
        "owner",
        "approval/request",
        {
            "threadId": "thread_1",
            "turnId": "thread_1/turn/prompt",
            "requestId": "approval_1",
        },
        request_id=RequestId("approval_1"),
    )
    _drain_available(owner)

    processor.process_message(
        "owner",
        {
            "jsonrpc": "2.0",
            "id": "approval_1",
            "result": {"decision": "allow"},
            "extra": True,
        },
    )
    processor.process_message(
        "owner",
        {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32603, "message": "unknown response"},
        },
    )

    assert router.get_pending_server_request(RequestId("approval_1")) is pending
    assert _drain_available(owner) == []


def test_jsonrpc_error_resolves_the_future_without_fabricating_a_decision() -> None:
    owner = ChannelTransport(connection_id="owner")
    router = OutgoingRouter()
    router.register_transport(owner)
    processor = MessageProcessor(router=router)
    pending = router.send_server_request(
        "owner",
        "approval/request",
        {
            "threadId": "thread_1",
            "turnId": "thread_1/turn/prompt",
            "requestId": "approval_1",
        },
        request_id=RequestId("approval_1"),
    )
    _drain_available(owner)

    processor.process_message(
        "owner",
        {
            "jsonrpc": "2.0",
            "id": "approval_1",
            "error": {"code": -32603, "message": "client cannot decide"},
        },
    )

    assert router.get_pending_server_request(RequestId("approval_1")) is None
    with pytest.raises(RuntimeError, match="client cannot decide"):
        pending.result(timeout=0)
    assert _drain_available(owner) == []


def test_explicit_host_approval_provider_uses_actual_run_id() -> None:
    transport = ChannelTransport(connection_id="owner")
    router = OutgoingRouter()
    router.register_transport(transport)
    provider = AppServerApprovalProvider(
        connection_id="owner",
        thread_id="thread",
        turn_id="thread/turn/stale",
        router=router,
        timeout_seconds=5,
        has_durable_answer=lambda request: False,
    )
    request = ApprovalRequest(
        request_id="approval_1",
        tool_name="dangerous",
        tool_call_id="call_1",
        run_id="thread/turn/actual",
    )
    outcome: list[Any] = []

    worker = threading.Thread(target=lambda: outcome.append(provider.decide(request)))
    worker.start()
    try:
        server_request = _drain_until(transport, lambda message: message.get("method") == "approval/request")
        assert server_request["params"]["turnId"] == request.run_id
        router.resolve_response(
            "owner",
            JsonRpcResponse(
                id=RequestId(request.request_id),
                result={"decision": "allow", "reason": "approved"},
            ),
        )
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert outcome[0] is not None
        assert outcome[0].action == "allow"
    finally:
        router.cancel_server_request(RequestId(request.request_id))
        worker.join(timeout=2)


def test_wrong_turn_response_leaves_pending_and_does_not_run_tool(tmp_path) -> None:
    calls: list[str] = []
    server, transport, root_store, _, _ = _app_server(tmp_path, calls)
    try:
        thread_id, turn_id, approval = _start_approval(server, transport)
        params = dict(cast(dict[str, Any], approval["params"]))
        request_id = cast(str, params["requestId"])
        pending = server.router.get_pending_server_request(RequestId(request_id))
        assert pending is not None
        assert server.router.cancel_server_request(RequestId(request_id))

        wrong_turn = params | {"turnId": f"{thread_id}/turn/other"}
        pending = server.router.send_server_request(
            transport.connection_id,
            "approval/request",
            wrong_turn,
            request_id=RequestId(request_id),
        )
        _drain_available(transport)
        _send(
            server,
            transport,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"decision": "allow"},
            },
        )

        assert server.router.get_pending_server_request(RequestId(request_id)) is pending
        assert calls == []
        assert server.kernel.store.read_input(thread_id, f"approval/{request_id}/answer") is None

        assert server.router.cancel_server_request(RequestId(request_id))
        unknown_request_id = "approval/unknown-request"
        pending = server.router.send_server_request(
            transport.connection_id,
            "approval/request",
            params | {"requestId": unknown_request_id},
            request_id=RequestId(unknown_request_id),
        )
        _drain_available(transport)
        _send(
            server,
            transport,
            {
                "jsonrpc": "2.0",
                "id": unknown_request_id,
                "result": {"decision": "allow"},
            },
        )

        assert server.router.get_pending_server_request(RequestId(unknown_request_id)) is pending
        assert calls == []
        assert server.kernel.store.read_input(thread_id, f"approval/{unknown_request_id}/answer") is None

        assert server.router.cancel_server_request(RequestId(unknown_request_id))
        server.wake(thread_id)
        pending = _wait_for_pending(server.router, RequestId(request_id))
        assert pending.params is not None
        assert pending.params["turnId"] == turn_id
        _drain_available(transport)
        _send(
            server,
            transport,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"decision": "allow"},
            },
        )
        _drain_until_turn_completed(transport)
        assert server.router.get_pending_server_request(RequestId(request_id)) is None
        assert calls == ["ran"]
    finally:
        _close(server, root_store)


def test_expired_approval_response_is_rejected_without_running_tool(tmp_path) -> None:
    calls: list[str] = []
    clock = _MutableClock()
    server, transport, root_store, _, _ = _app_server(
        tmp_path,
        calls,
        clock=clock,
        approval_timeout_seconds=60,
    )
    try:
        thread_id, _, approval = _start_approval(server, transport)
        params = cast(dict[str, Any], approval["params"])
        request_id = cast(str, params["requestId"])
        state, _, _ = server.kernel.store.read_state(thread_id)
        deadline = next(
            wait["deadline_ms"]
            for wait in state.waits.values()
            if wait["handle"]["kind"] == "approval" and wait["handle"]["request_id"] == request_id
        )
        assert isinstance(deadline, int)
        clock.jump_to(deadline)

        _send(
            server,
            transport,
            {
                "jsonrpc": "2.0",
                "id": approval["id"],
                "result": {"decision": "allow", "reason": "too late"},
            },
        )
        _drain_until_turn_completed(transport)

        input_id = f"approval/{request_id}/answer"
        applied = _approval_input_record(server, thread_id, input_id)
        assert applied["disposition"] == "rejected"
        assert applied["reason"] == "approval deadline expired"
        assert calls == []
    finally:
        _close(server, root_store)
