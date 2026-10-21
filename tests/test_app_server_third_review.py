"""Lease takeover and disconnected recovery through real App Server producers."""

import time
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
from test_app_server_distributed import node, request, submission
from test_app_server_review_regressions import record_output
from test_app_server_second_review import eventually, provider

from vv_agent import Agent, ScriptedModelProvider, function_tool
from vv_agent.app_server import ChannelTransport
from vv_agent.types import LLMResponse, ToolCall


def test_expiry_retry_survives_a_second_abandoned_holder(tmp_path, monkeypatch):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", provider())
        request(server, 1, "thread/start", {"inputId": "thread"})
        first = server.kernel.store.acquire("thread", owner="holder-A", ttl_ms=200)
        messages = record_output(server, monkeypatch)
        request(server, 2, "turn/start", submission())
        server.run_adapter.join()
        status = server.kernel.store.lease_status
        takeovers = []

        def takeover(sid):
            head, epoch, remaining = status(sid)
            if not takeovers and remaining == 0:
                second = server.kernel.store.acquire(sid, owner="holder-B", ttl_ms=300)
                assert second and second.epoch > first.epoch
                takeovers.append(second)
                return status(sid)
            return head, epoch, remaining

        monkeypatch.setattr(server.kernel.store, "lease_status", takeover)
        eventually(lambda: bool(takeovers))
        eventually(lambda: bool(server.kernel.store.read_state("thread")[0].turns))
        server.run_adapter.join()
        assert server.kernel.store.read_state("thread")[0].turns["thread/turn/prompt"].ended
        assert not any(m.get("params", {}).get("code") == "lease_retry_exhausted" for m in messages)


@pytest.mark.parametrize("trigger", ["retry", "wake"])
@pytest.mark.parametrize("approval", [False, True])
def test_disconnected_recovery_does_not_resubscribe_and_approval_is_durable(tmp_path, monkeypatch, trigger, approval):
    effects = []

    @function_tool(needs_approval=True)
    def protected() -> str:
        effects.append("once")
        return "ok"

    models = (
        ScriptedModelProvider.new(
            "test", "m", [LLMResponse("", [ToolCall("protected-call", "protected", {})]), LLMResponse("done")]
        )
        if approval
        else provider()
    )
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", models, agent=Agent("test", "Answer.", tools=[protected]))
        messages = record_output(server, monkeypatch)
        request(server, 1, "thread/start", {"inputId": "thread"})
        holder = server.kernel.store.acquire("thread", owner="dead", ttl_ms=300)
        tid = request(server, 2, "turn/start", submission())["result"]["turnId"]
        server.run_adapter.join()
        assert server.run_adapter._lease_retries
        server.processor.disconnect_connection("A")
        assert not server.router.is_registered("A")
        if trigger == "wake":
            server.kernel.store.release(holder)
            server.wake("thread")
        eventually(lambda: bool(server.kernel.store.read_state("thread")[0].turns))
        server.run_adapter.join()
        assert server.state_manager.subscribers("thread") == set()
        assert server.run_adapter._notification_positions == {}

        if approval:
            assert effects == []
            assert server.run_adapter.public_thread_status("thread")["status"] == "approval_wait"
            assert server.store.runtimes["thread"].config.approval_provider is None
            rid = server.kernel.waits("thread")[0][2]["request_id"]
            other = node(stack, tmp_path / "store", "B", models, agent=server.host._agent)
            assert (
                request(
                    other, 3, "approval/resolve", {"threadId": "thread", "turnId": tid, "requestId": rid, "decision": "allow"}
                )["result"]
                == {}
            )
            other.run_adapter.join()
            assert effects == ["once"]
        assert server.kernel.store.read_state("thread")[0].turns[tid].ended
        assert server.router.pending_server_request_count() == 0
        assert not any(m.get("method") == "approval/request" for m in messages)
        # Let the old timer fire too: it must not recreate notification state.
        eventually(lambda: not server.run_adapter._lease_retries)
        time.sleep(0.1)
        assert server.state_manager.subscribers("thread") == set()
        assert server.run_adapter._notification_positions == {}


@pytest.mark.parametrize("trigger", ["retry", "wake"])
def test_recovery_uses_a_registered_subscriber_after_original_disconnect(tmp_path, trigger):
    with ExitStack() as stack:
        server = node(stack, tmp_path / "store", "A", provider())
        request(server, 1, "thread/start", {"inputId": "thread"})
        transport = ChannelTransport(connection_id="B", outbound_capacity=10000)
        server.router.register_transport(transport)
        peer = SimpleNamespace(transport=transport, processor=server.processor)
        request(peer, "init-B", "initialize", {"clientInfo": {"name": "B"}})
        request(peer, 2, "thread/resume", {"threadId": "thread"})
        holder = server.kernel.store.acquire("thread", owner="dead", ttl_ms=300)
        tid = request(server, 3, "turn/start", submission())["result"]["turnId"]
        server.run_adapter.join()
        server.processor.disconnect_connection("A")
        if trigger == "wake":
            server.kernel.store.release(holder)
            server.wake("thread")
        eventually(lambda: bool(server.kernel.store.read_state("thread")[0].turns))
        server.run_adapter.join()
        assert server.kernel.store.read_state("thread")[0].turns[tid].ended
        assert server.state_manager.subscribers("thread") == {"B"}
        assert server.store.runtimes["thread"].config.approval_provider._connection_id == "B"
