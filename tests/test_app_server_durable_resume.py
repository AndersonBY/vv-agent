from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from typing import Any, cast

import pytest
from support import FixedModelProvider

from vv_agent import Agent, RunConfig, function_tool
from vv_agent.app_server import (
    AppServer,
    AppServerErrorCode,
    ChannelTransport,
    DefaultAppServerHost,
)
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.llm import ScriptedLLM
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


def _send(server: AppServer, payload: dict[str, Any]) -> None:
    server.processor.process_message("conn_1", payload)


def _drain_until_completed(transport: ChannelTransport) -> list[dict[str, Any]]:
    messages = []
    while True:
        message = transport.receive_outbound(timeout=5)
        messages.append(message)
        if message.get("method") == "turn/completed":
            return messages


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
    from vv_agent.session.surfaces import SessionDriver

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

    kernel = SessionDriver(path)
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
        store=kernel.store,
        host=DefaultAppServerHost(
            agent=Agent("assistant", "Work.", model="test-model", tools=[work]),
            run_config=RunConfig(model_provider=FixedModelProvider(ScriptedLLM([model, model]), _resolved_model()), max_cycles=2),
        ),
    )
    server.kernel = kernel
    server.store.kernel = kernel
    server.run_adapter.kernel = kernel
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
    from vv_agent.session.surfaces import SessionDriver

    leases = _inject_lease_loss(monkeypatch, cut, loss)
    calls = []

    def model(_request):
        calls.append("model")
        return LLMResponse("work", [ToolCall("work", "work", {})]) if calls == ["model"] else LLMResponse("done")

    @function_tool(needs_approval=surface_name == "approval")
    def work():
        calls.append("tool")
        return "worked"

    kernel = SessionDriver(tmp_path / "lease.sqlite")
    agent = Agent("assistant", "Work.", model="test-model", tools=[work])
    provider = FixedModelProvider(ScriptedLLM([model, model]), _resolved_model())
    server = None
    try:
        if surface_name != "interactive":
            transport = _KernelTestTransport(connection_id="conn_1")
            server = AppServer(
                transport=transport,
                store=kernel.store,
                host=DefaultAppServerHost(agent=agent, run_config=RunConfig(model_provider=provider, max_cycles=2)),
            )
            kernel = server.kernel
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
                options=AgentSessionOptions(model_provider=provider, workspace=tmp_path), store=kernel.store
            )
            kernel = client.driver
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
