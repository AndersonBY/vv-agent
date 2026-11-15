from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._hydration import MissingContentLink
from vv_agent.session import _recovery_phase
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_runtime import bind_content_key_provider
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import _Driver
from vv_agent.session.projection import hydrate_record_payload, hydration_for_store
from vv_agent.session.records import InboxItem
from vv_agent.session.result import project_result
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver

MASTER_KEY = b"k" * 32
THREAD = "thread"
INPUT = "input-1"
TURN = f"{THREAD}/turn/{INPUT}"


def _host() -> DefaultAppServerHost:
    return DefaultAppServerHost(
        agent=Agent("test", "Be precise.", model="scripted"),
        run_config=RunConfig(model_provider=ScriptedModelProvider.new("scripted", "scripted", [])),
    )


def _rpc(
    server: AppServer,
    transport: ChannelTransport,
    request_id: int,
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    transport.send_inbound(payload)
    server.processor.process_next(transport)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        message = transport.receive_outbound(timeout=5)
        if message.get("id") == request_id:
            assert "error" not in message
            result = message["result"]
            assert isinstance(result, dict)
            return result
    raise AssertionError(f"timed out waiting for {method}")


def _populate(store: SQLiteStore, provider: LocalKeyProvider) -> SessionDriver:
    sessions = SessionDriver(store=store, content_key_provider=provider)
    sessions.create(
        THREAD,
        "/tmp",
        {"app_server": {"agent_key": "default", "cwd": "/tmp", "metadata": {"label": "durable"}}},
    )
    content = {
        "text": "hello",
        "messages": [{"role": "user", "content": "hello"}],
        "app_server": {
            "input": [{"type": "text", "text": "hello"}],
            "metadata": {"source": "test"},
            "kind": "start",
            "expected_turn_id": None,
        },
    }
    sessions.push(THREAD, InboxItem(INPUT, "user", {"content": content}))
    runtime = sessions.runtime(
        Agent("test", "Be precise.", model="scripted"),
        RunConfig(
            model_provider=ScriptedModelProvider.new("scripted", "scripted", []),
            workspace="/tmp",
        ),
        session_id=THREAD,
    )
    lease = sessions.store.acquire(THREAD, owner="app-server-hydration-test", ttl_ms=15_000)
    assert lease is not None
    driver = _Driver(sessions.store, THREAD, runtime, _Scope(lease, runtime, now_ms=store._now()))
    assert _recovery_phase.run(driver, inputs.apply_input(driver, one_turn=False))
    assert _recovery_phase.run(driver, inputs.start_turn(driver))
    driver.commit(
        [
            driver.record(
                "turn_ended",
                {
                    "status": "completed",
                    "reason": None,
                    "result": "done",
                    "adopted_results": [],
                    "budget": {},
                    "unconfirmed_operations": [],
                },
            )
        ],
        guarded=True,
    )
    sessions.store.release(lease)
    return sessions


def test_s1_projection_reopens_with_hydrated_thread_context_and_result(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    provider = LocalKeyProvider(MASTER_KEY)
    with SQLiteStore.standalone(str(path)) as store:
        sessions = _populate(store, provider)
        try:
            records = store.read_state(THREAD)[1]
            creation = next(row for row in records if row.record.kind == "session_created")
            started = next(row for row in records if row.record.kind == "turn_started")
            applied = next(row for row in records if row.record.kind == "input_applied")
            ended = next(row for row in records if row.record.kind == "turn_ended")
            assert ContentRef.from_dict(creation.record._payload["attributes"])
            assert ContentRef.from_dict(started.record._payload["definition"])
            assert ContentRef.from_dict(applied.record._payload["input"]["payload"]["content"])
            assert ContentRef.from_dict(ended.record._payload["result"])
        finally:
            sessions.close()

    with SQLiteStore.standalone(str(path)) as reopened:
        bind_content_key_provider(reopened, provider)
        transport = ChannelTransport(connection_id="hydration-test")
        server = AppServer(transport=transport, store=reopened, host=_host())
        _rpc(
            server,
            transport,
            0,
            "initialize",
            {"clientInfo": {"name": "hydration-test"}},
        )
        transport.send_inbound({"jsonrpc": "2.0", "method": "initialized"})
        server.processor.process_next(transport)
        thread_wire = _rpc(server, transport, 1, "thread/read", {"threadId": THREAD})
        assert thread_wire["thread"]["agentKey"] == "default"
        assert thread_wire["thread"]["metadata"] == {"label": "durable"}
        assert thread_wire["turns"][0]["input"] == [{"type": "text", "text": "hello"}]
        assert thread_wire["turns"][0]["result"]["finalOutput"] == "done"
        context_wire = _rpc(server, transport, 2, "thread/context/read", {"threadId": THREAD})
        assert context_wire["threadId"] == THREAD
        assert context_wire["counts"]["modelVisibleHistoryMessages"] == 1
        listed = _rpc(server, transport, 3, "thread/list", {})
        assert [thread["threadId"] for thread in listed["threads"]] == [THREAD]
        snapshot = server.store.read_thread(THREAD)
        assert snapshot.thread.agent_key == "default"
        assert snapshot.thread.metadata == {"label": "durable"}
        assert len(snapshot.turns) == 1
        assert snapshot.turns[0].input == [{"type": "text", "text": "hello"}]
        assert snapshot.turns[0].result["finalOutput"] == "done"
        context = server.run_adapter.read_context(THREAD)
        assert context["threadId"] == THREAD
        assert context["counts"]["modelVisibleHistoryMessages"] == 1
        result = project_result(reopened, THREAD, TURN)
        assert result.status.value == "completed"
        assert result.input == "hello"
        assert result.final_output == "done"
        server.kernel.close()


def test_projection_rejects_cross_tenant_content_binding(tmp_path: Path) -> None:
    with SQLiteStore.standalone(str(tmp_path / "session.sqlite")) as store:
        sessions = _populate(store, LocalKeyProvider(MASTER_KEY))
        try:
            creation = store.read(THREAD, limit=1).records[0]
            wrong = hydration_for_store(store, tenant_id="other-tenant", session_id=THREAD)
            with pytest.raises(MissingContentLink):
                hydrate_record_payload(creation, wrong)
        finally:
            sessions.close()


def test_projection_fails_closed_when_linked_content_is_missing(tmp_path: Path) -> None:
    with SQLiteStore.standalone(str(tmp_path / "session.sqlite")) as store:
        sessions = _populate(store, LocalKeyProvider(MASTER_KEY))
        try:
            records = store.read_state(THREAD)[1]
            ended = next(row for row in records if row.record.kind == "turn_ended")
            result_ref = ContentRef.from_dict(ended.record._payload["result"])
            connection = store.connection
            connection.execute("PRAGMA foreign_keys=OFF")
            try:
                connection.execute(
                    "DELETE FROM sk_content WHERE tenant_id=? AND session_id=? AND content_id=?",
                    (store.tenant_id, THREAD, result_ref.content_id),
                )
                connection.commit()
            finally:
                connection.execute("PRAGMA foreign_keys=ON")
            server = AppServer(store=store, host=_host())
            with pytest.raises(MissingContentLink):
                server.store.read_thread(THREAD)
            server.kernel.close()
        finally:
            sessions.close()
