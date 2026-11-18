"""S1 creation attributes expose logical values through every public surface."""

from __future__ import annotations

from pathlib import Path

from vv_agent import (
    Agent,
    AgentSessionOptions,
    InteractiveAgentClient,
    Message,
    RunConfig,
    ScriptedModelProvider,
)
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.kernel import _driver_planning
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._content_publication import declared_content_ref
from vv_agent.session import _recovery_phase
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.kernel import _Driver
from vv_agent.session.replay import list_events

APP_SERVER_PRINCIPAL = {
    "tenant_id": "default",
    "subject_id": "app-server",
    "subject_kind": "service",
    "attributes": {"role": "app-server"},
}


def _send(server: AppServer, transport: ChannelTransport, request_id: int | str, method: str, params: dict) -> dict:
    server.processor.process_message(
        transport.connection_id,
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
    )
    while True:
        message = transport.receive_outbound(timeout=5)
        if message.get("id") == request_id:
            assert "result" in message, message
            return message["result"]


def test_session_surfaces_hydrate_s1_seed_messages_and_shared_state(surface, tmp_path: Path) -> None:
    client = InteractiveAgentClient(
        options=AgentSessionOptions(
            model_provider=ScriptedModelProvider.new("test", "m", []),
            workspace=tmp_path,
        )
    )
    history = [Message("user", "retained seed")]
    session = client.create_session(
        agent=Agent("test", "Work.", model="m"),
        session_id="seeded",
        session={"messages": [message.to_dict() for message in history], "shared_state": {"nested": {"value": 7}}},
    )

    creation = surface.store.read("seeded", limit=1).records[0].record
    ContentRef.from_dict(creation._payload["attributes"])
    assert session.messages == history
    assert session.shared_state == {"nested": {"value": 7}}

    restored = client.create_session(agent=Agent("test", "Work.", model="m"), session_id="seeded")
    assert restored.messages == history
    assert restored.shared_state == {"nested": {"value": 7}}


def test_list_events_hydrates_s1_app_server_creation_and_cursor(surface) -> None:
    seed_message = Message("user", "app-server-seed-secret")
    shared_state = {"nested": {"count": 7, "marker": "app-server-state-secret"}}
    surface.create(
        "thread",
        ".",
        {
            "app_server": {"agent_key": "default", "cwd": None, "metadata": {}},
            "seed": {"messages": [seed_message.to_dict()], "shared_state": shared_state},
        },
        principal=APP_SERVER_PRINCIPAL,
    )

    creation = surface.store.read("thread", limit=1).records[0].record
    attributes_ref = ContentRef.from_dict(creation._payload["attributes"])
    with surface.store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id="default", session_id="thread")
        attributes = writer.get_json(attributes_ref)
        message_ref = declared_content_ref(attributes["seed"]["messages"][0])
        assert message_ref is not None
        assert writer.get_json(message_ref) == seed_message.to_dict()
        links = {link.slot for link in writer.durable.read_links("default", "thread", record_seq=1)}
    assert attributes["seed"]["shared_state"] == shared_state
    assert {"session.attributes", "session.attributes.seed.messages[0]"} <= links

    record_body = surface.store._one(
        "SELECT body FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=1",
        ("default", "thread"),
    )[0]
    record_bytes = record_body if isinstance(record_body, bytes) else record_body.encode()
    assert b"app-server-seed-secret" not in record_bytes
    assert b"app-server-state-secret" not in record_bytes
    for row in surface.store._rows("SELECT * FROM sk_content WHERE tenant_id=%s AND session_id=%s", ("default", "thread")):
        for value in row:
            encoded = value if isinstance(value, bytes) else str(value).encode()
            assert b"app-server-seed-secret" not in encoded
            assert b"app-server-state-secret" not in encoded

    transport = ChannelTransport(connection_id="test", outbound_capacity=100)
    server = AppServer(
        transport=transport,
        host=DefaultAppServerHost(
            agent=Agent("test", "Work.", model="scripted"),
            run_config=RunConfig(model_provider=ScriptedModelProvider.new("test", "scripted", [])),
        ),
    )
    _send(server, transport, 0, "initialize", {"clientInfo": {"name": "test"}})
    replayed = _send(server, transport, 1, "thread/start", {"inputId": "thread", "agentKey": "default"})
    assert replayed["replayed"] is True
    assert surface.session_attributes("thread")["seed"] == {
        "messages": [seed_message.to_dict()],
        "shared_state": shared_state,
    }

    page = list_events(surface.store, "thread", limit=1)
    cursor = page["throughCursor"]

    assert page == {
        "events": [],
        "nextCursor": cursor,
        "throughCursor": cursor,
        "more": False,
    }
    assert list_events(surface.store, "thread", limit=1, after=cursor, through=cursor) == page


def test_effective_model_hydrates_s1_app_server_fork(surface, tmp_path: Path) -> None:
    effective = {
        "model": "fork-model",
        "model_settings": {"max_tokens": 321},
        "route": {"opaque": "frozen"},
        "binding": {"model": "fork-model"},
    }
    surface.create(
        "fork",
        str(tmp_path),
        {
            "seed": {"messages": [], "shared_state": {}},
            "app_server_fork": {"effective_model": effective},
        },
    )
    runtime = surface.runtime(
        Agent("fork", "Work.", model="scripted"),
        RunConfig(model_provider=ScriptedModelProvider.new("scripted", "scripted", []), workspace=tmp_path),
        session_id="fork",
    )
    lease = surface.store.acquire("fork", owner="s1-hydration-entry", ttl_ms=15_000)
    assert lease is not None
    driver = _Driver(surface.store, "fork", runtime, _Scope(lease, runtime, now_ms=surface.store._now()))
    assert _recovery_phase.run(driver, _driver_planning.effective_model(driver)) == effective
    assert surface.store.release(lease)


def test_thread_start_recovers_s1_frozen_app_server_context(surface, monkeypatch) -> None:
    frozen = {
        "agent_name": "frozen",
        "memory_settings": {},
        "child_tools": [],
        "model_binding": {"model": "frozen-model"},
        "capabilities": {},
        "task": {"model": "frozen-model"},
        "tools": [],
    }
    surface.create(
        "thread",
        ".",
        {
            "app_server": {"agent_key": "default", "cwd": None, "metadata": {"frozen": True}},
            "app_server_context": frozen,
            "seed": {"messages": [], "shared_state": {}},
        },
        principal=APP_SERVER_PRINCIPAL,
    )
    transport = ChannelTransport(connection_id="test", outbound_capacity=100)
    server = AppServer(
        transport=transport,
        host=DefaultAppServerHost(
            agent=Agent("test", "Work.", model="scripted"),
            run_config=RunConfig(model_provider=ScriptedModelProvider.new("test", "scripted", [])),
        ),
    )
    captured = []
    receipt = server.run_adapter._store.create_thread_receipt

    def capture_receipt(**kwargs):
        captured.append(kwargs["context"])
        return receipt(**kwargs)

    monkeypatch.setattr(server.run_adapter._store, "create_thread_receipt", capture_receipt)

    _send(server, transport, 0, "initialize", {"clientInfo": {"name": "test"}})
    _send(
        server,
        transport,
        1,
        "thread/start",
        {"inputId": "thread", "agentKey": "default", "metadata": {"frozen": True}},
    )

    creation = surface.store.read("thread", limit=1).records[0].record
    ContentRef.from_dict(creation._payload["attributes"])
    assert captured == [frozen]
