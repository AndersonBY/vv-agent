from __future__ import annotations

from typing import Any, cast

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.app_server import AppServer, ChannelTransport, DefaultAppServerHost
from vv_agent.kernel._content import ContentRef
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store
from vv_agent.types import LLMResponse


def _send(transport: ChannelTransport, server: AppServer, payload: dict[str, object]) -> None:
    transport.send_inbound(payload)
    server.processor.process_next(transport)


def _receive_id(transport: ChannelTransport, response_id: int) -> dict[str, Any]:
    while True:
        message = transport.receive_outbound(timeout=10)
        if message.get("id") == response_id:
            return message


def _drain_turn_completed(transport: ChannelTransport) -> None:
    while transport.receive_outbound(timeout=10).get("method") != "turn/completed":
        pass


def test_fork_hydrates_context_without_replacing_committed_content_refs(surface) -> None:
    provider = ScriptedModelProvider.new("test", "test", [LLMResponse("done")])
    host = DefaultAppServerHost(agent=Agent("test", "test", model="test"), run_config=RunConfig(model_provider=provider))
    transport = ChannelTransport(connection_id="test")
    server = AppServer(store=surface.store, host=host, transport=transport)
    try:
        _send(transport, server, {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}})
        surface.create(
            "source",
            "/tmp",
            {
                "app_server": {"agent_key": "default", "cwd": None, "metadata": {}},
                "seed": {"unseen_input_protection": True},
            },
        )
        _send(
            transport,
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "turn/start",
                "params": {
                    "threadId": "source",
                    "inputId": "first",
                    "expectedTurnId": None,
                    "input": [{"type": "text", "text": "first"}],
                },
            },
        )
        turn = cast(dict[str, Any], _receive_id(transport, 1)["result"])["turnId"]
        _drain_turn_completed(transport)
        cast(Any, server.run_adapter).join()

        _send(
            transport,
            server,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "thread/fork",
                "params": {"threadId": "source", "forkId": "copy", "throughTurnId": turn},
            },
        )
        fork_id = cast(dict[str, Any], _receive_id(transport, 2)["result"])["threadId"]
        stored = surface.store.read(fork_id).records[0]
        assert ContentRef.from_dict(stored.record._payload["attributes"]).content_id
        hydrated = hydrate_record_payload(
            stored,
            hydration_for_session_store(surface.store, fork_id),
        )
        assert hydrated["attributes"]["seed"]["unseen_input_protection"] is True
    finally:
        server.kernel.close()
