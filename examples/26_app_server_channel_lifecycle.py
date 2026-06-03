#!/usr/bin/env python3
"""In-process App Server lifecycle with ChannelTransport and a scripted model."""

from __future__ import annotations

import json
from typing import Any

from vv_agent import Agent, RunConfig
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.app_server.server import AppServer
from vv_agent.app_server.transport import ChannelTransport
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse, ToolCall


def _host() -> DefaultAppServerHost:
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="finish", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            )
        ]
    )

    def model_provider(agent: Agent, run_config: RunConfig):
        del agent, run_config
        endpoint = EndpointConfig(endpoint_id="example", api_key="example", api_base="https://example.invalid/v1")
        return (
            llm,
            ResolvedModelConfig(
                backend="example",
                requested_model="scripted-model",
                selected_model="scripted-model",
                model_id="scripted-model",
                endpoint_options=[EndpointOption(endpoint=endpoint, model_id="scripted-model")],
            ),
        )

    return DefaultAppServerHost(
        agent=Agent(name="assistant", instructions="Answer through task_finish.", model="scripted-model"),
        run_config=RunConfig(model_provider=model_provider, max_cycles=1),
    )


def _send(server: AppServer, transport: ChannelTransport, payload: dict[str, Any]) -> None:
    transport.send_inbound(payload)
    server.processor.process_next(transport)


def _print_message(message: dict[str, Any]) -> None:
    print(json.dumps(message, ensure_ascii=False, separators=(",", ":")))


def _drain_until(transport: ChannelTransport, *, response_id: int | None = None, method: str | None = None) -> None:
    while True:
        message = transport.receive_outbound(timeout=10)
        _print_message(message)
        if response_id is not None and message.get("id") == response_id:
            return
        if method is not None and message.get("method") == method:
            return


def main() -> None:
    transport = ChannelTransport(connection_id="example")
    server = AppServer(transport=transport, host=_host())

    _send(
        server,
        transport,
        {
            "id": 0,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "channel-example"},
                "capabilities": {"optOutNotificationMethods": []},
            },
        },
    )
    _drain_until(transport, response_id=0)

    _send(server, transport, {"id": 1, "method": "thread/start", "params": {"agentKey": "default", "cwd": "./workspace"}})
    _drain_until(transport, response_id=1)

    _send(
        server,
        transport,
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": "hello"}]}},
    )
    _drain_until(transport, method="turn/completed")

    _send(server, transport, {"id": 3, "method": "thread/list"})
    _drain_until(transport, response_id=3)

    _send(server, transport, {"id": 4, "method": "thread/archive", "params": {"threadId": "thread_1"}})
    _drain_until(transport, method="thread/status/changed")


if __name__ == "__main__":
    main()
