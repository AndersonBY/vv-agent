from __future__ import annotations

from vv_agent import Agent, RunConfig
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.app_server.server import AppServer
from vv_agent.app_server.transport import ChannelTransport
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse, ToolCall


def run_debug_message(message: str) -> list[dict[str, object]]:
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
        endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
        return (
            llm,
            ResolvedModelConfig(
                backend="debug",
                requested_model="debug-model",
                selected_model="debug-model",
                model_id="debug-model",
                endpoint_options=[EndpointOption(endpoint=endpoint, model_id="debug-model")],
            ),
        )

    transport = ChannelTransport(connection_id="debug")
    server = AppServer(
        transport=transport,
        host=DefaultAppServerHost(
            agent=Agent(name="assistant", instructions="Answer.", model="debug-model"),
            run_config=RunConfig(model_provider=model_provider, max_cycles=1),
        ),
    )
    for payload in [
        {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "debug"}}},
        {"id": 1, "method": "thread/start", "params": {"agentKey": "default"}},
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": message}]}},
    ]:
        transport.send_inbound(payload)
        server.processor.process_next(transport)

    outbound: list[dict[str, object]] = []
    while True:
        item = transport.receive_outbound(timeout=10)
        outbound.append(item)
        if item.get("method") == "turn/completed":
            return outbound
