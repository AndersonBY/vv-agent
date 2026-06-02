from __future__ import annotations

from vv_agent import Agent, RunConfig
from vv_agent.app_server import AppServer, ChannelTransport
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import ScriptedLLM
from vv_agent.types import LLMResponse, ToolCall


def _resolved_model(model: str = "test-model") -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    return ResolvedModelConfig(
        backend="test",
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id=model)],
    )


def test_processor_starts_thread_and_streams_turn_items() -> None:
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
        return llm, _resolved_model()

    transport = ChannelTransport(connection_id="conn_1")
    host = DefaultAppServerHost(
        agent=Agent(name="assistant", instructions="Answer.", model="test-model"),
        run_config=RunConfig(model_provider=model_provider, max_cycles=1),
    )
    server = AppServer(transport=transport, host=host)

    for payload in [
        {"id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}},
        {"id": 1, "method": "thread/start", "params": {"agentKey": "default", "cwd": "/tmp/work"}},
        {"id": 2, "method": "turn/start", "params": {"threadId": "thread_1", "input": [{"type": "text", "text": "hello"}]}},
    ]:
        transport.send_inbound(payload)
        server.processor.process_next(transport)

    outbound: list[dict[str, object]] = []
    while True:
        message = transport.receive_outbound(timeout=2)
        outbound.append(message)
        if message.get("method") == "turn/completed":
            break

    response_ids = [message.get("id") for message in outbound if "result" in message]
    methods = [message.get("method") for message in outbound if "method" in message]

    assert response_ids[:3] == [0, 1, 2]
    assert "thread/started" in methods
    assert "turn/started" in methods
    assert "item/started" in methods
    assert "item/completed" in methods
    assert methods[-1] == "turn/completed"
