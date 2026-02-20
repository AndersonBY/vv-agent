from __future__ import annotations

from collections.abc import Callable

from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.runtime import AgentRuntime
from vv_agent.runtime.context import ExecutionContext
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentStatus, AgentTask, LLMResponse, Message, ToolCall


class StreamCapturingLLM:
    """LLM that captures stream_callback and simulates token-level streaming."""

    def __init__(self, tokens: list[str], tool_calls: list[ToolCall] | None = None):
        self.tokens = tokens
        self.tool_calls = tool_calls or []

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, object]],
        stream_callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        content_parts = []
        for token in self.tokens:
            content_parts.append(token)
            if stream_callback is not None:
                stream_callback(token)
        return LLMResponse(
            content="".join(content_parts),
            tool_calls=self.tool_calls,
        )


class TestStreamCallback:
    def test_stream_callback_receives_tokens(self):
        tokens = ["Hello", " ", "world", "!"]
        llm = StreamCapturingLLM(tokens=tokens)
        runtime = AgentRuntime(
            llm_client=llm,
            tool_registry=build_default_registry(),
        )
        task = AgentTask(
            task_id="stream-test",
            model="test",
            system_prompt="sys",
            user_prompt="hi",
            max_cycles=1,
            no_tool_policy="finish",
        )
        received: list[str] = []
        ctx = ExecutionContext(stream_callback=lambda t: received.append(t))
        result = runtime.run(task, ctx=ctx)
        assert result.status == AgentStatus.COMPLETED
        assert received == tokens
        assert result.final_answer == "Hello world!"

    def test_no_stream_callback_still_works(self):
        tokens = ["Hi"]
        llm = StreamCapturingLLM(tokens=tokens)
        runtime = AgentRuntime(
            llm_client=llm,
            tool_registry=build_default_registry(),
        )
        task = AgentTask(
            task_id="no-stream-test",
            model="test",
            system_prompt="sys",
            user_prompt="hi",
            max_cycles=1,
            no_tool_policy="finish",
        )
        result = runtime.run(task)
        assert result.status == AgentStatus.COMPLETED
        assert result.final_answer == "Hi"

    def test_scripted_llm_ignores_stream_callback(self):
        llm = ScriptedLLM(steps=[LLMResponse(content="ok")])
        runtime = AgentRuntime(
            llm_client=llm,
            tool_registry=build_default_registry(),
        )
        task = AgentTask(
            task_id="scripted-stream",
            model="test",
            system_prompt="sys",
            user_prompt="hi",
            max_cycles=1,
            no_tool_policy="finish",
        )
        received: list[str] = []
        ctx = ExecutionContext(stream_callback=lambda t: received.append(t))
        result = runtime.run(task, ctx=ctx)
        assert result.status == AgentStatus.COMPLETED
        # ScriptedLLM doesn't call stream_callback
        assert received == []
