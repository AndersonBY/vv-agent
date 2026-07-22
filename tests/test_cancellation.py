from __future__ import annotations

import pytest
from support import FixedModelProvider

from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.runtime.cancellation import CancellationToken, CancelledError
from vv_agent.runtime.context import ExecutionContext


class TestCancellationToken:
    def test_initial_state(self):
        token = CancellationToken()
        assert not token.cancelled

    def test_cancel(self):
        token = CancellationToken()
        token.cancel()
        assert token.cancelled

    def test_check_raises_when_cancelled(self):
        token = CancellationToken()
        token.cancel()
        with pytest.raises(CancelledError):
            token.check()

    def test_check_ok_when_not_cancelled(self):
        token = CancellationToken()
        token.check()  # should not raise

    def test_on_cancel_callback(self):
        token = CancellationToken()
        called = []
        token.on_cancel(lambda: called.append(True))
        assert not called
        token.cancel()
        assert called == [True]

    def test_cancel_is_idempotent_and_callbacks_run_once(self):
        token = CancellationToken()
        called = []
        token.on_cancel(lambda: called.append(True))

        token.cancel()
        token.cancel()

        assert called == [True]

    def test_on_cancel_fires_immediately_if_already_cancelled(self):
        token = CancellationToken()
        token.cancel()
        called = []
        token.on_cancel(lambda: called.append(True))
        assert called == [True]

    def test_child_cancelled_when_parent_cancelled(self):
        parent = CancellationToken()
        child = parent.child()
        assert not child.cancelled
        parent.cancel()
        assert child.cancelled

    def test_child_not_cancelled_independently(self):
        parent = CancellationToken()
        child = parent.child()
        child.cancel()
        assert child.cancelled
        assert not parent.cancelled

    def test_grandchild_propagation(self):
        grandparent = CancellationToken()
        parent = grandparent.child()
        child = parent.child()
        grandparent.cancel()
        assert parent.cancelled
        assert child.cancelled

    def test_child_preserves_parent_cancellation_reason(self):
        parent = CancellationToken()
        child = parent.child()

        parent.cancel("host requested cancellation")

        assert child.reason == "host requested cancellation"
        with pytest.raises(CancelledError, match="host requested cancellation"):
            child.check()


class TestExecutionContext:
    def test_check_cancelled_with_no_token(self):
        ctx = ExecutionContext()
        ctx.check_cancelled()  # should not raise

    def test_check_cancelled_raises(self):
        token = CancellationToken()
        token.cancel()
        ctx = ExecutionContext(cancellation_token=token)
        with pytest.raises(CancelledError):
            ctx.check_cancelled()


class TestCancellationInRuntime:
    """Test cancellation integrated with AgentRuntime cycle loop."""

    def test_cancel_before_first_cycle(self):
        from vv_agent.llm.scripted import ScriptedLLM
        from vv_agent.runtime import AgentRuntime
        from vv_agent.tools import build_default_registry
        from vv_agent.types import AgentStatus, AgentTask, LLMResponse

        llm = ScriptedLLM(steps=[LLMResponse(content="hello")])
        runtime = AgentRuntime(llm_client=llm, tool_registry=build_default_registry())
        task = AgentTask(
            task_id="cancel-test",
            model="test",
            system_prompt="sys",
            user_prompt="hi",
            max_cycles=3,
            no_tool_policy="finish",
        )
        token = CancellationToken()
        token.cancel()
        ctx = ExecutionContext(cancellation_token=token)
        result = runtime.run(task, ctx=ctx)
        assert result.status == AgentStatus.FAILED
        assert "cancelled" in (result.error or "").lower()

    def test_cancel_between_cycles(self):
        from vv_agent.llm.scripted import ScriptedLLM
        from vv_agent.runtime import AgentRuntime
        from vv_agent.tools import build_default_registry
        from vv_agent.types import AgentStatus, AgentTask, LLMResponse

        # 3 cycles, cancel after first
        llm = ScriptedLLM(
            steps=[
                LLMResponse(content="cycle1"),
                LLMResponse(content="cycle2"),
                LLMResponse(content="cycle3"),
            ]
        )
        runtime = AgentRuntime(llm_client=llm, tool_registry=build_default_registry())
        task = AgentTask(
            task_id="cancel-test-2",
            model="test",
            system_prompt="sys",
            user_prompt="hi",
            max_cycles=3,
            no_tool_policy="continue",
        )
        token = CancellationToken()
        ctx = ExecutionContext(cancellation_token=token)

        # Cancel after first cycle via before_cycle_messages
        def cancel_on_cycle_2(cycle_index, messages, shared):
            if cycle_index == 2:
                token.cancel()
            return []

        result = runtime.run(task, ctx=ctx, before_cycle_messages=cancel_on_cycle_2)
        assert result.status == AgentStatus.FAILED
        assert "cancelled" in (result.error or "").lower()


def test_runner_emits_one_cancelled_terminal_event(tmp_path) -> None:
    from vv_agent import Agent, RunConfig, Runner
    from vv_agent.llm.scripted import ScriptedLLM
    from vv_agent.runtime.cancellation import CancellationToken
    from vv_agent.types import LLMResponse

    token = CancellationToken()
    token.cancel()
    llm = ScriptedLLM(steps=[LLMResponse(content="unused")])
    endpoint = EndpointConfig(
        endpoint_id="test",
        api_key="test-key",
        api_base="https://example.invalid/v1",
    )
    resolved = ResolvedModelConfig(
        requested_model="test-model",
        selected_model="test-model",
        backend="test",
        model_id="test-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="test-model")],
    )
    result = Runner.run_sync(
        Agent(name="cancelled", instructions="Stop.", model="test-model"),
        "stop",
        run_config=RunConfig(
            workspace=tmp_path,
            cancellation_token=token,
            model_provider=FixedModelProvider(llm, resolved),
        ),
    )

    terminal_types = [event.type for event in result.events if event.type in {"run_completed", "run_failed", "run_cancelled"}]
    assert terminal_types == ["run_cancelled"]
