from __future__ import annotations

import pytest

from v_agent.runtime.cancellation import CancellationToken, CancelledError
from v_agent.runtime.context import ExecutionContext


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
        from v_agent.llm.scripted import ScriptedLLM
        from v_agent.runtime import AgentRuntime
        from v_agent.tools import build_default_registry
        from v_agent.types import AgentStatus, AgentTask, LLMResponse

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
        from v_agent.llm.scripted import ScriptedLLM
        from v_agent.runtime import AgentRuntime
        from v_agent.tools import build_default_registry
        from v_agent.types import AgentStatus, AgentTask, LLMResponse

        # 3 cycles, cancel after first
        llm = ScriptedLLM(steps=[
            LLMResponse(content="cycle1"),
            LLMResponse(content="cycle2"),
            LLMResponse(content="cycle3"),
        ])
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
