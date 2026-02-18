from __future__ import annotations

from pathlib import Path

from v_agent.constants import TASK_FINISH_TOOL_NAME, TODO_WRITE_TOOL_NAME
from v_agent.llm import ScriptedLLM
from v_agent.runtime import AgentRuntime, BeforeLLMPatch
from v_agent.tools import ToolContext, ToolSpec, build_default_registry
from v_agent.types import (
    AgentStatus,
    AgentTask,
    LLMResponse,
    Message,
    ToolCall,
    ToolDirective,
    ToolExecutionResult,
    ToolResultStatus,
)


def test_runtime_hook_can_patch_before_llm_messages(tmp_path: Path) -> None:
    class InjectMessageHook:
        def before_llm(self, event):
            patched = list(event.messages)
            patched.append(Message(role="user", content="HOOK_CONTEXT"))
            return BeforeLLMPatch(messages=patched)

    def assert_hook_message(model: str, messages: list[Message]) -> LLMResponse:
        del model
        assert any(message.role == "user" and message.content == "HOOK_CONTEXT" for message in messages)
        return LLMResponse(
            content="finish",
            tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
        )

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[assert_hook_message]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        hooks=[InjectMessageHook()],
    )
    task = AgentTask(task_id="hook_before_llm", model="m", system_prompt="sys", user_prompt="start", max_cycles=2)

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "ok"
    assert any(message.content == "HOOK_CONTEXT" for message in result.messages)


def test_runtime_hook_can_short_circuit_tool_call(tmp_path: Path) -> None:
    class BlockTodoHook:
        def before_tool_call(self, event):
            if event.call.name != TODO_WRITE_TOOL_NAME:
                return None
            return ToolExecutionResult(
                tool_call_id=event.call.id,
                status="error",
                status_code=ToolResultStatus.ERROR,
                error_code="blocked_by_hook",
                content='{"ok":false,"error":"blocked"}',
            )

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="todo",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=TODO_WRITE_TOOL_NAME,
                        arguments={"todos": [{"title": "a", "status": "pending", "priority": "medium"}]},
                    )
                ],
            ),
            LLMResponse(
                content="finish",
                tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
        ]
    )
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        hooks=[BlockTodoHook()],
    )
    task = AgentTask(task_id="hook_tool_short_circuit", model="m", system_prompt="sys", user_prompt="go", max_cycles=4)

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.cycles[0].tool_results[0].error_code == "blocked_by_hook"


def test_runtime_hook_can_patch_after_tool_call_to_finish(tmp_path: Path) -> None:
    class FinishAfterTodoHook:
        def after_tool_call(self, event):
            if event.call.name != TODO_WRITE_TOOL_NAME:
                return None
            return ToolExecutionResult(
                tool_call_id=event.result.tool_call_id,
                content=event.result.content,
                status_code=ToolResultStatus.SUCCESS,
                directive=ToolDirective.FINISH,
                metadata={"final_message": "finished-by-hook"},
            )

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="todo",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=TODO_WRITE_TOOL_NAME,
                        arguments={"todos": [{"title": "a", "status": "completed", "priority": "medium"}]},
                    )
                ],
            )
        ]
    )
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        hooks=[FinishAfterTodoHook()],
    )
    task = AgentTask(task_id="hook_after_tool", model="m", system_prompt="sys", user_prompt="go", max_cycles=4)

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "finished-by-hook"


def test_runtime_hook_can_replace_llm_response(tmp_path: Path) -> None:
    class ReplaceResponseHook:
        def after_llm(self, event):
            del event
            return LLMResponse(
                content="forced finish",
                tool_calls=[ToolCall(id="h1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "hook-finish"})],
            )

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="plain")]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        hooks=[ReplaceResponseHook()],
    )
    task = AgentTask(task_id="hook_after_llm", model="m", system_prompt="sys", user_prompt="go", max_cycles=2)

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "hook-finish"


def test_runtime_steering_skips_remaining_tool_calls(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    def track(event: str, payload: dict[str, object]) -> None:
        events.append((event, payload))

    def _noop(context: ToolContext, arguments: dict[str, object]) -> ToolExecutionResult:
        del context, arguments
        return ToolExecutionResult(tool_call_id="", content='{"ok":true}')

    registry = build_default_registry()
    registry.register(ToolSpec(name="_demo_noop", handler=_noop))
    registry.register_schema(
        "_demo_noop",
        {
            "type": "function",
            "function": {"name": "_demo_noop", "description": "noop", "parameters": {"type": "object", "properties": {}}},
        },
    )

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="two tools",
                tool_calls=[
                    ToolCall(id="t1", name="_demo_noop", arguments={}),
                    ToolCall(id="t2", name="_demo_noop", arguments={}),
                ],
            ),
            LLMResponse(
                content="finish",
                tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
        ]
    )

    queued = {"used": False}

    def interruption_provider() -> list[Message]:
        if queued["used"]:
            return []
        queued["used"] = True
        return [Message(role="user", content="STEER_NOW")]

    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=registry,
        default_workspace=tmp_path,
        log_handler=track,
    )
    task = AgentTask(
        task_id="steer_skip",
        model="m",
        system_prompt="sys",
        user_prompt="go",
        max_cycles=4,
        extra_tool_names=["_demo_noop"],
    )
    result = runtime.run(task, interruption_messages=interruption_provider)

    assert result.status == AgentStatus.COMPLETED
    assert result.cycles[0].tool_results[1].error_code == "skipped_due_to_steering"
    assert any(message.content == "STEER_NOW" for message in result.messages)
    assert any(name == "run_steered" for name, _ in events)
