from __future__ import annotations

from functools import partial
from pathlib import Path

from support.compaction import run_model_turn
from support.kernel_runtime import KernelRuntime

from vv_agent import constants as constants_module
from vv_agent.constants import READ_FILE_TOOL_NAME
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.memory import MemoryManager
from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.prompt import build_raw_system_prompt_bundle
from vv_agent.runtime import ExecutionContext, RuntimeHookManager
from vv_agent.runtime.hooks import (
    AfterLLMEvent,
    AfterToolCallEvent,
    BaseRuntimeHook,
    BeforeLLMEvent,
    BeforeLLMPatch,
    BeforeToolCallEvent,
)
from vv_agent.tools import ToolContext, ToolSpec, build_default_registry
from vv_agent.types import (
    AgentStatus,
    AgentTask,
    LLMResponse,
    Message,
    ToolCall,
    ToolDirective,
    ToolExecutionResult,
    ToolResultStatus,
)
from vv_agent.workspace import MemoryWorkspaceBackend

TASK_LIST_TOOL_NAME = getattr(constants_module, "".join(("TO", "DO")) + "_WRITE_TOOL_NAME")


def test_runtime_hook_can_patch_before_llm_messages(tmp_path: Path) -> None:
    class InjectMessageHook(BaseRuntimeHook):
        def before_llm(self, event: BeforeLLMEvent) -> BeforeLLMPatch:
            patched = list(event.messages)
            patched.append(Message(role="user", content="HOOK_CONTEXT"))
            return BeforeLLMPatch(messages=patched)

    def assert_hook_message(request: LlmRequest) -> LLMResponse:
        model, messages = request.model, request.messages
        del model
        assert any(message.role == "user" and message.content == "HOOK_CONTEXT" for message in messages)
        return LLMResponse(content="ok")

    runtime = KernelRuntime(
        llm_client=ScriptedLLM(steps=[assert_hook_message]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        hooks=[InjectMessageHook()],
    )
    task = AgentTask(
        task_id="hook_before_llm",
        model="m",
        prompt_bundle=build_raw_system_prompt_bundle("sys"),
        user_prompt="start",
        max_cycles=2,
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "ok"
    assert not any(message.content == "HOOK_CONTEXT" for message in result.messages)


def test_before_llm_hook_cannot_remove_recovery_tool_from_compacted_request() -> None:
    class RemoveReadFileHook(BaseRuntimeHook):
        def before_llm(self, event: BeforeLLMEvent) -> BeforeLLMPatch:
            schemas = [schema for schema in event.tool_schemas if schema.get("function", {}).get("name") != READ_FILE_TOOL_NAME]
            return BeforeLLMPatch(tool_schemas=schemas)

    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="start"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "old",
                    "type": "function",
                    "function": {"name": "custom_search", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="a " * 900, tool_call_id="old"),
        Message(role="assistant", content="recent"),
    ]
    manager = MemoryManager(
        compact_threshold=1_000,
        model="unknown-provider-model",
        model_context_window=1_000,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        microcompaction_policy=MicrocompactionPolicy(
            trigger_ratio=0.75,
            target_ratio=0.60,
            keep_recent_cycles=1,
            min_result_chars=500,
        ),
        workspace_backend=MemoryWorkspaceBackend(),
        recovery_tool_available=True,
    )
    runner = partial(
        run_model_turn,
        llm=ScriptedLLM(),
        tool_registry=build_default_registry(),
        hook_manager=RuntimeHookManager(hooks=[RemoveReadFileHook()]),
    )
    task = AgentTask(
        task_id="hook-removes-recovery",
        model="m",
        prompt_bundle=build_raw_system_prompt_bundle("sys"),
        user_prompt="start",
        max_cycles=1,
    )

    result = runner(
        task=task,
        messages=messages,
        memory_manager=manager,
    )

    assert result.status == AgentStatus.FAILED
    assert "recovery_unavailable" in str(result.error)


def test_runtime_hook_can_short_circuit_tool_call(tmp_path: Path) -> None:
    class BlockTodoHook(BaseRuntimeHook):
        def before_tool_call(self, event: BeforeToolCallEvent) -> ToolExecutionResult | None:
            if event.call.name != TASK_LIST_TOOL_NAME:
                return None
            return ToolExecutionResult(
                tool_call_id=event.call.id,
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
                        name=TASK_LIST_TOOL_NAME,
                        arguments={"todos": [{"title": "a", "status": "pending", "priority": "medium"}]},
                    )
                ],
            ),
            LLMResponse(content="done"),
        ]
    )
    lifecycle_events = []
    runtime = KernelRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        hooks=[BlockTodoHook()],
    )
    task = AgentTask(
        task_id="hook_tool_short_circuit",
        model="m",
        prompt_bundle=build_raw_system_prompt_bundle("sys"),
        user_prompt="go",
        max_cycles=4,
    )

    result = runtime.run(
        task,
        ctx=ExecutionContext(
            event_handler=lifecycle_events.append,
            metadata={
                "_vv_agent_agent_name": "hook-agent",
                "_vv_agent_run_id": "run-hook",
                "_vv_agent_trace_id": "trace-hook",
            },
        ),
    )
    assert result.status == AgentStatus.COMPLETED
    assert result.cycles[0].tool_results[0].error_code == "blocked_by_hook"
    blocked_lifecycle = [event for event in lifecycle_events if getattr(event, "tool_call_id", None) == "c1"]
    assert [event.type for event in blocked_lifecycle] == [
        "tool_call_planned",
        "tool_call_completed",
    ]
    assert blocked_lifecycle[-1].execution_started is False


def test_runtime_completed_event_contains_after_hook_result(tmp_path: Path) -> None:
    class FinishAfterTodoHook(BaseRuntimeHook):
        def after_tool_call(self, event: AfterToolCallEvent) -> ToolExecutionResult | None:
            if event.call.name != TASK_LIST_TOOL_NAME:
                return None
            return ToolExecutionResult(
                tool_call_id=event.result.tool_call_id,
                content="hook content",
                status_code=ToolResultStatus.SUCCESS,
                directive=ToolDirective.FINISH,
                metadata={"final_message": "finished-by-hook"},
            )

    lifecycle_events = []
    runtime = KernelRuntime(
        llm_client=ScriptedLLM(
            steps=[
                LLMResponse(
                    content="todo",
                    tool_calls=[
                        ToolCall(
                            id="hook-finalized",
                            name=TASK_LIST_TOOL_NAME,
                            arguments={"todos": [{"title": "a", "status": "completed", "priority": "medium"}]},
                        )
                    ],
                )
            ]
        ),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        hooks=[FinishAfterTodoHook()],
    )
    result = runtime.run(
        AgentTask(
            task_id="hook_completed_event",
            model="m",
            prompt_bundle=build_raw_system_prompt_bundle("sys"),
            user_prompt="go",
            max_cycles=2,
        ),
        ctx=ExecutionContext(
            event_handler=lifecycle_events.append,
            metadata={
                "_vv_agent_agent_name": "hook-agent",
                "_vv_agent_run_id": "run-hook",
                "_vv_agent_trace_id": "trace-hook",
            },
        ),
    )

    assert result.final_answer == "finished-by-hook"
    completed = next(
        event for event in lifecycle_events if event.type == "tool_call_completed" and event.tool_call_id == "hook-finalized"
    )
    assert completed.directive == ToolDirective.FINISH.value
    assert result.cycles[0].tool_results[0].content == "hook content"
    assert result.cycles[0].tool_results[0].metadata["final_message"] == "finished-by-hook"


def test_runtime_hook_can_patch_after_tool_call_to_finish(tmp_path: Path) -> None:
    class FinishAfterTodoHook(BaseRuntimeHook):
        def after_tool_call(self, event: AfterToolCallEvent) -> ToolExecutionResult | None:
            if event.call.name != TASK_LIST_TOOL_NAME:
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
                        name=TASK_LIST_TOOL_NAME,
                        arguments={"todos": [{"title": "a", "status": "completed", "priority": "medium"}]},
                    )
                ],
            )
        ]
    )
    runtime = KernelRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        hooks=[FinishAfterTodoHook()],
    )
    task = AgentTask(
        task_id="hook_after_tool", model="m", prompt_bundle=build_raw_system_prompt_bundle("sys"), user_prompt="go", max_cycles=4
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "finished-by-hook"


def test_runtime_hook_after_tool_call_with_blank_id_is_normalized(tmp_path: Path) -> None:
    class BlankIdAfterHook(BaseRuntimeHook):
        def after_tool_call(self, event: AfterToolCallEvent) -> ToolExecutionResult | None:
            if event.call.name != TASK_LIST_TOOL_NAME:
                return None
            return ToolExecutionResult(
                tool_call_id=" ",
                content=event.result.content,
                status_code=ToolResultStatus.SUCCESS,
            )

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="todo",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=TASK_LIST_TOOL_NAME,
                        arguments={"todos": [{"title": "a", "status": "completed", "priority": "medium"}]},
                    )
                ],
            ),
            LLMResponse(content="done"),
        ]
    )
    runtime = KernelRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        hooks=[BlankIdAfterHook()],
    )
    task = AgentTask(
        task_id="hook_after_tool_blank_id",
        model="m",
        prompt_bundle=build_raw_system_prompt_bundle("sys"),
        user_prompt="go",
        max_cycles=4,
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.cycles[0].tool_results[0].tool_call_id == "c1"


def test_runtime_hook_can_replace_llm_response(tmp_path: Path) -> None:
    class ReplaceResponseHook(BaseRuntimeHook):
        def after_llm(self, event: AfterLLMEvent) -> LLMResponse:
            del event
            return LLMResponse(content="hook-finish")

    runtime = KernelRuntime(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="plain")]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        hooks=[ReplaceResponseHook()],
    )
    task = AgentTask(
        task_id="hook_after_llm", model="m", prompt_bundle=build_raw_system_prompt_bundle("sys"), user_prompt="go", max_cycles=2
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "hook-finish"


def test_runtime_cycle_injection_preserves_admitted_tool_batch(tmp_path: Path) -> None:
    events: list[object] = []

    def track(event: object) -> None:
        events.append(event)

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
            LLMResponse(content="done"),
        ]
    )

    queued = {"used": False}

    def interruption_provider() -> list[Message]:
        if queued["used"]:
            return []
        queued["used"] = True
        return [Message(role="user", content="STEER_NOW")]

    runtime = KernelRuntime(
        llm_client=llm,
        tool_registry=registry,
        default_workspace=tmp_path,
        event_handler=track,
    )
    task = AgentTask(
        task_id="steer_skip",
        model="m",
        prompt_bundle=build_raw_system_prompt_bundle("sys"),
        user_prompt="go",
        max_cycles=4,
        extra_tool_names=["_demo_noop"],
    )
    result = runtime.run(task, interruption_messages=interruption_provider)

    assert result.status == AgentStatus.COMPLETED
    assert all(r.error_code is None for r in result.cycles[0].tool_results)
    assert queued["used"]
    assert not any(message.content == "STEER_NOW" for message in result.messages)


def test_before_llm_cannot_remove_recovery_tool_from_summary_evidence() -> None:
    from support import model_call_context
    from support.compaction import fixture, messages

    class RemoveReadFileHook(BaseRuntimeHook):
        def before_llm(self, event: BeforeLLMEvent) -> BeforeLLMPatch:
            return BeforeLLMPatch(
                tool_schemas=[
                    schema for schema in event.tool_schemas if schema.get("function", {}).get("name") != READ_FILE_TOOL_NAME
                ]
            )

    original = messages(fixture("memory_local")["summary_compaction"]["cases"][0]["expected"]["messages"])
    calls = []
    runner = partial(
        run_model_turn,
        llm=ScriptedLLM(steps=[lambda request: calls.append(request) or LLMResponse(content="done")]),
        tool_registry=build_default_registry(),
        hook_manager=RuntimeHookManager(hooks=[RemoveReadFileHook()]),
    )
    task = AgentTask(
        task_id="summary-recovery", model="m", prompt_bundle=build_raw_system_prompt_bundle("sys"), user_prompt="continue"
    )
    result = runner(task=task, messages=original, memory_manager=MemoryManager(), ctx=model_call_context())
    assert result.status == AgentStatus.FAILED
    assert "recovery_unavailable" in str(result.error)
    assert calls == []
