from __future__ import annotations

from pathlib import Path

from vv_agent.constants import (
    ACTIVATE_SKILL_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    READ_IMAGE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
)
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime import AgentRuntime
from vv_agent.tools import ToolContext, ToolSpec, build_default_registry
from vv_agent.types import AgentStatus, AgentTask, LLMResponse, Message, ToolCall, ToolExecutionResult

_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a"
    "0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360000000020001e221bc330000000049454e44ae426082"
)


def test_runtime_finishes_via_task_finish(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="planning",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=TODO_WRITE_TOOL_NAME,
                        arguments={"todos": [{"title": "draft", "status": "completed", "priority": "medium"}]},
                    )
                ],
            ),
            LLMResponse(
                content="finalizing",
                tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "all done"})],
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=llm, tool_registry=build_default_registry(), default_workspace=tmp_path)

    task = AgentTask(
        task_id="task1",
        model="dummy-model",
        system_prompt="sys",
        user_prompt="finish this",
        max_cycles=4,
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "all done"
    assert len(result.cycles) == 2


def test_runtime_waits_for_user_when_ask_user_called(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="need input",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=ASK_USER_TOOL_NAME,
                        arguments={"question": "confirm?", "options": ["yes", "no"]},
                    )
                ],
            )
        ]
    )
    runtime = AgentRuntime(llm_client=llm, tool_registry=build_default_registry(), default_workspace=tmp_path)
    task = AgentTask(task_id="task2", model="m", system_prompt="sys", user_prompt="ask", max_cycles=3)

    result = runtime.run(task)
    assert result.status == AgentStatus.WAIT_USER
    assert "confirm" in (result.wait_reason or "")


def test_runtime_retries_after_todo_guard_error(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="create todo",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=TODO_WRITE_TOOL_NAME,
                        arguments={"todos": [{"title": "t1", "status": "pending", "priority": "medium"}]},
                    )
                ],
            ),
            LLMResponse(
                content="try finish",
                tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
            LLMResponse(
                content="mark done",
                tool_calls=[
                    ToolCall(
                        id="c3",
                        name=TODO_WRITE_TOOL_NAME,
                        arguments={"todos": [{"title": "t1", "status": "completed", "priority": "medium"}]},
                    )
                ],
            ),
            LLMResponse(
                content="finish",
                tool_calls=[ToolCall(id="c4", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done for real"})],
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=llm, tool_registry=build_default_registry(), default_workspace=tmp_path)
    task = AgentTask(task_id="task3", model="m", system_prompt="sys", user_prompt="todo", max_cycles=6)

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "done for real"
    assert len(result.cycles) == 4
    assert result.todo_list[0]["status"] == "completed"


def test_runtime_hits_max_cycles_with_continue_policy(tmp_path: Path) -> None:
    llm = ScriptedLLM(steps=[LLMResponse(content="step1"), LLMResponse(content="step2")])
    runtime = AgentRuntime(llm_client=llm, tool_registry=build_default_registry(), default_workspace=tmp_path)
    task = AgentTask(
        task_id="task4",
        model="m",
        system_prompt="sys",
        user_prompt="do",
        max_cycles=2,
        no_tool_policy="continue",
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.MAX_CYCLES
    assert len(result.cycles) == 2


def test_runtime_can_finish_without_tool_on_policy(tmp_path: Path) -> None:
    llm = ScriptedLLM(steps=[LLMResponse(content="direct answer")])
    runtime = AgentRuntime(llm_client=llm, tool_registry=build_default_registry(), default_workspace=tmp_path)
    task = AgentTask(
        task_id="task5",
        model="m",
        system_prompt="sys",
        user_prompt="answer",
        max_cycles=3,
        no_tool_policy="finish",
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "direct answer"


def test_runtime_emits_cycle_logs(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="planning",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=TODO_WRITE_TOOL_NAME,
                        arguments={"todos": [{"title": "draft", "status": "completed", "priority": "medium"}]},
                    )
                ],
            ),
            LLMResponse(
                content="finalizing",
                tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "all done"})],
            ),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []

    def handler(event: str, payload: dict[str, object]) -> None:
        events.append((event, payload))

    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        log_handler=handler,
    )
    task = AgentTask(
        task_id="task_log",
        model="dummy-model",
        system_prompt="sys",
        user_prompt="finish this",
        max_cycles=4,
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED

    event_names = [name for name, _ in events]
    assert "run_started" in event_names
    assert "cycle_started" in event_names
    assert "cycle_llm_response" in event_names
    assert "tool_result" in event_names
    assert "run_completed" in event_names


def test_runtime_injects_image_message_after_read_image(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    image_path.write_bytes(_PNG_1X1)

    def assert_image_message(model: str, messages: list[Message]) -> LLMResponse:
        del model
        image_messages = [msg for msg in messages if msg.role == "user" and isinstance(msg.image_url, str)]
        assert image_messages, "Expected image message with image_url in runtime history."
        assert image_messages[-1].content.startswith("[Image loaded]")
        assert image_messages[-1].image_url is not None
        assert image_messages[-1].image_url.startswith("data:image/png;base64,")
        return LLMResponse(
            content="done",
            tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
        )

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="read image",
                tool_calls=[ToolCall(id="c1", name=READ_IMAGE_TOOL_NAME, arguments={"path": "img.png"})],
            ),
            assert_image_message,
        ]
    )
    runtime = AgentRuntime(llm_client=llm, tool_registry=build_default_registry(), default_workspace=tmp_path)
    task = AgentTask(
        task_id="task_img",
        model="m",
        system_prompt="sys",
        user_prompt="read image",
        max_cycles=4,
        native_multimodal=True,
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "ok"


def test_runtime_keeps_tool_results_adjacent_before_image_notifications(tmp_path: Path) -> None:
    def _demo_image(context: ToolContext, arguments: dict[str, object]) -> ToolExecutionResult:
        del context, arguments
        return ToolExecutionResult(
            tool_call_id="",
            content='{"ok": true}',
            image_url="data:image/png;base64,AAAA",
        )

    def assert_order(model: str, messages: list[Message]) -> LLMResponse:
        del model
        assistant_index = next(
            index
            for index, message in enumerate(messages)
            if message.role == "assistant" and message.tool_calls
        )
        first_tool = messages[assistant_index + 1]
        second_tool = messages[assistant_index + 2]
        assert first_tool.role == "tool"
        assert first_tool.tool_call_id == "img1"
        assert second_tool.role == "tool"
        assert second_tool.tool_call_id == "todo1"

        image_messages = [msg for msg in messages[assistant_index + 3 :] if msg.role == "user" and msg.image_url]
        assert image_messages
        assert image_messages[0].content == ""
        return LLMResponse(
            content="done",
            tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
        )

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="run tools",
                tool_calls=[
                    ToolCall(id="img1", name="_demo_image", arguments={}),
                    ToolCall(
                        id="todo1",
                        name=TODO_WRITE_TOOL_NAME,
                        arguments={"todos": [{"title": "done", "status": "completed", "priority": "medium"}]},
                    ),
                ],
            ),
            assert_order,
        ]
    )
    registry = build_default_registry()
    registry.register(ToolSpec(name="_demo_image", handler=_demo_image))
    runtime = AgentRuntime(llm_client=llm, tool_registry=registry, default_workspace=tmp_path)
    task = AgentTask(
        task_id="task_image_order",
        model="m",
        system_prompt="sys",
        user_prompt="go",
        max_cycles=4,
        native_multimodal=True,
        extra_tool_names=["_demo_image"],
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "ok"


def test_runtime_skips_image_notifications_when_multimodal_disabled(tmp_path: Path) -> None:
    def _demo_image(context: ToolContext, arguments: dict[str, object]) -> ToolExecutionResult:
        del context, arguments
        return ToolExecutionResult(
            tool_call_id="",
            content='{"ok": true}',
            image_url="data:image/png;base64,AAAA",
        )

    def assert_no_image_message(model: str, messages: list[Message]) -> LLMResponse:
        del model
        assert not any(message.role == "user" and message.image_url for message in messages)
        return LLMResponse(
            content="done",
            tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
        )

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="capture",
                tool_calls=[ToolCall(id="img1", name="_demo_image", arguments={})],
            ),
            assert_no_image_message,
        ]
    )
    registry = build_default_registry()
    registry.register(ToolSpec(name="_demo_image", handler=_demo_image))
    runtime = AgentRuntime(llm_client=llm, tool_registry=registry, default_workspace=tmp_path)
    task = AgentTask(
        task_id="task_no_multimodal",
        model="m",
        system_prompt="sys",
        user_prompt="go",
        max_cycles=4,
        extra_tool_names=["_demo_image"],
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "ok"


def test_runtime_collects_cycle_and_total_token_usage(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="planning",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=TODO_WRITE_TOOL_NAME,
                        arguments={"todos": [{"title": "draft", "status": "completed", "priority": "medium"}]},
                    )
                ],
                raw={
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 25,
                        "total_tokens": 125,
                        "prompt_tokens_details": {"cached_tokens": 40},
                        "completion_tokens_details": {"reasoning_tokens": 10},
                    }
                },
            ),
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
                raw={
                    "usage": {
                        "input_tokens": 50,
                        "output_tokens": 30,
                        "total_tokens": 80,
                        "input_tokens_details": {"cache_creation_tokens": 12},
                    }
                },
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=llm, tool_registry=build_default_registry(), default_workspace=tmp_path)
    task = AgentTask(task_id="task_usage", model="m", system_prompt="sys", user_prompt="go", max_cycles=4)

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED

    assert len(result.token_usage.cycles) == 2
    assert result.token_usage.cycles[0].usage.prompt_tokens == 100
    assert result.token_usage.cycles[0].usage.completion_tokens == 25
    assert result.token_usage.cycles[0].usage.cached_tokens == 40
    assert result.token_usage.cycles[1].usage.prompt_tokens == 50
    assert result.token_usage.cycles[1].usage.completion_tokens == 30
    assert result.token_usage.cycles[1].usage.cache_creation_tokens == 12

    assert result.token_usage.prompt_tokens == 150
    assert result.token_usage.completion_tokens == 55
    assert result.token_usage.total_tokens == 205
    assert result.token_usage.cached_tokens == 40
    assert result.token_usage.reasoning_tokens == 10
    assert result.token_usage.cache_creation_tokens == 12


def test_runtime_propagates_available_skills_into_tool_context(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="activate skill",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=ACTIVATE_SKILL_TOOL_NAME,
                        arguments={"skill_name": "demo"},
                    )
                ],
            ),
            LLMResponse(
                content="finish",
                tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=llm, tool_registry=build_default_registry(), default_workspace=tmp_path)
    task = AgentTask(
        task_id="task_skill",
        model="m",
        system_prompt="sys",
        user_prompt="activate",
        max_cycles=4,
        metadata={"available_skills": [{"name": "demo", "instructions": "Use this"}]},
    )
    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.shared_state["active_skills"] == ["demo"]


def test_runtime_propagates_skill_directories_into_tool_context(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="activate skill",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=ACTIVATE_SKILL_TOOL_NAME,
                        arguments={"skill_name": "demo"},
                    )
                ],
            ),
            LLMResponse(
                content="finish",
                tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
        ]
    )

    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: demo
description: demo skill
---
Body
""",
        encoding="utf-8",
    )

    runtime = AgentRuntime(llm_client=llm, tool_registry=build_default_registry(), default_workspace=tmp_path)
    task = AgentTask(
        task_id="task_skill_dir",
        model="m",
        system_prompt="sys",
        user_prompt="activate",
        max_cycles=4,
        metadata={"skill_directories": ["skills"]},
    )
    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.shared_state["active_skills"] == ["demo"]
