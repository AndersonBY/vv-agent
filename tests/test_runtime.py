from __future__ import annotations

from pathlib import Path

from v_agent.constants import ASK_USER_TOOL_NAME, TASK_FINISH_TOOL_NAME, TODO_WRITE_TOOL_NAME
from v_agent.llm import ScriptedLLM
from v_agent.runtime import AgentRuntime
from v_agent.tools import build_default_registry
from v_agent.types import AgentStatus, AgentTask, LLMResponse, ToolCall


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
