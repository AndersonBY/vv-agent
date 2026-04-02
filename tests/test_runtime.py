from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from vv_agent.constants import (
    ACTIVATE_SKILL_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    READ_IMAGE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
)
from vv_agent.llm import ScriptedLLM
from vv_agent.memory import SessionMemoryEntry, SessionMemoryState
from vv_agent.runtime import AgentRuntime
from vv_agent.tools import ToolContext, ToolSpec, build_default_registry
from vv_agent.types import (
    AgentStatus,
    AgentTask,
    LLMResponse,
    Message,
    SubAgentConfig,
    SubTaskRequest,
    ToolCall,
    ToolExecutionResult,
)

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


def test_runtime_emits_run_max_cycles_log(tmp_path: Path) -> None:
    llm = ScriptedLLM(steps=[LLMResponse(content="step1"), LLMResponse(content="step2")])
    events: list[tuple[str, dict[str, object]]] = []

    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        log_handler=lambda event, payload: events.append((event, payload)),
    )
    task = AgentTask(
        task_id="task4_log",
        model="m",
        system_prompt="sys",
        user_prompt="do",
        max_cycles=2,
        no_tool_policy="continue",
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.MAX_CYCLES

    max_cycles_payload = next(payload for event, payload in events if event == "run_max_cycles")
    assert max_cycles_payload["cycle"] == 2
    assert max_cycles_payload["final_answer"] == "Reached max cycles without finish signal."


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
    cycle_payload = next(payload for name, payload in events if name == "cycle_llm_response")
    assert "token_usage" in cycle_payload
    assert isinstance(cycle_payload["token_usage"], dict)
    assert isinstance(cycle_payload.get("assistant_message"), str)
    assert isinstance(cycle_payload.get("tool_calls"), list)
    assert cycle_payload.get("memory_compacted") is False
    tool_calls = cycle_payload["tool_calls"]
    assert tool_calls
    first_tool_call = tool_calls[0] if isinstance(tool_calls, list) else None
    assert isinstance(first_tool_call, dict)
    first_tool_call_map = cast(dict[str, Any], first_tool_call)
    assert first_tool_call_map.get("name") == TODO_WRITE_TOOL_NAME
    assert isinstance(first_tool_call_map.get("arguments"), dict)
    assert cycle_payload.get("tool_call_names") == [TODO_WRITE_TOOL_NAME]

    tool_payload = next(payload for name, payload in events if name == "tool_result")
    assert isinstance(tool_payload.get("content"), str)
    assert "result" not in tool_payload
    assert isinstance(tool_payload.get("metadata"), dict)


def test_runtime_build_memory_manager_uses_model_token_limits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vv_agent.runtime.engine.resolve_model_token_limits", lambda _model: (64_000, 8_000))

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="task_memory_defaults",
        model="demo-model",
        system_prompt="sys",
        user_prompt="run",
    )

    manager = runtime._build_memory_manager(task=task, workspace_path=tmp_path)

    assert manager.model == "demo-model"
    assert manager.model_context_window == 64_000
    assert manager.reserved_output_tokens == 8_000
    assert manager.autocompact_buffer_tokens == 13_000
    assert manager.autocompact_threshold == 43_000
    assert manager.session_memory is not None
    assert manager.session_memory.config.extraction_model == "demo-model"


def test_runtime_build_memory_manager_metadata_overrides_model_token_limits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vv_agent.runtime.engine.resolve_model_token_limits", lambda _model: (64_000, 8_000))

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="task_memory_override",
        model="demo-model",
        system_prompt="sys",
        user_prompt="run",
        metadata={
            "model_context_window": 32_000,
            "reserved_output_tokens": 4_000,
            "autocompact_buffer_tokens": 2_000,
            "microcompact_trigger_ratio": 0.5,
            "microcompact_keep_recent_cycles": 5,
            "microcompact_min_result_length": 900,
            "microcompact_compactable_tools": ["read_file", "bash"],
            "session_memory_min_tokens": 2_000,
            "session_memory_max_tokens": 9_000,
            "session_memory_min_text_messages": 7,
            "session_memory_storage_dir": ".custom/session-memory",
        },
    )

    manager = runtime._build_memory_manager(task=task, workspace_path=tmp_path)

    assert manager.model_context_window == 32_000
    assert manager.reserved_output_tokens == 4_000
    assert manager.autocompact_buffer_tokens == 2_000
    assert manager.microcompact_trigger_ratio == 0.5
    assert manager.microcompact_keep_recent_cycles == 5
    assert manager.microcompact_min_result_length == 900
    assert manager.microcompact_compactable_tools == {"read_file", "bash"}
    assert manager.session_memory is not None
    assert manager.session_memory.config.min_tokens_before_extraction == 2_000
    assert manager.session_memory.config.max_tokens == 9_000
    assert manager.session_memory.config.min_text_messages == 7
    assert manager.session_memory.config.storage_dir == ".custom/session-memory"


def test_runtime_injects_loaded_session_memory_into_system_prompt(tmp_path: Path) -> None:
    storage_path = tmp_path / ".memory" / "session" / "task_session_memory" / "session_memory.json"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_path.write_text(
        json.dumps(
            SessionMemoryState(
                entries=[
                    SessionMemoryEntry(
                        category="key_fact",
                        content="persisted workspace fact",
                        source_cycle=3,
                        importance=9,
                    )
                ],
                initialized=True,
            ).to_dict(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def assert_session_memory(_model: str, messages: list[Message]) -> LLMResponse:
        assert messages[0].role == "system"
        assert "<Session Memory>" in messages[0].content
        assert "persisted workspace fact" in messages[0].content
        return LLMResponse(content="done")

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[assert_session_memory]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="task_session_memory",
        model="demo-model",
        system_prompt="sys",
        user_prompt="run",
        max_cycles=1,
        no_tool_policy="finish",
    )

    result = runtime.run(task)

    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "done"


def test_runtime_does_not_load_session_memory_from_other_task_scope(tmp_path: Path) -> None:
    scoped_storage_path = tmp_path / ".memory" / "session" / "task_with_memory" / "session_memory.json"
    scoped_storage_path.parent.mkdir(parents=True, exist_ok=True)
    scoped_storage_path.write_text(
        json.dumps(
            SessionMemoryState(
                entries=[
                    SessionMemoryEntry(
                        category="key_fact",
                        content="old task fact",
                        source_cycle=2,
                        importance=7,
                    )
                ],
                initialized=True,
            ).to_dict(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def assert_no_session_memory(_model: str, messages: list[Message]) -> LLMResponse:
        assert messages[0].role == "system"
        assert "<Session Memory>" not in messages[0].content
        assert "old task fact" not in messages[0].content
        return LLMResponse(content="done")

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[assert_no_session_memory]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    task = AgentTask(
        task_id="fresh_task",
        model="demo-model",
        system_prompt="sys",
        user_prompt="run",
        max_cycles=1,
        no_tool_policy="finish",
    )

    result = runtime.run(task)

    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "done"


def test_runtime_disables_session_memory_for_subtasks_by_default(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
    )
    parent_task = AgentTask(
        task_id="parent",
        model="demo-model",
        system_prompt="sys",
        user_prompt="run",
        metadata={"language": "en-US"},
        sub_agents={
            "worker": SubAgentConfig(
                model="demo-model",
                description="Worker agent",
            )
        },
    )
    sub_task = runtime._build_sub_agent_task(
        parent_task=parent_task,
        sub_task_id="sub-1",
        sub_session_id="session-sub-1",
        sub_agent_name="worker",
        sub_agent=parent_task.sub_agents["worker"],
        resolved_model_id="demo-model",
        request=SubTaskRequest(agent_name="worker", task_description="do work"),
        parent_shared_state={},
        workspace_path=tmp_path,
    )

    manager = runtime._build_memory_manager(task=sub_task, workspace_path=tmp_path)

    assert sub_task.metadata["session_memory_enabled"] is False
    assert manager.session_memory is None


def test_runtime_uses_prompt_tokens_for_followup_compaction_budget(tmp_path: Path) -> None:
    observed_cycle_two_messages: list[Message] = []

    def inspect_cycle_two_messages(_model: str, messages: list[Message]) -> LLMResponse:
        observed_cycle_two_messages.extend(messages)
        return LLMResponse(
            content="finish",
            tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
        )

    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="step1",
                raw={"usage": {"prompt_tokens": 10, "completion_tokens": 500, "total_tokens": 510}},
            ),
            inspect_cycle_two_messages,
        ]
    )
    runtime = AgentRuntime(llm_client=llm, tool_registry=build_default_registry(), default_workspace=tmp_path)
    task = AgentTask(
        task_id="task_prompt_budget",
        model="dummy-model",
        system_prompt="sys",
        user_prompt="keep context",
        max_cycles=2,
        no_tool_policy="continue",
        metadata={
            "model_context_window": 100,
            "reserved_output_tokens": 10,
            "autocompact_buffer_tokens": 10,
        },
    )

    result = runtime.run(task)

    assert result.status == AgentStatus.COMPLETED
    assert observed_cycle_two_messages
    assert all("<Compressed Agent Memory>" not in message.content for message in observed_cycle_two_messages)


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


def test_runtime_tool_result_event_keeps_full_content_by_default(tmp_path: Path) -> None:
    long_title = "x" * 500
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="write todo",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=TODO_WRITE_TOOL_NAME,
                        arguments={
                            "todos": [{"title": long_title, "status": "completed", "priority": "medium"}]
                        },
                    )
                ],
            ),
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
            ),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []

    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        log_handler=lambda event, payload: events.append((event, payload)),
    )
    task = AgentTask(
        task_id="task_long_tool_result",
        model="dummy-model",
        system_prompt="sys",
        user_prompt="go",
        max_cycles=4,
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED

    tool_payload = next(payload for name, payload in events if name == "tool_result")
    full_result = str(tool_payload.get("content") or "")
    assert long_title in full_result
    assert len(full_result) > 220
    assert str(tool_payload.get("content_preview") or "") == full_result


def test_runtime_tool_result_event_preview_can_be_truncated_explicitly(tmp_path: Path) -> None:
    long_title = "y" * 500
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="write todo",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=TODO_WRITE_TOOL_NAME,
                        arguments={
                            "todos": [{"title": long_title, "status": "completed", "priority": "medium"}]
                        },
                    )
                ],
            ),
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
            ),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []

    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        log_handler=lambda event, payload: events.append((event, payload)),
        log_preview_chars=120,
    )
    task = AgentTask(
        task_id="task_long_tool_result_truncated",
        model="dummy-model",
        system_prompt="sys",
        user_prompt="go",
        max_cycles=4,
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED

    tool_payload = next(payload for name, payload in events if name == "tool_result")
    full_result = str(tool_payload.get("content") or "")
    preview = str(tool_payload.get("content_preview") or "")
    assert long_title in full_result
    assert full_result != preview
    assert preview.endswith("...")
    assert len(preview) <= 120
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
        metadata={"available_skills": [{"name": "demo", "description": "Demo skill", "instructions": "Use this"}]},
    )
    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.shared_state["active_skills"] == ["demo"]


def test_runtime_propagates_available_skills_path_list_into_tool_context(tmp_path: Path) -> None:
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
        metadata={"available_skills": ["skills"]},
    )
    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.shared_state["active_skills"] == ["demo"]


def test_memory_summary_model_priority_uses_metadata_over_local_settings(tmp_path: Path) -> None:
    settings_file = tmp_path / "local_settings.py"
    settings_file.write_text(
        '\n'.join(
            (
                'DEFAULT_USER_MEMORY_SUMMARIZE_BACKEND = "settings-backend"',
                'DEFAULT_USER_MEMORY_SUMMARIZE_MODEL = "settings-model"',
            )
        ),
        encoding="utf-8",
    )

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        settings_file=settings_file,
        default_backend="fallback-backend",
    )
    task = AgentTask(
        task_id="task_memory_priority_metadata",
        model="task-model",
        system_prompt="sys",
        user_prompt="go",
        metadata={
            "memory_summary_backend": "metadata-backend",
            "memory_summary_model": "metadata-model",
        },
    )

    manager = runtime._build_memory_manager(task=task, workspace_path=tmp_path)
    assert manager.summary_backend == "metadata-backend"
    assert manager.summary_model == "metadata-model"


def test_memory_summary_model_priority_uses_local_settings_over_fallback(tmp_path: Path) -> None:
    settings_file = tmp_path / "local_settings.py"
    settings_file.write_text(
        '\n'.join(
            (
                'DEFAULT_USER_MEMORY_SUMMARIZE_BACKEND = "settings-backend"',
                'DEFAULT_USER_MEMORY_SUMMARIZE_MODEL = "settings-model"',
            )
        ),
        encoding="utf-8",
    )

    runtime = AgentRuntime(
        llm_client=ScriptedLLM(),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        settings_file=settings_file,
        default_backend="fallback-backend",
    )
    task = AgentTask(
        task_id="task_memory_priority_settings",
        model="task-model",
        system_prompt="sys",
        user_prompt="go",
    )

    manager = runtime._build_memory_manager(task=task, workspace_path=tmp_path)
    assert manager.summary_backend == "settings-backend"
    assert manager.summary_model == "settings-model"


def test_memory_summary_model_priority_uses_fallback_when_missing(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        default_backend="fallback-backend",
    )
    task = AgentTask(
        task_id="task_memory_priority_fallback",
        model="task-model",
        system_prompt="sys",
        user_prompt="go",
    )

    manager = runtime._build_memory_manager(task=task, workspace_path=tmp_path)
    assert manager.summary_backend == "fallback-backend"
    assert manager.summary_model == "task-model"
