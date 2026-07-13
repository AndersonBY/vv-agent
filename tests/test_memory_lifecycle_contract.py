from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from vv_agent.config import ResolvedModelConfig
from vv_agent.llm import LlmRequest, ScriptedLLM
from vv_agent.memory import (
    MemoryManager,
    MemoryProviderResult,
    MemorySaveResult,
    SessionMemory,
    SessionMemoryConfig,
    SessionMemoryEntry,
)
from vv_agent.runtime import AgentRuntime
from vv_agent.runtime.context import ExecutionContext
from vv_agent.runtime.cycle_runner import CycleRunner
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentStatus, AgentTask, LLMResponse, Message

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "memory_lifecycle_v1.json"
_CONTRACT = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _summary_payload() -> str:
    return json.dumps(
        {
            "summary_version": "2.0",
            "original_user_messages": ["original"],
            "user_constraints": [],
            "decisions": [],
            "files_examined_or_modified": [],
            "errors_and_fixes": [],
            "progress": ["done"],
            "key_facts": [],
            "open_issues": [],
            "current_work_state": "done",
            "next_steps": [],
        },
        ensure_ascii=False,
    )


def test_runtime_routes_summary_through_configured_backend_model_pair(tmp_path: Path) -> None:
    contract = _CONTRACT["summary_route"]
    builds: list[tuple[str, str]] = []
    requests: list[LlmRequest] = []

    def summarize(request: LlmRequest) -> LLMResponse:
        requests.append(request)
        return LLMResponse(content=_summary_payload())

    summary_llm = ScriptedLLM(steps=[summarize])

    def llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        builds.append((backend, model))
        return summary_llm, ResolvedModelConfig(
            backend=backend,
            requested_model=model,
            selected_model=model,
            model_id=model,
            endpoint_options=[],
        )

    settings_file = tmp_path / "settings.py"
    settings_file.write_text("", encoding="utf-8")
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="done")]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        settings_file=settings_file,
        llm_builder=llm_builder,
    )
    task = AgentTask(
        task_id="memory_route",
        model="main-model",
        system_prompt="system",
        user_prompt="continue",
        initial_messages=[
            Message(role="system", content="system"),
            Message(role="user", content="u" * 160),
            Message(role="assistant", content="a" * 160),
            Message(role="user", content="c" * 160),
        ],
        max_cycles=1,
        no_tool_policy="finish",
        memory_compact_threshold=40,
        metadata={
            "memory_summary_backend": contract["backend"],
            "memory_summary_model": contract["model"],
            "model_context_window": 60,
            "reserved_output_tokens": 10,
            "autocompact_buffer_tokens": 10,
            "session_memory_enabled": False,
        },
    )

    result = runtime.run(task)

    assert result.status == AgentStatus.COMPLETED
    assert builds == [(contract["backend"], contract["model"])]
    assert len(builds) == contract["resolution_count"]
    assert [request.model for request in requests] == [contract["request_model"]]


def test_runtime_routes_session_extraction_through_its_own_backend_model_pair(
    tmp_path: Path,
) -> None:
    contract = _CONTRACT["session_extraction_route"]
    builds: list[tuple[str, str]] = []
    extraction_requests: list[LlmRequest] = []

    def extract(request: LlmRequest) -> LLMResponse:
        extraction_requests.append(request)
        return LLMResponse(
            content='[{"category":"decision","content":"route extraction separately","importance":8}]'
        )

    extraction_llm = ScriptedLLM(steps=[extract])

    def llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        builds.append((backend, model))
        return extraction_llm, ResolvedModelConfig(
            backend=backend,
            requested_model=model,
            selected_model=model,
            model_id=model,
            endpoint_options=[],
        )

    def assert_injected(request: LlmRequest) -> LLMResponse:
        assert "route extraction separately" in request.messages[0].content
        return LLMResponse(content="done")

    settings_file = tmp_path / "settings.py"
    settings_file.write_text("", encoding="utf-8")
    runtime = AgentRuntime(
        llm_client=ScriptedLLM(steps=[assert_injected]),
        tool_registry=build_default_registry(),
        default_workspace=tmp_path,
        settings_file=settings_file,
        llm_builder=llm_builder,
    )
    task = AgentTask(
        task_id="session_memory_route",
        model="main-model",
        system_prompt="system",
        user_prompt="remember this decision",
        max_cycles=1,
        no_tool_policy="finish",
        memory_compact_threshold=10_000,
        metadata={
            "session_memory_enabled": True,
            "session_memory_min_tokens": 1,
            "session_memory_min_text_messages": 1,
            "session_memory_extraction_backend": contract["backend"],
            "session_memory_extraction_model": contract["model"],
            "model_context_window": 20_000,
            "reserved_output_tokens": 0,
            "autocompact_buffer_tokens": 0,
        },
    )

    result = runtime.run(task)

    assert result.status == AgentStatus.COMPLETED
    assert builds == [(contract["backend"], contract["model"])]
    assert len(builds) == contract["resolution_count"]
    assert [request.model for request in extraction_requests] == [contract["request_model"]]


class RecordingMemoryProvider:
    def __init__(self, *, fail_before: bool = False, fail_after: bool = False) -> None:
        self.fail_before = fail_before
        self.fail_after = fail_after
        self.started: list[Any] = []
        self.completed: list[Any] = []

    def search(self, request):
        del request
        return []

    def save(self, request):
        del request
        return MemorySaveResult()

    def before_compact(self, event):
        self.started.append(event)
        if self.fail_before:
            raise RuntimeError("before exploded")
        return MemoryProviderResult(metadata={"phase": "before"})

    def after_compact(self, event):
        self.completed.append(event)
        if self.fail_after:
            raise RuntimeError("after exploded")


def _ptl_memory_manager() -> MemoryManager:
    return MemoryManager(
        compact_threshold=10_000,
        model="main-model",
        model_context_window=20_000,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        summary_callback=lambda _prompt, _backend, _model: _summary_payload(),
    )


def _ptl_task() -> AgentTask:
    return AgentTask(
        task_id="memory_ptl",
        model="main-model",
        system_prompt="system",
        user_prompt="continue",
        no_tool_policy="finish",
    )


def test_ptl_forced_and_emergency_attempts_notify_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _CONTRACT["provider_attempts"]
    emitted: list[Any] = []
    provider = RecordingMemoryProvider()
    emergency_drop_ratios: list[float] = []
    original_emergency_compact = MemoryManager.emergency_compact

    def tracking_emergency_compact(self, messages, *, cycle_index=None, drop_ratio=0.2):
        emergency_drop_ratios.append(drop_ratio)
        return original_emergency_compact(
            self,
            messages,
            cycle_index=cycle_index,
            drop_ratio=drop_ratio,
        )

    monkeypatch.setattr(MemoryManager, "emergency_compact", tracking_emergency_compact)

    def prompt_too_long(_request: LlmRequest) -> LLMResponse:
        raise RuntimeError("Prompt is too long for this model")

    runner = CycleRunner(
        llm_client=ScriptedLLM(
            steps=[
                prompt_too_long,
                prompt_too_long,
                LLMResponse(content="done"),
            ]
        ),
        tool_registry=build_default_registry(),
    )
    next_messages, cycle = runner.run_cycle(
        task=_ptl_task(),
        messages=[
            Message(role="system", content="system"),
            Message(role="user", content="first"),
            Message(role="assistant", content="working"),
            Message(role="user", content="continue"),
        ],
        cycle_index=2,
        memory_manager=_ptl_memory_manager(),
        ctx=ExecutionContext(
            metadata={
                "_vv_agent_memory_providers": [provider],
                "_vv_agent_emit_event": emitted.append,
                "_vv_agent_run_id": "run_memory",
                "_vv_agent_trace_id": "trace_memory",
                "_vv_agent_agent_name": "assistant",
            }
        ),
    )

    assert cycle.memory_compacted is True
    assert next_messages[-1].content == "done"
    assert len(provider.started) == contract["started_count"]
    assert len(provider.completed) == contract["completed_count"]
    assert [event.type for event in emitted] == [
        "memory_compact_started",
        "memory_compact_completed",
        "memory_compact_started",
        "memory_compact_completed",
    ]
    assert emitted[0].metadata["memory_provider_results"]["RecordingMemoryProvider"] == contract["result_metadata"]
    assert emergency_drop_ratios == [contract["strategies"][1]["drop_ratio"]]


def test_memory_provider_attempt_errors_are_fail_open() -> None:
    contract = _CONTRACT["provider_attempts"]
    emitted: list[Any] = []
    provider = RecordingMemoryProvider(fail_before=True, fail_after=True)
    runner = CycleRunner(
        llm_client=ScriptedLLM(steps=[LLMResponse(content="done")]),
        tool_registry=build_default_registry(),
    )

    with pytest.warns(RuntimeWarning) as warnings:
        next_messages, cycle = runner.run_cycle(
            task=_ptl_task(),
            messages=[
                Message(role="system", content="system"),
                Message(role="user", content="u" * 120),
                Message(role="assistant", content="a" * 120),
                Message(role="user", content="c" * 120),
            ],
            cycle_index=2,
            memory_manager=MemoryManager(
                compact_threshold=40,
                model="main-model",
                model_context_window=60,
                reserved_output_tokens=10,
                autocompact_buffer_tokens=10,
                summary_callback=lambda _prompt, _backend, _model: _summary_payload(),
            ),
            previous_prompt_tokens=160,
            ctx=ExecutionContext(
                metadata={
                    "_vv_agent_memory_providers": [provider],
                    "_vv_agent_emit_event": emitted.append,
                }
            ),
        )

    assert len(warnings) == 2
    assert cycle.memory_compacted is True
    assert next_messages[-1].content == "done"
    before_error = emitted[0].metadata["memory_provider_errors"][0]
    after_error = emitted[1].metadata["memory_provider_errors"][0]
    assert before_error["stage"] == contract["before_error"]["stage"]
    assert before_error["error"] == contract["before_error"]["error"]
    assert after_error["stage"] == contract["after_error"]["stage"]
    assert after_error["error"] == contract["after_error"]["error"]


def test_session_memory_refreshes_in_place_and_resets_token_baseline() -> None:
    contract = _CONTRACT["session_memory"]
    session_memory = SessionMemory(SessionMemoryConfig())
    session_memory.state.entries = [
        SessionMemoryEntry("decision", contract["stale_fact"], source_cycle=1)
    ]
    manager = MemoryManager(
        compact_threshold=40,
        model="main-model",
        model_context_window=60,
        reserved_output_tokens=10,
        autocompact_buffer_tokens=10,
        base_system_prompt="system",
        session_memory=session_memory,
        summary_callback=lambda _prompt, _backend, _model: _summary_payload(),
    )
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="u" * 120),
        Message(role="assistant", content="a" * 120),
        Message(role="user", content="c" * 120),
    ]
    stale = manager.apply_session_memory_context(messages)
    session_memory.state.entries = [
        SessionMemoryEntry("decision", contract["fresh_fact"], source_cycle=2)
    ]

    refreshed = manager.apply_session_memory_context(stale)

    assert contract["stale_fact"] not in refreshed[0].content
    assert contract["fresh_fact"] in refreshed[0].content
    assert refreshed[0].content.count("<Session Memory>") == contract["block_count"]

    compacted, changed = manager.compact(refreshed, cycle_index=3, force=True)
    expected_baseline = manager._calculate_message_length(
        manager.apply_session_memory_context(compacted)
    )

    assert changed is True
    assert session_memory.state.initialized is contract["initialized_after_compaction"]
    assert session_memory.state.tokens_at_last_extraction == expected_baseline
    assert expected_baseline > manager._calculate_message_length(compacted)


def _artifact_path(message: Message) -> str:
    for line in message.content.splitlines():
        if line.startswith("artifact_path:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"artifact path missing from {message.content!r}")


def test_artifact_fallbacks_are_unique_and_fail_open_at_workspace_boundary(tmp_path: Path) -> None:
    contract = _CONTRACT["artifacts"]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = MemoryManager(
        workspace=workspace,
        tool_result_compact_threshold=10,
        tool_result_keep_last=0,
    )
    messages = [
        Message(role="tool", content="first artifact payload", tool_call_id="/"),
        Message(role="tool", content="second artifact payload", tool_call_id="/"),
    ]

    compacted, changed = manager._compact_tool_messages(messages, cycle_index=4)
    paths = [_artifact_path(message) for message in compacted]

    assert changed is True
    assert len(paths) == contract["fallback_count"]
    assert len(set(paths)) == len(paths)
    pattern = re.compile(contract["fallback_pattern"])
    assert all(pattern.fullmatch(PurePosixPath(path).name) for path in paths)
    assert all((workspace / path).is_file() for path in paths)

    blocked = workspace / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    failing = MemoryManager(
        workspace=workspace,
        tool_result_artifact_dir="blocked/nested",
        tool_result_compact_threshold=10,
        tool_result_keep_last=0,
    )
    failed_messages, failed_changed = failing._compact_tool_messages(
        [Message(role="tool", content="write failure payload", tool_call_id="call")],
        cycle_index=4,
    )
    assert failed_changed is True
    assert _artifact_path(failed_messages[0]) == contract["write_failure_path"]

    escaping = MemoryManager(
        workspace=workspace,
        tool_result_artifact_dir="../outside",
        tool_result_compact_threshold=10,
        tool_result_keep_last=0,
    )
    escaped_messages, escaped_changed = escaping._compact_tool_messages(
        [Message(role="tool", content="escape payload", tool_call_id="call")],
        cycle_index=4,
    )
    assert escaped_changed is True
    assert _artifact_path(escaped_messages[0]) == contract["escape_path"]
    assert not (tmp_path / "outside").exists()
