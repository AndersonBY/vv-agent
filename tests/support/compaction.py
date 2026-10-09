"""Real compaction producer harness for the canonical contract vectors."""

from __future__ import annotations

import json
from pathlib import Path
from types import EllipsisType
from typing import Any

import pytest

from vv_agent.memory import MemoryManager
from vv_agent.memory.manager import SummaryCallback
from vv_agent.types import LLMResponse, Message
from vv_agent.workspace import MemoryWorkspaceBackend

FIXTURES = Path(__file__).parents[1] / "fixtures" / "parity"


def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def messages(raw: list[dict[str, Any]]) -> list[Message]:
    return [Message.from_dict(item) for item in raw]


def section(prompt: str, name: str) -> Any:
    return json.loads(prompt.split(f"<{name}>\n", 1)[1].split(f"\n</{name}>", 1)[0])


def case_manager(
    case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    language: str = "zh-CN",
    summary_event_limit: int = 40,
    summary_callback: SummaryCallback | None | EllipsisType = ...,
) -> tuple[MemoryManager, list[str]]:
    inputs = case["input"]
    captured: list[str] = []

    def summarize(prompt: str, _backend: str | None, _model: str | None) -> str | None:
        captured.append(prompt)
        callback = inputs.get("callback", {"kind": "return", "value": inputs.get("summary_response")})
        if callback["kind"] == "raise":
            raise RuntimeError(callback["error"])
        return callback.get("value")

    class NoFileReads(MemoryWorkspaceBackend):
        def read_text(self, path: str) -> str:
            raise AssertionError(f"automatic read: {path}")

        def exists(self, path: str) -> bool:
            raise AssertionError(f"automatic exists: {path}")

        def stat(self, path: str) -> Any:
            raise AssertionError(f"automatic stat: {path}")

    manager = MemoryManager(
        language=language,
        summary_event_limit=summary_event_limit,
        keep_recent_messages=inputs["keep_recent_messages"],
        recovery_tool_available=inputs.get("recovery_tool_available", True),
        workspace_backend=NoFileReads(),
        summary_callback=summary_callback
        if summary_callback is not ...
        else (None if inputs.get("callback", {}).get("kind") == "absent" else summarize),
    )
    estimator = case.get("token_estimator")
    if estimator:
        original = inputs["messages"]

        def estimate(_self: MemoryManager, value: list[Message]) -> int:
            return estimator["input_messages"] if [m.to_dict() for m in value] == original else estimator["candidate_messages"]

        monkeypatch.setattr(MemoryManager, "_calculate_message_length", estimate)
    return manager, captured


def assert_case(case: dict[str, Any], result: list[Message], changed: bool, captured: list[str]) -> None:
    expected = case["expected"]
    assert changed is expected["changed"]
    assert [m.to_dict() for m in result] == expected["messages"]
    assert len(captured) == expected["summary_calls"]
    if captured and case.get("expected_summary_input") is not None:
        expected_input = case["expected_summary_input"]
        assert section(captured[0], "Previous Summary") == expected_input["previous_summary"]
        assert section(captured[0], "Conversation Prefix") == expected_input["conversation_prefix"]


def prune_case_manager(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> MemoryManager:
    from vv_agent.microcompaction import MicrocompactionPolicy

    inputs = case["input"]
    backend = MemoryWorkspaceBackend()
    original = messages(inputs["messages"])
    for message in original:
        if message.artifact_ref:
            backend.write_text_exclusive(message.artifact_ref.path, inputs["existing_artifact_text"])
    manager = MemoryManager(
        compact_threshold=inputs["compact_threshold"],
        keep_recent_messages=inputs["keep_recent_messages"],
        model_context_window=10000,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        microcompaction_policy=MicrocompactionPolicy.from_dict(inputs["microcompaction_policy"]),
        workspace_backend=backend,
        recovery_tool_available=True,
    )
    estimate = case["token_estimator"]
    monkeypatch.setattr(
        MemoryManager,
        "_calculate_message_length",
        lambda _self, ms: estimate["input_messages"] if ms == original else estimate["candidate_messages"],
    )
    monkeypatch.setattr(
        MemoryManager,
        "_estimate_message_tokens",
        lambda _self, m: (
            estimate["replacement_result"] if m.content.startswith("<Tool Result Compact>") else estimate["old_result"]
        ),
    )
    return manager


def run_model_turn(*, llm, tool_registry, task, messages, memory_manager, ctx=None, hook_manager=None):
    """Drive the current producer with scripted internal-model responses and a seeded transcript."""
    from dataclasses import replace

    from vv_agent import Agent, RunConfig, ScriptedModelProvider
    from vv_agent.runtime.context import ExecutionContext
    from vv_agent.session.surfaces import SessionDriver

    ctx = ctx or ExecutionContext()
    metadata = dict(task.metadata)
    memory = memory_manager.session_memory
    if memory is not None:
        metadata.update(
            session_memory_enabled=True,
            session_memory_min_tokens=memory.config.min_tokens_before_extraction,
            session_memory_min_text_messages=memory.config.min_text_messages,
        )
        metadata["vv_session"] = {"memory_initial_state": memory.state.to_dict()}
    task = replace(task, metadata=metadata, no_tool_policy="finish")

    def response(request):
        purpose = request.metadata.get("purpose")
        if purpose == "compaction":
            callback = memory_manager.summary_callback
            return LLMResponse(callback(request.messages[0].content, None, task.model) or "" if callback else "")
        if purpose == "session_memory":
            assert memory is not None and memory.config.extraction_callback is not None
            return LLMResponse(memory.config.extraction_callback(request.messages[0].content, None, task.model) or "")
        return llm.complete(request)

    provider = ScriptedModelProvider.from_callback("test", task.model, response).with_token_limits(
        memory_manager.model_context_window, memory_manager.model_max_output_tokens
    )
    config = RunConfig(
        model_provider=provider,
        tool_registry_factory=lambda: tool_registry,
        hooks=list(hook_manager.hooks) if hook_manager else [],
        memory_providers=ctx.metadata.get("_vv_agent_memory_providers", []),
        stream=lambda event: ctx.event_handler(event) if ctx.event_handler and event.type.startswith("memory_compact_") else None,
        workspace_backend=memory_manager.workspace_backend,
    )
    seed = list(messages)
    content = seed.pop().content if seed and seed[-1].role == "user" else task.user_prompt
    driver = SessionDriver()
    try:
        driver.create(task.task_id, ".", {"seed": {"messages": [m.to_dict() for m in seed], "shared_state": {}}})
        handle = driver.start(
            task.task_id, Agent("test", task.prompt_bundle, model=task.model), config, content, task=task, autostart=False
        )
        handle.runtime.memory_manager = memory_manager
        handle.start()
        result = handle.result().raw_result
        return result
    finally:
        driver.close()
