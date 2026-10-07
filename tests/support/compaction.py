"""Real compaction producer harness for the canonical contract vectors."""

from __future__ import annotations

import json
from pathlib import Path
from types import EllipsisType
from typing import Any

import pytest

from vv_agent.memory import MemoryManager
from vv_agent.memory.manager import SummaryCallback
from vv_agent.types import Message
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
