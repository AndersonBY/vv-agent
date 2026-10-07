from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from support.compaction import assert_case, case_manager
from support.compaction import messages as current_messages

from vv_agent.memory import MemoryManager, SessionMemory, SessionMemoryConfig
from vv_agent.memory.microcompact import COMPACT_MARKER_OPENING
from vv_agent.memory.token_utils import count_messages_tokens, count_tokens
from vv_agent.microcompaction import MicrocompactionPolicy
from vv_agent.tools.metadata import ToolResultRetention
from vv_agent.types import Message, ToolArtifactRef
from vv_agent.workspace import MemoryWorkspaceBackend

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "memory_local.json"
_CONTRACT: dict[str, Any] = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _messages_from_fixture(raw_messages: list[dict[str, Any]]) -> list[Message]:
    messages: list[Message] = []
    for raw_message in raw_messages:
        payload = dict(raw_message)
        normalized_tool_calls = payload.get("tool_calls")
        if isinstance(normalized_tool_calls, list):
            payload["tool_calls"] = [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(
                            tool_call["arguments"],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                }
                for tool_call in normalized_tool_calls
            ]
        messages.append(Message.from_dict(payload))
    return messages


def _first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
    raise AssertionError("expected a JSON object")


def test_memory_local_fixture_identity_and_fields() -> None:
    assert set(_CONTRACT) == {
        "contract",
        "character_unit",
        "token_counts",
        "message_tokens",
        "microcompact",
        "session_prompt_truncation",
        "summary",
        "recompression_originals",
        "unicode_excerpt",
        "session_extraction",
        "summary_parse",
        "summary_compaction",
        "evidence_manifest",
    }
    assert _CONTRACT["contract"] == "memory_local"
    assert _CONTRACT["character_unit"] == "unicode_code_point"


def test_memory_local_token_counts_match_fixture() -> None:
    token_cases = _CONTRACT["token_counts"]
    actual_cases: list[dict[str, Any]] = []
    for case in token_cases:
        text = case.get("text")
        if not isinstance(text, str):
            text = str(case["text_unit"]) * int(case["repeat"])
        actual_case = {key: value for key, value in case.items() if key != "tokens"}
        actual_case["tokens"] = count_tokens(text, model=str(case["model"]))
        actual_cases.append(actual_case)

    message_case = _CONTRACT["message_tokens"]
    actual_message_case = {key: value for key, value in message_case.items() if key != "tokens"}
    actual_message_case["tokens"] = count_messages_tokens(
        message_case["messages"],
        model=str(message_case["model"]),
    )

    assert actual_cases == token_cases
    assert actual_message_case == message_case


def test_memory_local_microcompact_boundaries_match_fixture() -> None:
    contract = _CONTRACT["microcompact"]
    actual_cases: list[dict[str, Any]] = []

    for case in contract["cases"]:
        if "repeat" not in case:
            continue

        class CaseWorkspaceBackend(MemoryWorkspaceBackend):
            def __init__(self, *, fail_writes: bool) -> None:
                super().__init__()
                self._fail_writes = fail_writes

            def write_text_exclusive(self, path: str, content: str) -> int:
                if self._fail_writes:
                    raise OSError("fixture archive failure")
                return super().write_text_exclusive(path, content)

        backend = CaseWorkspaceBackend(fail_writes=case.get("artifact_write_succeeds") is False)
        manager = MemoryManager(
            compact_threshold=1_000,
            model_context_window=1_000,
            reserved_output_tokens=0,
            autocompact_buffer_tokens=0,
            microcompaction_policy=MicrocompactionPolicy(
                keep_recent_cycles=int(contract["keep_recent_cycles_default"]),
                min_result_chars=int(contract["min_result_chars_default"]),
            ),
            tool_result_retentions={str(case["tool_name"]): ToolResultRetention(str(case["result_retention"]))},
            workspace_backend=backend,
            recovery_tool_available=True,
            artifact_scope="memory-local",
        )
        tool_content = str(contract["content_unit"]) * int(case["repeat"])
        call_id = "call_old"
        messages = [
            Message(role="system", content="system"),
            Message(
                role="assistant",
                content="old tool call",
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": case["tool_name"], "arguments": "{}"},
                    }
                ],
            ),
            Message(
                role="tool",
                content=tool_content,
                tool_call_id=call_id,
            ),
            Message(role="assistant", content="recent reply 1"),
            Message(role="assistant", content="recent reply 2"),
            Message(role="assistant", content="recent reply 3"),
        ]
        if "existing_artifact_ref" in case:
            artifact = ToolArtifactRef.from_dict(case["existing_artifact_ref"])
            backend.write_text_exclusive(artifact.path, case["persisted_utf8_text"])
            messages[2].artifact_ref = artifact

        plan = manager.plan_microcompaction(
            messages,
            cycle_index=5,
            current_tokens=1_000,
        )
        assert plan is not None
        applied = manager.apply_microcompaction(messages, plan=plan)
        actual_case: dict[str, Any] = {
            "name": case["name"],
            "repeat": case["repeat"],
            "replaced_with_compact_marker": applied.messages[2].content.startswith(COMPACT_MARKER_OPENING),
        }
        actual_cases.append(actual_case)

    expected_cases = [
        {
            "name": case["name"],
            "repeat": case["repeat"],
            "replaced_with_compact_marker": case["replaced_with_compact_marker"],
        }
        for case in contract["cases"]
        if "repeat" in case
    ]
    assert actual_cases == expected_cases


def test_memory_local_session_prompt_truncation_matches_fixture() -> None:
    contract = _CONTRACT["session_prompt_truncation"]
    unit = str(contract["content_unit"])
    actual_cases: list[dict[str, Any]] = []

    for case in contract["cases"]:
        original = unit * int(case["repeat"])
        payload = SessionMemory._message_to_text(Message(role="user", content=original))
        content = payload["content"]
        assert isinstance(content, str)

        expected_content = original
        if len(original) > int(contract["limit_chars"]):
            expected_content = unit * int(contract["head_chars"]) + str(contract["notice"]) + unit * int(contract["tail_chars"])
        assert content == expected_content
        actual_cases.append(
            {
                "repeat": case["repeat"],
                "truncated": content != original,
                "content_chars": len(content),
                "unit_chars": content.count(unit),
            }
        )

    assert actual_cases == contract["cases"]


def test_memory_local_summary_and_excerpt_match_fixture() -> None:
    summary_contract = _CONTRACT["summary"]
    manager = MemoryManager(
        summary_event_limit=int(summary_contract["event_limit"]),
        summary_callback=None,
    )

    summary = manager._build_local_summary(_messages_from_fixture(summary_contract["messages"]), [])
    assert _first_json_object(summary) == summary_contract["expected"]

    excerpt_contract = _CONTRACT["unicode_excerpt"]
    unit = str(excerpt_contract["content_unit"])
    suffix = str(excerpt_contract["suffix"])
    excerpt = manager._summarize_message_content(
        unit * int(excerpt_contract["repeat"]),
        limit=int(excerpt_contract["limit_chars"]),
    )
    actual_excerpt = {
        "content_unit": unit,
        "repeat": excerpt_contract["repeat"],
        "limit_chars": excerpt_contract["limit_chars"],
        "expected_unit_chars": excerpt.count(unit),
        "suffix": excerpt[-len(suffix) :],
    }

    assert excerpt == unit * int(excerpt_contract["expected_unit_chars"]) + suffix
    assert actual_excerpt == excerpt_contract


def test_session_memory_public_extract_matches_fixture() -> None:
    contract = _CONTRACT["session_extraction"]
    callback_calls = 0

    def extract_callback(_prompt: str, _backend: str | None, _model: str | None) -> str:
        nonlocal callback_calls
        callback_calls += 1
        return str(contract["raw"])

    memory = SessionMemory(SessionMemoryConfig(extraction_callback=extract_callback))
    merged = memory.extract(
        [Message(role="user", content="extract durable facts")],
        current_cycle=int(contract["cycle"]),
        current_tokens=10,
    )

    assert callback_calls == 1
    assert merged == len(contract["expected"])
    assert [entry.to_dict() for entry in memory.state.entries] == contract["expected"]


def test_session_memory_public_extract_handles_escaped_and_nested_json() -> None:
    content = 'keep ] and "quoted" plus \\ slash'
    raw_payload = [
        {
            "category": "decision",
            "content": content,
            "importance": 8,
            "metadata": {"nested": [1, {"value": "]"}]},
        }
    ]

    def extract_callback(_prompt: str, _backend: str | None, _model: str | None) -> str:
        return f"prefix {json.dumps(raw_payload)} suffix"

    memory = SessionMemory(SessionMemoryConfig(extraction_callback=extract_callback))
    merged = memory.extract(
        [Message(role="user", content="extract nested data")],
        current_cycle=4,
        current_tokens=10,
    )

    assert merged == 1
    assert [entry.to_dict() for entry in memory.state.entries] == [
        {
            "category": "decision",
            "content": content,
            "source_cycle": 4,
            "importance": 8,
        }
    ]


# Contract 23 producers compare complete transcripts and actual callback input.

_SUMMARY = _CONTRACT["summary_compaction"]
_SUMMARY_CASES = [case for case in _SUMMARY["cases"] if "variants" not in case]


@pytest.mark.parametrize("case", _SUMMARY_CASES, ids=lambda case: case["name"])
def test_history_preserving_summary_contract(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    manager, captured = case_manager(case, monkeypatch)
    result, changed = manager.compact(
        current_messages(case["input"]["messages"]), force=True, cycle_index=case["input"].get("cycle_index")
    )
    assert_case(case, result, changed, captured)


@pytest.mark.parametrize("variant", _SUMMARY["accepted_normalization_cases"]["variants"], ids=lambda case: case["name"])
def test_summary_normalization_contract(variant: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    case = _SUMMARY["accepted_normalization_cases"]
    case = {**case, "input": {**case["input"], "summary_response": variant["summary_response"]}, "expected": variant["expected"]}
    manager, captured = case_manager(case, monkeypatch)
    result, changed = manager.compact(current_messages(case["input"]["messages"]), force=True)
    assert_case(case, result, changed, captured)


_FAILURE = next(case for case in _SUMMARY["cases"] if "variants" in case)


@pytest.mark.parametrize(
    "variant", [v for v in _FAILURE["variants"] if "summary_input_fits" not in v], ids=lambda case: case["name"]
)
def test_failed_or_empty_summary_preserves_prefix(variant: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    case = {
        **_FAILURE,
        "input": {**_FAILURE["input"], **variant},
        "expected": {**_FAILURE["expected"], "summary_calls": variant["expected_summary_calls"]},
    }
    manager, captured = case_manager(case, monkeypatch)
    if variant.get("candidate_fits") is False:
        manager.model_context_window = 1
        manager.reserved_output_tokens = 0
    result, changed = manager.compact(current_messages(case["input"]["messages"]), force=True)
    assert_case(case, result, changed, captured)


@pytest.mark.parametrize("case", _SUMMARY["invalid_block_cases"], ids=lambda case: case["name"])
def test_invalid_atomic_blocks_preserve_history(case: dict[str, Any]) -> None:
    captured = []
    manager = MemoryManager(
        keep_recent_messages=1, summary_callback=lambda *args: captured.append(args) or '{"progress":["done"]}'
    )
    original = current_messages(case["messages"])
    result, changed = manager.compact(original, force=True)
    assert [m.to_dict() for m in result] == case["expected"]["messages"]
    assert changed is False
    assert captured == []


@pytest.mark.parametrize("language", ["zh-CN", "en-US"])
def test_localized_complete_prefix_prompt_bytes(language: str, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _SUMMARY_CASES[0]
    prompt_case = _SUMMARY["prompt_cases"][0]
    manager, captured = case_manager(case, monkeypatch, language=language, summary_event_limit=prompt_case["event_limit"])
    manager.compact(current_messages(case["input"]["messages"]), force=True)
    assert captured == [prompt_case["expected_prompts"][language]]


@pytest.mark.parametrize("case", _CONTRACT["microcompact"]["transcript_cases"], ids=lambda case: case["name"])
def test_relative_transcript_microcompact_contract(case: dict[str, Any]) -> None:
    inputs = case["input"]
    manager = MemoryManager(
        compact_threshold=1667,
        reserved_output_tokens=0,
        autocompact_buffer_tokens=0,
        keep_recent_messages=inputs.get("keep_recent_messages", 2),
        recovery_tool_available=True,
        microcompaction_policy=MicrocompactionPolicy.from_dict(inputs["policy"]),
    )
    original = current_messages(inputs["messages"])
    plan = manager.plan_microcompaction(original, cycle_index=inputs["cycle_index"], current_tokens=inputs["current_tokens"])
    assert plan is not None
    assert [candidate.message_index for candidate in plan.candidates] == case["expected"]["candidate_message_indices"]
    assert [m.to_dict() for m in original] == case["expected"].get("messages_before_application", inputs["messages"])


def test_standalone_summary_helpers_match_current_contract() -> None:
    manager = MemoryManager()
    parse = _CONTRACT["summary_parse"]
    assert manager._parse_summary_payload(parse["raw"]) == parse["expected"]
    originals = _CONTRACT["recompression_originals"]
    assert manager._collect_original_user_messages(current_messages(originals["messages"])) == originals["expected"]
