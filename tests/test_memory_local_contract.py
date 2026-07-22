from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from vv_agent.memory import MemoryManager, MicrocompactConfig, SessionMemory, SessionMemoryConfig, microcompact
from vv_agent.memory.token_utils import count_messages_tokens, count_tokens
from vv_agent.types import Message

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
        tool_content = str(contract["content_unit"]) * int(case["repeat"])
        messages = [
            Message(role="system", content="system"),
            Message(
                role="assistant",
                content="old tool call",
                tool_calls=[
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            ),
            Message(role="tool", content=tool_content, tool_call_id="call_old"),
            Message(role="assistant", content="recent reply"),
        ]
        compacted, cleared = microcompact(
            messages,
            current_cycle=3,
            config=MicrocompactConfig(
                keep_recent_cycles=int(contract["keep_recent_cycles"]),
                min_result_length=int(contract["minimum_chars"]),
            ),
        )
        actual_case: dict[str, Any] = {
            "repeat": case["repeat"],
            "cleared": cleared > 0,
        }
        if cleared:
            actual_case["original_chars"] = compacted[2].metadata["microcompact_original_chars"]
        actual_cases.append(actual_case)

    assert actual_cases == contract["cases"]


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

    compacted, changed = manager.compact(
        _messages_from_fixture(summary_contract["messages"]),
        force=True,
    )

    assert changed is True
    assert _first_json_object(compacted[1].content) == summary_contract["expected"]

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


def test_memory_local_recompression_originals_are_stable() -> None:
    contract = _CONTRACT["recompression_originals"]
    manager = MemoryManager(summary_callback=None)

    compacted, changed = manager.compact(
        _messages_from_fixture(contract["messages"]),
        force=True,
    )
    first_originals = _first_json_object(compacted[1].content)["original_user_messages"]

    recompressed, recompressed_changed = manager.compact(
        [*compacted, Message(role="assistant", content="continued work")],
        force=True,
    )
    second_originals = _first_json_object(recompressed[1].content)["original_user_messages"]

    assert changed is True
    assert recompressed_changed is True
    assert first_originals == contract["expected"]
    assert second_originals == contract["expected"]


def test_memory_summary_parse_and_prefixed_callback_restore_match_contract(tmp_path: Path) -> None:
    parse_contract = _CONTRACT["summary_parse"]
    manager = MemoryManager()

    assert manager._parse_summary_payload(str(parse_contract["raw"])) == parse_contract["expected"]

    restored_file = tmp_path / "restore.py"
    restored_file.write_text("RESTORED_FROM_PREFIX = True\n", encoding="utf-8")

    def summary_callback(_prompt: str, _backend: str | None, _model: str | None) -> str:
        return (
            "prefix text\n"
            '{"summary_version":"2.0","files_examined_or_modified":'
            '[{"path":"restore.py","action":"modified","summary":"updated"}]} trailing'
        )

    restore_manager = MemoryManager(workspace=tmp_path, summary_callback=summary_callback)
    compacted, changed = restore_manager.compact(
        [
            Message(role="system", content="system"),
            Message(role="user", content="update restore.py"),
            Message(role="assistant", content="updated"),
        ],
        force=True,
    )

    assert changed is True
    assert "prefix text" in compacted[1].content
    assert "<Post-Compaction File Context>" in compacted[1].content
    assert "RESTORED_FROM_PREFIX = True" in compacted[1].content
