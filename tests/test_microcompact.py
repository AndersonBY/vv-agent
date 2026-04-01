from __future__ import annotations

from vv_agent.memory.microcompact import CLEARED_MARKER, MicrocompactConfig, microcompact
from vv_agent.types import Message


def _build_messages() -> list[Message]:
    return [
        Message(role="system", content="sys"),
        Message(role="user", content="start"),
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
        Message(role="tool", content="x" * 800, tool_call_id="call_old"),
        Message(role="assistant", content="cycle two"),
        Message(role="user", content="continue"),
        Message(
            role="assistant",
            content="recent tool call",
            tool_calls=[
                {
                    "id": "call_recent",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="y" * 800, tool_call_id="call_recent"),
    ]


def test_microcompact_handles_empty_messages() -> None:
    compacted, cleared = microcompact([], current_cycle=5)

    assert compacted == []
    assert cleared == 0


def test_microcompact_clears_only_old_tool_results() -> None:
    messages, cleared = microcompact(
        _build_messages(),
        current_cycle=4,
        config=MicrocompactConfig(keep_recent_cycles=2, min_result_length=500),
    )

    assert cleared == 1
    tool_messages = [message for message in messages if message.role == "tool"]
    assert tool_messages[0].content == CLEARED_MARKER
    assert tool_messages[1].content != CLEARED_MARKER


def test_microcompact_does_not_clear_already_cleared_messages() -> None:
    messages = _build_messages()
    messages[3] = Message(role="tool", content=CLEARED_MARKER, tool_call_id="call_old")

    compacted, cleared = microcompact(
        messages,
        current_cycle=4,
        config=MicrocompactConfig(keep_recent_cycles=2, min_result_length=500),
    )

    assert cleared == 0
    assert compacted[3].content == CLEARED_MARKER


def test_microcompact_respects_compactable_tool_filter() -> None:
    messages = _build_messages()
    messages[2] = Message(
        role="assistant",
        content="old custom call",
        tool_calls=[
            {
                "id": "call_old",
                "type": "function",
                "function": {"name": "custom_tool", "arguments": "{}"},
            }
        ],
    )

    compacted, cleared = microcompact(
        messages,
        current_cycle=4,
        config=MicrocompactConfig(keep_recent_cycles=2, min_result_length=500),
    )

    assert cleared == 0
    assert compacted[3].content != CLEARED_MARKER


def test_microcompact_leaves_non_tool_messages_untouched() -> None:
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="u" * 1000),
        Message(role="assistant", content="a" * 1000),
    ]

    compacted, cleared = microcompact(
        messages,
        current_cycle=3,
        config=MicrocompactConfig(keep_recent_cycles=1, min_result_length=10),
    )

    assert cleared == 0
    assert compacted == messages


def test_microcompact_respects_min_result_length_boundary() -> None:
    def build_boundary_messages(length: int) -> list[Message]:
        return [
            Message(role="system", content="sys"),
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
            Message(role="tool", content="x" * length, tool_call_id="call_old"),
            Message(role="assistant", content="recent reply"),
        ]

    compacted_499, cleared_499 = microcompact(
        build_boundary_messages(499),
        current_cycle=3,
        config=MicrocompactConfig(keep_recent_cycles=1, min_result_length=500),
    )
    compacted_500, cleared_500 = microcompact(
        build_boundary_messages(500),
        current_cycle=3,
        config=MicrocompactConfig(keep_recent_cycles=1, min_result_length=500),
    )
    compacted_501, cleared_501 = microcompact(
        build_boundary_messages(501),
        current_cycle=3,
        config=MicrocompactConfig(keep_recent_cycles=1, min_result_length=500),
    )

    assert cleared_499 == 0
    assert compacted_499[2].content != CLEARED_MARKER
    assert cleared_500 == 0
    assert compacted_500[2].content != CLEARED_MARKER
    assert cleared_501 == 1
    assert compacted_501[2].content == CLEARED_MARKER


def test_microcompact_clamps_external_cycle_to_inferred_window() -> None:
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="compressed summary"),
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
        Message(role="tool", content="x" * 800, tool_call_id="call_old"),
        Message(role="assistant", content="middle reply"),
        Message(
            role="assistant",
            content="recent tool call",
            tool_calls=[
                {
                    "id": "call_recent",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="y" * 800, tool_call_id="call_recent"),
    ]

    compacted, cleared = microcompact(
        messages,
        current_cycle=15,
        config=MicrocompactConfig(keep_recent_cycles=1, min_result_length=500),
    )

    tool_messages = [message for message in compacted if message.role == "tool"]
    assert cleared == 1
    assert tool_messages[0].content == CLEARED_MARKER
    assert tool_messages[1].content != CLEARED_MARKER
