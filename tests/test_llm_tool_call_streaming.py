from types import SimpleNamespace
from typing import Any

import pytest

from vv_agent.llm.vv_llm_client import VvLlmClient


def _delta(
    arguments: str,
    *,
    index: int | None = None,
    call_id: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    return {"index": index, "id": call_id, "function": {"name": name, "arguments": arguments}}


@pytest.mark.parametrize(
    ("chunks", "incremental", "expected"),
    [
        pytest.param(
            [
                [_delta('{"x":', index=0, call_id="ca", name="a"), _delta('{"y":', index=1, call_id="cb", name="b")],
                [_delta("1}", index=0), _delta("2}", index=1)],
            ],
            True,
            {"a": '{"x":1}', "b": '{"y":2}'},
            id="issue67-two-deltas-per-chunk",
        ),
        pytest.param(
            [
                [_delta('{"x":', index=0, call_id="ca", name="a")],
                [_delta('{"y":', index=1, call_id="cb", name="b")],
                [_delta("1}", index=0)],
                [_delta("2}", index=1)],
            ],
            True,
            {"a": '{"x":1}', "b": '{"y":2}'},
            id="real-index-interleaving",
        ),
        pytest.param(
            [
                [_delta('{"x":', index=0, call_id="ca", name="a")],
                [_delta("1}", index=0)],
                [_delta('{"y":', index=0, call_id="cb", name="b")],
                [_delta("2}", index=0)],
            ],
            True,
            {"a": '{"x":1}', "b": '{"y":2}'},
            id="vv-llm-rewritten-sequential-index-zero",
        ),
        pytest.param(
            [
                [_delta('{"x":', call_id="ca", name="a"), _delta('{"y":', call_id="cb", name="b")],
                [_delta("1}", call_id="ca")],
                [_delta("2}", call_id="cb")],
            ],
            True,
            {"a": '{"x":1}', "b": '{"y":2}'},
            id="id-only-continuations",
        ),
        pytest.param(
            [
                [_delta('{"x":', index=0, call_id="ca", name="a"), _delta('{"y":', index=1, call_id="cb", name="b")],
                [_delta("1}", index=1, call_id="ca")],
                [_delta("2}", index=0, call_id="cb")],
            ],
            True,
            {"a": '{"x":1}', "b": '{"y":2}'},
            id="id-wins-over-conflicting-index",
        ),
        pytest.param(
            [
                [_delta('{"x":', index=0, call_id="ca", name="a")],
                [_delta("1", index=1, call_id="ca", name="a")],
                [_delta("}", index=1)],
            ],
            True,
            {"a": '{"x":1}'},
            id="repeated-name-id-with-changed-index",
        ),
        pytest.param(
            [[_delta('{"x":', name="a")], [_delta("1}")]],
            True,
            {"a": '{"x":1}'},
            id="single-call-without-index-or-id",
        ),
        pytest.param(
            [[_delta('{"x":', call_id="ca", name="a")], [_delta("1}")]],
            True,
            {"a": '{"x":1}'},
            id="single-call-without-index",
        ),
        pytest.param(
            [
                [_delta('{"x":', index=0, call_id="ca", name="a"), _delta('{"y":', index=1, call_id="cb", name="b")],
                [_delta("1}", index=0, call_id="ca", name="a"), _delta("2}", index=1, call_id="cb", name="b")],
            ],
            True,
            {"a": '{"x":1}', "b": '{"y":2}'},
            id="repeated-first-delta-fields",
        ),
        pytest.param(
            [
                [_delta('{"x":', index=0, call_id="ca", name="a"), _delta('{"y":', index=1, call_id="cb", name="b")],
                [_delta('{"x":1}', index=0), _delta('{"y":2}', index=1)],
            ],
            False,
            {"a": '{"x":1}', "b": '{"y":2}'},
            id="snapshot-replace",
        ),
        pytest.param(
            [
                [_delta('{"x":', index=0, call_id="ca", name="a"), _delta('{"y":', index=1, call_id="cb", name="b")],
                [_delta("unowned")],
            ],
            True,
            {"a": '{"x":', "b": '{"y":'},
            id="ambiguous-no-index-or-id",
        ),
        pytest.param(
            [[_delta('{"x":', index=0, call_id="ca", name="a")], [_delta("unowned", index=99)]],
            True,
            {"a": '{"x":'},
            id="unknown-index-does-not-use-last-active",
        ),
        pytest.param(
            [[_delta('{"x":', index=0, call_id="ca", name="a")], [_delta("unowned", index=0, call_id="unknown")]],
            True,
            {"a": '{"x":'},
            id="unknown-id-does-not-use-index",
        ),
        pytest.param(
            [[_delta('{"x":', index=0, name="a")], [_delta("1}", index=0, call_id="ca")]],
            True,
            {"a": '{"x":1}'},
            id="late-id-for-generated-slot",
        ),
    ],
)
def test_tool_call_delta_ownership(chunks: list[list[dict[str, Any]]], incremental: bool, expected: dict[str, str]) -> None:
    client = VvLlmClient(endpoint_targets=[], backend="openai", selected_model="test-model")
    parts: dict[str, dict[str, Any]] = {}
    active = None
    for chunk in chunks:
        for index, delta in enumerate(chunk):
            active = client._accumulate_tool_call_delta(
                tool_call_parts=parts,
                tool_delta=delta,
                default_index=index,
                last_active_tool_call_id=active,
                incremental=incremental,
                keep_extra_content=False,
            )

    assert len(parts) == len(expected)
    assert {slot["name"]: slot["arguments"] for slot in parts.values()} == expected
    for chunk in chunks:
        for delta in chunk:
            if delta["id"] and delta["id"] != "unknown":
                assert sum(slot["id"] == delta["id"] for slot in parts.values()) == 1


@pytest.mark.parametrize("with_ids", [False, True], ids=["no-ids", "ids"])
@pytest.mark.parametrize("streamed", [False, True], ids=["complete", "streamed"])
@pytest.mark.parametrize("incremental", [True, False], ids=["append", "snapshot"])
def test_same_name_calls_keep_separate_slots(with_ids: bool, streamed: bool, incremental: bool) -> None:
    client = VvLlmClient(endpoint_targets=[], backend="openai", selected_model="test-model")
    parts: dict[str, dict[str, Any]] = {}
    active = None
    for position, path in enumerate(["a", "b"]):
        complete = f'{{"p":"{path}"}}'
        deltas = [
            _delta(
                '{"p":' if streamed else complete,
                index=0,
                call_id=f"c{position}" if with_ids else None,
                name="read_file",
            )
        ]
        if streamed:
            deltas.append(_delta(f'"{path}"}}' if incremental else complete, index=0))
        for delta in deltas:
            active = client._accumulate_tool_call_delta(
                tool_call_parts=parts,
                tool_delta=delta,
                default_index=0,
                last_active_tool_call_id=active,
                incremental=incremental,
                keep_extra_content=False,
            )

    assert [(slot["name"], slot["arguments"]) for slot in parts.values()] == [
        ("read_file", '{"p":"a"}'),
        ("read_file", '{"p":"b"}'),
    ]
    assert len({slot["id"] for slot in parts.values()}) == 2
    if with_ids:
        assert [slot["id"] for slot in parts.values()] == ["c0", "c1"]


def test_repeated_name_without_id_keeps_new_slot_behavior() -> None:
    client = VvLlmClient(endpoint_targets=[], backend="openai", selected_model="test-model")
    parts: dict[str, dict[str, Any]] = {}
    active = None
    for delta in [_delta('{"x":', index=0, name="a"), _delta("1}", index=0, name="a")]:
        active = client._accumulate_tool_call_delta(
            tool_call_parts=parts,
            tool_delta=delta,
            default_index=0,
            last_active_tool_call_id=active,
            incremental=True,
            keep_extra_content=False,
        )

    # Without IDs, a repeated name cannot distinguish a continuation from a new call.
    assert [slot["arguments"] for slot in parts.values()] == ['{"x":', "1}"]


@pytest.mark.parametrize("incremental", [True, False], ids=["append", "snapshot"])
def test_stream_completion_attributes_arguments_and_progress_to_the_same_call(incremental: bool) -> None:
    chunks = [
        {
            "tool_calls": [
                _delta('{"x":', index=0, call_id="ca", name="a"),
                _delta('{"y":', index=1, call_id="cb", name="b"),
            ]
        },
        {"tool_calls": [_delta("unowned")]},
        {
            "tool_calls": [
                _delta("1}" if incremental else '{"x":1}', index=0),
                _delta("2}" if incremental else '{"y":2}', index=1),
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]

    def create_completion(**kwargs: Any):
        assert kwargs["stream"] is True
        return iter(chunks)

    client = VvLlmClient(endpoint_targets=[], backend="openai", selected_model="test-model")
    options = client._resolve_request_options("test-model", stream=True, endpoint_type="openai" if incremental else "snapshot")
    events: list[dict[str, Any]] = []
    response = client._stream_completion(
        chat_client=SimpleNamespace(create_completion=create_completion),
        options=options,
        messages=[],
        model_name="test-model",
        tool_payload=[],
        stream_callback=events.append,
    )

    assert {call.id: call.arguments for call in response.tool_calls} == {"ca": {"x": 1}, "cb": {"y": 2}}
    assert [
        (
            event["event"],
            event["tool_call_id"],
            event["tool_call_index"],
            event["function_name"],
            event["arguments_chars"],
        )
        for event in events
    ] == [
        ("tool_call_started", "ca", 0, "a", 5),
        ("tool_call_progress", "ca", 0, "a", 5),
        ("tool_call_started", "cb", 1, "b", 5),
        ("tool_call_progress", "cb", 1, "b", 5),
        ("tool_call_progress", "ca", 0, "a", 7),
        ("tool_call_progress", "cb", 1, "b", 7),
    ]
