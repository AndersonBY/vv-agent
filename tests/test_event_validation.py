from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vv_agent import CompletionReason, event_from_dict

FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "run_events_invalid.json"


def _contract() -> dict[str, Any]:
    return json.loads(FIXTURE.read_bytes())


def _run_completed_payload() -> dict[str, Any]:
    return {
        "version": "v1",
        "type": "run_completed",
        "event_id": "evt_completion_contract",
        "run_id": "run_completion_contract",
        "trace_id": "trace_completion_contract",
        "created_at": 1.0,
        "status": "completed",
        "final_output": "done",
    }


def test_invalid_run_event_inputs_are_rejected() -> None:
    contract = _contract()

    for case in contract["reject"]:
        with pytest.raises(ValueError, match=r".+"):
            event_from_dict(case["input"])


@pytest.mark.parametrize("reason", [reason.value for reason in CompletionReason])
def test_run_event_completion_reason_accepts_declared_values(reason: str) -> None:
    payload = _run_completed_payload()
    payload["completion_reason"] = reason

    event = event_from_dict(payload)

    assert event.to_dict()["completion_reason"] == reason


def test_run_event_completion_text_fields_accept_strings_and_null() -> None:
    payload = _run_completed_payload()
    payload.update(
        completion_reason=None,
        completion_tool_name="task_finish",
        partial_output="last draft",
    )

    event = event_from_dict(payload)

    encoded = event.to_dict()
    assert encoded.get("completion_reason") is None
    assert encoded["completion_tool_name"] == "task_finish"
    assert encoded["partial_output"] == "last draft"

    nullable_payload = _run_completed_payload()
    nullable_payload.update(
        completion_reason=None,
        completion_tool_name=None,
        partial_output=None,
    )
    nullable = event_from_dict(nullable_payload)
    nullable_encoded = nullable.to_dict()
    assert nullable_encoded.get("completion_reason") is None
    assert nullable_encoded.get("completion_tool_name") is None
    assert nullable_encoded.get("partial_output") is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("completion_reason", "future_reason"),
        ("completion_reason", 7),
        ("completion_tool_name", False),
        ("completion_tool_name", ["task_finish"]),
        ("partial_output", {"text": "last draft"}),
        ("partial_output", 7),
    ],
)
def test_run_event_completion_fields_reject_unknown_reason_and_wrong_types(
    field_name: str,
    value: Any,
) -> None:
    payload = _run_completed_payload()
    payload[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        event_from_dict(payload)


def test_run_event_rejects_unknown_fields_but_preserves_typed_metadata_extension() -> None:
    payload = _run_completed_payload()
    payload["future_field"] = {"ignored": True}
    payload["metadata"] = {"future_metadata": {"preserved": True}}

    with pytest.raises(ValueError, match="unknown fields: future_field"):
        event_from_dict(payload)

    payload.pop("future_field")
    encoded = event_from_dict(payload).to_dict()
    assert encoded["metadata"] == {"future_metadata": {"preserved": True}}


@pytest.mark.parametrize("duration_ms", [True, -1, 1.5, 9_007_199_254_740_992])
def test_tool_completion_duration_rejects_non_json_safe_values(duration_ms: Any) -> None:
    payload = {
        "version": "v1",
        "type": "tool_call_completed",
        "event_id": "evt_invalid_duration",
        "run_id": "run_invalid_duration",
        "trace_id": "trace_invalid_duration",
        "created_at": 1.0,
        "tool_name": "lookup",
        "tool_call_id": "call_invalid_duration",
        "status": "success",
        "directive": "continue",
        "error_code": None,
        "execution_started": True,
        "duration_ms": duration_ms,
    }

    with pytest.raises(ValueError, match="duration_ms"):
        event_from_dict(payload)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("trigger", 1),
        ("configured_threshold", None),
        ("configured_threshold", True),
        ("effective_threshold", -1),
        ("microcompact_threshold", 1.5),
        ("model_context_window", 9_007_199_254_740_992),
        ("model_max_output_tokens", "8192"),
        ("reserved_output_tokens", []),
        ("reserved_output_source", None),
        ("autocompact_buffer_tokens", {}),
    ],
)
def test_memory_compact_started_rejects_known_fields_with_wrong_types(
    field_name: str,
    value: Any,
) -> None:
    payload = {
        "version": "v1",
        "type": "memory_compact_started",
        "event_id": "evt_invalid_memory_started",
        "run_id": "run_invalid_memory",
        "trace_id": "trace_invalid_memory",
        "created_at": 1.0,
        "message_count": 3,
        "trigger": "full_threshold",
        "configured_threshold": 250_000,
        "effective_threshold": 250_000,
        "microcompact_threshold": 187_500,
        "model_context_window": 1_000_000,
        "model_max_output_tokens": None,
        "reserved_output_tokens": 16_000,
        "reserved_output_source": "framework_fallback",
        "autocompact_buffer_tokens": 13_000,
    }
    payload[field_name] = value

    with pytest.raises(ValueError, match=r".+"):
        event_from_dict(payload)


@pytest.mark.parametrize(("field_name", "value"), [("mode", 1), ("changed", 1)])
def test_memory_compact_completed_rejects_known_fields_with_wrong_types(
    field_name: str,
    value: Any,
) -> None:
    payload = {
        "version": "v1",
        "type": "memory_compact_completed",
        "event_id": "evt_invalid_memory_completed",
        "run_id": "run_invalid_memory",
        "trace_id": "trace_invalid_memory",
        "created_at": 1.0,
        "before_count": 3,
        "after_count": 2,
        "mode": "summary",
        "changed": True,
    }
    payload[field_name] = value

    with pytest.raises(ValueError, match=r".+"):
        event_from_dict(payload)
