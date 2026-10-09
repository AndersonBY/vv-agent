from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from vv_agent import event_from_dict

FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "run_events_invalid.json"


def _contract() -> dict[str, Any]:
    return json.loads(FIXTURE.read_bytes())


def _run_completed_payload() -> dict[str, Any]:
    return {
        "version": "v6",
        "type": "run_completed",
        "event_id": "evt_completion_contract",
        "session_id": "event-session",
        "run_id": "run_completion_contract",
        "trace_id": "trace_completion_contract",
        "created_at": 1.0,
        "status": "completed",
        "final_output": "done",
    }


def test_invalid_run_event_inputs_are_rejected() -> None:
    for case in _contract()["reject"]:
        payload = json.loads(base64.b64decode(case["bytes_base64"]))
        with pytest.raises(ValueError, match=r".+"):
            event_from_dict(payload)


def test_run_event_rejects_retired_completion_reason():
    with pytest.raises(ValueError, match="completion_reason"):
        event_from_dict(_run_completed_payload() | {"completion_reason": "retired"})


@pytest.mark.parametrize("field_name", ["completion_tool_name", "partial_output"])
def test_run_event_optional_completion_fields_round_trip(field_name):
    payload = _run_completed_payload() | {field_name: "value"}
    assert event_from_dict(payload).to_dict()[field_name] == "value"


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
        "version": "v6",
        "type": "tool_call_completed",
        "event_id": "evt_invalid_duration",
        "session_id": "event-session",
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
        "version": "v6",
        "type": "memory_compact_started",
        "event_id": "evt_invalid_memory_started",
        "session_id": "event-session",
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
        "version": "v6",
        "type": "memory_compact_completed",
        "event_id": "evt_invalid_memory_completed",
        "session_id": "event-session",
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
