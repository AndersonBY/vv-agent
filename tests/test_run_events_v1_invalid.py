from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from vv_agent import CompletionReason, event_from_dict

FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "run_events_v1_invalid.json"
FIXTURE_SHA256 = "deec9e8c56cdb39e70b8c40e776021ce669dc6ea3477bd9b23f947dd5b5f1e99"


def _contract() -> dict[str, Any]:
    fixture_bytes = FIXTURE.read_bytes()
    assert hashlib.sha256(fixture_bytes).hexdigest() == FIXTURE_SHA256
    return json.loads(fixture_bytes)


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


def test_run_event_v1_compatibility_inputs_canonicalize_to_fixture() -> None:
    contract = _contract()

    for case in contract["canonicalize"]:
        event = event_from_dict(case["input"])
        assert event.to_dict() == case["output"], case["id"]


def test_run_event_v1_invalid_inputs_are_rejected() -> None:
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


def test_run_event_completion_validation_preserves_non_completion_compatibility() -> None:
    payload = _run_completed_payload()
    payload["future_field"] = {"ignored": True}
    payload["metadata"] = {"future_metadata": {"preserved": True}}

    encoded = event_from_dict(payload).to_dict()

    assert "future_field" not in encoded
    assert encoded["metadata"] == {"future_metadata": {"preserved": True}}
