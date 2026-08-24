from __future__ import annotations

import json
from pathlib import Path

import pytest

from vv_agent.deferred import DeferredToolHandle
from vv_agent.events import (
    CheckpointCreatedEvent,
    CheckpointResumedEvent,
    ModelRetryDuplicateRiskEvent,
    OperationAmbiguousEvent,
    OperationReplayedEvent,
    ReconciliationRequiredEvent,
    ReconciliationResolvedEvent,
    ToolCallCompletedEvent,
    ToolCallDeferredEvent,
    event_from_dict,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity" / "resume_events.jsonl"


def _fixture_events() -> list[dict[str, object]]:
    return [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()]


def test_resume_event_fixture_round_trips_through_typed_producers() -> None:
    expected_types = (
        CheckpointCreatedEvent,
        CheckpointResumedEvent,
        OperationReplayedEvent,
        OperationAmbiguousEvent,
        ReconciliationRequiredEvent,
        OperationAmbiguousEvent,
        ModelRetryDuplicateRiskEvent,
        OperationAmbiguousEvent,
        ReconciliationRequiredEvent,
        ReconciliationResolvedEvent,
        ToolCallDeferredEvent,
        ToolCallCompletedEvent,
        ToolCallDeferredEvent,
        ToolCallCompletedEvent,
        ReconciliationResolvedEvent,
        ToolCallDeferredEvent,
    )

    for payload, expected_type in zip(_fixture_events(), expected_types, strict=True):
        event = event_from_dict(payload)
        assert isinstance(event, expected_type)
        assert event.to_dict() == payload


def test_resume_event_rejects_invalid_operation_boundaries() -> None:
    payload = _fixture_events()[3]
    with pytest.raises(ValueError, match="idempotency_support"):
        event_from_dict({**payload, "idempotency_support": None})

    replay = _fixture_events()[2]
    with pytest.raises(ValueError, match="receipt_state"):
        event_from_dict({**replay, "receipt_state": "started"})

    model_risk = _fixture_events()[6]
    with pytest.raises(ValueError, match="model operation_kind"):
        event_from_dict({**model_risk, "operation_kind": "tool"})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("operation_id", "op_tool_cycle_2_call_tampered", "deferred handle is invalid"),
        ("attempt", 2, "deferred handle is invalid"),
    ],
)
def test_deferred_event_rejects_top_level_identity_tampering(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = next(item for item in _fixture_events() if item["type"] == "tool_call_deferred")
    tampered = {**payload, field: value}

    with pytest.raises(ValueError, match=message):
        event_from_dict(tampered)


def test_deferred_event_constructor_rejects_model_operation_kind() -> None:
    payload = next(item for item in _fixture_events() if item["type"] == "tool_call_deferred")
    handle = DeferredToolHandle.from_dict(payload["handle"])

    with pytest.raises(ValueError, match="deferred operation_kind must be tool"):
        ToolCallDeferredEvent(
            run_id=payload["run_id"],
            trace_id=payload["trace_id"],
            cycle_index=payload["cycle_index"],
            tool_call_id=payload["tool_call_id"],
            tool_name=payload["tool_name"],
            operation_id=payload["operation_id"],
            attempt=payload["attempt"],
            handle=handle,
            execution_started=payload["execution_started"],
            duration_ms=payload["duration_ms"],
            checkpoint_key=payload["checkpoint_key"],
            operation_kind="model",
            event_id=payload["event_id"],
            created_at=payload["created_at"],
        )
