from __future__ import annotations

import json
from pathlib import Path

import pytest

from vv_agent.events import (
    CheckpointCreatedEvent,
    CheckpointResumedEvent,
    ModelRetryDuplicateRiskEvent,
    OperationAmbiguousEvent,
    OperationReplayedEvent,
    ReconciliationRequiredEvent,
    ReconciliationResolvedEvent,
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
