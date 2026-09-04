from __future__ import annotations

import pytest

from vv_agent.runtime.model_calls import ModelCallCoordinator, ModelCallLedger
from vv_agent.types import ModelCallOperation, ModelCallRecord, ModelCallStatus, TokenUsage


def _record(operation_id: str, status: ModelCallStatus) -> ModelCallRecord:
    return ModelCallRecord(
        call_id=f"{operation_id}:attempt:1",
        operation_id=operation_id,
        attempt=1,
        operation=ModelCallOperation.AGENT_CYCLE,
        cycle_index=1,
        backend="test",
        model="test-model",
        status=status,
        usage=TokenUsage(),
        error_code=None if status is ModelCallStatus.COMPLETED else "model_request_failed",
    )


def _coordinator(records: list[ModelCallRecord]) -> ModelCallCoordinator:
    ledger = ModelCallLedger()
    ledger.replace(records)
    return ModelCallCoordinator(
        ledger=ledger,
        run_id="run",
        trace_id="trace",
        agent_name="agent",
        session_id=None,
        parent_run_id=None,
        event_sink=None,
    )


@pytest.mark.parametrize(
    ("records", "expected_operation_id"),
    [
        ([_record("op_model_cycle_1_main", ModelCallStatus.COMPLETED)], "op_model_cycle_1_main_2"),
        ([_record("op_model_cycle_1_main", ModelCallStatus.AMBIGUOUS)], "op_model_cycle_1_main"),
        (
            [
                _record("op_model_cycle_1_main", ModelCallStatus.COMPLETED),
                _record("op_model_cycle_1_main_2", ModelCallStatus.AMBIGUOUS),
            ],
            "op_model_cycle_1_main_2",
        ),
        (
            [
                _record("op_model_cycle_1_main", ModelCallStatus.COMPLETED),
                _record("op_model_cycle_1_main_2", ModelCallStatus.COMPLETED),
            ],
            "op_model_cycle_1_main_3",
        ),
    ],
    ids=[
        "completed-advances",
        "ambiguous-replays",
        "completed-before-ambiguous",
        "completed-slots-avoid-collision",
    ],
)
def test_generated_model_slot_count_is_seeded_for_resume(
    records: list[ModelCallRecord],
    expected_operation_id: str,
) -> None:
    coordinator = _coordinator(records)

    identity = coordinator.new_identity(
        cycle_index=1,
        operation_slot="main",
        operation=ModelCallOperation.AGENT_CYCLE,
        backend="test",
        model="test-model",
    )

    assert identity.operation_id == expected_operation_id


def test_generated_slot_seeding_does_not_parse_another_slot_as_a_suffix() -> None:
    coordinator = _coordinator([_record("op_model_cycle_1_main_2", ModelCallStatus.COMPLETED)])

    identity = coordinator.new_identity(
        cycle_index=1,
        operation_slot="main",
        operation=ModelCallOperation.AGENT_CYCLE,
        backend="test",
        model="test-model",
    )

    assert identity.operation_id == "op_model_cycle_1_main"


def test_peek_identity_preserves_completed_operation_id_for_durable_replay() -> None:
    coordinator = _coordinator([_record("op_model_cycle_1_main", ModelCallStatus.COMPLETED)])

    identity = coordinator.peek_identity(
        cycle_index=1,
        operation_slot="main",
        operation=ModelCallOperation.AGENT_CYCLE,
        backend="test",
        model="test-model",
    )

    assert identity.operation_id == "op_model_cycle_1_main"
