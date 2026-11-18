"""Focused W3 reducer bindings and attempt projection."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from vv_agent.kernel._records import Record, make_record
from vv_agent.kernel._reducer import Fold, TransitionError
from vv_agent.kernel._state import Attempt
from vv_agent.session.environments import execution_id_for

from .helpers import record as ordinary_record
from .test_w3_record_level import (
    _contract_source,
    _fixture,
    _op_planned,
    _op_prepared,
    _principal,
    _turn_started,
)

SESSION_ID = "sess-1"
TURN_ID = "turn-1"
OPERATION_ID = "op-bg-1"


def _w3() -> dict[str, Any]:
    return _fixture(_contract_source(), "execution_worker_w3.json")


def _session(w3: dict[str, Any]) -> Record:
    tenant_id = w3["identity_vectors"][0]["identity"]["tenant_id"]
    return ordinary_record("session_created", sid=SESSION_ID, principal=_principal(tenant_id))


def _started(plan: Record) -> Record:
    return ordinary_record(
        "op_started",
        sid=plan.session_id,
        tid=plan.turn_id or "",
        oid=plan.operation_id or "",
        attempt=plan.attempt or 1,
    )


def _fold(records: list[Record]):
    return Fold(
        normalize_memory_settings=lambda settings, context_window: settings,
        count_messages=lambda messages, *, model: 0,
    ).extend(records)


def _plan_variant(
    w3: dict[str, Any],
    *,
    tenant_id: str | None = None,
    environment_id: str | None = None,
    provider_binding: str | None | object = ...,
) -> Record:
    plan = _op_planned(w3)
    payload = deepcopy(plan._payload)
    identity = payload["execution_identity"]
    if tenant_id is not None:
        identity["tenant_id"] = tenant_id
    if environment_id is not None:
        identity["environment_id"] = environment_id
    if tenant_id is not None or environment_id is not None:
        identity["execution_id"] = execution_id_for(
            identity["tenant_id"],
            identity["environment_id"],
            identity["operation_key"],
        )
    if provider_binding is not ...:
        payload["provider_binding"] = provider_binding
    return make_record(
        "op_planned",
        session_id=plan.session_id,
        turn_id=plan.turn_id,
        operation_id=plan.operation_id,
        attempt=plan.attempt,
        payload=payload,
        feature_level=3,
    )


def test_level_3_plan_and_preparation_project_final_command_identity() -> None:
    w3 = _w3()
    plan = _op_planned(w3)
    prepared = _op_prepared(w3)
    state = _fold([_session(w3), _turn_started(w3), plan, prepared, _started(plan)])
    attempt = state.operations[OPERATION_ID].attempts[1]
    execution_plan = attempt.execution_plan
    identity = plan._payload["execution_identity"]

    assert identity["execution_id"] == execution_id_for(
        identity["tenant_id"],
        identity["environment_id"],
        identity["operation_key"],
    )
    assert attempt.started and attempt.prepared is prepared
    assert execution_plan.record_id == plan.record_id
    assert execution_plan._payload["execution_identity"] == identity
    assert execution_plan._payload["request"] == prepared._payload["request"]
    assert execution_plan._payload["request_digest"] == prepared._payload["request_digest"]
    assert execution_plan._payload["command_digest"] == w3["preparation_binding"]["prepared_command_digest"]
    assert execution_plan._payload["command_digest"] != plan._payload["command_digest"]


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ({"tenant_id": "tenant-b"}, "execution identity changed session tenant"),
        ({"environment_id": "env-other"}, "execution identity changed turn environment"),
        ({"provider_binding": None}, "level-3 plan requires a provider"),
    ],
    ids=["tenant", "environment", "provider"],
)
def test_level_3_plan_rejects_cross_record_binding_changes(
    variant: dict[str, Any],
    message: str,
) -> None:
    w3 = _w3()
    with pytest.raises(TransitionError, match=message):
        _fold([_session(w3), _turn_started(w3), _plan_variant(w3, **variant)])


def test_level_3_dispatch_requires_preparation() -> None:
    w3 = _w3()
    plan = _op_planned(w3)
    with pytest.raises(TransitionError, match="level-3 dispatch requires preparation"):
        _fold([_session(w3), _turn_started(w3), plan, _started(plan)])


def test_preparation_cannot_replace_the_planned_provider() -> None:
    w3 = _w3()
    plan = _op_planned(w3)
    prepared = _op_prepared(w3)
    payload = deepcopy(prepared._payload) | {"provider_binding": "provider-other"}
    replacement = make_record(
        "op_prepared",
        session_id=prepared.session_id,
        turn_id=prepared.turn_id,
        operation_id=prepared.operation_id,
        attempt=prepared.attempt,
        payload=payload,
        feature_level=3,
    )

    with pytest.raises(TransitionError, match="preparation changed provider"):
        _fold([_session(w3), _turn_started(w3), plan, replacement, _started(plan)])


def test_ordinary_preparation_cannot_downgrade_a_level_3_plan() -> None:
    w3 = _w3()
    plan = _op_planned(w3)
    prepared = _op_prepared(w3)
    payload = deepcopy(prepared._payload)
    payload.pop("command_digest")
    downgraded = make_record(
        "op_prepared",
        session_id=prepared.session_id,
        turn_id=prepared.turn_id,
        operation_id=prepared.operation_id,
        attempt=prepared.attempt,
        payload=payload,
    )

    with pytest.raises(TransitionError, match="level-3 preparation does not match its plan"):
        _fold([_session(w3), _turn_started(w3), plan, downgraded, _started(plan)])


def test_attempt_projection_reuses_first_preparation_without_authorizing_resend() -> None:
    w3 = _w3()
    plan = _op_planned(w3)
    prepared = _op_prepared(w3)
    payload = deepcopy(plan._payload) | {"command_digest": "a" * 64, "not_before_ms": 0}
    retry_plan = make_record(
        "op_planned",
        session_id=plan.session_id,
        turn_id=plan.turn_id,
        operation_id=plan.operation_id,
        attempt=2,
        payload=payload,
        feature_level=3,
    )
    execution_plan = Attempt(retry_plan, prepared=prepared).execution_plan

    assert execution_plan.attempt == 2
    assert execution_plan._payload["execution_identity"] == plan._payload["execution_identity"]
    assert execution_plan._payload["request"] == prepared._payload["request"]
    assert execution_plan._payload["request_digest"] == prepared._payload["request_digest"]
    assert execution_plan._payload["command_digest"] == prepared._payload["command_digest"]


def test_ordinary_tool_dispatch_remains_preparation_free() -> None:
    records = [
        ordinary_record("session_created", sid="s"),
        ordinary_record("turn_started", sid="s", tid="t"),
        ordinary_record("op_planned", sid="s", tid="t", oid="o", op_kind="tool", purpose=None),
        ordinary_record("op_started", sid="s", tid="t", oid="o"),
    ]
    attempt = _fold(records).operations["o"].attempts[1]

    assert attempt.prepared is None
    assert attempt.started
    assert "command_digest" not in attempt.execution_plan._payload
