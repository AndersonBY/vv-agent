from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from support.s1_fixture_values import content_ref

from vv_agent.kernel import _driver_outcomes as outcomes
from vv_agent.kernel._hydration import IntegrityFailure, KeyUnavailable, MissingContentLink
from vv_agent.kernel._provider_outcomes import Definitive
from vv_agent.kernel._records import Record
from vv_agent.kernel._state import Attempt, ExecutionState, Operation, Turn


def _plan(
    operation_id: str,
    *,
    attempt: int = 1,
    purpose: str | None = "primary",
    kind: str = "model",
    retained: bool = False,
) -> Record:
    request = content_ref(f"{operation_id}/request") if retained else {"messages": [{"role": "user", "content": "history"}]}
    return Record(
        record_id=f"record/{operation_id}/{attempt}",
        kind="op_planned",
        session_id="s",
        turn_id="turn",
        operation_id=operation_id,
        attempt=attempt,
        payload={
            "op_kind": kind,
            "purpose": purpose,
            "request": request,
            "request_digest": "a" * 64,
            "provider_binding": "model" if kind == "model" else "function",
            "dependencies": [],
        },
    )


class _Driver:
    def __init__(self) -> None:
        self.state = ExecutionState(boundaries={})
        self.hydrate_calls: list[str] = []
        self.request_content_reads = 0
        self.definition_calls = 0
        self.hydration_error: Exception | None = None

    def add_operation(
        self,
        plan: Record,
        *,
        state: str = "completed",
        superseded: bool = False,
        wait: dict[str, Any] | None = None,
        kind: str | None = None,
    ) -> Operation:
        assert plan.operation_id is not None and plan.attempt is not None and plan.turn_id is not None
        attempts = {
            number: Attempt(
                plan if number == plan.attempt else _plan(plan.operation_id, attempt=number),
                state=state if number == plan.attempt else "completed",
                dispatch=SimpleNamespace(),  # type: ignore[arg-type]
                superseded=superseded if number == plan.attempt else True,
                wait=wait if number == plan.attempt else None,
            )
            for number in range(1, plan.attempt + 1)
        }
        operation = Operation(
            plan.turn_id,
            kind or plan._payload["op_kind"],
            attempts,
            selected_attempt=plan.attempt if state == "completed" else None,
        )
        self.state.operations[plan.operation_id] = operation
        self.state.turns[plan.turn_id] = Turn(start=SimpleNamespace())  # type: ignore[arg-type]
        return operation

    def add_tool(self, plan: Record, index: int = 0) -> None:
        assert plan.operation_id is not None and plan.turn_id is not None
        operation_id = f"{plan.operation_id}/attempt/{plan.attempt}/tool/{index}"
        tool_plan = _plan(operation_id, purpose=None, kind="tool")
        self.state.operations[operation_id] = Operation(
            plan.turn_id,
            "tool",
            {1: Attempt(tool_plan, state="completed", dispatch=SimpleNamespace())},  # type: ignore[arg-type]
            selected_attempt=1,
        )

    def hydrate_plan(self, record: Record) -> Record:
        self.hydrate_calls.append(record.record_id)
        self.request_content_reads += 1
        if self.hydration_error is not None:
            raise self.hydration_error
        request = record._payload["request"]
        if "content_id" in request:
            return replace(record, payload=record._payload | {"request": {"messages": [{"role": "user", "content": "history"}]}})
        return replace(record, payload=record._payload)

    def hydrated_definition(self, _record: Record) -> SimpleNamespace:
        self.definition_calls += 1
        return SimpleNamespace(capabilities={"effect": {"idempotency": "supported"}})

    def record(self, kind: str, _payload: dict[str, Any], plan: Record) -> Record:
        return Record(
            record_id=f"record/{kind}/{plan.operation_id}/{plan.attempt}",
            kind=kind,
            session_id="s",
            turn_id=plan.turn_id,
            operation_id=plan.operation_id,
            attempt=plan.attempt,
            payload={},
        )


def _no_budget(_driver: _Driver):
    if False:
        yield None
    return None


def _outcome(*calls: dict[str, Any]) -> Definitive:
    return Definitive({"content": "done", "tool_calls": list(calls), "raw": {}}, ())


def _finish(steps):
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def _effect_plan_steps(planned: list[tuple[str, dict[str, Any], str, list[str], dict[str, Any]]]):
    def plan_steps(
        _driver: _Driver,
        operation_id: str,
        request: dict[str, Any],
        kind: str,
        *,
        dependencies: list[str],
        capability: dict[str, Any],
    ):
        planned.append((operation_id, request, kind, dependencies, capability))
        if False:
            yield None
        return Record(
            record_id=operation_id,
            kind="op_planned",
            session_id="s",
            turn_id="turn",
            operation_id=operation_id,
            attempt=1,
            payload={},
        )

    return plan_steps


def test_non_normal_context_returns_before_any_content_read(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _Driver()
    driver.hydration_error = KeyUnavailable("must not read audit content")
    monkeypatch.setattr(outcomes, "live_budget_steps", _no_budget)
    plan = _plan("turn/model/1", retained=True)
    driver.add_operation(plan)

    call = {"id": "call", "name": "effect", "arguments": {}}
    for context in ("audit", "correction"):
        assert list(outcomes.completion_effects_steps(driver, plan, _outcome(call), context)) == []

    assert driver.hydrate_calls == []
    assert driver.request_content_reads == 0
    assert driver.definition_calls == 0


@pytest.mark.parametrize(
    ("name", "purpose", "kind", "attempt", "state", "wait"),
    [
        ("compaction", "compaction", "model", 1, "completed", None),
        ("session-memory", "session_memory", "model", 1, "completed", None),
        ("output-repair", "output_repair", "model", 1, "completed", None),
        ("interaction", None, "interaction", 1, "completed", None),
        ("retry", "primary", "model", 2, "completed", None),
        ("provider-wait", "primary", "model", 1, "completed", {"handle": {"kind": "provider"}}),
        ("active", "primary", "model", 1, "started", None),
    ],
)
def test_normal_paths_without_new_effects_do_not_hydrate(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    purpose: str | None,
    kind: str,
    attempt: int,
    state: str,
    wait: dict[str, Any] | None,
) -> None:
    driver = _Driver()
    driver.hydration_error = IntegrityFailure(f"{name} must not read content")
    monkeypatch.setattr(outcomes, "live_budget_steps", _no_budget)
    plan = _plan(f"turn/{name}", attempt=attempt, purpose=purpose, kind=kind, retained=True)
    driver.add_operation(plan, state=state, wait=wait, kind=kind)

    assert list(outcomes.completion_effects_steps(driver, plan, _outcome(), "normal")) == []
    assert driver.hydrate_calls == []
    assert driver.request_content_reads == 0
    assert driver.definition_calls == 0


def test_completed_steps_reuses_hydrated_plan_and_only_plans_missing_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _Driver()
    monkeypatch.setattr(outcomes, "live_budget_steps", _no_budget)
    plan = _plan("turn/model/1", retained=True)
    driver.add_operation(plan)
    driver.add_tool(plan, index=0)
    driver.add_tool(plan, index=2)
    calls = [
        {"id": "existing-0", "name": "effect", "arguments": {"value": 0}},
        {"id": "missing", "name": "effect", "arguments": {"value": 1}},
        {"id": "existing-2", "name": "effect", "arguments": {"value": 2}},
    ]
    planned: list[tuple[str, dict[str, Any], str, list[str], dict[str, Any]]] = []
    monkeypatch.setattr(outcomes, "plan_steps", _effect_plan_steps(planned))

    records = _finish(outcomes.completed_steps(driver, plan, _outcome(*calls)))

    assert driver.hydrate_calls == [plan.record_id]
    assert driver.request_content_reads == 1
    assert driver.definition_calls == 1
    assert planned == [
        (
            "turn/model/1/attempt/1/tool/1",
            calls[1],
            "tool",
            ["turn/model/1"],
            {"idempotency": "supported"},
        )
    ]
    assert [record.record_id for record in records] == [
        "record/op_completed/turn/model/1/1",
        "turn/model/1/attempt/1/tool/1",
    ]


def test_completed_steps_audit_uses_only_the_caller_hydration(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _Driver()
    monkeypatch.setattr(outcomes, "live_budget_steps", _no_budget)
    plan = _plan("turn/model/1", retained=True)
    driver.add_operation(plan)
    driver.state.turns["turn"].ended = True

    records = _finish(
        outcomes.completed_steps(
            driver,
            plan,
            _outcome({"id": "must-not-plan", "name": "effect", "arguments": {}}),
        )
    )

    assert driver.hydrate_calls == [plan.record_id]
    assert driver.request_content_reads == 1
    assert driver.definition_calls == 0
    assert [record.record_id for record in records] == ["record/op_completed/turn/model/1/1"]


def test_raw_plan_is_hydrated_once_only_when_constructing_the_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _Driver()
    monkeypatch.setattr(outcomes, "live_budget_steps", _no_budget)
    plan = _plan("turn/model/1", retained=True)
    driver.add_operation(plan)
    call = {"id": "call", "name": "effect", "arguments": {"value": 1}}
    planned: list[tuple[str, dict[str, Any], str, list[str], dict[str, Any]]] = []
    monkeypatch.setattr(outcomes, "plan_steps", _effect_plan_steps(planned))

    records = _finish(outcomes.completion_effects_steps(driver, plan, _outcome(call), "normal"))

    assert driver.hydrate_calls == [plan.record_id]
    assert driver.request_content_reads == 1
    assert driver.definition_calls == 1
    assert len(planned) == 1
    assert [record.record_id for record in records] == ["turn/model/1/attempt/1/tool/0"]


@pytest.mark.parametrize(
    "error",
    [
        KeyUnavailable("key unavailable"),
        MissingContentLink("missing content link"),
        IntegrityFailure("ciphertext tampered"),
    ],
    ids=["key-unavailable", "missing-link", "integrity-failure"],
)
def test_missing_effect_hydration_failure_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    driver = _Driver()
    driver.hydration_error = error
    monkeypatch.setattr(outcomes, "live_budget_steps", _no_budget)
    plan = _plan("turn/model/1", retained=True)
    driver.add_operation(plan)

    with pytest.raises(type(error)) as raised:
        list(
            outcomes.completion_effects_steps(
                driver,
                plan,
                _outcome({"id": "call", "name": "effect", "arguments": {}}),
                "normal",
            )
        )

    assert raised.value is error
