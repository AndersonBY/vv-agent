"""Cold S1 recovery keeps raw structural folding separate from semantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vv_agent import Agent, RunConfig, function_tool
from vv_agent.config import ResolvedModelConfig
from vv_agent.kernel._content_publication import declared_content_ref
from vv_agent.kernel._records import Record
from vv_agent.kernel._reducer import Fold, TransitionError, _operation, logical_operation_payloads
from vv_agent.kernel._state import Attempt, ExecutionState, Operation, Turn
from vv_agent.kernel._tool_values import ToolDirective, ToolExecutionResult, ToolResultStatus
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.session.kernel import _Driver as SessionDriver
from vv_agent.session.kernel import drive
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store
from vv_agent.session.records import InboxItem
from vv_agent.session.runtime import Runtime
from vv_agent.types import LLMResponse, ToolCall

from .conftest import open_store
from .helpers import content_ref, create_fixture_session, fixture_content_reader, push_fixture_input, record, wrap_fixture_store

SID = "s"
WORKSPACE = Path("/workspace")


def _runtime(database, store, llm: ScriptedLLM, tools=()) -> Runtime:
    return Runtime(
        agent=Agent("cold-recovery", "Resume the exact interaction.", tools=list(tools)),
        config=RunConfig(workspace=WORKSPACE, session_memory_enabled=False),
        resolved=ResolvedModelConfig("scripted", "scripted", "scripted", "scripted", []),
        llm=llm,
        heartbeat_store=lambda: open_store(database),
        content_reader=fixture_content_reader(store, SID),
    )


def _prepare_session(database) -> None:
    with open_store(database) as raw:
        raw.install_schema()
        store = wrap_fixture_store(raw)
        create_fixture_session(store, SID)
        push_fixture_input(store, SID, InboxItem("initial", "user", {"content": "start"}))


def _park_approval(database, effects: list[str]):
    @function_tool(needs_approval=True)
    def guarded() -> str:
        effects.append("ran")
        return "allowed"

    with open_store(database) as raw:
        store = wrap_fixture_store(raw)
        llm = ScriptedLLM([LLMResponse("", [ToolCall("guarded-call", "guarded", {})])])
        drive(store, SID, runtime=_runtime(database, store, llm, (guarded,)))
        return guarded, next(row for row in store.read(SID).records if row.record.kind == "op_parked")


@pytest.mark.persistent_store
def test_cold_approval_and_tool_result_resume_on_reopened_store(database, store_backend, monkeypatch: pytest.MonkeyPatch) -> None:
    del store_backend
    effects: list[str] = []
    _prepare_session(database)
    guarded, parked_row = _park_approval(database, effects)
    parked = parked_row.record
    assert declared_content_ref(parked.payload["handle"]["scope"]) is not None
    preflight_calls: list[tuple[str, ...]] = []
    original_preflight = SessionDriver._logical_preflight

    def spy_preflight(
        self: SessionDriver,
        new_records: list[Record],
        commit_records: list[Record],
        *,
        base_fold: Fold | None = None,
    ) -> None:
        assert base_fold is not None
        preflight_calls.append(tuple(new_record.kind for new_record in new_records))
        original_preflight(self, new_records, commit_records, base_fold=base_fold)

    monkeypatch.setattr(SessionDriver, "_logical_preflight", spy_preflight)

    with open_store(database) as raw:
        store = wrap_fixture_store(raw)
        hydration = hydration_for_session_store(store, SID)
        handle = hydrate_record_payload(parked_row, hydration)["handle"]
        push_fixture_input(
            store,
            SID,
            InboxItem(
                "approval-reply",
                "approval_answer",
                {
                    "operation_id": parked.operation_id,
                    "attempt": parked.attempt,
                    "request_id": handle["request_id"],
                    "request_digest": handle["request_digest"],
                    "scope": handle["scope"],
                    "decision": "approve",
                },
                target_turn_id=parked.turn_id,
                generation=0,
            ),
        )
        llm = ScriptedLLM([LLMResponse("complete")])
        drive(store, SID, runtime=_runtime(database, store, llm, (guarded,)))

        rows = store.read(SID).records
        applied = next(
            row.record
            for row in rows
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "approval-reply"
        )
        completed = next(
            row.record for row in rows if row.record.kind == "op_completed" and row.record.operation_id == parked.operation_id
        )
        assert applied.payload["disposition"] == "applied"
        assert declared_content_ref(applied.payload["input"]["payload"]["scope"]) is not None
        assert declared_content_ref(completed.payload["result"]) is not None
        assert effects == ["ran"]
        assert any("input_applied" in kinds for kinds in preflight_calls)
        assert any(row.record.kind == "turn_ended" and row.record.payload["status"] == "completed" for row in rows)


@pytest.mark.persistent_store
def test_cold_user_wait_resumes_on_reopened_store(database, store_backend) -> None:
    del store_backend
    question = "Which environment?"
    _prepare_session(database)

    with open_store(database) as raw:
        store = wrap_fixture_store(raw)
        llm = ScriptedLLM([LLMResponse("", [ToolCall("ask", "ask_user", {"question": question})])])
        drive(store, SID, runtime=_runtime(database, store, llm))
        parked_row = next(row for row in store.read(SID).records if row.record.kind == "op_parked")
        assert declared_content_ref(parked_row.record.payload["interaction_result"]) is not None

    with open_store(database) as raw:
        store = wrap_fixture_store(raw)
        hydration = hydration_for_session_store(store, SID)
        parked = parked_row.record
        wait = hydrate_record_payload(parked_row, hydration)
        push_fixture_input(
            store,
            SID,
            InboxItem(
                "user-reply",
                "user",
                {"content": {"interaction_id": wait["handle"]["interaction_id"], "text": "production"}},
                target_turn_id=parked.turn_id,
                generation=0,
            ),
        )
        llm = ScriptedLLM([LLMResponse("complete")])
        drive(store, SID, runtime=_runtime(database, store, llm))

        rows = store.read(SID).records
        applied = next(
            row.record
            for row in rows
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "user-reply"
        )
        assert applied.payload["disposition"] == "applied"
        assert declared_content_ref(applied.payload["input"]["payload"]["content"]) is not None
        assert not llm.steps
        assert any(row.record.kind == "turn_ended" and row.record.payload["status"] == "completed" for row in rows)


@pytest.mark.sqlite_file
def test_wrong_cold_approval_scope_is_rejected_before_dispatch(database, store_backend) -> None:
    del store_backend
    effects: list[str] = []
    _prepare_session(database)
    guarded, parked_row = _park_approval(database, effects)
    parked = parked_row.record

    with open_store(database) as raw:
        store = wrap_fixture_store(raw)
        hydration = hydration_for_session_store(store, SID)
        handle = hydrate_record_payload(parked_row, hydration)["handle"]
        dispatch_count = sum(row.record.kind == "op_started" for row in store.read(SID).records)
        push_fixture_input(
            store,
            SID,
            InboxItem(
                "wrong-approval",
                "approval_answer",
                {
                    "operation_id": parked.operation_id,
                    "attempt": parked.attempt,
                    "request_id": handle["request_id"],
                    "request_digest": handle["request_digest"],
                    "scope": ["wrong-tool-scope"],
                    "decision": "approve",
                },
                target_turn_id=parked.turn_id,
                generation=0,
            ),
        )
        drive(store, SID, runtime=_runtime(database, store, ScriptedLLM([]), (guarded,)))

        rows = store.read(SID).records
        applied = next(
            row.record
            for row in rows
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "wrong-approval"
        )
        assert applied.payload["disposition"] == "rejected"
        assert sum(row.record.kind == "op_started" for row in rows) == dispatch_count
        assert effects == []


def _interaction_state() -> tuple[ExecutionState, Any]:
    plan = Record(
        "op/o/planned",
        "op_planned",
        SID,
        "t",
        "o",
        1,
        {
            "op_kind": "interaction",
            "purpose": None,
            "request": content_ref("ask-request"),
            "request_digest": "a" * 64,
            "context_version": "1",
            "dependencies": [],
            "tool": content_ref("ask-user-tool"),
            "idempotency_key": None,
            "budget_admission": None,
            "not_before_ms": None,
            "provider_binding": "function",
            "consumed_unknowns": [],
            "policy_decision": None,
        },
    )
    state = ExecutionState(session_id=SID, active_turn_id="t")
    state.turns["t"] = Turn(record("turn_started"))
    state.operations["o"] = Operation("t", "interaction", attempts={1: Attempt(plan)})
    logical_plan = plan.payload | {
        "request": {
            "id": "ask-call",
            "name": "ask_user",
            "arguments": {"question": "Which?"},
            "provider_binding": None,
        }
    }
    return state, logical_plan


@pytest.mark.parametrize("retained", [False, True])
def test_interaction_result_schema_follows_raw_vs_semantic_boundary(retained: bool) -> None:
    state, logical_plan = _interaction_state()
    plan = state.operations["o"].attempts[1].execution_plan
    if retained:
        interaction_result = content_ref("interaction-result")
    else:
        interaction_result = ToolExecutionResult(
            tool_call_id="wrong-call",
            content="Which?",
            status_code=ToolResultStatus.WAIT_RESPONSE,
            directive=ToolDirective.WAIT_USER,
        ).to_dict()
    parked = Record(
        "op/o/parked",
        "op_parked",
        SID,
        "t",
        "o",
        1,
        {
            "phase": "before_dispatch",
            "handle": {"kind": "user", "interaction_id": "q", "question": content_ref("question")},
            "interaction_result": interaction_result,
            "poll_at_ms": None,
            "deadline_ms": None,
        },
    )

    with logical_operation_payloads({plan.record_id: logical_plan}):
        if retained:
            _operation(state, parked)
            assert state.operations["o"].attempts[1].wait is not None
        else:
            with pytest.raises(TransitionError, match="invalid interaction result"):
                _operation(state, parked)
            assert state.operations["o"].attempts[1].wait is None


def test_malformed_interaction_result_ref_is_rejected_before_state_mutation() -> None:
    state, logical_plan = _interaction_state()
    plan = state.operations["o"].attempts[1].execution_plan
    malformed = content_ref("interaction-result") | {"mac": "not-a-hash"}
    parked = Record(
        "op/o/parked",
        "op_parked",
        SID,
        "t",
        "o",
        1,
        {
            "phase": "before_dispatch",
            "handle": {"kind": "user", "interaction_id": "q", "question": content_ref("question")},
            "interaction_result": malformed,
            "poll_at_ms": None,
            "deadline_ms": None,
        },
    )

    with (
        logical_operation_payloads({plan.record_id: logical_plan}),
        pytest.raises(TransitionError, match="malformed ContentRef"),
    ):
        _operation(state, parked)
    assert state.operations["o"].attempts[1].wait is None
