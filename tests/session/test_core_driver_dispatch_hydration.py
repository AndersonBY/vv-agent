from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider, function_tool
from vv_agent.approval import ApprovalDecision
from vv_agent.kernel import _driver_dispatch as core_dispatch
from vv_agent.kernel import _driver_recovery as core_recovery
from vv_agent.kernel import _tool_admission as admission
from vv_agent.kernel._content_publication import declared_content_ref
from vv_agent.kernel._content_runtime import durable_content_reader
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_outcomes import PrepareToolResources
from vv_agent.kernel._lease_scope import LeaseOwnership
from vv_agent.kernel._provider_outcomes import Definitive
from vv_agent.kernel._reducer import TransitionError
from vv_agent.kernel._sql import stored_input_proof
from vv_agent.kernel._tool_metadata import ToolMetadata
from vv_agent.kernel._tool_schema import SchemaCheck
from vv_agent.kernel._tool_values import ToolExecutionResult, ToolResultStatus
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem, Record
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse, ToolCall

SESSION_ID = "core-dispatch-approval"
MASTER_KEY = b"d" * 32
WORKSPACE = "/workspace"


class _Cut(BaseException):
    pass


class _CoreScope(LeaseOwnership):
    def bind_turn(self, turn_id, generation, applied_inputs) -> None:
        del turn_id, generation, applied_inputs


def _sessions(store: SQLiteStore, provider: LocalKeyProvider) -> SessionDriver:
    store.install_schema()
    sessions = SessionDriver(store=store, content_key_provider=provider)
    sessions.create(SESSION_ID, WORKSPACE, {"seed": {"messages": [], "shared_state": {}}})
    sessions.push(SESSION_ID, InboxItem("initial", "user", {"content": "start"}))
    return sessions


def _runtime(sessions: SessionDriver, agent: Agent, responses: list[LLMResponse]):
    return sessions.runtime(
        agent,
        RunConfig(
            model_provider=ScriptedModelProvider.new("scripted", "scripted", responses),
            workspace=WORKSPACE,
        ),
        session_id=SESSION_ID,
    )


def _parked_row(store: SQLiteStore) -> Record:
    return next(row.record for row in store.read(SESSION_ID).records if row.record.kind == "op_parked")


def _applied_approval(store: SQLiteStore, input_id: str) -> Record:
    return next(
        row.record
        for row in store.read(SESSION_ID).records
        if row.record.kind == "input_applied"
        and row.record._payload["input"]["kind"] == "approval_answer"
        and row.record._payload["input"]["input_id"] == input_id
    )


@contextmanager
def _direct_driver(store: SQLiteStore, sessions: SessionDriver) -> Iterator[CoreDriver]:
    lease = sessions.store.acquire(SESSION_ID, owner="core-dispatch-test", ttl_ms=15_000)
    assert lease is not None
    scope = _CoreScope(
        lease,
        15_000,
        0.25,
        clock=store.clock,
        stop=Event(),
        ready=Event(),
        now_ms=store._now(),
    )
    driver = CoreDriver(
        sessions.store,
        SESSION_ID,
        scope,
        fold_factory=store.fold_factory,
        clock=store.clock,
        observe_cost=lambda: None,
        hook=lambda _point, _record: None,
        emit_phase=lambda _duration, _fields: None,
        content_reader=durable_content_reader(sessions.store, tenant_id=store.tenant_id, session_id=SESSION_ID),
    )
    try:
        yield driver
    finally:
        sessions.store.release(lease)


def _run_core_dispatch(driver: CoreDriver, attempt, *, invocations: list[str]) -> tuple[list[Any], list[dict[str, Any]]]:
    steps = core_dispatch.dispatch(driver, attempt)
    approved_calls: list[Any] = []
    denied_answers: list[dict[str, Any]] = []
    response: Any = None
    try:
        while True:
            request = steps.send(response)
            response = None
            if isinstance(request, (core_dispatch.PrepareContext, core_dispatch.PrepareAfterTool)):
                continue
            if isinstance(request, core_dispatch.FinishContext):
                response = {}
            elif request is core_dispatch.DispatchRead.HAS_HOOKS:
                response = False
            elif request is core_dispatch.DispatchRead.CONTEXT_CYCLE:
                response = 0
            elif isinstance(request, core_dispatch.SetContextCycle):
                continue
            elif isinstance(request, core_dispatch.ApproveSessionCall):
                approved_calls.append(request.call)
            elif isinstance(request, core_dispatch.ApprovalDenied):
                denied_answers.append(request.answer)
                response = ToolExecutionResult(
                    tool_call_id=request.plan._payload["request"]["id"],
                    status_code=ToolResultStatus.ERROR,
                    error_code="tool_approval_denied",
                    content="denied",
                )
            elif isinstance(request, core_dispatch.PreflightHandler):
                response = False
            elif isinstance(request, core_dispatch.BeginPreflight):
                response = [request.plan._payload["request"]["name"]]
            elif isinstance(request, core_dispatch.PrepareAdmission):
                response = True
            elif request is admission.Read.SCHEMA:
                response = {"type": "object", "properties": {}, "additionalProperties": False}
            elif isinstance(request, SchemaCheck):
                response = ()
            elif request is admission.Read.HANDLER_MANAGED:
                response = False
            elif request is admission.Read.POLICY_METADATA:
                response = {}
            elif request is admission.Read.TOOL_METADATA:
                response = ToolMetadata()
            elif isinstance(request, admission.ApprovedCall):
                response = approved_calls[-1] if approved_calls else None
            elif request is admission.Read.APPROVAL_MODE:
                response = "default"
            elif isinstance(request, admission.RequiresApproval):
                raise AssertionError("allow_session answer was not applied to the next call")
            elif isinstance(request, core_dispatch.InvokePreflight):
                response = (True, None)
            elif isinstance(request, core_dispatch.FinishPreflight) or request is core_dispatch.EndPreflight.CLEANUP:
                response = None
            elif isinstance(request, (core_dispatch.ChildBinding, core_dispatch.HasChild, core_dispatch.HasDelegation)):
                response = False
            elif isinstance(request, core_dispatch.SelectProvider):
                response = True
            elif request is core_dispatch.DispatchRead.POLL_LEASE:
                response = SimpleNamespace(inbox_seq=driver.watermark, db_now_ms=cast(Any, driver.store)._now())
            elif request is core_dispatch.DispatchRead.CANCELLED:
                response = False
            elif request is core_dispatch.DispatchRead.HANDLER_VERSION:
                assert driver.state.active_turn_id is not None
                response = driver.state.turns[driver.state.active_turn_id].start._payload["handler_version"]
            elif isinstance(request, core_dispatch.CommitDispatch):
                driver.commit(request.records, guarded=request.guarded)
            elif isinstance(request, core_dispatch.DispatchHook):
                response = None
            elif isinstance(request, core_dispatch.InvokeOperation):
                invocations.append(request.plan._payload["request"]["name"])
                response = Definitive(
                    ToolExecutionResult(
                        tool_call_id=request.plan._payload["request"]["id"],
                        content="executed",
                    ).to_dict(),
                    (f"test/{request.plan.operation_id}/evidence",),
                )
            elif isinstance(request, (core_dispatch.AfterTool, PrepareToolResources)):
                response = request.result
            elif request is core_dispatch.DispatchRead.PREPARE_AFTER_TOOL:
                response = None
            elif request is core_dispatch.DispatchRead.DURABLE_CONTEXT:
                response = driver.durable_shared_state()
            else:
                raise AssertionError(f"unexpected core dispatch request: {request!r}")
    except StopIteration:
        pass
    finally:
        steps.close()
    return approved_calls, denied_answers


def _database_snapshot(store: SQLiteStore) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    connection = store.connection
    names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    return tuple((name, tuple(sorted(connection.execute(f'SELECT * FROM "{name}"').fetchall(), key=repr))) for name in names)


@pytest.mark.sqlite_file
def test_core_dispatch_hydrates_allow_session_from_real_file_backed_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "allow-keys.sqlite3")
    effects: list[str] = []

    @function_tool(needs_approval=True)
    def effect() -> str:
        effects.append("effect")
        return "ok"

    agent = Agent("core-dispatch", "Exercise approval dispatch.", model="scripted", tools=[effect])
    try:
        with SQLiteStore.standalone(str(tmp_path / "allow.sqlite3")) as store:
            sessions = _sessions(store, key_provider)
            runtime = _runtime(
                sessions,
                agent,
                [
                    LLMResponse("", [ToolCall("effect-1", "effect", {})]),
                    LLMResponse("first turn complete"),
                    LLMResponse("", [ToolCall("effect-2", "effect", {})]),
                    LLMResponse("second turn complete"),
                ],
            )

            assert drive(sessions.store, SESSION_ID, runtime=runtime)
            parked = _parked_row(store)
            assert declared_content_ref(parked._payload["handle"]["scope"]) is not None
            assert effects == []
            sessions.approve(
                SESSION_ID,
                runtime,
                parked._payload["handle"]["request_id"],
                ApprovalDecision("allow_session", reason="scope granted", metadata={"channel": "test"}),
                "allow-session-answer",
            )
            assert drive(sessions.store, SESSION_ID, runtime=runtime)
            assert effects == ["effect"]
            retained_answer = _applied_approval(store, "allow-session-answer")
            assert declared_content_ref(retained_answer._payload["input"]["payload"]["scope"]) is not None

            sessions.push(SESSION_ID, InboxItem("again", "user", {"content": "run again"}))
            original_dispatch = core_recovery._dispatch
            target: dict[str, Any] = {}

            def cut_second_effect(driver, attempt):
                plan = driver.hydrate_plan(attempt.execution_plan)
                if (
                    plan.turn_id == f"{SESSION_ID}/turn/again"
                    and plan._payload["op_kind"] != "model"
                    and plan._payload["request"]["name"] == "effect"
                ):
                    target["operation_id"] = plan.operation_id
                    target["turn_id"] = plan.turn_id
                    raise _Cut
                return (yield from original_dispatch(driver, attempt))

            monkeypatch.setattr(core_recovery, "_dispatch", cut_second_effect)
            with pytest.raises(_Cut):
                drive(sessions.store, SESSION_ID, runtime=runtime)
            assert target

            invocations: list[str] = []
            with _direct_driver(store, sessions) as driver:
                operation = driver.state.operations[target["operation_id"]]
                attempt = operation.attempts[1]
                assert attempt.state == "planned"
                approved_calls, denied_answers = _run_core_dispatch(driver, attempt, invocations=invocations)
                assert [call.name for call in approved_calls] == ["effect"]
                assert denied_answers == []
                assert invocations == ["effect"]
                completed = [
                    row.record
                    for row in driver.records
                    if row.record.kind == "op_completed" and row.record.operation_id == target["operation_id"]
                ]
                assert len(completed) == 1
                assert completed[0]._payload["execution_started"] is True
            assert effects == ["effect"]
    finally:
        key_provider.close()


@pytest.mark.sqlite_file
def test_core_dispatch_hydrates_deny_answer_and_commits_no_external_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "deny-keys.sqlite3")
    effects: list[str] = []

    @function_tool(needs_approval=True)
    def effect() -> str:
        effects.append("effect")
        return "ok"

    agent = Agent("core-dispatch", "Exercise approval dispatch.", model="scripted", tools=[effect])
    try:
        with SQLiteStore.standalone(str(tmp_path / "deny.sqlite3")) as store:
            sessions = _sessions(store, key_provider)
            runtime = _runtime(sessions, agent, [LLMResponse("", [ToolCall("effect-1", "effect", {})])])
            assert drive(sessions.store, SESSION_ID, runtime=runtime)
            parked = _parked_row(store)
            sessions.approve(
                SESSION_ID,
                runtime,
                parked._payload["handle"]["request_id"],
                ApprovalDecision("deny", reason="not permitted", metadata={"channel": "test"}),
                "deny-answer",
            )
            original_dispatch = core_recovery._dispatch

            def cut_denied_dispatch(driver, attempt):
                if attempt.plan.operation_id == parked.operation_id and attempt.approval == "deny":
                    raise _Cut
                return (yield from original_dispatch(driver, attempt))

            monkeypatch.setattr(core_recovery, "_dispatch", cut_denied_dispatch)
            with pytest.raises(_Cut):
                drive(sessions.store, SESSION_ID, runtime=runtime)
            retained_answer = _applied_approval(store, "deny-answer")
            assert declared_content_ref(retained_answer._payload["input"]["payload"]["scope"]) is not None

            invocations: list[str] = []
            with _direct_driver(store, sessions) as driver:
                assert parked.operation_id is not None
                assert parked.attempt is not None
                attempt = driver.state.operations[parked.operation_id].attempts[parked.attempt]
                approved_calls, denied_answers = _run_core_dispatch(driver, attempt, invocations=invocations)
                assert approved_calls == []
                assert invocations == []
                assert len(denied_answers) == 1
                assert denied_answers[0]["scope"] == ["effect"]
                assert denied_answers[0]["reason"] == "not permitted"
                assert denied_answers[0]["metadata"] == {"channel": "test"}
                completed = [
                    row.record
                    for row in driver.records
                    if row.record.kind == "op_completed" and row.record.operation_id == parked.operation_id
                ]
                assert len(completed) == 1
                assert completed[0]._payload["execution_started"] is False
            assert effects == []
    finally:
        key_provider.close()


@pytest.mark.sqlite_file
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("scope", ["other"], "approval identity or scope mismatch"),
        ("operation", "wrong-operation", "approval operation mismatch"),
        ("request_digest", "0" * 64, "approval identity or scope mismatch"),
    ],
)
def test_core_commit_rolls_back_mismatched_approval_answer(
    tmp_path: Path,
    field: str,
    value: Any,
    error: str,
) -> None:
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / f"negative-{field}-keys.sqlite3")

    @function_tool(needs_approval=True)
    def effect() -> str:
        return "ok"

    agent = Agent("core-dispatch", "Exercise approval dispatch.", model="scripted", tools=[effect])
    try:
        with SQLiteStore.standalone(str(tmp_path / f"negative-{field}.sqlite3")) as store:
            sessions = _sessions(store, key_provider)
            runtime = _runtime(sessions, agent, [LLMResponse("", [ToolCall("effect-1", "effect", {})])])
            assert drive(sessions.store, SESSION_ID, runtime=runtime)
            parked = _parked_row(store)
            handle = parked._payload["handle"]
            generation = next(
                row.record._payload["generation"]
                for row in store.read(SESSION_ID).records
                if row.record.kind == "turn_started" and row.record.turn_id == parked.turn_id
            )
            payload = {
                "operation_id": parked.operation_id,
                "attempt": parked.attempt,
                "request_id": handle["request_id"],
                "request_digest": handle["request_digest"],
                "scope": ["effect"],
                "decision": "allow_session",
            }
            if field == "operation":
                payload["operation_id"] = value
            else:
                payload[field] = value
            input_id = f"negative-{field}"
            sessions.push(
                SESSION_ID,
                InboxItem(input_id, "approval_answer", payload, parked.turn_id, generation=generation),
            )
            stored_input = next(entry for entry in sessions.store.peek_inbox(SESSION_ID) if entry.item.input_id == input_id)
            assert declared_content_ref(stored_input.item.payload["scope"]) is not None
            with _direct_driver(store, sessions) as driver:
                retained = stored_input.item
                applied = driver.record(
                    "input_applied",
                    {
                        "input": retained.to_dict(),
                        "input_digest": retained.digest,
                        "input_proof": stored_input_proof(stored_input),
                        "disposition": "applied",
                        "reason": None,
                        "target_operation_id": retained.payload["operation_id"],
                        "target_wait_id": retained.payload["request_id"],
                        "position": "between_actions",
                    },
                    tid=parked.turn_id,
                )
                before = _database_snapshot(store)
                with pytest.raises(TransitionError, match=error):
                    driver.commit([applied], consumed=(input_id,))
                assert _database_snapshot(store) == before
                assert sessions.store.peek_inbox(SESSION_ID)[0].item.input_id == input_id
    finally:
        key_provider.close()
