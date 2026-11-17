from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent import Agent, RunConfig, ScriptedModelProvider
from vv_agent.kernel import _driver_recovery
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._content_publication import make_retained_record_before_publication
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._reducer import TransitionError
from vv_agent.kernel._store import Lease
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.runtime.hooks import BaseRuntimeHook, BeforeToolCallPatch
from vv_agent.session import _recovery_phase, kernel
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem, Record
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.tool_results import prepare_tool_resources
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.metadata import ToolIdempotency, ToolMetadata
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from .controlled import ControlledProvider
from .helpers import push_fixture_input

SESSION_ID = "core-retry-prepared"
MASTER_KEY = b"p" * 32


class _Cut(BaseException):
    pass


class _PatchedArgumentsHook(BaseRuntimeHook):
    def before_tool_call(self, event):
        return BeforeToolCallPatch(call=ToolCall(event.call.id, event.call.name, {"text": "patched"}))


def _producer(
    store: SQLiteStore,
    database: Path,
    provider: LocalKeyProvider,
) -> tuple[SessionDriver, object, ControlledProvider, ScriptedModelProvider]:
    store.install_schema()
    sessions = SessionDriver(store=store, content_key_provider=provider)
    sessions.create(SESSION_ID, "/workspace", {"seed": {"messages": [], "shared_state": {}}})
    push_fixture_input(store, SESSION_ID, InboxItem("input", "user", {"content": "go"}))

    external = ControlledProvider(database, "unknown", idempotent=True)
    external.install()
    tool = FunctionTool(
        name="effect",
        description="effect",
        params_json_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        on_invoke=lambda _context, _args: ToolExecutionResult(tool_call_id="unused", content="unused"),
        tool_metadata=ToolMetadata(idempotency=ToolIdempotency.SUPPORTED),
    )
    model = ScriptedModelProvider.new(
        "scripted",
        "scripted",
        [LLMResponse("", [ToolCall("call", "effect", {"text": "original"})]), LLMResponse("done")],
    )
    runtime = sessions.runtime(
        Agent("retry", "retry once", model="scripted", tools=[tool], hooks=[_PatchedArgumentsHook()]),
        RunConfig(
            model_provider=model,
            workspace="/workspace",
            session_memory_enabled=False,
            tool_providers={"effect": external},
        ),
        session_id=SESSION_ID,
    )
    return sessions, runtime, external, model


def _capture_unknown(
    monkeypatch: pytest.MonkeyPatch,
    sessions: SessionDriver,
    runtime,
) -> list[Record]:
    captured: list[Record] = []
    original = kernel._Driver.commit

    def cut_unknown_commit(self, records, consumed=(), *, guarded=False, prepare=None, _logical_base=None):
        if any(record.kind == "op_unknown" for record in records):
            captured.extend(records)
            raise _Cut
        return original(
            self,
            records,
            consumed,
            guarded=guarded,
            prepare=prepare,
            _logical_base=_logical_base,
        )

    monkeypatch.setattr(kernel._Driver, "commit", cut_unknown_commit)
    with pytest.raises(_Cut):
        drive(sessions.store, SESSION_ID, runtime=runtime)
    assert [record.kind for record in captured] == ["op_unknown"]
    return captured


def _direct_driver(store: SQLiteStore, sessions: SessionDriver, runtime) -> tuple[CoreDriver, Lease]:
    lease = sessions.store.acquire(SESSION_ID, owner="core-retry", ttl_ms=15_000)
    assert lease is not None
    scope = kernel._Scope(lease, runtime, now_ms=store._now())
    scope.poll(store)
    driver = CoreDriver(
        sessions.store,
        SESSION_ID,
        scope,
        fold_factory=store.fold_factory,
        clock=store.clock,
        observe_cost=lambda: None,
        hook=lambda _point, _record: None,
        emit_phase=lambda _duration, _fields: None,
        content_reader=runtime.content_reader,
    )
    cast(Any, driver).runtime = runtime
    assert not hasattr(driver, "_logical_state_scope")
    return driver, lease


def _run_core_step(driver: CoreDriver, *, one_turn: bool = False) -> bool:
    host = _recovery_phase._Host(driver)
    composed = host.steps(_driver_recovery._compose(driver, _driver_recovery.step(driver, one_turn=one_turn)))
    return kernel._Driver._run_outcome_steps(
        cast(Any, driver),
        cast(Any, composed),
        prepare_resources=prepare_tool_resources,
    )


def _table_counts(store: SQLiteStore) -> dict[str, int]:
    return {
        table: int(
            store._one(
                f"SELECT COUNT(*) FROM {table} WHERE tenant_id=? AND session_id=?",
                (store.tenant_id, SESSION_ID),
            )[0]
        )
        for table in ("sk_record", "sk_commit", "sk_session_key", "sk_content", "sk_content_link")
    }


def _prepared_attempt(driver: CoreDriver):
    tool_operations = [operation for operation in driver.state.operations.values() if operation.kind == "tool"]
    assert len(tool_operations) == 1
    attempt = tool_operations[0].attempts[1]
    assert attempt.prepared is not None
    return attempt


@pytest.mark.sqlite_file
def test_core_retry_uses_unprepared_plan_identity_without_dispatching(tmp_path: Path, monkeypatch) -> None:
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "keys.sqlite3")
    database = tmp_path / "session.sqlite3"
    try:
        with SQLiteStore.standalone(str(database)) as store:
            sessions, runtime, external, model = _producer(store, database, key_provider)
            model_steps = cast(ScriptedLLM, model.llm).steps
            unknown = _capture_unknown(monkeypatch, sessions, runtime)
            assert external.counts() == (1, 1)
            assert len(model_steps) == 1

            driver, lease = _direct_driver(store, sessions, runtime)
            try:
                driver.commit(unknown)
                attempt = _prepared_attempt(driver)
                assert ContentRef.from_dict(attempt.plan._payload["request"])
                hook_result = attempt.prepared._payload["hook_result"]
                assert hook_result is None or ContentRef.from_dict(hook_result)
                assert ContentRef.from_dict(attempt.prepared._payload["shared_state"])
                hydrated_prepared = driver.hydrate_plan(attempt.execution_plan)
                assert hydrated_prepared._payload["request"]["arguments"] == {"text": "patched"}
                assert hydrated_prepared._payload["budget_admission"]["shared_state"] == {}

                assert _run_core_step(driver)

                retry = driver.state.operations[attempt.plan.operation_id].attempts[2]
                assert retry.plan._payload["request"] == attempt.plan._payload["request"]
                assert retry.plan._payload["request_digest"] == attempt.plan._payload["request_digest"]
                assert ContentRef.from_dict(retry.plan._payload["budget_admission"])
                assert retry.plan._payload["budget_admission"] == attempt.plan._payload["budget_admission"]
                assert not any(record.record.kind == "op_started" and record.record.attempt == 2 for record in driver.records)
                assert external.counts() == (1, 1)
                assert len(model_steps) == 1
            finally:
                sessions.store.release(lease)
    finally:
        key_provider.close()


@pytest.mark.sqlite_file
def test_core_rejects_prepared_payload_retry_without_effects_or_ledger_change(tmp_path: Path, monkeypatch) -> None:
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "keys.sqlite3")
    database = tmp_path / "session.sqlite3"
    try:
        with SQLiteStore.standalone(str(database)) as store:
            sessions, runtime, external, model = _producer(store, database, key_provider)
            model_steps = cast(ScriptedLLM, model.llm).steps
            unknown = _capture_unknown(monkeypatch, sessions, runtime)
            assert external.counts() == (1, 1)
            assert len(model_steps) == 1

            driver, lease = _direct_driver(store, sessions, runtime)
            try:
                driver.commit(unknown)
                attempt = _prepared_attempt(driver)
                bad_retry = make_retained_record_before_publication(
                    "op_planned",
                    session_id=SESSION_ID,
                    turn_id=attempt.plan.turn_id,
                    operation_id=attempt.plan.operation_id,
                    attempt=2,
                    payload=attempt.execution_plan._payload | {"not_before_ms": 0},
                )
                before = _table_counts(store)

                with pytest.raises(TransitionError, match="retry changed request"):
                    driver.commit([bad_retry])

                assert _table_counts(store) == before
                assert 2 not in driver.state.operations[attempt.plan.operation_id].attempts
                assert bad_retry.record_id not in {row.record.record_id for row in store.read(SESSION_ID).records}
                assert external.counts() == (1, 1)
                assert len(model_steps) == 1
            finally:
                sessions.store.release(lease)
    finally:
        key_provider.close()
