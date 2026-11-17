from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, cast

from vv_agent import Agent, RunConfig, ScriptedModelProvider, function_tool
from vv_agent.approval import ApprovalDecision
from vv_agent.interaction import InteractionAnswer, InteractionReplyTarget
from vv_agent.kernel import _driver_inputs as inputs
from vv_agent.kernel import _driver_recovery
from vv_agent.kernel._content_publication import declared_content_ref
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_outcomes import PrepareToolResources
from vv_agent.kernel._interactions import reply_content
from vv_agent.session._driver_scope import _Scope
from vv_agent.session._recovery_phase import _Host
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import drive
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store
from vv_agent.session.records import InboxItem
from vv_agent.session.runtime import Runtime
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.tool_results import prepare_tool_resources
from vv_agent.types import LLMResponse, ToolCall

_KEY = LocalKeyProvider(b"k" * 32)


class _Calls:
    def __init__(self) -> None:
        self.models: list[Any] = []
        self.effects: list[str] = []


class _HydrationDriver(Protocol):
    runtime: Runtime
    scope: _Scope


def _runtime(sessions: SessionDriver, calls: _Calls, responses, *, no_tool_policy="finish"):
    def complete(request):
        calls.models.append(request)
        return responses[len(calls.models) - 1]

    return sessions.runtime(
        Agent("core-input", "Exercise retained input hydration.", model="scripted"),
        RunConfig(
            model_provider=ScriptedModelProvider.from_callback("scripted", "scripted", complete),
            workspace="/workspace",
            no_tool_policy=no_tool_policy,
        ),
        session_id="s",
    )


def _core_driver(sessions: SessionDriver, runtime):
    lease = sessions.store.acquire("s", owner="core-input-hydration", ttl_ms=15_000)
    assert lease is not None
    store = cast(SQLiteStore, sessions.store)
    scope = _Scope(lease, runtime, now_ms=store._now())
    driver = CoreDriver(
        sessions.store,
        "s",
        scope,
        fold_factory=store.fold_factory,
        clock=runtime.clock,
        observe_cost=lambda: None,
        hook=lambda _point, _record: None,
        emit_phase=lambda _duration, _fields: None,
        content_reader=runtime.content_reader,
    )
    cast(_HydrationDriver, driver).runtime = runtime
    return driver, lease


def _tool_context(driver: CoreDriver, plan):
    turn_id = driver.state.active_turn_id
    assert turn_id is not None
    task = driver.hydrated_definition(driver.state.turns[turn_id].start).task
    hydration_driver = cast(_HydrationDriver, driver)
    shared = hydration_driver.runtime.bind_state(driver.durable_shared_state(), task)
    return hydration_driver.runtime.context(
        task,
        plan,
        hydration_driver.scope.token,
        shared_state=shared,
        store=driver.store,
        session_id=driver.sid,
    )


def _apply_core_input(driver: CoreDriver) -> bool:
    steps = _driver_recovery._compose(driver, inputs.apply_input(driver, one_turn=False))
    host = _Host(driver)
    response = None
    while True:
        try:
            request = steps.send(response)
        except StopIteration as done:
            return done.value
        response = None
        if isinstance(request, inputs.PrepareToolContext):
            host.dispatch().context = _tool_context(driver, request.plan)
        elif isinstance(request, inputs.CommitInput):
            driver.commit(request.records, (request.input_id,), guarded=True)
            response = True
        elif isinstance(request, PrepareToolResources):
            runtime = cast(_HydrationDriver, driver).runtime
            response = prepare_tool_resources(request.plan, request.result, runtime.workspace_backend)
        else:
            response = host.handle(request)


def _content_rows(store: SQLiteStore) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        store._rows(
            "SELECT content_id,key_id,mac,size,labels,nonce,created_ms,state,pin_until_ms,ciphertext,object_locator "
            "FROM sk_content WHERE tenant_id=? AND session_id=? ORDER BY content_id",
            (store.tenant_id, "s"),
        )
    )


def _assert_retained_for_rejection(store: SQLiteStore, before_rows, before_kinds, calls: _Calls) -> None:
    records = store.read("s").records
    assert tuple(row.record.record_id for row in records[: len(before_kinds)]) == tuple(before_kinds)
    appended = records[len(before_kinds) :]
    assert [row.record.kind for row in appended] == ["usage_observed", "input_applied"]
    assert appended[-1].record.payload["disposition"] == "rejected"
    retained = {row[0]: row for row in before_rows}
    current = {row[0]: row for row in _content_rows(store)}
    assert all(current[content_id] == row for content_id, row in retained.items())
    assert len(calls.models) == 1
    assert calls.effects == []


def _input_application_reason(store: SQLiteStore, input_id: str) -> str:
    hydration = hydration_for_session_store(store, "s")
    stored = next(
        row
        for row in store.read("s").records
        if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == input_id
    )
    return hydrate_record_payload(stored, hydration)["reason"]


def _ask_user_target(sessions: SessionDriver, runtime) -> InteractionReplyTarget:
    form = next(wait["interaction"] for _sid, _runtime, wait in sessions.waits("s", runtime) if "interaction" in wait)
    names = ("sessionId", "turnId", "generation", "interactionId", "operationId", "attempt", "requestDigest")
    return InteractionReplyTarget.from_dict({name: form[name] for name in names})


def _park_ask_user(path: Path, calls: _Calls):
    calls.effects.clear()
    stack = ExitStack()
    store = stack.enter_context(SQLiteStore.standalone(str(path)))
    store.install_schema()
    sessions = SessionDriver(store=store, content_key_provider=_KEY)
    sessions.create("s", "/workspace", {"seed": {"messages": [], "shared_state": {}}})
    sessions.push("s", InboxItem("initial", "user", {"content": "ask me"}))
    runtime = _runtime(
        sessions,
        calls,
        [LLMResponse("", [ToolCall("question", "ask_user", {"question": "Which environment?"})])],
    )
    assert drive(sessions.store, "s", runtime=runtime)
    parked = next(row.record for row in store.read("s").records if row.record.kind == "op_parked")
    return stack, store, sessions, runtime, _ask_user_target(sessions, runtime), parked


def _park_turn_wait(path: Path, calls: _Calls):
    stack = ExitStack()
    store = stack.enter_context(SQLiteStore.standalone(str(path)))
    store.install_schema()
    sessions = SessionDriver(store=store, content_key_provider=_KEY)
    sessions.create("s", "/workspace", {"seed": {"messages": [], "shared_state": {}}})
    sessions.push("s", InboxItem("initial", "user", {"content": "ask me"}))
    runtime = _runtime(sessions, calls, [LLMResponse("Which environment?")], no_tool_policy="wait_user")
    assert drive(sessions.store, "s", runtime=runtime)
    parked = next(row.record for row in store.read("s").records if row.record.kind == "turn_parked")
    assert declared_content_ref(parked.payload["question"]) is not None
    return stack, store, sessions, runtime, _ask_user_target(sessions, runtime), parked


def test_core_driver_applies_retained_turn_parked_answer_once(tmp_path: Path) -> None:
    calls = _Calls()
    stack, store, sessions, runtime, target, _parked = _park_turn_wait(tmp_path / "turn-wait.sqlite", calls)
    try:
        sessions.answer("s", runtime, target, InteractionAnswer("text", text="staging"), "reply")
        driver, lease = _core_driver(sessions, runtime)
        try:
            assert not hasattr(driver, "_logical_state")
            assert _apply_core_input(driver)
        finally:
            assert sessions.store.release(lease)

        calls.models.clear()
        continuation = _runtime(sessions, calls, [LLMResponse("finished")])
        assert drive(sessions.store, "s", runtime=continuation)
        assert len(calls.models) == 1
        applied = [
            row.record
            for row in store.read("s").records
            if row.record.kind == "input_applied"
            and row.record.payload["input"]["input_id"] == "reply"
            and row.record.payload["disposition"] == "applied"
        ]
        assert len(applied) == 1
    finally:
        stack.close()


def test_core_driver_applies_retained_ask_user_answer_once(tmp_path: Path) -> None:
    calls = _Calls()
    stack, store, sessions, runtime, target, parked = _park_ask_user(tmp_path / "ask-valid.sqlite", calls)
    try:
        sessions.answer("s", runtime, target, InteractionAnswer("text", text="staging"), "reply")
        driver, lease = _core_driver(sessions, runtime)
        try:
            assert not hasattr(driver, "_logical_state")
            assert _apply_core_input(driver)
        finally:
            assert sessions.store.release(lease)

        calls.models.clear()
        continuation = _runtime(sessions, calls, [LLMResponse("finished")])
        assert drive(sessions.store, "s", runtime=continuation)
        assert len(calls.models) == 1
        applied = [
            row.record
            for row in store.read("s").records
            if row.record.kind == "input_applied"
            and row.record.payload["input"]["input_id"] == "reply"
            and row.record.payload["disposition"] == "applied"
        ]
        assert len(applied) == 1
        completed = [
            row.record
            for row in store.read("s").records
            if row.record.kind == "op_completed" and row.record.operation_id == parked.operation_id
        ]
        assert len(completed) == 1
    finally:
        stack.close()


def test_core_driver_rejects_stale_ask_user_target_without_retained_mutation(tmp_path: Path) -> None:
    calls = _Calls()
    stack, store, sessions, runtime, target, parked = _park_ask_user(tmp_path / "ask-stale.sqlite", calls)
    try:
        stale = replace(target, request_digest="0" * 64)
        sessions.push(
            "s",
            InboxItem(
                "reply",
                "user",
                {"content": reply_content(stale, InteractionAnswer("text", text="staging"))},
                stale.turn_id,
                stale.generation,
            ),
        )
        before_rows = _content_rows(store)
        before_records = [row.record.record_id for row in store.read("s").records]
        driver, lease = _core_driver(sessions, runtime)
        try:
            assert _apply_core_input(driver)
        finally:
            assert sessions.store.release(lease)
        _assert_retained_for_rejection(store, before_rows, before_records, calls)
        assert _input_application_reason(store, "reply") == "stale_interaction"
        assert parked.encode() in {row.record.encode() for row in store.read("s").records}
    finally:
        stack.close()


def _approval_wait(sessions: SessionDriver, runtime):
    return next(wait for _sid, _runtime, wait in sessions.waits("s", runtime) if wait.get("kind") == "approval")


def _park_approval(path: Path, calls: _Calls):
    @function_tool(needs_approval=True)
    def guarded() -> str:
        calls.effects.append("ran")
        return "allowed"

    stack = ExitStack()
    store = stack.enter_context(SQLiteStore.standalone(str(path)))
    store.install_schema()
    sessions = SessionDriver(store=store, content_key_provider=_KEY)
    sessions.create("s", "/workspace", {"seed": {"messages": [], "shared_state": {}}})
    sessions.push("s", InboxItem("initial", "user", {"content": "run guarded"}))
    runtime = sessions.runtime(
        Agent("core-input", "Exercise retained input hydration.", model="scripted", tools=[guarded]),
        RunConfig(
            model_provider=ScriptedModelProvider.from_callback(
                "scripted",
                "scripted",
                lambda request: (
                    calls.models.append(request)
                    or (
                        LLMResponse("", [ToolCall("guarded-call", "guarded", {})])
                        if len(calls.models) == 1
                        else LLMResponse("finished")
                    )
                ),
            ),
            workspace="/workspace",
        ),
        session_id="s",
    )
    assert drive(sessions.store, "s", runtime=runtime)
    parked = next(row.record for row in store.read("s").records if row.record.kind == "op_parked")
    return stack, store, sessions, runtime, _approval_wait(sessions, runtime), parked


def test_core_driver_applies_retained_approval_once(tmp_path: Path) -> None:
    calls = _Calls()
    stack, store, sessions, runtime, wait, _parked = _park_approval(tmp_path / "approval-valid.sqlite", calls)
    try:
        sessions.approve("s", runtime, wait["request_id"], ApprovalDecision.allow(), "approval-reply")
        driver, lease = _core_driver(sessions, runtime)
        try:
            assert not hasattr(driver, "_logical_state")
            assert _apply_core_input(driver)
        finally:
            assert sessions.store.release(lease)

        assert drive(sessions.store, "s", runtime=runtime)
        assert len(calls.models) == 2
        assert calls.effects == ["ran"]
        applied = [
            row.record
            for row in store.read("s").records
            if row.record.kind == "input_applied"
            and row.record.payload["input"]["input_id"] == "approval-reply"
            and row.record.payload["disposition"] == "applied"
        ]
        assert len(applied) == 1
    finally:
        stack.close()


def test_core_driver_rejects_stale_approval_request_digest_without_retained_mutation(tmp_path: Path) -> None:
    calls = _Calls()
    stack, store, sessions, runtime, wait, parked = _park_approval(tmp_path / "approval-stale.sqlite", calls)
    try:
        payload = {key: wait[key] for key in ("operation_id", "attempt", "request_id", "request_digest", "scope")}
        payload.update(request_digest="0" * 64, decision="approve")
        sessions.push(
            "s",
            InboxItem(
                "approval-reply",
                "approval_answer",
                payload,
                wait["turn_id"],
                0,
            ),
        )
        before_rows = _content_rows(store)
        before_records = [row.record.record_id for row in store.read("s").records]
        driver, lease = _core_driver(sessions, runtime)
        try:
            assert _apply_core_input(driver)
        finally:
            assert sessions.store.release(lease)
        _assert_retained_for_rejection(store, before_rows, before_records, calls)
        assert _input_application_reason(store, "approval-reply") == "approval identity or scope mismatch"
        assert parked.encode() in {row.record.encode() for row in store.read("s").records}
    finally:
        stack.close()
