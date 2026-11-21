"""W4 cursor reads and lease ownership at the public local worker boundary."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from gc import collect
from pathlib import Path
from threading import Event, Thread, current_thread
from time import monotonic, sleep
from typing import Any, cast

import pytest

import vv_agent.session.surfaces as surface_module
import vv_agent.session.timing as session_timing
from vv_agent import Agent, RunConfig, Runner, SubAgentConfig
from vv_agent.kernel._sqlite import SQLiteStore as CoreSQLiteStore
from vv_agent.model import ScriptedModelProvider
from vv_agent.session.children import ChildSession
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.runtime import static_session_requirements
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.store import LeaseLost, LeaseRetryExhausted
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.timing import ThreadLeaseKeeper
from vv_agent.tools.builtins import build_default_registry
from vv_agent.tools.function import function_tool
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import AgentStatus, LLMResponse, Message, ToolCall, ToolExecutionResult

from .conftest import open_store
from .helpers import (
    append_fixture_records,
    create_fixture_session,
    fixture_key_provider,
    record,
    wrap_fixture_store,
)
from .test_delegation_parity import routed
from .test_yieldable_waits import YieldProvider


def _wait_for(predicate, timeout: float = 8.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.02)
    raise AssertionError("condition was not reached before timeout")


def test_snapshot_status_linearizes_schedule_before_competing_claim(database, monkeypatch) -> None:
    provider = fixture_key_provider()
    claimed = []
    injected = False
    with ExitStack() as stack:
        first = stack.enter_context(open_store(database))
        first.install_schema()
        second = stack.enter_context(open_store(database))
        first.configure_worker(level=1, handler_versions=("1",), capabilities=())
        second.configure_worker(level=1, handler_versions=("1",), capabilities=())
        wrapped = wrap_fixture_store(first, provider)
        create_fixture_session(wrapped, "snapshot", consumers=())
        append_fixture_records(
            wrapped,
            "snapshot",
            (
                record("turn_started", sid="snapshot"),
                record("op_planned", sid="snapshot"),
                record("op_started", sid="snapshot"),
                record("op_completed", sid="snapshot"),
                record("turn_ended", sid="snapshot"),
            ),
            commit_id="snapshot-terminal",
        )
        with first.atomic():
            first._rows(
                "UPDATE sk_session SET next_drive_ms=0,drive_retry_at_ms=9223372036854775807 "
                "WHERE tenant_id=%s AND session_id=%s",
                (first.tenant_id, "snapshot"),
            )

        original_lease_status = type(first).lease_status

        def lease_status(store, session_id):
            nonlocal injected
            result = original_lease_status(store, session_id)
            if store is first and not injected:
                injected = True
                with second.atomic():
                    second._rows(
                        "UPDATE sk_session SET drive_retry_at_ms=0 WHERE tenant_id=%s AND session_id=%s",
                        (second.tenant_id, session_id),
                    )
                claimed.append(second.acquire(session_id, owner="racing-worker", ttl_ms=10_000))
            return result

        monkeypatch.setattr(type(first), "lease_status", lease_status)
        driver = SessionDriver(store=first, content_key_provider=provider)
        try:
            _, quiescent, error = driver._snapshot_status(first, "snapshot", None, "t")
            assert error is None
            assert quiescent is False
            assert injected and claimed[0] is not None
            assert claimed[0].owner == "racing-worker"
        finally:
            if claimed and claimed[0] is not None:
                second.release(claimed[0])
            driver.close()


@pytest.mark.persistent_store
def test_event_page_uses_one_captured_head_and_observer_cursor(database, monkeypatch) -> None:
    provider_key = fixture_key_provider()
    entered = Event()
    released = Event()
    writer_errors = []
    with ExitStack() as stack:
        first = stack.enter_context(open_store(database))
        first.install_schema()
        second = stack.enter_context(open_store(database))
        first.configure_worker(level=1, handler_versions=("1",), capabilities=())
        second.configure_worker(level=1, handler_versions=("1",), capabilities=())
        first_view = wrap_fixture_store(first, provider_key)
        second_view = wrap_fixture_store(second, provider_key)
        driver = SessionDriver(store=first_view)
        external = YieldProvider(declared=False)

        @function_tool
        def external_call() -> str:
            raise AssertionError("external_call provider owns dispatch")

        provider = ScriptedModelProvider.new(
            "scripted",
            "m",
            [
                LLMResponse("", [ToolCall("call", "external_call", {})]),
                LLMResponse("resumed done"),
            ],
        )
        agent = Agent("event-page", "Answer.", model="m", tools=[external_call])
        config = RunConfig(model_provider=provider, tool_providers={"external_call": external})
        driver.create("event-page", ".")
        initial = driver.start("event-page", agent, config, "go")
        try:
            assert initial.result(timeout=8).status == AgentStatus.WAIT_USER
            captured = first.read("event-page", limit=1).head_seq
            driver._worker.stop()
            assert driver._worker.join(5)
            original_status = driver._snapshot_status

            def paused_status(store, sid, state, requested_turn_id, *, captured_head=None):
                result = original_status(
                    store,
                    sid,
                    state,
                    requested_turn_id,
                    captured_head=captured_head,
                )
                entered.set()
                assert released.wait(5)
                return result

            def resume_from_second_store():
                try:
                    writer = SessionDriver(store=second_view)
                    try:
                        handle = external.handle
                        writer.push(
                            "event-page",
                            InboxItem(
                                "result",
                                "provider_result",
                                {
                                    "operation_id": handle["operation_id"],
                                    "attempt": handle["attempt"],
                                    "request_digest": handle["request_digest"],
                                    "provider_binding": handle["provider"],
                                    "result": ToolExecutionResult("call", "external done").to_dict(),
                                    "usage": {},
                                    "evidence": [handle["evidence"]],
                                },
                                initial.run_id,
                                0,
                            ),
                        )
                        runtime = driver._build_runtime_for_claim(second_view, "event-page")
                        drive(second_view, "event-page", runtime=runtime)
                    finally:
                        writer.close()
                except BaseException as exc:
                    writer_errors.append(exc)

            page_results = []

            def read_event_page():
                try:
                    page_results.append(
                        driver._handle_event_page(
                            "event-page",
                            initial.run_id,
                            after_seq=captured,
                        )
                    )
                except BaseException as exc:
                    writer_errors.append(exc)

            monkeypatch.setattr(driver, "_snapshot_status", paused_status)
            thread = Thread(target=read_event_page)
            thread.start()
            assert entered.wait(5)
            resume_from_second_store()
            released.set()
            thread.join(8)
            assert not thread.is_alive()
            assert not writer_errors
            assert len(page_results) == 1
            tid, events, next_seq, has_more, quiescent, error = page_results[0]
            assert (tid, events, next_seq, has_more, quiescent, error) == (
                initial.run_id,
                [],
                captured,
                False,
                True,
                None,
            )

            _, resumed, next_seq, has_more, quiescent, error = driver._handle_event_page(
                "event-page",
                initial.run_id,
                after_seq=captured,
            )
            assert resumed
            assert next_seq > captured
            assert has_more is False
            assert quiescent is False
            assert error is None
        finally:
            released.set()
            driver.close()


@function_tool
def background_child() -> str:
    raise AssertionError("background_child must use child admission")


@function_tool(needs_approval=True)
def approval_effect() -> str:
    return "approved"


def test_public_worker_support_includes_planned_tool_requirement(tmp_path: Path) -> None:
    tool = build_default_registry().get_executor("bash")
    agent = Agent("worker-tags", "Run the command.", model="m", tools=[tool])
    config = RunConfig(
        workspace=tmp_path,
        model_provider=ScriptedModelProvider.new(
            "scripted",
            "m",
            [
                LLMResponse("", [ToolCall("run", "bash", {"command": "printf kernel-parity"})]),
                LLMResponse("done"),
            ],
        ),
        tool_registry_factory=ToolRegistry,
    )
    required = static_session_requirements(
        agent,
        config,
        resolved=None,
        handler_version="1",
    )
    handle = Runner.start(agent, "go", run_config=config)
    assert set(required.capabilities) <= set(handle.kernel._worker.support.capabilities)
    result = handle.result(timeout=8)
    assert [message.content for message in result.raw_result.messages if message.role == "tool"] == ["kernel-parity"]


def test_worker_does_not_second_release_when_keeper_close_cannot_stop(tmp_path: Path, monkeypatch) -> None:
    renew_entered = Event()
    close_entered = Event()
    release_renew = Event()
    model_entered = Event()
    blocked_once = False
    original_renew = SQLiteStore.renew
    original_close = ThreadLeaseKeeper.close

    def blocked_renew(store, lease, *, ttl_ms):
        nonlocal blocked_once
        if model_entered.is_set() and not blocked_once:
            blocked_once = True
            renew_entered.set()
            assert release_renew.wait(10)
        return original_renew(store, lease, ttl_ms=ttl_ms)

    def tracked_close(self, heartbeat):
        close_entered.set()
        return original_close(self, heartbeat)

    monkeypatch.setattr(SQLiteStore, "renew", blocked_renew)
    monkeypatch.setattr(ThreadLeaseKeeper, "close", tracked_close)
    monkeypatch.setattr(session_timing, "HEARTBEAT_CLOSE_SECONDS", 0.05)

    with open_store(tmp_path / "keeper.sqlite") as store:
        store.install_schema()
        driver = SessionDriver(
            store=store,
            content_key_provider=LocalKeyProvider(b"k" * 32),
        )

        def runtime_factory(claim_store, agent, config, task, session_id):
            return replace(
                driver.runtime(
                    agent,
                    config,
                    task,
                    session_id=session_id,
                    store=claim_store,
                ),
                heartbeat_seconds=0.01,
            )

        driver.runtime_factory = runtime_factory

        def response(_request):
            model_entered.set()
            assert renew_entered.wait(5)
            return LLMResponse("done")

        try:
            driver.create("blocked", str(tmp_path))
            handle = driver.start(
                "blocked",
                Agent("agent", "Answer.", model="m"),
                RunConfig(model_provider=ScriptedModelProvider.new("scripted", "m", cast(Any, [response]))),
                "go",
            )
            assert renew_entered.wait(5)
            assert close_entered.wait(5)
            stalled_owner = store._one(
                "SELECT lease_owner, lease_epoch, drive_retry_at_ms FROM sk_session WHERE session_id='blocked'"
            )
            with pytest.raises(LeaseLost, match="heartbeat did not stop"):
                handle.result(timeout=8)

            retained_owner = store._one(
                "SELECT lease_owner, lease_epoch, drive_retry_at_ms FROM sk_session WHERE session_id='blocked'"
            )
            assert stalled_owner[0] is not None
            assert retained_owner == stalled_owner
        finally:
            release_renew.set()
            driver.close()


def test_public_worker_retries_transient_lease_loss_then_reports_bounded_exhaustion(tmp_path: Path, monkeypatch) -> None:
    from vv_agent.session.kernel import _Driver

    attempts: list[str] = []

    def lost(_driver):
        attempts.append("lease")
        raise LeaseLost("writer lost before durable progress")

    monkeypatch.setattr(_Driver, "step", lost)
    provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse("must not run")])
    with open_store(tmp_path / "retry.sqlite") as store:
        driver = SessionDriver(
            store=store,
            content_key_provider=LocalKeyProvider(b"r" * 32),
        )
        try:
            driver.create("retry", str(tmp_path))
            handle = driver.start(
                "retry",
                Agent("agent", "Answer.", model="m"),
                RunConfig(model_provider=provider),
                "go",
            )
            with pytest.raises(LeaseRetryExhausted) as caught:
                handle.result(timeout=8)
            assert caught.value.session_id == "retry"
            assert caught.value.attempts == len(attempts) == 5
            assert isinstance(caught.value.__cause__, LeaseLost)
            assert driver.store.read_state("retry")[0].turns == {}
            assert len(driver.store.peek_inbox("retry")) == 1
        finally:
            driver.close()


def test_takeover_progress_clears_terminal_lease_retry_for_existing_cursor(tmp_path: Path, monkeypatch) -> None:
    from vv_agent.session.kernel import _Driver

    original_step = _Driver.step

    def lost(_driver):
        raise LeaseLost("writer lost before durable progress")

    monkeypatch.setattr(_Driver, "step", lost)
    key_provider = LocalKeyProvider(b"t" * 32)
    with open_store(tmp_path / "takeover.sqlite") as store:
        store.install_schema()
        failing = SessionDriver(store=store, content_key_provider=key_provider)
        takeover = SessionDriver(store=store, content_key_provider=key_provider)
        try:
            agent = Agent("agent", "Answer.", model="m")
            failing.create("takeover", str(tmp_path))
            failed_cursor = failing.start(
                "takeover",
                agent,
                RunConfig(model_provider=ScriptedModelProvider.new("scripted", "m", [])),
                "go",
            )
            with pytest.raises(LeaseRetryExhausted):
                failed_cursor.result(timeout=8)
            # Keep A alive as an observer, but remove its drive authority so B
            # deterministically performs the takeover.
            failing._worker.support = replace(
                failing._worker.support,
                drive=None,
                project=None,
                projection_consumers=(),
            )

            monkeypatch.setattr(_Driver, "step", original_step)
            config = RunConfig(
                model_provider=ScriptedModelProvider.new(
                    "scripted",
                    "m",
                    [LLMResponse("takeover complete")],
                )
            )
            registration = takeover._static_registration(
                "takeover",
                agent,
                config,
                None,
                handler_version=takeover.default_handler_version,
            )
            takeover._runtimes["takeover"] = replace(registration, armed=True)
            winning_cursor = takeover.start("takeover", agent, config, None)

            assert winning_cursor.result(timeout=8).final_output == "takeover complete"
            assert failed_cursor.result(timeout=8).final_output == "takeover complete"
        finally:
            failing.close()
            takeover.close()
            key_provider.close()


def test_runner_start_returns_and_first_cursor_event_arrives_before_soft_budget() -> None:
    provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse("done")])
    started = monotonic()
    handle = Runner.start(
        Agent("early-cursor", "Answer.", model="m"),
        "go",
        run_config=RunConfig(model_provider=provider),
    )
    returned_after = monotonic() - started
    first_event = None
    first_event_after = None
    try:
        for event in handle.events():
            first_event = event
            first_event_after = monotonic() - started
            break
        assert returned_after < 0.5
        assert first_event is not None and first_event.type == "run_started"
        assert first_event_after is not None and first_event_after < 0.5
        assert handle.result(timeout=8).final_output == "done"
    finally:
        handle.kernel.close()


def test_events_use_bounded_after_seq_pages(monkeypatch) -> None:
    read_pages: list[tuple[int, int, int]] = []
    projection_pages: list[int] = []
    original_read_projection_page = CoreSQLiteStore.read_projection_page
    original_projection_context = CoreSQLiteStore.projection_context

    def read_projection_page(store, prefix, session_id, *, after_seq=0, through_seq=None, limit=256):
        if after_seq > 0:
            read_pages.append((after_seq, limit, through_seq or 0))
        return original_read_projection_page(
            store,
            prefix,
            session_id,
            after_seq=after_seq,
            through_seq=through_seq,
            limit=limit,
        )

    def projection_context(store, session_id, records, *, prefix=None):
        projection_pages.append(len(records))
        return original_projection_context(store, session_id, records, prefix=prefix)

    monkeypatch.setattr(CoreSQLiteStore, "read_projection_page", read_projection_page)
    monkeypatch.setattr(CoreSQLiteStore, "projection_context", projection_context)

    provider = ScriptedModelProvider.new("scripted", "m", [LLMResponse("done")])
    handle = Runner.start(
        Agent("cursor", "Answer.", model="m"),
        "go",
        run_config=RunConfig(model_provider=provider),
    )
    try:
        events = list(handle.events())
        result = handle.result(timeout=8)
        assert events[-1].type == "run_completed"
        event_ids = {event.event_id for event in events}
        assert {event.event_id for event in result.events}.issubset(event_ids)
        assert read_pages
        assert any(after_seq > 0 and limit == 256 for after_seq, limit, _ in read_pages)
        assert projection_pages and any(size <= 256 for size in projection_pages)
    finally:
        handle.kernel.close()


def test_events_advance_one_record_cursor_across_multiple_pages(monkeypatch) -> None:
    monkeypatch.setattr(surface_module, "_EVENT_PAGE_SIZE", 2)
    page_cursors: list[int] = []
    original_read = CoreSQLiteStore.read

    def read(store, session_id, *, after_seq=0, through_seq=None, limit=1024, expected_head=None):
        if after_seq > 0:
            page_cursors.append(after_seq)
        return original_read(
            store,
            session_id,
            after_seq=after_seq,
            through_seq=through_seq,
            limit=limit,
            expected_head=expected_head,
        )

    monkeypatch.setattr(CoreSQLiteStore, "read", read)
    handle = Runner.start(
        Agent("cursor", "Answer.", model="m"),
        "go",
        run_config=RunConfig(model_provider=ScriptedModelProvider.new("scripted", "m", [LLMResponse("done")])),
    )
    try:
        assert handle.result(timeout=8).final_output == "done"
        monkeypatch.setattr(
            handle.kernel,
            "_snapshot_status",
            lambda _store, _sid, _state, requested, **_kwargs: (requested, True, None),
        )
        events = list(handle.events())
        assert len(page_cursors) > 2
        assert max(page_cursors) > 2
        sequences = [event.metadata["session_seq"] for event in events]
        assert sequences == sorted(sequences)
        event_ids = [event.event_id for event in events]
        assert len(event_ids) == len(set(event_ids))
    finally:
        handle.kernel.close()


def test_periodic_scan_drives_follow_up_after_terminal_and_lost_wake(monkeypatch) -> None:
    driver = SessionDriver()
    try:
        provider = ScriptedModelProvider.new(
            "scripted",
            "m",
            [LLMResponse("first"), LLMResponse("second")],
        )
        driver.create("follow-up", ".")
        handle = driver.start(
            "follow-up",
            Agent("follow-up", "Answer.", model="m"),
            RunConfig(model_provider=provider),
            "one",
        )
        assert handle.result(timeout=8).final_output == "first"

        monkeypatch.setattr(driver._worker, "wake", lambda _session_id: None)
        handle.follow_up("two")
        cursor = driver.start(
            "follow-up",
            Agent("follow-up", "Answer.", model="m"),
            RunConfig(model_provider=provider),
            None,
        )
        assert cursor.result(timeout=8).final_output == "second"
    finally:
        driver.close()


@pytest.mark.parametrize("mode", ["run_sync", "stream_sync"])
def test_sync_runner_owners_wait_for_background_child_before_closing(
    tmp_path: Path,
    monkeypatch,
    mode: str,
) -> None:
    holder: dict[str, SessionDriver] = {}
    child_called = Event()
    real_driver = SessionDriver

    def runtime_factory(store, agent, config, task, session_id):
        runtime = holder["driver"].runtime(
            agent,
            config,
            task,
            session_id=session_id,
            store=store,
            handler_version="handler/v2",
        )

        def admit(plan):
            registration = holder["driver"]._runtime_registration(plan.session_id)
            return ChildSession(
                SessionSpec(
                    "child",
                    holder["driver"]._session_principal(plan.session_id).to_dict(),
                    cast(Any, str(tmp_path / "child")),
                    attributes={},
                    feature_level=registration.requirements.level,
                    bootstrap=(registration.requirements if registration.requirements.level == 2 else None),
                ),
                "background work",
                background=True,
            )

        runtime.children["background_child"] = admit
        return runtime

    def configured_driver(*_args, **_kwargs):
        driver = real_driver(
            runtime_factory=runtime_factory,
            worker_level=2,
            handler_versions=("handler/v2",),
            capabilities=("child:background",),
        )
        holder["driver"] = driver
        return driver

    monkeypatch.setattr(surface_module, "SessionDriver", configured_driver)

    def child_response(_request):
        child_called.set()
        return LLMResponse("child done")

    provider = ScriptedModelProvider.new(
        "scripted",
        "m",
        cast(
            Any,
            [
                LLMResponse("", [ToolCall("child-call", "background_child", {})]),
                LLMResponse("parent done"),
                child_response,
            ],
        ),
    )
    agent = Agent("parent", "Delegate.", model="m", tools=[background_child])
    config = RunConfig(model_provider=provider, workspace=tmp_path)
    if mode == "run_sync":
        result = Runner.run_sync(agent, "go", run_config=config)
        assert result.final_output == "parent done"
    else:
        events = list(Runner.stream_sync(agent, "go", run_config=config))
        assert events[-1].type == "run_completed"

    driver = holder["driver"]
    assert driver.path is not None
    database = Path(driver.path)
    assert child_called.is_set()
    assert not driver._worker.running
    assert not database.parent.exists()


@pytest.mark.parametrize("wait_kind", ["user", "approval"])
def test_sync_runner_retains_owner_for_waiting_background_child(
    tmp_path: Path,
    monkeypatch,
    wait_kind: str,
) -> None:
    holder: dict[str, SessionDriver] = {}
    real_driver = SessionDriver

    def runtime_factory(store, agent, config, task, session_id):
        runtime = holder["driver"].runtime(
            agent,
            config,
            task,
            session_id=session_id,
            store=store,
            handler_version="handler/v2",
        )

        def admit(plan):
            registration = holder["driver"]._runtime_registration(plan.session_id)
            return ChildSession(
                SessionSpec(
                    "child",
                    holder["driver"]._session_principal(plan.session_id).to_dict(),
                    cast(Any, str(tmp_path / "child")),
                    attributes={},
                    feature_level=registration.requirements.level,
                    bootstrap=(registration.requirements if registration.requirements.level == 2 else None),
                ),
                "background work",
                background=True,
            )

        runtime.children["background_child"] = admit
        return runtime

    def configured_driver(*_args, **_kwargs):
        driver = real_driver(
            runtime_factory=runtime_factory,
            worker_level=2,
            handler_versions=("handler/v2",),
            capabilities=("child:background",),
        )
        holder["driver"] = driver
        return driver

    monkeypatch.setattr(surface_module, "SessionDriver", configured_driver)
    provider = ScriptedModelProvider.new(
        "scripted",
        "m",
        [
            LLMResponse("", [ToolCall("child-call", "background_child", {})]),
            LLMResponse("parent done"),
            LLMResponse(
                "",
                [
                    ToolCall("effect", "approval_effect", {})
                    if wait_kind == "approval"
                    else ToolCall("ask", "ask_user", {"question": "child color?"})
                ],
            ),
        ],
    )
    agent = Agent("parent", "Delegate.", model="m", tools=[background_child, approval_effect])
    result = Runner.run_sync(
        agent,
        "go",
        run_config=RunConfig(model_provider=provider, workspace=tmp_path),
    )

    owner = result._session_driver
    assert owner is not None
    assert result.final_output == "parent done"
    assert owner._worker.running
    child_state, _, _ = owner._read_state("child")
    child_turn_id = next(reversed(child_state.turns))
    assert child_state.waits
    assert child_state.turns[child_turn_id].wait is None
    assert {wait["handle"]["kind"] for wait in child_state.waits.values()} == {wait_kind}
    assert any(wait["session_id"] == "child" for wait in result.metadata["session_waits"])

    resumed = Runner.resume("child", child_turn_id)
    assert resumed.status is AgentStatus.WAIT_USER
    assert resumed._session_driver is owner
    owner.close()


def test_public_runner_applies_llm_hooks_on_the_worker_path() -> None:
    from vv_agent.runtime.hooks import BaseRuntimeHook, BeforeLLMPatch

    hook_threads = []

    class Hook(BaseRuntimeHook):
        def before_llm(self, event):
            hook_threads.append("before")
            event.shared_state["hook"] = "before"
            return BeforeLLMPatch(messages=[*event.messages, Message("user", "hook input")])

        def after_llm(self, event):
            assert event.shared_state["hook"] == "before"
            hook_threads.append("after")
            return None

    def response(request):
        assert request.messages[-1].content == "hook input"
        return LLMResponse("hooked")

    result = Runner.run_sync(
        Agent("hook", "Be concise.", model="m", hooks=[Hook()]),
        "go",
        run_config=RunConfig(model_provider=ScriptedModelProvider.new("scripted", "m", cast(Any, [response]))),
    )
    assert result.final_output == "hooked"
    assert hook_threads == ["before", "after"]


def test_background_child_is_driven_by_worker_after_parent_handle_finishes(tmp_path: Path) -> None:
    holder: dict[str, SessionDriver] = {}

    def runtime_factory(store, agent, config, task, session_id):
        runtime = holder["driver"].runtime(
            agent,
            config,
            task,
            session_id=session_id,
            store=store,
            handler_version="handler/v2",
        )

        def admit(plan):
            registration = holder["driver"]._runtime_registration(plan.session_id)
            return ChildSession(
                SessionSpec(
                    "child",
                    holder["driver"]._session_principal(plan.session_id).to_dict(),
                    cast(Any, str(tmp_path / "child")),
                    attributes={},
                    feature_level=registration.requirements.level,
                    bootstrap=(registration.requirements if registration.requirements.level == 2 else None),
                ),
                "background work",
                background=True,
            )

        runtime.children["background_child"] = admit
        return runtime

    with open_store(tmp_path / "background.sqlite") as store:
        store.install_schema()
        driver = SessionDriver(
            store=store,
            content_key_provider=LocalKeyProvider(b"b" * 32),
            runtime_factory=runtime_factory,
            worker_level=2,
            handler_versions=("handler/v2",),
            capabilities=("child:background",),
        )
        holder["driver"] = driver
        provider = ScriptedModelProvider.new(
            "scripted",
            "m",
            [
                LLMResponse("", [ToolCall("child-call", "background_child", {})]),
                LLMResponse("parent done"),
                LLMResponse("child done"),
            ],
        )
        try:
            driver.create("parent", str(tmp_path))
            handle = driver.start(
                "parent",
                Agent("parent", "Delegate.", model="m", tools=[background_child]),
                RunConfig(model_provider=provider),
                "go",
                handler_version="handler/v2",
            )
            assert handle.result(timeout=8).final_output == "parent done"
            _wait_for(
                lambda: (
                    (child_state := driver._read_state("child")[0]).turns
                    and all(turn.ended for turn in child_state.turns.values())
                )
            )
            root, chain = driver._runtime_lineage(driver.store, "child")
            assert root.session_id == "parent"
            assert chain == ("child",)
            assert set(driver._runtimes) == {"parent"}
        finally:
            driver.close()


def test_status_wait_held_steer_does_not_starve_background_child(tmp_path: Path, monkeypatch) -> None:
    holder: dict[str, SessionDriver] = {}
    accepted = YieldProvider(declared=True, callback=True)
    real_driver = SessionDriver

    def configured_driver(*_args, **_kwargs):
        driver = real_driver(
            worker_level=2,
            handler_versions=("1",),
            capabilities=("child:configured",),
        )
        holder["driver"] = driver
        original_runtime = driver.runtime

        def runtime(*args, **kwargs):
            value = original_runtime(*args, **kwargs)
            value.providers["background_child"] = accepted
            return value

        monkeypatch.setattr(driver, "runtime", runtime)
        return driver

    monkeypatch.setattr(surface_module, "SessionDriver", configured_driver)

    def status_step(_request):
        child_id = next(sid for sid in holder["driver"].store.list_sessions() if "/child/" in sid)
        return LLMResponse(
            "",
            [
                ToolCall(
                    "status",
                    "sub_task_status",
                    {
                        "task_ids": [child_id],
                        "wait_for_completion": True,
                        "max_wait_seconds": 60,
                    },
                )
            ],
        )

    provider = routed(
        [
            LLMResponse(
                "",
                [
                    ToolCall(
                        "delegate",
                        "create_sub_task",
                        {
                            "agent_id": "worker",
                            "task_description": "work",
                            "wait_for_completion": False,
                        },
                    )
                ],
            ),
            status_step,
            LLMResponse("parent observed child"),
        ],
        [LLMResponse("", [ToolCall("wait", "background_child", {})]), LLMResponse("child done")],
    )
    agent = Agent(
        "parent",
        "Parent instructions.",
        model="parent",
        tools=[background_child],
        sub_agents={"worker": SubAgentConfig(model="m", description="Worker instructions.")},
    )
    try:
        handle = Runner.start(
            agent,
            "go",
            run_config=RunConfig(model=provider.default_model_ref(), model_provider=provider, workspace=tmp_path),
        )
        driver = holder["driver"]
        parent_sid = handle.session_id

        def parent_status():
            state = driver.store.read_state(parent_sid)[0]
            return state, next(
                (
                    wait
                    for wait in state.waits.values()
                    if wait["handle"].get("kind") == "provider" and wait["handle"].get("provider") == "sub_task_status"
                ),
                None,
            )

        _wait_for(lambda: parent_status()[1] is not None)
        parent_state, _ = parent_status()
        child_id = next(sid for sid in driver.store.list_sessions() if "/child/" in sid)

        def steer(sid: str, state) -> None:
            assert state.active_turn_id is not None
            turn = state.turns[state.active_turn_id]
            driver.push(
                sid,
                InboxItem(
                    f"{sid}-steer",
                    "steer",
                    {"content": "continue"},
                    state.active_turn_id,
                    turn.start._payload["generation"],
                ),
            )

        steer(parent_sid, parent_state)

        def child_wait_state():
            state = driver.store.read_state(child_id)[0]
            if any(
                wait["handle"].get("kind") == "provider" and wait["handle"].get("provider") == "background_child"
                for wait in state.waits.values()
            ):
                return state
            return None

        _wait_for(lambda: child_wait_state() is not None)
        child_state = child_wait_state()
        assert child_state is not None
        steer(child_id, child_state)
        result = handle.result(timeout=10)

        assert result.status is AgentStatus.COMPLETED
        assert result.final_output == "parent observed child"
        assert driver.store.read_state(child_id)[0].turns
        assert all(turn.ended for turn in driver.store.read_state(child_id)[0].turns.values())
        assert accepted.cancels == 0
    finally:
        driver.close()


def test_yield_on_steer_is_resumed_by_the_worker(tmp_path: Path) -> None:
    holder: dict[str, SessionDriver] = {}
    accepted = YieldProvider(declared=True)
    requests = []

    def runtime_factory(store, agent, config, task, session_id):
        runtime = holder["driver"].runtime(
            agent,
            config,
            task,
            session_id=session_id,
            store=store,
            handler_version="handler/v2",
        )
        runtime.providers["background_child"] = accepted
        return runtime

    def finish(request):
        requests.append(request)
        return LLMResponse("after steer")

    with open_store(tmp_path / "yield.sqlite") as store:
        store.install_schema()
        driver = SessionDriver(
            store=store,
            content_key_provider=LocalKeyProvider(b"y" * 32),
            runtime_factory=runtime_factory,
            worker_level=2,
            handler_versions=("handler/v2",),
            capabilities=("provider:yield",),
        )
        holder["driver"] = driver
        provider = ScriptedModelProvider.new(
            "scripted",
            "m",
            cast(Any, [LLMResponse("", [ToolCall("wait", "background_child", {})]), finish]),
        )
        try:
            driver.create("yield", str(tmp_path))
            handle = driver.start(
                "yield",
                Agent("yield", "Call the provider.", model="m", tools=[background_child]),
                RunConfig(model_provider=provider),
                "go",
                handler_version="handler/v2",
            )
            assert handle.result(timeout=8).status is AgentStatus.WAIT_USER
            parked = next(row.record for row in store.read("yield", limit=100).records if row.record.kind == "op_parked")
            assert parked.payload.get("yield_on_steer") is True

            handle.steer("change course")
            result = handle.result(timeout=8)
            assert result.status is AgentStatus.COMPLETED
            assert result.final_output == "after steer"
            assert accepted.cancels == 0
            assert requests and any("Wait interrupted by new input" in str(message.content) for message in requests[0].messages)
        finally:
            driver.close()


@pytest.mark.parametrize("dispose", ["gc", "close"])
def test_unobserved_waiting_stream_disposal_releases_owner_and_temporary_store(dispose: str) -> None:
    provider = ScriptedModelProvider.new(
        "scripted",
        "m",
        [LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Which color?"})])],
    )
    stream = cast(
        Any,
        Runner.stream_sync(
            Agent("waiting", "Ask once.", model="m"),
            "choose",
            run_config=RunConfig(model_provider=provider),
        ),
    )
    owner = stream._handle.kernel
    assert owner.path is not None
    database = Path(owner.path)
    worker_thread = owner._worker._thread
    assert database.exists()
    assert worker_thread is not None and worker_thread.is_alive()

    assert list(stream)
    # Exhaustion materializes the WAIT_USER result, but without reading the
    # public property the owner remains recoverable while the stream is alive.
    assert stream._result is not None
    assert not stream._result_observed
    assert owner._worker.running

    if dispose == "gc":
        del stream
        collect()
    else:
        stream.close()
    _wait_for(lambda: not owner._worker.running and not database.parent.exists())


def test_dropped_waiting_stream_retains_resume_owner(tmp_path: Path) -> None:
    provider = ScriptedModelProvider.new(
        "scripted",
        "m",
        [LLMResponse("", [ToolCall("ask", "ask_user", {"question": "Which color?"})])],
    )
    stream = cast(
        Any,
        Runner.stream_sync(
            Agent("waiting", "Ask once.", model="m"),
            "choose",
            run_config=RunConfig(model_provider=provider, workspace=tmp_path),
        ),
    )
    handle = stream._handle
    handle.result(timeout=8)
    list(stream)
    waiting = stream.result
    assert handle.kernel is waiting._session_driver
    assert waiting.status is AgentStatus.WAIT_USER
    owner = waiting._session_driver
    assert owner is not None
    assert owner.path is not None
    database = Path(owner.path)
    session_id = waiting.raw_result.session_id
    turn_id = waiting.raw_result.turn_id
    assert session_id is not None and turn_id is not None

    del stream
    collect()
    assert database.exists()
    assert not owner._is_closed

    resumed = Runner.resume(session_id, turn_id)
    assert resumed.status is AgentStatus.WAIT_USER
    assert resumed._session_driver is owner
    owner.close()


def test_observer_delivers_committed_events_while_drive_model_is_slow() -> None:
    calls = 0
    model_entered = Event()
    release_model = Event()
    observer_threads: list[str] = []
    stream_events = []

    class EventSink:
        def __init__(self) -> None:
            self.events = []

        def append(self, event) -> None:
            self.events.append(event)

    @function_tool
    def echo(value: str) -> str:
        return value

    def response(_request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return LLMResponse("", [ToolCall("echo-call", "echo", {"value": "committed"})])
        model_entered.set()
        assert release_model.wait(5)
        return LLMResponse("done")

    def observe(event) -> None:
        observer_threads.append(current_thread().name)
        stream_events.append(event)

    sink = EventSink()
    driver = SessionDriver()
    try:
        provider = ScriptedModelProvider.from_callback("scripted", "m", response)
        driver.create("slow-stream", ".")
        handle = driver.start(
            "slow-stream",
            Agent("slow-stream", "Answer.", model="m", tools=[echo]),
            RunConfig(
                model_provider=provider,
                stream=observe,
                event_store=sink,
                event_store_fail_closed=True,
            ),
            "go",
        )
        assert model_entered.wait(5)
        _wait_for(lambda: bool(stream_events), timeout=5)
        committed_ids = [event.event_id for event in stream_events]
        assert committed_ids
        assert all(name.startswith("session-local-worker") for name in observer_threads)
        assert [event.event_id for event in sink.events] == committed_ids
        release_model.set()
        assert handle.result(timeout=8).final_output == "done"
        assert [event.event_id for event in sink.events] == [event.event_id for event in stream_events]
        assert len(sink.events) == len({event.event_id for event in sink.events})
    finally:
        release_model.set()
        driver.close()
