"""Runtime factories rebuild bindings for each claimed drive."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from time import monotonic, sleep
from typing import Any, cast

import pytest

from vv_agent import Agent, RunConfig
from vv_agent.model import ScriptedModelProvider
from vv_agent.session.children import ChildSession, ChildTool
from vv_agent.session.kernel import drive
from vv_agent.session.projection import SessionHydration, hydrate_records
from vv_agent.session.providers import Provider
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.runtime import Runtime
from vv_agent.session.store import Conflict, SessionStore
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.timing import SystemClock, ThreadInvoker, ThreadLeaseKeeper
from vv_agent.tools.function import function_tool
from vv_agent.types import AgentStatus, LLMResponse, SubAgentConfig, ToolCall


class _WorkerDeath(Exception):
    pass


def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def _hydrated_records(driver: SessionDriver, session_id: str):
    _, source_records, _ = driver.store.read_state(session_id)
    return hydrate_records(
        source_records,
        SessionHydration.from_store(
            driver.store,
            tenant_id=driver.tenant_id,
            session_id=session_id,
        ),
    )


def test_factory_rebuilds_after_claim_and_fences_the_old_definition() -> None:
    driver = SessionDriver()
    session_id = "factory-resume"
    provider = ScriptedModelProvider.new("scripted", "model", [LLMResponse("must not run")])
    config = RunConfig(model_provider=provider)
    calls: list[tuple[int, Runtime]] = []

    def stop_after_turn_started(point: str, record: Any) -> None:
        if point == "after_commit" and record is not None and record.kind == "turn_started":
            raise _WorkerDeath("leave the frozen turn for another claim")

    def make_runtime(selected_session_id: str) -> Runtime:
        invocation = len(calls)
        _, epoch, remaining_ms = driver.store.lease_status(selected_session_id)
        assert epoch > 0
        assert remaining_ms > 0
        runtime = driver.runtime(
            Agent("first" if invocation == 0 else "second", "Answer.", model="model"),
            config,
            session_id=selected_session_id,
            store=driver.store,
        )
        if invocation == 0:
            runtime.hook = stop_after_turn_started
        calls.append((epoch, runtime))
        return runtime

    try:
        driver.create(session_id, ".")
        driver.push(session_id, InboxItem("go", "user", {"content": "go"}))

        with pytest.raises(_WorkerDeath):
            drive(driver.store, session_id, runtime=make_runtime, failure_backoff_ms=1)

        _wait_until(lambda: driver.store.is_runnable(session_id), timeout=1)
        drive(driver.store, session_id, runtime=make_runtime, failure_backoff_ms=1)

        epochs = [epoch for epoch, _ in calls]
        assert len(epochs) == 2 and epochs[0] < epochs[1]
        assert calls[0][1] is not calls[1][1]
        assert calls[0][1]._definitions is not calls[1][1]._definitions

        state, _, _ = driver.store.read_state(session_id)
        assert not any(operation.kind == "model" for operation in state.operations.values())
        ended = next(record for record in _hydrated_records(driver, session_id) if record.record.kind == "turn_ended")
        assert ended.record._payload["reason"] == "handler_schema_or_capability_mismatch"
    finally:
        driver.close()


def test_worker_factory_gets_a_fresh_store_and_runtime_for_each_claim() -> None:
    holder: dict[str, SessionDriver] = {}
    calls: list[tuple[int, int, int, SessionStore, Runtime]] = []

    def runtime_factory(
        store: SessionStore,
        agent: Agent,
        config: RunConfig,
        task: Any,
        session_id: str,
    ) -> Runtime:
        _, epoch, remaining_ms = store.lease_status(session_id)
        assert epoch > 0
        assert remaining_ms > 0
        runtime = holder["driver"].runtime(agent, config, task, session_id=session_id, store=store)
        calls.append((epoch, id(store), id(runtime), store, runtime))
        return runtime

    driver = SessionDriver(runtime_factory=runtime_factory)
    holder["driver"] = driver
    session_id = "factory-claims"
    provider = ScriptedModelProvider.new("scripted", "model", [LLMResponse("first"), LLMResponse("second")])

    try:
        driver.create(session_id, ".")
        handle = driver.start(
            session_id,
            Agent("assistant", "Answer.", model="model"),
            RunConfig(model_provider=provider),
            "first",
            autostart=True,
        )
        assert handle.result(timeout=8).status == AgentStatus.COMPLETED

        handle.follow_up("second")
        _wait_until(lambda: sum(stored.record.kind == "turn_ended" for stored in driver.store.read(session_id).records) == 2)
        assert handle.result(timeout=8).status == AgentStatus.COMPLETED

        assert len(calls) == 2
        assert [epoch for epoch, *_ in calls] == [1, 2]
        assert calls[0][1] != calls[1][1]
        assert calls[0][2] != calls[1][2]
        assert calls[0][4]._definitions is not calls[1][4]._definitions
        assert all(store is not driver.store for _, _, _, store, _ in calls)
    finally:
        driver.close()


def test_worker_factory_rejects_runtime_reused_from_previous_claim() -> None:
    holder: dict[str, SessionDriver] = {}
    calls: list[tuple[int, SessionStore, Runtime]] = []
    model_calls = 0

    def model(_request) -> LLMResponse:
        nonlocal model_calls
        model_calls += 1
        return LLMResponse("first")

    def runtime_factory(
        store: SessionStore,
        agent: Agent,
        config: RunConfig,
        task: Any,
        session_id: str,
    ) -> Runtime:
        _, epoch, remaining_ms = store.lease_status(session_id)
        assert epoch > 0
        assert remaining_ms > 0
        runtime = (
            calls[0][2]
            if calls
            else holder["driver"].runtime(
                agent,
                config,
                task,
                session_id=session_id,
                store=store,
            )
        )
        calls.append((epoch, store, runtime))
        return runtime

    driver = SessionDriver(runtime_factory=runtime_factory)
    holder["driver"] = driver
    provider = ScriptedModelProvider.from_callback("scripted", "model", model)

    try:
        driver.create("factory-reuse", ".")
        handle = driver.start(
            "factory-reuse",
            Agent("assistant", "Answer.", model="model"),
            RunConfig(model_provider=provider),
            "first",
            autostart=True,
        )
        assert handle.result(timeout=8).status == AgentStatus.COMPLETED

        handle.follow_up("second")
        with pytest.raises(Conflict, match="fresh Runtime for every claim"):
            handle.result(timeout=8)

        # The failure is deterministic. A later worker scan must not invoke the
        # factory or the model again while the same invalid definition remains.
        sleep(0.4)
        assert model_calls == 1
        assert len(calls) == 2
        assert calls[0][0] < calls[1][0]
        assert calls[0][1] is not calls[1][1]
        assert calls[0][2] is calls[1][2]
    finally:
        driver.close()


def test_worker_factory_rebinds_lifecycle_fields_when_store_is_omitted() -> None:
    holder: dict[str, SessionDriver] = {}
    calls: list[tuple[SessionStore, Runtime, object, object]] = []

    def runtime_factory(
        store: SessionStore,
        agent: Agent,
        config: RunConfig,
        task: Any,
        session_id: str,
    ) -> Runtime:
        # Deliberately omit ``store=store`` here. SessionDriver still owns the
        # heartbeat, content reader and hook bindings for this exact claim.
        runtime = holder["driver"].runtime(agent, config, task, session_id=session_id)
        calls.append((store, runtime, runtime.content_reader, runtime.hook))
        return runtime

    driver = SessionDriver(runtime_factory=runtime_factory)
    holder["driver"] = driver
    provider = ScriptedModelProvider.new("scripted", "model", [LLMResponse("done")])

    try:
        driver.create("factory-omitted-store", ".")
        handle = driver.start(
            "factory-omitted-store",
            Agent("assistant", "Answer.", model="model"),
            RunConfig(model_provider=provider),
            "go",
            autostart=True,
        )
        assert handle.result(timeout=8).status == AgentStatus.COMPLETED

        claim_store, runtime, factory_reader, factory_hook = calls[0]
        assert claim_store is not driver.store
        assert getattr(runtime.heartbeat_store, "__self__", None) is claim_store
        assert runtime.content_reader is not factory_reader
        assert runtime.hook is not factory_hook
    finally:
        driver.close()


def test_child_claim_uses_root_only_factory_and_rebinds_child_runtime(tmp_path: Path) -> None:
    holder: dict[str, SessionDriver] = {}
    factory_calls: list[tuple[str, str, SessionStore, Runtime]] = []
    claimed_runtimes: list[tuple[str, SessionStore, Runtime]] = []
    inherited_provider = cast(Provider, object())
    inherited_host_binding = "host-claim-binding"
    inherited_clock = SystemClock()
    inherited_lease_keeper = ThreadLeaseKeeper()
    inherited_invoker = ThreadInvoker()

    def admit_child(plan: Any) -> ChildSession:
        return ChildSession(
            SessionSpec(
                "child",
                holder["driver"]._session_principal(plan.session_id).to_dict(),
                cast(Any, str(tmp_path / "child")),
                attributes={},
            ),
            "child work",
        )

    inherited_child = ChildTool(admit_child)

    def runtime_factory(
        claim_store: SessionStore,
        agent: Agent,
        config: RunConfig,
        task: Any,
        session_id: str,
    ) -> Runtime:
        runtime = holder["driver"].runtime(
            agent,
            config,
            task,
            session_id=session_id,
            store=claim_store,
            handler_version="handler/v2",
        )
        if agent.name == "parent":
            runtime.children["inherited_child"] = inherited_child
            runtime.providers["inherited_provider"] = inherited_provider
            runtime.host_bindings["claim_host"] = inherited_host_binding
            runtime.clock = inherited_clock
            runtime.lease_keeper = inherited_lease_keeper
            runtime.invoker = inherited_invoker
            runtime.ttl_ms = 47_000
        factory_calls.append((session_id, agent.name, claim_store, runtime))
        return runtime

    driver = SessionDriver(
        runtime_factory=runtime_factory,
        worker_level=2,
        handler_versions=("handler/v2",),
    )
    holder["driver"] = driver
    original_build = driver._build_runtime_for_claim

    def build_runtime(claim_store: SessionStore, session_id: str) -> Runtime:
        runtime = original_build(claim_store, session_id)
        claimed_runtimes.append((session_id, claim_store, runtime))
        return runtime

    cast(Any, driver)._build_runtime_for_claim = build_runtime
    provider = ScriptedModelProvider.new(
        "scripted",
        "m",
        [
            LLMResponse(
                "",
                [ToolCall("child-call", "create_sub_task", {"agent_id": "worker", "task_description": "child work"})],
            ),
            LLMResponse("child done"),
            LLMResponse("parent done"),
        ],
    )

    try:
        driver.create("parent", str(tmp_path), {"seed": {"messages": [], "shared_state": {}}})
        handle = driver.start(
            "parent",
            Agent(
                "parent",
                "Use the child tool.",
                model="m",
                sub_agents={
                    "worker": SubAgentConfig(
                        model="m",
                        description="Worker instructions.",
                        exclude_tools=["forbidden"],
                    )
                },
            ),
            RunConfig(model_provider=provider, workspace=tmp_path),
            "go",
            handler_version="handler/v2",
        )
        assert handle.result(timeout=8).final_output == "parent done"

        child_session_id = next(session_id for session_id in driver.store.list_sessions() if session_id != "parent")
        child_claim_runtime = next(item for item in claimed_runtimes if item[0] == child_session_id)
        child_claim_store = child_claim_runtime[1]
        child_runtime = child_claim_runtime[2]
        child_claim_factory_calls = [call for call in factory_calls if call[2] is child_claim_store]
        child_created = driver._created_payload(driver.store, child_session_id)

        assert [(session_id, agent_name) for session_id, agent_name, _, _ in child_claim_factory_calls] == [("parent", "parent")]
        child_root = child_claim_factory_calls[0][3]
        assert child_runtime.agent.name == "worker"
        assert child_runtime.config.workspace == Path(child_created["workspace"])
        assert child_runtime.frozen_task is not None
        assert "forbidden" in child_runtime.frozen_task.metadata["_vv_agent_disallowed_tools"]
        assert child_runtime.children is child_root.children
        assert child_runtime.host_bindings is child_root.host_bindings
        assert child_runtime.children["inherited_child"] is inherited_child
        assert child_runtime.providers["inherited_provider"] is inherited_provider
        assert child_runtime.host_bindings["claim_host"] == inherited_host_binding
        assert child_runtime.clock is inherited_clock
        assert child_runtime.lease_keeper is inherited_lease_keeper
        assert child_runtime.invoker is inherited_invoker
        assert child_runtime.ttl_ms == 47_000
        assert getattr(child_runtime.heartbeat_store, "__self__", None) is child_claim_store
        assert cast(Any, child_runtime.content_reader).reader.binding == (driver.tenant_id, child_session_id)

        first_parent_claim = next(item for item in claimed_runtimes if item[0] == "parent")
        assert claimed_runtimes.index(first_parent_claim) < claimed_runtimes.index(child_claim_runtime)
        assert first_parent_claim[1] is not child_claim_store
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            cast(Any, first_parent_claim[1]).connection.execute("SELECT 1")
    finally:
        driver.close()


def test_custom_child_claim_rebuilds_with_child_identity_and_workspace(tmp_path: Path) -> None:
    holder: dict[str, SessionDriver] = {}
    factory_calls: list[tuple[str, str, Path | None, SessionStore, Runtime]] = []
    claimed_runtimes: list[tuple[str, SessionStore, Runtime]] = []
    inherited_host_binding = "custom-child-host-binding"

    @function_tool
    def custom_child() -> str:
        raise AssertionError("custom child must use the session kernel")

    def admit_child(plan: Any) -> ChildSession:
        registration = holder["driver"]._runtime_registration(plan.session_id)
        return ChildSession(
            SessionSpec(
                "custom-child",
                holder["driver"]._session_principal(plan.session_id).to_dict(),
                cast(Any, str(tmp_path / "custom-child")),
                attributes={},
                feature_level=registration.requirements.level,
                bootstrap=(registration.requirements if registration.requirements.level == 2 else None),
            ),
            "custom child work",
        )

    def runtime_factory(
        claim_store: SessionStore,
        agent: Agent,
        config: RunConfig,
        task: Any,
        session_id: str,
    ) -> Runtime:
        runtime = holder["driver"].runtime(
            agent,
            config,
            task,
            session_id=session_id,
            store=claim_store,
            handler_version="handler/v2",
        )
        if agent.name == "parent":
            runtime.children["custom_child"] = admit_child
            runtime.host_bindings["custom_host"] = inherited_host_binding
        factory_calls.append(
            (
                session_id,
                agent.name,
                Path(config.workspace) if config.workspace is not None else None,
                claim_store,
                runtime,
            )
        )
        return runtime

    driver = SessionDriver(
        runtime_factory=runtime_factory,
        worker_level=2,
        handler_versions=("handler/v2",),
    )
    holder["driver"] = driver
    original_build = driver._build_runtime_for_claim

    def build_runtime(claim_store: SessionStore, session_id: str) -> Runtime:
        runtime = original_build(claim_store, session_id)
        claimed_runtimes.append((session_id, claim_store, runtime))
        return runtime

    cast(Any, driver)._build_runtime_for_claim = build_runtime
    provider = ScriptedModelProvider.new(
        "scripted",
        "m",
        [
            LLMResponse("", [ToolCall("child-call", "custom_child", {})]),
            LLMResponse("custom child done"),
            LLMResponse("parent done"),
        ],
    )

    try:
        driver.create("parent", str(tmp_path))
        handle = driver.start(
            "parent",
            Agent("parent", "Use the custom child.", model="m", tools=[custom_child]),
            RunConfig(model_provider=provider, workspace=tmp_path),
            "go",
            handler_version="handler/v2",
        )
        assert handle.result(timeout=8).final_output == "parent done"

        child_claim_runtime = next(item for item in claimed_runtimes if item[0] == "custom-child")
        child_claim_store = child_claim_runtime[1]
        child_runtime = child_claim_runtime[2]
        child_calls = [call for call in factory_calls if call[3] is child_claim_store]

        assert [(session_id, agent_name, workspace) for session_id, agent_name, workspace, _, _ in child_calls] == [
            ("parent", "parent", tmp_path),
            ("custom-child", "parent", tmp_path / "custom-child"),
        ]
        assert child_runtime.agent.name == "parent"
        assert child_runtime.config.workspace is not None
        assert Path(child_runtime.config.workspace) == tmp_path / "custom-child"
        assert child_runtime.host_bindings["custom_host"] == inherited_host_binding
        assert getattr(child_runtime.heartbeat_store, "__self__", None) is child_claim_store
        assert cast(Any, child_runtime.content_reader).reader.binding == (driver.tenant_id, "custom-child")
    finally:
        driver.close()
