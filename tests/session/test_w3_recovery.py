"""W3 unknown recovery stays pinned to the submitted command identity."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent import Agent
from vv_agent.kernel._boundary_hydration import SessionHydration, hydrate_session_records
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._store import Lease
from vv_agent.session import _recovery_phase, kernel
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.tool_results import prepare_tool_resources
from vv_agent.tools import ToolRegistry, ToolSpec, build_default_registry
from vv_agent.tools.metadata import ToolIdempotency, ToolMetadata
from vv_agent.types import LLMResponse, ToolCall

from .conftest import open_store
from .controlled import ControlledProvider
from .test_recovery_matrix import effect_tool
from .test_recovery_matrix import runtime as ordinary_runtime
from .test_recovery_matrix import start as start_ordinary
from .test_w3_command_producer import (
    MASTER_KEY,
    SESSION_ID,
    TENANT_ID,
    HostCommandProvider,
    _binding,
    _config,
    _create_level_three_session,
    _not_called,
)


class WorkerLost(BaseException):
    pass


class LoseSubmittedReceipt:
    def __init__(self, provider: HostCommandProvider) -> None:
        self.provider = provider
        self.lost = False

    def invoke(self, callback, *, timeout, token, cancellation_grace, clock):
        del timeout, token, cancellation_grace, clock
        outcome = callback()
        if self.provider.probe.submitted_plans and not self.lost:
            self.lost = True
            raise WorkerLost("worker lost after provider submission")
        return outcome


def _idempotent_agent() -> Agent:
    registry = ToolRegistry()
    registry.register_schema("bash", build_default_registry().get_schema("bash"))
    registry.register(
        ToolSpec(
            name="bash",
            handler=_not_called,
            tool_metadata=ToolMetadata(idempotency=ToolIdempotency.SUPPORTED),
        )
    )
    registry._add_planner_extra_tool_name("bash")
    return Agent(
        "w3-command",
        "Run the bound command.",
        model="scripted",
        tools=[registry.get_executor("bash")],
        metadata={},
    )


def _records(store):
    rows = list(store.read(SESSION_ID, limit=200).records)
    hydration = SessionHydration.from_store(store, tenant_id=TENANT_ID, session_id=SESSION_ID)
    return rows, hydrate_session_records(tuple(rows), hydration)


def _level_three_lookup_item(plan, turn, *, input_id: str = "provider-lookup-running") -> InboxItem:
    identity = plan.payload["execution_identity"]
    provider_id = plan.payload["provider_binding"]
    binding = {
        "tenant_id": identity["tenant_id"],
        "session_id": plan.session_id,
        "operation_id": plan.operation_id,
        "attempt": plan.attempt,
        "request_digest": plan.payload["request_digest"],
        "idempotency_key": plan.payload["idempotency_key"],
        "execution_id": identity["execution_id"],
        "command_digest": plan.payload["command_digest"],
        "environment_id": identity["environment_id"],
        "generation": turn.payload["environment"]["generation"],
        "provider_id": provider_id,
    }
    handle = {
        "kind": "provider",
        "provider": provider_id,
        "job_id": "job-running-1",
        "operation_id": binding["operation_id"],
        "attempt": binding["attempt"],
        "request_digest": binding["request_digest"],
        "execution_id": binding["execution_id"],
        "command_digest": binding["command_digest"],
        "environment_id": binding["environment_id"],
        "generation": binding["generation"],
        "evidence": ["provider-registry:running"],
        "query_ref": None,
        "cancel_ref": None,
    }
    return InboxItem(
        input_id,
        "provider_evidence",
        {
            "operation_id": binding["operation_id"],
            "attempt": binding["attempt"],
            "request_digest": binding["request_digest"],
            "provider_lookup": {
                "schema_version": "vv-agent.provider-lookup.v1",
                "status": "running",
                "binding": binding,
                "outcome": {"kind": "Accepted", "handle": handle, "yield_on_steer": False},
                "evidence": ["provider-registry:running"],
            },
        },
        target_turn_id=plan.turn_id,
        generation=turn.payload["generation"],
        feature_level=3,
    )


def _push_lookup(sessions: SessionDriver, store, item: InboxItem) -> None:
    actor = {
        "tenant_id": TENANT_ID,
        "subject_id": "bash",
        "subject_kind": "service",
        "attributes": {},
    }
    with store.atomic() as tx:
        sessions._push_in_transaction(tx, SESSION_ID, item, actor=actor, actor_subject_id="provider")


def _direct_driver(store, sessions: SessionDriver, runtime) -> tuple[CoreDriver, Lease]:
    lease = store.acquire(SESSION_ID, owner="w3-recovery", ttl_ms=15_000)
    assert lease is not None
    scope = kernel._Scope(lease, runtime, now_ms=store._now())
    scope.poll(store)
    driver = CoreDriver(
        store,
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
    return driver, lease


def _run_core_step(driver: CoreDriver) -> bool:
    from vv_agent.kernel import _driver_recovery

    host = _recovery_phase._Host(driver)
    composed = host.steps(_driver_recovery._compose(driver, _driver_recovery.step(driver, one_turn=False)))
    return kernel._Driver._run_outcome_steps(
        cast(Any, driver),
        cast(Any, composed),
        prepare_resources=prepare_tool_resources,
    )


@contextmanager
def _lost_w3_submission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "w3-lookup.sqlite3"
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "w3-lookup-keys.sqlite3")
    provider = HostCommandProvider()
    agent = _idempotent_agent()
    config = _config(provider)
    _create_level_three_session(path, key_provider, agent, config)
    monkeypatch.setattr("vv_agent.kernel._driver_planning.validate_command_submit", lambda *args: None)
    invoker = LoseSubmittedReceipt(provider)
    holder: dict[str, SessionDriver] = {}

    with open_store(path) as store:
        store.configure_reader(level=3)

        def runtime_factory(drive_store, run_agent, run_config, task, session_id):
            runtime = holder["sessions"].runtime(
                run_agent,
                run_config,
                task,
                session_id=session_id,
                store=drive_store,
                handler_version="1",
            )
            runtime.trusted_environment_binding = _binding()
            runtime.providers["bash"] = provider
            runtime.invoker = invoker
            return runtime

        sessions = SessionDriver(
            store=store,
            content_key_provider=key_provider,
            runtime_factory=runtime_factory,
            worker_level=3,
        )
        holder["sessions"] = sessions
        try:
            handle = sessions.start(SESSION_ID, agent, config, "run", autostart=False)
            handle._pending_submission()
            handle._pending_submission = None
            handle._runtime_reservation = None
            with sessions._open_drive_store() as drive_store:
                runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
                with pytest.raises(WorkerLost):
                    drive(drive_store, SESSION_ID, runtime=runtime)
            yield path, store, sessions, provider
        finally:
            sessions.close()


def test_w3_worker_loss_unknown_never_retries_or_resubmits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "w3-recovery.sqlite3"
    key_provider = LocalKeyProvider(
        MASTER_KEY,
        state_path=tmp_path / "w3-keys.sqlite3",
    )
    provider = HostCommandProvider()
    agent = _idempotent_agent()
    config = _config(provider)
    _create_level_three_session(path, key_provider, agent, config)
    # Recovery policy is independent of command-submit digest validation.
    monkeypatch.setattr("vv_agent.kernel._driver_planning.validate_command_submit", lambda *args: None)
    invoker = LoseSubmittedReceipt(provider)
    holder: dict[str, SessionDriver] = {}

    with open_store(path) as store:
        store.configure_reader(level=3)

        def runtime_factory(drive_store, run_agent, run_config, task, session_id):
            runtime = holder["sessions"].runtime(
                run_agent,
                run_config,
                task,
                session_id=session_id,
                store=drive_store,
                handler_version="1",
            )
            runtime.trusted_environment_binding = _binding()
            runtime.providers["bash"] = provider
            runtime.invoker = invoker
            return runtime

        sessions = SessionDriver(
            store=store,
            content_key_provider=key_provider,
            runtime_factory=runtime_factory,
            worker_level=3,
        )
        holder["sessions"] = sessions
        try:
            handle = sessions.start(SESSION_ID, agent, config, "run", autostart=False)
            handle._pending_submission()
            handle._pending_submission = None
            handle._runtime_reservation = None

            with sessions._open_drive_store() as drive_store:
                runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
                with pytest.raises(WorkerLost):
                    drive(drive_store, SESSION_ID, runtime=runtime)

            with sessions._open_drive_store() as drive_store:
                before, hydrated_before = _records(drive_store)
                before_tool = [row.record for row in before if "/tool/" in str(row.record.operation_id)]
                tool_plan = next(
                    row.record
                    for row in hydrated_before
                    if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
                )
                assert tool_plan.feature_level == 3
                assert tool_plan.payload["tool"]["idempotency"] == "supported"
                assert [record.kind for record in before_tool].count("op_planned") == 1
                assert [record.kind for record in before_tool].count("op_started") == 1
                assert not any(
                    row.record.kind in {"op_unknown", "op_planned"}
                    and "/tool/" in str(row.record.operation_id)
                    and row.record.attempt == 2
                    for row in hydrated_before
                )

                runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
                drive(drive_store, SESSION_ID, runtime=runtime)

                rows, hydrated = _records(drive_store)
                unknown = next(row.record for row in hydrated if row.record.kind == "op_unknown")
                tool_records = [row.record for row in rows if "/tool/" in str(row.record.operation_id)]
                assert "/tool/" in str(unknown.operation_id)
                assert unknown.payload["retry"] == "stop"
                assert unknown.payload["retry_at_ms"] is None
                assert unknown.payload["observation"]["retryable"] is False
                assert len([row for row in tool_records if row.kind == "op_planned"]) == 1
                assert len([row for row in tool_records if row.kind == "op_started"]) == 1
                assert len(provider.probe.submitted_plans) == 1
                assert len(provider.probe.submitted_calls) == 1
        finally:
            sessions.close()


def test_level_three_provider_lookup_is_preserved_and_fails_closed_without_resubmission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with (
        _lost_w3_submission(tmp_path, monkeypatch) as (path, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, hydrated_before = _records(drive_store)
        plan = next(
            row.record for row in hydrated_before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
        )
        turn = next(row.record for row in hydrated_before if row.record.kind == "turn_started")

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        lookup = _level_three_lookup_item(plan, turn)
        _push_lookup(sessions, drive_store, lookup)
        stored = drive_store.peek_inbox(SESSION_ID)[0].item
        assert set(stored.payload) == {"operation_id", "attempt", "request_digest", "provider_lookup"}
        assert stored.payload["provider_lookup"]["content_id"]
        assert b"job-running-1" not in path.read_bytes()

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        applied = next(
            row.record
            for row in hydrated
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == lookup.input_id
        )
        unknown = next(row.record for row in hydrated if row.record.kind == "op_unknown")
        assert applied.payload["disposition"] == "rejected"
        assert applied.payload["reason"] == "provider_lookup_recovery_unavailable"
        assert applied.payload["target_operation_id"] == plan.operation_id
        assert applied.payload["input"]["payload"]["provider_lookup"]["status"] == "running"
        assert unknown.payload["retry"] == "stop"
        assert len([row for row in rows if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)]) == 1
        assert len(provider.probe.submitted_plans) == 1
        assert len(provider.probe.submitted_calls) == 1
        assert b"job-running-1" not in path.read_bytes()


def test_w3_recovery_retry_plan_preserves_feature_level_and_execution_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with (
        _lost_w3_submission(tmp_path, monkeypatch) as (_, store, sessions, _provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, hydrated = _records(drive_store)
        plan = next(
            row.record for row in hydrated if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
        )
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        driver, lease = _direct_driver(drive_store, sessions, runtime)
        try:
            attempt = driver.state.operations[plan.operation_id].attempts[1]
            unknown = driver.record(
                "op_unknown",
                {
                    "reason": "authenticated provider lookup proved not_started",
                    "dispatch_evidence": [attempt.dispatch.record_id],
                    "observation": {"code": "lookup_not_started", "retryable": True},
                    "retry": "retry",
                    "retry_at_ms": 0,
                    "duplicate_cost_risk": False,
                    "measurement_missing": True,
                },
                attempt.execution_plan,
            )
            driver.commit([unknown])

            assert _run_core_step(driver)

            retry = driver.state.operations[plan.operation_id].attempts[2]
            assert retry.plan.feature_level == 3
            assert retry.execution_plan.feature_level == 3
            assert retry.execution_plan.payload["execution_identity"] == plan.payload["execution_identity"]
            assert retry.execution_plan.payload["idempotency_key"] == plan.payload["idempotency_key"]
        finally:
            store.release(lease)


def test_ordinary_idempotent_unknown_still_retries(tmp_path: Path) -> None:
    database = tmp_path / "ordinary-retry.sqlite"
    provider = ControlledProvider(database, "unknown", idempotent=True)
    provider.install()
    with open_store(database) as store:
        store.install_schema()
        start_ordinary(store)
        drive(
            store,
            "s",
            runtime=ordinary_runtime(
                database,
                [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")],
                [effect_tool(idempotent=True)],
                providers={"effect": provider},
            ),
        )
        assert provider.counts() == (2, 1)
