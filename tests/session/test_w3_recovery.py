"""W3 unknown recovery stays pinned to the submitted command identity."""

from __future__ import annotations

from pathlib import Path

import pytest

from vv_agent import Agent
from vv_agent.kernel._boundary_hydration import SessionHydration, hydrate_session_records
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.kernel import drive
from vv_agent.session.surfaces import SessionDriver
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
