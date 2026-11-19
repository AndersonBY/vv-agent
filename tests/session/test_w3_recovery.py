"""W3 unknown recovery stays pinned to the submitted command identity."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent import Agent
from vv_agent.kernel._boundary_hydration import MissingContentLink, SessionHydration, hydrate_session_records
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._store import Lease
from vv_agent.session import _recovery_phase, kernel
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.environments import (
    LookupBinding,
    LookupStatus,
    ProviderLookupResult,
    execution_id_for,
    stable_operation_key,
)
from vv_agent.session.kernel import drive
from vv_agent.session.providers import Accepted, Definitive, Unknown
from vv_agent.session.records import InboxItem
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.tool_results import prepare_tool_resources
from vv_agent.tools import ToolRegistry, ToolSpec, build_default_registry
from vv_agent.tools.metadata import ToolIdempotency, ToolMetadata
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

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
    def __init__(self, provider: HostCommandProvider, *, losses: int = 1) -> None:
        self.provider = provider
        self.remaining = losses

    def invoke(self, callback, *, timeout, token, cancellation_grace, clock):
        del timeout, token, cancellation_grace, clock
        submitted = len(self.provider.probe.submitted_plans)
        outcome = callback()
        if len(self.provider.probe.submitted_plans) > submitted and self.remaining:
            self.remaining -= 1
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


def _expected_lookup_binding(plan, turn) -> LookupBinding:
    identity = plan.payload["execution_identity"]
    return LookupBinding(
        tenant_id=identity["tenant_id"],
        session_id=plan.session_id,
        operation_id=plan.operation_id,
        attempt=plan.attempt,
        request_digest=plan.payload["request_digest"],
        idempotency_key=plan.payload["idempotency_key"],
        execution_id=identity["execution_id"],
        command_digest=plan.payload["command_digest"],
        environment_id=identity["environment_id"],
        generation=turn.payload["environment"]["generation"],
        provider_id=plan.payload["provider_binding"],
    )


def _lookup_result(binding: LookupBinding, status: LookupStatus) -> ProviderLookupResult:
    evidence = (f"provider-registry:{status.value}",)
    handle = {
        "kind": "provider",
        "provider": binding.provider_id,
        "job_id": f"job-{status.value}",
        "operation_id": binding.operation_id,
        "attempt": binding.attempt,
        "request_digest": binding.request_digest,
        "execution_id": binding.execution_id,
        "command_digest": binding.command_digest,
        "environment_id": binding.environment_id,
        "generation": binding.generation,
        "evidence": list(evidence),
        "query_ref": None,
        "cancel_ref": None,
    }
    outcome = {
        LookupStatus.NOT_STARTED: None,
        LookupStatus.RUNNING: Accepted(handle),
        LookupStatus.EXITED: Definitive(
            ToolExecutionResult(tool_call_id="bash-1", content="provider-exited").to_dict(),
            evidence,
        ),
        LookupStatus.UNKNOWN: Unknown("provider registry unavailable"),
    }[status]
    return ProviderLookupResult(binding, status, outcome, evidence, feature_level=3)


def _other_lookup_binding(binding: LookupBinding) -> LookupBinding:
    operation_id = f"{binding.operation_id}/other"
    idempotency_key = stable_operation_key(binding.session_id, operation_id)
    return replace(
        binding,
        operation_id=operation_id,
        idempotency_key=idempotency_key,
        execution_id=execution_id_for(binding.tenant_id, binding.environment_id, idempotency_key),
    )


def _assert_w3_lookup_fail_closed(
    rows,
    hydrated,
    provider: HostCommandProvider,
    plan,
    *,
    status: str,
    evidence: list[str],
) -> None:
    unknown = next(row.record for row in hydrated if row.record.kind == "op_unknown")
    tool_records = [row.record for row in rows if "/tool/" in str(row.record.operation_id)]
    assert unknown.operation_id == plan.operation_id
    assert unknown.attempt == plan.attempt
    assert unknown.payload["retry"] == "stop"
    assert unknown.payload["retry_at_ms"] is None
    assert unknown.payload["observation"]["code"] == "provider_lookup_recovery_unavailable"
    assert unknown.payload["observation"]["retryable"] is False
    assert unknown.payload["observation"]["lookup_status"] == status
    assert unknown.payload["observation"]["lookup_evidence"] == evidence
    assert [record.kind for record in tool_records].count("op_planned") == 1
    assert [record.kind for record in tool_records].count("op_started") == 1
    assert not any(record.attempt == 2 for record in tool_records)
    assert len(provider.probe.submitted_plans) == 1
    assert len(provider.probe.submitted_calls) == 1


def _level_three_lookup_item(
    plan,
    turn,
    *,
    input_id: str = "provider-lookup-running",
    status: LookupStatus = LookupStatus.RUNNING,
) -> InboxItem:
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
        "job_id": f"job-{status.value}",
        "operation_id": binding["operation_id"],
        "attempt": binding["attempt"],
        "request_digest": binding["request_digest"],
        "execution_id": binding["execution_id"],
        "command_digest": binding["command_digest"],
        "environment_id": binding["environment_id"],
        "generation": binding["generation"],
        "evidence": [f"provider-registry:{status.value}"],
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
                "status": status.value,
                "binding": binding,
                "outcome": {"kind": "Accepted", "handle": handle, "yield_on_steer": False}
                if status is LookupStatus.RUNNING
                else None,
                "evidence": [f"provider-registry:{status.value}"],
            },
        },
        target_turn_id=plan.turn_id,
        generation=turn.payload["generation"],
        feature_level=3,
    )


def _push_lookup(
    sessions: SessionDriver,
    store,
    item: InboxItem,
    *,
    actor_subject_id: str = "bash",
) -> None:
    actor = {
        "tenant_id": TENANT_ID,
        "subject_id": actor_subject_id,
        "subject_kind": "service",
        "attributes": {"role": actor_subject_id},
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
def _lost_w3_submission(
    tmp_path: Path,
    *,
    provider: HostCommandProvider | None = None,
    losses: int = 1,
):
    path = tmp_path / "w3-lookup.sqlite3"
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "w3-lookup-keys.sqlite3")
    provider = provider or HostCommandProvider()
    agent = _idempotent_agent()
    config = _config(provider)
    _create_level_three_session(path, key_provider, agent, config)
    invoker = LoseSubmittedReceipt(provider, losses=losses)
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


def test_w3_worker_loss_lookup_unknown_never_retries_or_resubmits(tmp_path: Path) -> None:
    with (
        _lost_w3_submission(tmp_path) as (_path, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        before, hydrated_before = _records(drive_store)
        before_tool = [row.record for row in before if "/tool/" in str(row.record.operation_id)]
        tool_plan = next(
            row.record for row in hydrated_before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
        )
        turn = next(row.record for row in hydrated_before if row.record.kind == "turn_started")
        assert tool_plan.feature_level == 3
        assert tool_plan.payload["tool"]["idempotency"] == "supported"
        assert [record.kind for record in before_tool].count("op_planned") == 1
        assert [record.kind for record in before_tool].count("op_started") == 1

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        assert provider.probe.lookup_calls == [
            (tool_plan.payload["execution_identity"]["execution_id"], _expected_lookup_binding(tool_plan, turn))
        ]
        _assert_w3_lookup_fail_closed(
            rows,
            hydrated,
            provider,
            tool_plan,
            status="unknown",
            evidence=["test-host-lookup-not-wired"],
        )


@pytest.mark.parametrize("status", [LookupStatus.RUNNING, LookupStatus.EXITED])
def test_level_three_worker_loss_records_known_lookup_status_fail_closed(
    tmp_path: Path,
    status: LookupStatus,
) -> None:
    provider = HostCommandProvider()
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(binding, status)

    with (
        _lost_w3_submission(tmp_path, provider=provider) as (_path, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, before = _records(drive_store)
        plan = next(row.record for row in before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id))
        turn = next(row.record for row in before if row.record.kind == "turn_started")
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        expected_binding = _expected_lookup_binding(plan, turn)
        assert provider.probe.lookup_calls == [(expected_binding.execution_id, expected_binding)]
        _assert_w3_lookup_fail_closed(
            rows,
            hydrated,
            provider,
            plan,
            status=status.value,
            evidence=[f"provider-registry:{status.value}"],
        )


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_evidence"),
    [
        ("exception", "unknown", ["provider-lookup-error"]),
        ("timeout", "unknown", ["provider-lookup-timeout"]),
        ("malformed", "invalid", ["provider-lookup-result-invalid"]),
        ("mismatch", "invalid", ["provider-lookup-result-invalid"]),
        ("not_started_malformed", "invalid", ["provider-lookup-result-invalid"]),
        ("not_started_mismatch", "invalid", ["provider-lookup-result-invalid"]),
    ],
)
def test_level_three_worker_loss_rejects_lookup_failures_without_resubmission(
    tmp_path: Path,
    failure: str,
    expected_status: str,
    expected_evidence: list[str],
) -> None:
    provider = HostCommandProvider()

    def lookup(_execution_id, binding):
        if failure == "exception":
            raise RuntimeError("registry unavailable")
        if failure == "timeout":
            raise TimeoutError("lookup timed out")
        if failure == "malformed":
            return {"status": "unknown"}
        if failure == "not_started_malformed":
            return {"status": "not_started"}
        if failure == "not_started_mismatch":
            return _lookup_result(_other_lookup_binding(binding), LookupStatus.NOT_STARTED)
        return _lookup_result(_other_lookup_binding(binding), LookupStatus.UNKNOWN)

    provider.probe.lookup_handler = lookup
    with (
        _lost_w3_submission(tmp_path, provider=provider) as (_path, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, before = _records(drive_store)
        plan = next(row.record for row in before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id))
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        assert len(provider.probe.lookup_calls) == 1
        _assert_w3_lookup_fail_closed(
            rows,
            hydrated,
            provider,
            plan,
            status=expected_status,
            evidence=expected_evidence,
        )


def test_level_three_provider_lookup_is_preserved_and_fails_closed_without_resubmission(
    tmp_path: Path,
) -> None:
    with (
        _lost_w3_submission(tmp_path) as (path, _store, sessions, provider),
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
        assert len(provider.probe.lookup_calls) == 1
        assert b"job-running-1" not in path.read_bytes()


def test_w3_not_started_lookup_publishes_evidence_then_retries_once(tmp_path: Path) -> None:
    provider = HostCommandProvider()
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(binding, LookupStatus.NOT_STARTED)

    with (
        _lost_w3_submission(tmp_path, provider=provider) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, hydrated = _records(drive_store)
        plan = next(
            row.record for row in hydrated if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
        )
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        tool_records = [row for row in rows if row.record.operation_id == plan.operation_id]
        applied_row = next(
            row
            for row in hydrated
            if row.record.kind == "input_applied" and row.record.payload["input"]["kind"] == "provider_evidence"
        )
        applied = applied_row.record
        evidence_item = applied.payload["input"]
        evidence_id = evidence_item["input_id"]
        unknown_row = next(row for row in tool_records if row.record.kind == "op_unknown" and row.record.attempt == 1)
        unknown = next(row.record for row in hydrated if row.record.record_id == unknown_row.record.record_id)
        retry_plan = next(row for row in tool_records if row.record.kind == "op_planned" and row.record.attempt == 2)
        retry_started = next(row for row in tool_records if row.record.kind == "op_started" and row.record.attempt == 2)

        assert evidence_id.startswith("provider-recovery/lookup/")
        assert applied.payload["disposition"] == "applied"
        assert unknown.payload["retry"] == "retry"
        assert unknown.payload["observation"]["evidence_input_id"] == evidence_id
        assert unknown.payload["observation"]["evidence_owner"] == f"input:{evidence_id}"
        assert unknown.payload["observation"]["evidence_slot"] == "provider_lookup"
        assert applied_row.seq < unknown_row.seq < retry_plan.seq < retry_started.seq
        assert [row.record.attempt for row in tool_records if row.record.kind == "op_planned"] == [1, 2]
        assert [row.record.attempt for row in tool_records if row.record.kind == "op_prepared"] == [1]
        assert not any(row.record.attempt == 3 for row in tool_records)
        assert len(provider.probe.submitted_plans) == 2
        first, second = provider.probe.submitted_plans
        assert first.attempt == 1 and second.attempt == 2
        for field in ("execution_identity", "idempotency_key", "command_digest", "provider_binding"):
            assert second.payload[field] == first.payload[field]
        assert second.feature_level == first.feature_level == 3
        assert second.payload["execution_identity"]["operation_key"] == first.payload["execution_identity"]["operation_key"]
        assert sum(row.record.kind == "op_started" for row in tool_records) == 2

        link = drive_store._rows(
            "SELECT owner,slot,content_id FROM ("
            "SELECT 'input:' || input_id AS owner,slot,content_id FROM sk_content_link "
            "WHERE tenant_id=%s AND session_id=%s AND input_id=%s AND slot='provider_lookup'"
            ")",
            (TENANT_ID, SESSION_ID, evidence_id),
        )
        assert len(link) == 1
        assert link[0][0:2] == (f"input:{evidence_id}", "provider_lookup")
        assert link[0][2]
        inbox = drive_store._rows(
            "SELECT input_seq,consumed_seq FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
            (TENANT_ID, SESSION_ID, evidence_id),
        )
        assert len(inbox) == 1 and inbox[0][1] is not None


def test_w3_not_started_evidence_commit_failure_creates_no_retry_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = HostCommandProvider()
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(binding, LookupStatus.NOT_STARTED)
    declare_input_content = _recovery_phase.declare_input_content

    def fail_lookup_publication(writer, item):
        if item.kind == "provider_evidence":
            raise RuntimeError("evidence publication failed")
        return declare_input_content(writer, item)

    monkeypatch.setattr(_recovery_phase, "declare_input_content", fail_lookup_publication)
    with (
        _lost_w3_submission(tmp_path, provider=provider) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        with pytest.raises(RuntimeError, match="evidence publication failed"):
            drive(drive_store, SESSION_ID, runtime=runtime)

        rows, _ = _records(drive_store)
        assert len(provider.probe.submitted_plans) == 1
        assert not any(row.record.kind == "op_planned" and row.record.attempt == 2 for row in rows)


def test_w3_not_started_second_submission_unknown_does_not_create_attempt_three(tmp_path: Path) -> None:
    provider = HostCommandProvider(Unknown("second submission has no receipt"))
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(binding, LookupStatus.NOT_STARTED)

    with (
        _lost_w3_submission(tmp_path, provider=provider) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, _ = _records(drive_store)
        tool_records = [row.record for row in rows if "/tool/" in str(row.record.operation_id)]
        assert len(provider.probe.submitted_plans) == 2
        assert [record.attempt for record in tool_records if record.kind == "op_planned"] == [1, 2]
        assert [record.attempt for record in tool_records if record.kind == "op_started"] == [1, 2]
        assert not any(record.attempt == 3 for record in tool_records)
        assert not any(
            record.kind == "op_unknown" and record.attempt == 2 and record.payload["retry"] == "retry" for record in tool_records
        )


def test_w3_not_started_second_recovery_lookup_does_not_create_attempt_three(tmp_path: Path) -> None:
    provider = HostCommandProvider()
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(binding, LookupStatus.NOT_STARTED)

    with (
        _lost_w3_submission(tmp_path, provider=provider, losses=2) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        with pytest.raises(WorkerLost):
            drive(drive_store, SESSION_ID, runtime=runtime)

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        tool_records = [row.record for row in rows if "/tool/" in str(row.record.operation_id)]
        assert len(provider.probe.submitted_plans) == 2
        assert [record.attempt for record in tool_records if record.kind == "op_planned"] == [1, 2]
        assert not any(record.attempt == 3 for record in tool_records)
        attempt_two_unknown = next(row.record for row in hydrated if row.record.kind == "op_unknown" and row.record.attempt == 2)
        assert attempt_two_unknown.payload["retry"] == "stop"
        assert attempt_two_unknown.payload["observation"]["lookup_status"] == "not_started"
        assert attempt_two_unknown.payload["observation"]["code"] == "provider_lookup_recovery_unavailable"


def test_w3_not_started_evidence_from_an_untrusted_actor_does_not_retry(tmp_path: Path) -> None:
    with (
        _lost_w3_submission(tmp_path) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, hydrated = _records(drive_store)
        plan = next(
            row.record for row in hydrated if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
        )
        turn = next(row.record for row in hydrated if row.record.kind == "turn_started")
        _push_lookup(
            sessions,
            drive_store,
            _level_three_lookup_item(
                plan,
                turn,
                input_id="provider-recovery/lookup/untrusted",
                status=LookupStatus.NOT_STARTED,
            ),
        )

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        applied = next(
            row.record
            for row in hydrated
            if row.record.kind == "input_applied"
            and row.record.payload["input"]["input_id"] == "provider-recovery/lookup/untrusted"
        )
        assert applied.payload["disposition"] == "rejected"
        assert applied.payload["reason"] == "provider_lookup_recovery_unavailable"
        assert len(provider.probe.submitted_plans) == 1
        assert not any(row.record.kind == "op_planned" and row.record.attempt == 2 for row in rows)


def test_w3_not_started_evidence_with_missing_content_does_not_retry(tmp_path: Path) -> None:
    with (
        _lost_w3_submission(tmp_path) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, hydrated = _records(drive_store)
        plan = next(
            row.record for row in hydrated if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
        )
        turn = next(row.record for row in hydrated if row.record.kind == "turn_started")
        item = _level_three_lookup_item(
            plan,
            turn,
            input_id="provider-recovery/lookup/missing",
            status=LookupStatus.NOT_STARTED,
        )
        _push_lookup(sessions, drive_store, item, actor_subject_id="provider-recovery")
        drive_store._rows(
            "DELETE FROM sk_content_link WHERE tenant_id=%s AND session_id=%s AND input_id=%s AND slot='provider_lookup'",
            (TENANT_ID, SESSION_ID, item.input_id),
        )

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        with pytest.raises(MissingContentLink):
            drive(drive_store, SESSION_ID, runtime=runtime)

        rows, _ = _records(drive_store)
        assert len(provider.probe.submitted_plans) == 1
        assert not any(row.record.kind == "op_planned" and row.record.attempt == 2 for row in rows)


def test_ordinary_idempotent_unknown_still_retries(tmp_path: Path) -> None:
    database = tmp_path / "ordinary-retry.sqlite"

    class LookupTrackingProvider(ControlledProvider):
        def __init__(self) -> None:
            super().__init__(database, "unknown", idempotent=True)
            self.lookup_calls = 0

        def lookup(self, execution_id, binding):
            self.lookup_calls += 1
            raise AssertionError("level-1 recovery must not call provider.lookup")

    provider = LookupTrackingProvider()
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
        assert provider.lookup_calls == 0
