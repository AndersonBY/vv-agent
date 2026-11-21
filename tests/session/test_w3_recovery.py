"""W3 unknown recovery stays pinned to the submitted command identity."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from vv_agent import Agent
from vv_agent.kernel import _driver_inputs
from vv_agent.kernel._boundary_hydration import MissingContentLink, SessionHydration, hydrate_session_records
from vv_agent.kernel._content_publication import declare_input_content, link_declared_input_content
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_inputs import provider_recovery_lookup_input_id
from vv_agent.kernel._store import Lease
from vv_agent.runtime.remote_environment import RemoteProtocolError
from vv_agent.session import _recovery_phase, kernel
from vv_agent.session.content_runtime import SessionContentWriter
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.environments import (
    InvalidProviderLookupResult,
    LookupBinding,
    LookupStatus,
    ProviderLookupResult,
    execution_id_for,
    stable_operation_key,
)
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.providers import Accepted, Definitive, ProviderResultInvalid, Unknown
from vv_agent.session.records import InboxItem, RecordError
from vv_agent.session.store import Conflict
from vv_agent.session.surfaces import SessionDriver
from vv_agent.session.timing import ThreadInvoker
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
    RewriteCommandHook,
    _binding,
    _config,
    _create_level_three_session,
    _not_called,
)


class WorkerLost(BaseException):
    pass


class LoseSubmittedReceipt:
    def __init__(self, provider: HostCommandProvider, *, losses: int = 1, delegate=None) -> None:
        self.provider = provider
        self.remaining = losses
        self.delegate = delegate

    def invoke(self, callback, *, timeout, token, cancellation_grace, clock):
        submitted = len(self.provider.probe.submitted_plans)
        if self.delegate is None:
            outcome = callback()
        else:
            outcome = self.delegate.invoke(
                callback,
                timeout=timeout,
                token=token,
                cancellation_grace=cancellation_grace,
                clock=clock,
            )
        if len(self.provider.probe.submitted_plans) > submitted and self.remaining:
            self.remaining -= 1
            raise WorkerLost("worker lost after provider submission")
        return outcome


class CatchAllSynchronousInvoker:
    def invoke(self, callback, *, timeout, token, cancellation_grace, clock):
        del timeout, token, cancellation_grace, clock
        try:
            return callback()
        except Exception as exc:
            return Unknown(f"{type(exc).__name__}: synchronous outcome unavailable")


def _idempotent_agent(*, hooks: list[Any] | None = None) -> Agent:
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
        hooks=list(hooks or []),
        metadata={},
    )


class CountingRewriteCommandHook(RewriteCommandHook):
    def __init__(self, command: str) -> None:
        super().__init__(command)
        self.calls = 0

    def before_tool_call(self, event):
        self.calls += 1
        return super().before_tool_call(event)


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


def _lookup_result(
    binding: LookupBinding,
    status: LookupStatus,
    *,
    yield_on_steer: bool = False,
) -> ProviderLookupResult:
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
        LookupStatus.RUNNING: Accepted(handle, yield_on_steer=yield_on_steer),
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
                "outcome": (
                    {"kind": "Accepted", "handle": handle, "yield_on_steer": False}
                    if status is LookupStatus.RUNNING
                    else {
                        "kind": "Definitive",
                        "result": ToolExecutionResult(tool_call_id="bash-1", content="provider-exited").to_dict(),
                        "usage": {},
                        "shared_state": None,
                        "evidence": [f"provider-registry:{status.value}"],
                    }
                    if status is LookupStatus.EXITED
                    else {"kind": "Unknown", "reason": "provider registry unavailable"}
                    if status is LookupStatus.UNKNOWN
                    else None
                ),
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
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=TENANT_ID, session_id=SESSION_ID)
        actor, actor_ref = writer.service_principal(actor_subject_id)
        publication = declare_input_content(writer, replace(item, actor=actor))
        tx.push(SESSION_ID, publication.item)
        link_declared_input_content(writer, publication.item)
        writer.link_input_actor(item.input_id, actor_ref)


def _push_control(sessions: SessionDriver, store, item: InboxItem) -> None:
    with store.atomic() as tx:
        sessions._push_in_transaction(tx, SESSION_ID, item)


def _control_item(plan, turn, action: str) -> InboxItem:
    return InboxItem(
        f"control/{action}",
        "control",
        {"action": action},
        target_turn_id=plan.turn_id,
        generation=turn.payload["generation"],
    )


def _with_provider_lookup_input_id(item: InboxItem) -> InboxItem:
    lookup = item.payload["provider_lookup"]
    return replace(
        item,
        input_id=provider_recovery_lookup_input_id(
            item.payload["operation_id"],
            item.payload["attempt"],
            item.payload["request_digest"],
            lookup,
        ),
    )


def _invalid_durable_lookup_evidence(item: InboxItem, variant: str) -> InboxItem:
    if variant == "turn":
        item = replace(item, target_turn_id=f"{item.target_turn_id}/other")
    elif variant == "generation":
        assert item.generation is not None
        item = replace(item, generation=item.generation + 1)
    elif variant == "actor":
        pass
    elif variant == "tenant":
        binding = item.payload["provider_lookup"]["binding"]
        binding["tenant_id"] = "other-tenant"
        binding["execution_id"] = execution_id_for(binding["tenant_id"], binding["environment_id"], binding["idempotency_key"])
    elif variant in {"session", "operation"}:
        binding = item.payload["provider_lookup"]["binding"]
        binding[f"{variant}_id"] = f"other-{variant}"
        binding["idempotency_key"] = stable_operation_key(binding["session_id"], binding["operation_id"])
        binding["execution_id"] = execution_id_for(binding["tenant_id"], binding["environment_id"], binding["idempotency_key"])
    elif variant == "request_digest":
        item.payload["provider_lookup"]["binding"]["request_digest"] = "f" * 64
    elif variant != "input_id":
        raise AssertionError(f"unexpected evidence variant: {variant}")

    if variant == "input_id":
        return replace(item, input_id="provider-recovery/lookup/wrong")
    return _with_provider_lookup_input_id(item)


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


def _consume_one_input(store, sessions: SessionDriver, runtime) -> None:
    driver, lease = _direct_driver(store, sessions, runtime)
    try:
        assert _run_core_step(driver)
    finally:
        store.release(lease)


@contextmanager
def _lost_w3_submission(
    tmp_path: Path,
    *,
    provider: HostCommandProvider | None = None,
    losses: int = 1,
    hooks: list[Any] | None = None,
    invoker=None,
):
    path = tmp_path / "w3-lookup.sqlite3"
    key_provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "w3-lookup-keys.sqlite3")
    provider = provider or HostCommandProvider()
    agent = _idempotent_agent(hooks=hooks)
    config = _config(provider)
    _create_level_three_session(path, key_provider, agent, config)
    invoker = invoker if invoker is not None else LoseSubmittedReceipt(provider, losses=losses)
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
            assert handle._pending_submission is not None
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


def test_level_three_worker_loss_running_lookup_queries_original_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = HostCommandProvider()
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(
        binding, LookupStatus.RUNNING, yield_on_steer=True
    )
    provider.probe.query_handler = lambda handle: Accepted(handle, yield_on_steer=True)

    with (
        _lost_w3_submission(tmp_path, provider=provider) as (_path, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, before = _records(drive_store)
        plan = next(row.record for row in before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id))
        turn = next(row.record for row in before if row.record.kind == "turn_started")
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        runtime.poll_ms = 1
        monkeypatch.setattr(drive_store, "next_drive_delay_ms", lambda *_args, **_kwargs: 0)
        drive(drive_store, SESSION_ID, runtime=runtime)

        expected_binding = _expected_lookup_binding(plan, turn)
        assert provider.probe.lookup_calls == [(expected_binding.execution_id, expected_binding)]
        assert len(provider.probe.submitted_plans) == 1
        rows, hydrated = _records(drive_store)
        assert len(provider.probe.query_calls) == 1
        assert len([row for row in rows if row.record.kind == "op_planned" and row.record.operation_id == plan.operation_id]) == 1
        assert not any(row.record.kind == "op_planned" and row.record.attempt == 2 for row in rows)
        assert not any(row.record.kind == "op_started" and row.record.attempt == 2 for row in rows)
        assert not any(
            row.record.kind in {"op_unknown", "op_completed"}
            and row.record.operation_id == plan.operation_id
            and row.record.attempt == plan.attempt
            for row in rows
        )

        parked_rows = [row for row in hydrated if row.record.kind == "op_parked"]
        assert len(parked_rows) == 1
        parked = parked_rows[0].record
        assert parked.operation_id == plan.operation_id
        assert parked.attempt == plan.attempt
        assert parked.payload["yield_on_steer"] is True
        assert parked.payload["poll_at_ms"] is not None
        assert set(parked.payload["handle"]) == {
            "kind",
            "provider",
            "job_id",
            "operation_id",
            "attempt",
            "request_digest",
            "evidence",
            "query_ref",
            "cancel_ref",
        }

        query_handle = provider.probe.query_calls[0]
        assert set(query_handle) == {
            "kind",
            "provider",
            "job_id",
            "operation_id",
            "attempt",
            "request_digest",
            "evidence",
            "query_ref",
            "cancel_ref",
            "execution_id",
            "command_digest",
            "environment_id",
            "generation",
        }
        assert query_handle["execution_id"] == expected_binding.execution_id
        assert query_handle["command_digest"] == expected_binding.command_digest
        assert query_handle["environment_id"] == expected_binding.environment_id
        assert query_handle["generation"] == expected_binding.generation


def test_level_three_tampered_not_started_binding_never_creates_attempt_two(tmp_path: Path) -> None:
    with (
        _lost_w3_submission(tmp_path) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, before = _records(drive_store)
        plan = next(row.record for row in before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id))
        turn = next(row.record for row in before if row.record.kind == "turn_started")
        lookup = _level_three_lookup_item(plan, turn, status=LookupStatus.NOT_STARTED)
        lookup.payload["provider_lookup"]["binding"]["command_digest"] = "f" * 64
        lookup = _with_provider_lookup_input_id(lookup)
        _push_lookup(sessions, drive_store, lookup, actor_subject_id="provider-recovery")

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        applied = next(
            row.record
            for row in hydrated
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == lookup.input_id
        )
        assert applied.payload["disposition"] == "rejected"
        assert applied.payload["reason"] == "provider_result_invalid"
        assert not any(row.record.kind == "op_planned" and row.record.attempt == 2 for row in rows)
        assert len(provider.probe.submitted_plans) == 1


def test_level_three_worker_loss_exited_lookup_completes_original_operation(tmp_path: Path) -> None:
    provider = HostCommandProvider()
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(binding, LookupStatus.EXITED)

    with (
        _lost_w3_submission(tmp_path, provider=provider) as (_path, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, before = _records(drive_store)
        plan = next(row.record for row in before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id))
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        completed_rows = [
            row
            for row in hydrated
            if row.record.kind == "op_completed"
            and row.record.operation_id == plan.operation_id
            and row.record.attempt == plan.attempt
        ]
        assert len(completed_rows) == 1
        completed = completed_rows[0].record
        assert completed.payload["execution_started"] is True
        assert completed.payload["result"]["tool_call_id"] == plan.payload["request"]["id"]
        assert completed.payload["result"]["content"] == "provider-exited"
        assert len(provider.probe.submitted_plans) == 1
        assert provider.probe.query_calls == []
        assert not any(row.record.kind == "op_parked" for row in rows)
        assert not any(row.record.kind == "op_planned" and row.record.attempt == 2 for row in rows)


def test_w3_parked_exited_lookup_must_retain_acceptance_evidence(tmp_path: Path) -> None:
    with (
        _lost_w3_submission(tmp_path) as (_, _store, sessions, _provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, before = _records(drive_store)
        plan = next(row.record for row in before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id))
        turn = next(row.record for row in before if row.record.kind == "turn_started")

        running = _with_provider_lookup_input_id(_level_three_lookup_item(plan, turn, status=LookupStatus.RUNNING))
        _push_lookup(sessions, drive_store, running, actor_subject_id="provider-recovery")
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        _consume_one_input(drive_store, sessions, runtime)

        _, hydrated = _records(drive_store)
        parked_before = next(
            row.record
            for row in hydrated
            if row.record.kind == "op_parked"
            and row.record.operation_id == plan.operation_id
            and row.record.attempt == plan.attempt
        )
        assert "provider-registry:running" in parked_before.payload["handle"]["evidence"]

        exited = _with_provider_lookup_input_id(_level_three_lookup_item(plan, turn, status=LookupStatus.EXITED))
        _push_lookup(sessions, drive_store, exited, actor_subject_id="provider-recovery")
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        _consume_one_input(drive_store, sessions, runtime)

        rows, hydrated = _records(drive_store)
        applied = next(
            row.record
            for row in hydrated
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == exited.input_id
        )
        parked_after = [
            row.record
            for row in hydrated
            if row.record.kind == "op_parked"
            and row.record.operation_id == plan.operation_id
            and row.record.attempt == plan.attempt
        ]
        assert applied.payload["disposition"] == "rejected"
        assert applied.payload["reason"] == "provider_result_invalid"
        assert len(parked_after) == 1
        assert parked_after[0].record_id == parked_before.record_id
        assert parked_after[0].payload == parked_before.payload
        assert not any(
            row.record.kind == "op_completed"
            and row.record.operation_id == plan.operation_id
            and row.record.attempt == plan.attempt
            for row in rows
        )


def test_level_three_conflicting_exited_lookup_is_rejected_without_duplicate_completion(tmp_path: Path) -> None:
    with (
        _lost_w3_submission(tmp_path) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, before = _records(drive_store)
        plan = next(row.record for row in before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id))
        turn = next(row.record for row in before if row.record.kind == "turn_started")
        first = _with_provider_lookup_input_id(_level_three_lookup_item(plan, turn, status=LookupStatus.EXITED))
        conflicting = _level_three_lookup_item(plan, turn, status=LookupStatus.EXITED)
        conflicting.payload["provider_lookup"]["outcome"]["result"]["content"] = "different result"
        conflicting = _with_provider_lookup_input_id(conflicting)
        _push_lookup(sessions, drive_store, first, actor_subject_id="provider-recovery")
        _push_lookup(sessions, drive_store, conflicting, actor_subject_id="provider-recovery")

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        _, hydrated = _records(drive_store)
        applied = next(
            row.record
            for row in hydrated
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == conflicting.input_id
        )
        assert applied.payload["disposition"] == "rejected"
        assert applied.payload["reason"] == "result conflict"
        completed = [
            row.record
            for row in hydrated
            if row.record.kind == "op_completed"
            and row.record.operation_id == plan.operation_id
            and row.record.attempt == plan.attempt
        ]
        assert len(completed) == 1
        assert completed[0].payload["result"]["content"] == "provider-exited"
        assert len(provider.probe.submitted_plans) == 1


def test_level_three_worker_loss_running_handle_mismatch_fails_closed_without_submit(tmp_path: Path) -> None:
    provider = HostCommandProvider()

    def lookup(_execution_id, binding):
        result = _lookup_result(binding, LookupStatus.RUNNING)
        assert isinstance(result.outcome, Accepted)
        result.outcome.handle["execution_id"] = "f" * 64
        return result

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
        _assert_w3_lookup_fail_closed(
            rows,
            hydrated,
            provider,
            plan,
            status="invalid",
            evidence=["provider-lookup-result-invalid"],
        )
        assert provider.probe.query_calls == []


def test_level_three_worker_loss_malformed_query_fails_closed_without_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = HostCommandProvider()
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(binding, LookupStatus.RUNNING)

    def query(handle):
        malformed = handle | {"execution_id": "f" * 64}
        return Accepted(malformed)

    provider.probe.query_handler = query
    with (
        _lost_w3_submission(tmp_path, provider=provider) as (_path, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, before = _records(drive_store)
        plan = next(row.record for row in before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id))
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        runtime.poll_ms = 1
        monkeypatch.setattr(drive_store, "next_drive_delay_ms", lambda *_args, **_kwargs: 0)
        with pytest.raises(ProviderResultInvalid) as caught:
            drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        parked = [row.record for row in hydrated if row.record.kind == "op_parked"]
        assert caught.value._query_operation_key == (plan.operation_id, plan.attempt)
        assert len(provider.probe.submitted_plans) == 1
        assert len(provider.probe.query_calls) == 1
        assert len(parked) == 1
        assert parked[0].operation_id == plan.operation_id
        assert parked[0].attempt == plan.attempt
        assert not any(
            row.record.kind in {"op_unknown", "op_completed"}
            and row.record.operation_id == plan.operation_id
            and row.record.attempt == plan.attempt
            for row in rows
        )
        assert not any(row.record.kind == "op_planned" and row.record.attempt == 2 for row in rows)
        assert not any(row.record.kind == "op_started" and row.record.attempt == 2 for row in rows)


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


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_evidence"),
    [
        ("timeout", "unknown", ["provider-lookup-timeout"]),
        ("lookup_invalid", "invalid", ["provider-lookup-result-invalid"]),
        ("result_invalid", "invalid", ["provider-lookup-result-invalid"]),
        ("remote_invalid", "invalid", ["provider-lookup-result-invalid"]),
    ],
)
def test_w3_lookup_failure_classification_is_invoker_independent(
    tmp_path: Path,
    failure: str,
    expected_status: str,
    expected_evidence: list[str],
) -> None:
    observed = []
    for delegate in (ThreadInvoker(), CatchAllSynchronousInvoker()):
        provider = HostCommandProvider()

        def lookup(_execution_id, binding):
            del binding
            error = {
                "timeout": TimeoutError("lookup timed out"),
                "lookup_invalid": InvalidProviderLookupResult("invalid lookup"),
                "result_invalid": ProviderResultInvalid(),
                "remote_invalid": RemoteProtocolError("invalid remote receipt"),
            }[failure]
            raise error

        provider.probe.lookup_handler = lookup
        invoker = LoseSubmittedReceipt(provider, delegate=delegate)
        case_dir = tmp_path / type(delegate).__name__
        case_dir.mkdir()
        with (
            _lost_w3_submission(case_dir, provider=provider, invoker=invoker) as (_path, _store, sessions, provider),
            sessions._open_drive_store() as drive_store,
        ):
            _, before = _records(drive_store)
            plan = next(
                row.record for row in before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
            )
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
            unknown = next(row.record for row in hydrated if row.record.kind == "op_unknown")
            observed.append((unknown.payload["observation"]["lookup_status"], unknown.payload["observation"]["lookup_evidence"]))

    assert observed == [(expected_status, expected_evidence), (expected_status, expected_evidence)]


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

        lookup = _with_provider_lookup_input_id(_level_three_lookup_item(plan, turn))
        _push_lookup(sessions, drive_store, lookup, actor_subject_id="provider-recovery")
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


def test_w3_not_started_lookup_publishes_evidence_then_retries_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = HostCommandProvider()
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(binding, LookupStatus.NOT_STARTED)
    decoded_lookup = _driver_inputs._decoded_provider_lookup
    decode_count = 0

    def count_decode(*args, **kwargs):
        nonlocal decode_count
        decode_count += 1
        return decoded_lookup(*args, **kwargs)

    monkeypatch.setattr(_driver_inputs, "_decoded_provider_lookup", count_decode)

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
        assert decode_count == 1

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


def test_w3_prepared_rewrite_not_started_retry_reuses_hook_result(tmp_path: Path) -> None:
    provider = HostCommandProvider()
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(binding, LookupStatus.NOT_STARTED)
    hook = CountingRewriteCommandHook("printf 'rewritten during recovery\\n'")

    with (
        _lost_w3_submission(tmp_path, provider=provider, hooks=[hook]) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, hydrated = _records(drive_store)
        prepared = next(
            row.record for row in hydrated if row.record.kind == "op_prepared" and "/tool/" in str(row.record.operation_id)
        )
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, _ = _records(drive_store)
        retry_plan = next(
            row.record
            for row in rows
            if row.record.kind == "op_planned" and row.record.operation_id == prepared.operation_id and row.record.attempt == 2
        )
        assert len(provider.probe.submitted_plans) == 2
        first, retry = provider.probe.submitted_plans
        assert first.attempt == 1
        assert retry.attempt == retry_plan.attempt == 2
        assert retry.payload["request_digest"] == retry_plan.payload["request_digest"] == prepared.payload["request_digest"]
        assert retry.payload["command_digest"] == retry_plan.payload["command_digest"] == prepared.payload["command_digest"]
        assert (
            retry.payload["request"]["arguments"]["command"]
            == prepared.payload["request"]["arguments"]["command"]
            == hook.command
        )
        assert hook.calls == 1
        assert not any(
            row.record.kind == "op_prepared" and row.record.operation_id == prepared.operation_id and row.record.attempt == 2
            for row in rows
        )

        record_count = len(rows)
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)
        replayed_rows, _ = _records(drive_store)
        assert len(replayed_rows) == record_count
        assert len(provider.probe.submitted_plans) == 2
        assert hook.calls == 1


def test_w3_not_started_after_cancel_is_rejected_before_retry(tmp_path: Path) -> None:
    with (
        _lost_w3_submission(tmp_path) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, hydrated = _records(drive_store)
        plan = next(
            row.record for row in hydrated if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
        )
        turn = next(row.record for row in hydrated if row.record.kind == "turn_started")
        _push_control(sessions, drive_store, _control_item(plan, turn, "cancel"))
        lookup = _with_provider_lookup_input_id(_level_three_lookup_item(plan, turn, status=LookupStatus.NOT_STARTED))
        _push_lookup(sessions, drive_store, lookup, actor_subject_id="provider-recovery")

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        applied = next(
            row.record
            for row in hydrated
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == lookup.input_id
        )
        state, _, _ = read_state(drive_store, SESSION_ID)
        assert applied.payload["disposition"] == "rejected"
        assert applied.payload["reason"] == "provider_lookup_recovery_unavailable"
        assert state.turns[plan.turn_id].cancelled is True
        assert any(row.record.kind == "turn_ended" and row.record.payload["status"] == "cancelled" for row in rows)
        assert not any(
            row.record.operation_id == plan.operation_id
            and row.record.attempt == 2
            and row.record.kind in {"op_planned", "op_started"}
            for row in rows
        )
        assert not any(row.record.kind == "op_parked" and row.record.operation_id == plan.operation_id for row in rows)
        assert len(provider.probe.submitted_plans) == 1


def test_w3_not_started_after_suspend_keeps_turn_suspended_without_retry(tmp_path: Path) -> None:
    with (
        _lost_w3_submission(tmp_path) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, hydrated = _records(drive_store)
        plan = next(
            row.record for row in hydrated if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
        )
        turn = next(row.record for row in hydrated if row.record.kind == "turn_started")
        _push_control(sessions, drive_store, _control_item(plan, turn, "suspend"))
        lookup = _with_provider_lookup_input_id(_level_three_lookup_item(plan, turn, status=LookupStatus.NOT_STARTED))
        _push_lookup(sessions, drive_store, lookup, actor_subject_id="provider-recovery")

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        applied = next(
            row.record
            for row in hydrated
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == lookup.input_id
        )
        state, _, _ = read_state(drive_store, SESSION_ID)
        assert applied.payload["disposition"] == "rejected"
        assert applied.payload["reason"] == "provider_lookup_recovery_unavailable"
        assert state.active_turn_id == plan.turn_id
        assert state.turns[plan.turn_id].suspended is True
        assert state.phase == "suspended"
        assert not any(row.record.kind == "turn_ended" for row in rows)
        assert not any(
            row.record.operation_id == plan.operation_id
            and row.record.attempt == 2
            and row.record.kind in {"op_planned", "op_started"}
            for row in rows
        )
        assert not any(row.record.kind == "op_parked" and row.record.operation_id == plan.operation_id for row in rows)
        assert len(provider.probe.submitted_plans) == 1


def test_w3_not_started_after_suspend_resume_retries_one_pinned_attempt(tmp_path: Path) -> None:
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
        turn = next(row.record for row in hydrated if row.record.kind == "turn_started")

        _push_control(sessions, drive_store, _control_item(plan, turn, "suspend"))
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        state, _, _ = read_state(drive_store, SESSION_ID)
        assert state.turns[plan.turn_id].suspended is True
        assert provider.probe.lookup_calls == []

        _push_control(sessions, drive_store, _control_item(plan, turn, "resume"))
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        evidence = next(
            row.record
            for row in hydrated
            if row.record.kind == "input_applied"
            and row.record.payload["input"]["kind"] == "provider_evidence"
            and row.record.payload["input"]["payload"]["operation_id"] == plan.operation_id
        )
        unknown = next(
            row.record
            for row in hydrated
            if row.record.kind == "op_unknown" and row.record.operation_id == plan.operation_id and row.record.attempt == 1
        )
        retry_plans = [
            row.record
            for row in rows
            if row.record.kind == "op_planned" and row.record.operation_id == plan.operation_id and row.record.attempt == 2
        ]
        retry_starts = [
            row.record
            for row in rows
            if row.record.kind == "op_started" and row.record.operation_id == plan.operation_id and row.record.attempt == 2
        ]

        assert evidence.payload["disposition"] == "applied"
        assert unknown.payload["retry"] == "retry"
        assert len(retry_plans) == len(retry_starts) == 1
        assert provider.probe.lookup_calls == [
            (plan.payload["execution_identity"]["execution_id"], _expected_lookup_binding(plan, turn))
        ]
        assert len(provider.probe.submitted_plans) == 2
        first, retry = provider.probe.submitted_plans
        assert first.attempt == 1
        assert retry.attempt == 2
        for payload in (first.payload, retry.payload):
            assert payload["execution_identity"] == plan.payload["execution_identity"]
            assert payload["idempotency_key"] == plan.payload["idempotency_key"]

        submitted = len(provider.probe.submitted_plans)
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)
        assert len(provider.probe.submitted_plans) == submitted


def test_w3_rejected_evidence_after_suspend_resumes_the_running_command(tmp_path: Path) -> None:
    provider = HostCommandProvider()
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(binding, LookupStatus.RUNNING)

    with (
        _lost_w3_submission(tmp_path, provider=provider) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, hydrated = _records(drive_store)
        plan = next(
            row.record for row in hydrated if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
        )
        turn = next(row.record for row in hydrated if row.record.kind == "turn_started")
        _push_control(sessions, drive_store, _control_item(plan, turn, "suspend"))
        lookup = _with_provider_lookup_input_id(_level_three_lookup_item(plan, turn, status=LookupStatus.RUNNING))
        _push_lookup(sessions, drive_store, lookup, actor_subject_id="provider-recovery")

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        _, hydrated = _records(drive_store)
        rejected = next(
            row.record
            for row in hydrated
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == lookup.input_id
        )
        assert rejected.payload["disposition"] == "rejected"
        assert rejected.payload["reason"] == "provider_lookup_recovery_unavailable"
        assert provider.probe.lookup_calls == []

        _push_control(sessions, drive_store, _control_item(plan, turn, "resume"))
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        parked = next(
            row.record
            for row in hydrated
            if row.record.kind == "op_parked" and row.record.operation_id == plan.operation_id and row.record.attempt == 1
        )
        assert provider.probe.lookup_calls == [
            (plan.payload["execution_identity"]["execution_id"], _expected_lookup_binding(plan, turn))
        ]
        assert len(provider.probe.submitted_plans) == 1
        assert not any(
            row.record.operation_id == plan.operation_id
            and row.record.attempt == 2
            and row.record.kind in {"op_planned", "op_started"}
            for row in rows
        )
        assert parked.payload["phase"] == "after_dispatch"
        assert parked.payload["handle"]["job_id"] == "job-running"
        assert not any(
            row.record.kind == "op_unknown"
            and row.record.operation_id == plan.operation_id
            and row.record.payload["observation"]["code"] == "provider_lookup_replayed"
            for row in hydrated
        )


def test_w3_running_after_cancel_is_rejected_before_park(tmp_path: Path) -> None:
    with (
        _lost_w3_submission(tmp_path) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, hydrated = _records(drive_store)
        plan = next(
            row.record for row in hydrated if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
        )
        turn = next(row.record for row in hydrated if row.record.kind == "turn_started")
        _push_control(sessions, drive_store, _control_item(plan, turn, "cancel"))
        lookup = _with_provider_lookup_input_id(_level_three_lookup_item(plan, turn, status=LookupStatus.RUNNING))
        _push_lookup(sessions, drive_store, lookup, actor_subject_id="provider-recovery")

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        applied = next(
            row.record
            for row in hydrated
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == lookup.input_id
        )
        state, _, _ = read_state(drive_store, SESSION_ID)
        assert applied.payload["disposition"] == "rejected"
        assert applied.payload["reason"] == "provider_lookup_recovery_unavailable"
        assert state.turns[plan.turn_id].cancelled is True
        assert any(row.record.kind == "turn_ended" and row.record.payload["status"] == "cancelled" for row in rows)
        assert not any(row.record.kind == "op_parked" and row.record.operation_id == plan.operation_id for row in rows)
        assert not any(
            row.record.operation_id == plan.operation_id
            and row.record.attempt == 2
            and row.record.kind in {"op_planned", "op_started"}
            for row in rows
        )
        assert len(provider.probe.submitted_plans) == 1


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


@pytest.mark.parametrize(
    "variant",
    [
        "turn",
        "generation",
        "input_id",
        "actor",
    ],
)
def test_w3_rejected_provider_evidence_persisted_by_ingress_relooks_up_once(
    tmp_path: Path,
    variant: str,
) -> None:
    provider = HostCommandProvider()
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(binding, LookupStatus.RUNNING)

    with (
        _lost_w3_submission(tmp_path, provider=provider) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, before = _records(drive_store)
        plan = next(row.record for row in before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id))
        turn = next(row.record for row in before if row.record.kind == "turn_started")
        evidence = _invalid_durable_lookup_evidence(_level_three_lookup_item(plan, turn, status=LookupStatus.RUNNING), variant)
        _push_lookup(
            sessions,
            drive_store,
            evidence,
            actor_subject_id="bash" if variant == "actor" else "provider-recovery",
        )

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        applied = next(
            row.record
            for row in hydrated
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == evidence.input_id
        )
        expected_binding = _expected_lookup_binding(plan, turn)

        assert applied.payload["disposition"] == "rejected"
        assert provider.probe.lookup_calls == [(expected_binding.execution_id, expected_binding)]
        assert len(provider.probe.submitted_plans) == 1
        assert len(provider.probe.submitted_calls) == 1
        assert not any(
            row.record.operation_id == plan.operation_id
            and row.record.attempt == 2
            and row.record.kind in {"op_planned", "op_started"}
            for row in rows
        )


@pytest.mark.parametrize("variant", ["tenant", "session", "operation", "request_digest"])
def test_w3_tampered_canonical_binding_fails_before_recovery(tmp_path: Path, variant: str) -> None:
    provider = HostCommandProvider()
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(binding, LookupStatus.RUNNING)

    with (
        _lost_w3_submission(tmp_path, provider=provider) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, before = _records(drive_store)
        plan = next(row.record for row in before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id))
        turn = next(row.record for row in before if row.record.kind == "turn_started")
        evidence = _invalid_durable_lookup_evidence(
            _level_three_lookup_item(plan, turn, status=LookupStatus.RUNNING),
            variant,
        )
        _push_lookup(sessions, drive_store, evidence)

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        with pytest.raises(RecordError):
            drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        assert provider.probe.lookup_calls == []
        assert not any(
            row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == evidence.input_id
            for row in hydrated
        )
        assert not any(row.record.kind == "op_planned" and row.record.attempt == 2 for row in rows)


def test_w3_ignores_legacy_provider_evidence_skeleton_through_apply_input(tmp_path: Path) -> None:
    provider = HostCommandProvider()
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(binding, LookupStatus.RUNNING)

    with (
        _lost_w3_submission(tmp_path, provider=provider) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, before = _records(drive_store)
        plan = next(row.record for row in before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id))
        turn = next(row.record for row in before if row.record.kind == "turn_started")
        current = _level_three_lookup_item(plan, turn, status=LookupStatus.RUNNING)
        legacy = replace(
            current,
            input_id="provider-evidence/legacy",
            feature_level=1,
            payload={
                "operation_id": current.payload["operation_id"],
                "attempt": current.payload["attempt"],
                "request_digest": current.payload["request_digest"],
                "handle": {
                    "kind": "provider",
                    "provider": plan.payload["provider_binding"],
                    "job_id": "legacy-job",
                    "operation_id": current.payload["operation_id"],
                    "attempt": current.payload["attempt"],
                    "request_digest": current.payload["request_digest"],
                    "evidence": ["legacy-provider-evidence"],
                    "query_ref": None,
                    "cancel_ref": None,
                },
            },
        )
        _push_lookup(sessions, drive_store, legacy, actor_subject_id="provider-recovery")

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        applied = next(
            row.record
            for row in hydrated
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == legacy.input_id
        )
        assert applied.payload["disposition"] == "rejected"
        assert applied.payload["reason"] == "untrusted or mismatched provider evidence"
        assert len(provider.probe.lookup_calls) == 1
        assert len(provider.probe.submitted_plans) == 1
        assert not any(row.record.kind == "op_planned" and row.record.attempt == 2 for row in rows)


def test_w3_provider_evidence_actor_conflict_stops_current_attempt(tmp_path: Path) -> None:
    provider = HostCommandProvider()
    provider.probe.lookup_handler = lambda _execution_id, binding: _lookup_result(binding, LookupStatus.RUNNING)

    with (
        _lost_w3_submission(tmp_path, provider=provider) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, before = _records(drive_store)
        plan = next(row.record for row in before if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id))
        turn = next(row.record for row in before if row.record.kind == "turn_started")
        occupied = _with_provider_lookup_input_id(_level_three_lookup_item(plan, turn, status=LookupStatus.RUNNING))
        _push_lookup(sessions, drive_store, occupied, actor_subject_id="bash")

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        unknown = next(
            row.record
            for row in hydrated
            if row.record.kind == "op_unknown" and row.record.operation_id == plan.operation_id and row.record.attempt == 1
        )
        rejected = next(
            row.record
            for row in hydrated
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == occupied.input_id
        )

        assert rejected.payload["disposition"] == "rejected"
        assert unknown.payload["retry"] == "stop"
        assert unknown.payload["observation"]["code"] == "provider_lookup_recovery_unavailable"
        assert unknown.payload["observation"]["lookup_status"] == "running"
        assert unknown.payload["observation"]["evidence_input_id"] == occupied.input_id
        assert unknown.payload["observation"]["evidence_conflict"] is True
        assert len(provider.probe.lookup_calls) == 1
        assert len(provider.probe.submitted_plans) == 1
        assert not any(record.record.attempt == 2 for record in rows if "/tool/" in str(record.record.operation_id))
        assert len([row for row in rows if row.record.kind == "turn_ended"]) == 1


def test_w3_provider_evidence_push_status_has_no_overlapping_booleans(tmp_path: Path) -> None:
    from vv_agent.kernel import _driver_recovery

    with (
        _lost_w3_submission(tmp_path) as (_, _store, sessions, _provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, hydrated = _records(drive_store)
        plan = next(
            row.record for row in hydrated if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
        )
        turn = next(row.record for row in hydrated if row.record.kind == "turn_started")
        item = _with_provider_lookup_input_id(_level_three_lookup_item(plan, turn, status=LookupStatus.RUNNING))
        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        driver, lease = _direct_driver(drive_store, sessions, runtime)
        try:
            host = _recovery_phase._Host(driver)
            admitted = host.handle(_driver_recovery.PushProviderEvidence(item))
            replayed = host.handle(_driver_recovery.PushProviderEvidence(item))
            conflicting_item = replace(item, target_turn_id=f"{item.target_turn_id}/other")
            conflict = host.handle(_driver_recovery.PushProviderEvidence(conflicting_item))
        finally:
            drive_store.release(lease)

        assert admitted.status == "admitted"
        assert replayed.status == "replayed"
        assert conflict.status == "conflict"


def test_w3_public_push_cannot_squat_provider_recovery_input_identity(tmp_path: Path) -> None:
    from vv_agent.kernel import _driver_recovery

    with (
        _lost_w3_submission(tmp_path) as (_, _store, sessions, _provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, hydrated = _records(drive_store)
        plan = next(
            row.record for row in hydrated if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
        )
        turn = next(row.record for row in hydrated if row.record.kind == "turn_started")
        item = _with_provider_lookup_input_id(_level_three_lookup_item(plan, turn, status=LookupStatus.RUNNING))
        forged_actor = {
            "tenant_id": TENANT_ID,
            "subject_id": "provider-recovery",
            "subject_kind": "service",
            "attributes": {"role": "provider-recovery"},
        }

        with pytest.raises(Conflict, match="trusted recovery ingress"):
            sessions.push(SESSION_ID, item, actor=forged_actor)
        assert drive_store.read_input(SESSION_ID, item.input_id) is None

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        driver, lease = _direct_driver(drive_store, sessions, runtime)
        try:
            host = _recovery_phase._Host(driver)
            admitted = host.handle(_driver_recovery.PushProviderEvidence(item))
            replayed = host.handle(_driver_recovery.PushProviderEvidence(item))
        finally:
            drive_store.release(lease)

        stored = drive_store.read_input(SESSION_ID, item.input_id)
        assert stored is not None
        assert admitted.status == "admitted"
        assert replayed.status == "replayed"
        assert stored.item.target_turn_id == item.target_turn_id
        assert stored.item.generation == item.generation
        assert stored.item.payload["operation_id"] == item.payload["operation_id"]
        assert stored.item.payload["attempt"] == item.payload["attempt"]
        assert stored.item.payload["request_digest"] == item.payload["request_digest"]
        assert stored.item.actor["tenant_id"] == TENANT_ID
        assert stored.item.actor["subject_id"] == "provider-recovery"


def test_w3_public_push_rejects_reserved_provider_recovery_actor(tmp_path: Path) -> None:
    with _lost_w3_submission(tmp_path) as (_, _store, sessions, _provider):
        forged_actor = {
            "tenant_id": TENANT_ID,
            "subject_id": "provider-recovery",
            "subject_kind": "service",
            "attributes": {"role": "provider-recovery"},
        }
        item = InboxItem("reserved-actor", "user", {"content": "ordinary input"})

        with pytest.raises(Conflict, match="reserved for trusted recovery ingress"):
            sessions.push(SESSION_ID, item, actor=forged_actor)

        assert sessions.store.read_input(SESSION_ID, item.input_id) is None


@pytest.mark.parametrize("kind", ["provider_evidence", "steer"])
def test_w3_app_server_ingress_rejects_provider_recovery_inputs(tmp_path: Path, kind: str) -> None:
    from vv_agent.app_server import AppServer
    from vv_agent.session.app_server import SubmissionConflict

    with (
        _lost_w3_submission(tmp_path) as (_, _store, sessions, _provider),
        sessions._open_drive_store() as drive_store,
    ):
        state, _, _ = drive_store.read_state(SESSION_ID)
        actor = (
            None
            if kind == "provider_evidence"
            else {
                "tenant_id": TENANT_ID,
                "subject_id": "provider-recovery",
                "subject_kind": "service",
                "attributes": {"role": "provider-recovery"},
            }
        )
        app = AppServer(store=drive_store, content_key_provider=sessions.content_key_provider)
        try:
            with pytest.raises(SubmissionConflict, match="submission_conflict"):
                app.run_adapter.submit_input(
                    connection_id="test",
                    thread_id=SESSION_ID,
                    input_id="forged-provider-input",
                    expected_turn_id=state.active_turn_id,
                    kind=kind,
                    input=[{"type": "text", "text": "forged"}],
                    actor=actor,
                )
        finally:
            app.kernel.close()

        assert drive_store.read_input(SESSION_ID, "forged-provider-input") is None


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
        lookup = _with_provider_lookup_input_id(_level_three_lookup_item(plan, turn, status=LookupStatus.NOT_STARTED))
        _push_lookup(sessions, drive_store, lookup)

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        applied = next(
            row.record
            for row in hydrated
            if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == lookup.input_id
        )
        assert applied.payload["disposition"] == "rejected"
        assert applied.payload["reason"] == "untrusted or mismatched provider evidence"
        assert len(provider.probe.submitted_plans) == 1
        assert not any(row.record.kind == "op_planned" and row.record.attempt == 2 for row in rows)


@pytest.mark.parametrize("slot", ["provider_lookup", "input.actor.attributes"])
def test_w3_unconsumed_not_started_evidence_with_missing_content_fails_closed(
    tmp_path: Path,
    slot: str,
) -> None:
    with (
        _lost_w3_submission(tmp_path) as (_, _store, sessions, provider),
        sessions._open_drive_store() as drive_store,
    ):
        _, hydrated = _records(drive_store)
        plan = next(
            row.record for row in hydrated if row.record.kind == "op_planned" and "/tool/" in str(row.record.operation_id)
        )
        turn = next(row.record for row in hydrated if row.record.kind == "turn_started")
        item = _with_provider_lookup_input_id(
            _level_three_lookup_item(
                plan,
                turn,
                status=LookupStatus.NOT_STARTED,
            )
        )
        _push_lookup(sessions, drive_store, item, actor_subject_id="provider-recovery")
        drive_store._rows(
            "DELETE FROM sk_content_link WHERE tenant_id=%s AND session_id=%s AND input_id=%s AND slot=%s",
            (TENANT_ID, SESSION_ID, item.input_id, slot),
        )

        runtime = sessions._build_runtime_for_claim(drive_store, SESSION_ID)
        with pytest.raises(MissingContentLink):
            drive(drive_store, SESSION_ID, runtime=runtime)

        rows, hydrated = _records(drive_store)
        inbox = drive_store._rows(
            "SELECT consumed_seq FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND input_id=%s",
            (TENANT_ID, SESSION_ID, item.input_id),
        )
        assert inbox == [(None,)]
        assert not any(
            row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == item.input_id for row in hydrated
        )
        assert provider.probe.lookup_calls == []
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
