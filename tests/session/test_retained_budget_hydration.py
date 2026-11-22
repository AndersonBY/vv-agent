"""Core retained-budget hydration rejects malformed encrypted producer values."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vv_agent.kernel._budget_values import BudgetUsageSnapshot
from vv_agent.kernel._content_publication import (
    link_declared_record_content,
    make_record_before_publication,
    publish_record_content,
    publish_turn_started,
    resolve_record_content_links,
)
from vv_agent.kernel._content_reader import DurableContentReader, owner_scope, record_owner
from vv_agent.kernel._content_runtime import ContentRuntimeBinding, SessionContentWriter, wrap_session_content_store
from vv_agent.kernel._driver import CoreDriver, DriverLeaseScope
from vv_agent.kernel._driver_outcomes import live_budget_inputs_steps
from vv_agent.kernel._hydration import KeyUnavailable, TypedSchemaFailure, reference_digest
from vv_agent.kernel._records import SessionSpec, digest, make_record
from vv_agent.kernel._retained_budget import retained_budget_inputs
from vv_agent.session.content_store import AESGCMCipher, LocalKeyProvider
from vv_agent.session.sqlite import SQLiteContentStore, SQLiteStore

MASTER_KEY = b"b" * 32
TENANT = "default"
SESSION = "retained-budget"
TURN = "retained-budget/turn/1"

VALID_BUDGET = {
    "max_total_tokens": 100,
    "max_uncached_input_tokens": None,
    "max_tool_calls": None,
    "max_tool_calls_by_name": {},
    "max_wall_time_ms": None,
    "max_host_cost": None,
    "unavailable_metric_policy": "continue_and_mark",
}

DEFINITION = {
    "agent_name": "retained-budget",
    "memory_settings": {},
    "child_tools": [],
    "model_binding": {
        "backend": "scripted",
        "model": "scripted",
        "context_length": None,
        "max_output_tokens": None,
    },
    "capabilities": {},
    "task": {
        "task_id": "retained-budget/task",
        "model": "scripted",
        "prompt_bundle": {"sections": [{"role": "system", "content": "Work."}], "system": "Work."},
        "user_prompt": "go",
    },
    "tools": [],
}


class _BackendContentReader:
    """Expose backend failures without the durable reader's own normalization."""

    def __init__(self, store: SQLiteContentStore, *, tenant_id: str, session_id: str) -> None:
        self._store = store
        self.binding = (tenant_id, session_id)

    def read(self, tenant_id: str, session_id: str, owner: str, slot: str, ref: Any) -> bytes:
        assert (tenant_id, session_id) == self.binding
        owner_kind, owner_value = owner_scope(owner)
        self._store.assert_link(tenant_id, session_id, ref, slot, **{owner_kind: owner_value})
        return self._store.get(tenant_id, session_id, ref)


@contextmanager
def _open_driver(
    tmp_path: Path,
    *,
    budget: dict[str, Any] | None = None,
    boundary_usage: Any = ...,
    observation_usage: Any = ...,
    model_usage: Any = ...,
) -> Iterator[tuple[CoreDriver, Any, LocalKeyProvider]]:
    provider = LocalKeyProvider(MASTER_KEY, state_path=tmp_path / "keys.sqlite3")
    try:
        with SQLiteStore.standalone(str(tmp_path / "sessions.sqlite")) as raw_store:
            raw_store.install_schema()
            raw_store.configure_input_admission(allow_unsigned=True)
            binding = ContentRuntimeBinding(provider, AESGCMCipher())
            store = wrap_session_content_store(raw_store, binding)
            with store.atomic() as tx:
                writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=SESSION)
                writer.create_session(
                    lambda content: SessionSpec(
                        SESSION,
                        content.service_principal("retained-budget-host")[0],
                        content.put_json({"workspace": "retained-budget"}).to_dict(),
                        attributes={"purpose": "retained-budget"},
                    ),
                    consumers=(),
                )

            lease = store.acquire(SESSION, owner="retained-budget", ttl_ms=60_000)
            assert lease is not None
            try:
                with store.atomic() as tx:
                    writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=SESSION)
                    turn_content = publish_turn_started(
                        writer,
                        DEFINITION,
                        VALID_BUDGET if budget is None else budget,
                    )
                    turn = make_record(
                        "turn_started",
                        session_id=SESSION,
                        turn_id=TURN,
                        payload={
                            "input_ids": [],
                            "definition": turn_content.definition.to_dict(),
                            "definition_digest": turn_content.definition_digest,
                            "handler_version": "1",
                            "budget": turn_content.budget.to_dict(),
                            "binding": "task",
                            "generation": 1,
                            "policy_digest": None,
                            "requirements": {"level": 1, "handler_version": "1", "capabilities": []},
                            "tokenizer_binding": None,
                            "environment": None,
                        },
                    )
                    receipt = tx.append(
                        SESSION,
                        lease=lease,
                        expected_seq=1,
                        commit_id="retained-budget-turn",
                        records=(turn,),
                    )
                    record_seq = receipt.record_sequences[0][1]
                    writer.link_record(record_seq, turn_content.definition, "definition")
                    writer.link_record(record_seq, turn_content.budget, "budget")

                next_seq = 2
                if boundary_usage is not ...:
                    boundary = make_record_before_publication(
                        "boundary_recorded",
                        session_id=SESSION,
                        turn_id=TURN,
                        payload={
                            "stage": "budget",
                            "boundary_id": "malformed-boundary",
                            "source_operation_id": None,
                            "source_digest": None,
                            "data": {"tool_names": [], "usage": boundary_usage, "exhaustion": None},
                        },
                    )
                    next_seq = _append_published(store, lease, (boundary,), next_seq)
                if observation_usage is not ...:
                    observation = make_record_before_publication(
                        "usage_observed",
                        session_id=SESSION,
                        turn_id=TURN,
                        payload={
                            "meter_id": "host-meter",
                            "observation": 1,
                            "mode": "cumulative",
                            "usage": observation_usage,
                            "source": {"provider": "host"},
                        },
                    )
                    next_seq = _append_published(store, lease, (observation,), next_seq)
                if model_usage is not ...:
                    _append_model(store, lease, model_usage, next_seq)

                reader = DurableContentReader(
                    SQLiteContentStore(raw_store, provider),
                    tenant_id=TENANT,
                    session_id=SESSION,
                )
                driver = CoreDriver(
                    raw_store,
                    SESSION,
                    cast(DriverLeaseScope, SimpleNamespace(lease=lease)),
                    fold_factory=raw_store.fold_factory,
                    clock=raw_store.clock,
                    observe_cost=lambda: None,
                    hook=lambda _stage, _record: None,
                    emit_phase=lambda _duration, _value: None,
                    content_reader=reader,
                )
                yield driver, turn_content.budget, provider
            finally:
                store.release(lease)
    finally:
        provider.close()


def _append_published(store: Any, lease: Any, records: tuple[Any, ...], expected_seq: int) -> int:
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=SESSION)
        publication = publish_record_content(writer, records)
        receipt = tx.append(
            SESSION,
            lease=lease,
            expected_seq=expected_seq,
            commit_id=f"retained-budget-{expected_seq}",
            records=publication.records,
        )
        links = resolve_record_content_links(publication.links, receipt.record_sequences)
        link_declared_record_content(writer, links)
    return expected_seq + len(records)


def _append_model(store: Any, lease: Any, usage: Any, expected_seq: int) -> int:
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=TENANT, session_id=SESSION)
        request_ref = writer.put_json({"id": "call", "name": "model", "arguments": {}, "messages": [], "metadata": {}})
        records = _model_records(usage, request_ref, reference_digest(request_ref))
        publication = publish_record_content(writer, records)
        receipt = tx.append(
            SESSION,
            lease=lease,
            expected_seq=expected_seq,
            commit_id=f"retained-budget-{expected_seq}",
            records=publication.records,
        )
        links = resolve_record_content_links(publication.links, receipt.record_sequences)
        link_declared_record_content(writer, links)
    return expected_seq + len(records)


def _model_records(usage: Any, request_ref: Any, request_digest: str) -> tuple[Any, ...]:
    operation_id = "retained-budget/model/1"
    plan = make_record_before_publication(
        "op_planned",
        session_id=SESSION,
        turn_id=TURN,
        operation_id=operation_id,
        attempt=1,
        payload={
            "op_kind": "model",
            "purpose": "primary",
            "request": request_ref,
            "request_digest": request_digest,
            "context_version": "1",
            "dependencies": [],
            "tool": None,
            "idempotency_key": None,
            "budget_admission": {},
            "not_before_ms": None,
            "provider_binding": "scripted",
            "consumed_unknowns": [],
            "policy_decision": None,
        },
    )
    started = make_record(
        "op_started",
        session_id=SESSION,
        turn_id=TURN,
        operation_id=operation_id,
        attempt=1,
        payload={"dispatch_id": "dispatch", "authorization_version": "1", "epoch": 1, "mode": "sync"},
    )
    result = {"status": "completed"}
    completed = make_record_before_publication(
        "op_completed",
        session_id=SESSION,
        turn_id=TURN,
        operation_id=operation_id,
        attempt=1,
        payload={
            "result": result,
            "result_digest": digest(result),
            "usage": usage,
            "shared_state": None,
            "evidence": ["dispatch"],
            "execution_started": True,
            "context": "normal",
            "request_digest": request_digest,
            "provider_binding": "scripted",
            "policy_decision": None,
        },
    )
    return plan, started, completed


@pytest.mark.parametrize(
    "budget",
    [
        {},
        {k: v for k, v in VALID_BUDGET.items() if k != "max_total_tokens"},
        VALID_BUDGET | {"future_dimension": 1},
    ],
    ids=["empty", "missing", "unknown"],
)
def test_core_driver_budget_gate_rejects_non_closed_limits(tmp_path: Path, budget: dict[str, Any]) -> None:
    with _open_driver(tmp_path, budget=budget) as (driver, _budget_ref, _provider):
        steps = live_budget_inputs_steps(driver)
        with pytest.raises(TypedSchemaFailure):
            next(steps)


@pytest.mark.parametrize(
    "boundary_usage",
    [
        [],
        BudgetUsageSnapshot().to_dict() | {"future_snapshot_field": 1},
    ],
    ids=["wrong-shape", "unknown-field"],
)
def test_core_driver_budget_snapshot_requires_exact_retained_schema(tmp_path: Path, boundary_usage: Any) -> None:
    with _open_driver(tmp_path, boundary_usage=boundary_usage) as (driver, _budget_ref, _provider):
        tid = driver.fold.state.active_turn_id
        assert tid is not None
        with pytest.raises(TypedSchemaFailure):
            retained_budget_inputs(driver.fold.state, tid, hydrator=driver)


@pytest.mark.parametrize("usage_field", ["observation", "model"])
def test_core_driver_usage_rejects_non_object_provider_usage(tmp_path: Path, usage_field: str) -> None:
    options: dict[str, Any] = {f"{usage_field}_usage": []}
    with _open_driver(tmp_path, **options) as (driver, _budget_ref, _provider):
        tid = driver.fold.state.active_turn_id
        assert tid is not None
        with pytest.raises(TypedSchemaFailure):
            retained_budget_inputs(driver.fold.state, tid, hydrator=driver)


def test_core_driver_accepts_opaque_provider_usage_object(tmp_path: Path) -> None:
    opaque_usage = {"input_tokens": 7, "completion_tokens": 3, "provider_extension": {"cache": "warm"}}
    with _open_driver(tmp_path, observation_usage=opaque_usage, model_usage=opaque_usage) as (
        driver,
        _budget_ref,
        _provider,
    ):
        tid = driver.fold.state.active_turn_id
        assert tid is not None
        inputs = retained_budget_inputs(driver.fold.state, tid, hydrator=driver)
        assert inputs is not None
        assert inputs.model_usages == (opaque_usage,)


def test_core_driver_normalizes_backend_key_unavailable(tmp_path: Path) -> None:
    with _open_driver(tmp_path) as (driver, budget_ref, provider):
        turn = driver.fold.state.turns[TURN].start
        driver.content_reader = _BackendContentReader(
            SQLiteContentStore(cast(SQLiteStore, driver.store), provider),
            tenant_id=TENANT,
            session_id=SESSION,
        )
        provider.destroy(TENANT, budget_ref.key_id)
        with pytest.raises(KeyUnavailable):
            driver.read_content_json(record_owner(driver.stored_record(turn).seq), "budget", budget_ref)
