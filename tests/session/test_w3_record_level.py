from __future__ import annotations

import json
import os
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from vv_agent.kernel._boundary_hydration import (
    SessionHydration,
    hydrate_session_input,
    hydrate_session_input_payload,
    hydrate_session_records,
)
from vv_agent.kernel._content_publication import (
    link_declared_record_content,
    make_retained_record_before_publication,
    publish_record_content,
    resolve_record_content_links,
)
from vv_agent.kernel._content_runtime import SessionContentWriter
from vv_agent.kernel._hydration import MissingContentLink, TypedSchemaFailure
from vv_agent.kernel._records import InboxItem, Record, RecordError, digest, make_record, validate_provider_lookup_evidence
from vv_agent.kernel._sql import UnsupportedFeatureLevel, stored_input_proof
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.environments import execution_id_for
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver

_MASTER_KEY = b"w" * 32
_TENANT = "tenant-a"
_SESSION = "sess-1"
_TURN = "turn-1"
_INPUT = "provider-evidence-1"


def _contract_source() -> Path:
    configured = os.environ.get("VV_AGENT_CONTRACT_SOURCE")
    if not configured:
        pytest.skip("VV_AGENT_CONTRACT_SOURCE is not configured")
    source = Path(configured)
    if not source.is_dir():
        pytest.fail(f"VV_AGENT_CONTRACT_SOURCE does not exist: {source}")
    return source


def _ref(label: str = "content") -> dict[str, Any]:
    payload = label.encode()
    return {
        "content_id": f"content/test/{label}",
        "mac": sha256(payload).hexdigest(),
        "size": len(payload),
        "key_id": "key/test",
        "labels": {"source": [], "classification": "internal"},
    }


def _principal(tenant_id: str = "tenant-a") -> dict[str, Any]:
    return {
        "tenant_id": tenant_id,
        "subject_id": "provider",
        "subject_kind": "service",
        "attributes": _ref("principal"),
    }


def _fixture(source: Path, name: str) -> dict[str, Any]:
    return json.loads((source / "fixtures" / name).read_text(encoding="utf-8"))


def _environment(w3: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(w3["environment"]) | {
        "descriptor": _ref("environment"),
        "operational_policy": _ref("operational-policy"),
    }


def _turn_started(
    w3: dict[str, Any],
    *,
    environment: dict[str, Any] | None = None,
) -> Record:
    definition = _ref("definition")
    return make_record(
        "turn_started",
        session_id="sess-1",
        turn_id="turn-1",
        payload={
            "input_ids": [],
            "definition": definition,
            "definition_digest": digest(definition),
            "handler_version": "handler-v3",
            "budget": _ref("budget"),
            "binding": "task",
            "generation": 1,
            "policy_digest": None,
            "requirements": {"level": 3, "handler_version": "handler-v3", "capabilities": []},
            "tokenizer_binding": None,
            "environment": _environment(w3) if environment is None else environment,
        },
        feature_level=3,
    )


def _op_planned(w3: dict[str, Any]) -> Record:
    request = _ref("request")
    identity = deepcopy(w3["identity_vectors"][0]["identity"])
    return make_record(
        "op_planned",
        session_id=identity["session_id"],
        turn_id="turn-1",
        operation_id=identity["operation_id"],
        attempt=1,
        payload={
            "op_kind": "tool",
            "purpose": None,
            "request": request,
            "request_digest": digest(request),
            "context_version": "context-v1",
            "dependencies": [],
            "tool": request,
            "idempotency_key": identity["operation_key"],
            "budget_admission": _ref("budget-admission"),
            "not_before_ms": None,
            "provider_binding": "provider-remote-1",
            "consumed_unknowns": [],
            "policy_decision": None,
            "execution_identity": identity,
            "command_digest": w3["command_digest_vectors"][0]["command_digest"],
        },
        feature_level=3,
    )


def _op_prepared(w3: dict[str, Any]) -> Record:
    plan = _op_planned(w3)
    request = _ref("prepared-request")
    payload = {
        "request": request,
        "request_digest": digest(request),
        "op_kind": "tool",
        "tool": _ref("prepared-tool"),
        "idempotency_key": plan.payload["idempotency_key"],
        "provider_binding": plan.payload["provider_binding"],
        "hook_result": None,
        "shared_state": _ref("prepared-shared-state"),
        "command_digest": w3["preparation_binding"]["prepared_command_digest"],
    }
    return make_record(
        "op_prepared",
        session_id=plan.session_id,
        turn_id=plan.turn_id,
        operation_id=plan.operation_id,
        attempt=plan.attempt,
        payload=payload,
        feature_level=3,
    )


def _lookup_item(w3_result: dict[str, Any], *, actor: dict[str, Any] | None = None) -> InboxItem:
    binding = w3_result["binding"]
    return InboxItem(
        _INPUT,
        "provider_evidence",
        {
            "operation_id": binding["operation_id"],
            "attempt": binding["attempt"],
            "request_digest": binding["request_digest"],
            "provider_lookup": deepcopy(w3_result),
        },
        actor=_principal(binding["tenant_id"]) if actor is None else actor,
        target_turn_id=_TURN,
        feature_level=3,
    )


def _lookup_wire_item(w3_result: dict[str, Any], *, label: str = "provider-lookup") -> InboxItem:
    item = _lookup_item(w3_result)
    return replace(item, payload=item.payload | {"provider_lookup": _ref(label)})


def _encode_mutation(value: dict[str, Any]) -> bytes:
    return Record(**value).encode()


def _publish_provider_lookup(tmp_path: Path, result: dict[str, Any]) -> Path:
    database = tmp_path / "provider-lookup.sqlite"
    provider = LocalKeyProvider(_MASTER_KEY, state_path=tmp_path / "provider-lookup-keys.sqlite3")
    with SQLiteStore.standalone(str(database)) as raw_store:
        raw_store.install_schema()
        raw_store.configure_input_admission(allow_unsigned=True)
        raw_store._rows("UPDATE sk_settings SET write_level=3 WHERE singleton=1")
        raw_store.configure_reader(level=3)
        with closing(SessionDriver(store=raw_store, tenant_id=_TENANT, content_key_provider=provider, worker_level=3)) as driver:
            driver.create(_SESSION, "/workspace")
            driver.push(
                _SESSION,
                replace(_lookup_item(result), actor=None),
                actor={
                    "tenant_id": _TENANT,
                    "subject_id": result["binding"]["provider_id"],
                    "subject_kind": "service",
                    "attributes": {},
                },
            )
    return database


def test_level_3_record_and_inbox_shapes_round_trip_from_canonical_vectors() -> None:
    source = _contract_source()
    w3 = _fixture(source, "execution_worker_w3.json")
    turn = _turn_started(w3)
    plan = _op_planned(w3)
    prepared = _op_prepared(w3)
    lookup = InboxItem(**w3["lookup_storage_boundary"]["retained_evidence"])

    assert Record.parse(turn.encode()) == turn
    assert Record.parse(plan.encode()) == plan
    assert Record.parse(prepared.encode()) == prepared
    assert InboxItem.parse(lookup.encode()) == lookup
    assert turn.feature_level == plan.feature_level == prepared.feature_level == lookup.feature_level == 3

    record_schema = _fixture(source, "session_record.schema.json")
    inbox_schema = _fixture(source, "session_inbox.schema.json")
    Draft202012Validator(record_schema).validate(turn.to_dict())
    Draft202012Validator(record_schema).validate(plan.to_dict())
    Draft202012Validator(record_schema).validate(prepared.to_dict())
    Draft202012Validator(inbox_schema).validate(lookup.to_dict())


def test_level_2_rejects_level_3_kinds_and_fields() -> None:
    source = _contract_source()
    w3 = _fixture(source, "execution_worker_w3.json")
    plan = _op_planned(w3).to_dict()
    plan["feature_level"] = 2

    with pytest.raises(RecordError):
        Record(**plan).encode()

    lookup = _lookup_wire_item(w3["lookup_results"][0]["result"]).to_dict()
    lookup["feature_level"] = 2
    lookup["requirements"] = {"level": 2, "handler_version": "handler-v2", "capabilities": []}
    with pytest.raises(RecordError, match="invalid session schema"):
        InboxItem(**lookup).encode()


def test_level_3_op_planned_only_accepts_tool_operations() -> None:
    source = _contract_source()
    w3 = _fixture(source, "execution_worker_w3.json")
    wire = _op_planned(w3).to_dict()
    wire["payload"].update(op_kind="model", purpose="primary")

    with pytest.raises(RecordError):
        _encode_mutation(wire)


@pytest.mark.parametrize("key", [None, "other-key"])
def test_level_3_op_planned_requires_the_stable_key(key: str | None) -> None:
    w3 = _fixture(_contract_source(), "execution_worker_w3.json")
    wire = _op_planned(w3).to_dict()
    wire["payload"]["idempotency_key"] = key
    with pytest.raises(RecordError, match="planned idempotency_key"):
        _encode_mutation(wire)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("tenant_id",), "tenant-b"),
        (("environment_id",), "environment-b"),
        (("operation_id",), "operation-b"),
        (("operation_key",), "sess-1/operation-b"),
    ],
)
def test_execution_identity_tampering_is_rejected(path: tuple[str, ...], value: str) -> None:
    source = _contract_source()
    w3 = _fixture(source, "execution_worker_w3.json")
    wire = _op_planned(w3).to_dict()
    target = wire["payload"]["execution_identity"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(RecordError):
        _encode_mutation(wire)


def test_level_3_environment_capabilities_are_sorted_unique_and_closed() -> None:
    source = _contract_source()
    w3 = _fixture(source, "execution_worker_w3.json")

    unordered = _environment(w3)
    unordered["environment_capabilities"] = list(reversed(unordered["environment_capabilities"]))
    with pytest.raises(RecordError):
        _turn_started(w3, environment=unordered)

    duplicate = _environment(w3)
    duplicate["environment_capabilities"] = ["command.execute", "command.execute"]
    with pytest.raises(RecordError):
        _turn_started(w3, environment=duplicate)

    extra = _environment(w3) | {"unknown": True}
    with pytest.raises(RecordError):
        _turn_started(w3, environment=extra)

    bad_digest = _environment(w3) | {"capabilities_digest": "0" * 64}
    with pytest.raises(RecordError, match="capabilities_digest"):
        _turn_started(w3, environment=bad_digest)


@pytest.mark.parametrize("name", ["not_started", "running", "exited", "unknown"])
def test_provider_lookup_status_and_outcome_follow_canonical_four_states(name: str) -> None:
    source = _contract_source()
    w3 = _fixture(source, "execution_worker_w3.json")
    result = next(vector["result"] for vector in w3["lookup_results"] if vector["result"]["status"] == name)

    item = _lookup_item(result)
    validate_provider_lookup_evidence(_lookup_wire_item(result), item.payload["provider_lookup"], session_id=_SESSION)
    with pytest.raises(RecordError, match="invalid session schema"):
        item.encode()
    with pytest.raises(TypedSchemaFailure, match="requires a ContentRef"):
        hydrate_session_input_payload(item, None)


@pytest.mark.parametrize(
    ("source_status", "outcome_kind"),
    [
        ("not_started", "Unknown"),
        ("running", "Definitive"),
        ("exited", "Accepted"),
        ("unknown", "Definitive"),
    ],
)
def test_provider_lookup_rejects_mismatched_status_and_outcome(source_status: str, outcome_kind: str) -> None:
    source = _contract_source()
    w3 = _fixture(source, "execution_worker_w3.json")
    result = next(vector["result"] for vector in w3["lookup_results"] if vector["result"]["status"] == source_status)
    item = _lookup_item(result)
    replacement = next(
        vector["result"]["outcome"]
        for vector in w3["lookup_results"]
        if vector["result"]["outcome"] is not None and vector["result"]["outcome"]["kind"] == outcome_kind
    )
    logical = item.payload["provider_lookup"] | {"outcome": deepcopy(replacement)}

    with pytest.raises(RecordError, match="requires"):
        validate_provider_lookup_evidence(item, logical, session_id=_SESSION)


def test_provider_lookup_closed_fields_and_binding_are_enforced() -> None:
    source = _contract_source()
    w3 = _fixture(source, "execution_worker_w3.json")
    item = _lookup_item(w3["lookup_results"][1]["result"])

    lookup = item.payload["provider_lookup"]
    with pytest.raises(RecordError, match="invalid session schema"):
        validate_provider_lookup_evidence(item, lookup | {"unknown": True}, session_id=_SESSION)

    wrong_operation = lookup | {"binding": lookup["binding"] | {"operation_id": "other-operation"}}
    with pytest.raises(RecordError, match="binding does not match"):
        validate_provider_lookup_evidence(item, wrong_operation, session_id=_SESSION)


@pytest.mark.parametrize(
    ("status", "outcome_path", "value", "message"),
    [
        ("running", ("handle", "evidence"), [], "lost acceptance evidence"),
        ("exited", ("evidence",), [], "lost terminal evidence"),
        ("unknown", ("reason",), " ", "reason is empty"),
    ],
)
def test_provider_lookup_outcome_requires_its_own_evidence(status, outcome_path, value, message) -> None:
    w3 = _fixture(_contract_source(), "execution_worker_w3.json")
    result = next(vector["result"] for vector in w3["lookup_results"] if vector["result"]["status"] == status)
    outcome = result["outcome"]
    target = outcome["handle"] if outcome_path[0] == "handle" else outcome
    target[outcome_path[-1]] = value
    with pytest.raises(RecordError, match=message):
        validate_provider_lookup_evidence(_lookup_wire_item(result), result, session_id=_SESSION)


def test_canonical_execution_fixture_and_identity_algorithm_cross_validate() -> None:
    source = _contract_source()
    w3 = _fixture(source, "execution_worker_w3.json")
    schema = _fixture(source, "execution_worker_w3.schema.json")
    Draft202012Validator(schema).validate(w3)
    assert w3["feature_level"] == 3

    for vector in w3["identity_vectors"]:
        identity = vector["identity"]
        assert (
            execution_id_for(
                identity["tenant_id"],
                identity["environment_id"],
                identity["operation_key"],
            )
            == identity["execution_id"]
        )


def test_decrypted_lookup_must_name_the_session_that_owns_the_inbox(tmp_path: Path) -> None:
    w3 = _fixture(_contract_source(), "execution_worker_w3.json")
    result = next(vector["result"] for vector in w3["lookup_results"] if vector["result"]["status"] == "not_started")
    binding = result["binding"]
    wrong_key = f"other-session/{binding['operation_id']}"
    binding.update(
        session_id="other-session",
        idempotency_key=wrong_key,
        execution_id=execution_id_for(_TENANT, binding["environment_id"], wrong_key),
    )
    database = _publish_provider_lookup(tmp_path, result)
    provider = LocalKeyProvider(_MASTER_KEY, state_path=tmp_path / "provider-lookup-keys.sqlite3")
    with SQLiteStore.standalone(str(database)) as raw_store:
        raw_store.configure_input_admission(allow_unsigned=True)
        raw_store.configure_reader(level=3)
        with closing(SessionDriver(store=raw_store, tenant_id=_TENANT, content_key_provider=provider, worker_level=3)) as driver:
            item = driver.store.peek_inbox(_SESSION)[0].item
            hydration = SessionHydration.from_store(driver.store, tenant_id=_TENANT, session_id=_SESSION)
            with pytest.raises(RecordError, match="lookup session does not match"):
                hydrate_session_input_payload(item, hydration)


def test_provider_lookup_reopens_through_original_inbox_owner_and_slot(tmp_path: Path) -> None:
    w3 = _fixture(_contract_source(), "execution_worker_w3.json")
    result = w3["lookup_storage_boundary"]["content"]["logical_result"]
    database = _publish_provider_lookup(tmp_path, result)
    provider = LocalKeyProvider(_MASTER_KEY, state_path=tmp_path / "provider-lookup-keys.sqlite3")
    with SQLiteStore.standalone(str(database)) as raw_store:
        raw_store.configure_input_admission(allow_unsigned=True)
        with closing(SessionDriver(store=raw_store, tenant_id=_TENANT, content_key_provider=provider, worker_level=3)) as driver:
            raw_store.configure_reader(level=2)
            with pytest.raises(UnsupportedFeatureLevel):
                driver.store.peek_inbox(_SESSION)
            raw_store.configure_reader(level=3)
            inbox = driver.store.peek_inbox(_SESSION)[0]
            item = inbox.item
            assert set(item.payload) == {"operation_id", "attempt", "request_digest", "provider_lookup"}
            assert result["outcome"]["handle"]["job_id"].encode() not in database.read_bytes()
            hydration = SessionHydration.from_store(driver.store, tenant_id=_TENANT, session_id=_SESSION)
            assert hydrate_session_input(item, hydration).payload["provider_lookup"] == result

            with pytest.raises(MissingContentLink):
                hydrate_session_input(replace(item, input_id="other-input"), hydration)

            applied = make_record(
                "input_applied",
                session_id=_SESSION,
                turn_id=_TURN,
                payload={
                    "input": item.to_dict(),
                    "input_digest": item.digest,
                    "input_proof": stored_input_proof(inbox),
                    "disposition": "rejected",
                    "reason": None,
                    "target_operation_id": None,
                    "target_wait_id": None,
                    "position": "between_actions",
                },
            )
            lease = driver.store.acquire(_SESSION, owner="lookup-validation", ttl_ms=10_000)
            assert lease is not None
            with driver.store.atomic() as tx:
                writer = SessionContentWriter.bind(tx, tenant_id=_TENANT, session_id=_SESSION)
                publication = publish_record_content(writer, (applied,))
                receipt = tx.append(
                    _SESSION,
                    lease=lease,
                    expected_seq=1,
                    commit_id="apply-lookup-validation",
                    records=publication.records,
                    consume_input_ids=(item.input_id,),
                )
                link_declared_record_content(writer, resolve_record_content_links(publication.links, receipt.record_sequences))
            stored = driver.store.read(_SESSION, after_seq=1).records[0]
            assert stored.record.payload["input"] == item.to_dict()
            assert {link.slot for link in publication.links} == {"input.actor.attributes"}
            hydrated = hydrate_session_records((stored,), hydration)[0].record
            assert hydrated.payload["input"]["payload"]["provider_lookup"] == result
            assert hydrated.encode() == stored.record.encode()

            raw_store._rows(
                "UPDATE sk_content_link SET slot='provider.result' "
                "WHERE tenant_id=%s AND session_id=%s AND input_id=%s AND slot='provider_lookup'",
                (_TENANT, _SESSION, _INPUT),
            )
            with pytest.raises(MissingContentLink):
                hydrate_session_input(item, hydration)
            with pytest.raises(MissingContentLink):
                hydrate_session_records((stored,), hydration)


def test_retained_level_3_preparation_keeps_its_feature_level() -> None:
    w3 = _fixture(_contract_source(), "execution_worker_w3.json")
    prepared = _op_prepared(w3)
    rebuilt = make_retained_record_before_publication(
        prepared.kind,
        session_id=prepared.session_id,
        turn_id=prepared.turn_id,
        operation_id=prepared.operation_id,
        attempt=prepared.attempt,
        payload=prepared.payload,
        feature_level=prepared.feature_level,
    )
    assert rebuilt.feature_level == 3
    assert rebuilt.payload["command_digest"] == prepared.payload["command_digest"]
