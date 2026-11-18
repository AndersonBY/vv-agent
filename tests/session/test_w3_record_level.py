from __future__ import annotations

import json
import os
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from vv_agent.kernel._records import InboxItem, Record, RecordError, digest, make_record
from vv_agent.session.environments import execution_id_for


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


def _lookup_item(w3_result: dict[str, Any]) -> InboxItem:
    binding = w3_result["binding"]
    return InboxItem(
        "provider-evidence-1",
        "provider_evidence",
        {
            "operation_id": binding["operation_id"],
            "attempt": binding["attempt"],
            "request_digest": binding["request_digest"],
            "lookup": deepcopy(w3_result),
        },
        actor=_principal(binding["tenant_id"]),
        target_turn_id="turn-1",
        feature_level=3,
    )


def _encode_mutation(value: dict[str, Any]) -> bytes:
    return Record(**value).encode()


def test_level_3_record_and_inbox_shapes_round_trip_from_canonical_vectors() -> None:
    source = _contract_source()
    w3 = _fixture(source, "execution_worker_w3.json")
    turn = _turn_started(w3)
    plan = _op_planned(w3)
    lookup = _lookup_item(w3["lookup_results"][1]["result"])

    assert Record.parse(turn.encode()) == turn
    assert Record.parse(plan.encode()) == plan
    assert InboxItem.parse(lookup.encode()) == lookup
    assert turn.feature_level == plan.feature_level == lookup.feature_level == 3

    record_schema = _fixture(source, "session_record.schema.json")
    inbox_schema = _fixture(source, "session_inbox.schema.json")
    Draft202012Validator(record_schema).validate(turn.to_dict())
    Draft202012Validator(record_schema).validate(plan.to_dict())
    Draft202012Validator(inbox_schema).validate(lookup.to_dict())


def test_level_2_rejects_level_3_kinds_and_fields() -> None:
    source = _contract_source()
    w3 = _fixture(source, "execution_worker_w3.json")
    plan = _op_planned(w3).to_dict()
    plan["feature_level"] = 2

    with pytest.raises(RecordError):
        Record(**plan).encode()

    lookup = _lookup_item(w3["lookup_results"][0]["result"]).to_dict()
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
    assert InboxItem.parse(item.encode()) == item
    expected_outcome = {
        "not_started": None,
        "running": "Accepted",
        "exited": "Definitive",
        "unknown": "Unknown",
    }[name]
    assert (None if result["outcome"] is None else result["outcome"]["kind"]) == expected_outcome


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
    wire = item.to_dict()
    wire["payload"]["lookup"]["outcome"] = deepcopy(replacement)

    with pytest.raises(RecordError, match="requires"):
        InboxItem(**wire).encode()


def test_provider_lookup_closed_fields_and_binding_are_enforced() -> None:
    source = _contract_source()
    w3 = _fixture(source, "execution_worker_w3.json")
    item = _lookup_item(w3["lookup_results"][1]["result"])

    extra = item.to_dict()
    extra["payload"]["lookup"]["unknown"] = True
    with pytest.raises(RecordError, match="invalid session schema"):
        InboxItem(**extra).encode()

    wrong_operation = item.to_dict()
    wrong_operation["payload"]["lookup"]["binding"]["operation_id"] = "other-operation"
    with pytest.raises(RecordError):
        InboxItem(**wrong_operation).encode()


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

    for vector in w3["lookup_results"]:
        item = _lookup_item(vector["result"])
        Draft202012Validator(_fixture(source, "session_inbox.schema.json")).validate(item.to_dict())
