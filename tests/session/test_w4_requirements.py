"""Contract-41 typed requirements, codec boundaries and pure capability tags."""

from __future__ import annotations

import base64
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from support.contract import fixture_root

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._requirements import (
    SessionRequirements,
    capability_supports,
    model_capability_tag,
    model_capability_tags,
    static_capability_requirements,
    tool_capability_tag,
)
from vv_agent.session.records import InboxItem, Record, RecordError, SessionSpec, make_record


def ref(label: str) -> dict[str, Any]:
    payload = label.encode()
    return {
        "content_id": f"content/test/{label}",
        "mac": sha256(payload).hexdigest(),
        "size": len(payload),
        "key_id": "key/test",
        "labels": {"source": [], "classification": "internal"},
    }


def principal() -> dict[str, Any]:
    return {
        "tenant_id": "default",
        "subject_id": "subject",
        "subject_kind": "user",
        "attributes": ref("principal"),
    }


def requirement(level: int = 2, *capabilities: str) -> SessionRequirements:
    return SessionRequirements(
        level=level,
        handler_version="handler/h2",
        capabilities=tuple(sorted(set(capabilities))),
    )


def allocation_fixture() -> tuple[dict[str, Any], Path]:
    path = fixture_root() / "worker_compatibility.json"
    assert path.is_file(), "contract-41 worker compatibility fixture is not synced"
    return json.loads(path.read_text(encoding="utf-8")), path


def test_real_level_one_producer_bytes_stay_byte_exact() -> None:
    fixture = json.loads((fixture_root() / "trusted_storage_s1.json").read_text())
    seen = set()
    for case in fixture["accepted_cases"]:
        wire = case["wire"]
        body = base64.b64decode(case["bytes_base64"], validate=True)
        parsed = InboxItem.parse(body) if "actor" in wire else Record.parse(body)
        assert parsed.encode() == body
        assert parsed.to_dict() == wire
        assert "requirements" not in parsed.to_dict()
        assert "bootstrap" not in parsed.to_dict().get("payload", {})
        seen.add(wire["kind"])
    assert {"session_created", "user", "input_applied"} <= seen


def test_level_one_codec_rejects_each_new_control_field() -> None:
    fixture = json.loads((fixture_root() / "trusted_storage_s1.json").read_text())
    cases = {case["name"]: case for case in fixture["accepted_cases"]}

    created = cases["session_created_principal_and_content_ref"]["wire"]
    created["payload"]["bootstrap"] = requirement().to_dict()
    with pytest.raises(RecordError):
        Record.parse(canonical_json_bytes(created))

    user = cases["signed_user_inbox_admission_proof"]["wire"]
    user["requirements"] = requirement().to_dict()
    with pytest.raises(RecordError):
        InboxItem.parse(canonical_json_bytes(user))

    with pytest.raises(RecordError, match="bootstrap is limited"):
        SessionSpec(
            "s",
            principal(),
            ref("workspace"),
            attributes=ref("attributes"),
            bootstrap=requirement(),
        ).record()


def test_level_two_session_created_and_inbox_use_exact_canonical_bytes() -> None:
    requirements = requirement(2, "tool:z", "tool:a")
    created = SessionSpec(
        "s",
        principal(),
        ref("workspace"),
        attributes=ref("attributes"),
        feature_level=2,
        bootstrap=requirements,
    ).record()
    assert created.to_dict()["payload"]["bootstrap"] == requirements.to_dict()
    assert created.encode() == canonical_json_bytes(created.to_dict())
    assert Record.parse(created.encode()) == created

    expected = {
        "actor": principal(),
        "admission_proof": None,
        "available_ms": 0,
        "feature_level": 2,
        "generation": 0,
        "input_id": "user-l2",
        "kind": "user",
        "payload": {"content": ref("go")},
        "requirements": requirements.to_dict(),
        "schema_version": "S1",
        "target_turn_id": None,
    }
    body = canonical_json_bytes(expected)
    assert InboxItem.parse(body).encode() == body


@pytest.mark.parametrize("kind", ["user", "follow_up", "model_selection"])
def test_level_two_requirements_are_required_for_the_closed_inbox_kinds(kind: str) -> None:
    payload = {"content": ref("go")} if kind != "model_selection" else {"selection": ref("model")}
    item = InboxItem(
        kind,
        kind,
        payload,
        actor=principal(),
        feature_level=2,
        requirements=requirement(2, "model:x"),
    )
    assert InboxItem.parse(item.encode()).to_dict() == item.to_dict()

    missing = item.to_dict()
    missing.pop("requirements")
    with pytest.raises(RecordError):
        InboxItem.parse(canonical_json_bytes(missing))


def test_level_two_rejects_inapplicable_kinds_and_malformed_requirements() -> None:
    control = InboxItem(
        "control",
        "control",
        {"action": "cancel"},
        target_turn_id="s/turn/t",
        actor=principal(),
        feature_level=2,
        requirements=requirement(),
    )
    with pytest.raises(RecordError, match="limited to user"):
        control.encode()

    with pytest.raises(ValueError, match="exactly"):
        SessionRequirements.from_dict(
            {
                "level": 2,
                "handler_version": "handler/h2",
                "capabilities": [],
                "opaque": {},
            }
        )
    with pytest.raises(ValueError, match="duplicates"):
        SessionRequirements(2, "handler/h2", ("tool:a", "tool:a"))
    with pytest.raises(ValueError, match="not sorted"):
        SessionRequirements(2, "handler/h2", ("tool:z", "tool:a"))
    with pytest.raises(ValueError, match="positive integer"):
        SessionRequirements(0, "handler/h2", ())
    with pytest.raises(RecordError, match="limited to session_created"):
        make_record(
            "turn_started",
            session_id="s",
            turn_id="s/turn/t",
            payload={},
            feature_level=2,
        )

    wrong_selection_type = InboxItem(
        "selection",
        "model_selection",
        {"selection": ref("model")},
        actor=principal(),
        feature_level=2,
        requirements=requirement(2, "tool:not-a-model"),
    )
    with pytest.raises(RecordError, match="model capabilities only"):
        wrong_selection_type.encode()

    missing_family = SessionSpec(
        "s",
        principal(),
        ref("workspace"),
        attributes=ref("attributes"),
        feature_level=2,
        bootstrap=SessionRequirements(2, None, ()),
    )
    with pytest.raises(RecordError, match="family capability"):
        missing_family.record()


def test_nested_input_applied_parses_the_typed_level_two_inbox_item() -> None:
    incoming = InboxItem(
        "nested",
        "follow_up",
        {"content": ref("later")},
        actor=principal(),
        feature_level=2,
        requirements=requirement(2, "tool:nested"),
    )
    applied = make_record(
        "input_applied",
        session_id="s",
        payload={
            "input": incoming.to_dict(),
            "input_digest": incoming.digest,
            "input_proof": None,
            "disposition": "queued",
            "reason": None,
            "target_operation_id": None,
            "target_wait_id": None,
            "position": "pending",
        },
    )
    parsed = Record.parse(applied.encode())
    assert parsed.to_dict()["payload"]["input"]["feature_level"] == 2
    assert parsed.to_dict()["payload"]["input"]["requirements"] == requirement(2, "tool:nested").to_dict()

    malformed = applied.to_dict()
    malformed["payload"]["input"]["requirements"]["unexpected"] = True
    with pytest.raises(RecordError):
        Record.parse(canonical_json_bytes(malformed))


def test_tool_tags_hash_the_full_registry_schema_and_metadata() -> None:
    fixture, _ = allocation_fixture()
    for tool in fixture["trusted_catalog"]["tools"]:
        actual = tool_capability_tag(tool["name"], tool["schema"], tool["metadata"])
        assert actual == tool["tag"]
        with pytest.raises(ValueError, match="outer type=function"):
            tool_capability_tag(
                tool["name"],
                tool["schema"]["function"]["parameters"],
                tool["metadata"],
            )


def test_model_tags_ignore_endpoints_react_to_capacity_and_decompose_internal() -> None:
    fixture, _ = allocation_fixture()
    cases = {case["name"]: case for case in fixture["model_tag_cases"]}
    for case in cases.values():
        actual = tuple(sorted({tag for item in case["bindings"] for tag in model_capability_tags(item["binding"])}))
        assert actual == tuple(sorted(set(case["expected_tags"])))

    primary = cases["internal_reuse_emits_unique_atomic_tags"]["bindings"][0]["binding"]
    primary["internal"] = {
        "same": cases["internal_reuse_emits_unique_atomic_tags"]["bindings"][1]["binding"],
        "summary": cases["internal_reuse_emits_unique_atomic_tags"]["bindings"][2]["binding"],
    }
    assert model_capability_tags(primary) == tuple(sorted(set(cases["internal_reuse_emits_unique_atomic_tags"]["expected_tags"])))
    assert model_capability_tag({**primary, "route": "private", "credentials": "secret"}) == model_capability_tag(primary)


def test_static_bound_uses_complete_declarations_without_runtime_compile() -> None:
    fixture, _ = allocation_fixture()
    tools = [(tool["schema"], tool["metadata"]) for tool in fixture["trusted_catalog"]["tools"]]
    model_binding = fixture["model_tag_cases"][0]["bindings"][0]["binding"] | {
        "internal": {"summary": fixture["model_tag_cases"][2]["bindings"][2]["binding"]}
    }
    expected = set(capability_supports(tools=tools, models=[model_binding], extra_capabilities=["family:custom"]))
    bound = static_capability_requirements(
        level=2,
        handler_version="handler/h2",
        tools=tools,
        models=[model_binding],
        extra_capabilities=["family:custom"],
    )
    assert set(bound.capabilities) == expected
    assert bound.to_dict() == {
        "level": 2,
        "handler_version": "handler/h2",
        "capabilities": list(bound.capabilities),
    }
