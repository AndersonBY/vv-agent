"""Focused S1 content lifecycle contracts without a durable backend."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from support.contract import fixture_root

from vv_agent.kernel import _content as content


def _lifecycle() -> dict:
    return json.loads((fixture_root() / "session_storage_lifecycle.json").read_text())


def _head(case: dict) -> content.SessionHead:
    header = case["header"]
    return content.SessionHead(
        tenant_id=header["tenant_id"],
        session_id=header["session_id"],
        final_seq=header["head_seq"],
        chain_digest=header["chain_digest"],
        key_id=_lifecycle()["key_capability"]["key_id"],
    )


def _walk(value):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _refs(value) -> list[dict]:
    fields = {"content_id", "mac", "size", "key_id", "labels"}
    return [node for node in _walk(value) if isinstance(node, dict) and set(node) == fields]


class MemoryKeyProvider:
    """Deterministic test double; this is intentionally absent from production."""

    def __init__(self, capability: content.SessionKeyCapability) -> None:
        self.capability = capability
        self.evidence = 0

    def destroy(self, tenant_id: str, key_id: str) -> content.KeyDestroyOutcome:
        status = self.capability.destroy_status(tenant_id, key_id)
        if status is content.KeyDestroyStatus.BINDING_MISMATCH:
            return content.KeyDestroyOutcome(False, status, None, "Capability identity includes tenant.")
        self.evidence += 1
        evidence_id = f"destroy/evidence/{self.evidence}"
        if status is content.KeyDestroyStatus.DESTROYED:
            self.capability = replace(self.capability, state=content.KeyCapabilityState.DESTROYED)
        return content.KeyDestroyOutcome(True, status, evidence_id if status is not content.KeyDestroyStatus.DESTROYING else None)

    def generate_data_key(self, tenant_id: str, session_id: str) -> content.WrappedDataKey:
        raise NotImplementedError

    def unwrap(self, tenant_id: str, session_id: str, wrapped_key: content.WrappedDataKey) -> bytes:
        raise NotImplementedError


def test_shared_content_ref_schema_accepts_every_fixture_reference() -> None:
    lifecycle = _lifecycle()
    refs = [content.ContentRef.from_dict(value) for value in _refs(lifecycle)]
    assert refs
    assert all(content.ContentRef.from_dict(ref.to_dict()) == ref for ref in refs)


@pytest.mark.parametrize(
    "source,code",
    [
        (["user", "system"], content.ContentErrorCode.INVALID_CONTENT_REF),
        (["system", "system"], content.ContentErrorCode.INVALID_CONTENT_REF),
        (["unknown"], content.ContentErrorCode.INVALID_CONTENT_REF),
    ],
)
def test_content_ref_labels_are_closed_ordered_and_duplicate_free(source: list[str], code: content.ContentErrorCode) -> None:
    value = deepcopy(_refs(_lifecycle())[0])
    value["labels"]["source"] = source
    with pytest.raises(content.ContentLifecycleError) as caught:
        content.ContentRef.from_dict(value)
    assert caught.value.code is code


def test_content_ref_rejects_unknown_fields_with_the_shared_schema() -> None:
    value = deepcopy(_refs(_lifecycle())[0])
    value["plaintext"] = "must not cross the boundary"
    with pytest.raises(content.InvalidContentRef):
        content.ContentRef.from_dict(value)


def test_content_identity_is_session_key_mac_and_labels_only() -> None:
    lifecycle = _lifecycle()
    original = content.ContentRef.from_dict(_refs(lifecycle)[0])
    binding = content.ContentBinding("default", "session", original.key_id)
    duplicate_id = replace(original, content_id="content/other/opaque")
    different_key = replace(original, key_id="key/other")
    assert content.bind_content_ref(original, binding) == content.bind_content_ref(duplicate_id, binding)
    assert content.bind_content_ref(original, binding) != content.bind_content_ref(
        different_key, content.ContentBinding("default", "session", "key/other")
    )
    assert content.bind_content_ref(original, binding) != content.bind_content_ref(
        original,
        binding,
    ).__class__(
        original.key_id,
        original.mac,
        content.ContentLabels((content.SourceKind.USER,), content.Classification.INTERNAL),
    )
    with pytest.raises(content.ContentBindingMismatch):
        content.bind_content_ref(original, content.ContentBinding("default", "session", "key/other"))


def test_destroy_fixture_cases_are_idempotent_and_binding_bound() -> None:
    lifecycle = _lifecycle()
    capability = content.SessionKeyCapability.from_dict(lifecycle["key_capability"])
    for case in lifecycle["destroy_cases"]:
        provider = MemoryKeyProvider(capability)
        request = content.KeyDestroyRequest(**case["request"])
        if case["name"] == "destroy_is_idempotent":
            provider.destroy(request.tenant_id, request.key_id)
        outcome = provider.destroy(request.tenant_id, request.key_id)
        assert outcome.accepted is case["accepted"]
        assert outcome.status.value == case["status"]
        assert (outcome.evidence_id is None) is (case["evidence_id"] is None)
        if case["reason"] is not None:
            assert outcome.reason == case["reason"]


def test_destroying_is_accepted_without_claiming_completion() -> None:
    lifecycle = _lifecycle()
    capability = content.SessionKeyCapability.from_dict(lifecycle["key_capability"])
    provider = MemoryKeyProvider(replace(capability, state=content.KeyCapabilityState.DESTROYING))
    outcome = provider.destroy(capability.tenant_id, capability.key_id)
    assert outcome.accepted
    assert outcome.status is content.KeyDestroyStatus.DESTROYING
    assert outcome.evidence_id is None
    assert isinstance(provider, content.KeyProvider)


def test_capability_requires_tenant_session_and_key_binding() -> None:
    lifecycle = _lifecycle()
    capability = content.SessionKeyCapability.from_dict(lifecycle["key_capability"])
    capability.require_binding(capability.tenant_id, capability.session_id, capability.key_id)
    with pytest.raises(content.ContentBindingMismatch):
        capability.require_binding(capability.tenant_id, "other-session", capability.key_id)


@pytest.mark.parametrize("case", _lifecycle()["forget_verification_cases"], ids=lambda case: case["name"])
def test_forget_fixture_cases_are_classified(case: dict) -> None:
    lifecycle = _lifecycle()
    tombstone = content.ForgetTombstone.from_dict(case["tombstone"])
    head = _head(lifecycle["export_cases"][0])
    if case["accepted"]:
        tombstone.verify(head, scope_root_session_id=head.session_id)
        return
    with pytest.raises(content.ContentLifecycleError) as caught:
        tombstone.verify(head, scope_root_session_id=head.session_id)
    expected = {
        "tombstone_head_mismatch": content.ContentErrorCode.TOMBSTONE_HEAD_MISMATCH,
        "tombstone_session_reuse": content.ContentErrorCode.FORGOTTEN_SESSION_REUSE,
    }[case["name"]]
    assert caught.value.code is expected
    assert str(caught.value) == case["reason"]


def test_pending_tombstone_never_verifies_forget() -> None:
    lifecycle = _lifecycle()
    tombstone = content.ForgetTombstone.from_dict(lifecycle["forget_tombstone"])
    with pytest.raises(content.InvalidTombstone):
        replace(tombstone, state=content.TombstoneState.PENDING).verify(_head(lifecycle["export_cases"][0]))


@pytest.mark.parametrize("case", _lifecycle()["export_cases"], ids=lambda case: case["name"])
def test_export_fixture_cases_are_classified(case: dict) -> None:
    lifecycle = _lifecycle()
    head = _head(lifecycle["export_cases"][0])
    if case["accepted"]:
        content.ExportDocument.from_dict(case["header"], case["lines"]).verify(head)
        return
    with pytest.raises(content.ContentLifecycleError) as caught:
        content.ExportDocument.from_dict(case["header"], case["lines"]).verify(head)
    expected = {
        "unknown_content_mode": content.ContentErrorCode.EXPORT_MODE_MISMATCH,
        "skeleton_export_contains_content": content.ContentErrorCode.EXPORT_MODE_MISMATCH,
        "unsupported_export_version": content.ContentErrorCode.UNSUPPORTED_EXPORT_VERSION,
    }[case["name"]]
    assert caught.value.code is expected
    assert str(caught.value) == case["reason"]


def test_skeleton_export_remains_verifiable_after_key_destruction() -> None:
    lifecycle = _lifecycle()
    capability = content.SessionKeyCapability.from_dict(lifecycle["key_capability"])
    provider = MemoryKeyProvider(capability)
    assert provider.destroy(capability.tenant_id, capability.key_id).status is content.KeyDestroyStatus.DESTROYED
    skeleton = lifecycle["export_cases"][0]
    content.ExportDocument.from_dict(skeleton["header"], skeleton["lines"]).verify(_head(skeleton))


def test_include_content_line_requires_the_referenced_content_key() -> None:
    lifecycle = _lifecycle()
    case = deepcopy(lifecycle["export_cases"][1])
    content_line = next(line for line in case["lines"] if line["line"] == "content")
    content_line["key_id"] = "key/other"
    with pytest.raises(content.ContentBindingMismatch):
        content.ExportDocument.from_dict(case["header"], case["lines"]).verify(_head(case))


def test_export_content_refs_ignore_reference_shaped_opaque_json() -> None:
    opaque_ref = content.ContentRef(
        "content/opaque-provider-value",
        "a" * 64,
        4,
        "key/foreign-owner",
        content.ContentLabels(),
    ).to_dict()
    record = SimpleNamespace(
        kind="boundary_recorded",
        _payload={
            "stage": "before_memory",
            "data": {
                "messages": [],
                "shared_state": None,
                "provider_observation": {"reference": opaque_ref},
            },
        },
    )

    assert content._record_content_refs(record) == []
    content._verify_record_tenant(record, "default")


def test_export_tenant_validation_reads_only_closed_principal_fields() -> None:
    record = SimpleNamespace(
        kind="session_created",
        _payload={"principal": {"tenant_id": "other"}, "attributes": {"tenant_id": "other"}},
    )
    with pytest.raises(content.ContentBindingMismatch, match="record tenant does not match"):
        content._verify_record_tenant(record, "default")


def test_module_implementation_contains_no_test_double_or_key_material() -> None:
    source = Path(content.__file__).read_text()
    assert "MemoryKeyProvider" not in source
    assert "key_material" not in source
    assert "plaintext_key" not in source
