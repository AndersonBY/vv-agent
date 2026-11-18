from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

from vv_agent.kernel._budget_values import RunBudgetLimits
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import (
    Classification,
    ContentIntegrityError,
    ContentLabels,
    ContentRef,
    ContentUnavailable,
    SourceKind,
)
from vv_agent.kernel._content import (
    KeyUnavailable as StoreKeyUnavailable,
)
from vv_agent.kernel._execution_values import AgentTask
from vv_agent.kernel._hydration import (
    IntegrityFailure,
    KeyUnavailable,
    MissingContent,
    MissingContentLink,
    ReferenceDigestMismatch,
    SessionHydrator,
    TypedSchemaFailure,
    UnsupportedHydrationSlot,
    reference_digest,
)
from vv_agent.kernel._message_values import Message
from vv_agent.kernel._prompt_values import PromptBundle, PromptSection


def _ref(name: str, *, size: int = 0) -> ContentRef:
    return ContentRef(
        content_id=f"content/{name}",
        mac=hashlib.sha256(f"mac/{name}".encode()).hexdigest(),
        size=size,
        key_id="key/session",
        labels=ContentLabels((SourceKind.SYSTEM,), Classification.INTERNAL),
    )


def _with_body(ref: ContentRef, value) -> tuple[ContentRef, bytes]:
    body = canonical_json_bytes(value)
    return ContentRef(
        content_id=ref.content_id,
        mac=ref.mac,
        size=len(body),
        key_id=ref.key_id,
        labels=ref.labels,
    ), body


class FakeReader:
    def __init__(
        self,
        values: dict[tuple[str, str], bytes] | None = None,
        *,
        failure: Exception | None = None,
        non_bytes_output: object | None = None,
    ) -> None:
        self.values = values or {}
        self.failure = failure
        self.non_bytes_output = non_bytes_output
        self.calls: list[tuple[str, str, str, str, ContentRef]] = []

    def read(
        self,
        tenant_id: str,
        session_id: str,
        owner: str,
        slot: str,
        ref: ContentRef,
    ) -> bytes:
        self.calls.append((tenant_id, session_id, owner, slot, ref))
        if self.failure is not None:
            raise self.failure
        if self.non_bytes_output is not None:
            return cast(bytes, self.non_bytes_output)
        try:
            return self.values[(owner, slot)]
        except KeyError as exc:
            raise MissingContentLink("owner/slot/ref link is absent") from exc


def _hydrator(reader: FakeReader, *, owner: str = "record/turn-start") -> SessionHydrator:
    return SessionHydrator(reader, tenant_id="tenant-a", session_id="session-a", owner=owner)


def _task_payload() -> dict:
    task = AgentTask(
        task_id="session-a/task",
        model="scripted",
        prompt_bundle=PromptBundle((PromptSection("system", "Work precisely."),)),
        user_prompt="do the work",
    )
    return task.to_dict()


def test_reader_is_called_with_the_exact_tenant_session_owner_slot_and_ref() -> None:
    ref, body = _with_body(_ref("input"), {"text": "go"})
    reader = FakeReader({("record/turn-start", "input"): body})

    assert _hydrator(reader).input_content(ref) == {"text": "go"}
    assert reader.calls == [("tenant-a", "session-a", "record/turn-start", "input", ref)]


@pytest.mark.parametrize(
    "failure,error",
    [
        (MissingContentLink("missing link"), MissingContentLink),
        (MissingContent("missing content"), MissingContent),
        (KeyUnavailable("key service down"), KeyUnavailable),
        (IntegrityFailure("tampered"), IntegrityFailure),
        (ContentUnavailable("missing"), MissingContent),
        (StoreKeyUnavailable("unavailable"), KeyUnavailable),
        (ContentIntegrityError("tampered"), IntegrityFailure),
    ],
)
def test_reader_failures_have_stable_kernel_boundaries(failure: Exception, error: type[Exception]) -> None:
    ref, _ = _with_body(_ref("input"), {"text": "go"})
    with pytest.raises(error):
        _hydrator(FakeReader(failure=failure)).input_content(ref)


def test_missing_link_binding_is_checked_before_content_is_returned() -> None:
    ref, body = _with_body(_ref("workspace"), "/workspace")
    reader = FakeReader({("record/other", "workspace"): body})

    with pytest.raises(MissingContentLink):
        _hydrator(reader).workspace(ref)


def test_reference_digest_is_distinct_from_mac_and_plaintext_digest() -> None:
    ref = _ref("digest")
    plaintext_digest = hashlib.sha256(b"plaintext").hexdigest()

    assert reference_digest(ref) == hashlib.sha256(canonical_json_bytes(ref.to_dict())).hexdigest()
    assert len({reference_digest(ref), ref.mac, plaintext_digest}) == 3
    assert reference_digest(ContentRef.from_dict(ref.to_dict())) == reference_digest(ref)
    assert reference_digest(replace(ref, content_id="content/other")) != reference_digest(ref)


def test_reference_digest_mismatch_fails_before_reader_access() -> None:
    ref, body = _with_body(_ref("input"), {"text": "go"})
    reader = FakeReader({("record/turn-start", "input"): body})

    with pytest.raises(ReferenceDigestMismatch):
        _hydrator(reader).input_content(ref, expected_reference_digest=reference_digest(_ref("other")))
    assert reader.calls == []


def test_content_ref_json_object_is_parsed_only_after_reader_decryption() -> None:
    inner = _ref("inner")
    outer, body = _with_body(_ref("outer"), inner.to_dict())
    reader = FakeReader({("record/turn-start", "content_ref"): body})

    assert _hydrator(reader).content_ref(outer) == inner
    assert reader.calls[0][3] == "content_ref"


def test_content_ref_is_never_used_as_plaintext_fallback() -> None:
    ref = _ref("missing")
    reader = FakeReader(failure=MissingContent("missing"))

    with pytest.raises(MissingContent):
        _hydrator(reader).request(ref)


def test_nested_refs_are_preserved_without_recursive_hydration() -> None:
    nested = _ref("nested")
    outer, body = _with_body(_ref("input"), {"reply": nested.to_dict(), "values": [1, 2]})
    reader = FakeReader({("record/turn-start", "input"): body})

    value = _hydrator(reader).input_content(outer)

    assert value == {"reply": nested.to_dict(), "values": [1, 2]}
    assert len(reader.calls) == 1


@pytest.mark.parametrize("body", [b"{", b'{"a":1,"a":2}', b'{"a": 1}', b'{"a":NaN}'])
def test_noncanonical_or_malformed_json_fails_closed(body: bytes) -> None:
    ref = ContentRef("content/input", hashlib.sha256(b"mac").hexdigest(), len(body), "key/session", _ref("x").labels)
    reader = FakeReader({("record/turn-start", "input"): body})

    with pytest.raises(TypedSchemaFailure):
        _hydrator(reader).input_content(ref)


def test_unknown_agent_task_field_is_rejected() -> None:
    payload = _task_payload() | {"unknown": True}
    ref, body = _with_body(_ref("task"), payload)
    reader = FakeReader({("record/turn-start", "task"): body})

    with pytest.raises(TypedSchemaFailure):
        _hydrator(reader).task(ref)


def test_task_uses_the_existing_agent_task_parser() -> None:
    ref, body = _with_body(_ref("task"), _task_payload())
    reader = FakeReader({("record/turn-start", "task"): body})

    task = _hydrator(reader).task(ref)

    assert task.user_prompt == "do the work"
    assert task.prompt_bundle.flatten() == "Work precisely."


def test_principal_attributes_validate_the_declared_seed_subtree() -> None:
    state = {"response_rejection": True}
    ref, body = _with_body(_ref("attributes"), {"host": {"id": 7}, "seed": state})
    reader = FakeReader({("record/turn-start", "principal.attributes"): body})

    assert _hydrator(reader).principal_attributes(ref) == {"host": {"id": 7}, "seed": state}

    invalid_ref, invalid_body = _with_body(_ref("attributes-invalid"), {"seed": state | {"unknown": True}})
    with pytest.raises(TypedSchemaFailure):
        _hydrator(FakeReader({("record/turn-start", "principal.attributes"): invalid_body})).principal_attributes(invalid_ref)


def test_definition_parses_the_closed_container_and_its_embedded_task() -> None:
    payload = {
        "agent_name": "kernel",
        "memory_settings": {},
        "child_tools": [],
        "model_binding": {
            "backend": "scripted",
            "model": "scripted",
            "context_length": None,
            "max_output_tokens": None,
            "internal": {
                "compaction": {
                    "backend": "scripted",
                    "model": "summary",
                    "context_length": None,
                    "max_output_tokens": None,
                }
            },
        },
        "capabilities": {},
        "task": _task_payload(),
        "tools": [],
    }
    ref, body = _with_body(_ref("definition"), payload)
    reader = FakeReader({("record/turn-start", "definition"): body})

    definition = _hydrator(reader).definition(ref)

    assert definition.agent_name == "kernel"
    assert definition.task.user_prompt == "do the work"
    assert definition.tools == ()
    assert definition.model_binding == payload["model_binding"]

    unknown_ref, unknown_body = _with_body(_ref("definition-invalid"), payload | {"unknown": True})
    with pytest.raises(TypedSchemaFailure):
        _hydrator(FakeReader({("record/turn-start", "definition"): unknown_body})).definition(unknown_ref)

    endpoints_ref, endpoints_body = _with_body(
        _ref("definition-endpoints"),
        payload | {"model_binding": payload["model_binding"] | {"endpoints": []}},
    )
    with pytest.raises(TypedSchemaFailure):
        _hydrator(FakeReader({("record/turn-start", "definition"): endpoints_body})).definition(endpoints_ref)

    incomplete_binding = {
        "backend": "scripted",
        "model": "summary",
        "context_length": None,
    }
    incomplete_ref, incomplete_body = _with_body(
        _ref("definition-incomplete-internal"),
        payload | {"model_binding": payload["model_binding"] | {"internal": {"compaction": incomplete_binding}}},
    )
    with pytest.raises(TypedSchemaFailure):
        _hydrator(FakeReader({("record/turn-start", "definition"): incomplete_body})).definition(incomplete_ref)


def test_budget_is_exact_and_returns_run_budget_limits() -> None:
    budget = RunBudgetLimits(max_tool_calls=3)
    ref, body = _with_body(_ref("budget"), budget.to_dict())
    reader = FakeReader({("record/turn-start", "budget"): body})

    assert _hydrator(reader).budget(ref) == budget

    invalid_ref, invalid_body = _with_body(_ref("budget-invalid"), budget.to_dict() | {"unknown": 1})
    invalid_reader = FakeReader({("record/turn-start", "budget"): invalid_body})
    with pytest.raises(TypedSchemaFailure):
        _hydrator(invalid_reader).budget(invalid_ref)

    host_ref, host_body = _with_body(
        _ref("budget-host-invalid"),
        budget.to_dict()
        | {
            "max_host_cost": {
                "unit": "usd",
                "currency": "USD",
                "amount_microunits": 1,
                "unknown": True,
            }
        },
    )
    with pytest.raises(TypedSchemaFailure):
        _hydrator(FakeReader({("record/turn-start", "budget"): host_body})).budget(host_ref)


def test_message_array_round_trips_through_the_current_message_codec() -> None:
    messages = [Message("system", "be concise"), Message("user", "go")]
    ref, body = _with_body(_ref("messages"), [message.to_dict() for message in messages])
    reader = FakeReader({("record/turn-start", "messages"): body})

    hydrated = _hydrator(reader).messages(ref)

    assert [message.to_dict() for message in hydrated] == [message.to_dict() for message in messages]


def test_unknown_message_field_is_rejected() -> None:
    payload = [Message("user", "go").to_dict() | {"unknown": True}]
    ref, body = _with_body(_ref("messages"), payload)
    reader = FakeReader({("record/turn-start", "messages"): body})

    with pytest.raises(TypedSchemaFailure):
        _hydrator(reader).messages(ref)


def test_declared_opaque_slots_require_canonical_json_shapes() -> None:
    request = {"model": "scripted", "messages": [_ref("nested").to_dict()]}
    request_ref, request_body = _with_body(_ref("request"), request)
    result_ref, result_body = _with_body(_ref("result"), {"content": "done"})
    usage_ref, usage_body = _with_body(_ref("usage"), {"input_tokens": 2})
    evidence_ref, evidence_body = _with_body(_ref("evidence"), [request["messages"][0]])
    values = {
        ("record/turn-start", "request"): request_body,
        ("record/turn-start", "result"): result_body,
        ("record/turn-start", "usage"): usage_body,
        ("record/turn-start", "evidence"): evidence_body,
    }
    hydrator = _hydrator(FakeReader(values))

    assert hydrator.request(request_ref) == request
    assert hydrator.result(result_ref) == {"content": "done"}
    assert hydrator.usage(usage_ref) == {"input_tokens": 2}
    assert hydrator.evidence(evidence_ref) == request["messages"]


def test_slot_dispatch_rejects_shapes_without_a_typed_parser() -> None:
    ref, body = _with_body(_ref("workspace"), "/workspace")
    reader = FakeReader({("record/turn-start", "workspace"): body})
    hydrator = _hydrator(reader)

    assert hydrator.hydrate("workspace", ref) == "/workspace"
    with pytest.raises(UnsupportedHydrationSlot):
        hydrator.hydrate("request.messages[0]", ref)
    assert len(reader.calls) == 1


@pytest.mark.parametrize(
    "decode",
    [
        lambda hydrator, ref: hydrator.content_ref(ref),
        lambda hydrator, ref: hydrator.principal_attributes(ref),
        lambda hydrator, ref: hydrator.workspace(ref),
        lambda hydrator, ref: hydrator.definition(ref),
        lambda hydrator, ref: hydrator.input_content(ref),
        lambda hydrator, ref: hydrator.task(ref),
        lambda hydrator, ref: hydrator.budget(ref),
        lambda hydrator, ref: hydrator.request(ref),
        lambda hydrator, ref: hydrator.result(ref),
        lambda hydrator, ref: hydrator.usage(ref),
        lambda hydrator, ref: hydrator.evidence(ref),
        lambda hydrator, ref: hydrator.messages(ref),
    ],
)
def test_non_bytes_reader_output_is_an_integrity_failure(decode: Callable[[SessionHydrator, ContentRef], object]) -> None:
    ref = _ref("bad-output", size=2)
    reader = FakeReader(non_bytes_output="{}")

    with pytest.raises(IntegrityFailure):
        decode(_hydrator(reader), ref)
