"""S1 input_applied producer shape across unified input dispositions."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from vv_agent.kernel._content import ContentRef, Principal
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_inputs import CommitInput, ResolveModelSelection, apply_input
from vv_agent.kernel._records import InboxItem, Record, copy_json
from vv_agent.kernel._state import ExecutionState
from vv_agent.session.store import Conflict
from vv_agent.session.surfaces import SessionDriver


def _ref(content_id: str) -> dict:
    return {
        "content_id": content_id,
        "mac": "0" * 64,
        "size": 1,
        "key_id": "test",
        "labels": {"source": ["user"], "classification": "internal"},
    }


ACTOR = {
    "tenant_id": "tenant",
    "subject_id": "subject",
    "subject_kind": "user",
    "attributes": _ref("actor"),
}


def _item(
    kind: str,
    payload: dict,
    *,
    input_id: str = "input",
    target_turn_id: str | None = None,
) -> InboxItem:
    return InboxItem(
        input_id=input_id,
        kind=kind,
        payload=payload,
        target_turn_id=target_turn_id,
        actor=ACTOR,
    )


def _driver(item: InboxItem, state: ExecutionState | None = None, *, capture: bool = False) -> CoreDriver:
    driver = object.__new__(CoreDriver)
    driver.sid = "session"
    driver.watermark = 1
    driver.state = state or ExecutionState()
    driver.store = SimpleNamespace(peek_inbox=lambda *args, **kwargs: [SimpleNamespace(item=item)])
    driver.content_reader = None
    if capture:

        def record(kind, payload, plan=None, *, tid=None):
            payload = copy_json(payload)
            return Record(
                record_id="captured",
                kind=kind,
                session_id="session",
                turn_id=tid,
                operation_id=None,
                attempt=None,
                payload=payload,
            )

        driver.record = record
    return driver


def _apply(driver: CoreDriver, response: dict | BaseException | None = None):
    pending = apply_input(driver, one_turn=True)
    request = next(pending)
    if isinstance(request, ResolveModelSelection):
        request = pending.throw(response) if isinstance(response, BaseException) else pending.send(response)
    assert isinstance(request, CommitInput)
    try:
        next(pending)
    except StopIteration as done:
        assert done.value is True
    else:
        raise AssertionError("apply_input yielded after CommitInput")
    assert len(request.records) == 1
    return request.records[0]


def test_session_driver_uses_explicit_ingress_principal_and_owns_actor_attributes():
    driver = SessionDriver()
    principal = {
        "tenant_id": "default",
        "subject_id": "ingress-user",
        "subject_kind": "user",
        "attributes": {"role": "operator", "labels": "business", "size": 17},
    }
    try:
        driver.create("session", "/workspace", principal=principal)
        first = driver.push("session", _item("user", {"content": "go"}, input_id="input"), actor=principal)
        replay = driver.push("session", _item("user", {"content": "go"}, input_id="input"), actor=principal)
        assert not first.replayed
        assert replay.replayed
        assert replay.input_seq == first.input_seq
        driver.push("session", replace(_item("user", {"content": "owned"}, input_id="owned-input"), actor=None))

        created = driver.store.read("session", limit=1).records[0]
        created_principal = created.record.payload["principal"]
        assert created_principal["subject_id"] == "ingress-user"
        assert created_principal["subject_kind"] == "user"
        assert (
            driver.read_record_content(
                "session",
                created.seq,
                created_principal["attributes"],
                slot="session.principal.attributes",
            )
            == principal["attributes"]
        )

        stored = driver.store.peek_inbox("session")[0].item
        assert stored.actor["tenant_id"] == "default"
        assert stored.actor["subject_id"] == "ingress-user"
        assert stored.actor["subject_kind"] == "user"
        assert (
            driver.read_input_content(
                "session",
                stored.input_id,
                stored.actor["attributes"],
                slot="input.actor.attributes",
            )
            == principal["attributes"]
        )
        assert "ingress-user" not in str(stored.payload["content"])
        assert all("ingress-user" not in message.content for message in driver.messages("session"))
        owned = next(item.item for item in driver.store.peek_inbox("session") if item.item.input_id == "owned-input")
        assert owned.actor["subject_id"] == "ingress-user"
        assert owned.actor["subject_kind"] == "user"

        with pytest.raises(ValueError, match="session tenant"):
            driver.push(
                "session",
                _item("user", {"content": "foreign"}, input_id="foreign"),
                actor=principal | {"tenant_id": "tenant-b"},
            )
        assert all(item.item.input_id != "foreign" for item in driver.store.peek_inbox("session"))

        persisted = Principal.from_dict(created_principal)
        driver.push(
            "session",
            replace(_item("user", {"content": "from-principal"}, input_id="principal-object"), actor=None),
            actor=persisted,
        )
        principal_object_actor = driver.store.read_input("session", "principal-object").item.actor
        assert (
            driver.read_input_content(
                "session",
                "principal-object",
                principal_object_actor["attributes"],
                slot="input.actor.attributes",
            )
            == principal["attributes"]
        )

        wrong_slot = Principal(
            tenant_id=persisted.tenant_id,
            subject_id=persisted.subject_id,
            subject_kind=persisted.subject_kind,
            attributes=ContentRef.from_dict(created.record.payload["workspace"]),
        )
        with pytest.raises(ValueError, match="owner link"):
            driver.push(
                "session",
                replace(_item("user", {"content": "wrong-slot"}, input_id="wrong-slot"), actor=None),
                actor=wrong_slot,
            )

        driver.create(
            "other",
            "/workspace",
            principal={
                "tenant_id": "default",
                "subject_id": "other-user",
                "subject_kind": "user",
                "attributes": {"role": "other"},
            },
        )
        other_principal = Principal.from_dict(driver.store.read("other", limit=1).records[0].record.payload["principal"])
        with pytest.raises(ValueError, match="owner link"):
            driver.push(
                "session",
                replace(_item("user", {"content": "foreign-ref"}, input_id="foreign-ref"), actor=None),
                actor=other_principal,
            )
        with pytest.raises(ValueError, match="host ingress principal mapping"):
            driver.create("persisted-create", "/workspace", principal=persisted)

        with pytest.raises(Conflict):
            driver.create(
                "session",
                "/workspace",
                principal=principal | {"subject_id": "different-user"},
            )
    finally:
        driver.close()


@pytest.mark.parametrize(
    ("item", "state", "disposition", "reason", "target", "wait", "position", "schema_ready"),
    [
        (
            _item("control", {"action": "archive"}),
            None,
            "applied",
            None,
            None,
            None,
            "between_actions",
            True,
        ),
        (
            _item("follow_up", {"content": _ref("follow-up")}),
            None,
            "queued",
            None,
            None,
            None,
            "between_actions",
            True,
        ),
        (
            _item("control", {"action": "cancel"}),
            None,
            "rejected",
            "control requires active target",
            None,
            None,
            "between_actions",
            False,
        ),
        (
            _item("steer", {"content": _ref("steer")}),
            ExecutionState(active_turn_id="current-turn"),
            "applied",
            None,
            None,
            None,
            "before_model",
            True,
        ),
        (
            _item("user", {"content": _ref("stale")}, target_turn_id="old-turn"),
            ExecutionState(active_turn_id="current-turn"),
            "rejected",
            "stale turn or generation",
            None,
            None,
            "between_actions",
            False,
        ),
        (
            _item(
                "child_result",
                {
                    "session_id": "child",
                    "turn_id": "turn",
                    "operation_id": "child-operation",
                    "attempt": 1,
                    "status": "completed",
                    "result": _ref("result"),
                    "terminal_seq": 1,
                    "terminal_digest": "1" * 64,
                },
            ),
            None,
            "rejected",
            "untrusted or mismatched child completion",
            "child-operation",
            None,
            "between_actions",
            False,
        ),
        (
            _item(
                "provider_result",
                {
                    "operation_id": "provider-operation",
                    "attempt": 1,
                    "request_digest": "2" * 64,
                    "provider_binding": None,
                    "result": _ref("provider-result"),
                    "usage": _ref("usage"),
                    "evidence": _ref("evidence"),
                },
            ),
            None,
            "rejected",
            "untrusted or mismatched provider evidence",
            "provider-operation",
            None,
            "between_actions",
            False,
        ),
        (
            _item(
                "approval_answer",
                {
                    "operation_id": "approval-operation",
                    "attempt": 1,
                    "request_id": "request",
                    "request_digest": "3" * 64,
                    "decision": "approve",
                    "scope": _ref("scope"),
                },
            ),
            None,
            "rejected",
            "approval identity or scope mismatch",
            "approval-operation",
            "request",
            "between_actions",
            False,
        ),
    ],
)
def test_input_applied_routes_keep_shape_with_explicit_null_proof(
    item, state, disposition, reason, target, wait, position, schema_ready
):
    record = _apply(_driver(item, state, capture=not schema_ready))
    payload = record._payload

    assert payload["input_proof"] is None
    assert payload["disposition"] == disposition
    assert payload["reason"] == reason
    assert payload["target_operation_id"] == target
    assert payload["target_wait_id"] == wait
    assert payload["position"] == position
    if schema_ready:
        record.encode()


@pytest.mark.parametrize(
    ("selection", "response", "disposition", "reason", "has_binding", "schema_ready"),
    [
        (None, None, "applied", None, False, True),
        (_ref("selection"), ValueError("unresolvable"), "rejected", "model_selection_unresolvable", False, False),
        (_ref("selection"), _ref("binding"), "applied", None, True, True),
    ],
)
def test_model_selection_input_applied_paths(selection, response, disposition, reason, has_binding, schema_ready):
    item = _item("model_selection", {"selection": selection})
    record = _apply(_driver(item, capture=not schema_ready), response)
    payload = record._payload

    assert payload["input_proof"] is None
    assert payload["disposition"] == disposition
    assert payload["reason"] == reason
    assert ("model_binding" in payload) is has_binding
    if schema_ready:
        record.encode()


def test_retained_user_reply_remains_noop_with_null_proof():
    item = _item("user", {"content": _ref("retained")}, input_id="retained")
    first = _apply(_driver(item, capture=True))
    retained = replace(first, payload=first._payload | {"disposition": "applied"})
    state = ExecutionState(applied_inputs={item.input_id: retained})

    record = _apply(_driver(item, state, capture=True))
    payload = record._payload

    assert payload["input_proof"] is None
    assert payload["disposition"] == "noop"
    assert payload["reason"] == "retained user reply"
