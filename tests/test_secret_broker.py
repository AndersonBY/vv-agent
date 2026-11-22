from __future__ import annotations

import json
import logging
import pickle
from copy import copy, deepcopy
from typing import Any

import pytest

from vv_agent.kernel._content import ContentLabels, ContentRef, Principal
from vv_agent.runtime.secrets import (
    InMemorySecretBroker,
    InvalidSecretRequest,
    SecretAlreadyUsed,
    SecretBinding,
    SecretBindingMismatch,
    SecretBroker,
    SecretDenied,
    SecretExpired,
    SecretReference,
    SecretRevoked,
    UnsupportedSecretMediation,
)

SECRET_NAME = "secret://integrations/mail/api-key"
RAW_SECRET = "canary-secret-value-that-must-not-leak"


def _principal(*, tenant_id: str = "tenant-a", subject_id: str = "user-a") -> Principal:
    attributes = ContentRef(
        content_id=f"principal/{subject_id}",
        mac="0" * 64,
        size=0,
        key_id="principal-key",
        labels=ContentLabels(),
    )
    return Principal(tenant_id, subject_id, "user", attributes)


def _register(broker: InMemorySecretBroker, *, tenant_id: str = "tenant-a") -> None:
    broker.register(
        SECRET_NAME,
        RAW_SECRET,
        tenant_id=tenant_id,
        allowed_actions={"mail": ("send",)},
    )


def _context(**changes: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "principal": _principal(),
        "session_id": "session-a",
        "environment_id": "environment-a",
        "generation": 7,
        "operation_id": "operation-a",
        "service": "mail",
        "action": "send",
    }
    values.update(changes)
    return values


def _issue(broker: InMemorySecretBroker, **changes: Any) -> SecretReference:
    return broker.issue(SECRET_NAME, **_context(**changes), ttl_seconds=30)


def _use(broker: InMemorySecretBroker, reference: SecretReference, **changes: Any) -> Any:
    return broker.use(reference, **_context(**changes), mediated_operation=lambda material: material)


def test_correct_binding_mediates_exactly_one_use() -> None:
    broker = InMemorySecretBroker()
    _register(broker)
    reference = _issue(broker)
    observed: list[bytes] = []

    assert isinstance(broker, SecretBroker)
    assert _use(broker, reference) == RAW_SECRET.encode()

    with pytest.raises(SecretAlreadyUsed) as rejected:
        broker.use(
            reference,
            **_context(),
            mediated_operation=observed.append,
        )

    assert observed == []
    assert RAW_SECRET not in str(rejected.value)


@pytest.mark.parametrize(
    "changed_context",
    [
        {"principal": _principal(tenant_id="tenant-b")},
        {"principal": _principal(subject_id="user-b")},
        {"session_id": "session-b"},
        {"environment_id": "environment-b"},
        {"generation": 8},
        {"operation_id": "operation-b"},
        {"service": "calendar"},
        {"action": "read"},
    ],
    ids=[
        "tenant",
        "principal",
        "session",
        "environment",
        "generation",
        "operation",
        "service",
        "action",
    ],
)
def test_use_rechecks_every_binding_component_before_consuming(changed_context: dict[str, Any]) -> None:
    broker = InMemorySecretBroker()
    _register(broker)
    reference = _issue(broker)
    observed: list[bytes] = []

    with pytest.raises(SecretBindingMismatch):
        broker.use(reference, **_context(**changed_context), mediated_operation=observed.append)

    assert observed == []
    assert _use(broker, reference) == RAW_SECRET.encode()


def test_secret_name_and_knowledge_of_its_value_do_not_authorize() -> None:
    broker = InMemorySecretBroker()
    request = _context()

    with pytest.raises(SecretDenied):
        broker.issue(SECRET_NAME, **request, ttl_seconds=30)

    _register(broker, tenant_id="tenant-a")
    with pytest.raises(SecretDenied):
        broker.issue(SECRET_NAME, **_context(principal=_principal(tenant_id="tenant-b")), ttl_seconds=30)


@pytest.mark.parametrize(
    "secret_name",
    [
        "integrations/mail/api-key",
        "secret://",
        "secret:///api-key",
        "secret://integrations//api-key",
        "secret://integrations/mail?action=send",
        "secret://integrations/mail ",
    ],
)
def test_issue_and_register_reject_noncanonical_secret_names(secret_name: str) -> None:
    broker = InMemorySecretBroker()

    with pytest.raises(InvalidSecretRequest):
        broker.register(
            secret_name,
            RAW_SECRET,
            tenant_id="tenant-a",
            allowed_actions={"mail": ("send",)},
        )
    with pytest.raises(InvalidSecretRequest):
        broker.issue(secret_name, **_context(), ttl_seconds=30)


def test_unsupported_service_action_rejects_before_any_use() -> None:
    broker = InMemorySecretBroker()
    _register(broker)

    with pytest.raises(UnsupportedSecretMediation):
        _issue(broker, service="mail", action="read")
    with pytest.raises(UnsupportedSecretMediation):
        _issue(broker, service="calendar", action="send")


def test_expired_reference_rejects_without_calling_adapter() -> None:
    now = [100.0]
    broker = InMemorySecretBroker(clock=lambda: now[0])
    _register(broker)
    reference = broker.issue(SECRET_NAME, **_context(), ttl_seconds=5)
    observed: list[bytes] = []
    now[0] = 105.0

    with pytest.raises(SecretExpired):
        broker.use(reference, **_context(), mediated_operation=observed.append)

    assert observed == []


def test_revoke_blocks_the_reference() -> None:
    broker = InMemorySecretBroker()
    _register(broker)
    reference = _issue(broker)
    observed: list[bytes] = []

    broker.revoke(reference)

    with pytest.raises(SecretRevoked):
        broker.use(reference, **_context(), mediated_operation=observed.append)
    assert observed == []


def test_adapter_failure_still_consumes_the_one_shot_reference() -> None:
    broker = InMemorySecretBroker()
    _register(broker)
    reference = _issue(broker)

    def fail(material: bytes) -> None:
        assert material == RAW_SECRET.encode()
        raise RuntimeError("adapter failed after receiving authority")

    with pytest.raises(RuntimeError, match="adapter failed"):
        broker.use(reference, **_context(), mediated_operation=fail)
    with pytest.raises(SecretAlreadyUsed):
        _use(broker, reference)


def test_reference_is_opaque_nonserializable_and_safe_to_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    broker = InMemorySecretBroker()
    _register(broker)
    reference = _issue(broker)
    binding = SecretBinding(**_context())

    with pytest.raises(SecretDenied) as rejected:
        broker.revoke(SecretReference())

    rendered = "\n".join(
        (
            repr(broker),
            repr(reference),
            str(reference),
            repr(binding),
            repr(rejected.value),
            caplog.text,
        )
    )

    assert RAW_SECRET not in rendered
    assert SECRET_NAME not in repr(reference)
    assert "opaque" in repr(reference)
    with pytest.raises(TypeError):
        pickle.dumps(reference)
    with pytest.raises(TypeError):
        json.dumps(reference)
    with pytest.raises(TypeError):
        copy(reference)
    with pytest.raises(TypeError):
        deepcopy(reference)
