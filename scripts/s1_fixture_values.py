"""Shared S1-safe values and transaction helpers for producer fixtures."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import replace
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

from vv_agent.kernel._content import Principal
from vv_agent.kernel._content_publication import declare_input_content, link_declared_input_content
from vv_agent.kernel._content_runtime import (
    SessionContentWriter,
    bind_content_runtime,
    durable_content_reader,
    wrap_session_content_store,
)
from vv_agent.kernel._records import InboxItem, SessionSpec
from vv_agent.kernel._requirements import SessionRequirements

_FIXTURE_KEY = sha256(b"vv-agent-s1-test-fixtures").digest()


class _FixtureSecrets:
    def __init__(self) -> None:
        self._counter = 0

    def token_bytes(self, count: int) -> bytes:
        output = bytearray()
        while len(output) < count:
            self._counter += 1
            output.extend(sha256(f"vv-agent-s1-fixture-content:{self._counter}".encode()).digest())
        return bytes(output[:count])

    def token_urlsafe(self, count: int) -> str:
        return base64.urlsafe_b64encode(self.token_bytes(count)).decode().rstrip("=")


@contextmanager
def deterministic_content_secrets():
    from vv_agent.session import content_store

    with patch.object(content_store, "secrets", _FixtureSecrets()):
        yield


def deterministic_content_producer(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with deterministic_content_secrets():
            return function(*args, **kwargs)

    return wrapped


def content_ref(label: str = "content") -> dict[str, Any]:
    payload = label.encode("utf-8")
    return {
        "content_id": f"content/test/{label}",
        "mac": sha256(payload).hexdigest(),
        "size": len(payload),
        "key_id": "key/test",
        "labels": {"source": [], "classification": "internal"},
    }


def principal(subject_id: str = "p", subject_kind: str = "user") -> dict[str, Any]:
    return {
        "tenant_id": "default",
        "subject_id": subject_id,
        "subject_kind": subject_kind,
        "attributes": content_ref(f"principal/{subject_id}"),
    }


def fixture_key_provider(state_path: str | Path | None = None):
    from vv_agent.session.content_store import LocalKeyProvider

    return LocalKeyProvider(_FIXTURE_KEY, state_path=state_path)


def fixture_key_provider_for_store(store):
    path = getattr(store, "path", None)
    state_path = None if path is None else Path(f"{path}.content-key-state.sqlite3")
    return fixture_key_provider(state_path)


def wrap_fixture_store(store, provider=None, *, cipher=None):
    if provider is None:
        provider = fixture_key_provider()
    if cipher is None:
        from vv_agent.session.content_store import AESGCMCipher

        cipher = AESGCMCipher()
    binding = bind_content_runtime(store, key_provider=provider, cipher=cipher)
    return wrap_session_content_store(store, binding)


def create_fixture_session(
    store,
    session_id: str,
    *,
    attributes: dict[str, Any] | None = None,
    workspace: Any = "/tmp",
    consumers: tuple[str, ...] = ("events",),
    subject_id: str = "fixture/host",
    feature_level: int = 1,
    bootstrap: SessionRequirements | None = None,
):
    if store._ledger_integrity_mode() == "unconfigured":
        store.configure_input_admission(allow_unsigned=True)
    attributes = attributes if attributes is not None else {"seed": {"messages": [], "shared_state": {}}}
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=session_id)

        def session_spec(content: SessionContentWriter) -> SessionSpec:
            actor, _ = content.service_principal(subject_id)
            return SessionSpec(
                session_id,
                actor,
                content.put_json(workspace).to_dict(),
                attributes=attributes,
                feature_level=feature_level,
                bootstrap=bootstrap,
            )

        return writer.create_session(session_spec, consumers=consumers)


def push_fixture_input(store, session_id: str, value: InboxItem):
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=session_id)
        subject_kind = "service" if value.kind in {"provider_evidence", "provider_result", "child_result"} else "user"
        actor_attributes = writer.put_json({"role": f"fixture/{value.kind}"})
        actor = Principal(
            tenant_id=store.tenant_id,
            subject_id=f"fixture/{value.kind}",
            subject_kind=subject_kind,
            attributes=actor_attributes,
        ).to_dict()
        published = declare_input_content(writer, replace(value, actor=actor, admission_proof=None))
        receipt = tx.push(session_id, published.item)
        link_declared_input_content(writer, published.item)
        return receipt


def fixture_content_reader(store, session_id: str):
    return durable_content_reader(store, tenant_id=store.tenant_id, session_id=session_id)


def fixture_child_spec(store, parent_session_id: str, session_id: str) -> SessionSpec:
    created = store.read(parent_session_id, limit=1).records[0].record._payload
    return SessionSpec(session_id, created["principal"], created["workspace"], attributes={})


__all__ = [
    "content_ref",
    "create_fixture_session",
    "deterministic_content_producer",
    "deterministic_content_secrets",
    "fixture_child_spec",
    "fixture_content_reader",
    "fixture_key_provider",
    "fixture_key_provider_for_store",
    "principal",
    "push_fixture_input",
    "wrap_fixture_store",
]
