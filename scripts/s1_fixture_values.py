"""Shared S1-safe values and transaction helpers for producer fixtures."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Any

from vv_agent.kernel._content import Principal
from vv_agent.kernel._content_publication import declare_input_content, link_declared_input_content
from vv_agent.session.content_runtime import (
    SessionContentWriter,
    bind_content_key_provider,
    durable_content_reader,
    wrap_session_content_store,
)
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.records import InboxItem, SessionSpec

_FIXTURE_KEY = sha256(b"vv-agent-s1-test-fixtures").digest()


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


def fixture_key_provider() -> LocalKeyProvider:
    return LocalKeyProvider(_FIXTURE_KEY)


def wrap_fixture_store(store, provider: LocalKeyProvider | None = None):
    provider = provider or fixture_key_provider()
    bind_content_key_provider(store, provider)
    return wrap_session_content_store(store, provider)


def create_fixture_session(
    store,
    session_id: str,
    *,
    attributes: dict[str, Any] | None = None,
    workspace: str = "/tmp",
    consumers: tuple[str, ...] = ("events",),
    subject_id: str = "fixture/host",
):
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
    "fixture_child_spec",
    "fixture_content_reader",
    "fixture_key_provider",
    "principal",
    "push_fixture_input",
    "wrap_fixture_store",
]
