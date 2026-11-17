"""Shared S1-safe values and transaction helpers for test fixtures."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Any

from vv_agent.kernel._content import ContentSessionWriter, KeyProvider, Principal
from vv_agent.kernel._content_publication import (
    declare_input_content,
    link_declared_input_content,
    link_declared_record_content,
    publish_record_content,
    publish_session_created_content,
    resolve_record_content_links,
)
from vv_agent.session.content_runtime import (
    SessionContentWriter,
    content_key_provider_for,
    durable_content_reader,
    wrap_session_content_store,
)
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.projection import fold_hydrated_records, hydrate_records, hydration_for_session_store
from vv_agent.session.records import InboxItem, Record, SessionSpec
from vv_agent.session.reducer import _fold_factory_for

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


def wrap_fixture_store(store, provider: KeyProvider | None = None):
    selected = provider if provider is not None else content_key_provider_for(store)
    if selected is None:
        selected = fixture_key_provider()
    return wrap_session_content_store(store, selected)


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
        existing = bool(
            store._rows(
                "SELECT 1 FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                (store.tenant_id, session_id),
            )
        )

        def session_spec(content: ContentSessionWriter) -> SessionSpec:
            actor, _ = content.service_principal(subject_id)
            if existing:
                created = publish_session_created_content(
                    content,
                    principal=actor,
                    workspace=workspace,
                    attributes=attributes,
                )
                return SessionSpec(
                    session_id,
                    created.principal,
                    created.workspace.to_dict(),
                    attributes=created.attributes.to_dict(),
                )
            return SessionSpec(
                session_id,
                actor,
                content.put_json(workspace).to_dict(),
                attributes=attributes,
            )

        return writer.create_session(session_spec, consumers=consumers)


def push_fixture_input(store, session_id: str, value: InboxItem, *, preserve_actor: bool = False):
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=session_id)
        if preserve_actor:
            if value.actor is None:
                raise ValueError("preserve_actor requires an existing Principal")
            actor = value.actor
        else:
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


def append_fixture_records(
    store,
    session_id: str,
    records,
    *,
    commit_id: str = "fixture",
    consume_input_ids: tuple[str, ...] = (),
):
    """Append logical producer records through the S1 content inventory."""

    records = tuple(records)
    head = store.read(session_id, limit=1)
    lease = store.acquire(session_id, owner="s1-fixture-append", ttl_ms=60_000)
    if lease is None:
        raise RuntimeError("fixture could not acquire the session lease")
    try:
        with store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=store.tenant_id, session_id=session_id)
            publication = publish_record_content(writer, records)
            published = list(publication.records)
            request_digests = {
                record.operation_id: record._payload["request_digest"] for record in published if record.kind == "op_planned"
            }
            for index, record in enumerate(published):
                if record.kind == "op_completed" and record.operation_id in request_digests:
                    value = record.to_dict()
                    value["payload"]["request_digest"] = request_digests[record.operation_id]
                    record = Record(**value)
                    record.encode()
                    published[index] = record
            publication = replace(publication, records=tuple(published))
            receipt = tx.append(
                session_id,
                lease=lease,
                expected_seq=head.head_seq,
                commit_id=commit_id,
                records=publication.records,
                consume_input_ids=consume_input_ids,
            )
            links = resolve_record_content_links(publication.links, receipt.record_sequences)
            link_declared_record_content(writer, links)
    finally:
        store.release(lease)
    return receipt


def read_fixture_state(store, session_id: str):
    """Return the logical hydrated state and rows for test assertions."""

    page = store.read(session_id)
    rows = page.records
    hydration = hydration_for_session_store(store, session_id)
    hydrated = hydrate_records(rows, hydration)
    state = fold_hydrated_records(hydrated, rows, _fold_factory_for(store), hydration)
    return state, hydrated, page.inbox_seq


def clear_fixture_state_cache(store) -> None:
    """Clear the raw store cache even when tests hold the content-capability adapter."""

    root = getattr(store, "_root", None) or getattr(store, "_root_store", None) or store
    if hasattr(root, "_fold_cache"):
        root._fold_cache = None


def fixture_content_reader(store, session_id: str):
    return durable_content_reader(store, tenant_id=store.tenant_id, session_id=session_id)


def fixture_child_spec(store, parent_session_id: str, session_id: str) -> SessionSpec:
    created = store.read(parent_session_id, limit=1).records[0].record._payload
    return SessionSpec(session_id, created["principal"], created["workspace"], attributes={})


__all__ = [
    "append_fixture_records",
    "clear_fixture_state_cache",
    "content_ref",
    "create_fixture_session",
    "fixture_child_spec",
    "fixture_content_reader",
    "fixture_key_provider",
    "principal",
    "push_fixture_input",
    "read_fixture_state",
    "wrap_fixture_store",
]
