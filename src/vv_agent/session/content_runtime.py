"""Trusted session content ingress over caller-owned SQL transactions."""

from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from typing import Any

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import (
    Classification,
    ContentLabels,
    ContentRef,
    KeyProvider,
    Principal,
    SourceKind,
)
from vv_agent.kernel._content_store import DurableContentStore
from vv_agent.kernel._records import Record, SessionSpec
from vv_agent.kernel._sql import SQLSessionTx
from vv_agent.kernel._store import Conflict, CreateReceipt, SessionStore, SessionTx

from .content_store import SQLContentStore
from .postgres import PostgresContentStore, PostgresStore
from .sqlite import SQLiteContentStore, SQLiteStore

SESSION_PRINCIPAL_ATTRIBUTES = "session.principal.attributes"
SESSION_WORKSPACE = "session.workspace"
SESSION_ATTRIBUTES = "session.attributes"
INPUT_CONTENT = "input.content"
INPUT_SELECTION = "input.selection"
INPUT_ACTOR_ATTRIBUTES = "input.actor.attributes"
APPROVAL_SCOPE = "approval.scope"
APPROVAL_REASON = "approval.reason"
APPROVAL_METADATA = "approval.metadata"
CHILD_RESULT = "child.result"

_SLOTS = frozenset(
    {
        SESSION_PRINCIPAL_ATTRIBUTES,
        SESSION_WORKSPACE,
        SESSION_ATTRIBUTES,
        INPUT_CONTENT,
        INPUT_SELECTION,
        INPUT_ACTOR_ATTRIBUTES,
        APPROVAL_SCOPE,
        APPROVAL_REASON,
        APPROVAL_METADATA,
        CHILD_RESULT,
    }
)
_PROVIDER_ATTRIBUTE = "_vv_agent_content_key_provider"

INTERNAL_LABELS = ContentLabels(source=(SourceKind.INTERNAL,), classification=Classification.INTERNAL)
HOST_LABELS = ContentLabels(source=(SourceKind.USER,), classification=Classification.INTERNAL)


class ContentRuntimeUnavailable(RuntimeError):
    """The trusted ingress has no durable store or bound key provider."""


def bind_content_key_provider(store: SessionStore, key_provider: KeyProvider) -> None:
    """Attach one caller-injected provider to this store instance."""
    if not isinstance(key_provider, KeyProvider):
        raise TypeError("key_provider must implement the KeyProvider contract")
    root = _root_store(store)
    setattr(root, _PROVIDER_ATTRIBUTE, key_provider)
    if store is not root:
        setattr(store, _PROVIDER_ATTRIBUTE, key_provider)


def content_key_provider_for(store: object) -> KeyProvider | None:
    root = _root_store(store)
    provider = getattr(root, _PROVIDER_ATTRIBUTE, None)
    if provider is None and store is not root:
        provider = getattr(store, _PROVIDER_ATTRIBUTE, None)
    return provider if isinstance(provider, KeyProvider) else None


def durable_content_store(
    store: SessionStore,
    *,
    key_provider: KeyProvider | None = None,
) -> DurableContentStore:
    """Open a caller-selected durable adapter over an existing store instance."""
    root = _root_store(store)
    provider = key_provider or content_key_provider_for(root)
    if provider is None:
        raise ContentRuntimeUnavailable("a content key provider must be explicitly bound to the session store")
    if isinstance(root, SQLiteStore):
        return SQLiteContentStore(root, provider)
    if isinstance(root, PostgresStore):
        return PostgresContentStore(root, provider)
    raise ContentRuntimeUnavailable("content reads require a durable SQLite or PostgreSQL store")


def durable_content_reader(store: SessionStore, *, tenant_id: str, session_id: str):
    from .content_reader import DurableContentReader

    return DurableContentReader(
        durable_content_store(store),
        tenant_id=tenant_id,
        session_id=session_id,
    )


def read_record_json(
    store: SessionStore,
    *,
    tenant_id: str,
    session_id: str,
    record_seq: int,
    ref: ContentRef | dict[str, Any],
    slot: str,
) -> Any:
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=tenant_id, session_id=session_id)
        verified = writer.assert_record(record_seq, ref, slot)
        return writer.get_json(verified)


def read_input_json(
    store: SessionStore,
    *,
    tenant_id: str,
    session_id: str,
    input_id: str,
    ref: ContentRef | dict[str, Any],
    slot: str,
) -> Any:
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=tenant_id, session_id=session_id)
        verified = writer.assert_input(input_id, ref, slot)
        return writer.get_json(verified)


def _root_store(store: object) -> object:
    seen: set[int] = set()
    current: Any = store
    while id(current) not in seen:
        seen.add(id(current))
        parent = getattr(current, "_root_store", None) or getattr(current, "_root", None)
        if parent is None or parent is current:
            return current
        current = parent
    return current


def _slot(value: str) -> str:
    if value not in _SLOTS:
        raise ValueError("content links require a stable declared slot")
    return value


class SessionContentWriter:
    """Content operations bound to one tenant, session, transaction and provider."""

    def __init__(
        self,
        *,
        tx: SessionTx,
        tenant_id: str,
        session_id: str,
        key_provider: KeyProvider,
    ) -> None:
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("tenant_id must be a non-empty string")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(key_provider, KeyProvider):
            raise ContentRuntimeUnavailable("a bound content key provider is required")
        if not isinstance(tx, SQLSessionTx):
            raise ContentRuntimeUnavailable("content writes require a caller-owned SQL session transaction")
        store_tenant = getattr(tx.store, "tenant_id", None)
        if store_tenant is not None and store_tenant != tenant_id:
            raise ValueError("content writer tenant must match the transaction tenant")

        durable: DurableContentStore
        if isinstance(tx.store, SQLiteStore):
            durable = SQLiteContentStore(tx, key_provider)
        elif isinstance(tx.store, PostgresStore):
            durable = PostgresContentStore(tx, key_provider)
        else:
            raise ContentRuntimeUnavailable("content writes require a durable SQLite or PostgreSQL store")
        if not isinstance(durable, SQLContentStore):
            raise ContentRuntimeUnavailable("content store does not implement the S1 SQL contract")

        self.tx = tx
        self.tenant_id = tenant_id
        self.session_id = session_id
        self._key_provider = key_provider
        self.durable = durable

    @classmethod
    def bind(
        cls,
        tx: SessionTx,
        *,
        tenant_id: str,
        session_id: str,
        key_provider: KeyProvider | None = None,
    ) -> SessionContentWriter:
        if not isinstance(tx, SQLSessionTx):
            raise ContentRuntimeUnavailable("content writes require a caller-owned SQL session transaction")
        provider = key_provider or content_key_provider_for(tx.store)
        if provider is None:
            raise ContentRuntimeUnavailable("a content key provider must be explicitly bound to the session store")
        return cls(tx=tx, tenant_id=tenant_id, session_id=session_id, key_provider=provider)

    def create_session_key(self):
        return self.durable.create_session_key(self.tenant_id, self.session_id)

    def put_bytes(self, value: bytes, *, labels: ContentLabels = INTERNAL_LABELS) -> ContentRef:
        return self.durable.put(self.tenant_id, self.session_id, value, labels)

    def put_json(self, value: Any, *, labels: ContentLabels = INTERNAL_LABELS) -> ContentRef:
        return self.put_bytes(canonical_json_bytes(value), labels=labels)

    def get_bytes(self, ref: ContentRef | dict[str, Any]) -> bytes:
        return self.durable.get(self.tenant_id, self.session_id, _content_ref(ref))

    def get_json(self, ref: ContentRef | dict[str, Any]) -> Any:
        try:
            return json.loads(self.get_bytes(ref))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("stored content is not JSON") from exc

    def copy_json_from(self, source: SessionContentWriter, ref: ContentRef | dict[str, Any]) -> ContentRef:
        parsed = _content_ref(ref)
        return self.put_bytes(source.get_bytes(parsed), labels=parsed.labels)

    def service_principal(
        self,
        subject_id: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ContentRef]:
        if not isinstance(subject_id, str) or not subject_id:
            raise ValueError("subject_id must be a non-empty string")
        value = attributes if attributes is not None else {"role": subject_id}
        ref = self.put_json(value)
        principal = Principal(
            tenant_id=self.tenant_id,
            subject_id=subject_id,
            subject_kind="service",
            attributes=ref,
        ).to_dict()
        return principal, ref

    def link_input(self, input_id: str, ref: ContentRef, slot: str) -> None:
        self.durable.link_input(self.tenant_id, self.session_id, input_id, ref, _slot(slot))

    def link_input_actor(self, input_id: str, ref: ContentRef) -> None:
        self.link_input(input_id, ref, INPUT_ACTOR_ATTRIBUTES)

    def link_input_content(self, input_id: str, ref: ContentRef) -> None:
        self.link_input(input_id, ref, INPUT_CONTENT)

    def link_child_result(self, input_id: str, ref: ContentRef) -> None:
        self.link_input(input_id, ref, CHILD_RESULT)

    def link_record(self, record_seq: int, ref: ContentRef, slot: str) -> None:
        self.durable.link_record(self.tenant_id, self.session_id, record_seq, ref, _slot(slot))

    def assert_input(self, input_id: str, ref: ContentRef | dict[str, Any], slot: str) -> ContentRef:
        return self.durable.assert_link(
            self.tenant_id,
            self.session_id,
            _content_ref(ref),
            _slot(slot),
            input_id=input_id,
        )

    def assert_record(self, record_seq: int, ref: ContentRef | dict[str, Any], slot: str) -> ContentRef:
        return self.durable.assert_link(
            self.tenant_id,
            self.session_id,
            _content_ref(ref),
            _slot(slot),
            record_seq=record_seq,
        )

    def create_session(
        self,
        spec_factory: Callable[[SessionContentWriter], SessionSpec],
        *,
        consumers: tuple[str, ...],
    ) -> CreateReceipt:
        """Create session/key/content/record/links inside the current outer transaction."""
        self.tx._active()
        if any(not isinstance(consumer, str) or not consumer for consumer in consumers):
            raise ValueError("consumers must be non-empty strings")
        if len(consumers) != len(set(consumers)):
            raise Conflict("duplicate consumer")

        store = self.tx.store
        tenant_id = self.tenant_id
        session_id = self.session_id
        with store._transaction():
            if not store._rows(
                "SELECT 1 FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                (tenant_id, session_id),
            ):
                inserted = bool(
                    store._rows(
                        "INSERT INTO sk_session(tenant_id,session_id,schema_version,phase,level,deletion_state,created_ms) "
                        "VALUES (%s,%s,'S1','idle',1,'live',%s) RETURNING session_id",
                        (tenant_id, session_id, store._now()),
                    )
                )
            else:
                inserted = False
            store._lock(session_id)

            spec = spec_factory(self)
            if spec.session_id != session_id:
                raise ValueError("session spec does not match the content writer binding")
            principal = Principal.from_dict(spec.principal)
            if principal.tenant_id != tenant_id:
                raise Conflict("session principal belongs to another tenant")

            if inserted:
                store._rows(
                    "UPDATE sk_session SET parent_session_id=%s,parent_operation_id=%s WHERE tenant_id=%s AND session_id=%s",
                    (spec.parent_session_id, spec.parent_operation_id, tenant_id, session_id),
                )
            else:
                parent = store._rows(
                    "SELECT parent_session_id,parent_operation_id FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                    (tenant_id, session_id),
                )
                if not parent or tuple(parent[0]) != (spec.parent_session_id, spec.parent_operation_id):
                    raise Conflict("session parent identity has different bytes")

            existing_body = None
            if not inserted:
                rows = store._rows(
                    "SELECT body FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=1",
                    (tenant_id, session_id),
                )
                if not rows:
                    raise Conflict("session create row is incomplete")
                existing_body = rows[0][0]

            if spec.parent_session_id is not None and not store._rows(
                "SELECT 1 FROM sk_session WHERE tenant_id=%s AND session_id=%s",
                (tenant_id, spec.parent_session_id),
            ):
                raise Conflict("parent session not found in tenant")

            record = spec.record(_existing=Record.parse(existing_body)) if existing_body is not None else spec.record()
            body = canonical_json_bytes({"create": record.to_dict(), "consumers": sorted(consumers)})
            manifest = canonical_json_bytes({"create": record.record_id, "consumers": sorted(consumers)})
            if existing_body is not None:
                old = store._rows(
                    "SELECT body,created_ms FROM sk_commit WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
                    (tenant_id, session_id, "session/create"),
                )
                if not old or old[0][0] != manifest or existing_body != record.encode():
                    raise Conflict("create identity has different bytes")
                receipt = CreateReceipt(session_id, 1, old[0][1], True)
            else:
                now = store._now()
                state = store.fold_factory().extend((record,))
                self.tx._write(
                    record,
                    seq=1,
                    commit_id="session/create",
                    commit_digest=sha256(body).hexdigest(),
                    position=0,
                    epoch=0,
                    now=now,
                )
                self.tx._commit(
                    session_id,
                    "session/create",
                    body,
                    manifest,
                    ((record.record_id, 1),),
                    1,
                    now,
                )
                for consumer in consumers:
                    store._rows(
                        "INSERT INTO sk_consumer(tenant_id,session_id,consumer) VALUES (%s,%s,%s)",
                        (tenant_id, session_id, consumer),
                    )
                store._rows(
                    "UPDATE sk_session SET head_seq=1 WHERE tenant_id=%s AND session_id=%s",
                    (tenant_id, session_id),
                )
                store._schedule(session_id, state)
                receipt = CreateReceipt(session_id, 1, now)

            self.link_record(1, principal.attributes, SESSION_PRINCIPAL_ATTRIBUTES)
            self.link_record(1, _content_ref(spec.workspace), SESSION_WORKSPACE)
            if spec.attributes is None:
                raise ValueError("session attributes require a ContentRef")
            self.link_record(1, _content_ref(spec.attributes), SESSION_ATTRIBUTES)
            return receipt


def _content_ref(value: ContentRef | dict[str, Any]) -> ContentRef:
    return value if isinstance(value, ContentRef) else ContentRef.from_dict(value)


__all__ = [
    "APPROVAL_METADATA",
    "APPROVAL_REASON",
    "APPROVAL_SCOPE",
    "CHILD_RESULT",
    "HOST_LABELS",
    "INPUT_ACTOR_ATTRIBUTES",
    "INPUT_CONTENT",
    "INPUT_SELECTION",
    "INTERNAL_LABELS",
    "SESSION_ATTRIBUTES",
    "SESSION_PRINCIPAL_ATTRIBUTES",
    "SESSION_WORKSPACE",
    "ContentRuntimeUnavailable",
    "SessionContentWriter",
    "bind_content_key_provider",
    "content_key_provider_for",
    "durable_content_reader",
    "durable_content_store",
    "read_input_json",
    "read_record_json",
]
