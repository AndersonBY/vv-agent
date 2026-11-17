from __future__ import annotations

from typing import Any

from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._content_publication import declare_input_content, link_declared_input_content
from vv_agent.kernel._content_runtime import SessionContentWriter as CoreSessionContentWriter
from vv_agent.kernel._store import StoredRecord
from vv_agent.session.content_runtime import (
    SessionContentWriter,
    bind_content_key_provider,
    wrap_session_content_store,
)
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.store import SessionStore
from vv_agent.session.surfaces import SessionDriver

DEVELOPMENT_MASTER_KEY = b"app-server-s1-tests-use-this-key"


def development_key_provider() -> LocalKeyProvider:
    return LocalKeyProvider(DEVELOPMENT_MASTER_KEY)


def bind_development_content(target: SessionStore | SessionDriver) -> LocalKeyProvider:
    """Bind deterministic test-only keys before any session content is written."""

    provider = development_key_provider()
    if isinstance(target, SessionDriver):
        bind_content_key_provider(target._root_store, provider)
        target.content_key_provider = provider
        target.uses_development_content_key_provider = False
        target.store = wrap_session_content_store(target.store, provider)
        return provider
    bind_content_key_provider(target, provider)
    return provider


def create_s1_session(
    store: SessionStore,
    session_id: str,
    attributes: dict[str, Any],
    *,
    workspace: str = ".",
    consumers: tuple[str, ...] = ("app_server", "events", "traces"),
) -> None:
    """Create a session through the transaction-bound durable content writer."""

    tenant_id = getattr(store, "tenant_id", None)
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("S1 test session stores must be tenant-bound")
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=tenant_id, session_id=session_id)

        def session_spec(content: CoreSessionContentWriter) -> SessionSpec:
            principal, _ = content.service_principal("app-server-test")
            return SessionSpec(
                session_id,
                principal,
                content.put_json(workspace).to_dict(),
                attributes=content.put_json(attributes).to_dict(),
            )

        writer.create_session(session_spec, consumers=consumers)


def hydrate_stored_payload(store: SessionStore, session_id: str, stored: StoredRecord) -> dict[str, Any]:
    return hydrate_record_payload(
        stored,
        hydration_for_session_store(store, session_id),
    )


def push_s1_input(
    store: SessionStore,
    session_id: str,
    input_id: str,
    content: Any,
    *,
    kind: str = "user",
    target_turn_id: str | None = None,
    generation: int | None = None,
    available_ms: int = 0,
) -> None:
    """Push host input with its content and actor capabilities in one transaction."""

    tenant_id = getattr(store, "tenant_id", None)
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("S1 test session stores must be tenant-bound")
    with store.atomic() as tx:
        writer = SessionContentWriter.bind(tx, tenant_id=tenant_id, session_id=session_id)
        actor, actor_ref = writer.service_principal("app-server-test-user")
        payload = {"content": content} if kind in {"user", "steer", "follow_up"} else content
        if not isinstance(payload, dict):
            raise TypeError(f"{kind} input payload must be an object")
        published = declare_input_content(
            writer,
            InboxItem(
                input_id,
                kind,
                payload,
                target_turn_id,
                generation,
                available_ms,
                actor=actor,
            ),
        )
        tx.push(session_id, published.item)
        link_declared_input_content(writer, published.item)
        writer.link_input_actor(input_id, actor_ref)


def declared_content_ref(value: Any) -> ContentRef:
    return ContentRef.from_dict(value)


__all__ = [
    "DEVELOPMENT_MASTER_KEY",
    "bind_development_content",
    "create_s1_session",
    "declared_content_ref",
    "development_key_provider",
    "hydrate_stored_payload",
    "push_s1_input",
]
