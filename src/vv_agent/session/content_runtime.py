"""SDK defaults over the core S1 content runtime."""

from __future__ import annotations

from typing import cast

from vv_agent.kernel._content import ContentRef, KeyProvider
from vv_agent.kernel._content_runtime import (
    APPROVAL_METADATA,
    APPROVAL_REASON,
    APPROVAL_SCOPE,
    CHILD_RESULT,
    HOST_LABELS,
    INPUT_ACTOR_ATTRIBUTES,
    INPUT_CONTENT,
    INPUT_SELECTION,
    INTERNAL_LABELS,
    SESSION_ATTRIBUTES,
    SESSION_PRINCIPAL_ATTRIBUTES,
    SESSION_WORKSPACE,
    ContentRuntimeBinding,
    ContentRuntimeUnavailable,
    _ContentSessionStoreAdapter,
    _ContentSessionTxAdapter,
    _DeferredContentBackend,
    _ValidatedDurableContentStore,
    bind_content_runtime,
    content_key_provider_for,
    content_runtime_binding_for,
    durable_content_reader,
    read_input_json,
    read_record_json,
)
from vv_agent.kernel._content_runtime import (
    CONTENT_RUNTIME_BINDING_ATTRIBUTE as CONTENT_RUNTIME_BINDING_ATTRIBUTE,
)
from vv_agent.kernel._content_runtime import (
    SessionContentWriter as CoreSessionContentWriter,
)
from vv_agent.kernel._content_runtime import (
    durable_content_store as core_durable_content_store,
)
from vv_agent.kernel._content_runtime import (
    wrap_session_content_store as core_wrap_session_content_store,
)
from vv_agent.kernel._sql import SQLSessionTx
from vv_agent.kernel._store import ContentSessionTx, SessionStore, SessionTx

from .content_store import AESGCMCipher


class SessionContentWriter(CoreSessionContentWriter):
    """SDK writer preserving the existing explicit key-provider constructor."""

    def __init__(
        self,
        *,
        tx: SessionTx,
        tenant_id: str,
        session_id: str,
        key_provider: KeyProvider | None = None,
        binding: ContentRuntimeBinding | None = None,
        _published_refs: set[ContentRef] | None = None,
    ) -> None:
        if binding is None:
            if key_provider is None:
                raise ContentRuntimeUnavailable("a content key provider is required")
            binding = ContentRuntimeBinding(key_provider, AESGCMCipher())
        elif key_provider is not None:
            raise ValueError("key_provider and binding cannot both be provided")
        super().__init__(
            tx=tx,
            tenant_id=tenant_id,
            session_id=session_id,
            binding=binding,
            _published_refs=_published_refs,
        )

    @classmethod
    def bind(
        cls,
        tx: SessionTx,
        *,
        tenant_id: str,
        session_id: str,
        binding: ContentRuntimeBinding | None = None,
        key_provider: KeyProvider | None = None,
    ) -> SessionContentWriter:
        if isinstance(tx, ContentSessionTx):
            if binding is not None or key_provider is not None:
                raise ValueError("content runtime is already bound by the transaction content capability")
            return cast(SessionContentWriter, tx.content_writer(tenant_id=tenant_id, session_id=session_id))
        if not isinstance(tx, SQLSessionTx):
            raise ContentRuntimeUnavailable("content writes require a caller-owned SQL session transaction")
        if binding is not None and key_provider is not None:
            raise ValueError("key_provider and binding cannot both be provided")
        selected = binding or content_runtime_binding_for(tx.store)
        if key_provider is not None:
            selected = ContentRuntimeBinding(key_provider, AESGCMCipher())
        if selected is None:
            raise ContentRuntimeUnavailable("a content key provider must be explicitly bound to the session store")
        return cls(tx=tx, tenant_id=tenant_id, session_id=session_id, binding=selected)


def bind_content_key_provider(store: SessionStore, key_provider: KeyProvider) -> None:
    """Attach the SDK production cipher with the legacy provider-only API."""

    current = content_runtime_binding_for(store)
    if current is not None and current._object_content is not None:
        if current.key_provider is not key_provider:
            raise ContentRuntimeUnavailable("external content cannot switch key providers")
        return
    bind_content_runtime(store, key_provider=key_provider, cipher=AESGCMCipher())


def durable_content_store(
    store: SessionStore,
    *,
    key_provider: KeyProvider | None = None,
):
    """Preserve the provider override while using the core durable store owner."""

    if key_provider is not None:
        bind_content_key_provider(store, key_provider)
    return core_durable_content_store(store)


def wrap_session_content_store(store: SessionStore, key_provider: KeyProvider) -> SessionStore:
    bind_content_key_provider(store, key_provider)
    return core_wrap_session_content_store(store)


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
    "ContentRuntimeBinding",
    "ContentRuntimeUnavailable",
    "SessionContentWriter",
    "_ContentSessionStoreAdapter",
    "_ContentSessionTxAdapter",
    "_DeferredContentBackend",
    "_ValidatedDurableContentStore",
    "bind_content_key_provider",
    "bind_content_runtime",
    "content_key_provider_for",
    "content_runtime_binding_for",
    "durable_content_reader",
    "durable_content_store",
    "read_input_json",
    "read_record_json",
    "wrap_session_content_store",
]
