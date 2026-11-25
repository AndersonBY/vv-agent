"""Core S1 content runtime over caller-owned SQL transactions."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Protocol, cast

from ._canonical_json import canonical_json_bytes
from ._content import (
    APPROVAL_METADATA,
    APPROVAL_REASON,
    APPROVAL_SCOPE,
    CHILD_RESULT,
    INPUT_ACTOR_ATTRIBUTES,
    INPUT_CONTENT,
    INPUT_SELECTION,
    SESSION_ATTRIBUTES,
    SESSION_PRINCIPAL_ATTRIBUTES,
    SESSION_WORKSPACE,
    Classification,
    ContentLabels,
    ContentRef,
    ContentSessionWriter,
    ContentUnavailable,
    KeyProvider,
    KeyUnavailable,
    Principal,
    SessionHead,
    SourceKind,
)
from ._content_publication import (
    _BOUNDARY_CONTENT_SLOTS,
    _CONTENT_SLOTS,
    _INPUT_SLOTS,
    PROVIDER_EVIDENCE,
    PROVIDER_HANDLE_CANCEL_REF,
    PROVIDER_HANDLE_EVIDENCE,
    PROVIDER_HANDLE_JOB_ID,
    PROVIDER_HANDLE_QUERY_REF,
    PROVIDER_RESULT,
    PROVIDER_USAGE,
    ContentSlot,
    input_record_slots,
    publish_session_created_content,
)
from ._content_reader import _LinkValidatedContentStore, _OwnerBoundContentStore
from ._content_sql import CoreSQLContentStore, _ContentWriteMemo
from ._content_store import ContentCipher, DurableContentStore
from ._external_content import _PostgresObjectContent, _SQLiteObjectContent
from ._lease_scope import Clock
from ._postgres import CorePostgresContentStore, PostgresStore
from ._records import SessionSpec
from ._sql import SQLSessionTx, _ObjectDeadline, _TenantSQLStore
from ._sqlite import CoreSQLiteContentStore, SQLiteStore
from ._store import (
    CommitReceipt,
    ConsumerBatch,
    ContentSessionTx,
    CreateReceipt,
    InputReceipt,
    Lease,
    SessionStore,
    SessionTx,
)

CONTENT_RUNTIME_BINDING_ATTRIBUTE = "_vv_agent_content_runtime_binding"
_INDEXED_SLOT = re.compile(r"^(?P<base>[^\[\]]+)(?P<indices>(?:\[(?:0|[1-9][0-9]*)\])*)$")
_SLOT_INDEX = re.compile(r"\[(?:0|[1-9][0-9]*)\]")
_DEFERRED_SESSION_LIVE_SQL = "SELECT deletion_state FROM sk_session WHERE tenant_id={p} AND session_id={p}"
_DEFERRED_ADMISSION_TABLES = frozenset({"sk_session", "sk_session_key", "sk_content", "sk_content_link", "sk_record", "sk_inbox"})

_DEFAULT_LABELS = ContentLabels()
INTERNAL_LABELS = ContentLabels(source=(SourceKind.INTERNAL,), classification=Classification.INTERNAL)
HOST_LABELS = ContentLabels(source=(SourceKind.USER,), classification=Classification.INTERNAL)


class ContentRuntimeUnavailable(RuntimeError):
    """The trusted ingress has no durable store or bound content runtime."""


class _DriveStoreFactoryProvider(Protocol):
    def drive_store_factory(self) -> Callable[[], AbstractContextManager[SessionStore]]: ...


@dataclass(frozen=True, slots=True)
class ContentRuntimeBinding:
    """Explicit key provider and cipher pair attached to one store root."""

    key_provider: KeyProvider
    cipher: ContentCipher
    _object_content: _SQLiteObjectContent | _PostgresObjectContent | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key_provider, KeyProvider):
            raise TypeError("key_provider must implement the KeyProvider contract")


class _DeferredContentBackend:
    """Let SQLContentStore allocate target content before the session row exists.

    SQLSessionTx.create remains the admission authority. This backend only
    captures the existing content store's own SQL statements and replays them
    after the create transaction has admitted the session. SQLContentStore has
    no public pre-admission staging API today, so this adapter recognizes the
    small set of backend operations it emits and rejects any other operation on
    an admission table instead of delegating it to a session that does not yet
    exist.
    """

    def __init__(self, backend: Any, *, tenant_id: str, session_id: str) -> None:
        self._backend = backend
        self._tenant_id = tenant_id
        self._session_id = session_id
        self._key_values: dict[str, Any] | None = None
        self._content: dict[str, dict[str, Any]] = {}
        self._mutations: list[tuple[str, tuple[Any, ...]]] = []
        self.lock_clause = backend.lock_clause

    @staticmethod
    def _storage_error() -> RuntimeError:
        return RuntimeError("unexpected content SQL before session admission")

    @staticmethod
    def _select_columns(query: str) -> tuple[str, ...]:
        selected, separator, _rest = query.partition(" FROM ")
        if not separator or not selected.startswith("SELECT "):
            raise _DeferredContentBackend._storage_error()
        return tuple(column.strip() for column in selected.removeprefix("SELECT ").split(","))

    @classmethod
    def _insert_values(cls, query: str, params: tuple[Any, ...]) -> dict[str, Any]:
        _prefix, separator, suffix = query.partition("(")
        columns, separator_after_columns, _rest = suffix.partition(")")
        if not separator or not separator_after_columns:
            raise cls._storage_error()
        return dict(zip((column.strip() for column in columns.split(",")), params, strict=True))

    @classmethod
    def _select_row(cls, query: str, values: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
        if values is None:
            return None
        columns = cls._select_columns(query)
        try:
            return tuple(values[column] for column in columns)
        except KeyError as exc:
            raise cls._storage_error() from exc

    def _reads_admission_table(self, query: str) -> bool:
        padded = f" {query} "
        return any(f" {table} " in padded for table in _DEFERRED_ADMISSION_TABLES)

    def fetchone(self, query: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
        query = " ".join(query.split())
        if query == _DEFERRED_SESSION_LIVE_SQL and params == (self._tenant_id, self._session_id):
            return ("live",)
        if " FROM sk_session_key " in f" {query} ":
            if params != (self._tenant_id, self._session_id):
                raise self._storage_error()
            return self._select_row(query, self._key_values)
        if " FROM sk_content WHERE " in f" {query} ":
            return self._fetch_content(query, params)
        if self._reads_admission_table(query):
            raise self._storage_error()
        return self._backend.fetchone(query, params)

    def _fetch_content(self, query: str, params: tuple[Any, ...]) -> tuple[Any, ...] | None:
        if params[:2] != (self._tenant_id, self._session_id):
            raise self._storage_error()
        if len(params) == 3:
            values = self._content.get(cast(str, params[2]))
        elif len(params) == 5:
            key_id, mac, labels = params[2:]
            values = next(
                (
                    candidate
                    for candidate in self._content.values()
                    if (candidate["key_id"], candidate["mac"], candidate["labels"]) == (key_id, mac, labels)
                ),
                None,
            )
        else:
            raise self._storage_error()
        return self._select_row(query, values)

    def fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        if self._reads_admission_table(" ".join(query.split())):
            raise self._storage_error()
        return self._backend.fetchall(query, params)

    def run(self, query: str, params: tuple[Any, ...] = ()) -> int:
        query = " ".join(query.split())
        if query.startswith("INSERT INTO sk_session_key("):
            self._key_values = self._insert_values(query, params)
            self._mutations.append((query, params))
            return 1
        if query.startswith("INSERT INTO sk_content("):
            values = self._insert_values(query, params)
            self._content[cast(str, values["content_id"])] = values
            self._mutations.append((query, params))
            return 1
        if query.startswith("UPDATE sk_content SET state='live'"):
            _tenant, _session, content_id = params
            values = self._content[cast(str, content_id)]
            values["state"] = "live"
            values["pin_until_ms"] = None
            self._mutations.append((query, params))
            return 1
        if query.startswith("UPDATE sk_content SET pin_until_ms"):
            pin_until_ms, _tenant, _session, content_id = params
            self._content[cast(str, content_id)]["pin_until_ms"] = pin_until_ms
            self._mutations.append((query, params))
            return 1
        if query.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            raise self._storage_error()
        return self._backend.run(query, params)

    def require_transaction(self) -> None:
        self._backend.require_transaction()

    def now_ms(self) -> int:
        return self._backend.now_ms()

    def replay(self) -> None:
        for query, params in self._mutations:
            self._backend.run(query, params)


def _deferred_content_writer(writer: SessionContentWriter) -> tuple[SessionContentWriter, _DeferredContentBackend]:
    backend = _DeferredContentBackend(
        writer._durable._backend,
        tenant_id=writer.tenant_id,
        session_id=writer.session_id,
    )
    durable = CoreSQLContentStore(
        backend,
        writer._binding.key_provider,
        writer._binding.cipher,
        provider_id=writer._durable._provider_id,
    )
    durable.bind_transaction(writer.tx, write_memo=writer._write_memo)
    if writer._binding._object_content is not None:
        durable._bind_object_content(writer._binding._object_content, check_write=writer._durable._check_object_write)
    deferred = SessionContentWriter(
        tx=writer.tx,
        tenant_id=writer.tenant_id,
        session_id=writer.session_id,
        binding=writer._binding,
        _published_refs=writer._published_refs,
        _write_memo=writer._write_memo,
        _check_publication=writer._durable._check_object_write,
    )
    deferred._durable = durable
    deferred.durable = cast(DurableContentStore, _ValidatedDurableContentStore(durable))
    if isinstance(writer._binding._object_content, _PostgresObjectContent):
        deferred._creation_json = {}
    return deferred, backend


def _add_slot_spec(specs: dict[str, int], slot: str, index_count: int) -> None:
    previous = specs.setdefault(slot, index_count)
    if previous != index_count:
        raise RuntimeError(f"conflicting content inventory slot: {slot}")


def _content_slot_specs(content_slots: Iterable[ContentSlot]) -> dict[str, int]:
    specs: dict[str, int] = {}
    for content_slot in content_slots:
        _add_slot_spec(specs, content_slot.slot, content_slot.path.count("*"))
        for nested_slot in content_slot.nested:
            _add_slot_spec(
                specs,
                f"{content_slot.slot}.{nested_slot.slot}",
                nested_slot.path.count("*"),
            )
    return specs


_record_slot_specs = _content_slot_specs(
    content_slot
    for content_slots in (*_CONTENT_SLOTS.values(), *_BOUNDARY_CONTENT_SLOTS.values())
    for content_slot in content_slots
)
for _provider_record_slot in (
    PROVIDER_EVIDENCE,
    PROVIDER_HANDLE_CANCEL_REF,
    PROVIDER_HANDLE_EVIDENCE,
    PROVIDER_HANDLE_JOB_ID,
    PROVIDER_HANDLE_QUERY_REF,
    PROVIDER_RESULT,
    PROVIDER_USAGE,
):
    _add_slot_spec(_record_slot_specs, _provider_record_slot, 0)
for _input_kind in _INPUT_SLOTS:
    for _record_slot in input_record_slots(_input_kind):
        _add_slot_spec(_record_slot_specs, _record_slot.slot, _record_slot.path.count("*"))
_RECORD_SLOT_SPECS: Mapping[str, int] = MappingProxyType(_record_slot_specs)

_input_slot_specs = _content_slot_specs(content_slot for content_slots in _INPUT_SLOTS.values() for content_slot in content_slots)
_add_slot_spec(_input_slot_specs, INPUT_ACTOR_ATTRIBUTES, 0)
_INPUT_SLOT_SPECS: Mapping[str, int] = MappingProxyType(_input_slot_specs)


def _verify_content_binding(store: object, binding: ContentRuntimeBinding | None) -> None:
    root = _root_store(store)
    objects = None if binding is None else binding._object_content
    if isinstance(root, PostgresStore):
        if isinstance(objects, _PostgresObjectContent):
            root._verify_object_journal(objects.journal, objects._binding_digest())
            assert binding is not None
            objects.verify_installation(objects.journal.identity.database_id, binding.key_provider)
        elif objects is not None:
            raise ContentUnavailable("PostgreSQL requires its PostgreSQL object journal")
        else:
            installation = root._object_installation()
            if installation is not None and installation[1] is not None:
                raise ContentUnavailable("PostgreSQL installation requires its external content binding")
        return
    if not isinstance(root, SQLiteStore):
        if objects is not None:
            raise ContentUnavailable("external content requires a file-backed SQLite store")
        return
    installation = root._object_installation()
    if objects is None:
        if installation is not None and installation[1] is not None:
            raise ContentUnavailable("SQLite installation requires its external content binding")
        return
    if not isinstance(objects, _SQLiteObjectContent):
        raise ContentUnavailable("SQLite requires its SQLite object journal")
    if installation is None or installation[1] is None:
        raise ContentUnavailable("SQLite external content binding has not been provisioned")
    database_id, expected = installation
    assert binding is not None
    if objects.verify_installation(database_id, binding.key_provider) != expected:
        raise ContentUnavailable("external content binding does not match the SQLite installation")


def _provision_sqlite_object_content(store: SQLiteStore, *, key_provider: KeyProvider, objects: _SQLiteObjectContent) -> None:
    """Pin the object configuration on an installed, quiesced primary."""
    store._provision_object_content(lambda database_id: objects.verify_installation(database_id, key_provider))


def _provision_postgres_object_content(
    store: PostgresStore, *, key_provider: KeyProvider, objects: _PostgresObjectContent
) -> None:
    digest = objects.verify_installation(objects.journal.identity.database_id, key_provider)
    store._provision_object_journal(objects.journal, content_digest=digest)


def bind_content_runtime(
    store: SessionStore,
    *,
    key_provider: KeyProvider,
    cipher: ContentCipher,
    _object_content: _SQLiteObjectContent | _PostgresObjectContent | None = None,
) -> ContentRuntimeBinding:
    """Attach one explicit provider/cipher pair to a store's process-local root."""

    root = _root_store(store)
    binding = ContentRuntimeBinding(key_provider, cipher, _object_content)
    _verify_content_binding(root, binding)
    setattr(root, CONTENT_RUNTIME_BINDING_ATTRIBUTE, binding)
    if store is not root:
        setattr(store, CONTENT_RUNTIME_BINDING_ATTRIBUTE, binding)
    return binding


def content_runtime_binding_for(store: object) -> ContentRuntimeBinding | None:
    if isinstance(store, _ContentSessionStoreAdapter):
        return store._binding
    root = _root_store(store)
    binding = getattr(root, CONTENT_RUNTIME_BINDING_ATTRIBUTE, None)
    if binding is None and store is not root:
        binding = getattr(store, CONTENT_RUNTIME_BINDING_ATTRIBUTE, None)
    return binding if isinstance(binding, ContentRuntimeBinding) else None


def content_key_provider_for(store: object) -> KeyProvider | None:
    binding = content_runtime_binding_for(store)
    return None if binding is None else binding.key_provider


def durable_content_store(
    store: SessionStore,
    *,
    binding: ContentRuntimeBinding | None = None,
) -> DurableContentStore:
    """Open a caller-selected durable adapter over an existing store instance."""

    root = _root_store(store)
    selected = binding or content_runtime_binding_for(store)
    if selected is None:
        raise ContentRuntimeUnavailable("a content key provider must be explicitly bound to the session store")
    _verify_content_binding(root, selected)
    if isinstance(root, SQLiteStore):
        source = store._store if isinstance(store, _ContentSessionStoreAdapter) else store
        if not isinstance(source, (SQLiteStore, _TenantSQLStore)):
            raise ContentRuntimeUnavailable("SQLite tenant view must retain its concrete runtime binding")
        durable: CoreSQLContentStore = CoreSQLiteContentStore(
            source,
            selected.key_provider,
            selected.cipher,
        )
    elif isinstance(root, PostgresStore):
        source = store._store if isinstance(store, _ContentSessionStoreAdapter) else store
        if not isinstance(source, (PostgresStore, _TenantSQLStore)):
            raise ContentRuntimeUnavailable("PostgreSQL tenant view must retain its concrete runtime binding")
        durable = CorePostgresContentStore(source, selected.key_provider, selected.cipher)
    else:
        raise ContentRuntimeUnavailable("content reads require a durable SQLite or PostgreSQL store")
    if selected._object_content is not None:
        durable._bind_object_content(selected._object_content)
    return cast(DurableContentStore, _ValidatedDurableContentStore(durable))


def durable_content_reader(store: SessionStore, *, tenant_id: str, session_id: str):
    from ._content_reader import DurableContentReader

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
    ref: ContentRef | Mapping[str, Any],
    slot: str,
) -> Any:
    if _uses_postgres_objects(store):
        reader = durable_content_reader(store, tenant_id=tenant_id, session_id=session_id)
        with reader._read_frame(record_seqs=(record_seq,)) as frame:
            return json.loads(frame.read(tenant_id, session_id, f"record:{record_seq}", slot, _content_ref(ref)))
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
    ref: ContentRef | Mapping[str, Any],
    slot: str,
) -> Any:
    if _uses_postgres_objects(store):
        reader = durable_content_reader(store, tenant_id=tenant_id, session_id=session_id)
        with reader._read_frame(input_ids=(input_id,)) as frame:
            return json.loads(frame.read(tenant_id, session_id, f"input:{input_id}", slot, _content_ref(ref)))
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


def _uses_postgres_objects(store: object) -> bool:
    binding = content_runtime_binding_for(store)
    return binding is not None and isinstance(binding._object_content, _PostgresObjectContent)


def _slot(value: str, specs: Mapping[str, int]) -> str:
    match = _INDEXED_SLOT.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError("content links require a stable declared slot")
    base = match.group("base")
    declared_indexes = specs.get(base)
    if declared_indexes is None or declared_indexes != len(_SLOT_INDEX.findall(match.group("indices"))):
        raise ValueError("content links require a stable declared slot")
    return value


class _ValidatedContentFrame:
    """Apply the writer's declared owner-slot inventory to one read frame."""

    _supports_link_validated_get = True

    def __init__(self, frame: _LinkValidatedContentStore) -> None:
        self._frame = frame

    def assert_link(
        self,
        tenant_id: str,
        session_id: str,
        ref: ContentRef,
        slot: str,
        *,
        record_seq: int | None = None,
        input_id: str | None = None,
    ) -> ContentRef:
        specs = _INPUT_SLOT_SPECS if record_seq is None and input_id is not None else _RECORD_SLOT_SPECS
        return self._frame.assert_link(
            tenant_id,
            session_id,
            ref,
            _slot(slot, specs),
            record_seq=record_seq,
            input_id=input_id,
        )

    def get(
        self,
        tenant_id: str,
        session_id: str,
        ref: ContentRef,
        *,
        link_validated: bool = False,
    ) -> bytes:
        return self._frame.get(tenant_id, session_id, ref, link_validated=link_validated)


class _ValidatedDurableContentStore:
    """Keep direct durable link calls on the writer's owner/slot inventory."""

    def __init__(self, durable: CoreSQLContentStore) -> None:
        self._durable = durable

    def link_record(
        self,
        tenant_id: str,
        session_id: str,
        record_seq: int,
        ref: ContentRef,
        slot: str,
    ) -> None:
        self._durable.link_record(tenant_id, session_id, record_seq, ref, _slot(slot, _RECORD_SLOT_SPECS))

    def link_records(
        self,
        tenant_id: str,
        session_id: str,
        links: Iterable[tuple[int, ContentRef, str]],
    ) -> None:
        validated = tuple((record_seq, ref, _slot(slot, _RECORD_SLOT_SPECS)) for record_seq, ref, slot in links)
        self._durable.link_records(tenant_id, session_id, validated)

    def link_input(
        self,
        tenant_id: str,
        session_id: str,
        input_id: str,
        ref: ContentRef,
        slot: str,
    ) -> None:
        self._durable.link_input(tenant_id, session_id, input_id, ref, _slot(slot, _INPUT_SLOT_SPECS))

    def assert_link(
        self,
        tenant_id: str,
        session_id: str,
        ref: ContentRef,
        slot: str,
        *,
        record_seq: int | None = None,
        input_id: str | None = None,
    ) -> ContentRef:
        specs = _INPUT_SLOT_SPECS if record_seq is None and input_id is not None else _RECORD_SLOT_SPECS
        return self._durable.assert_link(
            tenant_id,
            session_id,
            ref,
            _slot(slot, specs),
            record_seq=record_seq,
            input_id=input_id,
        )

    def get(self, tenant_id: str, session_id: str, ref: ContentRef) -> bytes:
        return self._durable.get(tenant_id, session_id, ref)

    @contextmanager
    def _read_frame(
        self,
        tenant_id: str,
        session_id: str,
        *,
        record_seqs: Iterable[int] = (),
        input_ids: Iterable[str] = (),
        validate_each_read: bool = True,
    ) -> Iterator[_OwnerBoundContentStore]:
        opener = getattr(self._durable, "_read_frame", None)
        if not callable(opener):
            yield self
            return
        with opener(
            tenant_id,
            session_id,
            record_seqs=record_seqs,
            input_ids=input_ids,
            validate_each_read=validate_each_read,
        ) as frame:
            yield _ValidatedContentFrame(frame)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._durable, name)


class SessionContentWriter:
    """Content operations bound to one tenant, session, transaction and provider."""

    def __init__(
        self,
        *,
        tx: SessionTx,
        tenant_id: str,
        session_id: str,
        binding: ContentRuntimeBinding,
        _published_refs: set[ContentRef] | None = None,
        _write_memo: _ContentWriteMemo | None = None,
        _check_publication: Callable[[], None] | None = None,
    ) -> None:
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("tenant_id must be a non-empty string")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(tx, SQLSessionTx):
            raise ContentRuntimeUnavailable("content writes require a caller-owned SQL session transaction")
        store_tenant = getattr(tx.store, "tenant_id", None)
        if store_tenant is not None and store_tenant != tenant_id:
            raise ValueError("content writer tenant must match the transaction tenant")

        durable: CoreSQLContentStore
        if isinstance(tx.store, SQLiteStore):
            durable = CoreSQLiteContentStore(tx, binding.key_provider, binding.cipher)
        elif isinstance(tx.store, PostgresStore):
            durable = CorePostgresContentStore(tx, binding.key_provider, binding.cipher)
        else:
            raise ContentRuntimeUnavailable("content writes require a durable SQLite or PostgreSQL store")
        if not isinstance(durable, CoreSQLContentStore):
            raise ContentRuntimeUnavailable("content store does not implement the S1 SQL contract")

        self.tx = tx
        self.tenant_id = tenant_id
        self.session_id = session_id
        self._binding = binding
        self._key_provider = binding.key_provider
        self._durable: CoreSQLContentStore = durable
        self._write_memo = _write_memo if _write_memo is not None else _ContentWriteMemo()
        self._durable.bind_transaction(tx, write_memo=self._write_memo)
        if isinstance(binding._object_content, _PostgresObjectContent):
            if _check_publication is None:
                raise ContentRuntimeUnavailable("PostgreSQL object writes require the root session publication transaction")
            _check_publication()
            self._durable._bind_object_content(binding._object_content, check_write=_check_publication)
        elif binding._object_content is None:
            _verify_content_binding(tx.store, binding)
        else:
            if not isinstance(tx.store, SQLiteStore):
                raise ContentRuntimeUnavailable("external content requires a SQLite write transaction")
            sqlite = tx.store
            sqlite._require_content_write_fence(tx)
            deadline = sqlite.clock.monotonic() + 5

            def check_deadline() -> None:
                if sqlite.clock.monotonic() >= deadline:
                    raise KeyUnavailable("external content persistence budget expired")

            tx._bind_external_deadline(
                check_deadline, (int(deadline * 1_000_000_000), lambda: int(sqlite.clock.monotonic() * 1_000_000_000))
            )
            sqlite._require_content_write_fence(tx)
            _verify_content_binding(tx.store, binding)
            sqlite._require_content_write_fence(tx)
            self._durable._bind_object_content(
                binding._object_content, check_write=lambda: sqlite._require_content_write_fence(tx)
            )
        self.durable = cast(DurableContentStore, _ValidatedDurableContentStore(durable))
        self._published_refs = _published_refs if _published_refs is not None else set()
        self._creation_json: dict[ContentRef, bytes] | None = None

    @classmethod
    def bind(
        cls,
        tx: SessionTx,
        *,
        tenant_id: str,
        session_id: str,
        binding: ContentRuntimeBinding | None = None,
    ) -> SessionContentWriter:
        if isinstance(tx, ContentSessionTx):
            if binding is not None:
                raise ValueError("content runtime is already bound by the transaction capability")
            return cast(SessionContentWriter, tx.content_writer(tenant_id=tenant_id, session_id=session_id))
        if not isinstance(tx, SQLSessionTx):
            raise ContentRuntimeUnavailable("content writes require a caller-owned SQL session transaction")
        selected = binding or content_runtime_binding_for(tx.store)
        if selected is None:
            raise ContentRuntimeUnavailable("a content key provider must be explicitly bound to the session store")
        return cls(tx=tx, tenant_id=tenant_id, session_id=session_id, binding=selected)

    def create_session_key(self):
        return self._durable.create_session_key(self.tenant_id, self.session_id)

    def command_digest(
        self,
        *,
        command_spec: dict[str, Any],
        environment: dict[str, Any],
        operational_policy: dict[str, str],
    ) -> str:
        """Return the digest inside this writer's caller-owned transaction.

        The writer supplies the authenticated tenant and session, so callers
        cannot substitute either scope or provide a session data key directly.
        """

        self.tx._active()
        return self._durable.command_digest(
            self.tenant_id,
            self.session_id,
            command_spec=command_spec,
            environment=environment,
            operational_policy=operational_policy,
        )

    def put_bytes(self, value: bytes, *, labels: ContentLabels = _DEFAULT_LABELS) -> ContentRef:
        ref = self._durable.put(self.tenant_id, self.session_id, value, labels)
        self._published_refs.add(ref)
        return ref

    def put_json(self, value: Any, *, labels: ContentLabels = _DEFAULT_LABELS) -> ContentRef:
        body = canonical_json_bytes(value)
        ref = self.put_bytes(body, labels=labels)
        if self._creation_json is not None:
            self._creation_json[ref] = body
        return ref

    def get_bytes(self, ref: ContentRef | Mapping[str, Any]) -> bytes:
        return self._durable.get(self.tenant_id, self.session_id, _content_ref(ref))

    def get_json(self, ref: ContentRef | Mapping[str, Any]) -> Any:
        try:
            if self._creation_json is not None:
                self.tx._active()
                body = self._creation_json.get(_content_ref(ref))
                if body is not None:
                    return json.loads(body)
            return json.loads(self.get_bytes(ref))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("stored content is not JSON") from exc

    def copy_json_from(self, source: ContentSessionWriter, ref: ContentRef | Mapping[str, Any]) -> ContentRef:
        parsed = _content_ref(ref)
        return self.put_json(source.get_json(parsed), labels=parsed.labels)

    def service_principal(
        self,
        subject_id: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ContentRef]:
        if not isinstance(subject_id, str) or not subject_id:
            raise ValueError("subject_id must be a non-empty string")
        value = attributes if attributes is not None else {"role": subject_id}
        ref = self.put_json(value, labels=INTERNAL_LABELS)
        principal = Principal(
            tenant_id=self.tenant_id,
            subject_id=subject_id,
            subject_kind="service",
            attributes=ref,
        ).to_dict()
        return principal, ref

    def link_input(self, input_id: str, ref: ContentRef, slot: str) -> None:
        self._durable.link_input(
            self.tenant_id,
            self.session_id,
            input_id,
            ref,
            _slot(slot, _INPUT_SLOT_SPECS),
        )

    def link_input_actor(self, input_id: str, ref: ContentRef) -> None:
        self.link_input(input_id, ref, INPUT_ACTOR_ATTRIBUTES)

    def link_input_content(self, input_id: str, ref: ContentRef) -> None:
        self.link_input(input_id, ref, INPUT_CONTENT)

    def link_child_result(self, input_id: str, ref: ContentRef) -> None:
        self.link_input(input_id, ref, CHILD_RESULT)

    def link_record(self, record_seq: int, ref: ContentRef, slot: str) -> None:
        self._durable.link_record(
            self.tenant_id,
            self.session_id,
            record_seq,
            ref,
            _slot(slot, _RECORD_SLOT_SPECS),
        )

    def link_records(self, links: Iterable[tuple[int, ContentRef, str]]) -> None:
        validated = tuple((record_seq, ref, _slot(slot, _RECORD_SLOT_SPECS)) for record_seq, ref, slot in links)
        self._durable.link_records(self.tenant_id, self.session_id, validated)

    def assert_input(self, input_id: str, ref: ContentRef | Mapping[str, Any], slot: str) -> ContentRef:
        return self._durable.assert_link(
            self.tenant_id,
            self.session_id,
            _content_ref(ref),
            _slot(slot, _INPUT_SLOT_SPECS),
            input_id=input_id,
        )

    def assert_record(self, record_seq: int, ref: ContentRef | Mapping[str, Any], slot: str) -> ContentRef:
        return self._durable.assert_link(
            self.tenant_id,
            self.session_id,
            _content_ref(ref),
            _slot(slot, _RECORD_SLOT_SPECS),
            record_seq=record_seq,
        )

    def _require_child_creation(self) -> None:
        if self._binding._object_content is not None:
            raise ContentRuntimeUnavailable("external content child admission is unavailable")

    def _lock_creation_identity(self) -> None:
        # SQLite already serializes writers at the outer write transaction.
        self.tx._active()
        store = self.tx.store
        if isinstance(store, PostgresStore):
            store._lock_creation_identity(self.session_id)

    def create_session(
        self,
        spec_factory: Callable[[SessionContentWriter], SessionSpec],
        *,
        consumers: tuple[str, ...],
    ) -> CreateReceipt:
        """Create session/key/content/record/links inside the current outer transaction."""
        existing = self.tx.preflight_create(self.session_id)
        deferred_backend: _DeferredContentBackend | None = None
        factory_writer = self
        if not existing:
            factory_writer, deferred_backend = _deferred_content_writer(self)

        try:
            spec = spec_factory(factory_writer)
            if self._binding._object_content is not None and spec.parent_session_id is not None:
                raise ContentRuntimeUnavailable("external content child admission is unavailable")
            if spec.session_id != self.session_id:
                raise ValueError("session spec does not match the content writer binding")
            principal = Principal.from_dict(spec.principal)
            if principal.tenant_id != self.tenant_id:
                raise ValueError("session principal must use the content writer tenant")
            if spec.attributes is None:
                if deferred_backend is not None:
                    raise ValueError("new session attributes require content publication")
                return self.tx.create(spec, consumers=consumers)

            if existing:
                # Replay receives a committed reference-form spec. Validate its
                # declared roots without treating mappings as new producer values.
                _content_ref(spec.workspace)
                _content_ref(spec.attributes)
                return self.tx.create(spec, consumers=consumers)

            workspace_ref = _creation_ref(factory_writer, spec.workspace)
            attributes = _creation_ref(factory_writer, spec.attributes)
            principal_attributes = _creation_ref(factory_writer, spec.principal["attributes"])
            principal_payload = (
                spec.principal | {"attributes": principal_attributes} if principal_attributes is not None else spec.principal
            )

            created_content = publish_session_created_content(
                factory_writer,
                principal=principal_payload,
                workspace=workspace_ref if workspace_ref is not None else spec.workspace,
                attributes=attributes if attributes is not None else spec.attributes,
            )
        finally:
            if factory_writer._creation_json is not None:
                factory_writer._creation_json.clear()
        published = replace(
            spec,
            principal=created_content.principal,
            workspace=created_content.workspace.to_dict(),
            attributes=created_content.attributes.to_dict(),
        )
        receipt = self.tx.create(published, consumers=consumers)
        if deferred_backend is not None:
            if receipt.replayed:
                raise RuntimeError("a new deferred session cannot replay an existing create")
            deferred_backend.replay()

        self.link_record(1, _content_ref(created_content.principal["attributes"]), SESSION_PRINCIPAL_ATTRIBUTES)
        self.link_record(1, created_content.workspace, SESSION_WORKSPACE)
        self.link_record(1, created_content.attributes, SESSION_ATTRIBUTES)
        for nested in created_content.nested:
            self.link_record(1, nested.ref, nested.slot)
        return receipt


class _ContentSessionTxAdapter:
    """Expose the session writer as an explicit kernel transaction capability."""

    def __init__(
        self, tx: SessionTx, binding: ContentRuntimeBinding, *, check_publication: Callable[[], None] | None = None
    ) -> None:
        if not isinstance(tx, SQLSessionTx):
            raise ContentRuntimeUnavailable("content capability requires a caller-owned SQL transaction")
        self._tx = tx
        self._binding = binding
        self._check_publication = check_publication
        self._published_refs: set[ContentRef] = set()
        self._write_memo = _ContentWriteMemo()

    @property
    def _requires_early_persistence_deadline(self) -> bool:
        return self._binding._object_content is not None

    def content_writer(self, *, tenant_id: str, session_id: str):
        return SessionContentWriter(
            tx=self._tx,
            tenant_id=tenant_id,
            session_id=session_id,
            binding=self._binding,
            _published_refs=self._published_refs,
            _write_memo=self._write_memo,
            _check_publication=self._check_publication,
        )

    def lock_session(self, session_id: str) -> None:
        self._tx.lock_session(session_id)

    def bind_lease_guard(self, guard: Callable[[], Lease]) -> None:
        self._tx.bind_lease_guard(guard)

    def _bind_persistence_deadline(
        self, check_deadline: Callable[[], None], freeze_deadline: Callable[[], None], object_deadline: _ObjectDeadline
    ) -> bool:
        return self._tx._bind_persistence_deadline(check_deadline, freeze_deadline, object_deadline)

    def create(self, spec: SessionSpec, *, consumers: tuple[str, ...]) -> CreateReceipt:
        return self._tx.create(spec, consumers=consumers)

    def push(self, session_id: str, item) -> InputReceipt:
        return self._tx.push(session_id, item)

    def append(
        self,
        session_id: str,
        *,
        lease: Lease,
        expected_seq: int,
        commit_id: str,
        records,
        consume_input_ids: tuple[str, ...] = (),
        expected_inbox_seq: int | None = None,
        expected_head: SessionHead | tuple[int, str] | None = None,
    ) -> CommitReceipt:
        return self._tx.append(
            session_id,
            lease=lease,
            expected_seq=expected_seq,
            commit_id=commit_id,
            records=records,
            consume_input_ids=consume_input_ids,
            expected_inbox_seq=expected_inbox_seq,
            expected_head=expected_head,
        )

    def consumer_batch(self, session_id: str, consumer: str, *, limit: int = 256) -> ConsumerBatch | None:
        return self._tx.consumer_batch(session_id, consumer, limit=limit)

    def ack(self, batch: ConsumerBatch) -> None:
        self._tx.ack(batch)


class _ContentSessionStoreAdapter:
    """Decorate a tenant store so every transaction carries content capability."""

    def __init__(self, store: SessionStore, binding: ContentRuntimeBinding) -> None:
        self._store = store
        self._binding = binding

    @property
    def _root(self):
        return _root_store(self._store)

    def for_tenant(self, tenant_id: str) -> SessionStore:
        return cast(SessionStore, _ContentSessionStoreAdapter(self._store.for_tenant(tenant_id), self._binding))

    @contextmanager
    def heartbeat_store(self) -> Iterator[SessionStore]:
        with self._store.heartbeat_store() as store:
            yield cast(SessionStore, _ContentSessionStoreAdapter(store, self._binding))

    def drive_store_factory(self) -> Callable[[], AbstractContextManager[SessionStore]]:
        root_factory = cast(_DriveStoreFactoryProvider, self._store).drive_store_factory()

        @contextmanager
        def open_store() -> Iterator[SessionStore]:
            with root_factory() as store:
                if self._binding._object_content is not None:
                    _verify_content_binding(store, self._binding)
                yield cast(SessionStore, _ContentSessionStoreAdapter(store, self._binding))

        return open_store

    @contextmanager
    def atomic(self) -> Iterator[SessionTx]:
        if isinstance(self._binding._object_content, _PostgresObjectContent):
            raise ContentRuntimeUnavailable("PostgreSQL object sessions require a fenced root publication transaction")
        with self._store.atomic() as tx, self._adapt_transaction(tx) as adapter:
            yield adapter

    @contextmanager
    def _adapt_transaction(self, tx: SessionTx, *, check_publication: Callable[[], None] | None = None) -> Iterator[SessionTx]:
        adapter = _ContentSessionTxAdapter(tx, self._binding, check_publication=check_publication)
        try:
            yield cast(SessionTx, adapter)
        finally:
            adapter._write_memo.clear()
            adapter._published_refs.clear()

    def _root_publication_atomic(self, clock: Clock) -> AbstractContextManager[SessionTx]:
        deadline = clock.monotonic() + 5

        def check_deadline() -> None:
            if clock.monotonic() >= deadline:
                raise KeyUnavailable("external content persistence budget expired")

        return self._publication_atomic(
            check_deadline, (int(deadline * 1_000_000_000), lambda: int(clock.monotonic() * 1_000_000_000))
        )

    def _leased_publication_atomic(
        self, lease: Lease, check_deadline: Callable[[], None], object_deadline: _ObjectDeadline
    ) -> AbstractContextManager[SessionTx]:
        return self._publication_atomic(check_deadline, object_deadline, lease=lease)

    @contextmanager
    def _publication_atomic(
        self, check_deadline: Callable[[], None], object_deadline: _ObjectDeadline, *, lease: Lease | None = None
    ) -> Iterator[SessionTx]:
        objects = self._binding._object_content
        if not isinstance(objects, _PostgresObjectContent):
            with self.atomic() as tx:
                yield tx
            return
        root = _root_store(self)
        if not isinstance(root, PostgresStore):
            raise ContentRuntimeUnavailable("PostgreSQL object publication requires its PostgreSQL store")
        digest = objects._binding_digest()
        with root._object_publication_atomic(
            objects.journal, content_digest=digest, check_deadline=check_deadline, object_deadline=object_deadline, lease=lease
        ) as (tx, check_publication):
            if objects.verify_installation(objects.journal.identity.database_id, self._binding.key_provider) != digest:
                raise ContentUnavailable("PostgreSQL object content binding changed during publication")
            check_publication()
            with self._adapt_transaction(tx, check_publication=check_publication) as adapter:
                yield adapter

    def __getattr__(self, name: str):
        return getattr(self._store, name)


def wrap_session_content_store(
    store: SessionStore,
    binding: ContentRuntimeBinding | None = None,
) -> SessionStore:
    selected = binding or content_runtime_binding_for(store)
    if selected is None:
        raise ContentRuntimeUnavailable("a content key provider must be explicitly bound to the session store")
    base = store._store if isinstance(store, _ContentSessionStoreAdapter) else store
    return cast(SessionStore, _ContentSessionStoreAdapter(base, selected))


def _content_ref(value: ContentRef | Mapping[str, Any]) -> ContentRef:
    return value if isinstance(value, ContentRef) else ContentRef.from_dict(value)


def _creation_ref(writer: SessionContentWriter, value: Any) -> ContentRef | None:
    """Recognize only refs explicitly returned by this factory's writer."""

    if isinstance(value, ContentRef):
        return value
    try:
        ref = ContentRef.from_dict(value)
    except (TypeError, ValueError):
        return None
    return ref if ref in writer._published_refs else None


__all__ = [
    "APPROVAL_METADATA",
    "APPROVAL_REASON",
    "APPROVAL_SCOPE",
    "CHILD_RESULT",
    "CONTENT_RUNTIME_BINDING_ATTRIBUTE",
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
    "bind_content_runtime",
    "content_key_provider_for",
    "content_runtime_binding_for",
    "durable_content_reader",
    "durable_content_store",
    "read_input_json",
    "read_record_json",
    "wrap_session_content_store",
]
