"""Session host assembly and retained runtime bindings."""

from __future__ import annotations

import secrets
import sqlite3
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from typing import Any, cast
from uuid import uuid4
from weakref import WeakKeyDictionary, WeakValueDictionary

from vv_agent.agent import Agent
from vv_agent.approval import ApprovalDecision
from vv_agent.config import ResolvedModelConfig
from vv_agent.constants import SUB_TASK_STATUS_TOOL_NAME
from vv_agent.events import (
    AssistantDeltaEvent,
    ModelToolCallProgressEvent,
    ModelToolCallStartedEvent,
    ReasoningDeltaEvent,
    RunEvent,
)
from vv_agent.kernel._content import (
    INPUT_ACTOR_ATTRIBUTES,
    SESSION_PRINCIPAL_ATTRIBUTES,
    ContentRef,
    ContentUnavailable,
    KeyProvider,
    KeyUnavailable,
    Principal,
)
from vv_agent.kernel._content_publication import (
    declare_input_content,
    link_declared_input_content,
    publish_session_created_content,
)
from vv_agent.kernel._driver_inputs import _pending_batch_in_state, _steer_held_in_state
from vv_agent.kernel._postgres import PostgresStore as CorePostgresStore
from vv_agent.kernel._postgres import _bind_tenant_context
from vv_agent.kernel._provider_outcomes import ProviderResultInvalid
from vv_agent.kernel._requirements import (
    SessionRequirements,
    normalize_requirements,
)
from vv_agent.kernel._sql import (
    _DRIVE_RUNNABLE_SQL,
    LedgerIntegrityError,
    SessionNotLive,
    SQLStore,
    UnsupportedFeatureLevel,
    UnsupportedWorkerCapabilities,
    _Prefix,
    _RoutingCache,
    _TenantSQLStore,
)
from vv_agent.kernel._sqlite import SQLiteStore as CoreSQLiteStore
from vv_agent.kernel._store import DEFAULT_TENANT_ID, LeaseRetryExhausted
from vv_agent.memory import MemoryManager
from vv_agent.model import ModelError, resolve_model_selection, resolve_run_model
from vv_agent.result import RunResult
from vv_agent.run_config import RunConfig, effective_run_config
from vv_agent.run_handle import RunHandle
from vv_agent.tracing import trace_processors
from vv_agent.types import AgentTask, Message

from ._driver_scope import _sqlite_lock_error
from .children import child_delivery, child_handles
from .content_runtime import (
    INPUT_CONTENT,
    SessionContentWriter,
    _ContentSessionStoreAdapter,
    bind_content_key_provider,
    content_key_provider_for,
    content_runtime_binding_for,
    durable_content_reader,
    read_input_json,
    read_record_json,
    wrap_session_content_store,
)
from .content_store import LocalKeyProvider
from .context import project_context
from .postgres import PostgresStore as SessionPostgresStore
from .projection import (
    fold_hydrated_records,
    hydrate_record_payload,
    hydrate_records,
    hydration_for_session_store,
    project_records,
)
from .records import InboxItem, SessionSpec
from .reducer import _fold_factory_for
from .result import project_result
from .runtime import (
    Runtime,
    _resolve_static_model_config,
    static_model_selection_requirements,
    static_session_requirements,
    static_worker_supports,
)
from .sqlite import SQLiteStore
from .store import Conflict, Lease, LeaseLost, SessionStore
from .timing import SystemClock
from .tracing import deliver_spans
from .worker import LocalWorker, WorkerSupport

_DRIVERS: WeakValueDictionary[tuple[str, str], SessionDriver] = WeakValueDictionary()
_STORE_CLOSERS: WeakKeyDictionary[SessionStore, list[Any]] = WeakKeyDictionary()
_DEFAULT_SQLITE_PATH = object()
_ALL_RECORDS_LIMIT = (1 << 31) - 1
_EVENT_PAGE_SIZE = 256
_PROVIDER_RECOVERY_SUBJECT_ID = "provider-recovery"
_TRANSIENT_STREAM_EVENTS = (AssistantDeltaEvent, ReasoningDeltaEvent, ModelToolCallStartedEvent, ModelToolCallProgressEvent)
_RUNTIME_CLAIM_BOUND = object()
PrincipalInput = Principal | Mapping[str, Any]
RuntimeFactory = Callable[[SessionStore, Agent, RunConfig, AgentTask | None, str], Runtime]
"""Rebuild a host runtime against the exact store connection for one worker drive.

The factory is the explicit rebinding boundary. Callbacks that captured another
store or connection are unsupported and must be rebuilt by this factory. Any
host capabilities it adds must be declared through ``SessionDriver.capabilities``
so pre-admission requirements and LocalWorker support use the same trusted tags.
"""


def _session_requirements(store: SessionStore, sid: str) -> SessionRequirements:
    page = store.read(sid, limit=1)
    if not page.records or page.records[0].record.kind != "session_created":
        raise Conflict("session has no durable session_created record")
    created = page.records[0].record
    if created.feature_level == 1:
        return SessionRequirements(1, None, ())
    return normalize_requirements(created._payload["bootstrap"])


def _verified_routing_cache(
    store: SessionStore,
    sid: str,
    *,
    ceiling: int | None = None,
) -> tuple[_RoutingCache, int]:
    """Verify routing under the writer lock order: settings, then session."""

    sql_store = cast(Any, store)
    if ceiling is None:
        ceiling = sql_store._write_ceiling()
    row = sql_store._lock(sid)
    sql_store._require_live_row(row)
    return sql_store._require_routing_cache(sid, row), ceiling


def _merge_registration_requirements(
    requirements: SessionRequirements,
    registration: _RuntimeRegistration | None,
    *,
    static_requirements: SessionRequirements | None = None,
) -> SessionRequirements:
    registered = static_requirements
    if registration is not None:
        registered = registered or registration.requirements
    if registered is None:
        return requirements
    return SessionRequirements.normalized(
        level=max(requirements.level, registered.level),
        handler_version=(
            registration.handler_version
            if registration is not None
            else requirements.handler_version or registered.handler_version
        ),
        capabilities=(*requirements.capabilities, *registered.capabilities),
    )


def _static_agent_tree(agent: Agent, config: RunConfig) -> Iterator[tuple[Agent, RunConfig]]:
    """Yield trusted root/child declarations without constructing a Runtime."""

    from .delegation import _static_child_declarations

    queue: list[tuple[Agent, RunConfig, frozenset[int]]] = [(agent, config, frozenset())]
    while queue:
        current, current_config, ancestors = queue.pop()
        identity = id(current)
        if identity in ancestors:
            continue
        yield current, current_config
        path = ancestors | {identity}
        queue.extend((child.agent, child.config, path) for child in _static_child_declarations(current, current_config))


def _static_worker_supports_tree(
    agent: Agent,
    config: RunConfig,
    *,
    extra_capabilities: tuple[str, ...],
) -> tuple[str, ...]:
    """Declare every statically reachable child runtime this worker may claim.

    ``Agent.as_tool()`` and ``as_background_task()`` expose their child through
    the public tool metadata. A child claim is routed before its runtime can be
    built, so the root registration must include those child tools and models.
    """

    capabilities = set(extra_capabilities)
    for current, current_config in _static_agent_tree(agent, config):
        capabilities.update(
            static_worker_supports(
                current,
                current_config,
                resolved=None,
                extra_capabilities=extra_capabilities,
            )
        )
    return tuple(sorted(capabilities))


def _static_worker_requirement_level(
    agent: Agent,
    config: RunConfig,
    *,
    handler_version: str,
    extra_capabilities: tuple[str, ...],
) -> int:
    """Return the strongest level required by a statically reachable child."""

    return max(
        (
            static_session_requirements(
                current,
                current_config,
                resolved=None,
                handler_version=handler_version,
                extra_capabilities=extra_capabilities,
            ).level
            for current, current_config in _static_agent_tree(agent, config)
        ),
        default=1,
    )


@dataclass(slots=True)
class _SessionFailure:
    """The single operational failure state retained for one session."""

    turn_id: str | None
    error: BaseException
    attempts: int
    progress_head: int
    terminal: bool
    retry_attempts: int
    retry_base_seconds: float
    retry_cap_seconds: float
    retry_jitter: Callable[[float, float], float]
    warning_sent: bool = False
    fenced_operations: frozenset[tuple[str, int]] = frozenset()


@dataclass(frozen=True, slots=True)
class _RuntimeRegistration:
    """Trusted inputs used to rebuild one session runtime on each worker claim."""

    session_id: str
    agent: Agent
    config: RunConfig
    resolved: ResolvedModelConfig | None
    task: AgentTask | None
    handler_version: str
    requirements: SessionRequirements
    observer: Callable[[RunEvent], None] | None
    armed: bool = False


@dataclass(slots=True)
class _ProjectionRuntime:
    """Connection-free runtime fields used only by read-only projection."""

    owner: SessionDriver
    agent: Agent
    config: RunConfig
    _resolved: ResolvedModelConfig | None
    handler_version: str
    wake: Callable[[str], None] = lambda _sid: None
    hook: Callable[[str, Any], None] = lambda _point, _record: None
    clock: SystemClock = field(default_factory=SystemClock)
    content_reader: Any = None
    frozen_task: AgentTask | None = None
    _registry: Any = field(default=None, init=False, repr=False)

    @property
    def resolved(self) -> ResolvedModelConfig:
        if self._resolved is None:
            self._resolved = _resolve_static_model_config(self.agent, self.config)
        return self._resolved

    @property
    def registry(self):
        if self._registry is None:
            from .runtime import _build_configured_registry

            registry, _ = _build_configured_registry(self.agent, self.config, runtime=None)
            self._registry = registry
        return self._registry

    def child_runtime(self, store: SessionStore, session_id: str) -> _ProjectionRuntime:
        return self.owner._projection_runtime_view(session_id, store=store)

    def child_tasks(self, store: SessionStore, session_id: str):
        from .delegation import ChildTasks

        return ChildTasks(store, session_id, cast(Any, self))

    def for_agent(self, agent: Agent, config: RunConfig) -> _ProjectionRuntime:
        return _ProjectionRuntime(
            owner=self.owner,
            agent=agent,
            config=config,
            _resolved=None,
            handler_version=self.handler_version,
            wake=self.wake,
            hook=self.hook,
            clock=self.clock,
        )

    def durable_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return state

    def bind_state(self, state: dict[str, Any], task: AgentTask) -> dict[str, Any]:
        del task
        return state


class _PersistedPrincipalPayload(dict[str, Any]):
    """Keep persisted Principal identity distinct while crossing mapping-only assembly layers."""


class _IngressAttributes(dict[str, Any]):
    """Mark host ingress attributes as plaintext across mapping-only assembly layers."""


def _tenant_bound_sql_store(store: SessionStore) -> SessionStore:
    """Unwrap content decoration without discarding the tenant-bound SQL facade."""

    if isinstance(store, _ContentSessionStoreAdapter):
        store = store._store
    if isinstance(store, _TenantSQLStore):
        return cast(SessionStore, store._bound)
    return store


@contextmanager
def _snapshot_prefix_cache(concrete: SQLStore) -> Iterator[None]:
    previous_active = getattr(concrete, "_read_snapshot_active", False)
    concrete._read_snapshot_active = True
    concrete._read_snapshot_prefix = None
    try:
        yield
    finally:
        concrete._read_snapshot_active = previous_active
        concrete._read_snapshot_prefix = None


@contextmanager
def _read_snapshot(store: SessionStore) -> Iterator[None]:
    """Hold one read-only SQL snapshot without blocking a driving writer."""

    concrete = _tenant_bound_sql_store(store)
    if isinstance(concrete, CoreSQLiteStore):
        connection = concrete.connection
        with concrete._connection_lock:
            if connection.in_transaction:
                yield
                return
            with _snapshot_prefix_cache(concrete):
                connection.execute("BEGIN")
                try:
                    yield
                except BaseException:
                    with suppress(Exception):
                        connection.execute("ROLLBACK")
                    raise
                else:
                    connection.execute("COMMIT")
        return
    if isinstance(concrete, CorePostgresStore):
        from psycopg.pq import TransactionStatus

        connection = concrete.connection
        with concrete._connection_lock:
            if connection.info.transaction_status != TransactionStatus.IDLE:
                if connection.info.transaction_status == TransactionStatus.INTRANS:
                    _bind_tenant_context(connection, concrete.tenant_context_id)
                yield
                return
            with _snapshot_prefix_cache(concrete), connection.transaction():
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                _bind_tenant_context(connection, concrete.tenant_context_id)
                yield
        return
    with store.atomic():
        yield


def _drive_flags_in_snapshot(store: SessionStore, sid: str) -> tuple[bool, bool]:
    """Return current runnability and pending schedule from one SQL snapshot."""

    concrete = _tenant_bound_sql_store(store)
    if not isinstance(concrete, (CoreSQLiteStore, CorePostgresStore)):
        return store.is_runnable(sid), False
    candidate_predicate, worker_params = concrete._worker_candidate_predicate()
    row = concrete._one(
        f"WITH clock AS MATERIALIZED (SELECT {concrete.clock_sql} AS ms) "
        "SELECT "
        "CASE WHEN s.schema_version='S1' AND s.deletion_state='live' "
        f"AND {candidate_predicate} AND {_DRIVE_RUNNABLE_SQL} THEN 1 ELSE 0 END, "
        "CASE WHEN s.schema_version='S1' AND s.deletion_state='live' "
        "AND s.next_drive_ms<=t.ms THEN 1 ELSE 0 END "
        "FROM sk_session s CROSS JOIN clock t "
        "WHERE s.tenant_id=%s AND s.session_id=%s",
        (*worker_params, concrete.tenant_id, sid),
    )
    return bool(row[0]), bool(row[1])


def _is_runnable_in_snapshot(store: SessionStore, sid: str) -> bool:
    """Evaluate the normal worker predicate on the caller's current SQL snapshot."""

    return _drive_flags_in_snapshot(store, sid)[0]


@dataclass(slots=True)
class _ProjectionDelivery:
    """One verified ledger prefix captured for one caller-owned page read."""

    prefix: _Prefix | None


def _projection_delivery(store: SessionStore, sid: str) -> _ProjectionDelivery:
    """Create page-local projection state for the caller's current read snapshot."""

    concrete = _tenant_bound_sql_store(store)
    prefix = concrete.projection_prefix(sid) if isinstance(concrete, SQLStore) else None
    return _ProjectionDelivery(prefix=prefix)


def _read_hydrated_state(
    store: SessionStore,
    sid: str,
    *,
    expected_head=None,
):
    """Read and fold one session from a single database snapshot."""

    with _read_snapshot(store):
        read_options = {"expected_head": expected_head} if expected_head is not None else {}
        page = store.read(sid, limit=_ALL_RECORDS_LIMIT, **read_options)
        source_records = page.records
        if page.head_seq != len(source_records):
            raise Conflict("session ledger read did not capture the complete head")
    state = None
    if source_records:
        # Content decryption can cross a network boundary. Hydration and the
        # hydrated fold deliberately run after the SQL capture transaction ends.
        hydration = hydration_for_session_store(store, sid)
        records = hydrate_records(source_records, hydration)
        state = fold_hydrated_records(records, source_records, _fold_factory_for(store), hydration)
    return state, source_records, page.inbox_seq


def _has_pending_drive_input(store: SessionStore, sid: str, state) -> bool:
    """Return whether an otherwise waiting session already has a next input."""

    if _is_runnable_in_snapshot(store, sid) or store.peek_inbox(sid):
        return True
    return any(
        record._payload["disposition"] == "queued" and key not in state.admitted_inputs
        for key, record in state.applied_inputs.items()
    )


def _capture_record_page(
    store: SessionStore,
    sid: str,
    *,
    after_seq: int,
    through_seq: int | None = None,
    delivery: _ProjectionDelivery | None = None,
) -> tuple[Any, tuple[Any, ...]]:
    """Capture one bounded record page and its projection context."""

    concrete = _tenant_bound_sql_store(store)
    if delivery is not None and delivery.prefix is not None and isinstance(concrete, SQLStore):
        page = concrete.read_projection_page(
            delivery.prefix,
            sid,
            after_seq=after_seq,
            through_seq=through_seq,
            limit=_EVENT_PAGE_SIZE,
        )
        context = concrete.projection_context(sid, page.records, prefix=delivery.prefix) if page.records else ()
    else:
        page = store.read(
            sid,
            after_seq=after_seq,
            through_seq=through_seq,
            limit=_EVENT_PAGE_SIZE,
        )
        context = store.projection_context(sid, page.records) if page.records else ()
    return page, context


def _project_captured_record_page(
    store: SessionStore,
    sid: str,
    page,
    context: tuple[Any, ...],
    *,
    through_seq: int | None = None,
    run_id: str | None = None,
) -> tuple[int, list[RunEvent], bool]:
    """Hydrate and project one page through one disposable content frame."""

    if not page.records:
        return page.records[-1].seq if page.records else 0, [], False
    hydration = hydration_for_session_store(store, sid)
    page_sequences = {stored.seq for stored in page.records}
    events = [
        event
        for event in project_records(context, hydration=hydration)
        if event.metadata.get("session_seq") in page_sequences and (run_id is None or event.run_id == run_id)
    ]
    boundary = page.head_seq if through_seq is None else min(page.head_seq, through_seq)
    return page.records[-1].seq, events, page.records[-1].seq < boundary


def _project_record_page(
    store: SessionStore,
    sid: str,
    *,
    after_seq: int,
    through_seq: int | None = None,
    run_id: str | None = None,
) -> tuple[int, list[RunEvent], bool]:
    """Project one bounded record page from one consistent SQL snapshot."""

    with _read_snapshot(store):
        selected = _projection_delivery(store, sid)
        page, context = _capture_record_page(
            store,
            sid,
            after_seq=after_seq,
            through_seq=through_seq,
            delivery=selected,
        )
    if not page.records:
        return after_seq, [], False
    return _project_captured_record_page(
        store,
        sid,
        page,
        context,
        through_seq=through_seq,
        run_id=run_id,
    )


def _principal_identity(payload: dict[str, Any], *, tenant_id: str) -> dict[str, Any]:
    if set(payload) != {"tenant_id", "subject_id", "subject_kind", "attributes"}:
        raise ValueError("principal must contain tenant_id, subject_id, subject_kind and attributes")
    if not isinstance(payload["tenant_id"], str) or not payload["tenant_id"]:
        raise ValueError("tenant_id must be a non-empty string")
    if payload["tenant_id"] != tenant_id:
        raise ValueError("principal must use the session tenant")
    if not isinstance(payload["subject_id"], str) or not payload["subject_id"]:
        raise ValueError("subject_id must be a non-empty string")
    if payload["subject_kind"] not in {"user", "service", "agent"}:
        raise ValueError("subject_kind must be user, service or agent")
    return payload


def _principal_payload(value: PrincipalInput) -> dict[str, Any]:
    if isinstance(value, Principal):
        return _PersistedPrincipalPayload(value.to_dict())
    if isinstance(value, _PersistedPrincipalPayload):
        return value
    payload = dict(value)
    tenant_id = payload.get("tenant_id")
    if not isinstance(tenant_id, str):
        raise ValueError("principal tenant_id must be non-empty text")
    _principal_identity(payload, tenant_id=tenant_id)
    if not isinstance(payload["attributes"], Mapping) or isinstance(payload["attributes"], ContentRef):
        raise ValueError("host ingress principal attributes must be a plaintext mapping")
    payload["attributes"] = _IngressAttributes(payload["attributes"])
    return payload


def _ingress_principal(value: Mapping[str, Any], *, tenant_id: str) -> dict[str, Any]:
    """Validate a host ingress identity whose attributes are plaintext JSON."""

    if isinstance(value, Principal | _PersistedPrincipalPayload):
        raise ValueError("persisted Principal requires a target-session owner proof")
    payload = dict(value)
    _principal_identity(payload, tenant_id=tenant_id)
    attributes = payload["attributes"]
    if isinstance(attributes, ContentRef):
        raise ValueError("host ingress principal attributes must be a plaintext mapping")
    if not isinstance(attributes, Mapping):
        raise ValueError("host ingress principal attributes must be a plaintext mapping")
    payload["attributes"] = dict(attributes)
    return payload


def _persisted_principal(value: PrincipalInput) -> Principal | None:
    if isinstance(value, Principal):
        return value
    if isinstance(value, _PersistedPrincipalPayload):
        return Principal.from_dict(dict(value))
    return None


def _is_provider_recovery_actor(value: PrincipalInput | None) -> bool:
    if isinstance(value, Principal):
        return value.subject_id == _PROVIDER_RECOVERY_SUBJECT_ID
    if not isinstance(value, Mapping):
        return False
    if value.get("subject_id") == _PROVIDER_RECOVERY_SUBJECT_ID:
        return True
    attributes = value.get("attributes")
    return isinstance(attributes, Mapping) and attributes.get("role") == _PROVIDER_RECOVERY_SUBJECT_ID


def _reject_public_provider_ingress(
    item: InboxItem,
    *,
    actor: PrincipalInput | None,
    actor_subject_id: str,
) -> None:
    if item.kind == "provider_evidence":
        raise Conflict("provider evidence requires trusted recovery ingress")
    if (
        actor_subject_id == _PROVIDER_RECOVERY_SUBJECT_ID
        or _is_provider_recovery_actor(actor)
        or _is_provider_recovery_actor(item.actor)
    ):
        raise Conflict("provider-recovery actor is reserved for trusted recovery ingress")


class SessionDriver:
    def __init__(
        self,
        path: str | Path | object = _DEFAULT_SQLITE_PATH,
        *,
        store: SessionStore | None = None,
        tenant_id: str | None = None,
        model_resolver=None,
        content_key_provider: KeyProvider | None = None,
        runtime_factory: RuntimeFactory | None = None,
        open_store: Callable[[], AbstractContextManager[SessionStore]] | None = None,
        worker_level: int = 1,
        handler_versions: tuple[str, ...] = ("1",),
        capabilities: tuple[str, ...] = (),
    ) -> None:
        if runtime_factory is not None and not callable(runtime_factory):
            raise TypeError("runtime_factory must be callable or None")
        if open_store is not None and not callable(open_store):
            raise TypeError("open_store must be callable or None")
        if isinstance(worker_level, bool) or not isinstance(worker_level, int) or worker_level < 1:
            raise ValueError("worker_level must be a positive integer")
        normalized_handlers = tuple(sorted(set(handler_versions)))
        if not normalized_handlers or any(not isinstance(value, str) or not value for value in normalized_handlers):
            raise ValueError("handler_versions must contain non-empty strings")
        normalized_capabilities = tuple(sorted(set(capabilities)))
        if any(not isinstance(value, str) or not value for value in normalized_capabilities):
            raise ValueError("capabilities must contain non-empty strings")
        self.model_resolver = model_resolver
        self.runtime_factory = runtime_factory
        self.open_store = open_store
        self.handler_versions = normalized_handlers
        self.default_handler_version = normalized_handlers[0]
        self.capabilities = normalized_capabilities
        self._owns_store = store is None
        self._temporary_store_directory: TemporaryDirectory[str] | None = None
        if store is None and path is _DEFAULT_SQLITE_PATH:
            self._temporary_store_directory = TemporaryDirectory(prefix="vv-agent-session-")
            path = Path(self._temporary_store_directory.name) / "sessions.sqlite3"
        if store is not None:
            self._root_store = store
        else:
            self._root_store = SQLiteStore(str(path))
            if self._temporary_store_directory is not None:
                self._root_store.configure_input_admission(allow_unsigned=True)
        concrete_store = self._root_store
        seen: set[int] = set()
        while id(concrete_store) not in seen:
            seen.add(id(concrete_store))
            parent = getattr(concrete_store, "_root_store", None) or getattr(concrete_store, "_root", None)
            if parent is None or parent is concrete_store:
                break
            concrete_store = parent
        if isinstance(concrete_store, CorePostgresStore) and not isinstance(concrete_store, SessionPostgresStore):
            raise ValueError(
                "raw kernel PostgreSQL stores cannot be used as a runtime entry point; use session.postgres.PostgresStore"
            )
        bound_tenant = getattr(self._root_store, "tenant_id", None)
        if tenant_id is None:
            if store is None:
                tenant_id = DEFAULT_TENANT_ID
            elif isinstance(bound_tenant, str) and bound_tenant:
                tenant_id = bound_tenant
            else:
                raise ValueError("tenant_id is required for a session store without a tenant binding")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("tenant_id must be a non-empty string")
        credential_tenant = getattr(self._root_store, "credential_tenant_id", None)
        if credential_tenant is not None and credential_tenant != tenant_id:
            raise ValueError("session tenant must match the verified PostgreSQL tenant login")
        self.tenant_id = tenant_id
        self.credential_isolation = bool(
            getattr(self._root_store, "credential_isolation", False) and credential_tenant == tenant_id
        )
        runtime_mode = getattr(concrete_store, "runtime_mode", None)
        self._execution_capable = not isinstance(concrete_store, CorePostgresStore) or runtime_mode in {
            "worker",
            "unbound-internal",
        }
        self._store = self._root_store if bound_tenant == self.tenant_id else self._root_store.for_tenant(self.tenant_id)
        bound_provider = content_key_provider_for(self._root_store)
        self.uses_development_content_key_provider = content_key_provider is None and bound_provider is None
        self._temporary_content_key_state: TemporaryDirectory[str] | None = None
        if content_key_provider is not None:
            self.content_key_provider = content_key_provider
        elif bound_provider is not None:
            self.content_key_provider = bound_provider
        elif isinstance(concrete_store, CoreSQLiteStore) and (
            concrete_store.path == ":memory:" or self._temporary_store_directory is not None
        ):
            # Ephemeral local stores keep their development key lifecycle with
            # the database instead of sharing process-wide durable key state.
            self._temporary_content_key_state = TemporaryDirectory(prefix="vv-agent-session-keys-")
            self.content_key_provider = LocalKeyProvider(
                secrets.token_bytes(32),
                state_path=Path(self._temporary_content_key_state.name) / "key-state.sqlite3",
            )
        else:
            if self._owns_store and isinstance(self._root_store, CoreSQLiteStore):
                with self._root_store._connection_lock:
                    self._root_store.connection.close()
            if self._temporary_store_directory is not None:
                self._temporary_store_directory.cleanup()
            raise ValueError("durable SessionDriver stores require an explicit content_key_provider")
        bind_content_key_provider(self._root_store, self.content_key_provider)
        self._store = wrap_session_content_store(self._store, self.content_key_provider)
        cast(Any, self._store).configure_worker(
            level=worker_level,
            handler_versions=self.handler_versions,
            capabilities=self.capabilities,
        )
        self._drive_store_factory = None if open_store is not None else cast(Any, self._store).drive_store_factory()
        self._heartbeat_store_factory = cast(Any, self._store).heartbeat_store_factory()
        self.path = self._root_store.path if isinstance(self._root_store, CoreSQLiteStore) else None
        if (
            isinstance(self._root_store, CoreSQLiteStore)
            and not self._root_store.connection.execute("PRAGMA user_version").fetchone()[0]
        ):
            self._root_store.install_schema()
        self.handles: list[RunHandle] = []
        # Worker-visible configuration is immutable per assignment. Concurrent
        # starts are arbitrated by the SQL admission lock before this map is
        # replaced with the winning armed registration.
        self._runtimes: dict[str, _RuntimeRegistration] = {}
        self._observer_cursors: dict[str, int] = {}
        self._errors: dict[str, _SessionFailure] = {}
        self._close_callbacks = []
        self._next_session_requirements: SessionRequirements | None = None
        self._worker = LocalWorker(
            WorkerSupport(
                open_store=self._open_drive_store,
                drive=self._drive_worker_session if self._execution_capable else None,
                project=self._project_worker_session if self._execution_capable else None,
                observe=self._observe_registered_handles,
                projection_consumers=("child_delivery",) if self._execution_capable else (),
                level=worker_level,
                handler_versions=self.handler_versions,
                capabilities=self.capabilities,
            )
        )
        self._closed = False

    @property
    def store(self) -> SessionStore:
        return self._store

    @store.setter
    def store(self, value: SessionStore) -> None:
        self._store = value

    @property
    def _is_closed(self) -> bool:
        return self._closed

    @contextmanager
    def _open_drive_store(self):
        manager = self.open_store() if self.open_store is not None else self._open_default_drive_store()
        with manager as store:
            if getattr(store, "tenant_id", None) != self.tenant_id:
                store = store.for_tenant(self.tenant_id)
            if content_runtime_binding_for(store) is None:
                bind_content_key_provider(store, self.content_key_provider)
            store = wrap_session_content_store(store, self.content_key_provider)

            # The drive factory may return a bare core store. LeaseKeeper needs
            # the caller-owned root's renewal connection, never this drive store
            # and never a borrowed heartbeat connection for model/host work.
            def heartbeat_factory():
                return self._heartbeat_store_factory

            cast(Any, store).heartbeat_store_factory = heartbeat_factory
            # Worker path rebinding calls ``for_tenant`` on the content adapter,
            # which returns a fresh facade. Keep the same instance factory on
            # the wrapped tenant store so that rebind cannot lose keeper setup.
            inner = getattr(store, "_store", None)
            if inner is not None and inner is not store:
                cast(Any, inner).heartbeat_store_factory = heartbeat_factory
            self._worker._configure_store(store)
            yield store

    def _open_default_drive_store(self) -> AbstractContextManager[SessionStore]:
        factory = self._drive_store_factory
        if factory is None:
            raise ValueError("SessionDriver requires open_store=... for this store type")
        return factory()

    def _read_state(self, sid: str, store: SessionStore | None = None, *, expected_head=None):
        active = store or self.store
        # Keep the head, inbox sequence and records in one database snapshot. A
        # split autocommit read can combine two valid prefixes and then report a
        # false ledger gap while another writer appends.
        return _read_hydrated_state(active, sid, expected_head=expected_head)

    def _durable_content_reader(self, store: SessionStore, sid: str):
        return durable_content_reader(store, tenant_id=self.tenant_id, session_id=sid)

    def _handler_version(self, value: str | None) -> str:
        selected = value or self.default_handler_version
        if selected not in self.handler_versions:
            raise ValueError(f"handler_version is not declared by this SessionDriver: {selected}")
        return selected

    def _require_session_support(
        self,
        requirements: SessionRequirements,
        *,
        level: int,
        handler_version: str,
        capabilities: tuple[str, ...],
    ) -> None:
        """Match a persisted bound against trusted declarations without widening them."""

        if requirements.level > level:
            raise UnsupportedFeatureLevel(f"session feature level {requirements.level} exceeds worker feature level {level}")
        if requirements.level >= 2 and requirements.handler_version not in {None, handler_version}:
            raise Conflict("session bootstrap handler_version does not match RunHandle")
        missing = set(requirements.capabilities).difference(capabilities)
        if missing:
            raise UnsupportedWorkerCapabilities(
                "session required capabilities " + ",".join(sorted(missing)) + " are not supported by this worker"
            )

    def _ensure_static_worker_support(
        self,
        *,
        level: int,
        capabilities: tuple[str, ...],
    ) -> None:
        """Bind a session's complete static support to this worker before claims."""

        current = self._worker.support
        if level >= 2 and self._owns_store and isinstance(self._root_store, CoreSQLiteStore):
            write_level = self._root_store.max_feature_level
            if write_level < level:
                self._root_store._rows(
                    "UPDATE sk_settings SET write_level=? WHERE singleton=1",
                    (level,),
                )
        # Registrations are local trusted declarations. A running worker may
        # widen its support monotonically for a later session, but persisted
        # requirements never enter this union. The coordinator rebinds the
        # current support before every claim scan; future drive and keeper
        # stores capture the same support snapshot.
        support = replace(
            current,
            level=max(current.level, level),
            capabilities=tuple(sorted(set(current.capabilities) | set(capabilities))),
        )
        self._worker.support = support
        self._worker._configure_store(self.store)
        # Keeper heartbeat factories capture support when created. Rebuild the
        # caller-owned factory after every support upgrade so renewal uses the
        # same L2 level and capability set that admission just declared.
        self._heartbeat_store_factory = cast(Any, self.store).heartbeat_store_factory()

    def _configure_static_worker(
        self,
        *,
        level: int,
        capabilities: tuple[str, ...],
        requirements: SessionRequirements,
    ) -> None:
        """Save trusted high-level bootstrap and complete worker support before start."""

        self._ensure_static_worker_support(level=level, capabilities=capabilities)
        self._next_session_requirements = requirements

    def _static_registration(
        self,
        sid: str,
        agent: Agent,
        config: RunConfig,
        task: AgentTask | None,
        *,
        handler_version: str,
        armed: bool = False,
    ) -> _RuntimeRegistration:
        """Build the immutable worker input from trusted declarations only."""

        config = effective_run_config(agent, config)
        requirements = _session_requirements(self.store, sid)
        supports = _static_worker_supports_tree(
            agent,
            config,
            extra_capabilities=self.capabilities,
        )
        supports = tuple(sorted(set(supports) | set(self._worker.support.capabilities)))
        worker_level = max(
            self._worker.support.level,
            _static_worker_requirement_level(
                agent,
                config,
                handler_version=handler_version,
                extra_capabilities=self.capabilities,
            ),
        )
        self._require_session_support(
            requirements,
            level=worker_level,
            handler_version=handler_version,
            capabilities=supports,
        )
        self._ensure_static_worker_support(
            level=worker_level,
            capabilities=supports,
        )
        return _RuntimeRegistration(
            session_id=sid,
            agent=agent,
            config=config,
            resolved=None,
            task=task,
            handler_version=handler_version,
            requirements=requirements,
            observer=config.stream,
            armed=armed,
        )

    def _runtime_for_session(
        self,
        store: SessionStore,
        agent: Agent,
        config: RunConfig,
        task: AgentTask | None,
        sid: str,
        handler_version: str,
    ) -> Runtime:
        if self.runtime_factory is not None:
            runtime = self.runtime_factory(store, agent, config, task, sid)
            if not isinstance(runtime, Runtime):
                raise TypeError("SessionDriver runtime_factory must return Runtime")
            if runtime.handler_version != handler_version:
                raise Conflict("runtime handler_version does not match SessionDriver worker support")
            if getattr(runtime, "_session_driver_claim_bound", None) is _RUNTIME_CLAIM_BOUND:
                raise Conflict("SessionDriver runtime_factory must return a fresh Runtime for every claim")
            object.__setattr__(runtime, "_session_driver_claim_bound", _RUNTIME_CLAIM_BOUND)
            self._bind_claim_content_reader(runtime, store, sid)
            return runtime
        return self.runtime(agent, config, task, session_id=sid, store=store, handler_version=handler_version)

    def _bind_claim_content_reader(self, runtime: Runtime, store: SessionStore, sid: str) -> None:
        default_reader = getattr(runtime, "_session_driver_default_content_reader", None)
        if default_reader is None:
            return
        object.__setattr__(runtime, "_session_driver_default_content_reader", None)
        if runtime.content_reader is default_reader:
            runtime.content_reader = self._durable_content_reader(store, sid)

    def _register_runtime(
        self,
        sid: str,
        runtime: Runtime,
        *,
        tid: str | None = None,
        handler_version: str | None = None,
        armed: bool = False,
    ) -> _RuntimeRegistration:
        del tid
        selected_handler = self._handler_version(handler_version or runtime.handler_version)
        default_hook = Runtime.__dataclass_fields__["hook"].default
        if self.runtime_factory is None and (runtime.children or runtime.host_bindings or runtime.hook is not default_hook):
            from .bindings import MissingHostBinding

            raise MissingHostBinding("host-bound runtimes require SessionDriver(runtime_factory=...) before input admission")
        requirements = runtime.requirements_bound()
        supports = _static_worker_supports_tree(
            runtime.agent,
            runtime.config,
            extra_capabilities=self.capabilities,
        )
        supports = tuple(sorted(set(supports) | set(self._worker.support.capabilities)))
        worker_level = max(
            self._worker.support.level,
            _static_worker_requirement_level(
                runtime.agent,
                runtime.config,
                handler_version=selected_handler,
                extra_capabilities=self.capabilities,
            ),
        )
        self._ensure_static_worker_support(
            level=worker_level,
            capabilities=supports,
        )
        registration = _RuntimeRegistration(
            session_id=sid,
            agent=runtime.agent,
            config=runtime.config,
            resolved=runtime.resolved,
            task=runtime.frozen_task,
            handler_version=selected_handler,
            requirements=requirements,
            observer=runtime.config.stream,
            armed=armed,
        )
        self._runtimes[sid] = registration
        return registration

    def _runtime_registration(self, sid: str) -> _RuntimeRegistration:
        registration = self._runtimes.get(sid)
        if registration is None:
            from .bindings import MissingHostBinding

            raise MissingHostBinding(f"session {sid!r} has no trusted runtime configuration")
        return registration

    def _runtime_lineage(self, store: SessionStore, sid: str) -> tuple[_RuntimeRegistration, tuple[str, ...]]:
        """Resolve one durable child chain back to a trusted root registration."""

        current = sid
        children: list[str] = []
        seen: set[str] = set()
        while True:
            registration = self._runtimes.get(current)
            if registration is not None:
                return registration, tuple(reversed(children))
            if current in seen:
                raise Conflict("session parent chain contains a cycle")
            seen.add(current)
            page = store.read(current, limit=1)
            if not page.records or page.records[0].record.kind != "session_created":
                raise Conflict("child session has no durable session_created record")
            parent = page.records[0].record._payload.get("parent_session_id")
            if not isinstance(parent, str) or not parent:
                from .bindings import MissingHostBinding

                raise MissingHostBinding(f"session {sid!r} has no trusted root runtime configuration")
            children.append(current)
            current = parent

    def _created_payload(self, store: SessionStore, sid: str) -> dict[str, Any]:
        page = store.read(sid, limit=1)
        if not page.records or page.records[0].record.kind != "session_created":
            raise Conflict(f"session {sid!r} has no session_created record")
        return hydrate_record_payload(page.records[0], hydration_for_session_store(store, sid))

    def _build_runtime_for_claim(self, store: SessionStore, sid: str) -> Runtime:
        registration, child_chain = self._runtime_lineage(store, sid)
        runtime = self._runtime_for_session(
            store,
            registration.agent,
            registration.config,
            registration.task,
            registration.session_id,
            registration.handler_version,
        )
        for child_sid in child_chain:
            created = self._created_payload(store, child_sid)
            attributes = created.get("attributes", {})
            if "child_admission" in attributes:
                # The admitted child derives from the host-provided root runtime;
                # rebuilding it would discard bindings that are not factory inputs.
                runtime = runtime.child_runtime(store, child_sid)
            else:
                runtime = self._runtime_for_session(
                    store,
                    runtime.agent,
                    replace(runtime.config, workspace=created.get("workspace", runtime.config.workspace)),
                    None,
                    child_sid,
                    registration.handler_version,
                )
        if runtime.handler_version != registration.handler_version:
            raise Conflict("runtime handler_version does not match SessionDriver worker support")
        runtime.heartbeat_store = store.heartbeat_store
        original_stream = runtime.config.stream
        watermark = {"seq": store.read(sid, limit=1).head_seq}
        original_hook = runtime.hook

        def hook(point: str, record) -> None:
            original_hook(point, record)
            if point == "after_commit":
                # A commit can append many records. Read the committed head so
                # transient model deltas sort after the exact batch watermark.
                watermark["seq"] = store.lease_status(sid)[0]
                self._notify_session(sid)
            elif point in {"after_input", "after_drive"}:
                self._notify_session(sid)

        runtime.config = replace(
            runtime.config,
            stream=lambda event: self._emit_stream(
                sid,
                runtime.config,
                original_stream,
                watermark["seq"],
                event,
            ),
        )
        runtime.hook = hook
        runtime.wake = self._wake_session
        return runtime

    def _projection_runtime_view(
        self,
        sid: str,
        *,
        store: SessionStore | None = None,
    ) -> _ProjectionRuntime:
        selected_store = store or self.store
        registration, child_chain = self._runtime_lineage(selected_store, sid)
        agent = registration.agent
        resolved = registration.resolved
        if child_chain:
            state, records, _ = self._read_observed_state(selected_store, sid)
            turn = state.turns.get(state.active_turn_id or next(reversed(state.turns), "")) if state is not None else None
            agent_name = None
            if turn is not None:
                source_start = next(row for row in records if row.record.record_id == turn.start.record_id)
                definition = hydrate_record_payload(
                    source_start,
                    hydration_for_session_store(selected_store, sid),
                ).get("definition", {})
                agent_name = definition.get("agent_name")
            if isinstance(agent_name, str) and agent_name:
                agents: dict[str, Agent] = {}

                def collect(candidate: Agent) -> None:
                    if candidate.name in agents:
                        return
                    agents[candidate.name] = candidate
                    for handoff in candidate.handoffs:
                        collect(handoff.agent)

                collect(registration.agent)
                selected = agents.get(agent_name, agent)
                if selected is not agent:
                    agent = selected
                    resolved = _resolve_static_model_config(agent, registration.config)
        return _ProjectionRuntime(
            owner=self,
            agent=agent,
            config=registration.config,
            _resolved=resolved,
            handler_version=registration.handler_version,
            wake=self._wake_session,
        )

    def _runtime_view(self, sid: str) -> _ProjectionRuntime:
        return self._projection_runtime_view(sid)

    def _observe_registered_handles(self, store: SessionStore) -> None:
        """Advance one coordinator-owned durable projection cursor per session."""

        roots = tuple(self._runtimes)
        candidates = set(roots)
        candidates.update(handle.session_id for handle in self.handles)
        candidates.update(self._observer_cursors)
        for root in roots:
            registration = self._runtimes.get(root)
            if registration is not None:
                config = registration.config
                if config.event_store is None and config.stream is None and not trace_processors(config):
                    self._observer_cursors[root] = store.lease_status(root)[0]
                    self._notify_session(root)
                    continue
            try:
                child_ids = self._discover_child_runtimes(store, root)
            except Conflict as exc:
                if not self._is_deleted_session_conflict(exc):
                    raise
                self._drop_deleted_session_observation(root)
                candidates.discard(root)
                continue
            candidates.update(child_ids)
        for sid in sorted(candidates):
            try:
                registration = self._registration_for_cursor(sid, store=store)
                if registration is None:
                    continue
                config = registration.config
                if config.event_store is None and config.stream is None and not trace_processors(config):
                    self._notify_session(sid)
                    continue
                runtime = _ProjectionRuntime(
                    owner=self,
                    agent=registration.agent,
                    config=registration.config,
                    _resolved=registration.resolved,
                    handler_version=registration.handler_version,
                    wake=self._wake_session,
                )
                if self._observer_cursors.get(sid, 0) >= store.lease_status(sid)[0]:
                    self._notify_session(sid)
                    continue
                self._deliver_projected_events(store, sid, runtime)
            except Conflict as exc:
                if not self._is_deleted_session_conflict(exc):
                    raise
                self._drop_deleted_session_observation(sid)
                continue
            self._notify_session(sid)

    @staticmethod
    def _is_deleted_session_conflict(exc: Conflict) -> bool:
        return isinstance(exc, SessionNotLive) or str(exc) in {
            "session or cursor not found",
            "session not found in tenant",
        }

    def _drop_deleted_session_observation(self, sid: str) -> None:
        self._runtimes.pop(sid, None)
        self._observer_cursors.pop(sid, None)
        key = (self.tenant_id, sid)
        if _DRIVERS.get(key) is self:
            _DRIVERS.pop(key, None)

    def _wake_session(self, sid: str) -> None:
        self._worker.wake(sid)
        for handle in tuple(self.handles):
            if handle.session_id == sid:
                handle.notify()

    def _abandon_handle(self, handle: RunHandle) -> None:
        """Remove one client cursor and its unarmed start reservation, if any."""

        with suppress(ValueError):
            self.handles.remove(handle)
        registration = handle._runtime_reservation
        if registration is not None and self._runtimes.get(handle.session_id) is registration:
            self._runtimes.pop(handle.session_id, None)

    def _registration_for_cursor(
        self,
        sid: str,
        *,
        store: SessionStore | None = None,
    ) -> _RuntimeRegistration | None:
        from .bindings import MissingHostBinding

        registration = self._runtimes.get(sid)
        if registration is not None:
            return registration
        try:
            registration, _ = self._runtime_lineage(store or self.store, sid)
        except MissingHostBinding:
            return None
        return registration

    def _arm_handle(self, handle: RunHandle) -> None:
        if self._closed:
            raise RuntimeError("SessionDriver is closed")
        submission = handle._pending_submission
        if submission is not None:
            handle._pending_submission = None
            try:
                submission()
            except BaseException:
                if handle._pending_submission is None:
                    handle._pending_submission = submission
                raise
            handle._runtime_reservation = None
        else:
            registration = self._registration_for_cursor(handle.session_id)
            if registration is None:
                raise Conflict("run handle has no registered session runtime")
            if not registration.armed:
                raise Conflict("session already has an active RunHandle")
        self._worker.start()
        self._wake_session(handle.session_id)

    def _deliver_observer_event(self, config: RunConfig, event: RunEvent) -> None:
        if config.event_store is not None:
            try:
                config.event_store.append(event)
            except Exception as exc:
                if config.event_store_fail_closed:
                    raise
                warnings.warn(f"Run event store append failed: {exc}", RuntimeWarning, stacklevel=2)
        if config.stream is not None:
            with suppress(Exception):
                config.stream(event)

    def _emit_stream(
        self,
        sid: str,
        config: RunConfig,
        observer: Callable[[RunEvent], None] | None,
        watermark: int,
        event: RunEvent,
    ) -> None:
        del config
        if not isinstance(event, _TRANSIENT_STREAM_EVENTS):
            return
        if observer is not None:
            with suppress(Exception):
                observer(event)
        for handle in tuple(self.handles):
            if handle.session_id == sid:
                handle._add_stream_event(event, watermark)

    def _notify_session(self, sid: str) -> None:
        for handle in tuple(self.handles):
            if handle.session_id == sid:
                handle.notify()

    def _deliver_projected_events(
        self,
        store: SessionStore,
        sid: str,
        runtime: Runtime | _ProjectionRuntime,
    ) -> None:
        with _read_snapshot(store):
            concrete = _tenant_bound_sql_store(store)
            prefix = concrete.projection_prefix(sid) if isinstance(concrete, SQLStore) else None
            through_seq = prefix.head_seq if prefix is not None else store.read(sid, limit=1).head_seq
        cursor = self._observer_cursors.get(sid, 0)
        observer_config = runtime.config
        while cursor < through_seq:
            next_seq, events, _ = _project_record_page(
                store,
                sid,
                after_seq=cursor,
                through_seq=through_seq,
            )
            if next_seq <= cursor:
                break
            for event in events:
                self._deliver_observer_event(observer_config, event)
            cursor = next_seq
            self._observer_cursors[sid] = max(self._observer_cursors.get(sid, 0), cursor)
        processors = trace_processors(runtime.config)
        if processors:
            deliver_spans(store, sid, processors)

    def _discover_child_runtimes(self, store: SessionStore, sid: str) -> tuple[str, ...]:
        concrete = _tenant_bound_sql_store(store)
        transaction_in_progress = isinstance(concrete, SQLStore) and concrete._transaction_in_progress()
        parent, _ = self._runtime_lineage(store, sid)
        state, _, _ = store.read_state(sid)
        child_ids: list[str] = []
        for operation in state.operations.values():
            for attempt in operation.attempts.values():
                if attempt.child_handle is None:
                    continue
                for child in child_handles(attempt.child_handle):
                    child_sid = child["session_id"]
                    try:
                        page = store.read(child_sid, limit=1)
                        root, _ = self._runtime_lineage(store, child_sid)
                    except Conflict as exc:
                        if not self._is_deleted_session_conflict(exc):
                            raise
                        self._drop_deleted_session_observation(child_sid)
                        continue
                    if root is not parent:
                        raise Conflict("child session is not registered under this driver root")
                    child_ids.append(child_sid)
                    cursor = self._observer_cursors.get(child_sid)
                    is_new = cursor is None
                    if is_new:
                        self._observer_cursors[child_sid] = page.head_seq
                    _DRIVERS[(self.tenant_id, child_sid)] = self
                    wake = is_new or cursor < page.head_seq
                    if not wake and self._worker.drive_available and not transaction_in_progress:
                        wake = store.is_runnable(child_sid)
                    if wake:
                        self._worker.wake(child_sid)
        return tuple(dict.fromkeys(child_ids))

    def _deliver_child_session(self, store: SessionStore, sid: str) -> int:
        page = store.read(sid, limit=1)
        if not page.records:
            return 0
        parent_sid = page.records[0].record._payload.get("parent_session_id")
        if not isinstance(parent_sid, str) or not parent_sid:
            return 0
        delivered = 0

        def delivered_hook(point: str) -> None:
            nonlocal delivered
            if point == "after_parent_inbox":
                delivered += 1

        with store.atomic() as tx:
            count = child_delivery(store, tx, sid, hook=delivered_hook)
        if count and delivered:
            self._wake_session(parent_sid)
        return count

    def _project_worker_session(self, store: SessionStore, session_id: str, consumer: str) -> None:
        # Projection work is durable. If a worker died after the child appended
        # its terminal record but before delivery, replaying this consumer
        # closes that gap before the projection cursor is acknowledged.
        if consumer == "child_delivery":
            self._deliver_child_session(store, session_id)
            return
        # app_server, traces and events have independent owners. A local no-op
        # keeps their durable work pending without adding a writer transaction
        # to the drive/read critical path or acknowledging another owner.
        return

    @staticmethod
    def _lease_retry_policy(
        registration: _RuntimeRegistration,
        runtime: Runtime | None,
    ) -> tuple[int, float, float, Callable[[float, float], float]]:
        if runtime is not None:
            return (
                runtime.lease_retry_attempts,
                runtime.lease_retry_base_seconds,
                runtime.lease_retry_cap_seconds,
                runtime.lease_retry_jitter,
            )
        fields = Runtime.__dataclass_fields__
        return (
            cast(int, fields["lease_retry_attempts"].default),
            cast(float, fields["lease_retry_base_seconds"].default),
            cast(float, fields["lease_retry_cap_seconds"].default),
            cast(
                Callable[[float, float], float],
                fields["lease_retry_jitter"].default,
            ),
        )

    @staticmethod
    def _lease_retry_delay_ms(
        attempts: int,
        *,
        base_seconds: float,
        cap_seconds: float,
        jitter: Callable[[float, float], float],
    ) -> int:
        delay = min(cap_seconds, base_seconds * (2 ** max(0, attempts - 1)))
        selected = min(cap_seconds, max(0.0, jitter(0.0, delay)))
        return max(1, round(selected * 1000))

    def _terminal_failure(
        self,
        sid: str,
        turn_id: str | None,
        *,
        any_turn: bool = False,
    ) -> _SessionFailure | None:
        failure = self._errors.get(sid)
        if failure is None or not failure.terminal:
            return None
        if not any_turn and failure.turn_id not in {None, turn_id}:
            return None
        return failure

    def _failure_after_progress(
        self,
        store: SessionStore,
        sid: str,
        failure: _SessionFailure | None,
        *,
        state=None,
    ) -> _SessionFailure | None:
        """Clear a stale terminal error only after another writer made progress."""

        if failure is None or not failure.terminal or isinstance(failure.error, LedgerIntegrityError):
            return failure
        if store.read(sid, limit=1).head_seq <= failure.progress_head:
            return failure
        if isinstance(failure.error, ProviderResultInvalid):
            if not failure.fenced_operations:
                return failure
            if state is None:
                state, _, _ = store.read_state(sid)
            if state is not None and self._provider_failure_still_fenced(state, failure):
                return failure
        if self._errors.get(sid) is failure:
            self._errors.pop(sid, None)
        return None

    def _drive_worker_session(self, store: SessionStore, session_id: str, *, lease: Lease) -> None:
        try:
            registration, _ = self._runtime_lineage(store, session_id)
        except Exception:
            registration = None
        failure = self._failure_after_progress(store, session_id, self._errors.get(session_id))
        provider_failure = (
            failure if failure is not None and failure.terminal and isinstance(failure.error, ProviderResultInvalid) else None
        )
        if (
            registration is None
            or not registration.armed
            or (failure is not None and failure.terminal and provider_failure is None)
        ):
            try:
                store.defer_drive(lease, retry_after_ms=self._worker.support.failure_backoff_ms)
            except LeaseLost:
                pass
            finally:
                with suppress(LeaseLost):
                    store.release(lease)
            return
        provider_failure_polled: set[tuple[str, int]] = set()
        if provider_failure is not None:
            state, _, _ = store.read_state(session_id)
            provider_failure_polled = set(provider_failure.fenced_operations)
            consumable_input = self._provider_failure_has_consumable_input(store, session_id, state)
            if not provider_failure.fenced_operations and not consumable_input:
                # A non-query ProviderResultInvalid has no authority to infer a
                # current provider wait. Only an explicit consumable input may
                # pass through the core without inventing a fence identity.
                try:
                    store.defer_drive(lease, retry_after_ms=self._worker.support.failure_backoff_ms)
                except LeaseLost:
                    pass
                finally:
                    with suppress(LeaseLost):
                        store.release(lease)
                return
            if not consumable_input:
                if self._provider_failure_still_fenced(state, provider_failure):
                    try:
                        store.defer_drive(lease, retry_after_ms=self._worker.support.failure_backoff_ms)
                    except LeaseLost:
                        pass
                    finally:
                        with suppress(LeaseLost):
                            store.release(lease)
                    return
                if self._errors.get(session_id) is provider_failure:
                    self._errors.pop(session_id, None)
                failure = provider_failure = None
        try:
            before_head = store.read(session_id, limit=1).head_seq
        except LedgerIntegrityError as exc:
            fields = self._lease_retry_policy(registration, None)
            self._errors[session_id] = _SessionFailure(
                None,
                exc,
                1,
                0,
                True,
                fields[0],
                fields[1],
                fields[2],
                fields[3],
            )
            try:
                store.defer_drive(lease, retry_after_ms=self._worker.support.failure_backoff_ms)
            except LeaseLost:
                pass
            finally:
                with suppress(LeaseLost):
                    store.release(lease)
            self._notify_session(session_id)
            return
        except BaseException as exc:
            if isinstance(exc, sqlite3.OperationalError) and _sqlite_lock_error(exc):
                prior = failure
                attempts = 1 if prior is None else prior.attempts + 1
                first_cause = exc if prior is None else prior.error
                retry_attempts, retry_base, retry_cap, retry_jitter = self._lease_retry_policy(
                    registration,
                    None,
                )
                terminal = attempts >= retry_attempts
                self._errors[session_id] = _SessionFailure(
                    None if prior is None else prior.turn_id,
                    first_cause,
                    attempts,
                    0 if prior is None else prior.progress_head,
                    terminal,
                    retry_attempts,
                    retry_base,
                    retry_cap,
                    retry_jitter,
                )
                try:
                    if not terminal:
                        store.defer_drive(
                            lease,
                            retry_after_ms=self._lease_retry_delay_ms(
                                attempts + 1,
                                base_seconds=retry_base,
                                cap_seconds=retry_cap,
                                jitter=retry_jitter,
                            ),
                        )
                except LeaseLost:
                    pass
                finally:
                    with suppress(LeaseLost):
                        store.release(lease)
                self._notify_session(session_id)
                return
            try:
                store.defer_drive(lease, retry_after_ms=self._worker.support.failure_backoff_ms)
            except LeaseLost:
                pass
            finally:
                with suppress(LeaseLost):
                    store.release(lease)
            raise
        if failure is not None and not failure.terminal and before_head > failure.progress_head:
            # A keeper failure after a committed record is retryable, but it is
            # no longer consecutive once the durable cursor moved.
            self._errors.pop(session_id, None)
            failure = None
        runtime: Runtime | None = None
        ownership_transferred = False
        if failure is None:
            retry_attempts, retry_base, retry_cap, retry_jitter = self._lease_retry_policy(registration, runtime)
            initial_attempts = 0
        else:
            retry_attempts = failure.retry_attempts
            retry_base = failure.retry_base_seconds
            retry_cap = failure.retry_cap_seconds
            retry_jitter = failure.retry_jitter
            initial_attempts = failure.attempts
        retry_delay_ms = self._lease_retry_delay_ms(
            initial_attempts + 1,
            base_seconds=retry_base,
            cap_seconds=retry_cap,
            jitter=retry_jitter,
        )
        try:
            from . import kernel as kernel_module

            def make_runtime(selected_session_id: str) -> Runtime:
                nonlocal runtime
                runtime = self._build_runtime_for_claim(store, selected_session_id)
                return runtime

            # Once drive receives the claimed lease, DriveCore owns its release
            # and keeper handoff. In particular, a blocked keeper close that
            # leaves the database owner active must not be undone here.
            ownership_transferred = True
            kernel_module.drive(
                store,
                session_id,
                runtime=make_runtime,
                failure_backoff_ms=retry_delay_ms,
                _lease=lease,
                _initial_polled=provider_failure_polled,
            )
            if runtime is None:
                # RuntimeNotReady is a successful deferred claim inside
                # DriveCore. It owns the release and retry schedule; there is no
                # execution runtime to use for child delivery or projection.
                return
            if provider_failure is None:
                self._errors.pop(session_id, None)
            else:
                after_state, _, _ = store.read_state(session_id)
                after_failure = self._errors.get(session_id)
                if after_failure is provider_failure and self._provider_failure_still_fenced(after_state, provider_failure):
                    provider_failure.progress_head = store.read(session_id, limit=1).head_seq
                elif after_failure is provider_failure:
                    self._errors.pop(session_id, None)
            if store.read(session_id, limit=1).head_seq == before_head:
                self._defer_held_steer_claim(store, session_id)
        except BaseException as exc:
            after_head = store.read(session_id, limit=1).head_seq
            state = None
            try:
                state, _, _ = store.read_state(session_id)
                tid = state.active_turn_id or next(reversed(state.turns), None)
            except Exception:
                tid = None
            if isinstance(exc, LeaseLost):
                # Ownership is checked before terminal acknowledgement: a keeper
                # close failure can arrive after turn_ended and must remain a
                # visible operational failure while the DB lease is retained.
                _, current_epoch, remaining_ms = store.lease_status(session_id)
                retained_ownership = current_epoch == lease.epoch and remaining_ms > 0
                terminal_committed = bool(
                    state is not None
                    and tid is not None
                    and tid in state.turns
                    and state.turns[tid].ended
                    and not retained_ownership
                )
                if terminal_committed:
                    # The durable turn outcome already fences another dispatch.
                    # Treat the post-commit loss as acknowledgement of completed
                    # work instead of scheduling a retry that can never be claimed.
                    self._errors.pop(session_id, None)
                    self._notify_session(session_id)
                    return
                progressed = after_head > before_head
                prior = None if progressed else failure
                attempts = 1 if prior is None else prior.attempts + 1
                first_cause = exc if prior is None else prior.error
                retry_attempts, retry_base, retry_cap, retry_jitter = self._lease_retry_policy(
                    registration,
                    runtime,
                )
                # A heartbeat worker that could not stop still owns this exact
                # database lease. Surface that operational failure immediately,
                # but neither defer it nor attempt an outer release.
                terminal = retained_ownership or isinstance(exc, LeaseRetryExhausted) or attempts >= retry_attempts
                if terminal:
                    if retained_ownership or isinstance(exc, LeaseRetryExhausted):
                        reported: BaseException = exc
                        if isinstance(exc, LeaseRetryExhausted):
                            attempts = exc.attempts
                    else:
                        reported = LeaseRetryExhausted(session_id, tid or "", attempts)
                        reported.__cause__ = first_cause
                    self._errors[session_id] = _SessionFailure(
                        tid,
                        reported,
                        attempts,
                        after_head,
                        True,
                        retry_attempts,
                        retry_base,
                        retry_cap,
                        retry_jitter,
                    )
                else:
                    self._errors[session_id] = _SessionFailure(
                        tid,
                        first_cause,
                        attempts,
                        after_head,
                        False,
                        retry_attempts,
                        retry_base,
                        retry_cap,
                        retry_jitter,
                    )
                    # DriveCore normally defers before returning a LeaseLost. A
                    # pre-core failure has not transferred deferral ownership,
                    # so provide the same bounded backoff here. If keeper close
                    # retained ownership, defer fails closed and is left for the
                    # active lease to expire.
                    with suppress(LeaseLost):
                        store.defer_drive(
                            lease,
                            retry_after_ms=self._lease_retry_delay_ms(
                                attempts + 1,
                                base_seconds=retry_base,
                                cap_seconds=retry_cap,
                                jitter=retry_jitter,
                            ),
                        )
            elif isinstance(exc, sqlite3.OperationalError) and _sqlite_lock_error(exc):
                # Keep SQLite contention inside the normal bounded DriveCore retry
                # path. Any committed op_started marker is still reconciled as an
                # unknown outcome before another external call can be planned.
                if (
                    provider_failure is not None
                    and state is not None
                    and self._provider_failure_still_fenced(state, provider_failure)
                ):
                    provider_failure.progress_head = after_head
                    return
                progressed = after_head > before_head
                prior = None if progressed else failure
                attempts = 1 if prior is None else prior.attempts + 1
                first_cause = exc if prior is None else prior.error
                retry_attempts, retry_base, retry_cap, retry_jitter = self._lease_retry_policy(
                    registration,
                    runtime,
                )
                terminal = attempts >= retry_attempts
                self._errors[session_id] = _SessionFailure(
                    tid,
                    first_cause,
                    attempts,
                    after_head,
                    terminal,
                    retry_attempts,
                    retry_base,
                    retry_cap,
                    retry_jitter,
                )
                if not terminal and not ownership_transferred:
                    with suppress(LeaseLost):
                        store.defer_drive(
                            lease,
                            retry_after_ms=self._lease_retry_delay_ms(
                                attempts + 1,
                                base_seconds=retry_base,
                                cap_seconds=retry_cap,
                                jitter=retry_jitter,
                            ),
                        )
            elif isinstance(exc, KeyUnavailable):
                progressed = after_head > before_head
                prior = None if progressed else failure
                attempts = 1 if prior is None else prior.attempts + 1
                first_cause = exc if prior is None else prior.error
                retry_attempts, retry_base, retry_cap, retry_jitter = self._lease_retry_policy(
                    registration,
                    runtime,
                )
                self._errors[session_id] = _SessionFailure(
                    tid,
                    first_cause,
                    attempts,
                    after_head,
                    False,
                    retry_attempts,
                    retry_base,
                    retry_cap,
                    retry_jitter,
                )
                with suppress(LeaseLost):
                    store.defer_drive(
                        lease,
                        retry_after_ms=self._lease_retry_delay_ms(
                            attempts + 1,
                            base_seconds=retry_base,
                            cap_seconds=retry_cap,
                            jitter=retry_jitter,
                        ),
                    )
            else:
                # DriveCore already raises the original factory exception from
                # its backoff failure. Preserve identity and the existing cause
                # chain for the subscriber.
                if isinstance(exc, ProviderResultInvalid):
                    after_state, _, _ = store.read_state(session_id)
                    query_key = exc._query_operation_key
                    if (
                        provider_failure is not None
                        and query_key is not None
                        and query_key in provider_failure.fenced_operations
                        and self._provider_failure_still_fenced(after_state, provider_failure)
                    ):
                        provider_failure.progress_head = after_head
                        return
                    fields = self._lease_retry_policy(registration, runtime)
                    self._errors[session_id] = _SessionFailure(
                        tid,
                        exc,
                        1,
                        after_head,
                        True,
                        fields[0],
                        fields[1],
                        fields[2],
                        fields[3],
                        fenced_operations=frozenset({query_key}) if query_key is not None else frozenset(),
                    )
                    return
                reported = exc
                fields = self._lease_retry_policy(registration, runtime)
                self._errors[session_id] = _SessionFailure(
                    tid,
                    reported,
                    1,
                    after_head,
                    True,
                    fields[0],
                    fields[1],
                    fields[2],
                    fields[3],
                )
            # The subscriber receives a terminal typed failure from the single
            # session error slot. Returning normally prevents the outer worker
            # from second-guessing DriveCore's keeper-close ownership decision.
        finally:
            if not ownership_transferred:
                with suppress(LeaseLost):
                    store.release(lease)
            self._notify_session(session_id)

    def _defer_held_steer_claim(self, store: SessionStore, sid: str) -> None:
        """Rotate past a waiting parent whose undeclared steer cannot be consumed yet."""

        state, _, _ = store.read_state(sid)
        entries = self._inbox_entries(store, sid)
        if not entries or not all(_steer_held_in_state(state, entry.item, one_turn=False) for entry in entries):
            return
        retry_after_ms = max(1, round(self._worker.scan_interval * 1000))
        with suppress(LeaseLost):
            rotation_lease = store.acquire(
                sid,
                owner=f"local-rotation/{uuid4().hex}",
                ttl_ms=self._worker.support.lease_ttl_ms,
            )
            if rotation_lease is None:
                return
            try:
                store.defer_drive(rotation_lease, retry_after_ms=retry_after_ms)
            finally:
                store.release(rotation_lease)

    def _read_observed_state(self, store: SessionStore, sid: str):
        # Status and scheduling only need the raw folded state. Subscribers
        # hydrate the captured source records after this consistent raw read.
        with _read_snapshot(store):
            return store.read_state(sid)

    @staticmethod
    def _inbox_entries(store: SessionStore, sid: str):
        """Read the complete ready inbox without treating the first 256 rows as the boundary."""

        entries = store.peek_inbox(sid, limit=256)
        if len(entries) < 256:
            return entries
        inbox_seq = store.read(sid, limit=1).inbox_seq
        return store.peek_inbox(sid, limit=max(257, inbox_seq))

    def _pending_child_work(self, store: SessionStore, state) -> bool:
        seen: set[str] = set()

        def child_pending(child: dict[str, Any]) -> bool:
            child_sid = child["session_id"]
            if child_sid in seen:
                raise Conflict("session child wait graph contains a cycle")
            seen.add(child_sid)
            try:
                failure = self._terminal_failure(child_sid, None, any_turn=True)
                if failure is not None:
                    raise failure.error
                child_state, _, _ = self._read_observed_state(store, child_sid)
                if child_state is None:
                    return True
                requested = child_state.active_turn_id or child.get("turn_id")
                turn = child_state.turns.get(requested) if isinstance(requested, str) else None
                if turn is None:
                    turn = child_state.turns.get(next(reversed(child_state.turns), ""))
                if turn is not None and (turn.wait is not None or turn.suspended):
                    # A real answer/evidence resumes the paused child. Steering and
                    # future follow-ups stay queued until its current result exists.
                    return any(
                        entry.item.kind not in {"steer", "follow_up"}
                        and not (entry.item.kind == "user" and entry.item.target_turn_id is None)
                        for entry in self._inbox_entries(store, child_sid)
                    )
                handles = [wait["handle"] for wait in child_state.waits.values()]
                if any(handle.get("kind") in {"user", "approval"} for handle in handles):
                    return False
                _, _, lease_remaining = store.lease_status(child_sid)
                if lease_remaining > 0 or _is_runnable_in_snapshot(store, child_sid):
                    return True
                if _has_pending_drive_input(store, child_sid, child_state):
                    return True
                nested = [
                    grandchild
                    for handle in handles
                    if handle.get("kind") == "child"
                    for grandchild in child_handles(handle)
                    if not grandchild.get("background")
                ]
                if nested:
                    return any(child_pending(grandchild) for grandchild in nested)
                # A terminal child remains pending until child_delivery appends
                # the parent result. A live unpaused child is also pending.
                return True
            finally:
                seen.remove(child_sid)

        for wait in state.waits.values():
            handle = wait["handle"]
            if handle.get("kind") == "provider" and handle.get("provider") == SUB_TASK_STATUS_TOOL_NAME:
                # sub_task_status is the durable observation edge for a running
                # child. Its poll timeout is progress, not a public WAIT_USER edge.
                return True
            if handle.get("kind") != "child":
                continue
            for child in child_handles(handle):
                if child.get("background"):
                    continue
                if child_pending(child):
                    return True
        return False

    @staticmethod
    def _provider_failure_has_consumable_input(store: SessionStore, sid: str, state) -> bool:
        entries = SessionDriver._inbox_entries(store, sid)
        pending_batch = _pending_batch_in_state(state)
        return any(
            not (entry.item.kind == "model_selection" and pending_batch)
            and not (entry.item.kind == "steer" and _steer_held_in_state(state, entry.item, one_turn=False))
            for entry in entries
        )

    def _provider_failure_still_fenced(
        self,
        state,
        failure: _SessionFailure,
    ) -> bool:
        if failure.turn_id is None or not failure.fenced_operations:
            return False
        turn = state.turns.get(failure.turn_id)
        if state.closed or turn is None or turn.ended or turn.cancelled:
            return False
        return any(
            wait["handle"].get("kind") == "provider"
            for (oid, attempt), wait in state.waits.items()
            if (oid, attempt) in failure.fenced_operations
        )

    def _snapshot_status(
        self,
        store: SessionStore,
        sid: str,
        state,
        requested_turn_id: str,
        *,
        captured_head: int | None = None,
    ) -> tuple[str, bool, BaseException | None]:
        with _read_snapshot(store):
            tid = requested_turn_id
            failure = self._failure_after_progress(store, sid, self._errors.get(sid), state=state)
            failure = self._terminal_failure(sid, tid) if failure is not None else None
            if failure is not None and isinstance(failure.error, ProviderResultInvalid) and state is None:
                state, _, _ = self._read_observed_state(store, sid)
            pending_provider_input = bool(
                failure is not None
                and isinstance(failure.error, ProviderResultInvalid)
                and state is not None
                and self._provider_failure_has_consumable_input(store, sid, state)
            )
            error = failure.error if failure is not None and not pending_provider_input else None
            if error is not None and not isinstance(error, ProviderResultInvalid):
                return tid, True, error
            _, _, lease_remaining = store.lease_status(sid)
            if lease_remaining > 0:
                return tid, False, None
            if error is not None:
                return tid, True, error
            inbox_entries = self._inbox_entries(store, sid)
            records = ()
            if state is None:
                state, records, _ = self._read_observed_state(store, sid)
            if state is None:
                return tid, False, None
            captured_runnable, captured_schedule = _drive_flags_in_snapshot(store, sid)
            held_steer = bool(inbox_entries) and all(
                _steer_held_in_state(state, entry.item, one_turn=False) for entry in inbox_entries
            )
            if pending_provider_input or captured_schedule or (captured_runnable and not held_steer):
                # A captured schedule still owns the next drive. The durable wait
                # remains until the worker advances it, so a subscriber must not
                # project a stale WAIT result from the terminal-looking prefix.
                return tid, False, error
            if inbox_entries and not held_steer:
                return tid, False, None
            if requested_turn_id in state.turns:
                tid = requested_turn_id
            elif state.active_turn_id is not None:
                tid = state.active_turn_id
            elif state.turns:
                tid = next(reversed(state.turns))
            if tid not in state.turns:
                return tid, False, None
            # A terminal record is durable before DriveCore closes its keeper and
            # releases ownership. Keep the subscriber active until ownership is
            # gone, so a keeper-close failure can surface as the turn error.
            if self._pending_child_work(store, state):
                return tid, False, error
            turn = state.turns[tid]
            if not turn.ended and not turn.suspended and not turn.wait and not state.waits:
                return tid, False, error
            observer_cursor = self._observer_cursors.get(sid)
            head_seq = captured_head
            if head_seq is None:
                head_seq = records[-1].seq if records else store.read(sid, limit=1).head_seq
            if observer_cursor is not None and observer_cursor < head_seq:
                # The drive lease is released before the local observer projection
                # drains its complete committed prefix. Keep the turn active until
                # the stream owner has delivered the record cursor.
                return tid, False, error
            return tid, True, error

    def _handle_snapshot(
        self,
        sid: str,
        requested_turn_id: str,
    ) -> tuple[Any, tuple[Any, ...], int, str, bool, BaseException | None]:
        return self._snapshot_from_store(self.store, sid, requested_turn_id)

    def _wait_for_background_work(self, root_session_id: str) -> bool:
        """Wait for descendants and report whether a paused child owns resumability."""

        retain_owner = False
        while True:
            pending = False
            with self._open_drive_store() as store, _read_snapshot(store):
                known_sessions = {root_session_id}
                discovery_queue = list(known_sessions)
                while discovery_queue:
                    parent_id = discovery_queue.pop()
                    for child_id in self._discover_child_runtimes(store, parent_id):
                        if child_id in known_sessions:
                            continue
                        known_sessions.add(child_id)
                        discovery_queue.append(child_id)
                child_ids = known_sessions - {root_session_id}

                for child_id in sorted(child_ids):
                    state, _, _ = self._read_observed_state(store, child_id)
                    requested_turn_id = state.active_turn_id if state is not None else None
                    if requested_turn_id is None and state is not None and state.turns:
                        requested_turn_id = next(reversed(state.turns))
                    if requested_turn_id is None:
                        pending = True
                        continue
                    turn = state.turns.get(requested_turn_id) if state is not None else None
                    if turn is not None and not turn.ended and (turn.wait or turn.suspended or state.waits):
                        retain_owner = True
                        continue
                    _, quiescent, error = self._snapshot_status(
                        store,
                        child_id,
                        state,
                        requested_turn_id,
                    )
                    if error is not None:
                        raise error
                    if not quiescent:
                        pending = True
            if not pending:
                return retain_owner
            if self._closed:
                raise RuntimeError("SessionDriver closed before background work became quiescent")
            sleep(0.05)

    def _snapshot_from_store(
        self,
        store: SessionStore,
        sid: str,
        requested_turn_id: str,
    ) -> tuple[Any, tuple[Any, ...], int, str, bool, BaseException | None]:
        with _read_snapshot(store):
            failure = self._failure_after_progress(store, sid, self._errors.get(sid))
            failure = self._terminal_failure(sid, requested_turn_id) if failure is not None else None
            if failure is not None and not isinstance(failure.error, ProviderResultInvalid):
                return None, (), 0, failure.turn_id or requested_turn_id, True, failure.error
            state, records, inbox_seq = self._read_observed_state(store, sid)
            captured_head = records[-1].seq if records else store.read(sid, limit=1).head_seq
            tid, quiescent, error = self._snapshot_status(
                store,
                sid,
                state,
                requested_turn_id,
                captured_head=captured_head,
            )
            return state, records, inbox_seq, tid, quiescent, error

    def _handle_turn_id(self, sid: str, requested_turn_id: str) -> str:
        state, _, _ = self._read_observed_state(self.store, sid)
        if state is None:
            return requested_turn_id
        if requested_turn_id in state.turns:
            return requested_turn_id
        return state.active_turn_id or next(reversed(state.turns), requested_turn_id)

    def _handle_steer_generation(self, sid: str, turn_id: str) -> int | None:
        state, _, _ = self._read_observed_state(self.store, sid)
        turn = state.turns.get(turn_id) if state is not None else None
        if turn is None:
            return None
        generation = turn.start._payload.get("generation")
        return generation if type(generation) is int else None

    def _handle_event_page(
        self,
        sid: str,
        requested_turn_id: str,
        *,
        after_seq: int,
        store: SessionStore | None = None,
    ) -> tuple[str, list[RunEvent], int, bool, bool, BaseException | None]:
        """Project one page from the exact caller store without retaining iterator state."""

        selected_store = store or self.store
        with _read_snapshot(selected_store):
            delivery = _projection_delivery(selected_store, sid)
            if delivery.prefix is None:
                state, records, _ = self._read_observed_state(selected_store, sid)
                captured_head = records[-1].seq if records else selected_store.read(sid, limit=1).head_seq
            else:
                state = delivery.prefix.reducer.snapshot()
                records = delivery.prefix.records
                captured_head = delivery.prefix.head_seq
            tid, quiescent, error = self._snapshot_status(
                selected_store,
                sid,
                state,
                requested_turn_id,
                captured_head=captured_head,
            )
            page, context = _capture_record_page(
                selected_store,
                sid,
                after_seq=after_seq,
                delivery=delivery,
            )
        if page.records:
            next_seq, events, has_more = _project_captured_record_page(
                selected_store,
                sid,
                page,
                context,
                run_id=tid,
            )
        else:
            next_seq, events, has_more = after_seq, [], False
        return tid, events, next_seq, has_more, quiescent, error

    def _handle_done(self, sid: str, requested_turn_id: str) -> bool:
        if self._closed:
            return True
        try:
            _, quiescent, error = self._snapshot_status(self.store, sid, None, requested_turn_id)
        except LedgerIntegrityError:
            raise
        except (Conflict, KeyError):
            return False
        return quiescent or error is not None

    def _wait_for_handle(self, handle: RunHandle, timeout: float | None) -> None:
        from time import monotonic

        deadline = None if timeout is None else monotonic() + timeout
        while True:
            if self._handle_done(handle.session_id, handle.run_id):
                return
            if self._closed:
                raise RuntimeError("SessionDriver closed before the run became quiescent")
            remaining = None if deadline is None else deadline - monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("session execution is still active")
            handle._notify.wait(0.05 if remaining is None else min(0.05, remaining))
            handle._notify.clear()

    def _project_handle_result(self, handle: RunHandle) -> RunResult:
        store = self.store
        state, source_records, inbox_seq, tid, _, error = self._snapshot_from_store(
            store,
            handle.session_id,
            handle.run_id,
        )
        if error is not None:
            raise error
        if not tid:
            raise Conflict("session has no admitted turn")
        handle.run_id = tid
        runtime = self._projection_runtime_view(handle.session_id, store=store)
        result = project_result(
            store,
            handle.session_id,
            tid,
            runtime=cast(Runtime, runtime),
            snapshot=(state, source_records, inbox_seq),
        )
        waits = []
        has_waits = state is not None and (
            bool(state.waits)
            or any(turn.wait is not None for turn in state.turns.values())
            or any(
                attempt.child_handle is not None and any(child.get("background") for child in child_handles(attempt.child_handle))
                for operation in state.operations.values()
                for attempt in operation.attempts.values()
            )
        )
        if has_waits:
            waits = self.waits(
                handle.session_id,
                runtime,
                snapshot=(state, source_records, inbox_seq),
                store=store,
            )
        if waits:
            result.metadata["session_waits"] = [{"session_id": sid, **wait} for sid, _, wait in waits]
            result.raw_result.wait_reason = waits[0][2].get("question", result.raw_result.wait_reason)
        result._session_driver = self
        return result

    def close(self) -> None:
        self._closed = True
        for key, driver in tuple(_DRIVERS.items()):
            if driver is self:
                _DRIVERS.pop(key, None)
        if not self._worker.close():
            raise RuntimeError("SessionDriver worker did not stop")
        self._notify_session_all()
        callbacks = _STORE_CLOSERS.pop(self._root_store, []) if self._owns_store else list(self._close_callbacks)
        for callback in callbacks:
            callback()
            retained = _STORE_CLOSERS.get(self._root_store, [])
            if callback in retained:
                retained.remove(callback)
        self._close_callbacks.clear()
        if self._owns_store and isinstance(self._root_store, CoreSQLiteStore):
            with self._root_store._connection_lock:
                self._root_store.connection.close()
        if self._temporary_content_key_state is not None:
            cast(LocalKeyProvider, self.content_key_provider).close()
            self._temporary_content_key_state.cleanup()
            self._temporary_content_key_state = None
        if self._temporary_store_directory is not None:
            self._temporary_store_directory.cleanup()
            self._temporary_store_directory = None

    def _notify_session_all(self) -> None:
        for handle in list(self.handles):
            handle.notify()

    def _on_close(self, callback):
        self._close_callbacks.append(callback)
        _STORE_CLOSERS.setdefault(self._root_store, []).append(callback)

    def _bound_principal(
        self,
        principal: PrincipalInput | None,
        *,
        subject_id: str,
        writer: SessionContentWriter,
    ) -> dict[str, Any]:
        if principal is None:
            bound, _ = writer.service_principal(subject_id)
            return bound
        if isinstance(principal, Principal):
            raise ValueError("session creation requires a host ingress principal mapping")
        if not isinstance(principal, Mapping):
            raise ValueError("session creation requires a host ingress principal mapping")
        return _ingress_principal(cast(Mapping[str, Any], principal), tenant_id=self.tenant_id)

    def _persisted_actor(
        self,
        actor: Principal | Mapping[str, Any],
        *,
        writer: SessionContentWriter,
        input_id: str | None,
    ) -> tuple[dict[str, Any], ContentRef]:
        payload = actor.to_dict() if isinstance(actor, Principal) else Principal.from_dict(actor).to_dict()
        _principal_identity(payload, tenant_id=self.tenant_id)
        ref = ContentRef.from_dict(payload["attributes"])
        verified: ContentRef | None = None
        if input_id is not None:
            with suppress(ContentUnavailable):
                verified = writer.assert_input(input_id, ref, INPUT_ACTOR_ATTRIBUTES)
        if verified is None:
            try:
                verified = writer.assert_record(1, ref, SESSION_PRINCIPAL_ATTRIBUTES)
            except ContentUnavailable as exc:
                raise ValueError("persisted actor attributes require a target-session owner link") from exc
        copied = writer.copy_json_from(writer, verified)
        return payload | {"attributes": copied.to_dict()}, copied

    def _bound_actor(
        self,
        actor: PrincipalInput | None,
        *,
        subject_id: str,
        writer: SessionContentWriter,
        input_id: str | None = None,
        persisted: bool = False,
    ) -> tuple[dict[str, Any], ContentRef]:
        if actor is None:
            principal, attributes = writer.service_principal(subject_id)
            return principal, attributes
        persisted_actor = _persisted_principal(actor)
        if persisted or persisted_actor is not None:
            return self._persisted_actor(persisted_actor or actor, writer=writer, input_id=input_id)
        if isinstance(actor, Principal):
            raise TypeError("input actor must be a Principal or plaintext ingress mapping")
        if not isinstance(actor, Mapping):
            raise TypeError("input actor must be a Principal or plaintext ingress mapping")
        principal = _ingress_principal(cast(Mapping[str, Any], actor), tenant_id=self.tenant_id)
        ref = writer.put_json(principal["attributes"])
        return principal | {"attributes": ref.to_dict()}, ref

    def _prepare_item(self, sid: str, item: InboxItem, writer: SessionContentWriter) -> InboxItem:
        return declare_input_content(writer, item).item

    def _session_principal(self, sid: str) -> Principal:
        page = self.store.read(sid, limit=1)
        if not page.records:
            raise Conflict("session principal is unavailable")
        principal = Principal.from_dict(page.records[0].record.payload["principal"])
        if principal.tenant_id != self.tenant_id:
            raise Conflict("session principal belongs to another tenant")
        return principal

    def create(
        self,
        sid: str,
        workspace: str,
        attributes: dict[str, Any] | None = None,
        *,
        principal: PrincipalInput | None = None,
    ) -> None:
        if attributes is not None:
            seed = attributes.get("seed") if isinstance(attributes, Mapping) else None
            if isinstance(seed, Mapping) and "messages" in seed:
                messages = seed["messages"]
                if not isinstance(messages, list):
                    raise TypeError("seed.messages must be a list")
                for message in messages:
                    Message.from_dict(message)
        bootstrap = self._next_session_requirements
        self._next_session_requirements = None
        if bootstrap is not None and bootstrap.level == 1:
            # L1 static capabilities bind the worker; only L2/L3 requirements are durable bootstrap.
            bootstrap = None
        with self.store.atomic() as tx:
            writer = SessionContentWriter.bind(tx, tenant_id=self.tenant_id, session_id=sid)

            def session_spec(content: SessionContentWriter) -> SessionSpec:
                bound_principal = self._bound_principal(principal, subject_id="local", writer=content)
                created = publish_session_created_content(
                    content,
                    principal=bound_principal,
                    workspace=str(workspace),
                    attributes=attributes or {},
                )
                return SessionSpec(
                    sid,
                    created.principal,
                    created.workspace.to_dict(),
                    attributes=created.attributes.to_dict(),
                    feature_level=2 if bootstrap is not None else 1,
                    bootstrap=bootstrap,
                )

            writer.create_session(cast(Any, session_spec), consumers=("events", "app_server", "traces"))
        _DRIVERS[(self.tenant_id, sid)] = self

    def runtime(
        self,
        agent: Agent,
        config: RunConfig,
        task: AgentTask | None = None,
        *,
        session_id: str | None = None,
        store: SessionStore | None = None,
        handler_version: str | None = None,
    ) -> Runtime:

        llm, resolved = resolve_run_model(agent=agent, run_config=config)
        metadata = dict(agent.metadata) | config.metadata
        settings = config.model_provider.default_settings(resolved) if config.model_provider else agent.model_settings
        settings = settings.resolve(agent.model_settings).resolve(config.model_settings) if settings else config.model_settings
        reserved = settings.max_tokens if settings else None
        source = "model_settings" if reserved is not None else "framework_fallback"
        if reserved is None and isinstance(metadata.get("reserved_output_tokens"), int):
            reserved, source = metadata["reserved_output_tokens"], "task_metadata"
        if reserved is None:
            reserved = min(16_000, resolved.max_output_tokens) if resolved.max_output_tokens is not None else 16_000
            if reserved < 16_000:
                source = "framework_fallback_capped_by_model_capability"
        memory = MemoryManager(
            model_context_window=resolved.context_length or metadata.get("model_context_window") or 279_000,
            model_max_output_tokens=resolved.max_output_tokens,
            reserved_output_tokens=reserved,
            reserved_output_source=source,
            compact_threshold=task.memory_compact_threshold if task else 250_000,
            autocompact_buffer_tokens=metadata.get("autocompact_buffer_tokens", 13_000),
            keep_recent_messages=metadata.get("memory_keep_recent_messages", 10),
            microcompaction_policy=task.microcompaction_policy if task else config.microcompaction_policy,
        )
        content_reader = (
            durable_content_reader(store or self._root_store, tenant_id=self.tenant_id, session_id=session_id)
            if session_id is not None
            else None
        )
        runtime = Runtime(
            agent,
            config,
            resolved,
            llm,
            (store.heartbeat_store if store is not None else self._heartbeat_store),
            providers=dict(config.tool_providers),
            frozen_task=task,
            memory_manager=memory,
            model_resolver=self.model_resolver,
            handler_version=self._handler_version(handler_version),
            wake=self._wake_session if hasattr(self, "_worker") else lambda _sid: None,
            content_reader=content_reader,
        )
        if store is None and content_reader is not None:
            object.__setattr__(runtime, "_session_driver_default_content_reader", content_reader)
        provider = config.model_provider
        if self.model_resolver is None and provider is not None:
            runtime.model_resolver = lambda request: resolve_model_selection(provider, request)
        return runtime

    def _heartbeat_store(self):
        return self.store.heartbeat_store()

    def start(
        self,
        sid: str,
        agent: Agent,
        config: RunConfig,
        content: Any | None,
        *,
        input_id: str | None = None,
        task: AgentTask | None = None,
        handler_version: str | None = None,
        consumer: str = "events",
        autostart: bool = True,
        actor: PrincipalInput | None = None,
        actor_subject_id: str = "local",
    ) -> RunHandle:
        if self._closed:
            raise RuntimeError("SessionDriver is closed")
        state, source_records, inbox_seq = self.store.read_state(sid)
        if state.closed or state.archived:
            raise RuntimeError("session is closed or archived")
        if any(handle.session_id == sid and handle._runtime_reservation is not None for handle in self.handles):
            raise Conflict("session already has an active RunHandle")
        existing_registration = self._runtimes.get(sid)
        if existing_registration is None:
            existing_registration = self._registration_for_cursor(sid)
        if content is None and existing_registration is not None:
            if not existing_registration.armed:
                raise Conflict("session already has an active RunHandle")
            queued = next(
                (
                    record._payload["input"]
                    for key, record in state.applied_inputs.items()
                    if record._payload["disposition"] == "queued" and key not in state.admitted_inputs
                ),
                None,
            )
            pending_input = next(
                (
                    entry.item
                    for entry in self.store.peek_inbox(sid)
                    if entry.item.kind in {"user", "follow_up"} and entry.item.target_turn_id is None
                ),
                None,
            )
            tid = state.active_turn_id or (
                f"{sid}/turn/{queued['input_id']}"
                if queued
                else f"{sid}/turn/{pending_input.input_id}"
                if pending_input
                else next(reversed(state.turns), "")
            )
            cursor = RunHandle(
                self,
                sid,
                tid,
                None,
                consumer,
                handler_version=existing_registration.handler_version,
            )
            self.handles.append(cursor)
            self._observer_cursors.setdefault(sid, source_records[-1].seq if source_records else 0)
            if autostart:
                cursor.start()
            return cursor
        selected_handler = self._handler_version(handler_version)
        if content is not None and existing_registration is not None and not existing_registration.armed:
            raise Conflict("session already has an active RunHandle")
        if state.turns and config.shared_state is None:
            prior = project_result(
                self.store,
                sid,
                next(reversed(state.turns)),
                snapshot=(state, source_records, inbox_seq),
            )
            config = replace(config, shared_state=prior.raw_result.shared_state)
        if task is not None and config.shared_state is not None:
            task = replace(task, initial_shared_state=dict(config.shared_state))
        if content is not None:
            if state.active_turn_id is not None or any(
                entry.item.kind in {"user", "follow_up"} for entry in self.store.peek_inbox(sid)
            ):
                raise Conflict("session already has an active RunHandle or pending input")
            identity = input_id or uuid4().hex
            tid = f"{sid}/turn/{identity}"
        else:
            identity = None
            queued = next(
                (
                    record._payload["input"]
                    for key, record in state.applied_inputs.items()
                    if record._payload["disposition"] == "queued" and key not in state.admitted_inputs
                ),
                None,
            )
            pending_input = next(
                (
                    entry.item
                    for entry in self.store.peek_inbox(sid)
                    if entry.item.kind in {"user", "follow_up"} and entry.item.target_turn_id is None
                ),
                None,
            )
            tid = state.active_turn_id or (
                f"{sid}/turn/{queued['input_id']}"
                if queued
                else f"{sid}/turn/{pending_input.input_id}"
                if pending_input
                else next(reversed(state.turns), "")
            )
        registration: _RuntimeRegistration | None = None
        try:
            if identity is not None:
                registration = (
                    existing_registration
                    if existing_registration is not None and existing_registration.armed
                    else self._static_registration(
                        sid,
                        agent,
                        config,
                        task,
                        handler_version=selected_handler,
                    )
                )

                def submission() -> None:
                    assert registration is not None
                    self._admit_start(
                        sid,
                        registration,
                        InboxItem(identity, "user", {"content": content}),
                        actor=actor,
                        actor_subject_id=actor_subject_id,
                    )

            else:
                submission = None
            handle = RunHandle(self, sid, tid, None, consumer, handler_version=selected_handler)
            handle._pending_submission = submission
            handle._runtime_reservation = registration
            self._observer_cursors.setdefault(sid, source_records[-1].seq if source_records else 0)
            self.handles.append(handle)
            if autostart:
                handle.start()
        except BaseException:
            if "handle" in locals():
                with suppress(ValueError):
                    self.handles.remove(handle)
            raise
        return handle

    def _admit_start(
        self,
        sid: str,
        registration: _RuntimeRegistration,
        item: InboxItem,
        *,
        actor: PrincipalInput | None,
        actor_subject_id: str,
    ) -> None:
        """Bind one initial input under the session's existing SQL admission lock."""

        armed = replace(registration, armed=True)
        with self.store.atomic() as tx:
            ceiling = cast(Any, self.store)._write_ceiling()
            tx.lock_session(sid)
            previous_registration = self._runtimes.get(sid)
            try:
                state, _, _ = self.store.read_state(sid)
                if state.closed or state.archived:
                    raise RuntimeError("session is closed or archived")
                if state.active_turn_id is not None or any(
                    entry.item.kind in {"user", "follow_up"} for entry in self.store.peek_inbox(sid)
                ):
                    raise Conflict("session already has an active RunHandle or pending input")
                self._runtimes[sid] = armed
                self._push_in_transaction(
                    tx,
                    sid,
                    item,
                    actor=actor,
                    actor_subject_id=actor_subject_id,
                    registration=armed,
                    ceiling=ceiling,
                )
            except BaseException:
                if self._runtimes.get(sid) is armed:
                    if previous_registration is None:
                        self._runtimes.pop(sid, None)
                    else:
                        self._runtimes[sid] = previous_registration
                raise
        self._errors.pop(sid, None)

    def _route_inbox_item(
        self,
        sid: str,
        item: InboxItem,
        *,
        registration: _RuntimeRegistration | None = None,
        ceiling: int | None = None,
    ) -> InboxItem:
        """Attach the contract-41 requirement for one level-2 logical input."""

        if item.kind not in {"user", "follow_up", "model_selection"} or item.feature_level == 2:
            return item
        if registration is None:
            registration = self._registration_for_cursor(sid)
        routing, ceiling = _verified_routing_cache(self.store, sid, ceiling=ceiling)
        route_level = routing.route_level
        if route_level is None:
            raise LedgerIntegrityError("verified routing cache has no route level")
        registration_is_level_two = registration is not None and registration.requirements.level >= 2
        if route_level < 2 and item.kind != "model_selection" and not (registration_is_level_two and ceiling >= 2):
            return item
        if route_level < 2 and item.kind == "model_selection" and (ceiling < 2 or routing.requirements.handler_version is None):
            raise UnsupportedFeatureLevel("model selection requires a level-2 session route with a bound handler")
        if (
            item.kind == "model_selection"
            and registration is not None
            and routing.requirements.handler_version not in {None, registration.handler_version}
        ):
            raise Conflict("registered runtime handler does not match the current route")
        static_requirements = None
        if registration is not None:
            static_requirements = static_session_requirements(
                registration.agent,
                registration.config,
                registration.resolved,
                level=2,
                handler_version=registration.handler_version,
                extra_capabilities=self.capabilities,
            )
        else:
            static_requirements = _session_requirements(self.store, sid)
        session_requirements = _merge_registration_requirements(
            routing.requirements,
            registration,
            static_requirements=static_requirements,
        )
        if item.kind == "model_selection":
            requirements = self._route_model_selection(sid, item, session_requirements, registration)
        else:
            requirements = session_requirements
        return replace(item, feature_level=2, requirements=requirements)

    def _route_model_selection(
        self,
        sid: str,
        item: InboxItem,
        session_requirements: SessionRequirements,
        registration: _RuntimeRegistration | None,
    ) -> SessionRequirements:
        selection = item.payload["selection"]
        resolve_model = self.model_resolver
        if selection is not None and resolve_model is None:
            if registration is None:
                registration, _ = self._runtime_lineage(self.store, sid)
            provider = registration.config.model_provider
            if provider is None:
                raise Conflict("model selection requires the trusted session model provider")

            def resolve_model(value):
                return resolve_model_selection(provider, value)

        handler_version = session_requirements.handler_version
        if handler_version is None and registration is None:
            registration = self._runtimes.get(sid)
        if handler_version is None and registration is not None:
            handler_version = registration.handler_version
        handler_version = handler_version or self.default_handler_version
        if selection is None:
            return SessionRequirements(level=2, handler_version=handler_version, capabilities=())
        assert resolve_model is not None
        try:
            return static_model_selection_requirements(
                selection,
                resolve_model=resolve_model,
                handler_version=handler_version,
            )
        except (ModelError, ValueError):
            # Admission records selection commands before a holder resolves them.
            # Unknown or unauthorized choices therefore need a claimable bound;
            # apply records the stable rejection without changing the selection.
            return SessionRequirements(level=2, handler_version=handler_version, capabilities=())

    def push(
        self,
        sid: str,
        item: InboxItem,
        *,
        actor: PrincipalInput | None = None,
        actor_subject_id: str = "local",
    ):
        with self.store.atomic() as tx:
            receipt = self._push_in_transaction(
                tx,
                sid,
                item,
                actor=actor,
                actor_subject_id=actor_subject_id,
            )
        if receipt.replayed:
            return receipt
        self._worker.wake(sid)
        self._notify_session(sid)
        return receipt

    def _push_in_transaction(
        self,
        tx,
        sid: str,
        item: InboxItem,
        *,
        actor: PrincipalInput | None = None,
        actor_subject_id: str = "local",
        registration: _RuntimeRegistration | None = None,
        ceiling: int | None = None,
    ):
        _reject_public_provider_ingress(item, actor=actor, actor_subject_id=actor_subject_id)
        writer = SessionContentWriter.bind(tx, tenant_id=self.tenant_id, session_id=sid)
        item = self._route_inbox_item(sid, item, registration=registration, ceiling=ceiling)
        item = self._prepare_item(sid, item, writer)
        persisted = False
        actor = actor if actor is not None else item.actor
        if actor is None:
            actor = self._session_principal(sid)
            persisted = True
        elif actor is item.actor:
            persisted = True
        if _is_provider_recovery_actor(actor):
            raise Conflict("provider-recovery actor is reserved for trusted recovery ingress")
        bound_actor, actor_attributes = self._bound_actor(
            actor,
            subject_id=actor_subject_id,
            writer=writer,
            input_id=item.input_id,
            persisted=persisted,
        )
        item = replace(item, actor=bound_actor)
        receipt = tx.push(sid, item)
        link_declared_input_content(writer, item)
        writer.link_input_actor(item.input_id, actor_attributes)
        return receipt

    def read_input_content(
        self,
        sid: str,
        input_id: str,
        ref: ContentRef | dict[str, Any],
        *,
        slot: str = INPUT_CONTENT,
    ) -> Any:
        return read_input_json(
            self.store,
            tenant_id=self.tenant_id,
            session_id=sid,
            input_id=input_id,
            ref=ref,
            slot=slot,
        )

    def read_record_content(
        self,
        sid: str,
        record_seq: int,
        ref: ContentRef | dict[str, Any],
        *,
        slot: str,
    ) -> Any:
        return read_record_json(
            self.store,
            tenant_id=self.tenant_id,
            session_id=sid,
            record_seq=record_seq,
            ref=ref,
            slot=slot,
        )

    def control(
        self,
        sid: str,
        action: str,
        input_id: str,
        *,
        runtime: Runtime | None = None,
        actor: PrincipalInput | None = None,
    ):
        state, _, _ = self.store.read_state(sid)
        tid = None if action in {"archive", "close"} else state.active_turn_id
        registration = self._runtimes.get(sid) or self._registration_for_cursor(sid)
        if registration is not None and not registration.armed:
            raise Conflict("session has an unstarted RunHandle")
        selected_handler = self._handler_version(
            runtime.handler_version
            if runtime is not None
            else registration.handler_version
            if registration is not None
            else self.default_handler_version
        )
        if registration is None:
            if runtime is not None:
                self._register_runtime(sid, runtime, armed=True, handler_version=selected_handler)
            else:
                # Control-only sessions need a worker-routed configuration, not
                # an execution Runtime in the caller. The kernel factory will
                # construct the real Runtime inside its keeper scope if needed.
                from vv_agent.model import ScriptedModelProvider

                config = RunConfig(model_provider=ScriptedModelProvider.new("local", "control", []))
                agent = Agent("control", "Apply session controls.", model="control")
                registration = self._static_registration(
                    sid,
                    agent,
                    config,
                    None,
                    handler_version=selected_handler,
                    armed=True,
                )
                self._runtimes[sid] = registration
        self._worker.start()
        receipt = self.push(
            sid,
            InboxItem(input_id, "control", {"action": action}, tid),
            actor=actor,
            actor_subject_id="local-control",
        )
        self._wait_for_control_applied(sid, input_id)
        return receipt

    def _wait_for_control_applied(self, sid: str, input_id: str) -> None:
        while True:
            head = self.store.read(sid, limit=1).head_seq
            if any(
                record.record._payload["input"]["input_id"] == input_id
                for record in self.store.control_records(sid, through_seq=head)
            ):
                return
            resolved = self._failure_after_progress(self.store, sid, self._errors.get(sid))
            failure = self._terminal_failure(sid, None, any_turn=True) if resolved is not None else None
            if failure is not None and isinstance(failure.error, ProviderResultInvalid):
                state, _, _ = self._read_observed_state(self.store, sid)
                _, _, lease_remaining = self.store.lease_status(sid)
                if lease_remaining > 0 or (
                    state is not None and self._provider_failure_has_consumable_input(self.store, sid, state)
                ):
                    failure = None
            if failure is not None:
                raise failure.error
            if self._closed:
                raise RuntimeError("SessionDriver closed before the control was applied")
            sleep(0.01)

    def messages(self, sid: str) -> list[Message]:
        state, records, _ = self._read_state(sid)
        if records:
            hydration = hydration_for_session_store(self.store, sid)
            raw_records = records
            records = hydrate_records(raw_records, hydration)
            state = fold_hydrated_records(records, raw_records, _fold_factory_for(self.store), hydration)
        return [m for m in project_context(records, state, _fold_factory=_fold_factory_for(self.store)) if m.role != "system"]

    def session_attributes(self, sid: str) -> dict[str, Any]:
        page = self.store.read(sid, limit=1)
        if not page.records:
            raise KeyError(sid)
        attributes = hydrate_record_payload(
            page.records[0],
            hydration_for_session_store(self.store, sid),
        )["attributes"]
        if type(attributes) is not dict:
            raise ValueError("session attributes must be a JSON object")
        return attributes

    def waits(
        self,
        sid: str,
        runtime: Runtime | _ProjectionRuntime | None = None,
        *,
        snapshot=None,
        store: SessionStore | None = None,
    ) -> list[tuple[str, Runtime | _ProjectionRuntime | None, dict[str, Any]]]:
        selected_store = store or self.store
        state, _records, _ = snapshot or self._read_state(sid, selected_store)
        from .interactions import user_interaction

        found = []
        hydration = None

        def hydrated_payload(record_id: str) -> dict[str, Any]:
            nonlocal hydration
            if hydration is None:
                hydration = hydration_for_session_store(selected_store, sid)
            source = next(row for row in _records if row.record.record_id == record_id)
            return hydrate_record_payload(source, hydration)

        tid = state.active_turn_id or (next(reversed(state.turns), None) if state is not None else None)
        if tid is None:
            return found
        turn_wait = state.turns[tid].wait
        if turn_wait:
            wait_payload = hydrated_payload(turn_wait.record_id)
            found.append(
                (
                    sid,
                    runtime,
                    {
                        "interaction_id": wait_payload["interaction_id"],
                        "question": wait_payload["question"],
                        "turn_id": tid,
                        "interaction": user_interaction(sid, tid, state.turns[tid].start._payload["generation"], wait_payload),
                    },
                )
            )
        for (oid, number), wait in state.waits.items():
            if state.operations[oid].turn_id != tid:
                continue
            handle = wait["handle"]
            if handle["kind"] == "child":
                for child in child_handles(handle):
                    child_id = child["session_id"]
                    found.extend(
                        self.waits(
                            child_id,
                            None,
                            store=selected_store,
                        )
                    )
            elif handle["kind"] in {"user", "approval"}:
                public = {}
                parked = next(
                    row
                    for row in _records
                    if row.record.kind == "op_parked"
                    and row.record.turn_id == tid
                    and row.record.operation_id == oid
                    and row.record.attempt == number
                )
                wait_payload = hydrated_payload(parked.record.record_id)
                handle = wait_payload["handle"]
                if handle["kind"] == "user":
                    public["interaction"] = user_interaction(
                        sid,
                        tid,
                        state.turns[tid].start._payload["generation"],
                        wait_payload,
                        operation_id=oid,
                        attempt=number,
                    )
                found.append((sid, runtime, handle | {"operation_id": oid, "attempt": number, "turn_id": tid} | public))
        # Background admissions retain a child handle after their parent tool completes.
        seen = {child_sid for child_sid, _, _ in found}
        for op in state.operations.values():
            if op.turn_id != tid:
                continue
            for admitted in op.attempts.values():
                if admitted.child_handle is None:
                    continue
                for child in child_handles(admitted.child_handle):
                    child_id = child["session_id"]
                    if child["background"] and child_id not in seen:
                        found.extend(
                            self.waits(
                                child_id,
                                None,
                                store=selected_store,
                            )
                        )
                        seen.add(child_id)
        return found

    def answer(
        self,
        sid: str,
        runtime: Runtime | _ProjectionRuntime,
        target,
        answer,
        input_id: str,
        *,
        actor: PrincipalInput | None = None,
    ):
        from vv_agent.interaction import InteractionAnswer, InteractionReplyTarget

        from .interactions import InteractionError, reply_content, validate_reply
        from .tool_result_read import ToolResultReadError, validate_child_graph

        if not isinstance(target, InteractionReplyTarget) or not isinstance(answer, InteractionAnswer):
            raise TypeError("answer requires InteractionReplyTarget and InteractionAnswer")
        try:
            chain, root_turn = validate_child_graph(self.store, sid, target.session_id)
        except ToolResultReadError:
            raise InteractionError("stale_interaction") from None
        previous = self._retained_reply(sid, input_id)
        item = InboxItem(
            input_id,
            "user",
            {"content": reply_content(target, answer)},
            target.turn_id,
            target.generation,
        )
        if previous:
            if previous[0] != target.session_id:
                raise Conflict("input_id reused with different bytes")
            return self.push(
                target.session_id,
                item,
                actor=actor,
                actor_subject_id="interaction-reply",
            )
        captured = {identity: self._read_state(identity) for identity in [sid, *chain]}
        try:
            validate_child_graph(self.store, sid, target.session_id, active=True)
        except ToolResultReadError:
            raise InteractionError("stale_interaction") from None
        root_state, _, _ = captured[sid]
        if target.session_id != sid and root_turn != root_state.active_turn_id:
            raise InteractionError("stale_interaction")
        state, _, _ = captured[target.session_id]
        validate_reply(state, target.session_id, target, answer)
        with self.store.atomic() as tx:
            # Prepare outside row locks; recheck the captured heads before mutation.
            for identity in [sid, *chain]:
                tx.lock_session(identity)
                if self.store.lease_status(identity)[0] != captured[identity][1][-1].seq:
                    raise InteractionError("stale_interaction")
            return self.push(
                target.session_id,
                item,
                actor=actor,
                actor_subject_id="interaction-reply",
            )

    def approve(
        self,
        sid: str,
        runtime: Runtime | _ProjectionRuntime,
        request_id: str,
        decision: ApprovalDecision | str,
        input_id: str,
        *,
        actor: PrincipalInput | None = None,
    ):
        if isinstance(decision, str):
            decision = ApprovalDecision.from_input(decision)
        previous = self._retained_reply(sid, input_id)
        if previous:
            target, item = previous
            if item.kind != "approval_answer":
                raise Conflict("input_id reused with different bytes")
            payload = item.payload | {
                "request_id": request_id,
                "decision": "approve" if decision.action == "allow" else decision.action,
                "reason": decision.reason,
                "metadata": decision.metadata,
            }
            return self.push(
                target,
                replace(item, payload=payload),
                actor=actor,
                actor_subject_id="approval",
            )
        waits = [w for w in self.waits(sid, runtime) if w[2].get("request_id") == request_id]
        if len(waits) != 1:
            raise KeyError(f"Unknown approval request: {request_id}")
        target, _, wait = waits[0]
        state, _, _ = self._read_state(target)
        payload = {k: wait[k] for k in ("operation_id", "attempt", "request_id", "request_digest", "scope")}
        payload.update(
            decision="approve" if decision.action == "allow" else decision.action,
            reason=decision.reason,
            metadata=decision.metadata,
        )
        return self.push(
            target,
            InboxItem(
                input_id,
                "approval_answer",
                payload,
                wait["turn_id"],
                state.turns[wait["turn_id"]].start._payload["generation"],
            ),
            actor=actor,
            actor_subject_id="approval",
        )

    def _retained_reply(self, sid: str, input_id: str):
        state, records, _ = self._read_state(sid)
        if input_id in state.applied_inputs:
            return sid, InboxItem(**state.applied_inputs[input_id]._payload["input"])
        for pending in self.store.peek_inbox(sid):
            if pending.item.input_id == input_id:
                return sid, pending.item
        for stored in records:
            r = stored.record
            if r.kind == "op_parked" and r._payload["handle"]["kind"] == "child":
                for child in child_handles(r._payload["handle"]):
                    previous = self._retained_reply(child["session_id"], input_id)
                    if previous:
                        return previous
        return None


def _resume_turn_handle(
    session_id: str,
    turn_id: str,
    *,
    tenant_id: str = DEFAULT_TENANT_ID,
) -> tuple[RunHandle, RunResult]:
    from .bindings import MissingHostBinding

    if not isinstance(session_id, str) or not isinstance(turn_id, str):
        raise TypeError("resume requires session_id and turn_id strings")
    driver = _DRIVERS.get((tenant_id, session_id))
    if driver is None:
        raise MissingHostBinding("resume requires the original host runtime; use AgentSession for durable sessions")
    state, _, _ = driver.store.read_state(session_id)
    if state.closed or state.archived:
        raise ValueError("session is closed or archived")
    if turn_id not in state.turns or state.active_turn_id not in {None, turn_id}:
        raise ValueError("turn_id does not identify the resumable turn")
    handle = next((h for h in reversed(driver.handles) if h.session_id == session_id and h.run_id == turn_id), None)
    if handle is not None and not handle.done():
        return handle, handle.result()
    if state.turns[turn_id].ended:
        if handle is None:
            registration, _ = driver._runtime_lineage(driver.store, session_id)
            handle = driver.start(
                session_id,
                registration.agent,
                registration.config,
                None,
                task=registration.task,
                handler_version=registration.handler_version,
            )
        return handle, handle.result()
    if state.active_turn_id == turn_id:
        registration, _ = driver._runtime_lineage(driver.store, session_id)
        driver._worker.start()
        driver._wake_session(session_id)
        if handle is None:
            handle = driver.start(
                session_id,
                registration.agent,
                registration.config,
                None,
                task=registration.task,
                handler_version=registration.handler_version,
            )
        return handle, handle.result()
    if not driver.store.peek_inbox(session_id):
        if handle is None:
            raise MissingHostBinding("resume requires the original host runtime")
        return handle, handle.result()
    registration, _ = driver._runtime_lineage(driver.store, session_id)
    handle = driver.start(
        session_id,
        registration.agent,
        registration.config,
        None,
        task=registration.task,
        handler_version=registration.handler_version,
    )
    return handle, handle.result()


def resume_turn(session_id: str, turn_id: str, *, tenant_id: str = DEFAULT_TENANT_ID) -> RunResult:
    _, result = _resume_turn_handle(session_id, turn_id, tenant_id=tenant_id)
    return result
