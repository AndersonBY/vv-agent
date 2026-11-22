"""Internal PostgreSQL reference store, bound to caller-owned psycopg v3 connections."""

from __future__ import annotations

import base64
import json
import secrets
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal, LiteralString, cast

from vv_agent.kernel._allocation_journal import JournalIdentity
from vv_agent.kernel._canonical_json import canonical_json_bytes, validate_sha256
from vv_agent.kernel._content import ContentUnavailable, KeyProvider
from vv_agent.kernel._content_sql import _CONTENT_COLUMNS, _READ_FRAME_OWNER_BATCH, CoreSQLContentStore
from vv_agent.kernel._content_store import ContentCipher
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._sql import SQLSessionTx, SQLStore, _ObjectDeadline, _TenantSQLStore
from vv_agent.kernel._store import Conflict, Lease, LeaseLost, ObjectPublicationBusy, SessionStore

if TYPE_CHECKING:
    from psycopg import Connection

    from vv_agent.kernel._external_content import _PostgresContentJournal

CLOCK = "floor(extract(epoch FROM clock_timestamp()) * 1000)::bigint"
STORAGE_MAJOR = "S1"
SCHEMA_FINGERPRINT_VERSION = "S1-postgres-schema-v3"
# Update this digest with intentional DDL or fingerprint-shape changes and contract review.
SCHEMA_FINGERPRINT = "e19ae9287545c500edc810238875b1c008d8f8d4e9f4c48aa836766316aabc38"
RLS_ISOLATION = "explicit-tenant-credentials"
RLS_INSTALL_REASON = "The base DDL is unisolated. Runtime tenant access requires install_tenant_rls() and tenant_standalone()."
RLS_SHARED_GATEWAY_REASON = (
    "Shared gateway context prevents accidental query leakage but does not provide tenant credential isolation."
)
# The schema accepts stable ASCII identifier kinds so newer S1 feature-level producers can coexist with
# older workers. The active codec and producer remain strictly closed to their current PAYLOADS.

# The owned-table set is also the schema-completeness boundary for initialization.
SCHEMA_TABLES = frozenset(
    {
        "sk_settings",
        "sk_session",
        "sk_record",
        "sk_inbox",
        "sk_consumer",
        "sk_commit",
        "sk_checkpoint",
        "sk_session_key",
        "sk_content",
        "sk_content_link",
        "sk_tombstone",
    }
)
SESSION_TABLES = frozenset(SCHEMA_TABLES - {"sk_settings"})

_RLS_ROLE_MARKER_PREFIX = "vv-agent-rls-v1"


@dataclass(frozen=True)
class RLSInstallation:
    """Installed PostgreSQL role boundary and its exact isolation claim."""

    mode: Literal["tenant", "worker", "shared-gateway"]
    role: str
    tenant_id: str | None
    credential_isolation: bool


RuntimeMode = Literal["tenant", "worker", "shared-gateway", "unbound-internal"]


@dataclass(frozen=True)
class RuntimeBinding:
    """The credential and trust mode attached to one PostgreSQL store."""

    mode: RuntimeMode
    tenant_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"tenant", "worker", "shared-gateway", "unbound-internal"}:
            raise ValueError("runtime binding mode is invalid")
        if self.mode == "tenant":
            _validate_name(self.tenant_id or "", "tenant_id")
        elif self.tenant_id is not None:
            raise ValueError("only tenant runtime bindings may carry a tenant_id")

    @property
    def credential_isolation(self) -> bool:
        return self.mode == "tenant"


UNBOUND_INTERNAL_BINDING = RuntimeBinding("unbound-internal")


# Tenant RLS is deliberately absent here: safe installation requires generated per-tenant LOGIN
# roles and grants, rather than a shared role or a GUC-only policy that can forge tenant context.
DDL = """
CREATE TABLE sk_settings (
    singleton integer PRIMARY KEY CHECK (singleton = 1),
    storage_major text NOT NULL CHECK (storage_major = 'S1'),
    write_level integer NOT NULL CHECK (write_level >= 1),
    database_id text NOT NULL CHECK (database_id ~ '^[0-9a-f]{64}$'),
    object_journal_binding text CHECK (object_journal_binding ~ '^[0-9a-f]{64}$')
);
CREATE TABLE sk_session (
    tenant_id text NOT NULL, session_id text NOT NULL,
    schema_version text NOT NULL CHECK (schema_version = 'S1'),
    parent_session_id text, parent_operation_id text,
    head_seq bigint NOT NULL DEFAULT 0 CHECK (head_seq >= 0),
    inbox_seq bigint NOT NULL DEFAULT 0 CHECK (inbox_seq >= 0),
    lease_epoch bigint NOT NULL DEFAULT 0 CHECK (lease_epoch >= 0),
    lease_owner text, lease_until_ms bigint CHECK (lease_until_ms IS NULL OR lease_until_ms >= 0),
    phase text NOT NULL CHECK (phase IN ('idle','active','parked','suspended','closed')),
    active_turn_id text, next_drive_ms bigint CHECK (next_drive_ms IS NULL OR next_drive_ms >= 0),
    drive_retry_at_ms bigint NOT NULL DEFAULT 0 CHECK (drive_retry_at_ms >= 0),
    terminal_seq bigint NOT NULL DEFAULT 0 CHECK (terminal_seq >= 0 AND terminal_seq <= head_seq),
    level integer NOT NULL CHECK (level >= 1), handler_version text,
    required_capabilities bytea NOT NULL DEFAULT decode('5b5d','hex'),
    deletion_state text NOT NULL CHECK (deletion_state IN ('live','forgetting','forgotten','purging')),
    created_ms bigint NOT NULL CHECK (created_ms >= 0),
    PRIMARY KEY (tenant_id,session_id),
    CHECK ((lease_owner IS NULL) = (lease_until_ms IS NULL)),
    CHECK ((parent_session_id IS NULL) = (parent_operation_id IS NULL)),
    FOREIGN KEY (tenant_id,parent_session_id) REFERENCES sk_session(tenant_id,session_id)
);
CREATE INDEX sk_session_due ON sk_session(level,next_drive_ms,tenant_id,session_id)
    WHERE next_drive_ms IS NOT NULL AND deletion_state = 'live';
CREATE INDEX sk_session_parent ON sk_session(tenant_id,parent_session_id) WHERE parent_session_id IS NOT NULL;
CREATE INDEX sk_session_deletion ON sk_session(tenant_id,deletion_state,session_id);
CREATE TABLE sk_record (
    tenant_id text NOT NULL, session_id text NOT NULL, seq bigint NOT NULL CHECK (seq > 0),
    record_id text NOT NULL, schema_version text NOT NULL CHECK (schema_version = 'S1'),
    kind text NOT NULL CHECK (char_length(kind) BETWEEN 1 AND 64
        AND kind ~ '^[a-z]' AND kind !~ '[^a-z0-9_]'),
    turn_id text, operation_id text, attempt integer CHECK (attempt > 0), body bytea NOT NULL,
    digest text NOT NULL CHECK (digest ~ '^[0-9a-f]{64}$'),
    chain_digest text NOT NULL CHECK (chain_digest ~ '^[0-9a-f]{64}$'), commit_id text NOT NULL,
    commit_digest text NOT NULL CHECK (commit_digest ~ '^[0-9a-f]{64}$'),
    commit_position integer NOT NULL CHECK (commit_position >= 0), writer_epoch bigint NOT NULL CHECK (writer_epoch >= 0),
    created_ms bigint NOT NULL CHECK (created_ms >= 0),
    PRIMARY KEY (tenant_id,session_id,seq), UNIQUE (tenant_id,session_id,record_id),
    UNIQUE (tenant_id,session_id,commit_id,commit_position),
    FOREIGN KEY (tenant_id,session_id) REFERENCES sk_session(tenant_id,session_id)
);
CREATE INDEX sk_record_operation ON sk_record(tenant_id,session_id,operation_id,attempt,seq) WHERE operation_id IS NOT NULL;
CREATE INDEX sk_record_turn ON sk_record(tenant_id,session_id,turn_id,seq);
CREATE TABLE sk_inbox (
    tenant_id text NOT NULL, session_id text NOT NULL, input_id text NOT NULL,
    input_seq bigint NOT NULL CHECK (input_seq > 0),
    kind text NOT NULL CHECK (char_length(kind) BETWEEN 1 AND 64
        AND kind ~ '^[a-z]' AND kind !~ '[^a-z0-9_]'),
    body bytea NOT NULL, digest text NOT NULL CHECK (digest ~ '^[0-9a-f]{64}$'),
    admission_key_id text, admission_signature bytea,
    admission_signed_ms bigint CHECK (admission_signed_ms IS NULL OR admission_signed_ms >= 0),
    available_ms bigint NOT NULL CHECK (available_ms >= 0), received_ms bigint NOT NULL CHECK (received_ms >= 0),
    consumed_seq bigint CHECK (consumed_seq IS NULL OR consumed_seq > 0),
    routing_level integer NOT NULL DEFAULT 1 CHECK (routing_level >= 1), routing_handler_version text,
    routing_required_capabilities bytea NOT NULL DEFAULT decode('5b5d','hex'),
    PRIMARY KEY (tenant_id,session_id,input_id), UNIQUE (tenant_id,session_id,input_seq),
    CHECK ((admission_key_id IS NULL) = (admission_signature IS NULL)
        AND (admission_key_id IS NULL) = (admission_signed_ms IS NULL)),
    FOREIGN KEY (tenant_id,session_id) REFERENCES sk_session(tenant_id,session_id),
    FOREIGN KEY (tenant_id,session_id,consumed_seq) REFERENCES sk_record(tenant_id,session_id,seq)
);
CREATE INDEX sk_inbox_ready ON sk_inbox(tenant_id,session_id,available_ms,input_seq) WHERE consumed_seq IS NULL;
CREATE TABLE sk_consumer (
    tenant_id text NOT NULL, session_id text NOT NULL, consumer text NOT NULL,
    last_seq bigint NOT NULL DEFAULT 0 CHECK (last_seq >= 0),
    project_retry_at_ms bigint NOT NULL DEFAULT 0 CHECK (project_retry_at_ms >= 0),
    PRIMARY KEY (tenant_id,session_id,consumer),
    FOREIGN KEY (tenant_id,session_id) REFERENCES sk_session(tenant_id,session_id)
);
CREATE INDEX sk_consumer_retry ON sk_consumer(tenant_id,project_retry_at_ms,last_seq,session_id,consumer);
-- Commit metadata is necessary for empty/fully overlapping commits and their original head_seq.
-- It owns no execution state; record bodies remain solely in sk_record.
CREATE TABLE sk_commit (
    tenant_id text NOT NULL, session_id text NOT NULL, commit_id text NOT NULL,
    body bytea NOT NULL, digest text NOT NULL CHECK (digest ~ '^[0-9a-f]{64}$'),
    record_sequences bytea NOT NULL, head_seq bigint NOT NULL CHECK (head_seq >= 0),
    created_ms bigint NOT NULL CHECK (created_ms >= 0),
    PRIMARY KEY (tenant_id,session_id,commit_id),
    FOREIGN KEY (tenant_id,session_id) REFERENCES sk_session(tenant_id,session_id)
);
CREATE TABLE sk_checkpoint (
    tenant_id text NOT NULL, session_id text NOT NULL, seq bigint NOT NULL CHECK (seq > 0),
    chain_digest text NOT NULL CHECK (chain_digest ~ '^[0-9a-f]{64}$'),
    key_id text NOT NULL, signature bytea NOT NULL, signed_ms bigint NOT NULL CHECK (signed_ms >= 0),
    PRIMARY KEY (tenant_id,session_id,seq),
    FOREIGN KEY (tenant_id,session_id,seq) REFERENCES sk_record(tenant_id,session_id,seq)
);
CREATE TABLE sk_session_key (
    tenant_id text NOT NULL, session_id text NOT NULL, key_id text NOT NULL,
    provider_id text NOT NULL, wrapped_key bytea NOT NULL,
    state text NOT NULL CHECK (state IN ('active','destroying')), created_ms bigint NOT NULL CHECK (created_ms >= 0),
    PRIMARY KEY (tenant_id,session_id), UNIQUE (tenant_id,key_id),
    FOREIGN KEY (tenant_id,session_id) REFERENCES sk_session(tenant_id,session_id)
);
CREATE TABLE sk_content (
    tenant_id text NOT NULL, session_id text NOT NULL, content_id text NOT NULL,
    key_id text NOT NULL, mac text NOT NULL CHECK (mac ~ '^[0-9a-f]{64}$'),
    size bigint NOT NULL CHECK (size >= 0), labels bytea NOT NULL, nonce bytea NOT NULL CHECK (octet_length(nonce) = 12),
    created_ms bigint NOT NULL CHECK (created_ms >= 0),
    state text NOT NULL CHECK (state IN ('staged','live','gc_pending')),
    pin_until_ms bigint CHECK (pin_until_ms IS NULL OR pin_until_ms >= 0),
    ciphertext bytea, object_locator bytea,
    PRIMARY KEY (tenant_id,session_id,content_id),
    UNIQUE (tenant_id,session_id,key_id,mac,labels),
    CHECK ((ciphertext IS NOT NULL) <> (object_locator IS NOT NULL))
);
CREATE INDEX sk_content_gc ON sk_content(state,pin_until_ms,created_ms,tenant_id,session_id,content_id)
    WHERE state IN ('staged','gc_pending');
CREATE TABLE sk_content_link (
    tenant_id text NOT NULL, session_id text NOT NULL, content_id text NOT NULL,
    record_seq bigint, input_id text, slot text NOT NULL,
    CHECK ((record_seq IS NOT NULL)::int + (input_id IS NOT NULL)::int = 1),
    FOREIGN KEY (tenant_id,session_id) REFERENCES sk_session(tenant_id,session_id),
    FOREIGN KEY (tenant_id,session_id,record_seq) REFERENCES sk_record(tenant_id,session_id,seq),
    FOREIGN KEY (tenant_id,session_id,input_id) REFERENCES sk_inbox(tenant_id,session_id,input_id),
    FOREIGN KEY (tenant_id,session_id,content_id) REFERENCES sk_content(tenant_id,session_id,content_id)
);
CREATE UNIQUE INDEX sk_content_link_record_slot
    ON sk_content_link(tenant_id,session_id,record_seq,slot) WHERE record_seq IS NOT NULL;
CREATE UNIQUE INDEX sk_content_link_input_slot
    ON sk_content_link(tenant_id,session_id,input_id,slot) WHERE input_id IS NOT NULL;
CREATE INDEX sk_content_link_content ON sk_content_link(tenant_id,session_id,content_id);
CREATE TABLE sk_tombstone (
    tenant_id text NOT NULL, session_id text NOT NULL,
    final_seq bigint NOT NULL CHECK (final_seq >= 0),
    chain_digest text NOT NULL CHECK (chain_digest ~ '^[0-9a-f]{64}$'),
    deleted_ms bigint NOT NULL CHECK (deleted_ms >= 0), actor bytea NOT NULL,
    key_id text NOT NULL, signature bytea NOT NULL,
    state text NOT NULL CHECK (state IN ('pending','complete')),
    scope_root_session_id text NOT NULL,
    PRIMARY KEY (tenant_id,session_id)
);
CREATE INDEX sk_tombstone_state ON sk_tombstone(state,tenant_id,session_id);
"""


def _connection_conninfo(connection: Connection[Any]) -> str:
    from psycopg.conninfo import make_conninfo

    info = connection.info
    # ConnectionInfo.dsn deliberately omits the password.
    return make_conninfo(info.dsn, password=info.password) if info.password else info.dsn


def _role_marker(mode: str, tenant_id: str | None = None) -> str:
    suffix = ""
    if tenant_id is not None:
        suffix = ":" + base64.urlsafe_b64encode(tenant_id.encode("utf-8")).decode("ascii")
    return f"{_RLS_ROLE_MARKER_PREFIX}:{mode}{suffix}"


def _policy_name(role: str) -> str:
    return "vv_rls_" + sha256(role.encode("utf-8")).hexdigest()[:24]


def _validate_name(value: str, label: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not allow_empty and not value) or "\x00" in value:
        raise ValueError(f"{label} must be nonempty text without NUL")


def _query_rows(
    connection: Connection[Any], query: str, params: tuple[Any, ...] = (), *, binary: bool = False
) -> list[tuple[Any, ...]]:
    from psycopg.rows import tuple_row

    with connection.cursor(row_factory=tuple_row) as cursor:
        cursor.execute(cast(LiteralString, query), params, binary=binary)
        return cursor.fetchall() if cursor.description is not None else []


@contextmanager
def _read_only_probe(connection: Connection[Any]) -> Iterator[None]:
    """Run catalog probes without leaking a transaction on a caller-owned connection."""

    from psycopg.pq import TransactionStatus

    status = connection.info.transaction_status
    if status == TransactionStatus.INTRANS:
        yield
        return
    if status != TransactionStatus.IDLE:
        raise RuntimeError(f"PostgreSQL connection is not available for a read-only probe: {status.name}")
    if connection.autocommit:
        yield
        return
    with connection.transaction():
        yield


def _probe_rows(
    connection: Connection[Any], query: str, params: tuple[Any, ...] = (), *, binary: bool = False
) -> list[tuple[Any, ...]]:
    with _read_only_probe(connection):
        return _query_rows(connection, query, params, binary=binary)


def _bind_tenant_context(connection: Connection[Any], tenant_id: str | None) -> None:
    """Fill an unbound tenant GUC on this transaction without replacing caller state."""

    if tenant_id is None:
        return
    from psycopg.pq import TransactionStatus

    if connection.info.transaction_status != TransactionStatus.INTRANS:
        raise RuntimeError("tenant context binding requires an open transaction")
    current = _query_rows(connection, "SELECT current_setting('vv.tenant_id',true)")
    if current == [(None,)] or current == [("",)]:
        connection.execute("SELECT set_config('vv.tenant_id',%s,true)", (tenant_id,))


def _schema_relations(connection: Connection[Any]) -> list[tuple[Any, ...]]:
    with _read_only_probe(connection):
        return _probe_rows(
            connection,
            "SELECT c.relname,c.relkind,c.relpersistence FROM pg_catalog.pg_class c "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relkind IN ('r','p','v','m','f') "
            "AND c.relname LIKE 'sk\\_%%' ESCAPE '\\' ORDER BY c.relname",
        )


def _schema_tables(connection: Connection[Any]) -> set[str]:
    return {row[0] for row in _schema_relations(connection)}


def _rls_is_enabled(connection: Connection[Any]) -> bool:
    with _read_only_probe(connection):
        rows = _probe_rows(
            connection,
            "SELECT EXISTS(SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relkind IN ('r','p') "
            "AND c.relname LIKE 'sk\\_%%' ESCAPE '\\' "
            "AND (c.relrowsecurity OR c.relforcerowsecurity)) "
            "OR EXISTS(SELECT 1 FROM pg_policy p JOIN pg_class c ON c.oid=p.polrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=current_schema() "
            "AND c.relname LIKE 'sk\\_%%' ESCAPE '\\')",
        )
        return bool(rows[0][0])


def _catalog_manifest(
    connection: Connection[Any], relations: list[tuple[Any, ...]] | None = None
) -> dict[str, list[tuple[Any, ...]]]:
    """Return normalized catalog rows without OIDs, database names or scan order."""

    with _read_only_probe(connection):
        if relations is None:
            relations = _schema_relations(connection)
        tables = [row for row in relations if row[1] in {"r", "p"}]
        columns = _probe_rows(
            connection,
            "SELECT c.relname,a.attnum,a.attname,format_type(a.atttypid,a.atttypmod),"
            "a.attnotnull,pg_get_expr(d.adbin,d.adrelid,true),a.attidentity,a.attgenerated "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped "
            "LEFT JOIN pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum "
            "WHERE n.nspname=current_schema() AND c.relkind IN ('r','p') "
            "AND c.relname LIKE 'sk\\_%%' ESCAPE '\\' ORDER BY c.relname,a.attnum",
        )
        constraints = _probe_rows(
            connection,
            "SELECT c.relname,con.conname,con.contype,pg_get_constraintdef(con.oid,true),"
            "con.condeferrable,con.condeferred,con.convalidated "
            "FROM pg_constraint con JOIN pg_class c ON c.oid=con.conrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname LIKE 'sk\\_%%' ESCAPE '\\' "
            "ORDER BY c.relname,con.conname",
        )
        indexes = _probe_rows(
            connection,
            "SELECT tbl.relname,idx.relname,i.indisunique,i.indisprimary,i.indisvalid,i.indisready,"
            "i.indnkeyatts,i.indnatts,am.amname,pos,pg_get_indexdef(i.indexrelid,pos,true),"
            "pg_get_expr(i.indpred,i.indrelid,true) "
            "FROM pg_index i JOIN pg_class idx ON idx.oid=i.indexrelid "
            "JOIN pg_class tbl ON tbl.oid=i.indrelid JOIN pg_namespace n ON n.oid=tbl.relnamespace "
            "JOIN pg_am am ON am.oid=idx.relam "
            "CROSS JOIN LATERAL generate_series(1,i.indnatts) pos "
            "WHERE n.nspname=current_schema() AND tbl.relname LIKE 'sk\\_%%' ESCAPE '\\' "
            "ORDER BY tbl.relname,idx.relname,pos",
        )
        triggers = _probe_rows(
            connection,
            "SELECT tbl.relname,t.tgname,t.tgenabled,t.tgtype,pg_get_triggerdef(t.oid,true),"
            "pn.nspname,p.proname FROM pg_trigger t JOIN pg_class tbl ON tbl.oid=t.tgrelid "
            "JOIN pg_namespace n ON n.oid=tbl.relnamespace JOIN pg_proc p ON p.oid=t.tgfoid "
            "JOIN pg_namespace pn ON pn.oid=p.pronamespace "
            "WHERE n.nspname=current_schema() AND tbl.relname LIKE 'sk\\_%%' ESCAPE '\\' "
            "AND NOT t.tgisinternal ORDER BY tbl.relname,t.tgname",
        )
        return {
            "tables": tables,
            "columns": columns,
            "constraints": constraints,
            "indexes": indexes,
            "triggers": triggers,
        }


def _catalog_fingerprint(catalog: dict[str, list[tuple[Any, ...]]]) -> str:
    # PostgreSQL 18 exposes column NOT NULL as validated pg_constraint rows;
    # columns.attnotnull already carries that semantic on every supported version.
    catalog["constraints"] = [row for row in catalog["constraints"] if row[2] != "n" or not row[6]]
    material = {"version": SCHEMA_FINGERPRINT_VERSION, "catalog": catalog}
    encoded = json.dumps(material, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _schema_fingerprint(connection: Connection[Any]) -> str:
    with _read_only_probe(connection):
        return _catalog_fingerprint(_catalog_manifest(connection))


def _role_binding(marker: str | None) -> RuntimeBinding | None:
    if marker is None or not marker.startswith(_RLS_ROLE_MARKER_PREFIX + ":"):
        return None
    payload = marker.removeprefix(_RLS_ROLE_MARKER_PREFIX + ":")
    mode, separator, encoded_tenant = payload.partition(":")
    if mode == "tenant" and separator and encoded_tenant:
        try:
            tenant_id = base64.urlsafe_b64decode(encoded_tenant.encode("ascii")).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return None
        if _role_marker("tenant", tenant_id) == marker:
            return RuntimeBinding("tenant", tenant_id)
    if mode in {"worker", "shared-gateway"} and not separator:
        return RuntimeBinding(cast(RuntimeMode, mode))
    return None


def _runtime_role_state(connection: Connection[Any], role: str) -> tuple[Any, ...] | None:
    with _read_only_probe(connection):
        rows = _probe_rows(
            connection,
            "SELECT r.rolcanlogin,r.rolsuper,r.rolbypassrls,r.rolcreaterole,r.rolcreatedb,"
            "r.rolreplication,r.rolinherit,shobj_description(r.oid,'pg_authid'),"
            "EXISTS(SELECT 1 FROM pg_auth_members m WHERE m.member=r.oid),"
            "EXISTS(SELECT 1 FROM pg_class c WHERE c.relowner=r.oid AND c.relname=ANY(%s)),"
            "EXISTS(SELECT 1 FROM pg_namespace n WHERE n.nspowner=r.oid AND n.nspname=current_schema()),"
            "EXISTS(SELECT 1 FROM pg_database d WHERE d.datname=current_database() AND d.datdba=r.oid),"
            "has_schema_privilege(r.oid,current_schema(),'CREATE') "
            "FROM pg_roles r WHERE r.rolname=%s",
            (sorted(SCHEMA_TABLES), role),
        )
        return rows[0] if len(rows) == 1 else None


def _verified_role_binding(connection: Connection[Any], expected: RuntimeBinding) -> RuntimeBinding:
    with _read_only_probe(connection):
        identity = _probe_rows(connection, "SELECT session_user,current_user")
        if len(identity) != 1 or identity[0][0] != identity[0][1]:
            raise ValueError("runtime requires session_user and current_user to match")
        role = identity[0][0]
        row = _runtime_role_state(connection, role)
        if row is None:
            raise ValueError("runtime role is unavailable")
        if not row[0] or any(row[1:7]) or row[8] or row[9] or row[10] or row[11] or row[12]:
            raise ValueError("runtime role must be a dedicated login without elevated memberships or DDL rights")
        actual = _role_binding(row[7])
        if actual is None or actual != expected:
            raise ValueError(f"runtime role does not carry the expected {expected.mode} binding")
        return actual


def _read_runtime_binding(connection: Connection[Any], *, allow_unbound_internal: bool = True) -> RuntimeBinding:
    with _read_only_probe(connection):
        rows = _probe_rows(
            connection,
            "SELECT shobj_description(r.oid,'pg_authid') FROM pg_roles r WHERE r.rolname=current_user",
        )
        marker = rows[0][0] if len(rows) == 1 else None
        if marker is not None and (marker == _RLS_ROLE_MARKER_PREFIX or marker.startswith(_RLS_ROLE_MARKER_PREFIX + ":")):
            binding = _role_binding(marker)
            if binding is None:
                raise ValueError("runtime role has an unrecognized vv-agent RLS marker")
            return _verified_role_binding(connection, binding)
        if not allow_unbound_internal and _rls_is_enabled(connection):
            raise ValueError("unbound PostgreSQL roles cannot open an RLS-enabled runtime")
        return UNBOUND_INTERNAL_BINDING


def _strip_pg_casts(expression: str) -> str:
    """Remove PostgreSQL's display-only casts while preserving quoted literals."""

    result: list[str] = []
    index = 0
    quoted = False
    while index < len(expression):
        char = expression[index]
        if char == "'":
            result.append(char)
            if quoted and index + 1 < len(expression) and expression[index + 1] == "'":
                result.append("'")
                index += 2
                continue
            quoted = not quoted
            index += 1
            continue
        if not quoted and expression.startswith("::", index):
            index += 2
            while index < len(expression) and (expression[index].isalnum() or expression[index] in {"_", '"'}):
                index += 1
            continue
        result.append(char)
        index += 1
    return "".join(result)


def _expected_policy_predicate(connection: Connection[Any], binding: RuntimeBinding, role: str) -> str:
    from psycopg import sql

    role_literal = sql.Literal(role).as_string(connection)
    if binding.mode == "tenant":
        assert binding.tenant_id is not None
        tenant_literal = sql.Literal(binding.tenant_id).as_string(connection)
        return (
            f"SESSION_USER = {role_literal} AND tenant_id = {tenant_literal} "
            f"AND current_setting('vv.tenant_id', true) = {tenant_literal}"
        )
    if binding.mode == "worker":
        return f"SESSION_USER = {role_literal}"
    if binding.mode == "shared-gateway":
        return (
            f"SESSION_USER = {role_literal} AND current_setting('vv.tenant_id', true) IS NOT NULL "
            "AND tenant_id = current_setting('vv.tenant_id', true)"
        )
    raise ValueError("unbound internal stores do not install or verify RLS policies")


def _rls_state_is_current(connection: Connection[Any]) -> bool:
    with _read_only_probe(connection):
        state = _probe_rows(
            connection,
            "SELECT c.relname,c.relrowsecurity,c.relforcerowsecurity FROM pg_class c "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relkind IN ('r','p') "
            "AND c.relname LIKE 'sk\\_%%' ESCAPE '\\' ORDER BY c.relname",
        )
        by_table = {row[0]: (bool(row[1]), bool(row[2])) for row in state}
        settings_state = by_table.pop("sk_settings", None)
        if settings_state != (False, False) or set(by_table) != SESSION_TABLES:
            return False
        flags = set(by_table.values())
        if flags == {(False, False)}:
            policies = _probe_rows(
                connection,
                "SELECT count(*) FROM pg_policy p JOIN pg_class c ON c.oid=p.polrelid "
                "JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=current_schema() AND c.relname=ANY(%s)",
                (sorted(SESSION_TABLES),),
            )
            return policies == [(0,)]
        if flags != {(True, True)}:
            return False

        rows = _probe_rows(
            connection,
            "SELECT c.relname,p.polname,p.polcmd,p.polpermissive,"
            "pg_get_expr(p.polqual,p.polrelid,true),pg_get_expr(p.polwithcheck,p.polrelid,true),"
            "ARRAY(SELECT r.rolname FROM pg_roles r WHERE r.oid=ANY(p.polroles) ORDER BY r.rolname) "
            "FROM pg_policy p JOIN pg_class c ON c.oid=p.polrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname=ANY(%s) ORDER BY c.relname,p.polname",
            (sorted(SESSION_TABLES),),
        )
        if not rows:
            return False
        policy_roles = {role for row in rows for role in row[6]}
        markers = dict(
            _probe_rows(
                connection,
                "SELECT rolname,shobj_description(oid,'pg_authid') FROM pg_roles WHERE rolname=ANY(%s)",
                (sorted(policy_roles),),
            )
        )
        by_table_policies: dict[str, list[tuple[str, RuntimeBinding, str, str]]] = {table: [] for table in SESSION_TABLES}
        for table, name, command, permissive, using, check, roles in rows:
            if command != "*" or not permissive or not isinstance(roles, list) or len(roles) != 1:
                return False
            role = roles[0]
            binding = _role_binding(markers.get(role))
            if binding is None or name != _policy_name(role) or using is None or check is None:
                return False
            expected = _expected_policy_predicate(connection, binding, role)
            if _strip_pg_casts(using) != expected or _strip_pg_casts(check) != expected:
                return False
            by_table_policies[table].append((role, binding, using, check))
        expected_roles = {policy[0] for policy in by_table_policies[next(iter(SESSION_TABLES))]}
        return bool(expected_roles) and all(
            {policy[0] for policy in policies} == expected_roles for policies in by_table_policies.values()
        )


def _schema_status(connection: Connection[Any]) -> Literal["missing", "current", "invalid"]:
    with _read_only_probe(connection):
        relations = _schema_relations(connection)
        tables = {row[0] for row in relations}
        if not tables:
            return "missing"
        if tables != SCHEMA_TABLES:
            return "invalid"
        if _catalog_fingerprint(_catalog_manifest(connection, relations)) != SCHEMA_FINGERPRINT:
            return "invalid"
        rows = _probe_rows(connection, "SELECT singleton,storage_major,write_level FROM sk_settings")
        if not (
            len(rows) == 1
            and rows[0][0] == 1
            and rows[0][1] == STORAGE_MAJOR
            and type(rows[0][2]) is int
            and rows[0][2] >= 1
            and _rls_state_is_current(connection)
        ):
            return "invalid"
        return "current"


def _schema_is_current(connection: Connection[Any]) -> bool:
    return _schema_status(connection) == "current"


class PostgresStore(SQLStore):
    """A serialized connection; atomic() nests as a savepoint in host transactions."""

    # Session keys never change. This still serializes writers while allowing
    # child inserts to retain their parent foreign-key KEY SHARE locks.
    lock_clause = " FOR NO KEY UPDATE"
    _worker_handler_filter_sql = (
        "(s.handler_version IS NULL OR s.handler_version IN (SELECT jsonb_array_elements_text(%s::jsonb)))"
    )
    _worker_capability_filter_sql = (
        "NOT EXISTS (SELECT 1 FROM jsonb_array_elements_text("
        "convert_from(s.required_capabilities,'UTF8')::jsonb) AS required(value) "
        "WHERE required.value NOT IN (SELECT jsonb_array_elements_text(%s::jsonb)))"
    )
    clock_sql = CLOCK

    def __init__(
        self,
        connection: Connection[Any],
        *,
        fold_factory: Callable[[], Fold],
        runtime_binding: RuntimeBinding | None = None,
    ):
        super().__init__(fold_factory=fold_factory)
        self.connection = connection
        self._connection_lock = RLock()
        binding = runtime_binding or _read_runtime_binding(connection)
        if binding.mode == "tenant":
            assert binding.tenant_id is not None
            self.verify_tenant_connection(connection, binding.tenant_id)
            self._tenant_id = binding.tenant_id
        elif binding.mode in {"worker", "shared-gateway"}:
            _verified_role_binding(connection, binding)
            self._verify_bound_policy(connection, binding)
        self._runtime_binding = binding
        if _schema_status(connection) == "invalid":
            raise ValueError("unsupported PostgreSQL session schema version")

    @property
    def runtime_binding(self) -> RuntimeBinding:
        return getattr(self, "_runtime_binding", UNBOUND_INTERNAL_BINDING)

    @property
    def runtime_mode(self) -> RuntimeMode:
        return self.runtime_binding.mode

    @property
    def credential_isolation(self) -> bool:
        """Whether this store is backed by a verified per-tenant login."""

        return self.runtime_binding.credential_isolation

    @property
    def credential_tenant_id(self) -> str | None:
        """The tenant authenticated by the login, if this is a tenant binding."""

        return self.runtime_binding.tenant_id if self.credential_isolation else None

    @property
    def tenant_context_id(self) -> str | None:
        """The tenant bound by this exact runtime view for transaction-local RLS."""

        credential_tenant = self.credential_tenant_id
        if credential_tenant is not None:
            return credential_tenant
        if self._root_store is not None and self.runtime_mode in {"worker", "shared-gateway"}:
            return self.tenant_id
        return None

    @property
    def claims_all_tenants(self) -> bool:
        return self._root_store is None and (self.runtime_mode == "worker" or self._all_tenants)

    def _runnable_lock_clause(self) -> str:
        return " FOR NO KEY UPDATE OF s SKIP LOCKED" if self.runtime_mode == "worker" else ""

    def _require_claim_capable(self) -> None:
        # Trusted unbound-internal connections retain local schema/admin test
        # compatibility; production tenant and shared-gateway bindings cannot claim.
        if self.runtime_mode not in {"worker", "unbound-internal"}:
            raise PermissionError("PostgreSQL drive leases can only be claimed by a worker binding")

    def heartbeat_store(self):
        return self.heartbeat_store_factory()()

    def drive_store_factory(self) -> Callable[[], AbstractContextManager[SessionStore]]:
        """Capture a normal independent drive connection opener for this store."""

        conninfo = _connection_conninfo(self.connection)
        binding = self.runtime_binding
        tenant_id = self.tenant_id
        fold_factory = self.fold_factory
        clock_sql = self.clock_sql
        supports = self.worker_supports
        reader_level = self.reader_level
        all_tenants = self.claims_all_tenants
        configured, allow_unsigned, signer = self._input_admission_policy()
        store_type = type(self)

        @contextmanager
        def open_store() -> Iterator[SessionStore]:
            if binding.mode == "tenant":
                assert binding.tenant_id is not None
                manager = store_type.tenant_standalone(
                    conninfo,
                    tenant_id=binding.tenant_id,
                    fold_factory=fold_factory,
                )
            elif binding.mode == "worker":
                manager = store_type.worker_standalone(conninfo, fold_factory=fold_factory)
            elif binding.mode == "shared-gateway":
                manager = store_type.shared_gateway_standalone(conninfo, fold_factory=fold_factory)
            else:
                manager = store_type.standalone(conninfo, fold_factory=fold_factory)
            with manager as store:
                store.clock_sql = clock_sql
                store.configure_reader(level=reader_level)
                store.configure_worker(
                    level=supports.max_level,
                    handler_versions=supports.handler_versions,
                    capabilities=supports.capabilities,
                    all_tenants=all_tenants,
                )
                if configured:
                    store.configure_input_admission(signer, allow_unsigned=allow_unsigned)
                yield store if all_tenants else store.for_tenant(tenant_id)

        return open_store

    def heartbeat_store_factory(self) -> Callable[[], AbstractContextManager[SessionStore]]:
        """Capture connection-independent renewal configuration on the caller thread."""

        from psycopg.conninfo import conninfo_to_dict, make_conninfo

        conninfo = _connection_conninfo(self.connection)
        options = conninfo_to_dict(conninfo).get("options", "")
        heartbeat_conninfo = make_conninfo(
            conninfo,
            connect_timeout=3,
            options=f"{options} -c lock_timeout=250ms -c statement_timeout=1000ms",
        )
        binding = self.runtime_binding
        fold_factory = self.fold_factory
        clock_sql = self.clock_sql
        supports = self.worker_supports
        reader_level = self.reader_level
        configured, allow_unsigned, signer = self._input_admission_policy()
        store_type = type(self)

        @contextmanager
        def open_store() -> Iterator[SessionStore]:
            if binding.mode == "tenant":
                assert binding.tenant_id is not None
                manager = store_type.tenant_standalone(
                    heartbeat_conninfo,
                    tenant_id=binding.tenant_id,
                    fold_factory=fold_factory,
                )
            elif binding.mode == "worker":
                manager = store_type.worker_standalone(heartbeat_conninfo, fold_factory=fold_factory)
            elif binding.mode == "shared-gateway":
                manager = store_type.shared_gateway_standalone(heartbeat_conninfo, fold_factory=fold_factory)
            else:
                manager = store_type.standalone(heartbeat_conninfo, fold_factory=fold_factory)
            with manager as store:
                store.clock_sql = clock_sql
                store.configure_reader(level=reader_level)
                store.configure_worker(
                    level=supports.max_level,
                    handler_versions=supports.handler_versions,
                    capabilities=supports.capabilities,
                )
                if configured:
                    store.configure_input_admission(signer, allow_unsigned=allow_unsigned)
                yield cast(SessionStore, store)

        return open_store

    def _heartbeat_connection(self, conninfo: str):
        binding = self.runtime_binding
        if binding.mode == "tenant":
            assert binding.tenant_id is not None
            manager = PostgresStore.tenant_standalone(conninfo, tenant_id=binding.tenant_id, fold_factory=self.fold_factory)
        elif binding.mode == "worker":
            manager = PostgresStore.worker_standalone(conninfo, fold_factory=self.fold_factory)
        elif binding.mode == "shared-gateway":
            manager = PostgresStore.shared_gateway_standalone(conninfo, fold_factory=self.fold_factory)
        else:
            manager = PostgresStore.standalone(conninfo, fold_factory=self.fold_factory)
        return self._inherit_runtime_configuration(manager)

    @contextmanager
    def _bound_append(self, lease: Lease) -> Iterator[None]:
        # Rollback restores settings on failure; success must restore them explicitly.
        with self._transaction():
            previous = self._one("SELECT current_setting('lock_timeout'),current_setting('statement_timeout')")
            self._rows(
                f"WITH remaining AS (SELECT GREATEST(1,%s-({self.clock_sql})) AS ms) "
                "SELECT set_config('lock_timeout',ms::text || 'ms',true), "
                "set_config('statement_timeout',ms::text || 'ms',true) FROM remaining",
                (lease.expires_at_ms,),
            )
            yield
            self._rows(
                "SELECT set_config('lock_timeout',%s,true),set_config('statement_timeout',%s,true)",
                previous,
            )

    @staticmethod
    def _connect(conninfo: str) -> Connection[Any]:
        import psycopg

        return psycopg.connect(conninfo, autocommit=True)

    @classmethod
    @contextmanager
    def standalone(cls, conninfo: str, *, fold_factory: Callable[[], Fold]) -> Iterator[PostgresStore]:
        with cls._connect(conninfo) as connection:
            binding = _read_runtime_binding(connection, allow_unbound_internal=False)
            yield cls(connection, fold_factory=fold_factory, runtime_binding=binding)

    @classmethod
    @contextmanager
    def schema_admin_standalone(cls, conninfo: str, *, fold_factory: Callable[[], Fold]) -> Iterator[PostgresStore]:
        """Open a trusted migration/admin connection for schema or runtime-role installation."""

        with cls._connect(conninfo) as connection:
            yield cls(connection, fold_factory=fold_factory, runtime_binding=UNBOUND_INTERNAL_BINDING)

    @classmethod
    @contextmanager
    def worker_standalone(cls, conninfo: str, *, fold_factory: Callable[[], Fold]) -> Iterator[PostgresStore]:
        """Open a dedicated cross-tenant worker login without claiming tenant isolation."""

        with cls._connect(conninfo) as connection:
            yield cls(connection, fold_factory=fold_factory, runtime_binding=RuntimeBinding("worker"))

    @classmethod
    @contextmanager
    def shared_gateway_standalone(cls, conninfo: str, *, fold_factory: Callable[[], Fold]) -> Iterator[PostgresStore]:
        """Open an explicitly weaker shared-gateway login without claiming tenant isolation."""

        with cls._connect(conninfo) as connection:
            yield cls(connection, fold_factory=fold_factory, runtime_binding=RuntimeBinding("shared-gateway"))

    def _rows(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with self._connection_lock:
            context_tenant = self.tenant_context_id
            if context_tenant is not None:
                from psycopg.pq import TransactionStatus

                status = self.connection.info.transaction_status
                if status == TransactionStatus.IDLE:
                    with self.connection.transaction():
                        _bind_tenant_context(self.connection, context_tenant)
                        return _query_rows(
                            self.connection,
                            query,
                            params,
                            binary=query.lstrip().startswith("SELECT"),
                        )
                if status == TransactionStatus.INTRANS:
                    _bind_tenant_context(self.connection, context_tenant)
            return _query_rows(self.connection, query, params, binary=query.lstrip().startswith("SELECT"))

    def _execute(self, query: Any) -> None:
        with self._connection_lock, self.connection.cursor() as cursor:
            cursor.execute(query)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        from psycopg.pq import TransactionStatus

        try:
            with self._connection_lock:
                self._read_snapshot_prefix = None
                context_tenant = self.tenant_context_id
                with self.connection.transaction():
                    if self.connection.info.transaction_status == TransactionStatus.INTRANS:
                        _bind_tenant_context(self.connection, context_tenant)
                    yield
        except BaseException:
            self._fold_cache = None
            self._previous_prefix = None
            raise

    def _transaction_id(self) -> object:
        from psycopg.pq import TransactionStatus

        if self.connection.info.transaction_status != TransactionStatus.INTRANS:
            raise RuntimeError("SessionTx requires an open transaction")
        return self._one("SELECT pg_current_xact_id()::text")[0]

    def _require_create_isolation(self) -> None:
        isolation = self._one("SELECT current_setting('transaction_isolation')")[0]
        if isolation not in {"read committed", "read uncommitted"}:
            raise Conflict("PostgreSQL session creation requires READ COMMITTED isolation")

    def _lock_creation_identity(self, session_id: str) -> None:
        self._require_create_isolation()
        frame = canonical_json_bytes(["vv-agent/S1/postgres-session-creation/v1", self.tenant_id, session_id])
        lock_id = int.from_bytes(sha256(frame).digest()[:8], "big", signed=True)
        self._rows("SELECT pg_catalog.pg_advisory_xact_lock(%s::bigint)", (lock_id,))

    def _transaction_in_progress(self) -> bool:
        from psycopg.pq import TransactionStatus

        return self.connection.info.transaction_status != TransactionStatus.IDLE

    @staticmethod
    def _object_journal_digest(identity: JournalIdentity, content_digest: str) -> str:
        validate_sha256(content_digest, "PostgreSQL object content binding")
        binding = canonical_json_bytes(
            [
                "vv-agent/S1/postgres-object-journal/v1",
                identity.database_id,
                identity.journal_id,
                identity.namespace,
                content_digest,
            ]
        )
        return sha256(binding).hexdigest()

    def _provision_object_journal(self, journal: _PostgresContentJournal, *, content_digest: str) -> None:
        """Pin the authenticated journal installation on a quiesced primary."""

        from psycopg import sql
        from psycopg.rows import tuple_row

        with self._connection_lock:
            if self._transaction_in_progress() or self.runtime_mode != "unbound-internal":
                raise ContentUnavailable("object journal provisioning requires an idle PostgreSQL schema owner")
            with self._transaction():
                if not _schema_is_current(self.connection):
                    raise ContentUnavailable("object journal provisioning requires a current PostgreSQL S1 installation")
                (schema,) = self._one("SELECT pg_catalog.current_schema()")
                owner = self._rows(
                    "SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
                    "WHERE n.nspname=%s AND c.relname='sk_settings' AND session_user=current_user "
                    "AND pg_catalog.pg_get_userbyid(c.relowner)=session_user "
                    "AND pg_catalog.pg_has_role(session_user,n.nspowner,'USAGE')",
                    (schema,),
                )
                if owner != [(1,)]:
                    raise ContentUnavailable("object journal provisioning requires the PostgreSQL schema owner")
                bind = sql.SQL(
                    "UPDATE {}.sk_settings SET object_journal_binding=%s "
                    "WHERE singleton=1 AND database_id=%s AND object_journal_binding IS NULL RETURNING singleton"
                ).format(sql.Identifier(schema))
                with self.connection.cursor(row_factory=tuple_row) as cursor:
                    identity = journal.identity
                    cursor.execute(bind, (self._object_journal_digest(identity, content_digest), identity.database_id))
                    rows = cursor.fetchall()
                if rows != [(1,)]:
                    raise ContentUnavailable("object journal does not match an unbound PostgreSQL installation")

    @contextmanager
    def _object_publication_atomic(
        self,
        journal: _PostgresContentJournal,
        *,
        check_deadline: Callable[[], None],
        content_digest: str,
        object_deadline: _ObjectDeadline | None = None,
        lease: Lease | None = None,
    ) -> Iterator[tuple[SQLSessionTx, Callable[[], None]]]:
        """Open the primary transaction that owns one tenant's publication lock."""

        check_deadline()
        if not self._connection_lock.acquire(blocking=False):
            raise Conflict("object publication connection is busy")
        try:
            if self._transaction_in_progress():
                raise ContentUnavailable("object publication requires a store-owned outer PostgreSQL transaction")
            tenant_id = self.credential_tenant_id
            if tenant_id is None or tenant_id != journal.tenant_id or self.tenant_id != tenant_id:
                raise ContentUnavailable("object publication requires the authenticated journal tenant")
            check_deadline()
            with self._transaction():
                settings = self._one(
                    "SELECT current_setting('transaction_isolation'), current_setting('transaction_read_only'), "
                    "current_setting('vv.tenant_id',true)"
                )
                if settings != ("read committed", "off", tenant_id):
                    raise ContentUnavailable("object publication requires the authenticated READ COMMITTED write transaction")
                self._verify_object_journal(journal, content_digest)
                check_deadline()
                frame = canonical_json_bytes(
                    ["vv-agent/S1/postgres-object-publication/v1", journal.identity.namespace, tenant_id]
                )
                lock_id = int.from_bytes(sha256(frame).digest()[:8], "big", signed=True)
                if self._one("SELECT pg_catalog.pg_try_advisory_xact_lock(%s::bigint)", (lock_id,)) != (True,):
                    raise ObjectPublicationBusy("object publication is busy for this tenant")
                if lease is not None:
                    from psycopg import errors

                    if lease.tenant_id != tenant_id:
                        raise LeaseLost("object publication lease belongs to another tenant")
                    check_deadline()
                    self._rows("SELECT set_config('lock_timeout','250ms',true)")
                    try:
                        self._write_ceiling()
                        row = self._lock(lease.session_id)
                    except errors.LockNotAvailable as exc:
                        raise ObjectPublicationBusy("object publication could not lock the leased session") from exc
                    self._check_lease(lease.session_id, lease, row, self._now())
                    self._require_object_root_session(lease.session_id)
                tx = SQLSessionTx(self)
                tx._bind_external_deadline(check_deadline, object_deadline)

                def check_fence() -> None:
                    tx._check_content_persistence_deadline()
                    tx._active()
                    tx._check_content_persistence_deadline()

                check_fence()
                yield tx, check_fence
                check_fence()
                tx._finalize_lease_fence()
                if lease is not None:
                    self._check_lease(lease.session_id, lease, row, self._now())
                check_deadline()
        finally:
            self._connection_lock.release()

    def _require_object_root_session(self, session_id: str) -> None:
        if self._one(
            "SELECT parent_session_id FROM sk_session WHERE tenant_id=%s AND session_id=%s",
            (self.tenant_id, session_id),
        ) != (None,):
            raise ContentUnavailable("PostgreSQL object publication requires a root session")

    def _verify_object_journal(self, journal: _PostgresContentJournal, content_digest: str) -> None:
        if self.credential_tenant_id != journal.tenant_id or self.tenant_id != journal.tenant_id:
            raise ContentUnavailable("object publication requires the authenticated journal tenant")
        self.verify_tenant_connection(self.connection, journal.tenant_id)
        identity = journal.identity
        if self._object_installation() != (identity.database_id, self._object_journal_digest(identity, content_digest)):
            raise ContentUnavailable("object journal binding does not match the PostgreSQL installation")

    def _object_installation(self) -> tuple[str, str | None] | None:
        from psycopg import sql
        from psycopg.rows import tuple_row

        with self._connection_lock, _read_only_probe(self.connection):
            (schema,) = _query_rows(self.connection, "SELECT pg_catalog.current_schema()")[0]
            qualified_table = sql.Identifier(schema, "sk_settings")
            if not _query_rows(
                self.connection, "SELECT pg_catalog.to_regclass(%s)", (qualified_table.as_string(self.connection),)
            )[0][0]:
                return None
            pinned_settings = sql.SQL("SELECT database_id,object_journal_binding FROM {} WHERE singleton=1").format(
                qualified_table
            )
            with self.connection.cursor(row_factory=tuple_row) as cursor:
                cursor.execute(pinned_settings)
                return cursor.fetchone()

    def _schema_tables(self) -> set[str]:
        return _schema_tables(self.connection)

    def _install_runtime_role(
        self,
        *,
        role: str,
        password: str,
        mode: Literal["tenant", "worker", "shared-gateway"],
        tenant_id: str | None,
    ) -> RLSInstallation:
        from psycopg import sql

        _validate_name(role, "runtime role")
        _validate_name(password, "runtime role password")
        if tenant_id is not None:
            _validate_name(tenant_id, "tenant_id")
        marker = _role_marker(mode, tenant_id)
        policy = _policy_name(role)

        with self._transaction():
            if not _schema_is_current(self.connection):
                raise ValueError("PostgreSQL current S1 session schema is required before RLS")
            schema = self._one("SELECT current_schema()")[0]
            if not isinstance(schema, str):
                raise ValueError("PostgreSQL current schema is unavailable")

            existing = self._rows("SELECT shobj_description(oid,'pg_authid') FROM pg_roles WHERE rolname=%s", (role,))
            if existing:
                if existing[0][0] != marker:
                    raise ValueError("runtime role already exists without the matching vv-agent RLS marker")
            else:
                self._execute(
                    sql.SQL(
                        "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS NOCREATEROLE NOCREATEDB NOREPLICATION NOINHERIT"
                    ).format(sql.Identifier(role), sql.Literal(password))
                )
                self._execute(sql.SQL("COMMENT ON ROLE {} IS {}").format(sql.Identifier(role), sql.Literal(marker)))

            row = _runtime_role_state(self.connection, role)
            if row is None:
                raise ValueError("runtime role is unavailable")
            if not row[0] or any(row[1:7]) or row[8] or row[9] or row[10] or row[11] or row[12]:
                raise ValueError("runtime role must be a dedicated login without elevated memberships or DDL rights")

            tenant_predicate = None
            if mode == "tenant":
                assert tenant_id is not None
                tenant_predicate = sql.SQL(
                    "session_user = {} AND tenant_id = {} AND current_setting('vv.tenant_id', true) = {}"
                ).format(sql.Literal(role), sql.Literal(tenant_id), sql.Literal(tenant_id))
            elif mode == "worker":
                tenant_predicate = sql.SQL("session_user = {}").format(sql.Literal(role))
            else:
                tenant_predicate = sql.SQL(
                    "session_user = {} AND current_setting('vv.tenant_id', true) IS NOT NULL "
                    "AND tenant_id = current_setting('vv.tenant_id', true)"
                ).format(sql.Literal(role))

            for table in sorted(SESSION_TABLES):
                self._execute(
                    sql.SQL("ALTER TABLE {}.{} ENABLE ROW LEVEL SECURITY").format(sql.Identifier(schema), sql.Identifier(table))
                )
                self._execute(
                    sql.SQL("ALTER TABLE {}.{} FORCE ROW LEVEL SECURITY").format(sql.Identifier(schema), sql.Identifier(table))
                )
                self._execute(
                    sql.SQL("DROP POLICY IF EXISTS {} ON {}.{}").format(
                        sql.Identifier(policy), sql.Identifier(schema), sql.Identifier(table)
                    )
                )
                self._execute(
                    sql.SQL("CREATE POLICY {} ON {}.{} FOR ALL TO {} USING ({}) WITH CHECK ({})").format(
                        sql.Identifier(policy),
                        sql.Identifier(schema),
                        sql.Identifier(table),
                        sql.Identifier(role),
                        tenant_predicate,
                        tenant_predicate,
                    )
                )
                self._execute(
                    sql.SQL("REVOKE ALL ON TABLE {}.{} FROM PUBLIC").format(sql.Identifier(schema), sql.Identifier(table))
                )
                self._execute(
                    sql.SQL("GRANT SELECT,INSERT,UPDATE,DELETE ON TABLE {}.{} TO {}").format(
                        sql.Identifier(schema), sql.Identifier(table), sql.Identifier(role)
                    )
                )

            self._execute(
                sql.SQL("GRANT SELECT ON TABLE {}.sk_settings TO {}").format(sql.Identifier(schema), sql.Identifier(role))
            )
            self._execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(schema), sql.Identifier(role)))
            database = self._one("SELECT current_database()")[0]
            self._execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(database), sql.Identifier(role)))

        return RLSInstallation(
            mode=mode,
            role=role,
            tenant_id=tenant_id,
            credential_isolation=mode == "tenant",
        )

    def install_tenant_rls(self, tenant_id: str, role: str, password: str) -> RLSInstallation:
        """Install a credential-isolated tenant login and FORCE RLS policies."""

        return self._install_runtime_role(role=role, password=password, mode="tenant", tenant_id=tenant_id)

    def install_worker_rls(self, role: str, password: str) -> RLSInstallation:
        """Install a distinct cross-tenant worker login; it is not a tenant path."""

        return self._install_runtime_role(role=role, password=password, mode="worker", tenant_id=None)

    def install_shared_gateway_rls(self, role: str, password: str) -> RLSInstallation:
        """Install explicit context-only isolation; shared credentials can impersonate tenants."""

        return self._install_runtime_role(role=role, password=password, mode="shared-gateway", tenant_id=None)

    @classmethod
    def verify_tenant_connection(cls, connection: Connection[Any], tenant_id: str) -> None:
        """Reject any runtime login that cannot be trusted for the requested tenant."""

        _validate_name(tenant_id, "tenant_id")
        binding = RuntimeBinding("tenant", tenant_id)
        try:
            _verified_role_binding(connection, binding)
            if not _schema_is_current(connection):
                raise ValueError("current S1 schema and RLS policies are incomplete")
            cls._verify_bound_policy(connection, binding)
        except ValueError as exc:
            raise ValueError(f"tenant runtime role is not a safe credential-bound login for this tenant: {exc}") from exc

    @staticmethod
    def _verify_bound_policy(connection: Connection[Any], binding: RuntimeBinding) -> None:
        with _read_only_probe(connection):
            role = _probe_rows(connection, "SELECT current_user")[0][0]
            rows = _probe_rows(
                connection,
                "SELECT count(*) FROM pg_policy p JOIN pg_class c ON c.oid=p.polrelid "
                "JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=current_schema() AND c.relname=ANY(%s) AND p.polname=%s "
                "AND (SELECT oid FROM pg_roles WHERE rolname=session_user)=ANY(p.polroles)",
                (sorted(SESSION_TABLES), _policy_name(role)),
            )
            if rows != [(len(SESSION_TABLES),)]:
                raise ValueError(f"{binding.mode} runtime policies are not completely forced")

    @classmethod
    @contextmanager
    def tenant_standalone(
        cls,
        conninfo: str,
        *,
        tenant_id: str,
        fold_factory: Callable[[], Fold],
    ) -> Iterator[PostgresStore]:
        """Open a dedicated tenant LOGIN whose transactions bind its tenant context."""

        with cls._connect(conninfo) as connection:
            yield cls(connection, fold_factory=fold_factory, runtime_binding=RuntimeBinding("tenant", tenant_id))

    def install_schema(self) -> None:
        with self._transaction():
            if _schema_is_current(self.connection):
                return
            existing = _schema_tables(self.connection)
            if existing:
                raise ValueError("unsupported PostgreSQL session schema version")
            self._rows(DDL)
            self._rows(
                "INSERT INTO sk_settings(singleton,storage_major,write_level,database_id) VALUES (1,%s,1,%s)",
                (STORAGE_MAJOR, secrets.token_hex(32)),
            )
            if not _schema_is_current(self.connection):
                raise ValueError("PostgreSQL session schema fingerprint does not match this runtime")


class _PostgresContentBackend:
    lock_clause = " FOR UPDATE"

    def __init__(
        self,
        connection: Connection[Any],
        connection_lock: RLock | None,
        *,
        credential_tenant_id: str | None = None,
        tenant_context_id: str | None = None,
        runtime_mode: RuntimeMode = "unbound-internal",
    ) -> None:
        self.connection = connection
        self.connection_lock = connection_lock
        self.credential_tenant_id = credential_tenant_id
        self.tenant_context_id = tenant_context_id
        self.runtime_mode = runtime_mode

    @contextmanager
    def _serialized(self) -> Iterator[None]:
        if self.connection_lock is None:
            yield
            return
        with self.connection_lock:
            yield

    @contextmanager
    def read_snapshot(self) -> Iterator[None]:
        """Join a caller transaction or open one short repeatable read."""

        from psycopg.pq import TransactionStatus

        with self._serialized():
            status = self.connection.info.transaction_status
            if status == TransactionStatus.INTRANS:
                _bind_tenant_context(self.connection, self.tenant_context_id)
                yield
                return
            if status != TransactionStatus.IDLE:
                raise RuntimeError(f"PostgreSQL content snapshot cannot join transaction status {status.name}")
            self.connection.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            try:
                _bind_tenant_context(self.connection, self.tenant_context_id)
                yield
            except BaseException:
                with suppress(BaseException):
                    self.connection.rollback()
                raise
            try:
                self.connection.commit()
            except BaseException:
                with suppress(BaseException):
                    self.connection.rollback()
                raise

    @staticmethod
    def _query(query: str) -> LiteralString:
        return cast(LiteralString, query.replace("{p}", "%s"))

    def fetchone(self, query: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
        from psycopg.rows import tuple_row

        with self._serialized(), self.connection.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(self._query(query), params)
            row = cursor.fetchone()
            return None if row is None else tuple(row)

    def fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        from psycopg.rows import tuple_row

        with self._serialized(), self.connection.cursor(row_factory=tuple_row) as cursor:
            # Ciphertext rows contain BYTEA. Binary results avoid hex round trips.
            cursor.execute(self._query(query), params, binary=True)
            return [tuple(row) for row in cursor.fetchall()]

    def fetch_exact_record_links(
        self,
        tenant_id: str,
        session_id: str,
        first_record_seq: int,
        last_record_seq: int,
    ) -> list[tuple[Any, ...]]:
        """Read one exact dense owner range in link order."""

        return self.fetchall(
            "SELECT record_seq,input_id,slot,content_id FROM sk_content_link "
            "WHERE tenant_id={p} AND session_id={p} AND record_seq BETWEEN {p} AND {p} "
            "ORDER BY record_seq,slot",
            (tenant_id, session_id, first_record_seq, last_record_seq),
        )

    def fetch_exact_content_rows(
        self,
        tenant_id: str,
        session_id: str,
        content_ids: tuple[str, ...],
    ) -> list[tuple[Any, ...]]:
        """Read one exact content-id set inside the frame's existing snapshot."""

        if not content_ids:
            return []
        query = (
            f"SELECT {_CONTENT_COLUMNS} FROM sk_content "
            "WHERE tenant_id={p} AND session_id={p} AND content_id=ANY({p}::text[])"
        )
        parameters = (tenant_id, session_id, list(content_ids))
        if len(content_ids) <= _READ_FRAME_OWNER_BATCH:
            return self.fetchall(query, parameters)

        from psycopg import ServerCursor
        from psycopg.rows import tuple_row

        with self._serialized():
            cursor = ServerCursor(
                self.connection,
                f"vv_content_frame_{secrets.token_hex(8)}",
                row_factory=tuple_row,
            )
            try:
                cursor.itersize = _READ_FRAME_OWNER_BATCH
                cursor.execute(self._query(query), parameters, binary=True)
                rows: list[tuple[Any, ...]] = []
                while batch := cursor.fetchmany(_READ_FRAME_OWNER_BATCH):
                    rows.extend(tuple(row) for row in batch)
            except BaseException:
                with suppress(BaseException):
                    cursor.close()
                raise
            else:
                cursor.close()
                return rows

    def run(self, query: str, params: tuple[Any, ...] = ()) -> int:
        with self._serialized(), self.connection.cursor() as cursor:
            cursor.execute(self._query(query), params)
            return cursor.rowcount

    def require_transaction(self) -> None:
        from psycopg.pq import TransactionStatus

        with self._serialized():
            if self.connection.info.transaction_status != TransactionStatus.INTRANS:
                raise RuntimeError("durable content writes require the caller's outer session transaction")
            _bind_tenant_context(self.connection, self.tenant_context_id)

    def now_ms(self) -> int:
        row = self.fetchone(f"SELECT {CLOCK}")
        if row is None:
            raise RuntimeError("PostgreSQL content clock is unavailable")
        return row[0]


class CorePostgresContentStore(CoreSQLContentStore):
    """PostgreSQL content adapter over a caller-owned connection or transaction."""

    def __init__(
        self,
        source: PostgresStore | _TenantSQLStore | SQLSessionTx | Connection[Any],
        key_provider: KeyProvider,
        cipher: ContentCipher,
        *,
        provider_id: str | None = None,
    ) -> None:
        connection: Any
        connection_lock: RLock | None = None
        credential_tenant_id: str | None = None
        tenant_context_id: str | None = None
        runtime_mode: RuntimeMode = "unbound-internal"
        if isinstance(source, SQLSessionTx):
            if not isinstance(source.store, PostgresStore):
                raise TypeError("PostgresContentStore transaction must belong to PostgresStore")
            connection = cast(Any, source.store.connection)
            connection_lock = source.store._connection_lock
            credential_tenant_id = source.store.credential_tenant_id
            tenant_context_id = source.store.tenant_context_id
            runtime_mode = source.store.runtime_mode
        elif isinstance(source, _TenantSQLStore):
            bound = source._bound
            if not isinstance(bound, PostgresStore):
                raise TypeError("PostgresContentStore tenant view must belong to PostgresStore")
            connection = cast(Any, bound.connection)
            connection_lock = bound._connection_lock
            credential_tenant_id = bound.credential_tenant_id
            tenant_context_id = bound.tenant_context_id
            runtime_mode = bound.runtime_mode
        else:
            if isinstance(source, PostgresStore):
                connection = cast(Any, source.connection)
                connection_lock = source._connection_lock
                credential_tenant_id = source.credential_tenant_id
                tenant_context_id = source.tenant_context_id
                runtime_mode = source.runtime_mode
            else:
                connection = cast(Any, source)
                binding = _read_runtime_binding(connection, allow_unbound_internal=True)
                runtime_mode = binding.mode
                if binding.mode == "tenant":
                    credential_tenant_id = binding.tenant_id
                    tenant_context_id = binding.tenant_id
        super().__init__(
            _PostgresContentBackend(
                connection,
                connection_lock,
                credential_tenant_id=credential_tenant_id,
                tenant_context_id=tenant_context_id,
                runtime_mode=runtime_mode,
            ),
            key_provider,
            cipher,
            provider_id=provider_id,
        )
