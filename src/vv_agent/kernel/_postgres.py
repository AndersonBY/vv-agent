"""Internal PostgreSQL reference store, bound to caller-owned psycopg v3 connections."""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal, LiteralString, cast

from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._sql import SQLStore
from vv_agent.kernel._store import Lease

if TYPE_CHECKING:
    from psycopg import Connection

CLOCK = "floor(extract(epoch FROM clock_timestamp()) * 1000)::bigint"
STORAGE_MAJOR = "S1"
RLS_ISOLATION = "explicit-tenant-credentials"
RLS_INSTALL_REASON = "The base DDL is unisolated. Runtime tenant access requires install_tenant_rls() and tenant_standalone()."
RLS_SHARED_GATEWAY_REASON = (
    "Shared gateway context prevents accidental query leakage but does not provide tenant credential isolation."
)

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


# Tenant RLS is deliberately absent here: safe installation requires generated per-tenant LOGIN
# roles and grants, rather than a shared role or a GUC-only policy that can forge tenant context.
DDL = """
CREATE TABLE sk_settings (
    singleton integer PRIMARY KEY CHECK (singleton = 1),
    storage_major text NOT NULL CHECK (storage_major = 'S1'),
    write_level integer NOT NULL CHECK (write_level >= 1)
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
    kind text NOT NULL CHECK (kind IN ('session_created','turn_started','input_applied','op_planned','op_started',
        'op_prepared','turn_parked','op_parked','op_completed','op_unknown','context_compacted','boundary_recorded','usage_observed','turn_ended')),
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
    kind text NOT NULL CHECK (kind IN ('user','steer','follow_up','provider_result','approval_answer','child_result',
        'control','provider_evidence','model_selection')),
    body bytea NOT NULL, digest text NOT NULL CHECK (digest ~ '^[0-9a-f]{64}$'),
    admission_key_id text, admission_signature bytea,
    admission_signed_ms bigint CHECK (admission_signed_ms IS NULL OR admission_signed_ms >= 0),
    available_ms bigint NOT NULL CHECK (available_ms >= 0), received_ms bigint NOT NULL CHECK (received_ms >= 0),
    consumed_seq bigint CHECK (consumed_seq IS NULL OR consumed_seq > 0),
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


class PostgresStore(SQLStore):
    """A serialized connection; atomic() nests as a savepoint in host transactions."""

    # Session keys never change. This still serializes writers while allowing
    # child inserts to retain their parent foreign-key KEY SHARE locks.
    lock_clause = " FOR NO KEY UPDATE"
    clock_sql = CLOCK

    def __init__(self, connection: Connection[Any], *, fold_factory: Callable[[], Fold]):
        super().__init__(fold_factory=fold_factory)
        self.connection = connection
        self._connection_lock = RLock()

    def heartbeat_store(self):
        from psycopg.conninfo import conninfo_to_dict, make_conninfo

        conninfo = _connection_conninfo(self.connection)
        options = conninfo_to_dict(conninfo).get("options", "")
        return self._heartbeat_connection(
            make_conninfo(
                conninfo,
                connect_timeout=3,
                options=f"{options} -c lock_timeout=250ms -c statement_timeout=1000ms",
            )
        )

    def _heartbeat_connection(self, conninfo: str):
        return PostgresStore.standalone(conninfo, fold_factory=self.fold_factory)

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
            yield cls(connection, fold_factory=fold_factory)

    def _rows(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with self._connection_lock:
            return _query_rows(self.connection, query, params, binary=query.lstrip().startswith("SELECT"))

    def _execute(self, query: Any) -> None:
        with self._connection_lock, self.connection.cursor() as cursor:
            cursor.execute(query)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        try:
            with self._connection_lock, self.connection.transaction():
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

    def _schema_tables(self) -> set[str]:
        return {
            row[0]
            for row in self._rows(
                "SELECT c.relname FROM pg_catalog.pg_class c "
                "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=current_schema() AND c.relkind IN ('r','p') "
                "AND c.relname LIKE 'sk\\_%%' ESCAPE '\\'"
            )
        }

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
            if self._schema_tables() != SCHEMA_TABLES:
                raise ValueError("PostgreSQL session schema must be installed before RLS")
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

            role_state = self._rows(
                "SELECT r.rolcanlogin,r.rolsuper,r.rolbypassrls,r.rolcreaterole,r.rolcreatedb,"
                "r.rolreplication,r.rolinherit,"
                "EXISTS(SELECT 1 FROM pg_auth_members m WHERE m.member=r.oid),"
                "EXISTS(SELECT 1 FROM pg_class c WHERE c.relowner=r.oid AND c.relname=ANY(%s)),"
                "EXISTS(SELECT 1 FROM pg_namespace n WHERE n.nspowner=r.oid AND n.nspname=current_schema()),"
                "EXISTS(SELECT 1 FROM pg_database d WHERE d.datname=current_database() AND d.datdba=r.oid),"
                "has_schema_privilege(r.oid,current_schema(),'CREATE') "
                "FROM pg_roles r WHERE r.rolname=%s",
                (sorted(SCHEMA_TABLES), role),
            )
            if len(role_state) != 1:
                raise ValueError("runtime role is unavailable")
            row = role_state[0]
            if not row[0] or any(row[1:7]) or row[7] or row[8] or row[9] or row[10] or row[11]:
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
        identity = _query_rows(connection, "SELECT session_user,current_user")
        if len(identity) != 1 or identity[0][0] != identity[0][1]:
            raise ValueError("tenant runtime requires session_user and current_user to match")
        role = identity[0][0]
        state = _query_rows(
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
        if len(state) != 1:
            raise ValueError("tenant runtime role is unavailable")
        row = state[0]
        if (
            not row[0]
            or any(row[1:7])
            or row[7] != _role_marker("tenant", tenant_id)
            or row[8]
            or row[9]
            or row[10]
            or row[11]
            or row[12]
        ):
            raise ValueError("tenant runtime role is not a safe credential-bound login for this tenant")
        policy = _policy_name(role)
        policies = _query_rows(
            connection,
            "SELECT count(*) FROM pg_policy p JOIN pg_class c ON c.oid=p.polrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname=ANY(%s) AND p.polname=%s "
            "AND (SELECT oid FROM pg_roles WHERE rolname=session_user)=ANY(p.polroles)",
            (sorted(SESSION_TABLES), policy),
        )
        guarded = _query_rows(
            connection,
            "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname=ANY(%s) "
            "AND c.relrowsecurity AND c.relforcerowsecurity",
            (sorted(SESSION_TABLES),),
        )
        if policies != [(len(SESSION_TABLES),)] or guarded != [(len(SESSION_TABLES),)]:
            raise ValueError("tenant runtime policies are not completely forced")

    @classmethod
    @contextmanager
    def tenant_standalone(
        cls,
        conninfo: str,
        *,
        tenant_id: str,
        fold_factory: Callable[[], Fold],
    ) -> Iterator[PostgresStore]:
        """Open and verify a dedicated tenant LOGIN; callers still set transaction-local context."""

        with cls._connect(conninfo) as connection:
            cls.verify_tenant_connection(connection, tenant_id)
            yield cls(connection, fold_factory=fold_factory)

    def install_schema(self) -> None:
        with self._transaction():
            existing = self._schema_tables()
            if existing:
                if existing == SCHEMA_TABLES:
                    settings = self._rows("SELECT singleton,storage_major,write_level FROM sk_settings")
                    if settings == [(1, STORAGE_MAJOR, 1)]:
                        return
                raise ValueError("unsupported PostgreSQL session schema version")
            self._rows(DDL)
            self._rows(
                "INSERT INTO sk_settings(singleton,storage_major,write_level) VALUES (1,%s,1)",
                (STORAGE_MAJOR,),
            )
