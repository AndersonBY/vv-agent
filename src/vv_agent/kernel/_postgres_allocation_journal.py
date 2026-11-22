"""Independent PostgreSQL journal for S1 object reservations."""

from __future__ import annotations

import json
import secrets
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import astuple, dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, LiteralString

from ._allocation_journal import (
    Allocation,
    AllocationOwner,
    JournalIdentity,
    KeyIntent,
    _JournalTemporarilyUnavailable,
    _require_id,
    _require_text,
)
from ._content import (
    _EXACT_KEY_ID_PREFIX,
    ContentCleanupUnavailable,
    ContentUnavailable,
    KeyDestroyOutcome,
    KeyDestroyStatus,
    _ExactKeyIntent,
)
from ._postgres import CLOCK, _query_rows

if TYPE_CHECKING:
    from psycopg import Connection

_SCHEMA = "vv_allocation_journal"
_TABLE = f"{_SCHEMA}.allocation"
_IDENTITY = f"{_SCHEMA}.identity"
_KEY_TABLE = f"{_SCHEMA}.key_intent"
_OWNER = "tenant_id,session_id,content_id,admission_session_id,admission_id,provider_instance_id,key_id"
_COLUMNS = f"allocation_id,state,pin_until_ms,reservation_etag,delete_confirmed,{_OWNER}"
_KEY_OWNER = "key_id,provider_instance_id,tenant_id,session_id,admission_root_id,admission_id"
_KEY_COLUMNS = f"{_KEY_OWNER},state,pin_until_ms,evidence_id"
_VERSION = 7
_BOUND_PAGE_LIMIT = 8
_OWNER_MATCH = " AND ".join(f"{column} IS NOT DISTINCT FROM %s" for column in _OWNER.split(","))
_CLAIM_MATCH = (
    f"allocation_id=%s AND state='reaping' AND pin_until_ms=%s AND reservation_etag IS NOT DISTINCT FROM %s AND {_OWNER_MATCH}"
)

_DDL = (
    f"""CREATE TABLE {_IDENTITY} (
        singleton integer PRIMARY KEY CHECK (singleton = 1),
        revision integer NOT NULL,
        journal_id text NOT NULL,
        database_id text NOT NULL,
        namespace text NOT NULL,
        security_definition text NOT NULL
    )""",
    f"""CREATE TABLE {_TABLE} (
        allocation_id bytea PRIMARY KEY CHECK (octet_length(allocation_id) = 32),
        state text NOT NULL DEFAULT 'unbound' CHECK (state IN ('unbound','bound','reaping')),
        pin_until_ms bigint NOT NULL CHECK (pin_until_ms > 0),
        tenant_id text CHECK (length(tenant_id) > 0),
        session_id text CHECK (length(session_id) > 0),
        content_id text CHECK (length(content_id) > 0),
        admission_session_id text CHECK (length(admission_session_id) > 0),
        admission_id text CHECK (length(admission_id) > 0),
        provider_instance_id text CHECK (length(provider_instance_id) > 0),
        key_id text CHECK (length(key_id) > 0),
        reservation_etag text CHECK (length(reservation_etag) > 0),
        delete_confirmed boolean NOT NULL DEFAULT false CHECK (
            NOT delete_confirmed OR (state='reaping' AND tenant_id IS NOT NULL)
        ),
        CHECK (
            (state IN ('unbound','reaping') AND num_nonnulls({_OWNER},reservation_etag) = 0)
            OR
            (state IN ('bound','reaping') AND num_nulls({_OWNER},reservation_etag) = 0)
        )
    )""",
    f"CREATE INDEX allocation_expiry ON {_TABLE}(pin_until_ms,allocation_id) WHERE tenant_id IS NULL",
    f"CREATE INDEX allocation_tenant_expiry ON {_TABLE}(tenant_id,pin_until_ms,allocation_id) WHERE tenant_id IS NOT NULL",
    f"CREATE INDEX allocation_owner ON {_TABLE}(tenant_id,session_id)",
    f"CREATE INDEX allocation_admission ON {_TABLE}(tenant_id,admission_session_id)",
    f"CREATE INDEX allocation_key ON {_TABLE}(provider_instance_id,key_id) WHERE key_id IS NOT NULL",
    f"""CREATE TABLE {_KEY_TABLE} (
        key_id text NOT NULL CHECK (key_id ~ '^{_EXACT_KEY_ID_PREFIX}[0-9a-f]{{64}}$'),
        provider_instance_id text NOT NULL CHECK (length(provider_instance_id) > 0),
        tenant_id text NOT NULL CHECK (length(tenant_id) > 0),
        session_id text NOT NULL CHECK (length(session_id) > 0),
        admission_root_id text NOT NULL CHECK (length(admission_root_id) > 0),
        admission_id text NOT NULL CHECK (length(admission_id) > 0),
        state text NOT NULL DEFAULT 'reserved' CHECK (state IN ('reserved','allocating','reaping','retired')),
        pin_until_ms bigint NOT NULL CHECK (pin_until_ms > 0),
        evidence_id text CHECK (length(evidence_id) > 0),
        PRIMARY KEY (provider_instance_id,key_id),
        UNIQUE (provider_instance_id,key_id,tenant_id,session_id,admission_root_id,admission_id),
        CHECK ((state='retired') = (evidence_id IS NOT NULL))
    )""",
    f"CREATE UNIQUE INDEX key_intent_admission ON {_KEY_TABLE}(tenant_id,session_id,admission_root_id,admission_id) "
    "WHERE state<>'retired'",
    f"CREATE INDEX key_intent_owner ON {_KEY_TABLE}(tenant_id,session_id)",
    f"CREATE INDEX key_intent_admission_root ON {_KEY_TABLE}(tenant_id,admission_root_id)",
    f"""ALTER TABLE {_TABLE} ADD COLUMN exact_key_id text
        GENERATED ALWAYS AS (CASE WHEN starts_with(key_id,'{_EXACT_KEY_ID_PREFIX}') THEN key_id END) STORED,
        ADD FOREIGN KEY (provider_instance_id,exact_key_id,tenant_id,session_id,admission_session_id,admission_id)
        REFERENCES {_KEY_TABLE}(provider_instance_id,key_id,tenant_id,session_id,admission_root_id,admission_id)""",
    f"""CREATE FUNCTION {_SCHEMA}.validate_key_transition() RETURNS trigger
        LANGUAGE plpgsql SET search_path=pg_catalog AS $$
        BEGIN
            IF NOT ((OLD.state='reserved' AND NEW.state='allocating' AND NEW.pin_until_ms>{CLOCK})
                 OR (OLD.state IN ('reserved','allocating') AND NEW.state='reaping')
                 OR (OLD.state='reaping' AND NEW.state='retired')) THEN
                RAISE EXCEPTION 'invalid key intent transition' USING ERRCODE='42501';
            END IF;
            RETURN NEW;
        END
        $$""",
    f"CREATE TRIGGER validate_key_transition BEFORE UPDATE ON {_KEY_TABLE} "
    f"FOR EACH ROW EXECUTE FUNCTION {_SCHEMA}.validate_key_transition()",
    f"""CREATE FUNCTION {_SCHEMA}.validate_exact_binding() RETURNS trigger
        LANGUAGE plpgsql SET search_path=pg_catalog AS $$
        BEGIN
            IF OLD.state='unbound' AND NEW.state='bound' AND starts_with(NEW.key_id,'{_EXACT_KEY_ID_PREFIX}') THEN
                IF NOT EXISTS (
                    SELECT 1 FROM {_KEY_TABLE}
                    WHERE provider_instance_id=NEW.provider_instance_id AND key_id=NEW.key_id
                      AND tenant_id=NEW.tenant_id AND session_id=NEW.session_id AND state='allocating'
                      AND admission_root_id=NEW.admission_session_id AND admission_id=NEW.admission_id
                    FOR SHARE
                ) THEN
                    RAISE EXCEPTION 'exact key is unavailable for binding' USING ERRCODE='42501';
                END IF;
            END IF;
            RETURN NEW;
        END
        $$""",
    f"CREATE TRIGGER validate_exact_binding BEFORE UPDATE ON {_TABLE} "
    f"FOR EACH ROW EXECUTE FUNCTION {_SCHEMA}.validate_exact_binding()",
    f"""CREATE FUNCTION {_SCHEMA}.validate_delete_confirmation() RETURNS trigger
        LANGUAGE plpgsql SET search_path=pg_catalog AS $$
        BEGIN
            IF OLD.state<>'reaping' OR OLD.tenant_id IS NULL OR OLD.delete_confirmed
                OR NEW.state<>'reaping' OR NOT NEW.delete_confirmed THEN
                RAISE EXCEPTION 'invalid bound deletion confirmation' USING ERRCODE='42501';
            END IF;
            RETURN NEW;
        END
        $$""",
    f"CREATE TRIGGER validate_delete_confirmation BEFORE UPDATE OF delete_confirmed ON {_TABLE} "
    f"FOR EACH ROW EXECUTE FUNCTION {_SCHEMA}.validate_delete_confirmation()",
)

_SECURITY_DEFINITION = """
SELECT jsonb_build_array(
    n.nspacl,
    (SELECT jsonb_agg(jsonb_build_array(c.relname,c.relowner,c.relacl,c.relrowsecurity,c.relforcerowsecurity)
                     ORDER BY c.relname)
     FROM pg_class c WHERE c.relnamespace=n.oid AND c.relkind='r'),
    (SELECT jsonb_agg(jsonb_build_array(rel.relname,a.attname,a.attacl) ORDER BY rel.relname,a.attname)
     FROM pg_attribute a JOIN pg_class rel ON rel.oid=a.attrelid
     WHERE rel.relnamespace=n.oid AND rel.relkind='r' AND a.attnum>0 AND NOT a.attisdropped),
    (SELECT jsonb_agg(jsonb_build_array(rel.relname,p.polname,p.polcmd,p.polpermissive,p.polroles,
                    pg_get_expr(p.polqual,p.polrelid),pg_get_expr(p.polwithcheck,p.polrelid)) ORDER BY rel.relname,p.polname)
     FROM pg_policy p JOIN pg_class rel ON rel.oid=p.polrelid WHERE rel.relnamespace=n.oid),
    (SELECT jsonb_agg(pg_get_triggerdef(t.oid) ORDER BY t.tgname)
     FROM pg_trigger t JOIN pg_class rel ON rel.oid=t.tgrelid WHERE rel.relnamespace=n.oid AND NOT t.tgisinternal),
    (SELECT jsonb_agg(pg_get_functiondef(f.oid) ORDER BY f.proname)
     FROM pg_proc f WHERE f.pronamespace=n.oid),
    (SELECT pg_get_indexdef(i.oid) FROM pg_class i
     WHERE i.relnamespace=n.oid AND i.relname='allocation_tenant_expiry' AND i.relkind='i')
)::text
FROM pg_class allocation JOIN pg_namespace n ON n.oid=allocation.relnamespace
WHERE n.nspname=%s AND allocation.relname='allocation' AND allocation.relkind='r'
"""


@dataclass(frozen=True, slots=True)
class JournalRole:
    mode: Literal["anonymous", "binder", "scope", "key-intent", "cleanup"]
    tenant_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "anonymous" and self.tenant_id is None:
            return
        if self.mode in {"scope", "key-intent", "binder", "cleanup"} and self.tenant_id is not None:
            _require_text(self.tenant_id)
            return
        raise ValueError("invalid allocation journal role binding")

    def marker(self, identity: JournalIdentity) -> str:
        return "vv-allocation-journal:" + json.dumps([identity.journal_id, self.mode, self.tenant_id], ensure_ascii=True)


def _role_state(connection: Connection[Any], login: str) -> list[tuple[Any, ...]]:
    return _query_rows(
        connection,
        """
        SELECT shobj_description(r.oid,'pg_authid'),
           rolcanlogin AND NOT (rolsuper OR rolbypassrls OR rolcreaterole OR rolcreatedb OR rolreplication OR rolinherit)
           AND NOT EXISTS (SELECT 1 FROM pg_auth_members WHERE member=r.oid)
           AND NOT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname=%s AND nspowner=r.oid)
           AND NOT EXISTS (SELECT 1 FROM pg_database WHERE datname=current_database() AND datdba=r.oid)
           AND NOT EXISTS (
               SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
               WHERE n.nspname=%s AND c.relowner=r.oid
           )
        FROM pg_roles r WHERE rolname=%s
        """,
        (_SCHEMA, _SCHEMA, login),
    )


def _install_role(connection: Connection[Any], login: str, role: JournalRole, identity: JournalIdentity) -> None:
    from psycopg import sql

    principal = sql.Identifier(login)
    connection.execute(sql.SQL("COMMENT ON ROLE {} IS {}").format(principal, sql.Literal(role.marker(identity))))
    connection.execute(sql.SQL(f"GRANT USAGE ON SCHEMA {_SCHEMA} TO {{}}").format(principal))
    connection.execute(sql.SQL(f"GRANT SELECT ON {_IDENTITY} TO {{}}").format(principal))
    authority = sql.SQL("session_user = {}").format(sql.Literal(login))
    tenant_scope = sql.SQL("{} AND tenant_id = {}").format(authority, sql.Literal(role.tenant_id))

    def install(
        table: LiteralString,
        grants: tuple[LiteralString, ...],
        transitions: tuple[tuple[LiteralString, sql.Composable | None, sql.Composable | None], ...],
    ) -> None:
        for grant in grants:
            connection.execute(sql.SQL(f"GRANT {grant} ON {table} TO {{}}").format(principal))
        for command, using, check in transitions:
            policy = sql.Identifier(login + "_" + command.lower())
            statement = sql.SQL(f"CREATE POLICY {{}} ON {table} FOR {command} TO {{}}").format(policy, principal)
            if using is not None:
                statement += sql.SQL(" USING ({})").format(using)
            if check is not None:
                statement += sql.SQL(" WITH CHECK ({})").format(check)
            connection.execute(statement)

    if role.mode == "key-intent":
        install(
            _KEY_TABLE,
            ("SELECT", f"INSERT({_KEY_OWNER},pin_until_ms)", "UPDATE(state)"),
            (
                ("SELECT", tenant_scope, None),
                ("INSERT", None, tenant_scope + sql.SQL(f" AND state='reserved' AND pin_until_ms>{CLOCK}")),
                (
                    "UPDATE",
                    tenant_scope + sql.SQL(" AND state='reserved'"),
                    tenant_scope + sql.SQL(" AND state='allocating'"),
                ),
            ),
        )
        return
    if role.mode == "scope":
        install(_KEY_TABLE, ("SELECT",), (("SELECT", tenant_scope, None),))
        install(_TABLE, ("SELECT",), (("SELECT", tenant_scope, None),))
        return
    if role.mode == "cleanup":
        install(
            _KEY_TABLE,
            ("SELECT", "UPDATE(state,evidence_id)", "DELETE"),
            (
                ("SELECT", tenant_scope, None),
                (
                    "UPDATE",
                    tenant_scope + sql.SQL(" AND state<>'retired'"),
                    tenant_scope + sql.SQL(" AND state IN ('reaping','retired')"),
                ),
                ("DELETE", tenant_scope + sql.SQL(" AND state='retired'"), None),
            ),
        )
        grants = ("SELECT", "UPDATE(state,delete_confirmed)", "DELETE")
        transitions = (
            ("SELECT", tenant_scope, None),
            (
                "UPDATE",
                tenant_scope + sql.SQL(" AND state IN ('bound','reaping') AND NOT delete_confirmed"),
                tenant_scope + sql.SQL(" AND state='reaping'"),
            ),
            ("DELETE", tenant_scope + sql.SQL(" AND state='reaping' AND delete_confirmed"), None),
        )
    elif role.mode == "binder":
        install(
            _KEY_TABLE,
            (
                "SELECT(provider_instance_id,key_id,tenant_id,session_id,admission_root_id,admission_id,state)",
                "UPDATE(state)",
            ),
            # FOR SHARE needs an UPDATE grant. The false WITH CHECK blocks every key update.
            (("SELECT", tenant_scope, None), ("UPDATE", tenant_scope + sql.SQL(" AND state='allocating'"), sql.SQL("false"))),
        )
        grants = ("SELECT(allocation_id,state,pin_until_ms)", f"UPDATE(state,reservation_etag,{_OWNER})")
        visible = authority + sql.SQL(" AND (tenant_id IS NULL OR tenant_id = {})").format(sql.Literal(role.tenant_id))
        # PostgreSQL checks SELECT visibility on the new row as well as the old one.
        transitions = (
            ("SELECT", visible, None),
            (
                "UPDATE",
                authority + sql.SQL(f" AND tenant_id IS NULL AND state='unbound' AND pin_until_ms>{CLOCK}"),
                tenant_scope + sql.SQL(f" AND state='bound' AND pin_until_ms>{CLOCK}"),
            ),
        )
    else:
        visible = authority + sql.SQL(" AND tenant_id IS NULL")
        grants = ("SELECT", "INSERT(allocation_id,pin_until_ms)", "UPDATE(state)", "DELETE")
        transitions = (
            ("SELECT", visible, None),
            ("INSERT", None, visible + sql.SQL(f" AND state='unbound' AND pin_until_ms>{CLOCK}")),
            (
                "UPDATE",
                visible + sql.SQL(f" AND state='unbound' AND pin_until_ms<={CLOCK}"),
                visible + sql.SQL(" AND state='reaping'"),
            ),
            ("DELETE", visible + sql.SQL(" AND state='reaping'"), None),
        )
    install(_TABLE, grants, transitions)


class PostgresAllocationJournal:
    """One journal authority per independent connection, pinned to a provisioned identity."""

    @staticmethod
    def provision(
        admin: Connection[Any], *, database_id: str, namespace: str, roles: Mapping[str, JournalRole]
    ) -> JournalIdentity:
        """Install once in a dedicated database using fresh, pre-created logins.

        Include each tenant's separate cleanup login during provisioning if
        that tenant requires erasure. The recorded security definition covers
        the installed grants and policies.
        """
        from psycopg.pq import TransactionStatus

        if admin.info.transaction_status != TransactionStatus.IDLE:
            raise ValueError("journal provisioning requires an idle admin connection")
        required_modes = {"anonymous", "binder", "scope", "key-intent"}
        if not required_modes.issubset(role.mode for role in roles.values()):
            raise ValueError("journal provisioning requires the four publication authorities")
        identity = JournalIdentity(secrets.token_hex(32), database_id, namespace)
        with admin.transaction():
            admin.execute("SET LOCAL search_path = pg_catalog")
            core_tables = _query_rows(admin, "SELECT 1 FROM pg_class WHERE relname='sk_session' AND relkind IN ('r','p') LIMIT 1")
            if core_tables:
                raise ValueError("the allocation journal requires a dedicated database")
            admin.execute(f"CREATE SCHEMA {_SCHEMA}")
            admin.execute(f"REVOKE ALL ON SCHEMA {_SCHEMA} FROM PUBLIC")
            for statement in _DDL:
                admin.execute(statement)
            admin.execute(f"REVOKE ALL ON {_TABLE},{_KEY_TABLE},{_IDENTITY} FROM PUBLIC")
            for table in (_TABLE, _KEY_TABLE):
                admin.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
                admin.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
            for login, role in roles.items():
                if _role_state(admin, login) != [(None, True)]:
                    raise ValueError("journal roles must be fresh dedicated logins without elevated privileges")
                _install_role(admin, login, role, identity)
            [(security_definition,)] = _query_rows(admin, _SECURITY_DEFINITION, (_SCHEMA,))
            admin.execute(
                f"INSERT INTO {_IDENTITY} VALUES (1,%s,%s,%s,%s,%s)",
                (_VERSION, *astuple(identity), security_definition),
            )
        return identity

    def __init__(
        self,
        conninfo: str,
        identity: JournalIdentity,
        role: JournalRole,
        *,
        lock_timeout_ms: int = 250,
        statement_timeout_ms: int = 1000,
    ) -> None:
        if any(type(value) is not int or not 0 < value <= 2**31 - 1 for value in (lock_timeout_ms, statement_timeout_ms)):
            raise ValueError("journal timeouts must be positive PostgreSQL millisecond values")
        self._conninfo = conninfo
        self._identity = identity
        self._role = role
        self._lock_timeout_ms = lock_timeout_ms
        self._statement_timeout_ms = statement_timeout_ms
        with self._transaction():
            pass

    @contextmanager
    def _transaction(self) -> Iterator[Connection[Any]]:
        import psycopg

        connection = None
        body_completed = False
        try:
            with psycopg.connect(self._conninfo, autocommit=True, connect_timeout=3) as connection, connection.transaction():
                connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                connection.execute("SET LOCAL search_path = pg_catalog")
                _query_rows(
                    connection,
                    "SELECT set_config('lock_timeout',%s,true),set_config('statement_timeout',%s,true),"
                    "set_config('synchronous_commit','on',true)",
                    (f"{self._lock_timeout_ms}ms", f"{self._statement_timeout_ms}ms"),
                )
                login, effective_role = _query_rows(connection, "SELECT session_user,current_user")[0]
                if login != effective_role or _role_state(connection, login) != [(self._role.marker(self._identity), True)]:
                    raise ContentUnavailable("allocation journal role is unavailable")
                flags = _query_rows(
                    connection,
                    """
                    SELECT c.relrowsecurity,c.relforcerowsecurity,
                        has_schema_privilege(session_user,n.oid,'CREATE')
                    FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                    WHERE n.nspname=%s AND c.relname='allocation'
                    """,
                    (_SCHEMA,),
                )
                definition = _query_rows(connection, _SECURITY_DEFINITION, (_SCHEMA,))
                if flags != [(True, True, False)] or _query_rows(
                    connection, f"SELECT revision,journal_id,database_id,namespace,security_definition FROM {_IDENTITY}"
                ) != [(_VERSION, *astuple(self._identity), definition[0][0] if definition else None)]:
                    raise ContentUnavailable("allocation journal identity or policy does not match")
                yield connection
                body_completed = True
        except psycopg.Error as exc:
            state = exc.sqlstate
            # A failed COMMIT can have an unknown outcome. A generic query
            # cancellation (57014) also does not identify a lock timeout.
            if not body_completed and (
                state in {"08003", "08006", "40001", "40P01", "53300", "55P03", "57P01", "57P02", "57P03"}
                or isinstance(exc, psycopg.errors.ConnectionTimeout)
                or (connection is not None and connection.broken and state is None and isinstance(exc, psycopg.OperationalError))
            ):
                raise _JournalTemporarilyUnavailable(f"allocation journal is temporarily unavailable ({state})") from None
            raise ContentUnavailable(f"allocation journal is unavailable ({state})") from None

    def _allocations(self, rows: list[tuple[Any, ...]]) -> tuple[Allocation, ...]:
        return tuple(
            Allocation(
                self._identity.journal_id,
                allocation_id,
                state,
                pin,
                AllocationOwner(*owner) if owner[0] is not None else None,
                etag,
                delete_confirmed,
            )
            for allocation_id, state, pin, etag, delete_confirmed, *owner in rows
        )

    def _get(self, connection: Connection[Any], allocation_id: bytes, *, lock: bool = False) -> Allocation | None:
        suffix = " FOR UPDATE" if lock else ""
        rows = _query_rows(connection, f"SELECT {_COLUMNS} FROM {_TABLE} WHERE allocation_id=%s{suffix}", (allocation_id,))
        return self._allocations(rows)[0] if rows else None

    def get(self, allocation_id: bytes) -> Allocation | None:
        _require_id(allocation_id)
        with self._transaction() as connection:
            return self._get(connection, allocation_id)

    def _get_owned(self, allocation_id: bytes, owner: AllocationOwner) -> Allocation | None:
        _require_id(allocation_id)
        owner_match = " AND ".join(f"{column}=%s" for column in _OWNER.split(","))
        with self._transaction() as connection:
            rows = _query_rows(
                connection,
                f"SELECT {_COLUMNS} FROM {_TABLE} WHERE allocation_id=%s AND {owner_match}",
                (allocation_id, *astuple(owner)),
            )
            return self._allocations(rows)[0] if rows else None

    def _live_allocation_ms(self, expected: Allocation) -> int | None:
        _require_id(expected.allocation_id)
        with self._transaction() as connection:
            rows = _query_rows(
                connection,
                f"SELECT {_COLUMNS},pin_until_ms-({CLOCK}) FROM {_TABLE} WHERE allocation_id=%s",
                (expected.allocation_id,),
            )
            if not rows:
                return None
            (row,) = rows
            remaining_ms = row[-1]
            if remaining_ms <= 0 or self._allocations([row[:-1]])[0] != expected:
                return None
            return remaining_ms

    def reserve(self, allocation_id: bytes, *, pin_until_ms: int) -> bool:
        _require_id(allocation_id)
        if type(pin_until_ms) is not int or not 0 < pin_until_ms < 2**63:
            raise ValueError("allocation pin must be a positive PostgreSQL millisecond timestamp")
        with self._transaction() as connection:
            return (
                connection.execute(
                    f"INSERT INTO {_TABLE}(allocation_id,pin_until_ms) SELECT %s,%s WHERE %s>{CLOCK} "
                    "ON CONFLICT (allocation_id) DO NOTHING",
                    (allocation_id, pin_until_ms, pin_until_ms),
                ).rowcount
                == 1
            )

    def bind(self, allocation_id: bytes, owner: AllocationOwner, *, reservation_etag: str) -> bool:
        """Bind after a verified marker ACK while the caller owns the primary fence."""
        _require_id(allocation_id)
        _require_text(reservation_etag)
        if self._role != JournalRole("binder", owner.tenant_id):
            return False
        with self._transaction() as connection:
            # Test pin expiry after a conflicting row lock has released.
            if not _query_rows(
                connection,
                f"SELECT allocation_id FROM {_TABLE} WHERE allocation_id=%s AND state='unbound' FOR UPDATE",
                (allocation_id,),
            ):
                return False
            if owner.key_id.startswith(_EXACT_KEY_ID_PREFIX) and not _query_rows(
                connection,
                f"SELECT key_id FROM {_KEY_TABLE} WHERE provider_instance_id=%s AND key_id=%s "
                "AND tenant_id=%s AND session_id=%s AND admission_root_id=%s AND admission_id=%s "
                "AND state='allocating' FOR SHARE",
                (
                    owner.provider_instance_id,
                    owner.key_id,
                    owner.tenant_id,
                    owner.session_id,
                    owner.admission_session_id,
                    owner.admission_id,
                ),
            ):
                return False
            return (
                connection.execute(
                    f"UPDATE {_TABLE} SET state='bound',({_OWNER},reservation_etag)=(%s,%s,%s,%s,%s,%s,%s,%s) "
                    f"WHERE allocation_id=%s AND state='unbound' AND pin_until_ms>{CLOCK}",
                    (*astuple(owner), reservation_etag, allocation_id),
                ).rowcount
                == 1
            )

    def anonymous_candidates(self, *, limit: int, after: tuple[int, bytes] = (0, b"")) -> tuple[Allocation, ...]:
        if type(limit) is not int or not 0 < limit <= 2**31 - 1:
            raise ValueError("anonymous candidate limit must be positive")
        with self._transaction() as connection:
            return self._allocations(
                _query_rows(
                    connection,
                    f"SELECT {_COLUMNS} FROM {_TABLE} WHERE tenant_id IS NULL AND pin_until_ms<={CLOCK} "
                    "AND (pin_until_ms,allocation_id)>(%s,%s) ORDER BY pin_until_ms,allocation_id LIMIT %s",
                    (*after, limit),
                )
            )

    def expired_bound_candidates(self, *, limit: int, after: tuple[int, bytes] = (0, b"")) -> tuple[Allocation, ...]:
        """Return one cleanup-role page of expired tenant-bound journal rows."""
        if self._role.mode != "cleanup" or self._role.tenant_id is None:
            raise ContentCleanupUnavailable("bound allocation cleanup requires the tenant cleanup role")
        if type(limit) is not int or not 0 < limit <= _BOUND_PAGE_LIMIT:
            raise ValueError(f"bound allocation candidate limit must be between 1 and {_BOUND_PAGE_LIMIT}")
        if (
            not isinstance(after, tuple)
            or len(after) != 2
            or type(after[0]) is not int
            or after[0] < 0
            or not isinstance(after[1], bytes)
            or len(after[1]) not in {0, 32}
        ):
            raise ValueError("bound allocation candidate cursor is invalid")
        with self._transaction() as connection:
            return self._allocations(
                _query_rows(
                    connection,
                    f"SELECT {_COLUMNS} FROM {_TABLE} WHERE tenant_id=%s "
                    "AND state IN ('bound','reaping') "
                    f"AND pin_until_ms<=(SELECT {CLOCK}) AND (pin_until_ms,allocation_id)>(%s,%s) "
                    "ORDER BY pin_until_ms,allocation_id LIMIT %s",
                    (self._role.tenant_id, *after, limit),
                )
            )

    def scoped_allocations(self, tenant_id: str, session_ids: tuple[str, ...]) -> tuple[Allocation, ...]:
        _require_text(tenant_id, *session_ids)
        with self._transaction() as connection:
            return self._allocations(
                _query_rows(
                    connection,
                    f"SELECT {_COLUMNS} FROM {_TABLE} WHERE tenant_id=%s "
                    "AND (session_id=ANY(%s) OR admission_session_id=ANY(%s)) ORDER BY allocation_id",
                    (tenant_id, list(session_ids), list(session_ids)),
                )
            )

    def exact_key_allocations_exist(self, expected: KeyIntent) -> bool:
        """Return whether one exact key still owns any journal allocation."""
        if expected.journal_id != self._identity.journal_id or self._role != JournalRole("cleanup", expected.intent.tenant_id):
            raise ContentCleanupUnavailable("exact key allocation lookup requires the tenant cleanup role")
        with self._transaction() as connection:
            return bool(
                _query_rows(
                    connection,
                    f"SELECT 1 FROM {_TABLE} WHERE provider_instance_id=%s AND key_id=%s LIMIT 1",
                    (expected.intent.provider_instance_id, expected.intent.key_id),
                )
            )

    def key_intent_pin_expired(self, expected: KeyIntent) -> bool:
        """Read exact intent expiry from the journal database clock."""
        if expected.journal_id != self._identity.journal_id or self._role != JournalRole("cleanup", expected.intent.tenant_id):
            raise ContentCleanupUnavailable("exact key expiry lookup requires the tenant cleanup role")
        owner_match = " AND ".join(f"{column}=%s" for column in _KEY_OWNER.split(","))
        with self._transaction() as connection:
            return bool(
                _query_rows(
                    connection,
                    f"SELECT 1 FROM {_KEY_TABLE} WHERE {owner_match} AND pin_until_ms<={CLOCK} LIMIT 1",
                    astuple(expected.intent),
                )
            )

    def claim_anonymous(self, allocation_id: bytes) -> Allocation | None:
        _require_id(allocation_id)
        with self._transaction() as connection:
            connection.execute(
                f"UPDATE {_TABLE} SET state='reaping' WHERE allocation_id=%s AND state='unbound' AND pin_until_ms<={CLOCK}",
                (allocation_id,),
            )
            claimed = self._get(connection, allocation_id)
            if claimed is not None and claimed.state == "reaping" and claimed.owner is None:
                return claimed
            return None

    def claim_bound(self, expected: Allocation, *, require_expired_pin: bool = False) -> Allocation | None:
        """Claim one exact binding under its caller's primary fence.

        Orphan maintenance requires an expired pin; authorized root erasure keeps
        the default so it can claim a live allocation inside its deletion path.
        """
        _require_id(expected.allocation_id)
        if (
            expected.journal_id != self._identity.journal_id
            or expected.state == "unbound"
            or expected.owner is None
            or self._role != JournalRole("cleanup", expected.owner.tenant_id)
        ):
            return None
        with self._transaction() as connection:
            actual = self._get(connection, expected.allocation_id, lock=expected.state == "bound")
            if actual != expected:
                return None
            if require_expired_pin and _query_rows(connection, f"SELECT {CLOCK}>=%s", (expected.pin_until_ms,)) != [(True,)]:
                return None
            if expected.state == "reaping":
                return actual
            return self._allocations(
                _query_rows(
                    connection,
                    f"UPDATE {_TABLE} SET state='reaping' WHERE allocation_id=%s RETURNING {_COLUMNS}",
                    (expected.allocation_id,),
                )
            )[0]

    def confirm_bound_delete(self, claim: Allocation) -> Allocation:
        """Persist terminal provider evidence for the exact claim under the primary fence."""
        _require_id(claim.allocation_id)
        if (
            claim.state != "reaping"
            or claim.owner is None
            or claim.journal_id != self._identity.journal_id
            or type(claim.delete_confirmed) is not bool
            or self._role != JournalRole("cleanup", claim.owner.tenant_id)
        ):
            raise ContentCleanupUnavailable("terminal bound deletion requires the exact cleanup claim")
        confirmed = replace(claim, delete_confirmed=True)
        with self._transaction() as connection:
            rows = _query_rows(
                connection,
                f"UPDATE {_TABLE} SET delete_confirmed=true WHERE {_CLAIM_MATCH} "
                "AND NOT delete_confirmed AND NOT %s "
                f"RETURNING {_COLUMNS}",
                (claim.allocation_id, claim.pin_until_ms, claim.reservation_etag, *astuple(claim.owner), claim.delete_confirmed),
            )
            if rows:
                return self._allocations(rows)[0]
            if self._get(connection, claim.allocation_id) == confirmed:
                return confirmed
        raise ContentCleanupUnavailable("terminal bound deletion claim changed")

    def finish_reap(self, claim: Allocation, *, remote_terminal: bool) -> bool:
        _require_id(claim.allocation_id)
        if remote_terminal is not True:
            raise ContentCleanupUnavailable("allocation erasure has no terminal remote evidence")
        if claim.state != "reaping" or claim.journal_id != self._identity.journal_id:
            return False
        if claim.owner is None:
            if self._role != JournalRole("anonymous") or claim.delete_confirmed is not False:
                return False
            owner = (None,) * 7
        else:
            if self._role != JournalRole("cleanup", claim.owner.tenant_id) or claim.delete_confirmed is not True:
                return False
            owner = astuple(claim.owner)
        with self._transaction() as connection:
            return (
                connection.execute(
                    f"DELETE FROM {_TABLE} WHERE {_CLAIM_MATCH} AND delete_confirmed=%s",
                    (claim.allocation_id, claim.pin_until_ms, claim.reservation_etag, *owner, claim.delete_confirmed),
                ).rowcount
                == 1
            )

    def _key_rows(self, rows: list[tuple[Any, ...]]) -> tuple[KeyIntent, ...]:
        return tuple(
            KeyIntent(self._identity.journal_id, _ExactKeyIntent(*owner), state, pin, evidence)
            for (*owner, state, pin, evidence) in rows
        )

    def _get_key(self, connection: Connection[Any], provider: str, key_id: str, *, lock: bool = False) -> KeyIntent | None:
        suffix = " FOR UPDATE" if lock else ""
        rows = self._key_rows(
            _query_rows(
                connection,
                f"SELECT {_KEY_COLUMNS} FROM {_KEY_TABLE} WHERE provider_instance_id=%s AND key_id=%s{suffix}",
                (provider, key_id),
            )
        )
        return rows[0] if rows else None

    def get_key_intent(self, provider_instance_id: str, key_id: str) -> KeyIntent | None:
        _require_text(provider_instance_id, key_id)
        with self._transaction() as connection:
            return self._get_key(connection, provider_instance_id, key_id)

    def reserve_key_intent(self, intent: _ExactKeyIntent, *, pin_until_ms: int) -> KeyIntent | None:
        if type(pin_until_ms) is not int or not 0 < pin_until_ms < 2**63:
            raise ValueError("key intent pin must be a positive PostgreSQL millisecond timestamp")
        with self._transaction() as connection:
            rows = self._key_rows(
                _query_rows(
                    connection,
                    f"INSERT INTO {_KEY_TABLE}({_KEY_OWNER},pin_until_ms) SELECT %s,%s,%s,%s,%s,%s,%s "
                    f"WHERE %s>{CLOCK} ON CONFLICT DO NOTHING RETURNING {_KEY_COLUMNS}",
                    (*astuple(intent), pin_until_ms, pin_until_ms),
                )
            )
            return rows[0] if rows else None

    def key_intents_for_admission(
        self, tenant_id: str, session_id: str, admission_root_id: str, admission_id: str
    ) -> tuple[KeyIntent, ...]:
        _require_text(tenant_id, session_id, admission_root_id, admission_id)
        with self._transaction() as connection:
            return self._key_rows(
                _query_rows(
                    connection,
                    f"SELECT {_KEY_COLUMNS} FROM {_KEY_TABLE} WHERE tenant_id=%s AND session_id=%s "
                    "AND admission_root_id=%s AND admission_id=%s ORDER BY provider_instance_id,key_id",
                    (tenant_id, session_id, admission_root_id, admission_id),
                )
            )

    def scoped_key_intents(self, tenant_id: str, session_ids: tuple[str, ...]) -> tuple[KeyIntent, ...]:
        _require_text(tenant_id, *session_ids)
        with self._transaction() as connection:
            return self._key_rows(
                _query_rows(
                    connection,
                    f"SELECT {_KEY_COLUMNS} FROM {_KEY_TABLE} WHERE tenant_id=%s "
                    "AND (session_id=ANY(%s) OR admission_root_id=ANY(%s)) ORDER BY provider_instance_id,key_id",
                    (tenant_id, list(session_ids), list(session_ids)),
                )
            )

    def claim_key_allocation(self, expected: KeyIntent) -> KeyIntent | None:
        """Only an acknowledged transition authorizes one provider dispatch."""
        if expected.journal_id != self._identity.journal_id or expected.state != "reserved":
            return None
        provider, key_id = expected.intent.provider_instance_id, expected.intent.key_id
        with self._transaction() as connection:
            if self._get_key(connection, provider, key_id, lock=True) != expected:
                return None
            rows = self._key_rows(
                _query_rows(
                    connection,
                    f"UPDATE {_KEY_TABLE} SET state='allocating' WHERE provider_instance_id=%s AND key_id=%s "
                    f"AND pin_until_ms>{CLOCK} RETURNING {_KEY_COLUMNS}",
                    (provider, key_id),
                )
            )
            return rows[0] if rows else None

    def claim_key_intent(self, expected: KeyIntent, *, require_expired_pin: bool = True) -> KeyIntent | None:
        """The trusted reconciler must hold the primary fence and verify SQL disposition."""
        if expected.journal_id != self._identity.journal_id or expected.state == "retired":
            return None
        provider, key_id = expected.intent.provider_instance_id, expected.intent.key_id
        with self._transaction() as connection:
            if self._get_key(connection, provider, key_id, lock=True) != expected:
                return None
            if expected.state == "reaping":
                return expected
            rows = self._key_rows(
                _query_rows(
                    connection,
                    f"UPDATE {_KEY_TABLE} SET state='reaping' WHERE provider_instance_id=%s AND key_id=%s "
                    f"AND (NOT %s OR pin_until_ms<={CLOCK}) RETURNING {_KEY_COLUMNS}",
                    (provider, key_id, require_expired_pin),
                )
            )
            return rows[0] if rows else None

    def record_key_retirement(self, claim: KeyIntent, outcome: KeyDestroyOutcome) -> KeyIntent | None:
        if outcome.status not in {KeyDestroyStatus.DESTROYED, KeyDestroyStatus.ALREADY_DESTROYED} or not outcome.evidence_id:
            raise ContentCleanupUnavailable("exact key retirement has no terminal provider evidence")
        if claim.journal_id != self._identity.journal_id or claim.state != "reaping":
            return None
        provider, key_id = claim.intent.provider_instance_id, claim.intent.key_id
        retired = replace(claim, state="retired", evidence_id=outcome.evidence_id)
        with self._transaction() as connection:
            if self._get_key(connection, provider, key_id, lock=True) != claim:
                current = self._get_key(connection, provider, key_id)
                return current if current == retired else None
            rows = self._key_rows(
                _query_rows(
                    connection,
                    f"UPDATE {_KEY_TABLE} SET state='retired',evidence_id=%s "
                    f"WHERE provider_instance_id=%s AND key_id=%s RETURNING {_KEY_COLUMNS}",
                    (outcome.evidence_id, provider, key_id),
                )
            )
            return rows[0] if rows else None

    def finish_key_intent(self, retired: KeyIntent) -> bool:
        """Release exact scope metadata after the key retires and its objects are gone."""
        if retired.journal_id != self._identity.journal_id or retired.state != "retired":
            return False
        provider, key_id = retired.intent.provider_instance_id, retired.intent.key_id
        owner_match = " AND ".join(f"{column}=%s" for column in _KEY_OWNER.split(","))
        with self._transaction() as connection:
            return (
                connection.execute(
                    f"DELETE FROM {_KEY_TABLE} WHERE {owner_match} AND state='retired' AND pin_until_ms=%s AND evidence_id=%s "
                    f"AND NOT EXISTS (SELECT 1 FROM {_TABLE} WHERE provider_instance_id=%s AND exact_key_id=%s)",
                    (*astuple(retired.intent), retired.pin_until_ms, retired.evidence_id, provider, key_id),
                ).rowcount
                == 1
            )


class PostgresObjectJournal:
    """Compose object and key allocation roles for one authenticated tenant."""

    def __init__(
        self,
        *,
        anonymous: PostgresAllocationJournal,
        binder: PostgresAllocationJournal,
        scope: PostgresAllocationJournal,
        keys: PostgresAllocationJournal,
    ) -> None:
        tenant_id = scope._role.tenant_id
        if (
            tenant_id is None
            or scope._role != JournalRole("scope", tenant_id)
            or binder._role != JournalRole("binder", tenant_id)
            or keys._role != JournalRole("key-intent", tenant_id)
            or anonymous._role != JournalRole("anonymous")
            or not (scope._identity == binder._identity == anonymous._identity == keys._identity)
        ):
            raise ValueError("object journal requires matching provisioned role bindings")
        self._tenant_id = tenant_id
        self._anonymous = anonymous
        self._binder = binder
        self._scope = scope
        self._keys = keys

    @property
    def identity(self) -> JournalIdentity:
        return self._scope._identity

    def verify_identity(self) -> JournalIdentity:
        for role in (self._anonymous, self._binder, self._scope, self._keys):
            with role._transaction():
                pass
        return self.identity

    def _pin_until_ms(self, ttl_ms: int) -> int:
        with self._scope._transaction() as connection:
            return _query_rows(connection, f"SELECT {CLOCK}+%s", (ttl_ms,))[0][0]

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    def get_key_intent(self, provider_instance_id: str, key_id: str) -> KeyIntent | None:
        return self._scope.get_key_intent(provider_instance_id, key_id)

    def key_intents_for_admission(
        self, tenant_id: str, session_id: str, admission_root_id: str, admission_id: str
    ) -> tuple[KeyIntent, ...]:
        if tenant_id != self._tenant_id:
            return ()
        return self._scope.key_intents_for_admission(tenant_id, session_id, admission_root_id, admission_id)

    def reserve_key_intent(self, intent: _ExactKeyIntent, *, pin_until_ms: int) -> KeyIntent | None:
        if intent.tenant_id != self._tenant_id:
            return None
        return self._keys.reserve_key_intent(intent, pin_until_ms=pin_until_ms)

    def claim_key_allocation(self, expected: KeyIntent) -> KeyIntent | None:
        if expected.intent.tenant_id != self._tenant_id:
            return None
        return self._keys.claim_key_allocation(expected)

    def reserve(self, allocation_id: bytes, *, pin_until_ms: int) -> bool:
        return self._anonymous.reserve(allocation_id, pin_until_ms=pin_until_ms)

    def bind(self, allocation_id: bytes, owner: AllocationOwner, *, reservation_etag: str) -> bool:
        if owner.tenant_id != self._tenant_id or not owner.key_id.startswith(_EXACT_KEY_ID_PREFIX):
            return False
        return self._binder.bind(allocation_id, owner, reservation_etag=reservation_etag)

    def get_anonymous(self, allocation_id: bytes) -> Allocation | None:
        row = self._anonymous.get(allocation_id)
        return row if row is not None and row.owner is None else None

    def get_owned(self, allocation_id: bytes, owner: AllocationOwner) -> Allocation | None:
        if owner.tenant_id != self._tenant_id:
            return None
        return self._scope._get_owned(allocation_id, owner)

    def _live_allocation_ms(self, expected: Allocation) -> int | None:
        if expected.owner is None:
            return self._anonymous._live_allocation_ms(expected)
        if expected.owner.tenant_id != self._tenant_id:
            return None
        return self._scope._live_allocation_ms(expected)
