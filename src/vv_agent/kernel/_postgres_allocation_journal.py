"""Independent PostgreSQL journal for S1 object reservations."""

from __future__ import annotations

import json
import secrets
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import astuple, dataclass
from typing import TYPE_CHECKING, Any, Literal

from ._allocation_journal import Allocation, AllocationOwner, JournalIdentity, _require_id, _require_text
from ._content import ContentCleanupUnavailable, ContentUnavailable
from ._postgres import CLOCK, _query_rows

if TYPE_CHECKING:
    from psycopg import Connection

_SCHEMA = "vv_allocation_journal"
_TABLE = f"{_SCHEMA}.allocation"
_IDENTITY = f"{_SCHEMA}.identity"
_OWNER = "tenant_id,session_id,content_id,admission_session_id,admission_id,provider_instance_id,key_id"
_COLUMNS = f"allocation_id,state,pin_until_ms,reservation_etag,{_OWNER}"
_VERSION = 1

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
        CHECK (
            (state IN ('unbound','reaping') AND num_nonnulls({_OWNER},reservation_etag) = 0)
            OR
            (state IN ('bound','reaping') AND num_nulls({_OWNER},reservation_etag) = 0)
        )
    )""",
    f"CREATE INDEX allocation_expiry ON {_TABLE}(pin_until_ms,allocation_id) WHERE tenant_id IS NULL",
    f"CREATE INDEX allocation_owner ON {_TABLE}(tenant_id,session_id)",
    f"CREATE INDEX allocation_admission ON {_TABLE}(tenant_id,admission_session_id)",
)

_SECURITY_DEFINITION = """
SELECT jsonb_build_array(
    n.nspacl,
    (SELECT jsonb_agg(jsonb_build_array(c.relname,c.relowner,c.relacl,c.relrowsecurity,c.relforcerowsecurity)
                     ORDER BY c.relname)
     FROM pg_class c WHERE c.relnamespace=n.oid AND c.relkind='r'),
    (SELECT jsonb_agg(jsonb_build_array(a.attname,a.attacl) ORDER BY a.attname)
     FROM pg_attribute a WHERE a.attrelid=allocation.oid AND a.attnum>0 AND NOT a.attisdropped),
    (SELECT jsonb_agg(jsonb_build_array(p.polname,p.polcmd,p.polpermissive,p.polroles,
                    pg_get_expr(p.polqual,p.polrelid),pg_get_expr(p.polwithcheck,p.polrelid)) ORDER BY p.polname)
     FROM pg_policy p WHERE p.polrelid=allocation.oid)
)::text
FROM pg_class allocation JOIN pg_namespace n ON n.oid=allocation.relnamespace
WHERE n.nspname=%s AND allocation.relname='allocation' AND allocation.relkind='r'
"""


@dataclass(frozen=True, slots=True)
class JournalRole:
    mode: Literal["anonymous", "binder", "scope"]
    tenant_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "scope" and self.tenant_id is not None:
            _require_text(self.tenant_id)
        elif self.tenant_id is not None or self.mode not in {"anonymous", "binder"}:
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
    if role.mode == "scope":
        visible = sql.SQL("{} AND tenant_id = {}").format(authority, sql.Literal(role.tenant_id))
        grants = ("SELECT", "UPDATE(state)", "DELETE")
        transitions = (
            ("SELECT", visible, None),
            ("UPDATE", visible + sql.SQL(" AND state='bound'"), visible + sql.SQL(" AND state='reaping'")),
            ("DELETE", visible + sql.SQL(" AND state='reaping'"), None),
        )
    elif role.mode == "binder":
        visible = authority + sql.SQL(" AND tenant_id IS NULL")
        grants = ("SELECT(allocation_id,state,pin_until_ms)", f"UPDATE(state,reservation_etag,{_OWNER})")
        # PostgreSQL checks SELECT visibility on the new row as well as the old one.
        transitions = (
            ("SELECT", authority, None),
            (
                "UPDATE",
                visible + sql.SQL(f" AND state='unbound' AND pin_until_ms>{CLOCK}"),
                authority + sql.SQL(f" AND state='bound' AND pin_until_ms>{CLOCK}"),
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
    for grant in grants:
        connection.execute(sql.SQL(f"GRANT {grant} ON {_TABLE} TO {{}}").format(principal))
    for command, using, check in transitions:
        policy = sql.Identifier(login + "_" + command.lower())
        statement = sql.SQL(f"CREATE POLICY {{}} ON {_TABLE} FOR {command} TO {{}}").format(policy, principal)
        if using is not None:
            statement += sql.SQL(" USING ({})").format(using)
        if check is not None:
            statement += sql.SQL(" WITH CHECK ({})").format(check)
        connection.execute(statement)


class PostgresAllocationJournal:
    """One journal authority per independent connection, pinned to a provisioned identity."""

    @staticmethod
    def provision(
        admin: Connection[Any], *, database_id: str, namespace: str, roles: Mapping[str, JournalRole]
    ) -> JournalIdentity:
        """Install once in a dedicated database using fresh, pre-created logins."""
        from psycopg.pq import TransactionStatus

        if admin.info.transaction_status != TransactionStatus.IDLE:
            raise ValueError("journal provisioning requires an idle admin connection")
        if {role.mode for role in roles.values()} != {"anonymous", "binder", "scope"}:
            raise ValueError("journal provisioning requires all three authorities")
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
            admin.execute(f"REVOKE ALL ON {_TABLE},{_IDENTITY} FROM PUBLIC")
            admin.execute(f"ALTER TABLE {_TABLE} ENABLE ROW LEVEL SECURITY")
            admin.execute(f"ALTER TABLE {_TABLE} FORCE ROW LEVEL SECURITY")
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
        except psycopg.Error as exc:
            raise ContentUnavailable(f"allocation journal is unavailable ({exc.sqlstate})") from None

    def _allocations(self, rows: list[tuple[Any, ...]]) -> tuple[Allocation, ...]:
        return tuple(
            Allocation(
                self._identity.journal_id,
                allocation_id,
                state,
                pin,
                AllocationOwner(*owner) if owner[0] is not None else None,
                etag,
            )
            for allocation_id, state, pin, etag, *owner in rows
        )

    def _get(self, connection: Connection[Any], allocation_id: bytes, *, lock: bool = False) -> Allocation | None:
        suffix = " FOR UPDATE" if lock else ""
        rows = _query_rows(connection, f"SELECT {_COLUMNS} FROM {_TABLE} WHERE allocation_id=%s{suffix}", (allocation_id,))
        return self._allocations(rows)[0] if rows else None

    def get(self, allocation_id: bytes) -> Allocation | None:
        _require_id(allocation_id)
        with self._transaction() as connection:
            return self._get(connection, allocation_id)

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
        with self._transaction() as connection:
            # Test pin expiry after a conflicting row lock has released.
            if not _query_rows(
                connection,
                f"SELECT allocation_id FROM {_TABLE} WHERE allocation_id=%s AND state='unbound' FOR UPDATE",
                (allocation_id,),
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

    def claim_bound(self, expected: Allocation) -> Allocation | None:
        _require_id(expected.allocation_id)
        if expected.journal_id != self._identity.journal_id or expected.state == "unbound" or expected.owner is None:
            return None
        with self._transaction() as connection:
            # Reaping rows cannot be row-locked by this role's UPDATE policy.
            actual = self._get(connection, expected.allocation_id, lock=expected.state == "bound")
            if actual != expected:
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

    def finish_reap(self, claim: Allocation, *, remote_terminal: bool) -> bool:
        _require_id(claim.allocation_id)
        if remote_terminal is not True:
            raise ContentCleanupUnavailable("allocation erasure has no terminal remote evidence")
        if claim.state != "reaping" or claim.journal_id != self._identity.journal_id:
            return False
        owner = astuple(claim.owner) if claim.owner is not None else (None,) * 7
        owner_match = " AND ".join(f"{column} IS NOT DISTINCT FROM %s" for column in _OWNER.split(","))
        with self._transaction() as connection:
            return (
                connection.execute(
                    f"DELETE FROM {_TABLE} WHERE allocation_id=%s AND state='reaping' AND pin_until_ms=%s "
                    f"AND reservation_etag IS NOT DISTINCT FROM %s AND {owner_match}",
                    (claim.allocation_id, claim.pin_until_ms, claim.reservation_etag, *owner),
                ).rowcount
                == 1
            )
