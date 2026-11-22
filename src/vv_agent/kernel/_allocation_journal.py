"""SQLite sidecar for reservations and exact external content ownership."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Literal

from ._content import (
    _EXACT_KEY_ID_PREFIX,
    ContentBindingMismatch,
    ContentCleanupUnavailable,
    ContentUnavailable,
    KeyDestroyOutcome,
    KeyDestroyStatus,
    _ExactKeyIntent,
)

_VERSION = 4
_OWNER_COLUMNS = """
    tenant_id, session_id, content_id, admission_session_id, admission_id, provider_instance_id, key_id
"""
_ALLOCATION_COLUMNS = f"allocation_id,state,pin_until_ms,reservation_etag,delete_confirmed,{_OWNER_COLUMNS}"
_KEY_OWNER_COLUMNS = "key_id,provider_instance_id,tenant_id,session_id,admission_root_id,admission_id"
_KEY_INTENT_COLUMNS = f"{_KEY_OWNER_COLUMNS},state,pin_until_ms,evidence_id"
_NOW_MS = "(CAST(strftime('%s','now') AS INTEGER)*1000 + CAST(substr(strftime('%f','now'),4) AS INTEGER))"
_CHECK_EXACT_BINDING = f"""WHEN substr(NEW.key_id,1,{len(_EXACT_KEY_ID_PREFIX)}) = '{_EXACT_KEY_ID_PREFIX}'
        AND NOT EXISTS (
            SELECT 1 FROM key_intent
            WHERE provider_instance_id=NEW.provider_instance_id AND key_id=NEW.key_id
              AND tenant_id=NEW.tenant_id AND session_id=NEW.session_id
              AND admission_root_id=NEW.admission_session_id AND admission_id=NEW.admission_id
              AND state='allocating'
        )
        BEGIN
            SELECT RAISE(ABORT, 'exact key is unavailable for binding');
        END"""
_SCHEMA = (
    """CREATE TABLE journal_identity (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        journal_id TEXT NOT NULL,
        database_id TEXT NOT NULL,
        namespace TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE allocation (
        allocation_id BLOB PRIMARY KEY NOT NULL CHECK (length(allocation_id) = 32),
        state TEXT NOT NULL CHECK (state IN ('unbound','bound','reaping')),
        pin_until_ms INTEGER NOT NULL CHECK (pin_until_ms > 0),
        tenant_id TEXT CHECK (length(tenant_id) > 0),
        session_id TEXT CHECK (length(session_id) > 0),
        content_id TEXT CHECK (length(content_id) > 0),
        admission_session_id TEXT CHECK (length(admission_session_id) > 0),
        admission_id TEXT CHECK (length(admission_id) > 0),
        provider_instance_id TEXT CHECK (length(provider_instance_id) > 0),
        key_id TEXT CHECK (length(key_id) > 0),
        reservation_etag TEXT CHECK (length(reservation_etag) > 0),
        delete_confirmed INTEGER NOT NULL DEFAULT 0 CHECK (
            delete_confirmed IN (0,1) AND (delete_confirmed=0 OR (state='reaping' AND tenant_id IS NOT NULL))
        ),
        CHECK (
            (state IN ('unbound','reaping') AND tenant_id IS NULL AND session_id IS NULL
             AND content_id IS NULL AND admission_session_id IS NULL AND admission_id IS NULL
             AND provider_instance_id IS NULL AND key_id IS NULL AND reservation_etag IS NULL)
            OR
            (state IN ('bound','reaping') AND tenant_id IS NOT NULL AND session_id IS NOT NULL
             AND content_id IS NOT NULL AND admission_session_id IS NOT NULL AND admission_id IS NOT NULL
             AND provider_instance_id IS NOT NULL AND key_id IS NOT NULL AND reservation_etag IS NOT NULL)
        )
    ) STRICT""",
    "CREATE INDEX allocation_expiry ON allocation(pin_until_ms, allocation_id) WHERE tenant_id IS NULL",
    "CREATE INDEX allocation_owner ON allocation(tenant_id, session_id)",
    "CREATE INDEX allocation_admission ON allocation(tenant_id, admission_session_id)",
    "CREATE INDEX allocation_key ON allocation(provider_instance_id, key_id) WHERE key_id IS NOT NULL",
    """CREATE TABLE key_intent (
        key_id TEXT NOT NULL CHECK (
            substr(key_id,1,13) = 's1-exact-key-' AND length(key_id) = 77
            AND substr(key_id,14) NOT GLOB '*[^0-9a-f]*'
        ),
        provider_instance_id TEXT NOT NULL CHECK (length(provider_instance_id) > 0),
        tenant_id TEXT NOT NULL CHECK (length(tenant_id) > 0),
        session_id TEXT NOT NULL CHECK (length(session_id) > 0),
        admission_root_id TEXT NOT NULL CHECK (length(admission_root_id) > 0),
        admission_id TEXT NOT NULL CHECK (length(admission_id) > 0),
        state TEXT NOT NULL CHECK (state IN ('reserved','allocating','reaping','retired')),
        pin_until_ms INTEGER NOT NULL CHECK (pin_until_ms > 0),
        evidence_id TEXT CHECK (length(evidence_id) > 0),
        PRIMARY KEY (provider_instance_id, key_id),
        CHECK ((state = 'retired') = (evidence_id IS NOT NULL))
    ) STRICT""",
    "CREATE UNIQUE INDEX key_intent_admission ON key_intent(tenant_id, session_id, admission_root_id, admission_id) "
    "WHERE state <> 'retired'",
    "CREATE INDEX key_intent_owner ON key_intent(tenant_id, session_id)",
    "CREATE INDEX key_intent_admission_root ON key_intent(tenant_id, admission_root_id)",
    f"CREATE TRIGGER validate_exact_binding_insert BEFORE INSERT ON allocation {_CHECK_EXACT_BINDING}",
    f"CREATE TRIGGER validate_exact_binding_update BEFORE UPDATE OF {_OWNER_COLUMNS} ON allocation {_CHECK_EXACT_BINDING}",
)
_EXPECTED_SCHEMA = frozenset(_SCHEMA)


@dataclass(frozen=True, slots=True)
class JournalIdentity:
    journal_id: str
    database_id: str
    namespace: str

    def __post_init__(self) -> None:
        _require_text(*astuple(self))


@dataclass(frozen=True, slots=True)
class AllocationOwner:
    tenant_id: str
    session_id: str
    content_id: str
    admission_session_id: str
    admission_id: str
    provider_instance_id: str
    key_id: str

    def __post_init__(self) -> None:
        _require_text(*astuple(self))


@dataclass(frozen=True, slots=True)
class Allocation:
    journal_id: str
    allocation_id: bytes
    state: Literal["unbound", "bound", "reaping"]
    pin_until_ms: int
    owner: AllocationOwner | None = None
    reservation_etag: str | None = None
    delete_confirmed: bool = False


@dataclass(frozen=True, slots=True)
class KeyIntent:
    journal_id: str
    intent: _ExactKeyIntent
    state: Literal["reserved", "allocating", "reaping", "retired"]
    pin_until_ms: int
    evidence_id: str | None = None


def _require_text(*values: str) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("journal identifiers and ETags must be non-empty text")


def _require_id(allocation_id: bytes) -> None:
    if not isinstance(allocation_id, bytes) or len(allocation_id) != 32:
        raise ValueError("allocation ID must be 32 bytes")


def _require_pin(pin_until_ms: int) -> None:
    if type(pin_until_ms) is not int or not 0 < pin_until_ms < 2**63:
        raise ValueError("allocation pin must be a positive SQLite millisecond timestamp")


def _check_owner_only(path: Path, *, directory: bool = False) -> None:
    info = path.lstat()
    matches_type = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not matches_type or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ContentUnavailable("allocation journal requires owner-only storage")


class SQLiteAllocationJournal:
    """Trusted internal journal. The caller supplies the primary fence and currentness proof."""

    def __init__(self, path: Path, identity: JournalIdentity, *, busy_timeout_ms: int = 250) -> None:
        if type(busy_timeout_ms) is not int or busy_timeout_ms < 0:
            raise ValueError("journal busy timeout must be a nonnegative number of milliseconds")
        self._path = path.absolute()
        self._identity = identity
        self._busy_timeout_ms = busy_timeout_ms
        with self._transaction() as connection:
            if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ContentUnavailable("allocation journal integrity check failed")

    @classmethod
    def provision(cls, path: Path, *, database_id: str, namespace: str) -> JournalIdentity:
        identity = JournalIdentity(secrets.token_hex(32), database_id, namespace)
        path = path.absolute()
        _check_owner_only(path.parent, directory=True)
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)) as connection:
            if connection.execute("PRAGMA journal_mode=WAL").fetchone() != ("wal",):
                raise ContentUnavailable("allocation journal requires WAL")
            connection.execute("PRAGMA synchronous=FULL")
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                for statement in _SCHEMA:
                    connection.execute(statement)
                connection.execute("INSERT INTO journal_identity VALUES (1,?,?,?)", astuple(identity))
                connection.execute(f"PRAGMA user_version={_VERSION}")
        return identity

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        try:
            _check_owner_only(self._path.parent, directory=True)
            _check_owner_only(self._path)
            with closing(
                sqlite3.connect(self._path.as_uri() + "?mode=rw", uri=True, timeout=self._busy_timeout_ms / 1000)
            ) as connection:
                if connection.execute("PRAGMA journal_mode").fetchone() != ("wal",):
                    raise ContentUnavailable("allocation journal requires WAL")
                connection.execute("PRAGMA synchronous=FULL")
                with connection:
                    connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                    version = connection.execute("PRAGMA user_version").fetchone()
                    schema = connection.execute("SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL").fetchall()
                    if version != (_VERSION,) or {row[0] for row in schema} != _EXPECTED_SCHEMA:
                        raise ContentUnavailable("allocation journal schema does not match")
                    rows = connection.execute("SELECT journal_id,database_id,namespace FROM journal_identity").fetchall()
                    if rows != [astuple(self._identity)]:
                        raise ContentUnavailable("allocation journal identity does not match")
                    yield connection
        except (OSError, sqlite3.Error) as exc:
            raise ContentUnavailable("allocation journal is unavailable") from exc

    def _allocation(self, row: tuple) -> Allocation:
        allocation_id, state, pin_until_ms, etag, delete_confirmed, *owner = row
        return Allocation(
            self._identity.journal_id,
            allocation_id,
            state,
            pin_until_ms,
            AllocationOwner(*owner) if owner[0] is not None else None,
            etag,
            bool(delete_confirmed),
        )

    def _get(self, connection: sqlite3.Connection, allocation_id: bytes) -> Allocation | None:
        row = connection.execute(
            f"SELECT {_ALLOCATION_COLUMNS} FROM allocation WHERE allocation_id=?",
            (allocation_id,),
        ).fetchone()
        return self._allocation(row) if row is not None else None

    def get(self, allocation_id: bytes) -> Allocation | None:
        _require_id(allocation_id)
        with self._transaction() as connection:
            return self._get(connection, allocation_id)

    @property
    def identity(self) -> JournalIdentity:
        return self._identity

    def get_anonymous(self, allocation_id: bytes) -> Allocation | None:
        row = self.get(allocation_id)
        return row if row is not None and row.owner is None else None

    def get_owned(self, allocation_id: bytes, owner: AllocationOwner) -> Allocation | None:
        row = self.get(allocation_id)
        if row is None or row.owner is None:
            return None
        if row.owner != owner:
            raise ContentBindingMismatch("external ciphertext locator does not match its exact owner")
        return row

    def reserve(self, allocation_id: bytes, *, pin_until_ms: int) -> bool:
        """Commit a newly generated random ID before dispatching its first PUT."""
        _require_id(allocation_id)
        _require_pin(pin_until_ms)
        with self._transaction(write=True) as connection:
            return (
                connection.execute(
                    f"INSERT INTO allocation(allocation_id,state,pin_until_ms) SELECT ?,'unbound',? "
                    f"WHERE ?>{_NOW_MS} ON CONFLICT(allocation_id) DO NOTHING",
                    (allocation_id, pin_until_ms, pin_until_ms),
                ).rowcount
                == 1
            )

    def bind(self, allocation_id: bytes, owner: AllocationOwner, *, reservation_etag: str) -> bool:
        """Bind only a live anonymous reservation while holding the primary publication fence."""
        _require_id(allocation_id)
        _require_text(reservation_etag)
        with self._transaction(write=True) as connection:
            if owner.key_id.startswith(_EXACT_KEY_ID_PREFIX):
                key = self._get_key_intent(connection, owner.provider_instance_id, owner.key_id)
                if (
                    key is None
                    or key.state != "allocating"
                    or (key.intent.tenant_id, key.intent.session_id, key.intent.admission_root_id, key.intent.admission_id)
                    != (owner.tenant_id, owner.session_id, owner.admission_session_id, owner.admission_id)
                ):
                    return False
            return (
                connection.execute(
                    f"UPDATE allocation SET state='bound', ({_OWNER_COLUMNS},reservation_etag)=(?,?,?,?,?,?,?,?) "
                    f"WHERE allocation_id=? AND state='unbound' AND pin_until_ms>{_NOW_MS}",
                    (*astuple(owner), reservation_etag, allocation_id),
                ).rowcount
                == 1
            )

    def anonymous_candidates(self, *, limit: int, after: tuple[int, bytes] = (0, b"")) -> tuple[Allocation, ...]:
        """Page through expired reservations and pending claims by (pin, allocation ID)."""
        if type(limit) is not int or limit <= 0:
            raise ValueError("anonymous candidate limit must be positive")
        with self._transaction() as connection:
            rows = connection.execute(
                f"SELECT {_ALLOCATION_COLUMNS} FROM allocation "
                f"WHERE tenant_id IS NULL AND pin_until_ms<={_NOW_MS} AND (pin_until_ms,allocation_id)>(?,?) "
                "ORDER BY pin_until_ms,allocation_id LIMIT ?",
                (*after, limit),
            )
            return tuple(self._allocation(row) for row in rows)

    def scoped_allocations(self, tenant_id: str, session_ids: tuple[str, ...]) -> tuple[Allocation, ...]:
        """Read exact bindings by target or trusted admission session under the caller's fence."""
        _require_text(tenant_id, *session_ids)
        with self._transaction() as connection:
            rows = connection.execute(
                f"SELECT {_ALLOCATION_COLUMNS} FROM allocation "
                "WHERE tenant_id=?1 AND (session_id IN (SELECT value FROM json_each(?2)) OR "
                "admission_session_id IN (SELECT value FROM json_each(?2))) ORDER BY allocation_id",
                (tenant_id, json.dumps(session_ids)),
            )
            return tuple(self._allocation(row) for row in rows)

    def claim_anonymous(self, allocation_id: bytes) -> Allocation | None:
        _require_id(allocation_id)
        with self._transaction(write=True) as connection:
            connection.execute(
                f"UPDATE allocation SET state='reaping' WHERE allocation_id=? AND state='unbound' AND pin_until_ms<={_NOW_MS}",
                (allocation_id,),
            )
            claimed = self._get(connection, allocation_id)
            if claimed is not None and claimed.state == "reaping" and claimed.owner is None:
                return claimed
            return None

    def claim_bound(self, expected: Allocation) -> Allocation | None:
        """Claim only the exact binding the caller found safe to reap under the primary fence."""
        _require_id(expected.allocation_id)
        if expected.journal_id != self._identity.journal_id or expected.state == "unbound" or expected.owner is None:
            return None
        with self._transaction(write=True) as connection:
            if self._get(connection, expected.allocation_id) != expected:
                return None
            row = connection.execute(
                f"UPDATE allocation SET state='reaping' WHERE allocation_id=? RETURNING {_ALLOCATION_COLUMNS}",
                (expected.allocation_id,),
            ).fetchone()
            return self._allocation(row) if row is not None else None

    def confirm_bound_delete(self, claim: Allocation) -> Allocation:
        """Record terminal deletion for an exact bound claim while retaining ownership."""
        _require_id(claim.allocation_id)
        if claim.state != "reaping" or claim.owner is None or claim.journal_id != self._identity.journal_id:
            raise ContentCleanupUnavailable("terminal bound deletion requires the exact scoped claim")
        with self._transaction(write=True) as connection:
            if self._get(connection, claim.allocation_id) != claim:
                raise ContentCleanupUnavailable("terminal bound deletion claim changed")
            row = connection.execute(
                f"UPDATE allocation SET delete_confirmed=1 WHERE allocation_id=? RETURNING {_ALLOCATION_COLUMNS}",
                (claim.allocation_id,),
            ).fetchone()
            assert row is not None
            return self._allocation(row)

    def finish_reap(self, claim: Allocation, *, remote_terminal: bool) -> bool:
        """Remove an exact claim once the caller verifies deletion and terminal request state."""
        _require_id(claim.allocation_id)
        if remote_terminal is not True:
            raise ContentCleanupUnavailable("allocation erasure has no terminal remote evidence")
        if claim.state != "reaping" or claim.journal_id != self._identity.journal_id:
            return False
        with self._transaction(write=True) as connection:
            if self._get(connection, claim.allocation_id) != claim:
                return False
            connection.execute("DELETE FROM allocation WHERE allocation_id=?", (claim.allocation_id,))
            return True

    def _key_intent(self, row: tuple) -> KeyIntent:
        *binding, state, pin_until_ms, evidence_id = row
        return KeyIntent(self._identity.journal_id, _ExactKeyIntent(*binding), state, pin_until_ms, evidence_id)

    def _get_key_intent(self, connection: sqlite3.Connection, provider_instance_id: str, key_id: str) -> KeyIntent | None:
        row = connection.execute(
            f"SELECT {_KEY_INTENT_COLUMNS} FROM key_intent WHERE provider_instance_id=? AND key_id=?",
            (provider_instance_id, key_id),
        ).fetchone()
        return self._key_intent(row) if row is not None else None

    def get_key_intent(self, provider_instance_id: str, key_id: str) -> KeyIntent | None:
        _require_text(provider_instance_id, key_id)
        with self._transaction() as connection:
            return self._get_key_intent(connection, provider_instance_id, key_id)

    def reserve_key_intent(self, intent: _ExactKeyIntent, *, pin_until_ms: int) -> KeyIntent | None:
        """Commit a fresh exact key ID and its scope while holding the primary fence."""
        _require_pin(pin_until_ms)
        with self._transaction(write=True) as connection:
            row = connection.execute(
                f"INSERT INTO key_intent({_KEY_OWNER_COLUMNS},state,pin_until_ms) "
                f"SELECT ?,?,?,?,?,?,'reserved',? WHERE ?>{_NOW_MS} "
                f"ON CONFLICT DO NOTHING RETURNING {_KEY_INTENT_COLUMNS}",
                (*astuple(intent), pin_until_ms, pin_until_ms),
            ).fetchone()
            return self._key_intent(row) if row is not None else None

    def key_intents_for_admission(
        self, tenant_id: str, session_id: str, admission_root_id: str, admission_id: str
    ) -> tuple[KeyIntent, ...]:
        _require_text(tenant_id, session_id, admission_root_id, admission_id)
        with self._transaction() as connection:
            rows = connection.execute(
                f"SELECT {_KEY_INTENT_COLUMNS} FROM key_intent "
                "WHERE tenant_id=? AND session_id=? AND admission_root_id=? AND admission_id=? "
                "ORDER BY provider_instance_id,key_id",
                (tenant_id, session_id, admission_root_id, admission_id),
            )
            return tuple(self._key_intent(row) for row in rows)

    def scoped_key_intents(self, tenant_id: str, session_ids: tuple[str, ...]) -> tuple[KeyIntent, ...]:
        _require_text(tenant_id, *session_ids)
        with self._transaction() as connection:
            rows = connection.execute(
                f"SELECT {_KEY_INTENT_COLUMNS} FROM key_intent "
                "WHERE tenant_id=?1 AND (session_id IN (SELECT value FROM json_each(?2)) OR "
                "admission_root_id IN (SELECT value FROM json_each(?2))) ORDER BY provider_instance_id,key_id",
                (tenant_id, json.dumps(session_ids)),
            )
            return tuple(self._key_intent(row) for row in rows)

    def claim_key_allocation(self, expected: KeyIntent) -> KeyIntent | None:
        """Grant a single acknowledged dispatch. A read of allocating cannot authorize retry."""
        if expected.journal_id != self._identity.journal_id or expected.state != "reserved":
            return None
        intent = expected.intent
        with self._transaction(write=True) as connection:
            if self._get_key_intent(connection, intent.provider_instance_id, intent.key_id) != expected:
                return None
            row = connection.execute(
                f"UPDATE key_intent SET state='allocating' WHERE provider_instance_id=? AND key_id=? "
                f"AND pin_until_ms>{_NOW_MS} RETURNING {_KEY_INTENT_COLUMNS}",
                (intent.provider_instance_id, intent.key_id),
            ).fetchone()
            return self._key_intent(row) if row is not None else None

    def claim_key_intent(self, expected: KeyIntent, *, require_expired_pin: bool = True) -> KeyIntent | None:
        """Claim under the primary fence after verifying exact SQL disposition.

        A committed erasure fence may bypass the orphan pin check. The journal
        cannot establish scope authority or prove absence from primary SQL.
        """
        if expected.journal_id != self._identity.journal_id or expected.state == "retired":
            return None
        intent = expected.intent
        with self._transaction(write=True) as connection:
            if self._get_key_intent(connection, intent.provider_instance_id, intent.key_id) != expected:
                return None
            if expected.state == "reaping":
                return expected
            row = connection.execute(
                f"UPDATE key_intent SET state='reaping' WHERE provider_instance_id=? AND key_id=? "
                f"AND (NOT ? OR pin_until_ms<={_NOW_MS}) RETURNING {_KEY_INTENT_COLUMNS}",
                (intent.provider_instance_id, intent.key_id, require_expired_pin),
            ).fetchone()
            return self._key_intent(row) if row is not None else None

    def record_key_retirement(self, claim: KeyIntent, outcome: KeyDestroyOutcome) -> KeyIntent | None:
        """Record terminal provider evidence obtained for this exact claim."""
        if outcome.status not in {KeyDestroyStatus.DESTROYED, KeyDestroyStatus.ALREADY_DESTROYED} or not outcome.evidence_id:
            raise ContentCleanupUnavailable("exact key retirement has no terminal provider evidence")
        if claim.journal_id != self._identity.journal_id or claim.state != "reaping":
            return None
        intent = claim.intent
        with self._transaction(write=True) as connection:
            current = self._get_key_intent(connection, intent.provider_instance_id, intent.key_id)
            if current is not None and current.state == "retired":
                if (
                    current.intent == intent
                    and current.pin_until_ms == claim.pin_until_ms
                    and current.evidence_id == outcome.evidence_id
                ):
                    return current
                return None
            if current != claim:
                return None
            row = connection.execute(
                f"UPDATE key_intent SET state='retired',evidence_id=? WHERE provider_instance_id=? AND key_id=? "
                f"RETURNING {_KEY_INTENT_COLUMNS}",
                (outcome.evidence_id, intent.provider_instance_id, intent.key_id),
            ).fetchone()
            return self._key_intent(row) if row is not None else None

    def finish_key_intent(self, retired: KeyIntent) -> bool:
        """Release scope metadata only after the exact key and all its objects are gone."""
        if retired.journal_id != self._identity.journal_id or retired.state != "retired":
            return False
        intent = retired.intent
        with self._transaction(write=True) as connection:
            if self._get_key_intent(connection, intent.provider_instance_id, intent.key_id) != retired:
                return False
            return (
                connection.execute(
                    "DELETE FROM key_intent WHERE provider_instance_id=?1 AND key_id=?2 "
                    "AND NOT EXISTS (SELECT 1 FROM allocation WHERE provider_instance_id=?1 AND key_id=?2)",
                    (intent.provider_instance_id, intent.key_id),
                ).rowcount
                == 1
            )
