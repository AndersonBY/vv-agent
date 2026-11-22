from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from secrets import token_hex
from threading import Event

import pytest

from vv_agent.kernel._allocation_journal import AllocationOwner
from vv_agent.kernel._content import ContentCleanupUnavailable, ContentUnavailable
from vv_agent.kernel._postgres import CLOCK
from vv_agent.kernel._postgres_allocation_journal import JournalRole, PostgresAllocationJournal

pytestmark = pytest.mark.postgres
TABLE = "vv_allocation_journal.allocation"
OWNER = AllocationOwner("tenant-a", "child", "content", "parent", "admission", "provider", "key")
ETAG = '"marker"'


@pytest.fixture
def provisioned(database):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    roles = {
        f"aj_{suffix}_{token_hex(8)}": role
        for suffix, role in (
            ("anonymous", JournalRole("anonymous")),
            ("binder", JournalRole("binder")),
            ("tenant_a", JournalRole("scope", "tenant-a")),
            ("tenant_b", JournalRole("scope", "tenant-b")),
        )
    }
    outside_role = f"aj_outside_{token_hex(8)}"
    password = token_hex(32)
    with psycopg.connect(database, autocommit=True) as admin:
        try:
            for login in (*roles, outside_role):
                admin.execute(
                    sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} NOINHERIT").format(sql.Identifier(login), sql.Literal(password))
                )
            identity = PostgresAllocationJournal.provision(admin, database_id="main", namespace="bucket", roles=roles)
            logins = {role: make_conninfo(database, user=login, password=password) for login, role in roles.items()}
            journals = {
                role: PostgresAllocationJournal(dsn, identity, role, lock_timeout_ms=4500, statement_timeout_ms=5000)
                for role, dsn in logins.items()
            }
            yield admin, identity, journals, logins, make_conninfo(database, user=outside_role, password=password)
        finally:
            admin.execute("DROP SCHEMA IF EXISTS vv_allocation_journal CASCADE")
            for login in (*roles, outside_role):
                admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(login)))


def _now(admin):
    return admin.execute(f"SELECT {CLOCK}").fetchone()[0]


def test_bound_row_survives_primary_rollback_and_requires_exact_claim(provisioned, postgres_admin_dsn):
    import psycopg

    admin, identity, journals, _, _ = provisioned
    anonymous, binder = journals[JournalRole("anonymous")], journals[JournalRole("binder")]
    scope, foreign = journals[JournalRole("scope", "tenant-a")], journals[JournalRole("scope", "tenant-b")]
    allocation_id = bytes(32)
    with psycopg.connect(postgres_admin_dsn, autocommit=True) as primary, primary.transaction(force_rollback=True):
        primary.execute("CREATE TEMP TABLE journal_test_business (id integer)")
        primary.execute("INSERT INTO journal_test_business VALUES (1)")
        assert primary.execute("SELECT current_database()").fetchone() != admin.execute("SELECT current_database()").fetchone()
        assert anonymous.reserve(allocation_id, pin_until_ms=_now(admin) + 10_000)
        assert not anonymous.reserve(allocation_id, pin_until_ms=_now(admin) + 10_000)
        assert anonymous.get(allocation_id).owner is None
        assert binder.bind(allocation_id, OWNER, reservation_etag=ETAG)
        assert scope.get(allocation_id).owner == OWNER
    assert anonymous.get(allocation_id) is None
    assert foreign.get(allocation_id) is None
    assert foreign.scoped_allocations("tenant-a", ("parent",)) == ()
    (bound,) = scope.scoped_allocations("tenant-a", ("parent",))
    assert scope.scoped_allocations("tenant-a", ("child",)) == (bound,)
    assert bound.reservation_etag == ETAG
    assert not binder.bind(allocation_id, replace(OWNER, key_id="new"), reservation_etag=ETAG)
    assert scope.claim_bound(replace(bound, journal_id="other")) is None
    assert scope.claim_bound(replace(bound, reservation_etag='"wrong"')) is None
    assert scope.claim_bound(replace(bound, owner=replace(OWNER, key_id="new"))) is None
    claim = scope.claim_bound(bound)
    assert claim is not None and claim.state == "reaping"
    assert scope.claim_bound(bound) is None
    assert scope.claim_bound(claim) == claim
    with pytest.raises(ContentCleanupUnavailable):
        scope.finish_reap(claim, remote_terminal=False)
    assert not scope.finish_reap(replace(claim, owner=None), remote_terminal=True)
    assert not foreign.finish_reap(claim, remote_terminal=True)
    assert scope.finish_reap(claim, remote_terminal=True)
    assert not scope.finish_reap(claim, remote_terminal=True)
    assert scope.get(allocation_id) is None
    with pytest.raises(ContentUnavailable, match="role"):
        PostgresAllocationJournal(
            provisioned[3][JournalRole("scope", "tenant-a")],
            replace(identity, journal_id="wrong"),
            JournalRole("scope", "tenant-a"),
        )


def test_database_grants_and_rls_limit_authenticated_logins(provisioned):
    import psycopg

    admin, identity, journals, logins, outside_dsn = provisioned
    anonymous_role, binder_role = JournalRole("anonymous"), JournalRole("binder")
    scope_role = JournalRole("scope", "tenant-a")
    bound, pending = bytes(32), bytes([1]) * 32
    for allocation_id in (bound, pending):
        assert journals[anonymous_role].reserve(allocation_id, pin_until_ms=_now(admin) + 10_000)
    assert journals[binder_role].bind(bound, OWNER, reservation_etag=ETAG)
    with psycopg.connect(outside_dsn, autocommit=True) as connection, pytest.raises(psycopg.errors.InsufficientPrivilege):
        connection.execute(f"SELECT * FROM {TABLE}")
    with psycopg.connect(logins[anonymous_role], autocommit=True) as connection:
        assert connection.execute(f"SELECT allocation_id FROM {TABLE}").fetchall() == [(pending,)]
        assert connection.execute(f"DELETE FROM {TABLE} WHERE allocation_id=%s", (bound,)).rowcount == 0
        assert connection.execute(f"DELETE FROM {TABLE} WHERE allocation_id=%s", (pending,)).rowcount == 0
        assert connection.execute(f"UPDATE {TABLE} SET state='reaping' WHERE allocation_id=%s", (pending,)).rowcount == 0
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            connection.execute(f"UPDATE {TABLE} SET tenant_id='tenant-b'")
    with psycopg.connect(logins[binder_role], autocommit=True) as connection:
        rows = connection.execute(f"SELECT allocation_id FROM {TABLE} ORDER BY allocation_id").fetchall()
        assert rows == [(bound,), (pending,)]
        assert connection.execute(f"UPDATE {TABLE} SET tenant_id='tenant-b' WHERE allocation_id=%s", (bound,)).rowcount == 0
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            connection.execute(f"SELECT tenant_id FROM {TABLE}")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            connection.execute(f"DELETE FROM {TABLE}")
    with psycopg.connect(logins[scope_role], autocommit=True) as connection:
        connection.execute("SELECT set_config('vv.tenant_id','tenant-b',false)")
        assert connection.execute(f"SELECT allocation_id FROM {TABLE}").fetchall() == [(bound,)]
        assert connection.execute(f"DELETE FROM {TABLE} WHERE allocation_id=%s", (bound,)).rowcount == 0
        assert connection.execute(f"UPDATE {TABLE} SET state='reaping' WHERE allocation_id=%s", (pending,)).rowcount == 0
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            connection.execute(f"UPDATE {TABLE} SET tenant_id='tenant-b'")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            connection.execute(f"TRUNCATE {TABLE}")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            connection.execute(f"UPDATE {TABLE} SET state='unbound' WHERE allocation_id=%s", (bound,))
    with pytest.raises(ContentUnavailable, match="role"):
        PostgresAllocationJournal(logins[scope_role], identity, JournalRole("scope", "tenant-b"))


def test_bind_rechecks_after_lock_and_anonymous_claim_cannot_rebind(provisioned):
    admin, _, journals, _, _ = provisioned
    anonymous, binder = journals[JournalRole("anonymous")], journals[JournalRole("binder")]
    allocation_id = bytes(32)
    pin = _now(admin) + 1500
    assert anonymous.reserve(allocation_id, pin_until_ms=pin)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with admin.transaction():
            admin.execute(f"SELECT allocation_id FROM {TABLE} WHERE allocation_id=%s FOR UPDATE", (allocation_id,))
            future = pool.submit(binder.bind, allocation_id, OWNER, reservation_etag=ETAG)
            for _ in range(100):
                if admin.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type='Lock' AND query LIKE %s",
                    ("%allocation_id%FOR UPDATE%",),
                ).fetchone()[0]:
                    break
                admin.execute("SELECT pg_stat_clear_snapshot()")
                Event().wait(0.01)
            else:
                pytest.fail("binder did not wait for the reservation lock")
            assert _now(admin) < pin
            delay = (pin - _now(admin) + 20) / 1000
            admin.execute("SELECT pg_sleep(%s)", (delay,))
        assert not future.result(timeout=5)
    (candidate,) = anonymous.anonymous_candidates(limit=1)
    assert anonymous.anonymous_candidates(limit=1, after=(candidate.pin_until_ms, candidate.allocation_id)) == ()
    claim = anonymous.claim_anonymous(allocation_id)
    assert claim is not None and claim.owner is None and claim.state == "reaping"
    assert anonymous.claim_anonymous(allocation_id) == claim
    assert not binder.bind(allocation_id, OWNER, reservation_etag=ETAG)
    assert anonymous.finish_reap(claim, remote_terminal=True)


@pytest.mark.parametrize(
    "statement",
    [f"ALTER TABLE {TABLE} NO FORCE ROW LEVEL SECURITY", f"GRANT SELECT ON {TABLE} TO PUBLIC"],
)
def test_runtime_rejects_policy_or_privilege_drift(provisioned, statement):
    admin, _, journals, _, _ = provisioned
    scope = journals[JournalRole("scope", "tenant-a")]
    admin.execute(statement)
    with pytest.raises(ContentUnavailable, match="identity or policy"):
        scope.get(bytes(32))


def test_concurrent_binders_can_commit_only_one_exact_owner(provisioned):
    admin, _, journals, _, _ = provisioned
    anonymous, binder = journals[JournalRole("anonymous")], journals[JournalRole("binder")]
    scope = journals[JournalRole("scope", "tenant-a")]
    allocation_id = bytes(32)
    assert anonymous.reserve(allocation_id, pin_until_ms=_now(admin) + 10_000)
    owners = (OWNER, replace(OWNER, key_id="replacement"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(binder.bind, allocation_id, owner, reservation_etag=ETAG) for owner in owners]
        results = [future.result(timeout=5) for future in futures]
    assert results.count(True) == 1
    assert scope.get(allocation_id).owner == owners[results.index(True)]
