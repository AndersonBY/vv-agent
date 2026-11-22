from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from secrets import token_hex
from threading import Event

import pytest

from vv_agent.kernel._allocation_journal import AllocationOwner
from vv_agent.kernel._content import (
    ContentCleanupUnavailable,
    ContentUnavailable,
    KeyDestroyOutcome,
    KeyDestroyStatus,
    _ExactKeyIntent,
)
from vv_agent.kernel._postgres import CLOCK
from vv_agent.kernel._postgres_allocation_journal import JournalRole, PostgresAllocationJournal

pytestmark = pytest.mark.postgres
TABLE = "vv_allocation_journal.allocation"
KEY_TABLE = "vv_allocation_journal.key_intent"
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
            ("binder_a", JournalRole("binder", "tenant-a")),
            ("key_a", JournalRole("key-intent", "tenant-a")),
            ("key_b", JournalRole("key-intent", "tenant-b")),
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


def _wait_for_row_lock(admin, query_pattern):
    for _ in range(100):
        if admin.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type='Lock' AND query LIKE %s",
            (query_pattern,),
        ).fetchone()[0]:
            return
        admin.execute("SELECT pg_stat_clear_snapshot()")
        Event().wait(0.01)
    pytest.fail("journal operation did not wait for the row lock")


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
            _wait_for_row_lock(admin, "%allocation_id%FOR UPDATE%")
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


def test_exact_key_readback_mutations_and_cleanup_require_authority(provisioned):
    import psycopg

    admin, identity, journals, logins, _ = provisioned
    key_role = JournalRole("key-intent", "tenant-a")
    keys, foreign = journals[key_role], journals[JournalRole("key-intent", "tenant-b")]
    anonymous, scope = journals[JournalRole("anonymous")], journals[JournalRole("scope", "tenant-a")]
    binder_role = JournalRole("binder", "tenant-a")
    binder, legacy = journals[binder_role], journals[JournalRole("binder")]
    intent = _ExactKeyIntent("s1-exact-key-" + "a" * 64, "provider", "tenant-a", "child", "parent", "admission")
    owner = replace(OWNER, key_id=intent.key_id)
    allocation_id = bytes(32)
    pin = _now(admin) + 10_000
    assert keys.reserve_key_intent(intent, pin_until_ms=pin)
    reopened = PostgresAllocationJournal(logins[key_role], identity, key_role)
    reserved = reopened.get_key_intent(intent.provider_instance_id, intent.key_id)
    assert reserved is not None and reserved.state == "reserved" and reserved.intent == intent
    assert scope.get_key_intent(intent.provider_instance_id, intent.key_id) == reserved
    assert scope.scoped_key_intents("tenant-a", ("parent",)) == (reserved,)
    assert foreign.get_key_intent(intent.provider_instance_id, intent.key_id) is None
    assert foreign.key_intents_for_admission("tenant-a", "child", "parent", "admission") == ()
    with pytest.raises(ContentUnavailable):
        foreign.reserve_key_intent(replace(intent, key_id="s1-exact-key-" + "b" * 64), pin_until_ms=pin)
    assert anonymous.reserve(allocation_id, pin_until_ms=pin)
    assert not binder.bind(allocation_id, owner, reservation_etag=ETAG)
    assert not legacy.bind(allocation_id, owner, reservation_etag=ETAG)

    assert keys.claim_key_allocation(reserved)
    allocating = reopened.get_key_intent(intent.provider_instance_id, intent.key_id)
    assert allocating is not None and allocating.state == "allocating"
    assert not reopened.claim_key_allocation(reserved)
    assert not reopened.claim_key_allocation(allocating)
    assert not binder.bind(allocation_id, replace(owner, tenant_id="tenant-b"), reservation_etag=ETAG)
    assert not binder.bind(allocation_id, replace(owner, session_id="replacement"), reservation_etag=ETAG)
    with psycopg.connect(logins[binder_role], autocommit=True) as client, pytest.raises(psycopg.errors.InsufficientPrivilege):
        client.execute(f"UPDATE {KEY_TABLE} SET state='reaping'")
    assert binder.bind(allocation_id, owner, reservation_etag=ETAG)
    assert reopened.claim_key_intent(allocating) is None
    claim = reopened.claim_key_intent(allocating, require_expired_pin=False)
    assert claim is not None and claim.state == "reaping"
    assert reopened.claim_key_intent(claim) == claim
    replacement_id = bytes([1]) * 32
    assert anonymous.reserve(replacement_id, pin_until_ms=pin)
    assert not binder.bind(replacement_id, owner, reservation_etag=ETAG)
    outcome = KeyDestroyOutcome(True, KeyDestroyStatus.DESTROYED, "receipt")
    retired = keys.record_key_retirement(claim, outcome)
    assert retired is not None and retired.state == "retired"
    assert reopened.record_key_retirement(claim, outcome) == retired
    assert reopened.record_key_retirement(claim, replace(outcome, evidence_id="other")) is None
    assert not reopened.finish_key_intent(retired)
    bound = scope.get(allocation_id)
    assert bound is not None
    object_claim = scope.claim_bound(bound)
    assert object_claim is not None
    assert scope.finish_reap(object_claim, remote_terminal=True)
    assert reopened.finish_key_intent(retired)
    assert reopened.get_key_intent(intent.provider_instance_id, intent.key_id) is None


def test_key_allocation_pin_is_rechecked_after_waiting_for_row_lock(provisioned):
    admin, _, journals, _, _ = provisioned
    keys = journals[JournalRole("key-intent", "tenant-a")]
    pin = _now(admin) + 1500
    intent = _ExactKeyIntent("s1-exact-key-" + "b" * 64, "provider", "tenant-a", "child", "parent", "admission")
    reserved = keys.reserve_key_intent(intent, pin_until_ms=pin)
    assert reserved is not None
    with ThreadPoolExecutor(max_workers=1) as pool:
        with admin.transaction():
            admin.execute(f"SELECT key_id FROM {KEY_TABLE} FOR UPDATE")
            future = pool.submit(keys.claim_key_allocation, reserved)
            _wait_for_row_lock(admin, "%key_intent%FOR UPDATE%")
            assert _now(admin) < pin
            admin.execute("SELECT pg_sleep(%s)", ((pin - _now(admin) + 20) / 1000,))
        assert future.result(timeout=5) is None
    assert keys.get_key_intent(intent.provider_instance_id, intent.key_id) == reserved
    assert keys.claim_key_intent(reserved).state == "reaping"


def test_exact_bind_waits_for_key_cleanup_claim(provisioned):
    admin, _, journals, logins, _ = provisioned
    key_role = JournalRole("key-intent", "tenant-a")
    keys, anonymous = journals[key_role], journals[JournalRole("anonymous")]
    binder_role = JournalRole("binder", "tenant-a")
    intent = _ExactKeyIntent("s1-exact-key-" + "c" * 64, "provider", "tenant-a", "child", "parent", "admission")
    pin = _now(admin) + 10_000
    reserved = keys.reserve_key_intent(intent, pin_until_ms=pin)
    assert reserved is not None and keys.claim_key_allocation(reserved) is not None
    assert anonymous.reserve(bytes(32), pin_until_ms=pin)

    import psycopg

    def raw_bind():
        with (
            psycopg.connect(logins[binder_role], autocommit=True) as client,
            pytest.raises(psycopg.errors.InsufficientPrivilege, match="exact key is unavailable"),
        ):
            client.execute(
                f"UPDATE {TABLE} SET state='bound',tenant_id=%s,session_id=%s,content_id=%s,"
                "admission_session_id=%s,admission_id=%s,provider_instance_id=%s,key_id=%s,reservation_etag=%s "
                "WHERE allocation_id=%s",
                ("tenant-a", "child", "content", "parent", "admission", "provider", intent.key_id, ETAG, bytes(32)),
            )

    with ThreadPoolExecutor(max_workers=1) as pool:
        with psycopg.connect(logins[key_role], autocommit=True) as cleanup, cleanup.transaction():
            cleanup.execute(f"UPDATE {KEY_TABLE} SET state='reaping' WHERE key_id=%s", (intent.key_id,))
            pending = pool.submit(raw_bind)
            _wait_for_row_lock(admin, f"UPDATE {TABLE}%")
        pending.result(timeout=5)
    assert anonymous.get(bytes(32)).state == "unbound"
