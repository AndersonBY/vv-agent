from concurrent.futures import ThreadPoolExecutor
from dataclasses import astuple, replace
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
from vv_agent.kernel._postgres_allocation_journal import JournalRole, PostgresAllocationJournal, PostgresObjectJournal
from vv_agent.session.s3_content import CiphertextTemporarilyUnavailable, S3CiphertextStore

from .test_s3_reservation_adapter import _CIPHERTEXT, _Client, _pin

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
            ("binder_b", JournalRole("binder", "tenant-b")),
            ("key_a", JournalRole("key-intent", "tenant-a")),
            ("key_b", JournalRole("key-intent", "tenant-b")),
            ("tenant_a", JournalRole("scope", "tenant-a")),
            ("tenant_b", JournalRole("scope", "tenant-b")),
            ("cleanup_a", JournalRole("cleanup", "tenant-a")),
            ("cleanup_b", JournalRole("cleanup", "tenant-b")),
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


@pytest.mark.parametrize("operation", ["reserve", "bind"])
@pytest.mark.parametrize("matches", [True, False])
def test_object_store_reconciles_lost_journal_commit_reply(provisioned, monkeypatch, operation, matches):
    _, identity, journals, logins, _ = provisioned
    anonymous = journals[JournalRole("anonymous")]
    binder_a, scope_a = JournalRole("binder", "tenant-a"), JournalRole("scope", "tenant-a")
    binder_b, scope_b = JournalRole("binder", "tenant-b"), JournalRole("scope", "tenant-b")
    keys_a, keys_b = JournalRole("key-intent", "tenant-a"), JournalRole("key-intent", "tenant-b")
    journal = PostgresObjectJournal(
        anonymous=anonymous, binder=journals[binder_a], scope=journals[scope_a], keys=journals[keys_a]
    )
    foreign = PostgresObjectJournal(
        anonymous=anonymous, binder=journals[binder_b], scope=journals[scope_b], keys=journals[keys_b]
    )
    with pytest.raises(ValueError, match="matching provisioned role"):
        PostgresObjectJournal(anonymous=anonymous, binder=journals[binder_a], scope=journals[scope_b], keys=journals[keys_a])
    intent = _ExactKeyIntent("s1-exact-key-" + "a" * 64, "provider", "tenant-a", "child", "parent", "admission")
    pin = _pin()
    reserved = journal.reserve_key_intent(intent, pin_until_ms=pin)
    assert reserved is not None and journal.claim_key_allocation(reserved) is not None
    owner = replace(OWNER, key_id=intent.key_id)
    client = _Client(lambda allocation_id: journal.get_anonymous(allocation_id) or journal.get_owned(allocation_id, owner), owner)
    objects = S3CiphertextStore(client, bucket="fixture", journal=journal)
    allocation_ids: list[bytes] = []
    if operation == "reserve":
        reserve = anonymous.reserve

        def lost_reserve(allocation_id, *, pin_until_ms):
            allocation_ids.append(allocation_id)
            assert reserve(allocation_id, pin_until_ms=pin_until_ms if matches else pin_until_ms + 1)
            raise ContentUnavailable("reservation reply lost after commit")

        monkeypatch.setattr(anonymous, "reserve", lost_reserve)
    else:
        bind = journals[binder_a].bind

        def lost_bind(allocation_id, owner, *, reservation_etag):
            allocation_ids.append(allocation_id)
            assert bind(allocation_id, owner, reservation_etag=reservation_etag if matches else '"wrong-etag"')
            raise ContentUnavailable("binding reply lost after commit")

        monkeypatch.setattr(journals[binder_a], "bind", lost_bind)

    if not matches:
        with pytest.raises(CiphertextTemporarilyUnavailable):
            objects.put(owner, _CIPHERTEXT, pin_until_ms=pin, check_fence=client.fence)
        assert len(client.calls) == (0 if operation == "reserve" else 1)
        return

    locator = objects.put(owner, _CIPHERTEXT, pin_until_ms=pin, check_fence=client.fence)
    assert allocation_ids == [locator[1:]]
    assert len(client.calls) == 2
    reopened = PostgresObjectJournal(
        anonymous=anonymous,
        binder=journals[binder_a],
        scope=PostgresAllocationJournal(logins[scope_a], identity, scope_a),
        keys=PostgresAllocationJournal(logins[keys_a], identity, keys_a),
    )
    row = reopened.get_owned(locator[1:], owner)
    assert row is not None and row.owner == owner and row.state == "bound"
    assert reopened.get_owned(locator[1:], replace(owner, admission_id="different")) is None
    assert reopened.get_anonymous(locator[1:]) is None
    assert foreign.get_owned(locator[1:], replace(owner, tenant_id="tenant-b")) is None
    assert foreign.get_anonymous(locator[1:]) is None
    denied = S3CiphertextStore(client, bucket="fixture", journal=foreign)
    with pytest.raises(ContentUnavailable):
        denied.get(replace(owner, tenant_id="tenant-b"), locator, ciphertext_size=len(_CIPHERTEXT))


def test_object_journal_uses_exact_tenant_key_authority(provisioned):
    admin, identity, journals, logins, _ = provisioned
    anonymous = journals[JournalRole("anonymous")]
    tenant = "tenant-a"
    key_role = JournalRole("key-intent", tenant)
    binder, scope, keys = journals[JournalRole("binder", tenant)], journals[JournalRole("scope", tenant)], journals[key_role]
    with pytest.raises(ContentUnavailable, match="identity or policy"):
        PostgresAllocationJournal(logins[key_role], replace(identity, database_id="other-primary"), key_role)
    for wrong in (scope, journals[JournalRole("key-intent", "tenant-b")]):
        with pytest.raises(ValueError, match="matching provisioned role"):
            PostgresObjectJournal(anonymous=anonymous, binder=binder, scope=scope, keys=wrong)

    journal = PostgresObjectJournal(anonymous=anonymous, binder=binder, scope=scope, keys=keys)
    assert journal.tenant_id == tenant and journal.identity == identity
    intent = _ExactKeyIntent("s1-exact-key-" + "a" * 64, "provider", tenant, "child", "parent", "admission")
    pin = _now(admin) + 10_000
    foreign = replace(intent, tenant_id="tenant-b", key_id="s1-exact-key-" + "b" * 64)
    assert journal.reserve_key_intent(foreign, pin_until_ms=pin) is None
    assert journals[JournalRole("key-intent", foreign.tenant_id)].reserve_key_intent(foreign, pin_until_ms=pin) is not None
    assert journal.get_key_intent(foreign.provider_instance_id, foreign.key_id) is None
    assert journal.reserve_key_intent(intent, pin_until_ms=_now(admin)) is None
    reserved = journal.reserve_key_intent(intent, pin_until_ms=pin)
    assert reserved is not None and reserved.intent == intent
    assert journal.reserve_key_intent(intent, pin_until_ms=pin) is None
    assert journal.key_intents_for_admission("tenant-b", "child", "parent", "admission") == ()
    assert journal.key_intents_for_admission(tenant, "child", "parent", "different") == ()
    assert journal.key_intents_for_admission(tenant, "child", "parent", "admission") == (reserved,)
    assert journal.get_key_intent("other-provider", intent.key_id) is None
    assert journal.claim_key_allocation(replace(reserved, intent=replace(intent, session_id="different"))) is None
    assert journal.claim_key_allocation(replace(reserved, journal_id="different")) is None
    assert journal.claim_key_allocation(replace(reserved, intent=replace(intent, tenant_id="tenant-b"))) is None
    assert journal.get_key_intent(intent.provider_instance_id, intent.key_id) == reserved
    allocated = journal.claim_key_allocation(reserved)
    assert allocated is not None and allocated.state == "allocating"
    assert journal.get_key_intent(intent.provider_instance_id, intent.key_id) == allocated
    assert journal.claim_key_allocation(reserved) is None


def test_bound_row_survives_primary_rollback_and_requires_exact_claim(provisioned, postgres_admin_dsn):
    import psycopg

    admin, identity, journals, _, _ = provisioned
    anonymous, binder = journals[JournalRole("anonymous")], journals[JournalRole("binder")]
    scope, foreign = journals[JournalRole("scope", "tenant-a")], journals[JournalRole("scope", "tenant-b")]
    cleanup, foreign_cleanup = journals[JournalRole("cleanup", "tenant-a")], journals[JournalRole("cleanup", "tenant-b")]
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
    assert scope.claim_bound(bound) is None
    assert foreign_cleanup.claim_bound(bound) is None
    assert cleanup.claim_bound(replace(bound, journal_id="other")) is None
    assert cleanup.claim_bound(replace(bound, reservation_etag='"wrong"')) is None
    assert cleanup.claim_bound(replace(bound, owner=replace(OWNER, key_id="new"))) is None
    claim = cleanup.claim_bound(bound)
    assert claim is not None and claim.state == "reaping" and not claim.delete_confirmed
    assert cleanup.claim_bound(bound) is None
    assert cleanup.claim_bound(claim) == claim
    with pytest.raises(ContentCleanupUnavailable):
        cleanup.finish_reap(claim, remote_terminal=False)
    assert not cleanup.finish_reap(claim, remote_terminal=True)
    assert not cleanup.finish_reap(replace(claim, delete_confirmed=True), remote_terminal=True)
    with pytest.raises(ContentCleanupUnavailable, match="claim changed"):
        cleanup.confirm_bound_delete(replace(claim, delete_confirmed=True))
    for mismatch in (
        replace(claim, journal_id="other"),
        replace(claim, pin_until_ms=claim.pin_until_ms + 1),
        replace(claim, reservation_etag='"wrong"'),
        replace(claim, owner=replace(OWNER, key_id="different")),
        replace(claim, owner=replace(OWNER, admission_session_id="different")),
        replace(claim, owner=None),
    ):
        with pytest.raises(ContentCleanupUnavailable):
            cleanup.confirm_bound_delete(mismatch)
    for unauthorized in (scope, foreign_cleanup, anonymous, binder):
        with pytest.raises(ContentCleanupUnavailable, match="exact cleanup claim"):
            unauthorized.confirm_bound_delete(claim)
    assert scope.get(allocation_id) == claim
    confirmed = cleanup.confirm_bound_delete(claim)
    assert confirmed == replace(claim, delete_confirmed=True)
    cleanup_role = JournalRole("cleanup", "tenant-a")
    reopened = PostgresAllocationJournal(provisioned[3][cleanup_role], identity, cleanup_role)
    assert reopened.scoped_allocations("tenant-a", ("parent",)) == (confirmed,)
    assert reopened.confirm_bound_delete(claim) == confirmed
    assert reopened.confirm_bound_delete(confirmed) == confirmed
    assert reopened.claim_bound(confirmed) == confirmed
    assert reopened.claim_bound(claim) is None
    assert not reopened.finish_reap(claim, remote_terminal=True)
    assert not reopened.finish_reap(replace(confirmed, owner=None), remote_terminal=True)
    assert not scope.finish_reap(confirmed, remote_terminal=True)
    assert not foreign_cleanup.finish_reap(confirmed, remote_terminal=True)
    assert reopened.finish_reap(confirmed, remote_terminal=True)
    assert not reopened.finish_reap(confirmed, remote_terminal=True)
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
    cleanup_role = JournalRole("cleanup", "tenant-a")
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
        for command in ("DELETE FROM", "UPDATE"):
            statement = f"{command} {TABLE}"
            if command == "UPDATE":
                statement += " SET state='reaping'"
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                connection.execute(statement)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            connection.execute(f"TRUNCATE {TABLE}")
    with psycopg.connect(logins[cleanup_role], autocommit=True) as connection:
        assert connection.execute(f"UPDATE {TABLE} SET state='reaping' WHERE allocation_id=%s", (pending,)).rowcount == 0
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="invalid bound deletion confirmation"):
            connection.execute(f"UPDATE {TABLE} SET state='reaping',delete_confirmed=true WHERE allocation_id=%s", (bound,))
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            connection.execute(f"UPDATE {TABLE} SET tenant_id='tenant-b'")
        assert connection.execute(f"UPDATE {TABLE} SET state='reaping' WHERE allocation_id=%s", (bound,)).rowcount == 1
        assert connection.execute(f"DELETE FROM {TABLE} WHERE allocation_id=%s", (bound,)).rowcount == 0
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
            connection.execute(f"UPDATE {TABLE} SET state='bound' WHERE allocation_id=%s", (bound,))
        assert connection.execute(f"UPDATE {TABLE} SET delete_confirmed=true WHERE allocation_id=%s", (bound,)).rowcount == 1
        assert connection.execute(f"UPDATE {TABLE} SET delete_confirmed=false WHERE allocation_id=%s", (bound,)).rowcount == 0
        assert connection.execute(f"DELETE FROM {TABLE} WHERE allocation_id=%s", (bound,)).rowcount == 1
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
    with psycopg.connect(logins[key_role], autocommit=True) as client:
        assert client.execute(f"UPDATE {KEY_TABLE} SET state='reaping' WHERE state='allocating'").rowcount == 0
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            client.execute(f"DELETE FROM {KEY_TABLE}")
    cleanup = journals[JournalRole("cleanup", "tenant-a")]
    assert cleanup.claim_key_intent(allocating) is None
    claim = cleanup.claim_key_intent(allocating, require_expired_pin=False)
    assert claim is not None and claim.state == "reaping"
    assert cleanup.claim_key_intent(claim) == claim
    replacement_id = bytes([1]) * 32
    assert anonymous.reserve(replacement_id, pin_until_ms=pin)
    assert not binder.bind(replacement_id, owner, reservation_etag=ETAG)
    outcome = KeyDestroyOutcome(True, KeyDestroyStatus.DESTROYED, "receipt")
    assert reopened.record_key_retirement(claim, outcome) is None
    retired = cleanup.record_key_retirement(claim, outcome)
    assert retired is not None and retired.state == "retired"
    assert cleanup.record_key_retirement(claim, outcome) == retired
    assert cleanup.record_key_retirement(claim, replace(outcome, evidence_id="other")) is None
    assert not cleanup.finish_key_intent(retired)
    bound = scope.get(allocation_id)
    assert bound is not None
    object_claim = cleanup.claim_bound(bound)
    assert object_claim is not None
    confirmed = cleanup.confirm_bound_delete(object_claim)
    assert cleanup.finish_reap(confirmed, remote_terminal=True)
    with pytest.raises(ContentUnavailable):
        reopened.finish_key_intent(retired)
    assert cleanup.finish_key_intent(retired)
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
    assert journals[JournalRole("cleanup", "tenant-a")].claim_key_intent(reserved).state == "reaping"


@pytest.mark.parametrize("field", ["admission_session_id", "admission_id"])
def test_exact_bind_rejects_mismatched_admission(provisioned, field):
    import psycopg

    admin, _, journals, logins, _ = provisioned
    keys = journals[JournalRole("key-intent", "tenant-a")]
    anonymous = journals[JournalRole("anonymous")]
    binder_role = JournalRole("binder", "tenant-a")
    binder = journals[binder_role]
    scope = journals[JournalRole("scope", "tenant-a")]
    intent = _ExactKeyIntent("s1-exact-key-" + "c" * 64, "provider", "tenant-a", "child", "parent", "admission")
    pin = _now(admin) + 10_000
    reserved = keys.reserve_key_intent(intent, pin_until_ms=pin)
    assert reserved is not None and keys.claim_key_allocation(reserved) is not None
    owner = replace(OWNER, key_id=intent.key_id)
    wrong = replace(owner, **{field: "wrong"})
    allocation_id = bytes(32)
    assert anonymous.reserve(allocation_id, pin_until_ms=pin)
    assert not binder.bind(allocation_id, wrong, reservation_etag=ETAG)
    with (
        psycopg.connect(logins[binder_role], autocommit=True) as raw,
        pytest.raises(psycopg.errors.InsufficientPrivilege, match="exact key is unavailable for binding"),
    ):
        raw.execute(
            f"UPDATE {TABLE} SET state='bound',tenant_id=%s,session_id=%s,content_id=%s,"
            "admission_session_id=%s,admission_id=%s,provider_instance_id=%s,key_id=%s,reservation_etag=%s "
            "WHERE allocation_id=%s",
            (*astuple(wrong), ETAG, allocation_id),
        )
    assert binder.bind(allocation_id, owner, reservation_etag=ETAG)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        admin.execute(f"UPDATE {TABLE} SET {field}=%s WHERE allocation_id=%s", ("wrong", allocation_id))
    assert scope.scoped_allocations("tenant-a", ("parent",))[0].owner == owner


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
        with (
            psycopg.connect(logins[JournalRole("cleanup", "tenant-a")], autocommit=True) as cleanup,
            cleanup.transaction(),
        ):
            assert cleanup.execute(f"UPDATE {KEY_TABLE} SET state='reaping' WHERE key_id=%s", (intent.key_id,)).rowcount == 1
            pending = pool.submit(raw_bind)
            _wait_for_row_lock(admin, f"UPDATE {TABLE}%")
        pending.result(timeout=5)
    assert anonymous.get(bytes(32)).state == "unbound"


def test_obsolete_journal_revision_fails_closed(provisioned):
    admin, identity, journals, logins, _ = provisioned
    role = JournalRole("key-intent", "tenant-a")
    admin.execute("UPDATE vv_allocation_journal.identity SET revision=4")
    with pytest.raises(ContentUnavailable, match="identity or policy"):
        PostgresAllocationJournal(logins[role], identity, role)
    with pytest.raises(ContentUnavailable, match="identity or policy"):
        journals[role].get_key_intent("provider", "unavailable")
