from contextlib import closing
from dataclasses import replace
from uuid import uuid4

import psycopg
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from psycopg import sql
from psycopg.conninfo import make_conninfo
from session.test_s3_reservation_adapter import _Client
from test_s1_postgres_rls import TENANT_A, TENANT_B, _dsn, _provision
from test_s1_postgres_rls import disposable_pg as disposable_pg
from test_s1_postgres_rls import rls_database as rls_database

from vv_agent.kernel._content import ContentCleanupUnavailable, Principal
from vv_agent.kernel._content_runtime import _provision_postgres_object_content, bind_content_runtime
from vv_agent.kernel._external_content import _ErasureCurrentnessChallenge, _PostgresObjectContent, _PostgresRootErasure
from vv_agent.kernel._postgres_allocation_journal import JournalRole, PostgresAllocationJournal, PostgresObjectJournal
from vv_agent.kernel._store import Conflict
from vv_agent.session.content_store import AESGCMCipher, LocalKeyProvider
from vv_agent.session.deletion import DeletionAuthenticity, SessionDeletionStore
from vv_agent.session.ledger import Ed25519LedgerSigner
from vv_agent.session.postgres import PostgresContentStore, PostgresStore
from vv_agent.session.s3_content import S3CiphertextStore
from vv_agent.session.surfaces import SessionDriver


class _EraseClient(_Client):
    def __init__(self, journal, cleanup):
        super().__init__(lambda aid: cleanup.get(aid) or journal.get_anonymous(aid))
        self.cleanup = cleanup

    def put_object(self, **request):
        if "IfMatch" in request:
            row = self.cleanup.get(bytes.fromhex(request["Key"].rsplit("/", 1)[1]))
            assert row is not None and row.owner is not None and row.state == "bound"
            self.owner = row.owner
        return super().put_object(**request)


@pytest.fixture
def erasure_storage(rls_database, tmp_path):
    database = rls_database
    _provision(database)
    admin_dsn = _dsn(database, database.cluster.admin_role, database.cluster.admin_password)
    owner_dsn = _dsn(database, database.migration_role, database.migration_password)
    tenant_dsn = _dsn(database, database.tenant_a_role, database.tenant_a_password)
    nonce = uuid4().hex
    journal_name = f"vv_erase_{nonce}"
    journal_password = uuid4().hex
    roles = {
        f"vv_erase_{i}_{nonce}": role
        for i, role in enumerate(
            (
                JournalRole("anonymous"),
                JournalRole("binder", TENANT_A),
                JournalRole("scope", TENANT_A),
                JournalRole("key-intent", TENANT_A),
                JournalRole("cleanup", TENANT_A),
                JournalRole("cleanup", TENANT_B),
            )
        )
    }
    with (
        psycopg.connect(admin_dsn, autocommit=True) as admin,
        PostgresStore.schema_admin_standalone(owner_dsn) as primary_owner,
        closing(LocalKeyProvider(b"x" * 32, state_path=tmp_path / "keys.sqlite")) as provider,
    ):
        primary_owner.install_tenant_rls(TENANT_A, database.tenant_a_role, database.tenant_a_password)
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(journal_name)))
        try:
            for login in roles:
                admin.execute(
                    sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} NOINHERIT").format(
                        sql.Identifier(login), sql.Literal(journal_password)
                    )
                )
            journal_dsn = make_conninfo(admin_dsn, dbname=journal_name)
            (database_id,) = primary_owner._one("SELECT database_id FROM sk_settings")
            with psycopg.connect(journal_dsn, autocommit=True) as journal_owner:
                identity = PostgresAllocationJournal.provision(
                    journal_owner, database_id=database_id, namespace="erasure-test", roles=roles
                )
            logins = {
                role: PostgresAllocationJournal(make_conninfo(journal_dsn, user=login, password=journal_password), identity, role)
                for login, role in roles.items()
            }
            journal = PostgresObjectJournal(
                anonymous=logins[JournalRole("anonymous")],
                binder=logins[JournalRole("binder", TENANT_A)],
                scope=logins[JournalRole("scope", TENANT_A)],
                keys=logins[JournalRole("key-intent", TENANT_A)],
            )
            cleanup = logins[JournalRole("cleanup", TENANT_A)]
            client = _EraseClient(journal, cleanup)
            content = _PostgresObjectContent(
                journal, S3CiphertextStore(client, bucket="fixture", journal=journal), provider._provision_exact_key_lifecycle()
            )
            _provision_postgres_object_content(primary_owner, key_provider=provider, objects=content)
            yield tenant_dsn, provider, content, cleanup, logins[JournalRole("cleanup", TENANT_B)], client
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(journal_name)))
            for login in roles:
                admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(login)))


@pytest.mark.parametrize("operation", ["forget", "purge"])
def test_signed_object_root_erasure_requires_authority_and_resumes(erasure_storage, monkeypatch, operation):
    dsn, provider, objects, cleanup, foreign, client = erasure_storage
    signer = Ed25519LedgerSigner("fixture", Ed25519PrivateKey.generate())
    currentness = True
    previous_proof = None

    def prove_currentness(challenge: _ErasureCurrentnessChallenge) -> _ErasureCurrentnessChallenge | None:
        nonlocal previous_proof
        assert challenge.binding_digest == objects._binding_digest()
        assert challenge.tenant_id == TENANT_A and challenge.session_id == "objects"
        assert challenge.journal_id == objects.journal.identity.journal_id
        if not currentness:
            return previous_proof
        previous_proof = challenge
        return challenge

    def open_deletion(store, authority=None):
        store.configure_input_admission(signer)
        bind_content_runtime(store, key_provider=provider, cipher=AESGCMCipher(), _object_content=objects)
        return SessionDeletionStore(store, PostgresContentStore(store, provider), _postgres_root_erasure=authority)

    def snapshot(store):
        (state,) = store._one("SELECT deletion_state FROM sk_session WHERE session_id='objects'")
        return state, store._tombstone("objects")

    authority = _PostgresRootErasure(objects, cleanup, prove_currentness, terminal_bound_delete=True)
    with PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as store:
        deletion = open_deletion(store)
        with closing(SessionDriver(store=store)) as driver:
            driver.create("objects", "/erase")
            (created,) = driver.store.read("objects", limit=1).records
            actor = Principal.from_dict(created.record.payload["principal"])
        (key_id,) = store._one("SELECT key_id FROM sk_session_key WHERE session_id='objects'")
        before = dict(client.objects)

        def erase(deletion):
            method = deletion.purge_session if operation == "purge" else deletion.forget_session
            return method(TENANT_A, "objects", actor=actor)

        with pytest.raises(ContentCleanupUnavailable, match="explicit cleanup authority"):
            erase(deletion)
        wrong = _PostgresRootErasure(objects, foreign, prove_currentness, terminal_bound_delete=True)
        with pytest.raises(ContentCleanupUnavailable, match="authenticated tenant cleanup role"):
            erase(open_deletion(store, wrong))
        wrong_scope = replace(authority, prove_currentness=lambda proof: replace(proof, session_id="other"))
        with pytest.raises(ContentCleanupUnavailable, match="current storage evidence"):
            erase(open_deletion(store, wrong_scope))
        deletion = open_deletion(store, authority)
        currentness = False
        with pytest.raises(ContentCleanupUnavailable, match="current storage evidence"):
            erase(deletion)
        currentness = True
        assert snapshot(store) == ("live", None)
        assert client.objects == before
        with store.connection.transaction(), pytest.raises(RuntimeError, match="idle connection"):
            erase(deletion)
        with (
            PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as other,
            other._object_publication_atomic(
                objects.journal, check_deadline=lambda: None, content_digest=objects._binding_digest()
            ),
            pytest.raises(Conflict, match="publication is busy"),
        ):
            erase(deletion)
        assert snapshot(store) == ("live", None)
        with PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as observer:
            observer.configure_input_admission(signer)
            seen_fenced = []
            delete_object = client.delete_object

            def lost_delete_ack(**request):
                state, tombstone = snapshot(observer)
                assert state == ("purging" if operation == "purge" else "forgetting")
                assert tombstone is not None and tombstone.state == "pending"
                observer._verify_tombstone_object(tombstone, signer, allow_unsigned=False)
                seen_fenced.append(request["Key"])
                delete_object(**request)
                raise TimeoutError("fixture lost terminal acknowledgement")

            with monkeypatch.context() as patch, pytest.raises(ContentCleanupUnavailable, match="did not confirm"):
                patch.setattr(client, "delete_object", lost_delete_ack)
                erase(deletion)
            assert len(seen_fenced) == 1
            assert seen_fenced[0] not in client.objects
            assert len(client.objects) == len(before) - 1
            claims = cleanup.scoped_allocations(TENANT_A, ("objects",))
            assert sum(claim.state == "reaping" for claim in claims) == 1
            assert not any(claim.delete_confirmed for claim in claims)
            intent = cleanup.get_key_intent(objects.provider_instance_id, key_id)
            assert intent is not None and intent.state == "retired"

    with PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as restored:
        resumed = open_deletion(restored, authority)
        currentness = False
        with pytest.raises(ContentCleanupUnavailable, match="current storage evidence"):
            erase(resumed)
        assert len(client.objects) == len(before) - 1
        currentness = True
        verify_completed = resumed._verify_purge_outcome if operation == "purge" else resumed._verify_forget_outcome
        verify_name = "_verify_purge_outcome" if operation == "purge" else "_verify_forget_outcome"

        def lose_receipt(*args, **kwargs):
            verify_completed(*args, **kwargs)
            raise TimeoutError("fixture lost response after committed completion")

        with monkeypatch.context() as patch, pytest.raises(TimeoutError, match="committed completion"):
            patch.setattr(resumed, verify_name, lose_receipt)
            erase(resumed)

    with PostgresStore.tenant_standalone(dsn, tenant_id=TENANT_A) as restored:
        resumed = open_deletion(restored, authority)
        receipt = erase(resumed)
        assert receipt.operation == operation and receipt.authenticity is DeletionAuthenticity.SIGNED
        assert erase(resumed) == receipt
        assert not client.objects
        assert cleanup.scoped_allocations(TENANT_A, ("objects",)) == ()
        assert cleanup.scoped_key_intents(TENANT_A, ("objects",)) == ()
        assert not restored._rows("SELECT 1 FROM sk_session_key WHERE session_id='objects'")
        assert not restored._rows("SELECT 1 FROM sk_content WHERE session_id='objects'")
        assert not restored._rows("SELECT 1 FROM sk_content_link WHERE session_id='objects'")
        tombstone = restored._tombstone("objects")
        assert tombstone is not None and tombstone.state == "complete"
        restored._verify_tombstone_object(tombstone, signer, allow_unsigned=False)
        assert restored._rows("SELECT deletion_state FROM sk_session WHERE session_id='objects'") == (
            [("forgotten",)] if operation == "forget" else []
        )
