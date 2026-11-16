from __future__ import annotations

import json
import os
from hashlib import sha256

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import ContentBindingMismatch, Principal
from vv_agent.kernel._sql import (
    InputAdmissionProofError,
    LedgerIntegrityError,
    _extend_chain_digest,
    _initial_chain_digest,
)
from vv_agent.session.content_store import LocalKeyProvider
from vv_agent.session.export import export_session
from vv_agent.session.records import SessionSpec
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import SessionDriver

from .helpers import content_ref, item, principal, record


class Signer:
    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.key_id = "integrity-consumer-key"

    def bind_key(self):
        signer = self

        class Bound:
            key_id = signer.key_id

            @staticmethod
            def sign(message: bytes) -> bytes:
                return signer.key.sign(message)

        return Bound()

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        if key_id != self.key_id:
            return False
        try:
            self.key.public_key().verify(signature, message)
        except InvalidSignature:
            return False
        return True


def _create(store: SQLiteStore, session_id: str = "s", *, parent: str | None = None, consumers=("events",)) -> None:
    with store.atomic() as tx:
        tx.create(
            SessionSpec(
                session_id,
                principal(f"owner/{session_id}"),
                content_ref(f"workspace/{session_id}"),
                parent_session_id=parent,
                parent_operation_id=f"operation/{session_id}" if parent is not None else None,
                attributes=content_ref(f"attributes/{session_id}"),
            ),
            consumers=consumers,
        )


def _actor(subject_id: str) -> Principal:
    return Principal.from_dict(principal(subject_id))


def _rewrite_records_and_commits(store: SQLiteStore, changed_seq: int, changed_body: bytes) -> None:
    rows = store._rows(
        "SELECT seq,body,digest FROM sk_record WHERE tenant_id=%s AND session_id='s' ORDER BY seq",
        (store.tenant_id,),
    )
    digests = {seq: digest for seq, _body, digest in rows}
    bodies = {seq: body for seq, body, _digest in rows}
    bodies[changed_seq] = changed_body
    digests[changed_seq] = sha256(changed_body).hexdigest()

    chains: dict[int, str] = {}
    previous = _initial_chain_digest(store.tenant_id, "s")
    for seq, _body, _digest in rows:
        previous = _extend_chain_digest(previous, seq, digests[seq])
        chains[seq] = previous
        store.connection.execute(
            "UPDATE sk_record SET body=?,digest=?,chain_digest=? WHERE seq=?",
            (bodies[seq], digests[seq], chains[seq], seq),
        )

    for commit_id, commit_body in store._rows(
        "SELECT commit_id,body FROM sk_commit WHERE tenant_id=%s AND session_id='s'",
        (store.tenant_id,),
    ):
        manifest = json.loads(commit_body)
        for reference in manifest["records"]:
            reference["digest"] = digests[_record_seq(store, reference["record_id"])]
        head_seq = manifest["head_seq"]
        manifest["head_chain_digest"] = _initial_chain_digest(store.tenant_id, "s") if head_seq == 0 else chains[head_seq]
        rewritten = canonical_json_bytes(manifest)
        commit_digest = sha256(rewritten).hexdigest()
        store.connection.execute(
            "UPDATE sk_commit SET body=?,digest=? WHERE commit_id=?",
            (rewritten, commit_digest, commit_id),
        )
        store.connection.execute(
            "UPDATE sk_record SET commit_digest=? WHERE commit_id=?",
            (commit_digest, commit_id),
        )
    for seq, chain in chains.items():
        store.connection.execute("UPDATE sk_checkpoint SET chain_digest=? WHERE seq=?", (chain, seq))


def _record_seq(store: SQLiteStore, record_id: str) -> int:
    return store._one("SELECT seq FROM sk_record WHERE record_id=%s", (record_id,))[0]


def _tamper_record_payload(store: SQLiteStore, seq: int, mutate) -> None:
    body = store._one("SELECT body FROM sk_record WHERE seq=%s", (seq,))[0]
    value = json.loads(body)
    mutate(value["payload"])
    _rewrite_records_and_commits(store, seq, canonical_json_bytes(value))


def _append_turn(store: SQLiteStore) -> None:
    lease = store.acquire("s", owner="writer", ttl_ms=10_000)
    assert lease is not None
    with store.atomic() as tx:
        tx.append(
            "s",
            lease=lease,
            expected_seq=1,
            commit_id="turn",
            records=(
                record("turn_started", sid="s"),
                record("op_planned", sid="s"),
                record("op_started", sid="s"),
                record("op_completed", sid="s"),
            ),
        )


def test_consumers_and_projection_reject_rehashed_usage_record(tmp_path):
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / "consumer-tamper.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        _create(store)
        _append_turn(store)
        completed = next(stored for stored in store.read("s").records if stored.record.kind == "op_completed")

        _tamper_record_payload(
            store,
            completed.seq,
            lambda payload: payload["usage"].__setitem__("content_id", "content/test/forged-usage"),
        )

        with pytest.raises(LedgerIntegrityError, match="checkpoint signature"), store.atomic() as tx:
            tx.consumer_batch("s", "events")
        with pytest.raises(LedgerIntegrityError, match="checkpoint signature"):
            store.control_records("s", through_seq=store._head("s")[0])
        with pytest.raises(LedgerIntegrityError, match="checkpoint signature"):
            store.projection_context("s", (completed,))
        assert store._one("SELECT last_seq FROM sk_consumer WHERE consumer='events'")[0] == 0


def test_deleted_signed_proofs_and_keyless_rehash_cannot_downgrade_trusted_paths(tmp_path):
    path = tmp_path / "proof-deletion.sqlite"
    signer = Signer()
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        _create(store)
        with store.atomic() as tx:
            tx.push("s", item("pending"))
        store.connection.execute("DELETE FROM sk_checkpoint")
        input_body = store._one("SELECT body FROM sk_inbox WHERE input_id='pending'")[0]
        input_value = json.loads(input_body)
        input_value["payload"]["content"]["content_id"] = "content/test/forged-input"
        forged_input = canonical_json_bytes(input_value)
        store.connection.execute(
            "UPDATE sk_inbox SET body=?,digest=?,admission_key_id=NULL,admission_signature=NULL,"
            "admission_signed_ms=NULL WHERE input_id='pending'",
            (forged_input, sha256(forged_input).hexdigest()),
        )
        record_body = store._one("SELECT body FROM sk_record WHERE seq=1")[0]
        record_value = json.loads(record_body)
        record_value["payload"]["workspace"]["content_id"] = "content/test/forged-workspace"
        _rewrite_records_and_commits(store, 1, canonical_json_bytes(record_value))

    with SQLiteStore.standalone(str(path)) as reopened:
        reopened.configure_input_admission(signer)
        for operation in (
            lambda: reopened.read_state("s"),
            lambda: reopened.verify_ledger("s"),
            lambda: export_session(reopened, "s"),
        ):
            with pytest.raises(LedgerIntegrityError, match="checkpoint is missing at seq 1"):
                operation()
        with pytest.raises(LedgerIntegrityError, match="checkpoint is missing at seq 1"), reopened.atomic() as tx:
            tx.consumer_batch("s", "events")

        lease = reopened.acquire("s", owner="writer", ttl_ms=10_000)
        assert lease is not None
        with pytest.raises(LedgerIntegrityError, match="checkpoint is missing at seq 1"), reopened.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=1,
                commit_id="must-not-extend",
                records=(record("turn_started", sid="s"),),
            )
        assert reopened._head("s")[0] == 1

    with SQLiteStore.standalone(str(path)) as hash_only:
        hash_only.configure_input_admission(allow_unsigned=True)
        assert hash_only.verify_ledger("s")[0] == 1
        assert hash_only._ledger_integrity_mode() == "hash-only"
        chain = hash_only._one("SELECT chain_digest FROM sk_record WHERE seq=1")[0]
        assert hash_only.read_state("s", expected_head=(1, chain))[2] == 1
        with pytest.raises(ContentBindingMismatch, match="expected_head"):
            hash_only.read_state("s", expected_head=(2, chain))


def test_cached_prefix_does_not_skip_input_proof_revalidation_before_append(tmp_path):
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / "cached-input-proof.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        _create(store)
        with store.atomic() as tx:
            tx.push("s", item("pending"))
        store.read_state("s")
        store.connection.execute("UPDATE sk_inbox SET admission_key_id=NULL,admission_signature=NULL,admission_signed_ms=NULL")

        lease = store.acquire("s", owner="writer", ttl_ms=10_000)
        assert lease is not None
        with pytest.raises(InputAdmissionProofError, match="proof is missing"), store.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=1,
                commit_id="must-not-extend",
                records=(record("turn_started", sid="s"),),
            )


def test_trusted_verify_and_export_require_explicit_integrity_mode(tmp_path):
    with SQLiteStore.standalone(str(tmp_path / "unconfigured.sqlite")) as store:
        store.install_schema()
        _create(store)

        assert store.read_state("s")[2] == 0
        chain = store._one("SELECT chain_digest FROM sk_record WHERE seq=1")[0]
        with pytest.raises(LedgerIntegrityError, match="explicit signed signer or allow_unsigned hash-only"):
            store.read_state("s", expected_head=(1, chain))
        with pytest.raises(LedgerIntegrityError, match="explicit signed signer or allow_unsigned hash-only"):
            store.verify_ledger("s")
        with pytest.raises(LedgerIntegrityError, match="explicit signed signer or allow_unsigned hash-only"):
            export_session(store, "s")


def test_sql_deletion_rebinds_descendant_and_purge_actor_with_new_signatures(tmp_path):
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / "deletion.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        _create(store, "root", consumers=())
        _create(store, "child", parent="root", consumers=())

        cached_bound = signer.bind_key()
        store.begin_forget(
            "child",
            _actor("child-forget"),
            bound_signer=cached_bound,
            verifier=signer,
            allow_unsigned=False,
        )
        store.complete_forget("child")
        assert store._one("SELECT scope_root_session_id,actor FROM sk_tombstone WHERE session_id='child'")[0] == "child"

        store.begin_forget(
            "root",
            _actor("parent-forget"),
            bound_signer=cached_bound,
            verifier=signer,
            allow_unsigned=False,
        )
        store.complete_forget("root")
        for session_id in ("root", "child"):
            tombstone = store._tombstone(session_id)
            assert tombstone is not None
            assert tombstone.scope_root_session_id == "root"
            assert tombstone.actor.subject_id == "parent-forget"
            store._verify_tombstone_object(tombstone, signer, allow_unsigned=False)

        store.begin_purge(
            "root",
            _actor("purger"),
            bound_signer=cached_bound,
            verifier=signer,
            allow_unsigned=False,
        )
        assert set(store.complete_purge("root")) == {"root", "child"}
        for session_id in ("root", "child"):
            tombstone = store._tombstone(session_id)
            assert tombstone is not None
            assert tombstone.state.value == "complete"
            assert tombstone.scope_root_session_id == "root"
            assert tombstone.actor.subject_id == "purger"
            store._verify_tombstone_object(tombstone, signer, allow_unsigned=False)


def test_session_driver_requires_explicit_content_keys_for_durable_stores(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="explicit content_key_provider"):
        SessionDriver(tmp_path / "durable.sqlite")

    shared_state = tmp_path / "ambient-local-key-state.sqlite3"
    monkeypatch.setenv("VV_AGENT_LOCAL_KEY_STATE", os.fspath(shared_state))
    first = SessionDriver()
    second = SessionDriver()
    try:
        assert first.content_key_provider is not second.content_key_provider
        assert not shared_state.exists()
    finally:
        first.close()
        second.close()

    provider = LocalKeyProvider(b"k" * 32, state_path=tmp_path / "explicit-key-state.sqlite3")
    explicit = SessionDriver(tmp_path / "explicit.sqlite", content_key_provider=provider)
    explicit.close()
