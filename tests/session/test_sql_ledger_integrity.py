from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import ContentBindingMismatch
from vv_agent.kernel._records import InboxItem
from vv_agent.kernel._sql import InputAdmissionProofError, LedgerIntegrityError
from vv_agent.session.records import SessionSpec
from vv_agent.session.sqlite import SQLiteContentStore, SQLiteStore

from .helpers import content_ref, item, principal, record

EMPTY_CHAIN = "a2d9c7a09d36dc9a7d96c74bc641caa1d46fecffb66b884945e32c5d03cf2409"


class Signer:
    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.key_id = "ledger-key"
        self.fail = False
        self.reject_signed_messages = False
        self.reject_input_messages = False

    def bind_key(self):
        signer = self

        class Bound:
            key_id = signer.key_id

            @staticmethod
            def sign(message: bytes) -> bytes:
                if signer.fail:
                    raise RuntimeError("signing failed")
                return signer.key.sign(message)

        return Bound()

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        if (
            self.reject_signed_messages
            or (self.reject_input_messages and b'"domain":"vv-agent/S1/input"' in message)
            or key_id != self.key_id
        ):
            return False
        try:
            self.key.public_key().verify(signature, message)
        except InvalidSignature:
            return False
        return True


def create(store: SQLiteStore) -> None:
    with store.atomic() as tx:
        tx.create(
            SessionSpec("s", principal(), content_ref("workspace"), attributes=content_ref("attributes")),
            consumers=(),
        )


def extend(previous: str, seq: int, record_digest: str) -> str:
    return sha256(bytes.fromhex(previous) + seq.to_bytes(8, "big") + bytes.fromhex(record_digest)).hexdigest()


def test_chain_vector_uses_binary_length_framing_and_sequence(tmp_path):
    with SQLiteStore.standalone(str(tmp_path / "chain.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        create(store)
        body, digest, chain = store._one("SELECT body,digest,chain_digest FROM sk_record WHERE seq=1")
        assert sha256(body).hexdigest() == digest
        assert chain == extend(EMPTY_CHAIN, 1, digest)

        lease = store.acquire("s", owner="worker", ttl_ms=10_000)
        assert lease is not None
        with store.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=1,
                commit_id="turn",
                records=(record("turn_started"),),
            )
        next_digest, next_chain = store._one("SELECT digest,chain_digest FROM sk_record WHERE seq=2")
        assert next_chain == extend(chain, 2, next_digest)
        commit = store._one("SELECT body,digest FROM sk_commit WHERE commit_id='turn'")
        manifest = json.loads(commit[0])
        assert manifest["head_seq"] == 1
        assert manifest["head_chain_digest"] == chain
        assert manifest["records"] == [{"record_id": record("turn_started").record_id, "digest": next_digest}]
        assert commit[0] == canonical_json_bytes(manifest)
        assert commit[1] == sha256(commit[0]).hexdigest()


@pytest.mark.parametrize(
    "column,value",
    [
        ("record_id", "forged-record"),
        ("schema_version", "S2"),
        ("kind", "usage_observed"),
        ("turn_id", "forged-turn"),
        ("operation_id", "forged-operation"),
        ("attempt", 1),
    ],
)
def test_cached_record_body_requires_matching_fresh_sql_identity(tmp_path, column, value):
    with SQLiteStore.standalone(str(tmp_path / f"identity-{column}.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        create(store)
        lease = store.acquire("s", owner="worker", ttl_ms=10_000)
        assert lease is not None
        with store.atomic() as tx:
            tx.append("s", lease=lease, expected_seq=1, commit_id="turn", records=(record("turn_started", sid="s"),))
        assert store.read("s").head_seq == 2

        if column == "schema_version":
            store.connection.execute("PRAGMA ignore_check_constraints=ON")
            try:
                store.connection.execute(f"UPDATE sk_record SET {column}=? WHERE seq=2", (value,))
            finally:
                store.connection.execute("PRAGMA ignore_check_constraints=OFF")
        else:
            store.connection.execute(f"UPDATE sk_record SET {column}=? WHERE seq=2", (value,))
        with pytest.raises(LedgerIntegrityError, match="SQL identity does not match its body at seq 2"):
            store.read("s")


@pytest.mark.parametrize(
    "target,message",
    [
        ("chain", "chain digest mismatch at seq 2"),
        ("checkpoint", "checkpoint signature is invalid at seq 2"),
        ("commit", "commit body digest mismatch"),
    ],
)
def test_warm_record_bytes_still_revalidate_chain_checkpoints_and_commits(tmp_path, target, message):
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / f"warm-{target}.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        create(store)
        lease = store.acquire("s", owner="worker", ttl_ms=10_000)
        assert lease is not None
        with store.atomic() as tx:
            tx.append("s", lease=lease, expected_seq=1, commit_id="turn", records=(record("turn_started", sid="s"),))
        assert store.read("s").head_seq == 2

        if target == "chain":
            store.connection.execute("UPDATE sk_record SET chain_digest=? WHERE seq=2", ("0" * 64,))
        elif target == "checkpoint":
            store.connection.execute("UPDATE sk_checkpoint SET signature=? WHERE seq=2", (b"forged",))
        else:
            store.connection.execute("UPDATE sk_commit SET body=? WHERE commit_id='turn'", (b"{}",))

        with pytest.raises(LedgerIntegrityError, match=message):
            store.read("s")


def test_signed_checkpoint_rejects_recomputed_keyless_chain(tmp_path):
    path = tmp_path / "tamper.sqlite"
    signer = Signer()
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        create(store)
        body = store._one("SELECT body FROM sk_record WHERE seq=1")[0]
        value = json.loads(body)
        value["payload"]["attributes"]["content_id"] = "content/test/tampered"
        tampered = canonical_json_bytes(value)
        tampered_digest = sha256(tampered).hexdigest()
        tampered_chain = extend(EMPTY_CHAIN, 1, tampered_digest)
        store.connection.execute(
            "UPDATE sk_record SET body=?,digest=?,chain_digest=? WHERE seq=1",
            (tampered, tampered_digest, tampered_chain),
        )
        manifest = json.loads(store._one("SELECT body FROM sk_commit WHERE commit_id='session/create'")[0])
        manifest["records"][0]["digest"] = tampered_digest
        commit_body = canonical_json_bytes(manifest)
        store.connection.execute(
            "UPDATE sk_commit SET body=?,digest=? WHERE commit_id='session/create'",
            (commit_body, sha256(commit_body).hexdigest()),
        )
        store.connection.execute("UPDATE sk_record SET commit_digest=? WHERE seq=1", (sha256(commit_body).hexdigest(),))
        store.connection.execute("UPDATE sk_checkpoint SET chain_digest=? WHERE seq=1", (tampered_chain,))

    with SQLiteStore.standalone(str(path)) as reopened:
        reopened.configure_input_admission(signer)
        with pytest.raises(LedgerIntegrityError, match="checkpoint signature"):
            reopened.read("s")


def test_recovery_rejects_sequence_gap_after_recomputed_digest(tmp_path):
    path = tmp_path / "gap.sqlite"
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        create(store)
        lease = store.acquire("s", owner="worker", ttl_ms=10_000)
        assert lease is not None
        with store.atomic() as tx:
            tx.append("s", lease=lease, expected_seq=1, commit_id="turn", records=(record("turn_started"),))
        store.connection.execute("DELETE FROM sk_record WHERE seq=2")

    with SQLiteStore.standalone(str(path)) as reopened:
        reopened.configure_input_admission(allow_unsigned=True)
        with pytest.raises(LedgerIntegrityError, match="sequence gap"):
            reopened.read("s")


def test_checkpoint_signing_failure_rolls_back_append(tmp_path):
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / "rollback.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        create(store)
        lease = store.acquire("s", owner="worker", ttl_ms=10_000)
        assert lease is not None
        signer.fail = True
        with pytest.raises(RuntimeError, match="signing failed"), store.atomic() as tx:
            tx.append("s", lease=lease, expected_seq=1, commit_id="turn", records=(record("turn_started"),))
        assert store.read("s").head_seq == 1
        assert store._one("SELECT count(*) FROM sk_record")[0] == 1
        assert store._one("SELECT count(*) FROM sk_commit")[0] == 1
        assert store._one("SELECT count(*) FROM sk_checkpoint")[0] == 1


def test_signer_key_binding_is_verified_before_checkpoint_or_input_commit(tmp_path):
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / "key-binding.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        signer.reject_signed_messages = True
        with pytest.raises(LedgerIntegrityError, match=r"key binding.*seq 1"), store.atomic() as tx:
            tx.create(
                SessionSpec("s", principal(), content_ref("workspace"), attributes=content_ref("attributes")),
                consumers=(),
            )
        assert store._one("SELECT count(*) FROM sk_session")[0] == 0
        assert store._one("SELECT count(*) FROM sk_checkpoint")[0] == 0

        signer.reject_signed_messages = False
        create(store)
        signer.reject_input_messages = True
        with pytest.raises(InputAdmissionProofError, match=r"key binding.*input_seq 1"), store.atomic() as tx:
            tx.push("s", item("input"))
        assert store._one("SELECT count(*) FROM sk_inbox")[0] == 0
        assert store._one("SELECT inbox_seq FROM sk_session WHERE session_id='s'")[0] == 0


def test_signed_input_duplicate_reuses_and_verifies_the_original_proof(tmp_path):
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / "input-replay.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        create(store)
        with store.atomic() as tx:
            first = tx.push("s", item("input"))
        stored = store.read_input("s", "input")
        assert stored is not None and stored.item.admission_proof is not None

        with store.atomic() as tx:
            replay = tx.push("s", stored.item)
        assert replay.replayed is True
        assert (replay.input_seq, replay.received_ms) == (first.input_seq, first.received_ms)
        assert store.read_input("s", "input") == stored

        proof = stored.item.admission_proof
        signature = proof["admission_signature"]
        forged = replace(
            stored.item,
            admission_proof=proof | {"admission_signature": ("A" if signature[0] != "A" else "B") + signature[1:]},
        )
        with pytest.raises(InputAdmissionProofError, match="does not match the original"), store.atomic() as tx:
            tx.push("s", forged)
        assert store._one("SELECT count(*) FROM sk_inbox")[0] == 1


def test_expected_head_gates_reads_and_appends_before_signing(tmp_path):
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / "expected-head.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        create(store)
        first_chain = store._one("SELECT chain_digest FROM sk_record WHERE seq=1")[0]
        lease = store.acquire("s", owner="worker", ttl_ms=10_000)
        assert lease is not None
        with store.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=1,
                commit_id="first-tail",
                records=(record("turn_started", sid="s"),),
            )
        second_chain = store._one("SELECT chain_digest FROM sk_record WHERE seq=2")[0]
        assert store.read("s", expected_head=(2, second_chain)).head_seq == 2
        with pytest.raises(ContentBindingMismatch, match=r"expected_head.*seq 2"):
            store.read("s", expected_head=(2, first_chain))

        assert store.release(lease)
        lease = store.acquire("s", owner="worker", ttl_ms=10_000)
        assert lease is not None
        before = store._one("SELECT count(*) FROM sk_checkpoint")[0]
        with pytest.raises(ContentBindingMismatch, match=r"expected_head.*seq 2"), store.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=2,
                commit_id="wrong-anchor",
                records=(record("usage_observed", sid="s", meter_id="m", observation=1),),
                expected_head=(2, first_chain),
            )
        assert store._one("SELECT count(*) FROM sk_checkpoint")[0] == before
        assert store._head("s")[0] == 2
        with store.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=2,
                commit_id="correct-anchor",
                records=(record("usage_observed", sid="s", meter_id="m", observation=1),),
                expected_head=(2, second_chain),
            )
        assert store._head("s")[0] == 3
        assert store._one("SELECT count(*) FROM sk_checkpoint")[0] == before + 1


def test_expected_head_detects_deleted_signed_tail_without_a_db_derived_anchor(tmp_path):
    path = tmp_path / "rollback.sqlite"
    signer = Signer()
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        create(store)
        lease = store.acquire("s", owner="worker", ttl_ms=10_000)
        assert lease is not None
        with store.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=1,
                commit_id="tail",
                records=(record("turn_started", sid="s"),),
            )
        trusted_head = (
            2,
            store._one("SELECT chain_digest FROM sk_record WHERE seq=2")[0],
        )
        # Renewal must prove the intact source before it grants a lease. Keep this
        # earlier lease valid while simulating a deleted persisted tail.
        assert lease.expires_at_ms > store._now()
        store.connection.execute("DELETE FROM sk_checkpoint WHERE seq=2")
        store.connection.execute("DELETE FROM sk_record WHERE seq=2")
        store.connection.execute("DELETE FROM sk_commit WHERE commit_id='tail'")
        store.connection.execute("UPDATE sk_session SET head_seq=1 WHERE session_id='s'")

    with SQLiteStore.standalone(str(path)) as reopened:
        reopened.configure_input_admission(signer)
        assert reopened.verify_ledger("s") == (1, reopened._one("SELECT chain_digest FROM sk_record WHERE seq=1")[0])
        with pytest.raises(ContentBindingMismatch, match=r"expected_head.*seq 1"):
            reopened.read("s", expected_head=trusted_head)
        with pytest.raises(ContentBindingMismatch, match=r"expected_head.*seq 1"), reopened.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=1,
                commit_id="after-rollback",
                records=(record("turn_started", sid="s", tid="after-rollback"),),
                expected_head=trusted_head,
            )
        assert reopened._head("s")[0] == 1
        assert reopened._one("SELECT count(*) FROM sk_checkpoint")[0] == 1


def test_unsigned_or_rehashed_tail_after_checkpoint_256_cannot_be_extended(tmp_path):
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / "unsigned-tail.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        create(store)
        lease = store.acquire("s", owner="worker", ttl_ms=10_000)
        assert lease is not None
        observations = (
            record("turn_started", sid="s"),
            *(record("usage_observed", sid="s", meter_id="m", observation=number) for number in range(1, 259)),
        )
        with store.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=1,
                commit_id="tail-257-260",
                records=observations,
            )
        assert store._head("s")[0] == 260
        assert store._one("SELECT count(*) FROM sk_checkpoint")[0] == 260

        store.connection.execute("DELETE FROM sk_checkpoint WHERE seq>256")
        previous = store._one("SELECT chain_digest FROM sk_record WHERE seq=256")[0]
        for seq in range(257, 261):
            body, _digest = store._one("SELECT body,digest FROM sk_record WHERE seq=?", (seq,))
            value = json.loads(body)
            if value["kind"] == "usage_observed":
                value["payload"]["observation"] += 1000
            else:
                value["payload"]["handler_version"] = "forged"
            forged = canonical_json_bytes(value)
            forged_digest = sha256(forged).hexdigest()
            previous = extend(previous, seq, forged_digest)
            store.connection.execute(
                "UPDATE sk_record SET body=?,digest=?,chain_digest=? WHERE seq=?",
                (forged, forged_digest, previous, seq),
            )

        with pytest.raises(LedgerIntegrityError, match="checkpoint is missing at seq 257"), store.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=260,
                commit_id="honest-extension",
                records=(record("usage_observed", sid="s", meter_id="m", observation=260),),
            )
        assert store._head("s")[0] == 260
        assert store._one("SELECT count(*) FROM sk_record")[0] == 260
        assert store._one("SELECT count(*) FROM sk_checkpoint")[0] == 256


def test_existing_hash_only_history_cannot_be_retroactively_signed(tmp_path):
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / "retroactive-sign.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        create(store)
        store.configure_input_admission(signer)
        with pytest.raises(LedgerIntegrityError, match="checkpoint is missing at seq 1"):
            store.acquire("s", owner="worker", ttl_ms=10_000)
        assert store._head("s")[0] == 1
        assert store._one("SELECT count(*) FROM sk_record")[0] == 1
        assert store._one("SELECT count(*) FROM sk_checkpoint")[0] == 0


def test_tampered_signed_input_blocks_recovery_before_any_append_or_content_use(tmp_path, monkeypatch):
    signer = Signer()
    with SQLiteStore.standalone(str(tmp_path / "input-proof.sqlite")) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        create(store)
        with store.atomic() as tx:
            tx.push("s", item("input", actor=principal("attacker", "user")))
        assert store.read("s").head_seq == 1
        lease = store.acquire("s", owner="worker", ttl_ms=10_000)
        assert lease is not None
        body, checksum = store._one("SELECT body,digest FROM sk_inbox WHERE input_id='input'")
        original = InboxItem.parse(body)
        assert original.actor is not None
        forged = replace(original, actor=original.actor | {"subject_id": "forged"})
        forged_body = forged.encode()
        store.connection.execute(
            "UPDATE sk_inbox SET body=?,digest=? WHERE input_id='input'",
            (forged_body, sha256(forged_body).hexdigest()),
        )

        def forbidden_content_read(*_args, **_kwargs):
            raise AssertionError("content must not be read before input admission verification")

        monkeypatch.setattr(SQLiteContentStore, "get", forbidden_content_read)
        with pytest.raises(InputAdmissionProofError, match="input_seq 1"):
            store.read("s")
        with pytest.raises(InputAdmissionProofError, match="input_seq 1"), store.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=1,
                commit_id="must-not-land",
                records=(record("turn_started", sid="s"),),
            )
        assert store._head("s")[0] == 1
        assert store._one("SELECT count(*) FROM sk_record")[0] == 1
        assert store._one("SELECT count(*) FROM sk_checkpoint")[0] == 1
        assert checksum != sha256(forged_body).hexdigest()
        assert store.release(lease)
