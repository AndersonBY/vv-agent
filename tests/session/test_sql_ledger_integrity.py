from __future__ import annotations

import json
from hashlib import sha256

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._sql import LedgerIntegrityError
from vv_agent.session.records import SessionSpec
from vv_agent.session.sqlite import SQLiteStore

from .helpers import content_ref, principal, record

EMPTY_CHAIN = "a2d9c7a09d36dc9a7d96c74bc641caa1d46fecffb66b884945e32c5d03cf2409"


class Signer:
    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.key_id = "ledger-key"
        self.fail = False

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
        if key_id != self.key_id:
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
        create(store)
        lease = store.acquire("s", owner="worker", ttl_ms=10_000)
        assert lease is not None
        with store.atomic() as tx:
            tx.append("s", lease=lease, expected_seq=1, commit_id="turn", records=(record("turn_started"),))
        store.connection.execute("DELETE FROM sk_record WHERE seq=2")

    with SQLiteStore.standalone(str(path)) as reopened, pytest.raises(LedgerIntegrityError, match="sequence gap"):
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
