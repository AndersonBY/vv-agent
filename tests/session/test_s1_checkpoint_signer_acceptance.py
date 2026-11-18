"""Real signer and checkpoint acceptance coverage for trusted storage S1-A7."""

from __future__ import annotations

import json
from hashlib import sha256

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import ContentBindingMismatch, SessionHead
from vv_agent.kernel._sql import (
    LedgerIntegrityError,
    SQLStore,
    _extend_chain_digest,
    _initial_chain_digest,
)
from vv_agent.session.records import SessionSpec

from .conftest import Database, open_store
from .helpers import content_ref, principal, record

SESSION = "s"


class Signer:
    def __init__(self, key_id: str = "checkpoint-key") -> None:
        self.key = Ed25519PrivateKey.generate()
        self.key_id = key_id
        self.bind_count = 0
        self.sign_count = 0

    def bind_key(self):
        self.bind_count += 1
        signer = self

        class Bound:
            key_id = signer.key_id

            @staticmethod
            def sign(message: bytes) -> bytes:
                signer.sign_count += 1
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


class RotatingVerifier:
    def __init__(self, *signers: Signer) -> None:
        self.signers = {signer.key_id: signer for signer in signers}

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        signer = self.signers.get(key_id)
        return signer is not None and signer.verify(key_id, message, signature)


class PreservingRotationSigner(RotatingVerifier):
    """Rotate after binding while keeping the returned binding's key immutable."""

    def __init__(self, old: Signer, new: Signer) -> None:
        super().__init__(old, new)
        self.old = old
        self.new = new
        self.rotate_on_bind = True

    def bind_key(self):
        active = self.old if self.rotate_on_bind else self.new
        bound = active.bind_key()
        if self.rotate_on_bind:
            self.rotate_on_bind = False
        return bound


class DriftingRotationSigner(RotatingVerifier):
    """Rotate the signing material after returning the previous key identity."""

    def __init__(self, old: Signer, new: Signer) -> None:
        super().__init__(old, new)
        self.old = old
        self.new = new

    def bind_key(self):
        signer = self

        class Bound:
            key_id = signer.old.key_id

            @staticmethod
            def sign(message: bytes) -> bytes:
                return signer.new.key.sign(message)

        bound = Bound()
        return bound


def _create(store: SQLStore) -> None:
    with store.atomic() as tx:
        tx.create(
            SessionSpec(
                SESSION,
                principal(),
                content_ref("workspace"),
                attributes=content_ref("attributes"),
            ),
            consumers=("events",),
        )


def _append(store: SQLStore, lease, *, expected_seq: int, commit_id: str, records) -> None:
    with store.atomic() as tx:
        tx.append(
            SESSION,
            lease=lease,
            expected_seq=expected_seq,
            commit_id=commit_id,
            records=tuple(records),
        )


def _counts(store: SQLStore) -> tuple[int, int, int]:
    params = (store.tenant_id, SESSION)
    return (
        int(store._one("SELECT count(*) FROM sk_record WHERE tenant_id=%s AND session_id=%s", params)[0]),
        int(store._one("SELECT count(*) FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s", params)[0]),
        int(store._one("SELECT count(*) FROM sk_commit WHERE tenant_id=%s AND session_id=%s", params)[0]),
    )


def _rewrite_keyless_chain(store: SQLStore, replacements: dict[int, bytes] | None = None) -> None:
    """Rewrite every digest not protected by the signer, leaving signatures stale."""

    rows = store._rows(
        "SELECT seq,body FROM sk_record WHERE tenant_id=%s AND session_id=%s ORDER BY seq",
        (store.tenant_id, SESSION),
    )
    bodies = {seq: bytes(body) for seq, body in rows}
    if replacements:
        bodies.update(replacements)
    digests = {seq: sha256(body).hexdigest() for seq, body in bodies.items()}
    chains: dict[int, str] = {}
    previous = _initial_chain_digest(store.tenant_id, SESSION)
    for seq in sorted(bodies):
        previous = _extend_chain_digest(previous, seq, digests[seq])
        chains[seq] = previous
        store._rows(
            "UPDATE sk_record SET body=%s,digest=%s,chain_digest=%s WHERE tenant_id=%s AND session_id=%s AND seq=%s",
            (bodies[seq], digests[seq], chains[seq], store.tenant_id, SESSION, seq),
        )

    commits = store._rows(
        "SELECT commit_id,body FROM sk_commit WHERE tenant_id=%s AND session_id=%s",
        (store.tenant_id, SESSION),
    )
    for commit_id, body in commits:
        manifest = json.loads(bytes(body))
        sequences = store._rows(
            "SELECT record_id,seq FROM sk_record WHERE tenant_id=%s AND session_id=%s AND commit_id=%s ORDER BY commit_position",
            (store.tenant_id, SESSION, commit_id),
        )
        manifest["records"] = [{"record_id": record_id, "digest": digests[seq]} for record_id, seq in sequences]
        head_seq = manifest["head_seq"]
        manifest["head_chain_digest"] = _initial_chain_digest(store.tenant_id, SESSION) if head_seq == 0 else chains[head_seq]
        rewritten = canonical_json_bytes(manifest)
        commit_digest = sha256(rewritten).hexdigest()
        store._rows(
            "UPDATE sk_commit SET body=%s,digest=%s,record_sequences=%s WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
            (
                rewritten,
                commit_digest,
                canonical_json_bytes(sequences),
                store.tenant_id,
                SESSION,
                commit_id,
            ),
        )
        store._rows(
            "UPDATE sk_record SET commit_digest=%s WHERE tenant_id=%s AND session_id=%s AND commit_id=%s",
            (commit_digest, store.tenant_id, SESSION, commit_id),
        )

    for seq, chain in chains.items():
        store._rows(
            "UPDATE sk_checkpoint SET chain_digest=%s WHERE tenant_id=%s AND session_id=%s AND seq=%s",
            (chain, store.tenant_id, SESSION, seq),
        )


def _rewrite_session_actor(store: SQLStore) -> None:
    body = bytes(
        store._one(
            "SELECT body FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=1",
            (store.tenant_id, SESSION),
        )[0]
    )
    value = json.loads(body)
    value["payload"]["principal"]["subject_id"] = "sql-forged-actor"
    _rewrite_keyless_chain(store, {1: canonical_json_bytes(value)})


def _swap_records(store: SQLStore, left: int, right: int) -> None:
    fields = "record_id,schema_version,kind,turn_id,operation_id,attempt,body"
    left_row = store._one(
        f"SELECT {fields} FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=%s",
        (store.tenant_id, SESSION, left),
    )
    right_row = store._one(
        f"SELECT {fields} FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=%s",
        (store.tenant_id, SESSION, right),
    )
    temporary = f"swap:{left}:{right}"
    store._rows(
        "UPDATE sk_record SET record_id=%s WHERE tenant_id=%s AND session_id=%s AND seq=%s",
        (temporary, store.tenant_id, SESSION, left),
    )
    store._rows(
        "UPDATE sk_record SET record_id=%s WHERE tenant_id=%s AND session_id=%s AND seq=%s",
        (left_row[0], store.tenant_id, SESSION, right),
    )
    store._rows(
        "UPDATE sk_record SET schema_version=%s,kind=%s,turn_id=%s,operation_id=%s,attempt=%s,body=%s "
        "WHERE tenant_id=%s AND session_id=%s AND seq=%s",
        (*left_row[1:], store.tenant_id, SESSION, right),
    )
    store._rows(
        "UPDATE sk_record SET record_id=%s,schema_version=%s,kind=%s,turn_id=%s,"
        "operation_id=%s,attempt=%s,body=%s WHERE tenant_id=%s AND session_id=%s AND seq=%s",
        (*right_row, store.tenant_id, SESSION, left),
    )
    _rewrite_keyless_chain(store)


def _tamper_checkpoint(store: SQLStore, seq: int) -> None:
    store._rows(
        "UPDATE sk_checkpoint SET signature=%s WHERE tenant_id=%s AND session_id=%s AND seq=%s",
        (b"forged-checkpoint", store.tenant_id, SESSION, seq),
    )


def _delete_checkpoint(store: SQLStore, seq: int) -> None:
    store._rows(
        "DELETE FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s AND seq=%s",
        (store.tenant_id, SESSION, seq),
    )


@pytest.mark.persistent_store
@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("body-actor-rehash", "checkpoint signature is invalid at seq 1"),
        ("checkpoint-signature", "checkpoint signature is invalid at seq 2"),
        ("record-reorder", "checkpoint signature is invalid at seq 2"),
        ("checkpoint-delete", "checkpoint is missing at seq 4"),
    ],
)
def test_signed_tamper_is_rejected_by_every_consumer_before_append(
    database: Database,
    tamper: str,
    message: str,
) -> None:
    signer = Signer()
    with open_store(database) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        _create(store)
        lease = store.acquire(SESSION, owner="signed-writer", ttl_ms=10_000)
        assert lease is not None
        _append(
            store,
            lease,
            expected_seq=1,
            commit_id="signed-tail",
            records=(
                record("turn_started", sid=SESSION),
                record("usage_observed", sid=SESSION, meter_id="m", observation=1),
                record("usage_observed", sid=SESSION, meter_id="m", observation=2),
            ),
        )
        assert store._head(SESSION)[0] == 4
        assert _counts(store)[:2] == (4, 4)

        if tamper == "body-actor-rehash":
            _rewrite_session_actor(store)
        elif tamper == "checkpoint-signature":
            _tamper_checkpoint(store, 2)
        elif tamper == "record-reorder":
            _swap_records(store, 2, 3)
        else:
            _delete_checkpoint(store, 4)

        before = _counts(store)
        signatures = signer.sign_count
        with pytest.raises(LedgerIntegrityError, match=message):
            store.read(SESSION)
        with pytest.raises(LedgerIntegrityError, match=message):
            store.verify_ledger(SESSION)
        with pytest.raises(LedgerIntegrityError, match=message):
            store.acquire(SESSION, owner="must-not-claim", ttl_ms=10_000)
        with pytest.raises(LedgerIntegrityError, match=message), store.atomic() as tx:
            tx.append(
                SESSION,
                lease=lease,
                expected_seq=4,
                commit_id=f"must-not-land-{tamper}",
                records=(record("usage_observed", sid=SESSION, meter_id="m", observation=99),),
            )

        assert store._head(SESSION)[0] == 4
        assert _counts(store) == before
        assert signer.sign_count == signatures


def _append_checkpoint_256_tail(store: SQLStore, lease) -> None:
    records = (
        record("turn_started", sid=SESSION),
        *(record("usage_observed", sid=SESSION, meter_id="m", observation=number) for number in range(1, 259)),
    )
    _append(
        store,
        lease,
        expected_seq=1,
        commit_id="tail-2-260",
        records=records,
    )


@pytest.mark.persistent_store
def test_rewritten_unsigned_tail_after_checkpoint_256_rejects_before_signing(
    database: Database,
) -> None:
    signer = Signer()
    with open_store(database) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        _create(store)
        lease = store.acquire(SESSION, owner="signed-writer", ttl_ms=10_000)
        assert lease is not None
        _append_checkpoint_256_tail(store, lease)
        assert store._head(SESSION)[0] == 260
        assert _counts(store)[:2] == (260, 260)

        store._rows(
            "DELETE FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s AND seq>256",
            (store.tenant_id, SESSION),
        )
        replacements: dict[int, bytes] = {}
        rows = store._rows(
            "SELECT seq,body FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq>256 ORDER BY seq",
            (store.tenant_id, SESSION),
        )
        for seq, body in rows:
            value = json.loads(bytes(body))
            value["payload"]["observation"] = int(value["payload"]["observation"]) + 1000
            replacements[seq] = canonical_json_bytes(value)
        _rewrite_keyless_chain(store, replacements)

        before = _counts(store)
        signatures = signer.sign_count
        with pytest.raises(LedgerIntegrityError, match="checkpoint is missing at seq 257"), store.atomic() as tx:
            tx.append(
                SESSION,
                lease=lease,
                expected_seq=260,
                commit_id="honest-extension",
                records=(record("usage_observed", sid=SESSION, meter_id="m", observation=260),),
            )

        assert store._head(SESSION)[0] == 260
        assert _counts(store) == before
        assert before[:2] == (260, 256)
        assert signer.sign_count == signatures


@pytest.mark.persistent_store
def test_expected_head_detects_latest_signed_head_removal_without_upgrading_weak_reads(
    database: Database,
) -> None:
    signer = Signer()
    with open_store(database) as store:
        store.install_schema()
        store.configure_input_admission(signer)
        _create(store)
        lease = store.acquire(SESSION, owner="signed-writer", ttl_ms=10_000)
        assert lease is not None
        _append(
            store,
            lease,
            expected_seq=1,
            commit_id="latest-signed-tail",
            records=(
                record("turn_started", sid=SESSION),
                record("usage_observed", sid=SESSION, meter_id="m", observation=1),
            ),
        )
        final_seq, chain_digest = store.verify_ledger(SESSION)
        trusted_head = SessionHead(store.tenant_id, SESSION, final_seq, chain_digest, signer.key_id)
        store._rows(
            "DELETE FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s AND seq>1",
            (store.tenant_id, SESSION),
        )
        store._rows(
            "DELETE FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq>1",
            (store.tenant_id, SESSION),
        )
        store._rows(
            "DELETE FROM sk_commit WHERE tenant_id=%s AND session_id=%s AND commit_id='latest-signed-tail'",
            (store.tenant_id, SESSION),
        )
        store._rows(
            "UPDATE sk_session SET head_seq=1 WHERE tenant_id=%s AND session_id=%s",
            (store.tenant_id, SESSION),
        )

    with open_store(database) as recovered:
        recovered.configure_input_admission(signer)
        persisted = recovered._one(
            "SELECT chain_digest FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=1",
            (recovered.tenant_id, SESSION),
        )[0]
        assert recovered._ledger_integrity_mode() == "signed"
        assert recovered.verify_ledger(SESSION) == (1, persisted)
        with pytest.raises(ContentBindingMismatch, match=r"expected_head.*seq 1"):
            recovered.verify_ledger(SESSION, expected_head=trusted_head)
        with pytest.raises(ContentBindingMismatch, match=r"expected_head.*seq 1"):
            recovered.read(SESSION, expected_head=trusted_head)

        assert recovered._head(SESSION)[0] == 1
        assert _counts(recovered)[:2] == (1, 1)


@pytest.mark.persistent_store
def test_hash_only_history_remains_weaker_and_cannot_be_retroactively_signed(database: Database) -> None:
    signer = Signer()
    with open_store(database) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        _create(store)
        assert store._ledger_integrity_mode() == "hash-only"
        before = _counts(store)
        chain_digest = store.verify_ledger(SESSION)[1]
        assert (
            chain_digest
            == store._one(
                "SELECT chain_digest FROM sk_record WHERE tenant_id=%s AND session_id=%s AND seq=1",
                (store.tenant_id, SESSION),
            )[0]
        )

        store.configure_input_admission(signer)
        with pytest.raises(LedgerIntegrityError, match="checkpoint is missing at seq 1"):
            store.acquire(SESSION, owner="must-not-upgrade", ttl_ms=10_000)
        assert store._head(SESSION)[0] == 1
        assert _counts(store) == before
        assert signer.sign_count == 0


@pytest.mark.persistent_store
@pytest.mark.parametrize("rotation", ["preserve-binding", "rollback"])
def test_signer_rotation_between_bind_and_sign_preserves_identity_or_rolls_back(
    database: Database,
    rotation: str,
) -> None:
    old = Signer("rotation-old")
    new = Signer("rotation-new")
    with open_store(database) as store:
        store.install_schema()
        store.configure_input_admission(old)
        _create(store)
        lease = store.acquire(SESSION, owner="signed-writer", ttl_ms=10_000)
        assert lease is not None
        before = _counts(store)

        if rotation == "preserve-binding":
            rotating = PreservingRotationSigner(old, new)
            store.configure_input_admission(rotating)
            _append(
                store,
                lease,
                expected_seq=1,
                commit_id="rotated-bind",
                records=(record("turn_started", sid=SESSION),),
            )
            checkpoint = store._one(
                "SELECT key_id,signature FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s AND seq=2",
                (store.tenant_id, SESSION),
            )
            assert checkpoint[0] == old.key_id
            assert store.verify_ledger(SESSION)[0] == 2

            _append(
                store,
                lease,
                expected_seq=2,
                commit_id="rotated-sign",
                records=(record("usage_observed", sid=SESSION, meter_id="m", observation=1),),
            )
            assert (
                store._one(
                    "SELECT key_id FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s AND seq=3",
                    (store.tenant_id, SESSION),
                )[0]
                == new.key_id
            )
        else:
            store.configure_input_admission(DriftingRotationSigner(old, new))
            with (
                pytest.raises(LedgerIntegrityError, match="key binding does not verify checkpoint at seq 2"),
                store.atomic() as tx,
            ):
                tx.append(
                    SESSION,
                    lease=lease,
                    expected_seq=1,
                    commit_id="drifted-bind",
                    records=(record("turn_started", sid=SESSION),),
                )
            assert store._head(SESSION)[0] == 1
            assert _counts(store) == before
