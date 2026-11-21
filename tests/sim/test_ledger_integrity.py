from __future__ import annotations

import json
from hashlib import sha256

import pytest

from scripts.s1_fixture_values import content_ref, create_fixture_session
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._records import make_record
from vv_agent.kernel._sql import LedgerIntegrityError, _extend_chain_digest

from .scenarios import run_scenario
from .world import InvariantViolation, SimLedgerSigner, SimWorld


def test_ledger_signer_caches_exact_proofs_and_rejections(monkeypatch):
    signer = SimLedgerSigner(seed=31)
    public_key = signer._public_key

    class CountingPublicKey:
        def __init__(self):
            self.calls = 0

        def verify(self, signature, message):
            self.calls += 1
            return public_key.verify(signature, message)

    counting_public_key = CountingPublicKey()
    monkeypatch.setattr(signer, "_public_key", counting_public_key)

    message = b"tenant/default/session/s/checkpoint/7"
    signature = signer.bind_key().sign(message)
    assert signer.verify(signer.key_id, message, signature) is True
    assert signer.verify(signer.key_id, message, signature) is True
    assert counting_public_key.calls == 1

    assert signer.verify(signer.key_id, message + b"/different", signature) is False
    assert counting_public_key.calls == 2

    invalid_signature = bytes([signature[0] ^ 1]) + signature[1:]
    assert signer.verify(signer.key_id, message, invalid_signature) is False
    assert signer.verify(signer.key_id, message, invalid_signature) is False
    assert counting_public_key.calls == 3

    other_seed = SimLedgerSigner(seed=32)
    assert signer.verify(other_seed.key_id, message, signature) is False
    assert counting_public_key.calls == 3


def test_i9_recomputes_real_sqlite_chain_and_checkpoints(tmp_path):
    with SimWorld(tmp_path / "ledger.sqlite", seed=17) as world:
        run_scenario(world, "tools")
        world.check(quiescent=True)
        assert world.signed is True
        assert world.summary()["ledger_proof"] == "ed25519"
        assert "I9" in world.summary()["covered"]
        assert "I9" not in world.summary()["pending"]

        store = world.connection("fault")
        checkpoints = store._rows(
            "SELECT key_id,signature FROM sk_checkpoint WHERE tenant_id='default' AND session_id='s' ORDER BY seq"
        )
        admissions = store._rows(
            "SELECT admission_key_id,admission_signature FROM sk_inbox WHERE tenant_id='default' AND session_id='s'"
        )
        assert checkpoints and all(
            key_id == world.ledger_signer.key_id and len(signature) == 64 for key_id, signature in checkpoints
        )
        assert admissions and all(
            key_id == world.ledger_signer.key_id and len(signature) == 64 for key_id, signature in admissions
        )

        rows = store._rows(
            "SELECT seq,body,chain_digest FROM sk_record WHERE tenant_id='default' AND session_id='s' ORDER BY seq"
        )
        previous = rows[0][2]
        with store.atomic():
            for seq, body, _stored_chain in rows[1:]:
                value = json.loads(body)
                if seq == 2:
                    value["payload"]["handler_version"] = "forged"
                    body = canonical_json_bytes(value)
                    digest = sha256(body).hexdigest()
                    store._rows("UPDATE sk_record SET body=?,digest=? WHERE seq=?", (body, digest, seq))
                else:
                    digest = sha256(body).hexdigest()
                previous = _extend_chain_digest(previous, seq, digest)
                store._rows("UPDATE sk_record SET chain_digest=? WHERE seq=?", (previous, seq))
                if seq == 2:
                    store._rows("UPDATE sk_checkpoint SET chain_digest=? WHERE seq=?", (previous, seq))

        with pytest.raises(InvariantViolation, match=r"I9.*seed=17.*seq 2") as error:
            world.check()
        assert "checkpoint signature is invalid at seq 2" in str(error.value)


def _warm_verified_cache(world, session_id="s"):
    observer = world.connection("observer")
    head = observer._head(session_id)[0]
    observer._verified_records(session_id, head)
    return observer, world.connection("fault"), head


def test_i9_cache_rechecks_commit_digest(sim_database):
    with SimWorld(sim_database, seed=17) as world:
        run_scenario(world, "tools")
        observer, fault, head = _warm_verified_cache(world)
        fault._rows(
            "UPDATE sk_record SET commit_digest=%s WHERE tenant_id=%s AND session_id=%s AND seq=1",
            ("0" * 64, observer.tenant_id, "s"),
        )
        with pytest.raises(LedgerIntegrityError, match="commit record metadata"):
            observer._verified_records("s", head)


def test_i9_cache_rechecks_creation_consumers(sim_database):
    with SimWorld(sim_database, seed=17) as world:
        create_fixture_session(world.connection("host"), "consumer", consumers=("events",))
        observer, fault, head = _warm_verified_cache(world, "consumer")
        fault._rows(
            "DELETE FROM sk_consumer WHERE tenant_id=%s AND session_id=%s AND consumer=%s",
            (observer.tenant_id, "consumer", "events"),
        )
        with pytest.raises(LedgerIntegrityError, match="commit creation consumer is missing"):
            observer._verified_records("consumer", head)


def test_i9_cache_rechecks_records_beyond_head(sim_database):
    with SimWorld(sim_database, seed=17) as world:
        run_scenario(world, "tools")
        observer, fault, head = _warm_verified_cache(world)
        record = make_record(
            "usage_observed",
            session_id="s",
            payload={
                "meter_id": "beyond-head",
                "observation": 1,
                "mode": "cumulative",
                "usage": content_ref("beyond-head/usage"),
                "source": content_ref("beyond-head/source"),
            },
        )
        body = record.encode()
        fault._rows(
            "INSERT INTO sk_record(tenant_id,session_id,seq,record_id,schema_version,kind,turn_id,operation_id,"
            "attempt,body,digest,chain_digest,commit_id,commit_digest,commit_position,writer_epoch,created_ms) "
            "VALUES (%s,%s,%s,%s,'S1',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                observer.tenant_id,
                "s",
                head + 1,
                record.record_id,
                record.kind,
                record.turn_id,
                record.operation_id,
                record.attempt,
                body,
                sha256(body).hexdigest(),
                "0" * 64,
                "beyond-head",
                "0" * 64,
                0,
                1,
                1,
            ),
        )
        with pytest.raises(LedgerIntegrityError, match="ledger record outlives"):
            observer._verified_records("s", head)


def test_i9_cache_rechecks_integrity_mode(sim_database):
    with SimWorld(sim_database, seed=17) as world:
        run_scenario(world, "tools")
        observer, _fault, head = _warm_verified_cache(world)
        observer.configure_input_admission(allow_unsigned=True)
        with pytest.raises(LedgerIntegrityError, match="ledger verifier is required"):
            observer._verified_records("s", head)


def test_i9_explicit_hash_only_negative_rejects_rehashed_record_without_updating_chain(tmp_path):
    with SimWorld(tmp_path / "hash-only-ledger.sqlite", seed=23, signed=False) as world:
        run_scenario(world, "tools")
        world.check(quiescent=True)

        store = world.connection("fault")
        body = store._one("SELECT body FROM sk_record WHERE tenant_id='default' AND session_id='s' AND seq=2")[0]
        value = json.loads(body)
        value["payload"]["handler_version"] = "forged"
        forged = canonical_json_bytes(value)
        store._rows(
            "UPDATE sk_record SET body=?,digest=? WHERE tenant_id='default' AND session_id='s' AND seq=2",
            (forged, sha256(forged).hexdigest()),
        )

        with pytest.raises(InvariantViolation, match=r"I9.*seed=23.*chain digest mismatch at seq 2"):
            world.check()
        assert world.signed is False
        assert world.summary()["ledger_proof"] == "hash-only"
        assert "I9" in world.summary()["covered"]


def test_i9_real_check_verifies_commit_metadata(tmp_path):
    with SimWorld(tmp_path / "commit-ledger.sqlite", seed=29) as world:
        run_scenario(world, "accepted")
        world.check(quiescent=True)
        store = world.connection("fault")
        store._rows(
            "UPDATE sk_commit SET digest=? WHERE tenant_id='default' AND session_id='s' AND commit_id='session/create'",
            ("0" * 64,),
        )

        with pytest.raises(InvariantViolation, match=r"I9.*seed=29.*commit body digest mismatch"):
            world.check()
