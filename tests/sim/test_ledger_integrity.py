from __future__ import annotations

import json
from hashlib import sha256

import pytest

from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._sql import _extend_chain_digest

from .scenarios import run_scenario
from .world import InvariantViolation, SimWorld


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
