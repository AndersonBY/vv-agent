"""Mutate executed production functions in memory; source files stay untouched."""

import __future__

import inspect
import json
import textwrap
import threading

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from scripts.s1_fixture_values import content_ref, create_fixture_session, push_fixture_input
from vv_agent.kernel import _driver_outcomes, _driver_recovery
from vv_agent.kernel._content import ContentIntegrityError, ContentRef, WrappedDataKey
from vv_agent.kernel._content_runtime import durable_content_store
from vv_agent.kernel._content_sql import _content_aad
from vv_agent.kernel._lease_scope import LeaseOwnership
from vv_agent.kernel._records import InboxItem, make_record
from vv_agent.kernel._sql import SQLSessionTx, SQLStore
from vv_agent.kernel._store import WorkerSupports

from .content import SimAESGCMCipher
from .scenarios import run_scenario
from .world import Choices, InvariantViolation, SimWorld, require, sqlite_root

INPUT_DEDUPE_MUTATION = "push+dropped-input-identity-constraint"


def mutate(function, before, after):
    source = textwrap.dedent(inspect.getsource(function))
    assert source.count(before) == 1, "mutation anchor changed; update against the actual implementation"
    namespace = dict(function.__globals__)
    exec(
        compile(
            source.replace(before, after),
            f"<mutation:{function.__qualname__}>",
            "exec",
            flags=function.__code__.co_flags & __future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        namespace,
    )
    return namespace[function.__name__]


def drop_input_identity_constraint(world):
    """Allow duplicate input identities only in this disposable test database."""
    backend = world.summary()["backend"]
    store = world.connection("mutation/schema")
    if backend == "postgres":
        input_link = store._one(
            "SELECT conname FROM pg_constraint "
            "WHERE conrelid='sk_content_link'::regclass AND contype='f' AND confrelid='sk_inbox'::regclass "
            "AND pg_get_constraintdef(oid) LIKE %s",
            ("%(tenant_id, session_id, input_id)%",),
        )[0]
        constraints = [
            row[0]
            for row in store._rows(
                "SELECT conname FROM pg_constraint WHERE conrelid='sk_inbox'::regclass AND ("
                "(contype='p' AND pg_get_constraintdef(oid) LIKE %s) OR "
                "(contype='u' AND pg_get_constraintdef(oid) LIKE %s))",
                (
                    "PRIMARY KEY (tenant_id, session_id, input_id)%",
                    "UNIQUE (tenant_id, session_id, input_seq)%",
                ),
            )
        ]
        assert len(constraints) == 2, "input identity constraints changed"
        store._rows(f'ALTER TABLE sk_content_link DROP CONSTRAINT "{input_link}"')
        for constraint in constraints:
            store._rows(f'ALTER TABLE sk_inbox DROP CONSTRAINT "{constraint}"')
        return

    with store.atomic():
        inbox_ddl = store._one("SELECT sql FROM sqlite_master WHERE type='table' AND name='sk_inbox'")[0]
        link_ddl = store._one("SELECT sql FROM sqlite_master WHERE type='table' AND name='sk_content_link'")[0]
        indexes = store._rows(
            "SELECT name,sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL "
            "AND tbl_name IN ('sk_inbox','sk_content_link') ORDER BY name"
        )
        constraints = (
            "PRIMARY KEY (tenant_id,session_id,input_id),",
            "UNIQUE (tenant_id,session_id,input_seq),",
        )
        assert all(inbox_ddl.count(constraint) == 1 for constraint in constraints), "input identity constraint anchor changed"
        store._rows("DROP INDEX IF EXISTS sk_inbox_ready")
        store._rows("DROP TABLE sk_content_link")
        store._rows("DROP TABLE sk_inbox")
        for constraint in constraints:
            inbox_ddl = inbox_ddl.replace(constraint, "", 1)
        store._rows(inbox_ddl)
        input_link = "FOREIGN KEY (tenant_id,session_id,input_id) REFERENCES sk_inbox(tenant_id,session_id,input_id),"
        assert link_ddl.count(input_link) == 1, "content-link input foreign key anchor changed"
        store._rows(link_ddl.replace(input_link, "", 1))
        for _name, ddl in indexes:
            store._rows(ddl)


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("append_fence", "I2"),
        ("input_dedupe", "I3"),
        ("retry_key", "I1"),
        ("unsafe_unknown", "I10"),
    ],
)
def test_actual_implementation_mutations(sim_database, monkeypatch, record_property, mutation, expected):
    if mutation == "append_fence":
        before = "s._check_lease(session_id, lease, row, "
        source = textwrap.dedent(inspect.getsource(SQLSessionTx.append))
        assert source.count(before) == 3
        # Remove only append's real admission/revalidation fences, retaining CAS,
        # reducer validation, SQL constraints, acquisition and renewal rules.
        source = source.replace("s._check_lease(session_id, lease, row, s._now())", "None")
        source = source.replace("s._check_lease(session_id, lease, row, now)", "None")
        namespace = dict(SQLSessionTx.append.__globals__)
        exec(
            compile(
                source,
                "<mutation:SQLSessionTx.append>",
                "exec",
                flags=SQLSessionTx.append.__code__.co_flags & __future__.annotations.compiler_flag,
                dont_inherit=True,
            ),
            namespace,
        )
        monkeypatch.setattr(SQLSessionTx, "append", namespace["append"])
    elif mutation == "input_dedupe":
        input_mutation = mutate(SQLSessionTx.push, "if old:", "if False and old:")
        input_mutation.__dict__["_sim_mutation"] = INPUT_DEDUPE_MUTATION
        monkeypatch.setattr(SQLSessionTx, "push", input_mutation)
    elif mutation == "retry_key":
        monkeypatch.setattr(
            _driver_recovery,
            "step",
            mutate(
                _driver_recovery.step,
                'payload=payload | {"not_before_ms": due},',
                'payload=payload | {"not_before_ms": due, "idempotency_key": f"changed/{a.plan.attempt}"},',
            ),
        )
    else:
        monkeypatch.setattr(
            _driver_outcomes,
            "unknown_steps",
            mutate(
                _driver_outcomes.unknown_steps,
                'and (is_model or (plan._payload["tool"] or {}).get("idempotency") == "supported")',
                "and True",
            ),
        )
    from .report import versions

    identity = versions()
    assert len(identity["in_memory_mutations"]) == 1
    if mutation == "input_dedupe":
        assert identity["in_memory_mutations"][0]["mutation"] == INPUT_DEDUPE_MUTATION
    record_property("mutation_identity", identity)
    scenario = {
        "retry_key": "unknown_idempotent",
        "unsafe_unknown": "unknown_unsafe",
    }.get(mutation, "tools")
    expected_message = f"{expected}: duplicate input got a new receipt" if mutation == "input_dedupe" else expected
    with SimWorld(sim_database, seed=17) as world:
        if mutation == "input_dedupe":
            drop_input_identity_constraint(world)
        with pytest.raises(InvariantViolation, match=expected_message):
            run_scenario(world, scenario)


def test_worker_filter_mutation_is_killed(sim_database, monkeypatch):
    monkeypatch.setattr(WorkerSupports, "accepts", lambda self, **kwargs: True)
    monkeypatch.setattr(SQLStore, "worker_level", property(lambda self: 99))
    monkeypatch.setattr(SQLStore, "_worker_handler_filter_sql", "(%s IS NOT NULL)")
    monkeypatch.setattr(SQLStore, "_worker_capability_filter_sql", "(%s IS NOT NULL)")
    with SimWorld(sim_database, seed=17) as world, pytest.raises(InvariantViolation, match="I11"):
        run_scenario(world, "worker_compatibility")


def test_i2_oracle_rejects_zero_writer_epoch(sim_database, monkeypatch):
    original = SQLSessionTx._write

    def zero_writer_epoch(tx, record, **kwargs):
        kwargs["epoch"] = 0
        return original(tx, record, **kwargs)

    monkeypatch.setattr(SQLSessionTx, "_write", zero_writer_epoch)
    with SimWorld(sim_database, seed=17) as world, pytest.raises(InvariantViolation, match="I2"):
        run_scenario(world, "tools")


def lock_order_probe(world):
    """Check that production poll does not hold a real SQL write lock across renewal."""
    world.create()
    driving = world.connection("driver")
    renewal = world.connection("renewal")
    lease = driving.acquire("s", owner="lock-probe", ttl_ms=15000)
    heartbeat = LeaseOwnership(
        lease,
        15000,
        0.25,
        clock=world.clock,
        stop=threading.Event(),
        ready=threading.Event(),
    )
    if world.summary()["backend"] == "sqlite":
        sqlite_root(driving).connection.execute("PRAGMA busy_timeout=10")
    original = renewal.renew
    committed = []

    def append_then_renew(current, *, ttl_ms):
        try:
            rec = make_record(
                "usage_observed",
                session_id="s",
                payload={
                    "meter_id": "lock-probe",
                    "observation": 1,
                    "mode": "cumulative",
                    "usage": content_ref("lock-probe/usage"),
                    "source": content_ref("lock-probe/source"),
                },
            )
            with driving.atomic() as tx:
                receipt = tx.append("s", lease=current, expected_seq=1, commit_id="lock-probe", records=(rec,))
            committed.append(receipt)
        except Exception as exc:
            detail = str(exc).lower()
            if "locked" not in detail and "lock timeout" not in detail:
                raise
            require(False, "L1", "LeaseOwnership.poll held a SQL write lock across renewal")
        return original(current, ttl_ms=ttl_ms)

    renewal.renew = append_then_renew
    heartbeat.poll(renewal)
    assert committed[0].head_seq == 2
    assert world.connection("observer").read("s").records[-1].record.payload["meter_id"] == "lock-probe"


@pytest.mark.parametrize("mutated", [False, True])
def test_l1_write_lock_free_at_renewal_boundary(sim_database, monkeypatch, record_property, mutated):
    if mutated:
        original = LeaseOwnership.poll

        def hold_write_lock(self, store):
            with store.atomic():
                store._rows(
                    "UPDATE sk_session SET lease_until_ms=lease_until_ms WHERE tenant_id=%s AND session_id=%s",
                    (store.tenant_id, self.lease.session_id),
                )
                return original(self, store)

        hold_write_lock.__dict__["_sim_mutation"] = "hold_write_lock_during_renewal"
        monkeypatch.setattr(LeaseOwnership, "poll", hold_write_lock)
    from .report import versions

    identity = versions()
    assert bool(identity["in_memory_mutations"]) == mutated
    record_property("mutation_identity", identity)
    with SimWorld(sim_database) as world:
        if mutated:
            with pytest.raises(InvariantViolation, match="L1"):
                lock_order_probe(world)
        else:
            lock_order_probe(world)


def test_sqlite_failure_replay_minimization(tmp_path, monkeypatch, record_property):
    from .report import minimize, replay_provenance, versions

    baseline = versions()
    monkeypatch.setattr(
        _driver_recovery,
        "step",
        mutate(
            _driver_recovery.step,
            'payload=payload | {"not_before_ms": due},',
            'payload=payload | {"not_before_ms": due, "idempotency_key": f"changed/{a.plan.attempt}"},',
        ),
    )
    with SimWorld(tmp_path / "original.sqlite", seed=17) as failed:
        with pytest.raises(InvariantViolation, match="I1"):
            run_scenario(failed, "unknown_idempotent")
        report = failed.summary() | versions()
    assert len(report["code_digest"]) == 64
    assert all(
        report[key] == baseline[key]
        for key in ("contract_selection", "contract_version", "contract_revision", "contract_manifest_sha256", "pinned_contract")
    )
    assert len(report["in_memory_mutations"]) == 1
    assert report["in_memory_mutations"][0]["target"].endswith("_driver_recovery.step")
    provenance = replay_provenance(report, baseline)
    assert not provenance["matched"] and "in_memory_mutations" in provenance["differences"]
    record_property("failure_report", report)
    if report["code_revision"] is None:
        assert report["dirty"] is None
    trials = 0

    def reproduces(decisions):
        nonlocal trials
        trials += 1
        try:
            with SimWorld(tmp_path / f"trial-{trials}.sqlite", seed=999, replay=decisions) as world:
                run_scenario(world, report["scenario"])
        except InvariantViolation as exc:
            return exc.invariant == "I1"
        except Exception:
            return False
        return False

    reduced, attempts = minimize(report["decisions"], reproduces, budget=2)
    assert attempts == 2 and len(reduced) < len(report["decisions"])
    assert reproduces(reduced)


def test_sqlite_real_content_cipher_tamper_and_aad_failures(tmp_path):
    marker = b"sim-content-plaintext-marker-7e31c9d2"
    session_id = "s"
    with SimWorld(tmp_path / "content.sqlite", seed=17) as world:
        store = world.connection("host")
        create_fixture_session(store, session_id, consumers=())
        push_fixture_input(store, session_id, InboxItem("secret", "user", {"content": marker.decode("ascii")}))

        row = store._one(
            "SELECT c.content_id,c.key_id,c.mac,c.size,c.labels,c.state,c.nonce,c.ciphertext,c.object_locator "
            "FROM sk_content c JOIN sk_content_link l ON l.tenant_id=c.tenant_id AND l.session_id=c.session_id "
            "AND l.content_id=c.content_id WHERE c.tenant_id=%s AND c.session_id=%s "
            "AND l.input_id=%s AND l.slot=%s LIMIT 1",
            (store.tenant_id, session_id, "secret", "input.content"),
        )
        assert row is not None
        ref = ContentRef.from_dict(
            {
                "content_id": row[0],
                "mac": row[2],
                "size": row[3],
                "key_id": row[1],
                "labels": json.loads(row[4]),
            }
        )
        nonce = bytes(row[6])
        ciphertext = bytes(row[7])
        stored_blobs = store._rows(
            "SELECT nonce,ciphertext FROM sk_content WHERE tenant_id=%s AND session_id=%s",
            (store.tenant_id, session_id),
        )
        assert all(
            marker not in bytes(stored_nonce) and marker not in bytes(stored_ciphertext)
            for stored_nonce, stored_ciphertext in stored_blobs
        )

        key_id, _provider_id, wrapped_key, _state = store._one(
            "SELECT key_id,provider_id,wrapped_key,state FROM sk_session_key WHERE tenant_id=%s AND session_id=%s",
            (store.tenant_id, session_id),
        )
        data_key = world.content_provider.unwrap(
            store.tenant_id,
            session_id,
            WrappedDataKey(store.tenant_id, session_id, key_id, bytes(wrapped_key)),
        )
        aad = _content_aad(
            store.tenant_id,
            session_id,
            ref.content_id,
            ref.key_id,
            ref.mac,
            ref.size,
            ref.labels,
        )
        content_store = durable_content_store(store)
        plaintext = content_store.get(store.tenant_id, session_id, ref)
        assert marker in plaintext

        tampered = bytes([ciphertext[0] ^ 1]) + ciphertext[1:]
        with pytest.raises(ContentIntegrityError, match="content failed AEAD authentication") as caught:
            world.content_cipher.decrypt(data_key, nonce, tampered, aad)
        assert isinstance(caught.value.__cause__, InvalidTag)

        with store.atomic():
            store._rows(
                "UPDATE sk_content SET ciphertext=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (tampered, store.tenant_id, session_id, ref.content_id),
            )
        with pytest.raises(ContentIntegrityError, match="content failed AEAD authentication"):
            content_store.get(store.tenant_id, session_id, ref)
        wrong_aad_ciphertext = AESGCM(data_key).encrypt(nonce, plaintext, aad + b"/wrong-tenant")
        with store.atomic():
            store._rows(
                "UPDATE sk_content SET ciphertext=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (wrong_aad_ciphertext, store.tenant_id, session_id, ref.content_id),
            )
        with pytest.raises(ContentIntegrityError, match="content failed AEAD authentication"):
            content_store.get(store.tenant_id, session_id, ref)


def test_sim_content_randomness_is_independent_of_scheduler_choices():
    seed = 17
    unperturbed = SimAESGCMCipher(seed)
    perturbed = SimAESGCMCipher(seed)
    choices = Choices(seed)
    for _ in range(32):
        choices.choose("independent/content-stream", ("renew", "drive", "project"))
    assert perturbed.random_bytes(96) == unperturbed.random_bytes(96)
