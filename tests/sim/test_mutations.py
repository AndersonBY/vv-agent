"""Mutate executed production functions in memory; source files stay untouched."""

import __future__

import inspect
import textwrap
import threading

import pytest
from session.helpers import content_ref

from vv_agent.kernel import _driver_outcomes, _driver_recovery
from vv_agent.kernel._lease_scope import LeaseOwnership
from vv_agent.kernel._records import make_record
from vv_agent.kernel._sql import SQLSessionTx

from .scenarios import run_scenario
from .world import InvariantViolation, SimWorld, require

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
    column_type = "TEXT" if backend == "sqlite" else "text"
    integer_type = "INTEGER" if backend == "sqlite" else "bigint"
    blob_type = "BLOB" if backend == "sqlite" else "bytea"
    record_xor_input = (
        "((record_seq IS NOT NULL) + (input_id IS NOT NULL) = 1)"
        if backend == "sqlite"
        else "((record_seq IS NOT NULL)::int + (input_id IS NOT NULL)::int = 1)"
    )
    strict = " STRICT" if backend == "sqlite" else ""
    store = world.connection("mutation/schema")
    with store.atomic():
        store._rows("DROP INDEX IF EXISTS sk_inbox_ready")
        # Foreign-key targets must stay unique, so rebuild only the link table
        # without its input_id reference to the intentionally non-unique inbox.
        store._rows("DROP TABLE sk_content_link")
        store._rows("DROP TABLE sk_inbox")
        store._rows(
            f"""
            CREATE TABLE sk_inbox (
                tenant_id {column_type} NOT NULL, session_id {column_type} NOT NULL, input_id {column_type} NOT NULL,
                input_seq {integer_type} NOT NULL CHECK (input_seq > 0),
                kind {column_type} NOT NULL CHECK (kind IN ('user','steer','follow_up','provider_result','approval_answer',
                    'child_result','control','provider_evidence','model_selection')),
                body {blob_type} NOT NULL, digest {column_type} NOT NULL CHECK (length(digest)=64),
                admission_key_id {column_type}, admission_signature {blob_type},
                admission_signed_ms {integer_type} CHECK (admission_signed_ms IS NULL OR admission_signed_ms >= 0),
                available_ms {integer_type} NOT NULL CHECK (available_ms >= 0),
                received_ms {integer_type} NOT NULL CHECK (received_ms >= 0),
                consumed_seq {integer_type} CHECK (consumed_seq IS NULL OR consumed_seq > 0),
                UNIQUE(tenant_id,session_id,input_seq),
                CHECK ((admission_key_id IS NULL) = (admission_signature IS NULL)
                    AND (admission_key_id IS NULL) = (admission_signed_ms IS NULL)),
                FOREIGN KEY(tenant_id,session_id) REFERENCES sk_session(tenant_id,session_id),
                FOREIGN KEY(tenant_id,session_id,consumed_seq) REFERENCES sk_record(tenant_id,session_id,seq)
            ){strict}
            """
        )
        store._rows(
            "CREATE INDEX sk_inbox_ready ON sk_inbox(tenant_id,session_id,available_ms,input_seq) WHERE consumed_seq IS NULL"
        )
        store._rows(
            f"""
            CREATE TABLE sk_content_link (
                tenant_id {column_type} NOT NULL, session_id {column_type} NOT NULL,
                content_id {column_type} NOT NULL, record_seq {integer_type}, input_id {column_type},
                slot {column_type} NOT NULL,
                CHECK {record_xor_input},
                FOREIGN KEY(tenant_id,session_id) REFERENCES sk_session(tenant_id,session_id),
                FOREIGN KEY(tenant_id,session_id,record_seq) REFERENCES sk_record(tenant_id,session_id,seq),
                FOREIGN KEY(tenant_id,session_id,content_id) REFERENCES sk_content(tenant_id,session_id,content_id)
            ){strict}
            """
        )
        store._rows(
            "CREATE UNIQUE INDEX sk_content_link_record_slot "
            "ON sk_content_link(tenant_id,session_id,record_seq,slot) WHERE record_seq IS NOT NULL"
        )
        store._rows(
            "CREATE UNIQUE INDEX sk_content_link_input_slot "
            "ON sk_content_link(tenant_id,session_id,input_id,slot) WHERE input_id IS NOT NULL"
        )
        store._rows("CREATE INDEX sk_content_link_content ON sk_content_link(tenant_id,session_id,content_id)")


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
                'payload=a.execution_plan._payload | {"not_before_ms": due},',
                'payload=a.execution_plan._payload | {"not_before_ms": due, "idempotency_key": f"changed/{a.plan.attempt}"},',
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


def lock_order_probe(world):
    """Check that production poll leaves commit_lock free at its renewal call.

    The substituted renewal callback proves the commit latch is available, then
    commits one real SQL record, then invokes the actual store renewal. This is
    a bounded single-threaded lock-order probe; it does not create concurrent
    database lock waits. A mutant holding the latch across poll fails before the
    append while a nonblocking assertion prevents the test from hanging.
    """
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
        lock=threading.Lock(),
        poll_lock=threading.Lock(),
        commit_lock=threading.Lock(),
        latch_entry=threading.local(),
        actor_identity=threading.get_ident,
    )
    original = renewal.renew
    committed = []

    def append_then_renew(current, *, ttl_ms):
        require(
            heartbeat.commit_lock.acquire(blocking=False),
            "L1",
            "LeaseOwnership.poll held commit_lock across its renewal boundary",
        )
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
        finally:
            heartbeat.commit_lock.release()
        return original(current, ttl_ms=ttl_ms)

    renewal.renew = append_then_renew
    heartbeat.poll(renewal)
    assert committed[0].head_seq == 2
    assert world.connection("observer").read("s").records[-1].record.payload["meter_id"] == "lock-probe"


@pytest.mark.parametrize("mutated", [False, True])
def test_l1_commit_lock_free_at_renewal_boundary(sim_database, monkeypatch, record_property, mutated):
    if mutated:
        original = LeaseOwnership.poll

        def hold_commit_lock(self, store):
            with self.commit_lock:
                return original(self, store)

        hold_commit_lock.__dict__["_sim_mutation"] = "hold_commit_lock_during_renewal"
        monkeypatch.setattr(LeaseOwnership, "poll", hold_commit_lock)
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
            'payload=a.execution_plan._payload | {"not_before_ms": due},',
            'payload=a.execution_plan._payload | {"not_before_ms": due, "idempotency_key": f"changed/{a.plan.attempt}"},',
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
