"""Mutate executed production functions in memory; source files stay untouched."""

import __future__

import inspect
import textwrap

import pytest

from vv_agent.session import _driver_recovery, kernel
from vv_agent.session.records import make_record
from vv_agent.session.sql import SQLSessionTx

from .scenarios import run_scenario
from .world import InvariantViolation, SimWorld, require


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
        monkeypatch.setattr(SQLSessionTx, "push", mutate(SQLSessionTx.push, "if old:", "if False and old:"))
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
            _driver_recovery,
            "unknown",
            mutate(
                _driver_recovery.unknown,
                'and (is_model or (plan._payload["tool"] or {}).get("idempotency") == "supported")',
                "and True",
            ),
        )
    from .report import versions

    identity = versions()
    assert len(identity["in_memory_mutations"]) == 1
    record_property("mutation_identity", identity)
    scenario = "unknown_idempotent" if mutation == "retry_key" else "unknown_unsafe" if mutation == "unsafe_unknown" else "tools"
    with SimWorld(sim_database, seed=17) as world, pytest.raises(InvariantViolation, match=expected):
        run_scenario(world, scenario)


def lock_order_probe(world):
    """A real SQL append progresses while production poll is about to renew.

    At this scheduling point renewal may be waiting for a database row lock.
    Holding commit_lock there would prevent the driver from doing its append.
    The nonblocking acquisition turns that deadlock into a bounded failure.
    """
    world.create()
    driving = world.connection("driver")
    renewal = world.connection("renewal")
    lease = driving.acquire("s", owner="lock-probe", ttl_ms=15000)
    heartbeat = kernel._LeaseHeartbeat(lease, lambda: world.heartbeat_connection("lock-probe"), 15000, 0.25, clock=world.clock)
    original = renewal.renew
    committed = []

    def waiting_on_database(current, *, ttl_ms):
        require(heartbeat.commit_lock.acquire(blocking=False), "L1", "SQL renewal holds commit_lock; real append cannot progress")
        try:
            rec = make_record(
                "usage_observed",
                session_id="s",
                payload={
                    "meter_id": "lock-probe",
                    "observation": 1,
                    "mode": "cumulative",
                    "usage": {},
                    "source": "host",
                },
            )
            with driving.atomic() as tx:
                receipt = tx.append("s", lease=current, expected_seq=1, commit_id="lock-probe", records=(rec,))
            committed.append(receipt)
        finally:
            heartbeat.commit_lock.release()
        return original(current, ttl_ms=ttl_ms)

    renewal.renew = waiting_on_database
    heartbeat.poll(renewal)
    assert committed[0].head_seq == 2
    assert world.connection("observer").read("s").records[-1].record.payload["meter_id"] == "lock-probe"


@pytest.mark.parametrize("mutated", [False, True])
def test_l1_real_append_during_renewal(sim_database, monkeypatch, record_property, mutated):
    if mutated:
        original = kernel._LeaseHeartbeat.poll

        def hold_commit_lock(self, store):
            with self.commit_lock:
                return original(self, store)

        hold_commit_lock.__dict__["_sim_mutation"] = "hold_commit_lock_during_renewal"
        monkeypatch.setattr(kernel._LeaseHeartbeat, "poll", hold_commit_lock)
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
