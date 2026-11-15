"""SDK-free receipt algorithms consume real current SQL/SDK-produced prefixes."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from session.test_core_driver_outcomes import receipt_fixture
from support.sdk_runtime import SDKWorld

from vv_agent.budget import RunBudgetLimits
from vv_agent.kernel._budget import BudgetObservation


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
@pytest.mark.parametrize("budgeted", [False, True])
def test_core_receipt_unknown_and_parking_use_real_storage_without_sdk_imports(tmp_path, flags, budgeted, monkeypatch):
    database = tmp_path / "prefix.sqlite"
    with SDKWorld(database, seed=17) as world:
        worker, plan, outcome = receipt_fixture(world, limits=RunBudgetLimits(max_total_tokens=100) if budgeted else None)
        clock_observations = []
        original_clock = worker.runtime.clock.monotonic_ns
        recording_budget_clock = False

        def observe_clock():
            nonlocal recording_budget_clock
            value = original_clock()
            if recording_budget_clock:
                clock_observations.append(value)
                recording_budget_clock = False
            return value

        monkeypatch.setattr(worker.runtime.clock, "monotonic_ns", observe_clock)
        run_outcome_steps = worker._run_outcome_steps

        def observe_budget_reads(steps):
            nonlocal recording_budget_clock
            response = None
            while True:
                try:
                    request = steps.send(response)
                except StopIteration as done:
                    return done.value
                recording_budget_clock = request is BudgetObservation.CLOCK_NS
                response = yield request

        def run_marked_outcome_steps(steps, **kwargs):
            return run_outcome_steps(observe_budget_reads(steps), **kwargs)

        monkeypatch.setattr(worker, "_run_outcome_steps", run_marked_outcome_steps)
        expected = worker.completed(plan, outcome)
        source = {
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "outcome": {
                "result": outcome.result,
                "usage": outcome.usage,
                "shared_state": outcome.shared_state,
                "evidence": list(outcome.evidence),
            },
            "records": [r.to_dict() for r in expected],
            "clock_ms": world.clock.now_ms(),
            "budget_clocks": clock_observations,
        }
        assert worker.store.release(worker.scope.lease)
    facts = tmp_path / "input.json"
    facts.write_text(json.dumps(source))
    script = r"""
import importlib.abc
import json
import sys
from pathlib import Path
from threading import Event, Lock, local, get_ident
sys.path.insert(0, sys.argv[1])
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {'vv_agent','vv_agent.kernel'} or fullname.startswith('vv_agent.kernel.')
                or fullname.split('.')[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel._budget import BudgetObservation
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._driver_outcomes import completed_steps, unknown_steps, accepted_steps, OutcomeRead
from vv_agent.kernel._driver_planning import ProviderAvailable
from vv_agent.kernel._lease_scope import LeaseOwnership
from vv_agent.kernel._provider_outcomes import Accepted, Definitive, ProviderResultInvalid
from vv_agent.kernel._reducer import Fold
from vv_agent.kernel._sqlite import SQLiteStore
source=json.loads(Path(sys.argv[3]).read_text())
class Clock:
    def now_ms(self): return source['clock_ms']
    def monotonic(self): return self.now_ms()/1000
    def monotonic_ns(self): return self.now_ms()*1000000
    def sleep(self,seconds): raise AssertionError('no scheduler requested')
clock=Clock()
def unused(*args,**kwargs): raise AssertionError('no compaction in this produced prefix')
def factory(): return Fold(normalize_memory_settings=unused,count_messages=unused)
class Scope(LeaseOwnership):
    turn_id=None
    generation=None
with SQLiteStore.standalone(sys.argv[2],clock=clock,fold_factory=factory) as store:
    lease=store.acquire('s',owner='core-reader',ttl_ms=15000)
    assert lease is not None
    scope=Scope(lease,15000,.25,clock=clock,stop=Event(),ready=Event(),lock=Lock(),poll_lock=Lock(),
                commit_lock=Lock(),latch_entry=local(),actor_identity=get_ident)
    owner=CoreDriver(store,'s',scope,fold_factory=factory,clock=clock,observe_cost=lambda:None,
                     hook=lambda *a:None,emit_phase=lambda *a:None)
    owner.bind_turn()
    plan=owner.state.operations[source['operation_id']].attempts[source['attempt']].execution_plan
    outcome=Definitive(source['outcome']['result'],tuple(source['outcome']['evidence']),
                       source['outcome']['usage'],source['outcome']['shared_state'])
    observed=[]
    clock_reads=iter(source["budget_clocks"])
    def drain(steps):
        value=None
        while True:
            try: request=steps.send(value)
            except StopIteration as done: return done.value
            observed.append(type(request).__name__ if isinstance(request,ProviderAvailable) else request.value)
            if isinstance(request,ProviderAvailable):
                assert request.name=='effect'
                value=False
            elif request is OutcomeRead.CANCELLATION_REASON: value=None
            elif request is OutcomeRead.ACTIVE_INTERVAL_NS: value=clock.monotonic_ns()
            elif request is OutcomeRead.BIND_BUDGET_METER: value=None
            elif request is BudgetObservation.CLOCK_NS: value=next(clock_reads)
            elif request in {OutcomeRead.SCOPE_CANCELLED, OutcomeRead.HOST_CANCELLED}: value=False
            elif request is OutcomeRead.POLL_LEASE: value=scope.poll(store)
            elif request is OutcomeRead.POLL_INTERVAL: value=73
            else: raise AssertionError(('unexpected observation',request))
    result=drain(completed_steps(owner,plan,outcome))
    assert [r.to_dict() for r in result]==source['records']
    assert observed[-1]=='ProviderAvailable'
    assert observed.count('clock_ns')==len(source['budget_clocks'])
    assert list(clock_reads)==[]
    missing=drain(unknown_steps(owner,plan,'worker lost before durable result'))
    assert missing.payload['retry']=='retry' and missing.payload['measurement_missing']
    assert missing.payload['observation']['usage'] is None
    handle={'kind':'provider','provider':plan.payload['provider_binding'],'job_id':'job',
            'operation_id':plan.operation_id,'attempt':plan.attempt,'request_digest':plan.payload['request_digest'],
            'evidence':'receipt','query_ref':'job','cancel_ref':None}
    parked=drain(accepted_steps(owner,plan,Accepted(handle,True)))
    assert parked.payload['poll_at_ms']==clock.now_ms()+73 and parked.payload['yield_on_steer'] is True
    old=len(observed)
    try: drain(accepted_steps(owner,plan,Accepted(handle|{'request_digest':'0'*64})))
    except ProviderResultInvalid: pass
    else: raise AssertionError('invalid acceptance was accepted')
    assert len(observed)==old
assert all(n in {'vv_agent','vv_agent.kernel'} or n.startswith('vv_agent.kernel.')
           for n in sys.modules if n.startswith('vv_agent'))
"""
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(Path(__file__).parents[1] / "src"), str(database), str(facts)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _started_model_plan(tmp_path):
    from vv_agent import Agent, RunConfig, ScriptedModelProvider
    from vv_agent.session._driver_scope import _Scope
    from vv_agent.session.content_store import LocalKeyProvider
    from vv_agent.session.kernel import _Driver, drive
    from vv_agent.session.records import InboxItem
    from vv_agent.session.sqlite import SQLiteStore
    from vv_agent.session.surfaces import SessionDriver
    from vv_agent.types import LLMResponse

    class Started(BaseException):
        pass

    store = SQLiteStore(str(tmp_path / "outcomes.sqlite"))
    store.install_schema()
    sessions = SessionDriver(store=store, content_key_provider=LocalKeyProvider(b"k" * 32))
    sessions.create("s", "/workspace", {})
    sessions.push("s", InboxItem("input", "user", {"content": "produce a result"}))
    runtime = sessions.runtime(
        Agent("outcomes", "Work.", model="scripted"),
        RunConfig(
            model_provider=ScriptedModelProvider.new("scripted", "scripted", [LLMResponse("unused")]),
            workspace="/workspace",
        ),
        session_id="s",
    )

    def cut(point, record):
        if point == "after_commit" and record is not None and record.kind == "op_started":
            raise Started

    runtime.hook = cut
    with pytest.raises(Started):
        drive(sessions.store, "s", runtime=runtime)
    runtime.hook = lambda *_: None
    lease = sessions.store.acquire("s", owner="outcomes-test", ttl_ms=15_000)
    assert lease is not None
    driver = _Driver(sessions.store, "s", runtime, _Scope(lease, runtime, now_ms=store._now()))
    driver.bind_turn()
    operation = next(op for op in driver.state.operations.values() if op.kind == "model")
    return store, sessions, driver, operation.attempts[1].execution_plan


def test_rejected_retry_reuses_the_committed_plan_refs_and_digest(tmp_path):
    from vv_agent.kernel._content import ContentRef
    from vv_agent.kernel._hydration import reference_digest
    from vv_agent.kernel._provider_outcomes import Definitive
    from vv_agent.kernel._records import digest

    _store, sessions, driver, plan = _started_model_plan(tmp_path)
    try:
        outcome = Definitive(
            {
                "content": "retry",
                "error_code": "model_request_rejected",
                "rejection": {"retryable": True, "status_code": 429},
            },
            (),
            {},
            {},
        )
        records = driver.completed(plan, outcome)
        retry = records[-1]
        request_ref = ContentRef.from_dict(plan.payload["request"])
        request = driver.hydrate_plan(plan).payload["request"]

        assert [record.kind for record in records] == ["op_completed", "op_planned"]
        assert retry.payload["request"] == plan.payload["request"]
        assert retry.payload["tool"] == plan.payload["tool"]
        assert retry.payload["budget_admission"] == plan.payload["budget_admission"]
        assert retry.payload["request_digest"] == plan.payload["request_digest"] == reference_digest(request_ref)
        assert plan.payload["request_digest"] != digest(request)
    finally:
        sessions.close()


def test_completed_result_and_empty_usage_are_hydratable_and_deduplicated(tmp_path):
    from vv_agent.kernel._content import ContentRef
    from vv_agent.kernel._hydration import reference_digest
    from vv_agent.kernel._provider_outcomes import Definitive
    from vv_agent.kernel._records import digest

    store, sessions, driver, plan = _started_model_plan(tmp_path)
    try:
        records = driver.completed(plan, Definitive({}, (), {}, {}))
        assert [record.kind for record in records] == ["op_completed"]
        driver.commit(records, guarded=True)
        driver.refresh()
        assert plan.operation_id is not None
        completed = driver.state.operations[plan.operation_id].attempts[1].result
        assert completed is not None

        result_ref = ContentRef.from_dict(completed.payload["result"])
        usage_ref = ContentRef.from_dict(completed.payload["usage"])
        shared_ref = ContentRef.from_dict(completed.payload["shared_state"])
        evidence_ref = ContentRef.from_dict(completed.payload["evidence"])
        assert result_ref == usage_ref == shared_ref
        assert driver.hydrate_record_field(completed, "result", result_ref) == {}
        assert driver.hydrate_record_field(completed, "usage", usage_ref) == {}
        assert driver.hydrate_record_field(completed, "shared_state", shared_ref) == {}
        assert driver.hydrate_record_field(completed, "evidence", evidence_ref) == []
        assert completed.payload["result_digest"] == reference_digest(result_ref)
        assert completed.payload["result_digest"] != digest({})
        rows = store._one(
            "SELECT COUNT(*) FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
            ("default", "s", result_ref.content_id),
        )
        assert rows == (1,)
    finally:
        sessions.close()
