"""Receipt decisions use real SQL prefixes, original host leaves and suspended observations."""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
from sim.scenarios import Provider
from sim.test_lease_gap_evidence import driver, until
from sim.world import SimWorld

import vv_agent.budget as budget_module
from vv_agent.budget import BudgetEvaluator, HostCost, RunBudgetLimits
from vv_agent.kernel import _driver_outcomes as core
from vv_agent.kernel._driver_planning import ProviderAvailable
from vv_agent.session import kernel, tool_results
from vv_agent.session.records import InboxItem, digest


class AtReceipt(BaseException):
    pass


def receipt_fixture(world, *, kind="model", limits=None, meter=None):
    world.create()
    worker = driver(world)
    if kind != "model":
        worker.runtime.providers["effect"] = Provider(world)
    worker.runtime.config = replace(worker.runtime.config, budget_limits=limits, host_cost_meter=meter)
    original = worker.completed
    captured = []

    def cut(plan, outcome):
        if plan._payload["op_kind"] == kind:
            captured.append((plan, outcome))
            raise AtReceipt()
        return original(plan, outcome)

    worker.completed = cut
    try:
        with pytest.raises(AtReceipt):
            until(worker, lambda: False)
    finally:
        worker.completed = original
    plan, outcome = captured[0]
    assert worker.state.operations[plan.operation_id].attempts[plan.attempt].started
    return worker, plan, outcome


def assert_core_suspended():
    frame = inspect.currentframe()
    assert frame is not None
    frame = frame.f_back
    while frame is not None:
        assert not frame.f_globals.get("__name__", "").startswith("vv_agent.kernel."), frame.f_code.co_name
        frame = frame.f_back


def test_budget_default_clock_and_bound_meter_keep_original_sampling_order(tmp_path, monkeypatch):
    original = inspect.signature(BudgetEvaluator).parameters["clock_ns"].default
    assert original is budget_module._DEFAULT_BUDGET_CLOCK_NS is kernel._DEFAULT_BUDGET_CLOCK_NS
    events, current = [], {"active": False, "switched": False, "ns": 0}

    class Meter:
        def __init__(self, name, amount):
            self.name, self.amount = name, amount

        def read(self):
            if current["active"]:
                assert_core_suspended()
                events.append("meter:" + self.name)
            return HostCost("credits", self.amount)

    first, replacement = Meter("bound", 3), Meter("replacement", 1001)
    with SimWorld(tmp_path / "clock.sqlite", seed=17) as world:
        worker, plan, outcome = receipt_fixture(
            world, limits=RunBudgetLimits(max_total_tokens=100, max_host_cost=HostCost("credits", 100)), meter=first
        )
        runtime_clock = worker.runtime.clock

        class RuntimeClock:
            def now_ms(self):
                return runtime_clock.now_ms()

            def monotonic(self):
                return runtime_clock.monotonic()

            def sleep(self, seconds):
                return runtime_clock.sleep(seconds)

            def monotonic_ns(self):
                assert_core_suspended()
                events.append("runtime-clock")
                return runtime_clock.monotonic_ns()

        worker.runtime.clock = RuntimeClock()

        def budget_clock():
            assert_core_suspended()
            events.append("budget-clock")
            current["ns"] += 1_000_000
            if not current["switched"]:
                worker.runtime.config = replace(worker.runtime.config, host_cost_meter=replacement)
                current["switched"] = True
            return current["ns"]

        # Replace both actual observation bindings; the product default identity above is unchanged.
        defaults = BudgetEvaluator.__init__.__kwdefaults__
        assert defaults is not None
        monkeypatch.setitem(defaults, "clock_ns", budget_clock)
        monkeypatch.setattr(kernel, "_DEFAULT_BUDGET_CLOCK_NS", budget_clock)
        current["active"] = True
        result = worker.completed(plan, outcome)
        assert events == ["runtime-clock", "budget-clock", "budget-clock", "meter:bound", "budget-clock", "meter:bound"]
        boundaries = [r.payload["data"] for r in result if r.kind == "boundary_recorded"]
        assert len(boundaries) == 2 and all(r["exhaustion"] is None for r in boundaries)
        assert all(r["usage"]["host_cost"]["amount_microunits"] == 3 for r in boundaries)
        assert len([r for r in result if r.kind == "op_planned"]) == 1
        assert worker.runtime.config.host_cost_meter is replacement


def test_resource_preparation_precedes_actual_durable_cancellation_audit(tmp_path, monkeypatch):
    with SimWorld(tmp_path / "resource.sqlite", seed=17) as world:
        worker, plan, outcome = receipt_fixture(world, kind="tool")
        original = tool_results.prepare_tool_resources
        calls = []

        def prepare(record, result, backend):
            prepared = original(record, result, backend)
            item = InboxItem("cancel-at-preparation", "control", {"action": "cancel"}, plan.turn_id, 0)
            with worker.store.atomic() as tx:
                tx.push(worker.sid, item)
            worker.refresh()
            applied = worker.record(
                "input_applied",
                {
                    "input": item.to_dict(),
                    "input_digest": digest(item.to_dict()),
                    "disposition": "applied",
                    "reason": None,
                    "target_operation_id": None,
                    "target_wait_id": None,
                    "position": "before_dispatch",
                },
            )
            worker.commit([applied], consumed=(item.input_id,), guarded=True)
            calls.append(prepared)
            return prepared

        monkeypatch.setattr(tool_results, "prepare_tool_resources", prepare)
        results = worker.completed(plan, outcome)
        assert len(calls) == 1 and len(results) == 1
        assert results[0].payload["context"] == "audit"
        assert results[0].payload["result"] == calls[0]
        applied = worker.store.read_input(worker.sid, "cancel-at-preparation")
        assert applied is not None and applied.consumed_seq is not None


def test_tool_plans_are_built_by_the_shared_core_planner(tmp_path):
    with SimWorld(tmp_path / "plans.sqlite", seed=17) as world:
        worker, plan, outcome = receipt_fixture(world)
        steps = core.completed_steps(worker, plan, outcome)
        request = next(steps)
        assert isinstance(request, ProviderAvailable) and request.name == "effect"
        with pytest.raises(StopIteration) as finished:
            steps.send(True)
        planned = finished.value.value[-1]
        assert planned.kind == "op_planned" and planned.payload["provider_binding"] == "effect"
        assert planned.payload["dependencies"] == [plan.operation_id]
        assert planned.operation_id == f"{plan.operation_id}/attempt/{plan.attempt}/tool/0"


def test_approval_timeout_and_provider_interval_are_sampled_around_actual_poll(tmp_path, monkeypatch):
    with SimWorld(tmp_path / "parking.sqlite", seed=17) as world:
        worker, plan, _ = receipt_fixture(world)
        handle = {
            "kind": "provider",
            "provider": plan.payload["provider_binding"],
            "job_id": "job",
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "evidence": "receipt",
            "query_ref": "job",
            "cancel_ref": None,
        }
        worker.runtime.poll_ms = 1000
        poll = worker.scope.poll
        calls = []

        def changed(store):
            result = poll(store)
            calls.append(result)
            worker.runtime.poll_ms = 73
            return result

        monkeypatch.setattr(worker.scope, "poll", changed)
        parked = worker.parked(plan, handle, after=True)
        assert parked.payload["poll_at_ms"] == calls[0].db_now_ms + 73
        assert parked.payload["deadline_ms"] is None


def test_invalid_acceptance_is_rejected_before_poll(tmp_path, monkeypatch):
    from vv_agent.kernel._provider_outcomes import Accepted, ProviderResultInvalid

    with SimWorld(tmp_path / "invalid.sqlite", seed=17) as world:
        worker, plan, _ = receipt_fixture(world)

        def unexpected(*args, **kwargs):
            raise AssertionError("invalid acceptance must not renew or observe time")

        monkeypatch.setattr(worker.scope, "poll", unexpected)
        with pytest.raises(ProviderResultInvalid):
            worker.accepted(plan, Accepted({"kind": "provider"}))


@pytest.mark.parametrize("timeout", [None, 1.25])
def test_approval_timeout_is_bound_before_poll_and_none_skips_poll(tmp_path, monkeypatch, timeout):
    with SimWorld(tmp_path / "approval-time.sqlite", seed=17) as world:
        worker, plan, _ = receipt_fixture(world)
        worker.runtime.config = replace(worker.runtime.config, approval_timeout_seconds=timeout)
        actual_poll = worker.scope.poll
        polls = []

        def poll(store):
            assert timeout is not None, "unbounded approval must not poll"
            result = actual_poll(store)
            polls.append(result)
            worker.runtime.config = replace(worker.runtime.config, approval_timeout_seconds=99)
            return result

        monkeypatch.setattr(worker.scope, "poll", poll)
        handle = {"kind": "approval", "request_id": "approval", "request_digest": plan.payload["request_digest"], "scope": []}
        result = worker.parked(plan, handle, after=False)
        assert result.payload["poll_at_ms"] is None
        assert result.payload["deadline_ms"] == (polls[0].db_now_ms + 1250 if timeout is not None else None)
        assert len(polls) == (0 if timeout is None else 1)


def test_unknown_samples_the_actual_reason_only_after_retry_is_eligible(tmp_path):
    from vv_agent.runtime.cancellation import CancellationToken

    class Token(CancellationToken):
        reads = 0

        @property
        def reason(self):
            self.reads += 1
            return super().reason

    with SimWorld(tmp_path / "unknown-reason.sqlite", seed=17) as world:
        worker, plan, _ = receipt_fixture(world)
        token = Token()
        worker.scope.token = token
        assert worker.unknown(plan, "provider_result_invalid").payload["retry"] == "stop"
        assert token.reads == 0
        result = worker.unknown(plan, "worker lost before durable result")
        assert token.reads == 1 and result.payload["retry"] == "retry"
        assert result.payload["measurement_missing"] and result.payload["observation"]["usage"] is None
        token.cancel("cancel")
        assert worker.unknown(plan, "lost reply").payload["retry"] == "stop"
        assert token.reads == 2


@pytest.mark.parametrize("error_type", [StopIteration, RuntimeError, KeyboardInterrupt])
def test_resource_host_errors_preserve_identity_and_cause(tmp_path, monkeypatch, error_type):
    failure = error_type("resource observation")
    cause = StopIteration("original cause")
    failure.__cause__ = cause
    with SimWorld(tmp_path / "resource-error.sqlite", seed=17) as world:
        worker, plan, outcome = receipt_fixture(world, kind="tool")

        def fail(*args):
            assert_core_suspended()
            raise failure

        monkeypatch.setattr(tool_results, "prepare_tool_resources", fail)
        with pytest.raises(error_type) as raised:
            worker.completed(plan, outcome)
        assert raised.value is failure and raised.value.__cause__ is cause
