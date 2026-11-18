"""Shared budget decisions and the outer observation boundary."""

import inspect
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vv_agent.budget import BudgetEvaluator, ModelCallBudgetExhausted
from vv_agent.kernel._budget import CoreBudgetEvaluator
from vv_agent.kernel._budget_values import (
    MAX_WIRE_INTEGER,
    BudgetDimension,
    BudgetUnavailableReason,
    BudgetUsageSnapshot,
    HostCost,
    RunBudgetLimits,
)

CORE_SCRIPT = r"""
import importlib.abc
import sys
import time
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
class OnlyCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (fullname in {'vv_agent', 'vv_agent.kernel'} or fullname.startswith('vv_agent.kernel.')
                or fullname.split('.')[0] in sys.stdlib_module_names), fullname
sys.meta_path.insert(0, OnlyCore())
from vv_agent.kernel._budget import CoreBudgetEvaluator, BudgetObservation, HostCostObservation
from vv_agent.kernel._budget_values import RunBudgetLimits, HostCost, BudgetExhaustionReason

def forbidden(*args): raise AssertionError('core read host time')
time.monotonic = time.monotonic_ns = time.time = time.sleep = forbidden

def audit(event, args):
    if event in {'open', 'os.listdir', 'os.scandir', 'socket.connect', 'socket.getaddrinfo', 'subprocess.Popen'}:
        raise AssertionError('core effect: ' + event)
sys.addaudithook(audit)

trace = []
def drain(steps):
    response = None
    while True:
        try: request = steps.send(response)
        except StopIteration as done: return done.value
        trace.append(request.value)
        response = 0 if request is BudgetObservation.CLOCK_NS else HostCostObservation(HostCost('credits', 0))
state = drain(CoreBudgetEvaluator.create_steps(RunBudgetLimits(max_total_tokens=3, max_tool_calls=1,
    max_host_cost=HostCost('credits', 10))))
assert trace == ['clock_ns']
assert state.enabled and drain(state.run_start_steps()) is None
assert drain(state.cycle_start_steps()) is None and state.snapshot().cycles == 1
usage = SimpleNamespace(total_tokens=3, cache_usage=SimpleNamespace(uncached_input_tokens=0))
assert drain(state.model_call_complete_steps(usage)) is None
assert drain(state.model_call_start_steps()).reason is BudgetExhaustionReason.LIMIT_REACHED
assert state.snapshot().cycles == 1
assert drain(state.preflight_tools_steps(['a'])) is None
assert drain(state.preflight_tools_steps(['b'])).attempted_increment == 1
assert state.snapshot().tool_calls_by_name == {'a': 1}
assert drain(state.tool_batch_complete_steps(operation_failed=True)) is None
assert drain(state.terminal_steps()) is None
assert trace == ['clock_ns'] + ['clock_ns', 'host_cost'] * 8
assert all(n in {'vv_agent','vv_agent.kernel'} or n.startswith('vv_agent.kernel.')
           for n in sys.modules if n == 'vv_agent' or n.startswith('vv_agent.'))
"""


@pytest.mark.parametrize("isolated", [False, True], ids=["normal", "no-site"])
def test_core_budget_runs_with_only_stdlib_and_core(isolated):
    command = [sys.executable, *(["-S"] if isolated else []), "-c", CORE_SCRIPT, str(Path(__file__).parents[1] / "src")]
    result = subprocess.run(command, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr


class Meter:
    def __init__(self, read):
        self._read = read

    def read(self):
        return self._read()


def assert_core_suspended():
    frame = sys._getframe(1)
    while frame is not None:
        assert not frame.f_globals.get("__name__", "").startswith("vv_agent.kernel."), frame.f_code.co_name
        frame = frame.f_back


def test_outer_reads_are_ordered_and_core_is_suspended():
    trace = []
    evaluator = None

    def sample(kind):
        assert_core_suspended()
        trace.append((kind, None if evaluator is None else evaluator.snapshot().total_tokens))

    def clock():
        sample("clock")
        return 0

    def read():
        sample("meter")
        return HostCost("credits", 0)

    evaluator = BudgetEvaluator(
        RunBudgetLimits(max_total_tokens=10, max_host_cost=HostCost("credits", 10)),
        host_cost_meter=Meter(read),
        clock_ns=clock,
    )
    assert evaluator.run_start() is None
    assert evaluator.cycle_start() is None
    assert (
        evaluator.model_call_complete(SimpleNamespace(total_tokens=2, cache_usage=SimpleNamespace(uncached_input_tokens=0)))
        is None
    )
    assert evaluator.preflight_tools(["a"]) is None
    assert evaluator.tool_batch_complete(operation_failed=True) is None
    assert evaluator.terminal() is None
    assert trace == [("clock", None)] + [(k, 0) for _ in range(2) for k in ("clock", "meter")] + [
        (k, 2) for _ in range(4) for k in ("clock", "meter")
    ]
    assert BudgetEvaluator.snapshot is CoreBudgetEvaluator.snapshot
    assert BudgetEvaluator.observe_model_usage is CoreBudgetEvaluator.observe_model_usage
    assert inspect.signature(BudgetEvaluator).parameters["clock_ns"].default is time.monotonic_ns
    assert ModelCallBudgetExhausted.__module__ == "vv_agent.budget"


def test_unavailable_latches_suppress_only_their_own_reads():
    trace = []

    def clock():
        assert_core_suspended()
        trace.append("clock")
        return 0 if trace == ["clock"] else (MAX_WIRE_INTEGER + 1) * 1_000_000

    def meter():
        assert_core_suspended()
        trace.append("meter")
        return None

    evaluator = BudgetEvaluator(
        RunBudgetLimits(max_wall_time_ms=1, max_host_cost=HostCost("credits", 10)),
        host_cost_meter=Meter(meter),
        clock_ns=clock,
    )
    assert evaluator.run_start() is None
    assert evaluator.terminal() is None
    assert trace == ["clock", "clock", "meter"]
    assert [(item.dimension, item.reason) for item in evaluator.snapshot().unavailable_dimensions] == [
        (BudgetDimension.WALL_TIME, BudgetUnavailableReason.INTEGER_OVERFLOW),
        (BudgetDimension.HOST_COST, BudgetUnavailableReason.METER_UNAVAILABLE),
    ]


@pytest.mark.parametrize("kind", ["clock", "meter", "usage"])
@pytest.mark.parametrize("error_kind", [StopIteration, RuntimeError, KeyboardInterrupt])
def test_observation_exception_identity_and_original_classification(kind, error_kind):
    failure = error_kind("original failure")
    failure.__cause__ = ValueError("original cause")
    reads = []

    def clock():
        assert_core_suspended()
        reads.append("clock")
        if kind == "clock" and len(reads) > 1:
            raise failure
        return 0

    def meter():
        assert_core_suspended()
        reads.append("meter")
        if kind == "meter":
            raise failure
        return HostCost("credits", 0)

    class Usage:
        @property
        def total_tokens(self):
            if kind == "usage":
                raise failure
            return 0

    evaluator = BudgetEvaluator(
        RunBudgetLimits(max_host_cost=HostCost("credits", 10)),
        host_cost_meter=Meter(meter),
        clock_ns=clock,
    )
    if kind == "meter" and issubclass(error_kind, Exception):
        assert evaluator.model_call_complete(Usage()) is None
        assert evaluator.snapshot().unavailable_dimensions[-1].reason is BudgetUnavailableReason.METER_ERROR
        evaluator.run_start()
        assert reads == ["clock", "clock", "meter", "clock"]
    else:
        with pytest.raises(error_kind) as caught:
            evaluator.model_call_complete(Usage())
        assert caught.value is failure and caught.value.__cause__.args == ("original cause",)
        assert reads == (["clock"] if kind == "usage" else ["clock", "clock", *(["meter"] if kind == "meter" else [])])


def test_constructor_validates_before_clock_and_samples_before_initial_usage():
    reads = []
    usage = BudgetUsageSnapshot(elapsed_ms=10)

    def clock():
        reads.append("clock")
        # The supplied snapshot is sampled after the original clock callback.
        object.__setattr__(usage, "elapsed_ms", 20)
        return 0

    with pytest.raises(ValueError, match="requires at least one"):
        BudgetEvaluator(RunBudgetLimits(), clock_ns=clock)
    assert not reads
    evaluator = BudgetEvaluator(RunBudgetLimits(max_wall_time_ms=30), initial_usage=usage, clock_ns=clock)
    assert reads == ["clock"] and evaluator.snapshot().elapsed_ms == 20
