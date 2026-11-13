"""Yield receipts retain actual SQL waits and callback exception boundaries."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sim.world import SimWorld

from vv_agent.kernel import _driver_outcomes as core
from vv_agent.session.records import InboxItem
from vv_agent.types import ToolDirective, ToolExecutionResult

from .test_core_driver_outcomes import assert_core_suspended, receipt_fixture
from .test_yieldable_waits import YieldProvider


def yield_fixture(world):
    original = core.completed_steps

    def completed(owner, plan, outcome):
        if plan._payload["op_kind"] == "tool":
            return (yield from owner.completed(plan, outcome))
        return (yield from original(owner, plan, outcome))

    with patch.object(core, "completed_steps", completed):
        worker, plan, _ = receipt_fixture(world, kind="tool")
    provider = YieldProvider(callback=None)
    worker.runtime.providers[plan._payload["provider_binding"]] = provider
    accepted = provider.submit(plan, context=None)
    worker.commit([worker.accepted(plan, accepted)])
    attempt = worker.state.operations[plan.operation_id].attempts[plan.attempt]
    assert attempt.state == "parked" and attempt.wait["yield_on_steer"]
    item = InboxItem("steer-proof", "steer", {"content": "change direction"}, plan.turn_id, worker.scope.generation)
    return worker, attempt, provider, item


@pytest.mark.parametrize("mode", ["none", "valid", "wrong-id", "finish", "bad-body", "noncanonical", "stop", "error"])
def test_provider_yield_validation_and_fallback_use_actual_completed_owner(tmp_path, monkeypatch, mode):
    with SimWorld(tmp_path / "yield.sqlite", seed=17) as world:
        worker, attempt, provider, item = yield_fixture(world)
        handle = dict(attempt.wait["handle"])
        calls, completions = [], []
        original = core.completed_steps

        def completed(owner, plan, outcome):
            completions.append((owner, plan, outcome))
            return (yield from original(owner, plan, outcome))

        monkeypatch.setattr(core, "completed_steps", completed)

        def callback(plan, copied, reason):
            assert_core_suspended()
            calls.append((plan, dict(copied), reason))
            assert copied == handle and copied is not attempt.wait["handle"]
            copied["evidence"] = "callback-local-change"
            if mode == "stop":
                raise StopIteration("private callback details")
            if mode == "error":
                raise RuntimeError("private callback details")
            result = ToolExecutionResult(plan._payload["request"]["id"], "custom interruption")
            if mode == "wrong-id":
                result.tool_call_id = "unrelated"
            elif mode == "finish":
                result.directive = ToolDirective.FINISH
            elif mode == "bad-body":
                result.metadata["output_type"] = "json"
            elif mode == "noncanonical":
                result.metadata["invalid"] = float("nan")
            return result

        provider.on_yield = None if mode == "none" else callback
        records = worker.yield_results([attempt], item)
        assert len(completions) == 1 and completions[0][0] is worker
        receipts = [record for record in records if record.kind == "op_completed"]
        assert len(receipts) == 1
        receipt = receipts[0]
        result = receipt.payload["result"]
        assert receipt.payload["yielded_by"] == item.input_id
        assert receipt.payload["evidence"] == [f"steer/{item.input_id}", handle["evidence"]]
        assert attempt.wait["handle"] == handle
        assert len(calls) == (mode != "none")
        if mode == "valid":
            assert result["content"] == "custom interruption"
            assert "yield_callback_failed" not in result.get("metadata", {})
        else:
            assert result["content"] == "Wait interrupted by new input; observed work has not been stopped."
            assert result["metadata"] == {"yield_reason": "steer"} | ({"yield_callback_failed": True} if mode != "none" else {})
        assert "private callback details" not in str(receipt.payload)


@pytest.mark.parametrize("phase,error_type", [("lookup", RuntimeError), ("lookup", StopIteration), ("invoke", KeyboardInterrupt)])
def test_yield_callback_lookup_and_base_errors_escape_unchanged(tmp_path, phase, error_type):
    with SimWorld(tmp_path / "errors.sqlite", seed=17) as world:
        worker, attempt, provider, item = yield_fixture(world)
        error = error_type("exact host error")
        cause = ValueError("original cause")
        error.__cause__ = cause
        before = worker.store._head("s")[:2]

        def callback(*args):
            assert_core_suspended()
            raise error

        if phase == "lookup":

            class Binding:
                @property
                def on_yield(self):
                    assert_core_suspended()
                    raise error

            worker.runtime.providers[attempt.plan._payload["provider_binding"]] = Binding()
        else:
            provider.on_yield = callback
        with pytest.raises(error_type) as caught:
            worker.yield_results([attempt], item)
        assert caught.value is error and caught.value.__cause__ is cause
        assert worker.store._head("s")[:2] == before
        assert worker.state.operations[attempt.plan.operation_id].attempts[attempt.plan.attempt].state == "parked"


def test_yield_prepares_resources_after_callback_with_current_leaf(tmp_path, monkeypatch):
    from vv_agent.session import tool_results

    with SimWorld(tmp_path / "resources.sqlite", seed=17) as world:
        worker, attempt, provider, item = yield_fixture(world)
        events = []
        original = tool_results.prepare_tool_resources

        def prepare(plan, result, backend):
            assert_core_suspended()
            events.append("prepare")
            return original(plan, result, backend)

        def callback(plan, handles, reason):
            events.append("callback")
            monkeypatch.setattr(tool_results, "prepare_tool_resources", prepare)
            return ToolExecutionResult(plan._payload["request"]["id"], "custom interruption")

        provider.on_yield = callback
        records = worker.yield_results([attempt], item)
        assert records and events == ["callback", "prepare"]
