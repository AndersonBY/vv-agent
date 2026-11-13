"""The historical approval facade delegates decision flow to core recovery."""

from dataclasses import replace

import pytest

from vv_agent import Agent, RunConfig, function_tool
from vv_agent.approval import ApprovalBroker, ApprovalDecision
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.session import approval, kernel
from vv_agent.session._driver_scope import _Scope
from vv_agent.types import LLMResponse, ToolCall

from .test_tools_control_parity import Decisions, admit, records, runtime

pytestmark = pytest.mark.sqlite_file


def _park(store, database, tmp_path, effects):
    @function_tool(needs_approval=True)
    def effect() -> str:
        effects.append("ran")
        return "allowed"

    agent = Agent("test", "test", tools=[effect])
    config = RunConfig(workspace=tmp_path, approval_broker=ApprovalBroker())
    admit(store, config)
    llm = ScriptedLLM([LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")])
    kernel.drive(store, "control", runtime=runtime(store, database, agent, config, llm))
    park = records(store, kind="op_parked")[0]
    return agent, config, llm, park


@pytest.mark.parametrize("decision", ["approve", "deny"])
def test_resolve_approval_facade_uses_core_decision_and_delivery_path(store, database, tmp_path, decision):
    effects = []
    agent, config, llm, park = _park(store, database, tmp_path, effects)
    provider = Decisions(ApprovalDecision.allow() if decision == "approve" else ApprovalDecision.deny("not safe"))
    rt = runtime(store, database, agent, replace(config, approval_provider=provider), llm)
    lease = store.acquire("control", owner="approval-facade", ttl_ms=rt.ttl_ms)
    assert lease is not None
    driver = kernel._Driver(store, "control", rt, _Scope(lease, rt, now_ms=store._now()))
    driver.bind_turn()
    attempt = driver.state.operations[park.operation_id].attempts[park.attempt]

    assert approval.resolve_approval(driver, attempt)
    queued = next(item.item for item in store.peek_inbox("control") if item.item.kind == "approval_answer")
    assert queued.payload["request_id"] == park.payload["handle"]["request_id"]
    assert queued.payload["decision"] == decision
    assert store.release(lease)

    kernel.drive(store, "control", runtime=runtime(store, database, agent, config, llm))
    completed = next(r for r in records(store, kind="op_completed") if r.operation_id == park.operation_id)
    assert (completed.payload["result"]["error_code"] if decision == "deny" else None) == (
        "tool_approval_denied" if decision == "deny" else None
    )
    assert effects == ([] if decision == "deny" else ["ran"])


def test_resolve_approval_facade_preserves_transport_exception_chain(store, database, tmp_path):
    effects = []
    agent, config, llm, park = _park(store, database, tmp_path, effects)

    class Provider(Decisions):
        def should_request(self, request):
            raise error

    error = RuntimeError("approval transport failed")
    cause = ValueError("original transport cause")
    error.__cause__ = cause
    rt = runtime(store, database, agent, replace(config, approval_provider=Provider(None)), llm)
    lease = store.acquire("control", owner="approval-facade", ttl_ms=rt.ttl_ms)
    assert lease is not None
    driver = kernel._Driver(store, "control", rt, _Scope(lease, rt, now_ms=store._now()))
    driver.bind_turn()
    attempt = driver.state.operations[park.operation_id].attempts[park.attempt]

    with pytest.raises(RuntimeError) as caught:
        approval.resolve_approval(driver, attempt)
    assert caught.value is error
    assert caught.value.__cause__ is cause
    assert effects == []
    assert store.release(lease)
